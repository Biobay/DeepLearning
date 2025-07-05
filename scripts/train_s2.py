import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizer

from src.config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE_S2, EPOCHS_S2, 
    TEXT_EMBEDDING_DIM, Z_DIM, 
    CHECKPOINT_DIR_S1, CHECKPOINT_DIR_S2, GENERATED_IMAGE_DIR_S2,
    STAGE1_IMAGE_SIZE, STAGE2_IMAGE_SIZE
)
from src.data.dataset import PokemonDataset
from src.models.decoder import GeneratorS1 # Rinominato da Generator a GeneratorS1
from src.models.generator_s2 import GeneratorS2
from src.models.discriminator_s2 import DiscriminatorS2

# --- Setup ---
def setup_directories():
    os.makedirs(CHECKPOINT_DIR_S2, exist_ok=True)
    os.makedirs(GENERATED_IMAGE_DIR_S2, exist_ok=True)

# --- Funzione di Training ---
def train_stage2():
    setup_directories()

    # --- Modelli ---
    # Carica tokenizer e encoder di testo
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
    text_encoder.eval() # L'encoder non viene addestrato

    # Generatore Fase I (caricato e non addestrato)
    gen_s1 = GeneratorS1(z_dim=Z_DIM, text_embedding_dim=TEXT_EMBEDDING_DIM).to(DEVICE)
    checkpoint_s1_path = os.path.join(CHECKPOINT_DIR_S1, 'generator.pth')
    if not os.path.exists(checkpoint_s1_path):
        print(f"Errore: Checkpoint del generatore S1 non trovato in {checkpoint_s1_path}")
        print("Assicurati di aver prima addestrato la Fase I.")
        return
    gen_s1.load_state_dict(torch.load(checkpoint_s1_path, map_location=DEVICE))
    gen_s1.eval()

    # Modelli Fase II (da addestrare)
    gen_s2 = GeneratorS2(text_embedding_dim=TEXT_EMBEDDING_DIM).to(DEVICE)
    disc_s2 = DiscriminatorS2(text_embedding_dim=TEXT_EMBEDDING_DIM).to(DEVICE)

    # --- Ottimizzatori e Loss ---
    opt_gen_s2 = optim.Adam(gen_s2.parameters(), lr=LEARNING_RATE_S2, betas=(0.5, 0.999))
    opt_disc_s2 = optim.Adam(disc_s2.parameters(), lr=LEARNING_RATE_S2, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # --- Dataset ---
    dataset = PokemonDataset(csv_file='data/pokemon.csv', root_dir='data/images', tokenizer=tokenizer, target_size=STAGE2_IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Logging ---
    log_file = 'loss_log_s2.csv'
    if not os.path.exists(log_file):
        pd.DataFrame(columns=['epoch', 'd_loss', 'g_loss']).to_csv(log_file, index=False)

    # --- Ciclo di Training ---
    for epoch in range(EPOCHS_S2):
        d_losses = []
        g_losses = []
        for batch_idx, (real_images, texts) in enumerate(dataloader):
            real_images = real_images.to(DEVICE)
            
            # Ottieni embedding del testo
            with torch.no_grad():
                inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
                text_embedding = text_encoder(**inputs).last_hidden_state[:, 0, :] # [CLS] token

            # --- Addestramento Discriminatore ---
            disc_s2.zero_grad()

            # Loss su immagini reali
            real_labels = torch.ones(real_images.size(0), device=DEVICE)
            output_real = disc_s2(real_images, text_embedding)
            d_loss_real = criterion(output_real, real_labels)
            d_loss_real.backward()

            # Loss su immagini false
            noise = torch.randn(real_images.size(0), Z_DIM, device=DEVICE)
            with torch.no_grad():
                low_res_images = gen_s1(noise, text_embedding)
            
            fake_images = gen_s2(low_res_images, text_embedding)
            fake_labels = torch.zeros(real_images.size(0), device=DEVICE)
            output_fake = disc_s2(fake_images.detach(), text_embedding)
            d_loss_fake = criterion(output_fake, fake_labels)
            d_loss_fake.backward()

            d_loss = d_loss_real + d_loss_fake
            opt_disc_s2.step()
            d_losses.append(d_loss.item())

            # --- Addestramento Generatore ---
            gen_s2.zero_grad()
            output_gen = disc_s2(fake_images, text_embedding)
            g_loss = criterion(output_gen, real_labels) # Inganna il discriminatore
            g_loss.backward()
            opt_gen_s2.step()
            g_losses.append(g_loss.item())

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS_S2}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"
                )

        # --- Fine Epoch ---
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        print(f"Fine Epoch {epoch}: Avg D_Loss: {avg_d_loss:.4f}, Avg G_Loss: {avg_g_loss:.4f}")

        # Logga le loss
        new_log = pd.DataFrame([{'epoch': epoch, 'd_loss': avg_d_loss, 'g_loss': avg_g_loss}])
        new_log.to_csv(log_file, mode='a', header=False, index=False)

        # Salva immagini generate e checkpoint
        if epoch % 5 == 0 or epoch == EPOCHS_S2 - 1:
            with torch.no_grad():
                # Usa un batch fisso per il confronto visivo
                fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=DEVICE)
                fixed_texts = [texts[i] for i in range(min(len(texts), BATCH_SIZE))]
                fixed_inputs = tokenizer(fixed_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
                fixed_text_embedding = text_encoder(**fixed_inputs).last_hidden_state[:, 0, :]
                
                fixed_low_res = gen_s1(fixed_noise, fixed_text_embedding)
                fixed_fake_images = gen_s2(fixed_low_res, fixed_text_embedding)
                save_image(fixed_fake_images, f"{GENERATED_IMAGE_DIR_S2}/epoch_{epoch}.png", normalize=True)

            torch.save(gen_s2.state_dict(), os.path.join(CHECKPOINT_DIR_S2, f'generator_s2_epoch_{epoch}.pth'))
            torch.save(disc_s2.state_dict(), os.path.join(CHECKPOINT_DIR_S2, f'discriminator_s2_epoch_{epoch}.pth'))

if __name__ == '__main__':
    train_stage2()
