import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms # Importa transforms
import os
import pandas as pd
from transformers import BertModel, BertTokenizer # Modificato per usare BERT
import sys

# Aggiungi la root del progetto al sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src import config
from src.data.dataset import PokemonDataset, collate_fn # Importa collate_fn
from src.models.decoder import GeneratorS1
from src.models.generator_s2 import GeneratorS2
from src.models.discriminator_s2 import DiscriminatorS2

# --- Setup ---
def setup_directories():
    os.makedirs(config.CHECKPOINT_DIR_S2, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGE_DIR_S2, exist_ok=True)

# --- Funzione di Training ---
def train_stage2():
    setup_directories()

    # --- Modelli ---
    # Carica tokenizer e encoder di testo
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    text_encoder = BertModel.from_pretrained(config.ENCODER_MODEL_NAME).to(config.DEVICE)
    text_encoder.eval() # L'encoder non viene addestrato

    # Generatore Fase I (caricato e non addestrato)
    gen_s1 = GeneratorS1(config).to(config.DEVICE)
    # Assicurati che il checkpoint della fase 1 esista - USA UN PERCORSO ASSOLUTO
    checkpoint_s1_path = os.path.join(PROJECT_ROOT, config.CHECKPOINT_DIR_S1, 'generator_s1.pth')
    if not os.path.exists(checkpoint_s1_path):
        print(f"Errore: Checkpoint del generatore S1 non trovato in {checkpoint_s1_path}")
        print("Assicurati di aver prima addestrato la Fase I.")
        return
    gen_s1.load_state_dict(torch.load(checkpoint_s1_path, map_location=config.DEVICE))
    gen_s1.eval()

    # Modelli Fase II (da addestrare)
    gen_s2 = GeneratorS2(config).to(config.DEVICE)
    disc_s2 = DiscriminatorS2(config).to(config.DEVICE)

    # --- Ottimizzatori e Loss ---
    opt_gen_s2 = optim.Adam(gen_s2.parameters(), lr=config.LEARNING_RATE_S2, betas=(0.5, 0.999))
    opt_disc_s2 = optim.Adam(disc_s2.parameters(), lr=config.LEARNING_RATE_S2, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # --- Dataset ---
    transform_s2 = transforms.Compose([
        transforms.Resize((config.STAGE2_IMAGE_SIZE, config.STAGE2_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = PokemonDataset(
        csv_path=config.CSV_PATH, 
        img_dir=config.IMAGE_DIR, 
        tokenizer=tokenizer, 
        transform=transform_s2, # Usa la trasformazione per la Fase II
        max_seq_len=config.MAX_SEQ_LEN
    )
    # Usa collate_fn per gestire eventuali campioni None
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    # --- Logging ---
    log_file = 'loss_log_s2.csv'
    if not os.path.exists(log_file):
        pd.DataFrame(columns=['epoch', 'd_loss', 'g_loss']).to_csv(log_file, index=False)

    # --- Ciclo di Training ---
    for epoch in range(config.EPOCHS_S2):
        d_losses = []
        g_losses = []
        for batch_idx, batch in enumerate(dataloader):
            # Salta batch vuoti se collate_fn restituisce None
            if batch is None:
                continue

            real_images = batch['image'].to(config.DEVICE)
            texts = batch['description'] # Lista di stringhe
            
            # Ottieni embedding del testo
            with torch.no_grad():
                inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_SEQ_LEN).to(config.DEVICE)
                # Estrai sia gli hidden states che l'embedding [CLS]
                encoder_outputs = text_encoder(**inputs)
                hidden_states = encoder_outputs.last_hidden_state
                cls_embedding = hidden_states[:, 0, :]

            # --- Addestramento Discriminatore ---
            disc_s2.zero_grad()

            # Loss su immagini reali
            # Nota: la dimensione del batch potrebbe essere inferiore a BATCH_SIZE nell'ultimo batch
            current_batch_size = real_images.size(0)
            real_labels = torch.ones(current_batch_size, 1).to(config.DEVICE) # Target shape (batch, 1)

            output_real = disc_s2(real_images, cls_embedding).view(-1, 1)
            d_loss_real = criterion(output_real, real_labels)
            

            # Loss su immagini false
            noise = torch.randn(current_batch_size, config.Z_DIM, device=config.DEVICE)
            with torch.no_grad():
                # Passa gli argomenti corretti a gen_s1, che restituisce (immagine, pesi_attenzione)
                low_res_images, _ = gen_s1(cls_embedding=cls_embedding, hidden_states=hidden_states, z_noise=noise)
            
            # gen_s2 e disc_s2 usano solo il cls_embedding per il condizionamento
            fake_images = gen_s2(low_res_images, cls_embedding)
            fake_labels = torch.zeros(current_batch_size, 1).to(config.DEVICE) # Target shape (batch, 1)
            
            output_fake = disc_s2(fake_images.detach(), cls_embedding).view(-1, 1)
            d_loss_fake = criterion(output_fake, fake_labels)
            
            # Calcola la loss totale e aggiorna i pesi
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_disc_s2.step()
            d_losses.append(d_loss.item())

            # --- Addestramento Generatore ---
            gen_s2.zero_grad()
            # Vogliamo che il generatore inganni il discriminatore
            output_gen = disc_s2(fake_images, cls_embedding).view(-1, 1)
            g_loss = criterion(output_gen, real_labels) # Usa real_labels per la loss del generatore
            g_loss.backward()
            opt_gen_s2.step()
            g_losses.append(g_loss.item())

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config.EPOCHS_S2}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"
                )

        # --- Fine Epoch ---
        # Evita divisione per zero se non ci sono batch
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else 0
        avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else 0
        print(f"Fine Epoch {epoch}: Avg D_Loss: {avg_d_loss:.4f}, Avg G_Loss: {avg_g_loss:.4f}")

        # Logga le loss
        new_log = pd.DataFrame([{'epoch': epoch, 'd_loss': avg_d_loss, 'g_loss': avg_g_loss}])
        new_log.to_csv(log_file, mode='a', header=False, index=False)

        # Salva immagini generate e checkpoint
        if epoch % 5 == 0 or epoch == config.EPOCHS_S2 - 1:
            with torch.no_grad():
                # Usa un batch fisso di rumore per il confronto visivo
                fixed_noise = torch.randn(config.BATCH_SIZE, config.Z_DIM, device=config.DEVICE)
                # Prendi un batch fisso di descrizioni dal dataloader per coerenza
                try:
                    fixed_batch = next(iter(dataloader))
                    if fixed_batch is not None:
                        fixed_texts = fixed_batch['description']
                        fixed_inputs = tokenizer(fixed_texts, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_SEQ_LEN).to(config.DEVICE)
                        
                        # Estrai embedding per le immagini fisse
                        fixed_encoder_outputs = text_encoder(**fixed_inputs)
                        fixed_hidden_states = fixed_encoder_outputs.last_hidden_state
                        fixed_cls_embedding = fixed_hidden_states[:, 0, :]
                        
                        # Assicurati che la dimensione del rumore corrisponda a quella del testo
                        num_samples = fixed_cls_embedding.size(0)

                        # Genera immagini S1 e S2
                        fixed_low_res, _ = gen_s1(cls_embedding=fixed_cls_embedding, hidden_states=fixed_hidden_states, z_noise=fixed_noise[:num_samples])
                        fixed_fake_images = gen_s2(fixed_low_res, fixed_cls_embedding)
                        save_image(fixed_fake_images, f"{config.GENERATED_IMAGE_DIR_S2}/epoch_{epoch}.png", normalize=True)
                except StopIteration:
                    print("Dataloader vuoto, non posso generare immagini fisse.")

            # Salva i checkpoint dei modelli
            torch.save(gen_s2.state_dict(), os.path.join(config.CHECKPOINT_DIR_S2, f'generator_s2_epoch_{epoch}.pth'))
            torch.save(disc_s2.state_dict(), os.path.join(config.CHECKPOINT_DIR_S2, f'discriminator_s2_epoch_{epoch}.pth'))

if __name__ == '__main__':
    train_stage2()
