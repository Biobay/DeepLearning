# scripts/train.py (Versione GAN)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import src.config as config
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def train(cfg):
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )

    # Inizializza i modelli
    model = PikaPikaGen(cfg).to(device)
    generator = model # Per chiarezza, chiamiamo il modello completo 'generator'
    discriminator = model.discriminator

    # Ottimizzatori separati per generatore e discriminatore
    opt_gen = optim.Adam(list(generator.encoder.parameters()) + list(generator.decoder.parameters()), lr=cfg.LEARNING_RATE_GEN, betas=(cfg.BETA1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE_DISC, betas=(cfg.BETA1, 0.999))
    
    # Funzioni di costo
    bce_loss = nn.BCEWithLogitsLoss() # Più stabile di BCE + Sigmoid
    l1_loss = nn.L1Loss()
    
    history = {'gen_loss': [], 'disc_loss': []}

    print("\nInizio addestramento con architettura GAN...")
    for epoch in range(cfg.EPOCHS):
        generator.train()
        discriminator.train()
        
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        for batch in progress_bar:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images_full = batch['image'].to(device) # (256x256)
            
            # Ridimensiona le immagini reali a 215x215 per il confronto
            real_images = F.interpolate(real_images_full, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))

            # Genera immagini false
            fake_images, _ = generator.forward_generator(input_ids, attention_mask)

            # --- Fase 1: Addestramento del Discriminatore ---
            opt_disc.zero_grad()
            
            # Loss su immagini reali
            disc_real_pred = discriminator(real_images, real_images)
            loss_disc_real = bce_loss(disc_real_pred, torch.ones_like(disc_real_pred))
            
            # Loss su immagini false
            disc_fake_pred = discriminator(fake_images.detach(), real_images)
            loss_disc_fake = bce_loss(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # --- Fase 2: Addestramento del Generatore ---
            opt_gen.zero_grad()
            
            disc_pred_for_gen = discriminator(fake_images, real_images)
            loss_gen_gan = bce_loss(disc_pred_for_gen, torch.ones_like(disc_pred_for_gen))
            
            loss_gen_l1 = l1_loss(fake_images, real_images) * cfg.LAMBDA_L1
            
            loss_gen = loss_gen_gan + loss_gen_l1
            loss_gen.backward()
            opt_gen.step()
            
            total_gen_loss += loss_gen.item()
            total_disc_loss += loss_disc.item()
            progress_bar.set_postfix(G_loss=loss_gen.item(), D_loss=loss_disc.item())
        
        avg_gen_loss = total_gen_loss / len(train_loader)
        avg_disc_loss = total_disc_loss / len(train_loader)
        history['gen_loss'].append(avg_gen_loss)
        history['disc_loss'].append(avg_disc_loss)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Gen Loss: {avg_gen_loss:.4f} | Disc Loss: {avg_disc_loss:.4f}")
        
        # Salvataggio
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            save_image(real_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
            save_image(fake_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate.")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(generator.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"discriminator_epoch_{epoch+1}.pth"))
            print(f"Checkpoint salvati.")

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    train(config)