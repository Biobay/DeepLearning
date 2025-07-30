# scripts/train.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

import src.config as config
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def calculate_metrics(real_images, generated_images):
    """Calcola SSIM e PSNR. Le immagini devono essere nello stesso range [0, 1]."""
    real_images_np = real_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    generated_images_np = generated_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    
    # Denormalizza [-1, 1] -> [0, 1]
    real_images_np = (real_images_np + 1) / 2.0
    generated_images_np = (generated_images_np + 1) / 2.0
    
    batch_ssim, batch_psnr = 0.0, 0.0
    
    for i in range(real_images_np.shape[0]):
        batch_ssim += ssim(real_images_np[i], generated_images_np[i], multichannel=True, data_range=1.0, channel_axis=-1)
        batch_psnr += psnr(real_images_np[i], generated_images_np[i], data_range=1.0)
        
    return batch_ssim / real_images_np.shape[0], batch_psnr / real_images_np.shape[0]

def train(cfg):
    """Funzione principale per l'addestramento del modello U-Net."""
    
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    print("Creazione dei Dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )
    print("Dataloaders creati con successo.")

    model = PikaPikaGen(cfg).to(device)
    print("Modello PikaPikaGen con architettura U-Net creato.")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.L1Loss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=cfg.SCHEDULER_PATIENCE, 
        factor=cfg.SCHEDULER_FACTOR
    )

    history = {'train_loss': [], 'val_loss': [], 'val_ssim': [], 'val_psnr': []}
    
    print("\nInizio addestramento con architettura U-Net e L1 Loss...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        for batch in progress_bar:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device) # Questa è (N, 3, 256, 256)

            generated_images, _ = model(input_ids, attention_mask) # Questa è (N, 3, 215, 215)
            
            # Ridimensiona le immagini reali a 215x215 per il calcolo della loss
            real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
            
            loss = criterion(generated_images, real_images_resized)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Fase di Validazione
        model.eval()
        total_val_loss, total_ssim, total_psnr = 0.0, 0.0, 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                val_batches += 1
                
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                real_images = val_batch['image'].to(device) # (N, 3, 256, 256)

                generated_images, _ = model(input_ids, attention_mask) # (N, 3, 215, 215)
                
                real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
                
                val_loss = criterion(generated_images, real_images_resized)
                ssim_score, psnr_score = calculate_metrics(real_images_resized, generated_images)
                
                total_val_loss += val_loss.item()
                total_ssim += ssim_score
                total_psnr += psnr_score

        if val_batches > 0:
            avg_val_loss = total_val_loss / val_batches
            avg_ssim = total_ssim / val_batches
            avg_psnr = total_psnr / val_batches
        else:
            avg_val_loss, avg_ssim, avg_psnr = 0, 0, 0

        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(avg_ssim)
        history['val_psnr'].append(avg_psnr)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val SSIM: {avg_ssim:.4f} | Val PSNR: {avg_psnr:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0 and 'real_images_resized' in locals():
            save_image(real_images_resized, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
            save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate per l'epoca {epoch+1}")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    train(config)