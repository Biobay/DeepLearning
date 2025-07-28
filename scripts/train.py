# scripts/train.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

import src.config as config
from src.data.dataset import PokemonDataset, create_dataloaders
from src.models.model import PikaPikaGen

def calculate_metrics(real_images, generated_images):
    # Funzione invariata
    real_images_np = real_images.cpu().numpy().transpose(0, 2, 3, 1)
    generated_images_np = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
    real_images_np = (real_images_np + 1) / 2
    generated_images_np = (generated_images_np + 1) / 2
    batch_ssim, batch_psnr = 0, 0
    for i in range(real_images_np.shape[0]):
        batch_ssim += ssim(real_images_np[i], generated_images_np[i], multichannel=True, data_range=1.0, channel_axis=-1)
        batch_psnr += psnr(real_images_np[i], generated_images_np[i], data_range=1.0)
    return batch_ssim / real_images_np.shape[0], batch_psnr / real_images_np.shape[0]

def train(cfg):
    """
    Funzione principale per l'addestramento del modello con LR Scheduler e Dropout.
    """
    
    # --- 1. SETUP ---
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)

    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )
    print("Dataloaders creati con successo.")

    model = PikaPikaGen(config).to(device)
    print(f"Modello PikaPikaGen creato. Dropout Rate: {getattr(config, 'DROPOUT_RATE', 'N/A')}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.L1Loss()
    
    # =============================================================================
    # ## MODIFICA 1: INIZIALIZZAZIONE DELLO SCHEDULER ##
    # =============================================================================
    # Riduce il learning rate se la val_loss non migliora per 5 epoche
    # In scripts/train.py

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ssim': [],
        'val_psnr': []
    }

    # --- 3. CICLO DI ADDESTRAMENTO ---
    print("Inizio dell'addestramento...")
    for epoch in range(cfg.EPOCHS):
        # --- Fase di Training ---
        model.train() # ## MODIFICA: Assicura che il Dropout sia ATTIVO ##
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        # (Ciclo di training interno invariato)
        for batch in progress_bar:
            if batch is None: continue
            input_ids, attention_mask, real_images = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device)
            generated_images, _ = model(input_ids, attention_mask)
            loss = criterion(generated_images, real_images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Fase di Validazione ---
        model.eval() # ## MODIFICA: Assicura che il Dropout sia DISATTIVATO ##
        total_val_loss, total_ssim, total_psnr = 0, 0, 0
        with torch.no_grad():
            # ## MODIFICA: Ora iteriamo su tutto il validation set per una metrica più stabile ##
            for val_batch in val_loader:
                if val_batch is None: continue
                
                input_ids, attention_mask, real_images = val_batch['input_ids'].to(device), val_batch['attention_mask'].to(device), val_batch['image'].to(device)
                generated_images, _ = model(input_ids, attention_mask)
                
                val_loss = criterion(generated_images, real_images)
                ssim_score, psnr_score = calculate_metrics(real_images, generated_images)

                total_val_loss += val_loss.item()
                total_ssim += ssim_score
                total_psnr += psnr_score

        # Calcoliamo le medie sul validation set completo
        avg_val_loss = total_val_loss / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)

        # Salvataggio immagini (usando l'ultimo batch di validazione)
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0 and 'real_images' in locals():
            save_image(real_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
            save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate per l'epoca {epoch+1}")
        
        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(avg_ssim)
        history['val_psnr'].append(avg_psnr)

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val SSIM: {avg_ssim:.4f} | Val PSNR: {avg_psnr:.4f}"
        )
        
        # =============================================================================
        # ## MODIFICA 2: APPLICAZIONE DELLO SCHEDULER ##
        # =============================================================================
        scheduler.step(avg_val_loss)

        # Salvataggio checkpoint
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")

    print("Addestramento completato.")
    
    return history

if __name__ == '__main__':
    training_history = train(config)
    print("\n--- Riepilogo Training Standalone ---")
    if training_history:
        for metric, values in training_history.items():
            print(f"Metrica '{metric}': Valore Finale = {values[-1]:.4f}")