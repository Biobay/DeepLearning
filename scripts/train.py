# scripts/train.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import numpy as np

import src.config as config
from src.data.dataset import create_dataloaders
# Assicurati che model.py importi il nuovo UNetDecoder
from src.models.model import PikaPikaGen 

def calculate_metrics(real, generated):
    # ... (questa funzione può rimanere la stessa)
    pass 

def train(cfg):
    # --- 1. SETUP ---
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    train_loader, val_loader, _ = create_dataloaders(...) # Il tuo codice qui

    # PikaPikaGen ora userà UNetDecoder internamente
    model = PikaPikaGen(cfg).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # --- USIAMO SOLO L1 LOSS PER STABILITÀ ---
    criterion = nn.L1Loss() 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=cfg.SCHEDULER_PATIENCE, 
        factor=cfg.SCHEDULER_FACTOR, 
        verbose=True
    )

    history = {'train_loss': [], 'val_loss': [], 'val_ssim': [], 'val_psnr': []}
    
    print("Inizio addestramento con architettura U-Net e L1 Loss...")
    for epoch in range(cfg.EPOCHS):
        # --- Fase di Training ---
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        for batch in progress_bar:
            # ... logica del training batch ...
            # Forward pass, calcolo loss L1, backward, step...
            loss = criterion(...)
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- Fase di Validazione ---
        model.eval()
        # ... calcola avg_val_loss, avg_ssim, avg_psnr ...
        
        # Stampa e aggiorna scheduler
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: ...")
        scheduler.step(avg_val_loss)
        
        # ... salvataggio checkpoint e immagini ...

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    train(config)