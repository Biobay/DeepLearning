# scripts/train.py (Versione con GAN DISATTIVATA per testare il nuovo decoder)

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
    
    train_loader, val_loader, _ = create_dataloaders( # Riattiviamo il val_loader
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )

    # Inizializza solo il Generatore (Encoder + Decoder)
    model = PikaPikaGen(cfg).to(device)
    
    ## --- DISATTIVATO --- ##
    # Non creiamo né usiamo il Discriminatore in questo esperimento
    # discriminator = model.discriminator

    # --- OTTIMIZZATORE SINGOLO PER IL GENERATORE ---
    optimizer = optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), 
        lr=cfg.LEARNING_RATE # Usiamo un unico learning rate per semplicità
    )
    
    # --- LOSS SEMPLICE: SOLO L1 ---
    criterion = nn.L1Loss()
    
    # Riattiviamo lo scheduler, utile per il training con L1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=cfg.SCHEDULER_PATIENCE)
    
    history = {'train_loss': [], 'val_loss': []}

    print("\nInizio addestramento di DEBUG (solo Generatore con L1 Loss)...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        for batch in progress_bar:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images_full_res = batch['image'].to(device)
            
            real_images = F.interpolate(real_images_full_res, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))

            # --- SOLO FORWARD PASS DEL GENERATORE ---
            generated_images, _ = model.forward_generator(input_ids, attention_mask)
            
            # --- CALCOLO DELLA LOSS L1 ---
            loss = criterion(generated_images, real_images)
            
            # --- BACKWARD PASS E OTTIMIZZAZIONE (SOLO PER IL GENERATORE) ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # --- FASE DI VALIDAZIONE (CON L1 LOSS) ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                val_batches += 1
                
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                real_images = F.interpolate(val_batch['image'].to(device), size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))

                generated_images, _ = model.forward_generator(input_ids, attention_mask)
                val_loss = criterion(generated_images, real_images)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # --- SALVATAGGI ---
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            save_image(real_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
            save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate.")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            # Salviamo l'intero modello (che ora è solo il generatore)
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth"))
            print(f"Checkpoint salvato.")

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    train(config)