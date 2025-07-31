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

import src.config as cfg
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

    model = PikaPikaGen(cfg).to(device)

    # =============================================================================
    # ## OTTIMIZZATORE CON LEARNING RATES DIFFERENZIATI ##
    # =============================================================================
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': cfg.LEARNING_RATE_ENCODER},
        {'params': model.decoder.parameters(), 'lr': cfg.LEARNING_RATE_DECODER}
    ], weight_decay=cfg.WEIGHT_DECAY)
    
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=cfg.SCHEDULER_PATIENCE, 
        factor=cfg.SCHEDULER_FACTOR
    )

    history = {'train_loss': [], 'val_loss': []}
    
    print("\nInizio addestramento con Learning Rates Differenziati...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        for batch in progress_bar:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device)

            generated_images, _ = model.forward_generator(input_ids, attention_mask)
            
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
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                real_images = val_batch['image'].to(device)

                generated_images, _ = model.forward_generator(input_ids, attention_mask)
                real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
                
                val_loss = criterion(generated_images, real_images_resized)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        # Salvataggio
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            save_image(real_images_resized, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
            save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate.")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth"))
            print(f"Checkpoint salvato.")

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    train(cfg)