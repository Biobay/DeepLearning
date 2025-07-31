# scripts/train.py (Versione Finale per testare il Decoder con Input Strutturato)

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
from src.models.model import PikaPikaGen # model.py deve usare il nuovo decoder

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

    # Un singolo ottimizzatore per l'intero modello generativo (Encoder + Decoder)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Usiamo solo L1 Loss per questo test
    criterion = nn.L1Loss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', 
        patience=cfg.SCHEDULER_PATIENCE, 
        factor=cfg.SCHEDULER_FACTOR
    )

    history = {'train_loss': [], 'val_loss': []}
    
    print("\nInizio addestramento con Input Strutturato e L1 Loss...")
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

        # --- FASE DI VALIDAZIONE COMPLETA ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                val_batches += 1
                
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                real_images = val_batch['image'].to(device)

                generated_images, _ = model.forward_generator(input_ids, attention_mask)
                
                real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
                val_loss = criterion(generated_images, real_images_resized)
                
                total_val_loss += val_loss.item()
        
        if val_batches > 0:
            avg_val_loss = total_val_loss / val_batches
        else:
            avg_val_loss = 0.0 # Se non ci sono batch di validazione
        
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Lo scheduler dovrebbe basarsi sulla loss di validazione
        scheduler.step(avg_val_loss)
        
        # --- SALVATAGGIO CHECKPOINT E IMMAGINI ---
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            # Salva l'ultimo batch di validazione per un confronto visivo
            if 'real_images_resized' in locals():
                save_image(real_images_resized, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
                save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
                print(f"Immagini di esempio salvate per l'epoca {epoch+1}")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth")
            # Salva solo il generatore, dato che non c'è il discriminatore
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    # Esegue il training se lo script viene lanciato direttamente
    # L'orchestratore chiamerà questa funzione e userà il suo output
    train(config)