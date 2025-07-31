# scripts/train.py (Versione SEMPLIFICATA per testare il nuovo decoder)

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

    # In PikaPikaGen, l'encoder e il nuovo decoder vengono assemblati
    model = PikaPikaGen(cfg).to(device)

    # Un singolo ottimizzatore per l'intero modello generativo
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

        # Semplice validazione (solo loss)
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                # ... (logica simile al training per calcolare avg_val_loss)
                pass # Aggiungi la logica di validazione qui
        
        avg_val_loss = 0 # Placeholder
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Train Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss) # O sulla val_loss se la calcoli
        
        # ... (logica di salvataggio checkpoint e immagini)
        
    return history

if __name__ == '__main__':
    train(config)