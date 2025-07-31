# scripts/train.py
import os, sys, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.utils import save_image
from tqdm import tqdm

import src.config as cfg
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def train(cfg):
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True); os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME), img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR, config=cfg)

    model = PikaPikaGen(cfg).to(device)
    optimizer = optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    history = {'train_loss': [], 'val_loss': []}

    print("\nInizio addestramento finale con U-Net a Cross-Attention e L1 Loss...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            if batch is None: continue
            ids, mask, real_full = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device)
            real = F.interpolate(real_full, size=cfg.IMAGE_OUTPUT_SIZE)
            gen_img, _ = model.forward_generator(ids, mask)
            loss = criterion(gen_img, real)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                ids, mask, real_full = val_batch['input_ids'].to(device), val_batch['attention_mask'].to(device), val_batch['image'].to(device)
                real = F.interpolate(real_full, size=cfg.IMAGE_OUTPUT_SIZE)
                gen_img, _ = model.forward_generator(ids, mask)
                total_val_loss += criterion(gen_img, real).item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        history['val_loss'].append(avg_val_loss)
        print(f"-> Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        
        if (epoch+1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            save_image(torch.cat([real[:4], gen_img[:4]]), os.path.join(cfg.GENERATED_IMAGE_DIR, f"e{epoch+1}.png"), normalize=True, nrow=4)
            print("Immagini salvate.")
        if (epoch+1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"gen_e{epoch+1}.pth"))
            print("Checkpoint salvato.")
            
    return history

if __name__ == '__main__': train(cfg)