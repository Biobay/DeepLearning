# scripts/train.py (Versione con Loss L1 + LPIPS)

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
import lpips ## MODIFICA: Importa la libreria LPIPS

import src.config as config
from src.data.dataset import PokemonDataset, create_dataloaders
from src.models.model import PikaPikaGen

def calculate_metrics(real_images, generated_images):
    # (Funzione invariata)
    # ...
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
    Funzione di training con loss combinata (L1 + LPIPS).
    """
    
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
    print("Dataloaders creati.")

    model = PikaPikaGen(config).to(device)
    print(f"Modello PikaPikaGen creato. Dropout Rate: {getattr(config, 'DROPOUT_RATE', 'N/A')}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # =============================================================================
    # ## MODIFICA 1: Inizializza entrambe le funzioni di costo ##
    # =============================================================================
    criterion_l1 = nn.L1Loss()
    criterion_lpips = lpips.LPIPS(net='vgg').to(device) # Usa VGG come backbone, è più potente
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

    history = {'train_loss': [], 'val_loss': [], 'val_ssim': [], 'val_psnr': [], 'val_lpips': []} # Aggiungi lpips alla history
    
    print("Inizio dell'addestramento con loss combinata (L1 + LPIPS)...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Training]")
        for batch in progress_bar:
            if batch is None: continue
            input_ids, attention_mask, real_images = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device)
            
            generated_images, _ = model(input_ids, attention_mask)
            
            # =============================================================================
            # ## MODIFICA 2: Calcola e combina le loss ##
            # =============================================================================
            loss_l1 = criterion_l1(generated_images, real_images)
            loss_p = criterion_lpips(generated_images, real_images).mean()
            
            # Loss totale pesata
            loss = loss_l1 + cfg.LAMBDA_LPIPS * loss_p
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), l1=loss_l1.item(), lpips=loss_p.item())

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Fase di Validazione
        model.eval()
        total_val_loss, total_ssim, total_psnr, total_lpips = 0, 0, 0, 0
        with torch.no_grad():
            for val_batch in val_loader:
                if val_batch is None: continue
                input_ids, attention_mask, real_images = val_batch['input_ids'].to(device), val_batch['attention_mask'].to(device), val_batch['image'].to(device)
                generated_images, _ = model(input_ids, attention_mask)
                
                # Calcola tutte le metriche di validazione
                val_loss_l1 = criterion_l1(generated_images, real_images)
                val_loss_p = criterion_lpips(generated_images, real_images).mean()
                
                # La loss per lo scheduler è la loss L1, che è più stabile
                total_val_loss += val_loss_l1.item() 
                total_lpips += val_loss_p.item()
                
                ssim_score, psnr_score = calculate_metrics(real_images, generated_images)
                total_ssim += ssim_score
                total_psnr += psnr_score
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        avg_lpips = total_lpips / len(val_loader)

        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(avg_ssim)
        history['val_psnr'].append(avg_psnr)
        history['val_lpips'].append(avg_lpips) # Salva la nuova metrica

        print(
            f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val LPIPS: {avg_lpips:.4f} | Val SSIM: {avg_ssim:.4f} | Val PSNR: {avg_psnr:.4f}"
        )
        
        scheduler.step(avg_val_loss) # Lo scheduler continua a monitorare la Val Loss L1

        # Salvataggi (invariati) ...

    print("Addestramento completato.")
    return history

if __name__ == '__main__':
    training_history = train(config)
    # Salva la storia dell'addestramento
    with open("training_history.json", "w") as f:
        json.dump(training_history, f)