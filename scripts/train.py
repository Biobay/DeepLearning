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

import src.config as config
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def train(cfg):
    device = torch.device(cfg.DEVICE)
    
    # --- SETUP INIZIALE (MODELLI, DATALOADER, ETC.) ---
    train_loader, _, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )
    
    model = PikaPikaGen(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # =============================================================================
    # ## INIZIO SESSIONE DI DEBUG ##
    # =============================================================================
    print("\nINIZIO SESSIONE DI DEBUG (1 epoca, 5 batch)...")
    
    # ESEGUIAMO PER UNA SOLA EPOCA
    for epoch in range(1):
        model.train()
        
        # ESEGUIAMO SOLO PER I PRIMI 5 BATCH
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print(f"\n--- ATTENZIONE BATCH {batch_idx}: Batch vuoto, saltato. ---")
                continue
            if batch_idx >= 5:
                break

            print(f"\n--- DEBUG BATCH {batch_idx} ---")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device)

            # 1. CONTROLLO DATI DI INPUT
            print(f"Shape input_ids: {input_ids.shape}")
            print(f"Shape real_images: {real_images.shape}")
            print(f"Valore medio real_images: {real_images.mean().item():.4f} | Min: {real_images.min().item():.4f} | Max: {real_images.max().item():.4f}")
            if torch.isnan(real_images).any() or torch.isinf(real_images).any():
                print("!!!!!! ERRORE: NaN o Inf nelle immagini reali !!!!!!")
                return

            # --- ESEGUIAMO IL FORWARD PASS ---
            # Usiamo il forward_generator per coerenza
            generated_images, _ = model.forward_generator(input_ids, attention_mask)
            
            # 2. CONTROLLO OUTPUT GENERATORE
            print(f"Shape generated_images: {generated_images.shape}")
            print(f"Valore medio generated_images: {generated_images.mean().item():.4f} | Min: {generated_images.min().item():.4f} | Max: {generated_images.max().item():.4f}")
            if torch.isnan(generated_images).any() or torch.isinf(generated_images).any():
                print("!!!!!! ERRORE: NaN o Inf nelle immagini generate !!!!!!")
                return
            
            real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
            loss = criterion(generated_images, real_images_resized)
            
            print(f"Loss Iniziale: {loss.item():.4f}")

            # --- BACKWARD PASS E CONTROLLO GRADIENTI ---
            optimizer.zero_grad()
            loss.backward()
            
            # 3. CONTROLLO GRADIENTI
            total_norm_encoder = 0
            for p in model.encoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_encoder += param_norm.item() ** 2
            total_norm_encoder = total_norm_encoder ** 0.5
            print(f"Norma L2 dei gradienti dell'Encoder: {total_norm_encoder:.4f}")

            total_norm_decoder = 0
            for p in model.decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_decoder += param_norm.item() ** 2
            total_norm_decoder = total_norm_decoder ** 0.5
            print(f"Norma L2 dei gradienti del Decoder: {total_norm_decoder:.4f}")
            
            if total_norm_encoder < 1e-6 or total_norm_decoder < 1e-6:
                print("!!!!!! ATTENZIONE: Gradiente quasi nullo! Il modello potrebbe non stare imparando. !!!!!!")
            if torch.isnan(torch.tensor(total_norm_encoder)) or torch.isnan(torch.tensor(total_norm_decoder)):
                print("!!!!!! ERRORE: Gradiente è NaN! Il training è esploso. !!!!!!")


            optimizer.step()
            
    print("\n--- DEBUG COMPLETATO ---")
    
    # Interrompiamo l'esecuzione dopo il debug
    return {}

if __name__ == '__main__':
    train(config)