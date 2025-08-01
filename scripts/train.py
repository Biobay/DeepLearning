# scripts/train.py
import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from itertools import chain

# Aggiunge la root del progetto al path di Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config
from src.data.dataset import create_dataloaders
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1
from src.models.discriminator import DiscriminatorS1 # Ora è il Multi-Scale

def train(cfg):
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    log_file_path = os.path.join(cfg.LOG_DIR, "loss_log_stage1.csv")
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'loss_d', 'loss_g', 'loss_g_adv', 'loss_g_l1'])
    
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=cfg.CSV_PATH, img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR, config=cfg)

    text_encoder = TextEncoder(fine_tune=cfg.FINE_TUNE_ENCODER).to(device)
    netG = GeneratorS1(config=cfg).to(device)
    netD = DiscriminatorS1(config=cfg).to(device)

    optimizerG = optim.Adam(chain(text_encoder.parameters(), netG.parameters()), lr=cfg.LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.5, 0.999))
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    print("Inizio addestramento GAN Stage-I con Discriminatore Multi-Scala...")
    for epoch in range(cfg.EPOCHS):
        text_encoder.train()
        netG.train()
        netD.train()
        
        total_loss_d, total_loss_g = 0.0, 0.0
        
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", total=len(train_loader))
        
        for i, batch in progress_bar:
            if batch is None: continue
            ids, mask, real_images = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device)
            batch_size = real_images.size(0)

            # --- Train Discriminator ---
            netD.zero_grad()
            with torch.no_grad():
                cls_embedding, hidden_states = text_encoder(ids, mask)
                noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
                fake_images, _ = netG(cls_embedding, hidden_states, noise)
            
            real_preds = netD(real_images, cls_embedding.detach())
            loss_d_real = 0
            for pred in real_preds:
                loss_d_real += adversarial_loss(pred, torch.ones_like(pred) * 0.9) # Label Smoothing
            
            fake_preds = netD(fake_images.detach(), cls_embedding.detach())
            loss_d_fake = 0
            for pred in fake_preds:
                loss_d_fake += adversarial_loss(pred, torch.zeros_like(pred))

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            optimizerD.step()

            # --- Train Generator ---
            netG.zero_grad(); text_encoder.zero_grad()
            cls_embedding, hidden_states = text_encoder(ids, mask)
            noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
            fake_images, _ = netG(cls_embedding, hidden_states, noise)
            
            gen_preds = netD(fake_images, cls_embedding)
            loss_g_adv = 0
            for pred in gen_preds:
                loss_g_adv += adversarial_loss(pred, torch.ones_like(pred))
            
            loss_g_l1 = l1_loss(fake_images, real_images) * cfg.LAMBDA_L1
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimizerG.step()
            
            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            progress_bar.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())
            log_writer.writerow([epoch + 1, i + 1, loss_d.item(), loss_g.item(), loss_g_adv.item(), loss_g_l1.item()])
        
        # =============================================================================
        # ## LOGICA DI FINE EPOCA COMPLETA ##
        # =============================================================================
        
        # 1. Calcola e stampa le loss medie dell'epoca
        avg_loss_d = total_loss_d / len(train_loader)
        avg_loss_g = total_loss_g / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} -> Avg Loss D: {avg_loss_d:.4f}, Avg Loss G: {avg_loss_g:.4f}")

        # 2. Salva un batch di immagini di esempio dalla validazione
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            text_encoder.eval()
            netG.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader), None)
                if val_batch:
                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    noise = torch.randn(input_ids.size(0), cfg.Z_DIM, device=device)
                    
                    cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
                    generated_images, _ = netG(cls_embedding, hidden_states, noise)
                    
                    save_image(val_batch['image'], os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_epoch_{epoch+1}.png"), normalize=True)
                    save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"fake_epoch_{epoch+1}.png"), normalize=True)
                    print(f"Immagini di esempio salvate per l'epoca {epoch+1}")

        # 3. Salva i checkpoint dei modelli
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"netG_s1_epoch_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"netD_s1_epoch_{epoch+1}.pth"))
            torch.save(text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"text_encoder_epoch_{epoch+1}.pth"))

            # Salva anche il checkpoint fisso per Stage-II
            generator_s1_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_s1.pth")
            torch.save(netG.state_dict(), generator_s1_path)
            
            print(f"Checkpoint salvati per l'epoca {epoch+1}")

    log_file.close()

if __name__ == '__main__':
    train(config)