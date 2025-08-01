# scripts/train_stage2.py

import os
import sys
import csv

# Aggiungi la root del progetto al path di Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
from itertools import chain

# Importa i moduli del progetto
import src.config as config
from src.data.dataset import create_dataloaders
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1, GeneratorS2
from src.models.discriminator import DiscriminatorS2

def train_stage2(cfg):
    """Funzione principale per l'addestramento del modello GAN Stage-II (con Generatore U-Net)."""
    
    # Setup
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    
    # Crea le directory di output per Stage-II
    stage2_checkpoint_dir = os.path.join(cfg.CHECKPOINT_DIR, "stage2")
    stage2_generated_dir = os.path.join(cfg.GENERATED_IMAGE_DIR, "stage2")
    os.makedirs(stage2_checkpoint_dir, exist_ok=True)
    os.makedirs(stage2_generated_dir, exist_ok=True)
    
    # Prepara il file di log per le loss Stage-II
    log_file_path = os.path.join(cfg.LOG_DIR, "loss_log_stage2.csv")
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'loss_d', 'loss_g', 'loss_g_adv', 'loss_g_l1'])
    
    # Dataloader per immagini ad alta risoluzione
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=cfg.CSV_PATH, img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR, config=cfg,
        img_size=cfg.STAGE2_IMAGE_SIZE
    )
    print(f"Dataloaders Stage-II creati ({cfg.STAGE2_IMAGE_SIZE}x{cfg.STAGE2_IMAGE_SIZE}).")
    
    # Modelli
    text_encoder = TextEncoder(model_name=cfg.ENCODER_MODEL_NAME, fine_tune=cfg.FINE_TUNE_ENCODER).to(device)
    
    netG_s1 = GeneratorS1(config=cfg).to(device)
    s1_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_s1.pth")
    if not os.path.exists(s1_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint Stage-I non trovato: {s1_checkpoint_path}")
    netG_s1.load_state_dict(torch.load(s1_checkpoint_path, map_location=device))
    netG_s1.eval()
    for param in netG_s1.parameters():
        param.requires_grad = False
    print(f"Generatore Stage-I caricato da: {s1_checkpoint_path}")
    
    netG_s2 = GeneratorS2(config=cfg).to(device) # Ora è la U-Net
    netD_s2 = DiscriminatorS2(config=cfg).to(device)
    print("Generatore Stage-II (U-Net) e Discriminatore Stage-II creati.")
    
    # Ottimizzatori con learning rates differenziati
    params_g_s2 = chain(text_encoder.parameters(), netG_s2.parameters())
    optimizerG_s2 = optim.Adam(params_g_s2, lr=cfg.LEARNING_RATE_S2, betas=(0.5, 0.999))
    # NOTA: Se vuoi LR diversi per D, dovrai aggiungerlo al config e usarlo qui
    optimizerD_s2 = optim.Adam(netD_s2.parameters(), lr=cfg.LEARNING_RATE_S2, betas=(0.5, 0.999))
    
    # Loss
    adversarial_loss = nn.BCEWithlogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Ciclo di addestramento Stage-II
    print(f"Inizio dell'addestramento GAN Stage-II con Generatore U-Net...")
    for epoch in range(cfg.EPOCHS_S2):
        text_encoder.train()
        netG_s2.train()
        netD_s2.train()
        
        total_loss_d, total_loss_g = 0, 0
        
        progress_bar = tqdm(enumerate(train_loader), desc=f"Stage-II Epoch {epoch+1}/{cfg.EPOCHS_S2}", total=len(train_loader))
        
        for i, batch in progress_bar:
            if batch is None: continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images_s2 = batch['image'].to(device)
            batch_size = real_images_s2.size(0)
            
            # --- Fase 1: Addestramento del Discriminatore ---
            netD_s2.zero_grad()
            
            with torch.no_grad():
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
                noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
                stage1_images, _ = netG_s1(cls_embedding, hidden_states, noise)
            
            # Loss su immagini reali
            labels_real = torch.full((batch_size,), 1.0, dtype=torch.float, device=device) # Label smoothing può essere aggiunto qui
            output_real = netD_s2(real_images_s2, cls_embedding.detach())
            loss_d_real = adversarial_loss(output_real, labels_real)
            
            # Genera immagini S2 con la U-Net
            fake_images_s2, _ = netG_s2(stage1_images.detach(), cls_embedding.detach())
            
            # Loss su immagini false
            labels_fake = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)
            output_fake = netD_s2(fake_images_s2.detach(), cls_embedding.detach())
            loss_d_fake = adversarial_loss(output_fake, labels_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD_s2.step()
            
            # --- Fase 2: Addestramento del Generatore ---
            netG_s2.zero_grad()
            text_encoder.zero_grad()
            
            cls_embedding, _ = text_encoder(input_ids, attention_mask)
            
            # La U-Net non usa stage1_mu
            fake_images_s2, _ = netG_s2(stage1_images, cls_embedding)
            
            output_g = netD_s2(fake_images_s2, cls_embedding)
            labels_gen = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
            
            loss_g_adv = adversarial_loss(output_g, labels_gen)
            loss_g_l1 = l1_loss(fake_images_s2, real_images_s2) * cfg.LAMBDA_L1_S2
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimizerG_s2.step()
            
            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            progress_bar.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())
            
            log_writer.writerow([epoch + 1, i + 1, loss_d.item(), loss_g.item(), loss_g_adv.item(), loss_g_l1.item()])
        
        avg_loss_d = total_loss_d / len(train_loader)
        avg_loss_g = total_loss_g / len(train_loader)
        print(f"Stage-II Epoch {epoch+1}/{cfg.EPOCHS_S2}, Avg Loss D: {avg_loss_d:.4f}, Avg Loss G: {avg_loss_g:.4f}")
        
        # Salvataggio
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            with torch.no_grad():
                val_batch = next(iter(val_loader), None)
                if val_batch:
                    # Logica per generare e salvare un'immagine di esempio
                    input_ids = val_batch['input_ids'].to(device)
                    attention_mask = val_batch['attention_mask'].to(device)
                    noise = torch.randn(input_ids.size(0), cfg.Z_DIM, device=device)
                    cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
                    
                    stage1_images, _ = netG_s1(cls_embedding, hidden_states, noise)
                    stage2_images, _ = netG_s2(stage1_images, cls_embedding)
                    
                    save_image(val_batch['image'], os.path.join(stage2_generated_dir, f"real_e{epoch+1}.png"), normalize=True)
                    save_image(stage1_images, os.path.join(stage2_generated_dir, f"s1_e{epoch+1}.png"), normalize=True)
                    save_image(stage2_images, os.path.join(stage2_generated_dir, f"s2_e{epoch+1}.png"), normalize=True)
                    print(f"Immagini Stage-II salvate.")

        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(netG_s2.state_dict(), os.path.join(stage2_checkpoint_dir, f"netG_s2_e{epoch+1}.pth"))
            torch.save(netD_s2.state_dict(), os.path.join(stage2_checkpoint_dir, f"netD_s2_e{epoch+1}.pth"))
            torch.save(text_encoder.state_dict(), os.path.join(stage2_checkpoint_dir, f"text_encoder_s2_e{epoch+1}.pth"))
            torch.save(netG_s2.state_dict(), os.path.join(stage2_checkpoint_dir, "generator_s2.pth"))
            print(f"Checkpoint Stage-II salvati.")
    
    log_file.close()

if __name__ == '__main__':
    train_stage2(config)