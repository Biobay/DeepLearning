import os
import sys
import csv

# Aggiungi la root del progetto al path di Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm import tqdm
from itertools import chain

# Importa i moduli del progetto
import src.config as config
from src.data.dataset import create_dataloaders
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1, GeneratorS2
from src.models.discriminator import DiscriminatorS2

def train_stage2(cfg):
    """Funzione principale per l'addestramento del modello GAN Stage-II (64x64 -> 256x256)."""
    
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
    
    # Dataloader - per Stage-II usiamo immagini a 256x256
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg,
        img_size=256  # Stage-II lavora con immagini 256x256
    )
    print("Dataloaders Stage-II creati con successo (256x256).")
    
    # Modelli Stage-II
    text_encoder = TextEncoder(model_name=cfg.ENCODER_MODEL_NAME, fine_tune=cfg.FINE_TUNE_ENCODER).to(device)
    
    # Carica il generatore Stage-I pre-addestrato (frozen)
    netG_s1 = GeneratorS1(config=cfg).to(device)
    s1_checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_s1.pth")
    
    if not os.path.exists(s1_checkpoint_path):
        raise FileNotFoundError(f"Checkpoint Stage-I non trovato: {s1_checkpoint_path}")
    
    netG_s1.load_state_dict(torch.load(s1_checkpoint_path, map_location=device))
    netG_s1.eval()  # Stage-I rimane frozen durante Stage-II
    for param in netG_s1.parameters():
        param.requires_grad = False
    print(f"Generatore Stage-I caricato da: {s1_checkpoint_path}")
    
    # Generatore e Discriminatore Stage-II
    netG_s2 = GeneratorS2(config=cfg).to(device)
    netD_s2 = DiscriminatorS2(config=cfg).to(device)
    print("Generatore Stage-II (GeneratorS2) e Discriminatore Stage-II (DiscriminatorS2) creati.")
    
    # Ottimizzatori Stage-II
    # L'encoder viene ottimizzato insieme al generatore Stage-II
    params_g_s2 = chain(text_encoder.parameters(), netG_s2.parameters())
    optimizerG_s2 = optim.Adam(params_g_s2, lr=cfg.LEARNING_RATE_S2, betas=(0.5, 0.999))
    optimizerD_s2 = optim.Adam(netD_s2.parameters(), lr=cfg.LEARNING_RATE_S2, betas=(0.5, 0.999))
    
    # Loss
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Etichette per la loss avversaria
    real_label = 1.
    fake_label = 0.
    
    # Ciclo di addestramento Stage-II
    print("Inizio dell'addestramento GAN Stage-II (64x64 -> 256x256)...")
    for epoch in range(cfg.EPOCHS_S2):
        text_encoder.train()
        netG_s2.train()
        netD_s2.train()
        # netG_s1 rimane in eval mode
        
        total_loss_d = 0
        total_loss_g = 0
        
        progress_bar = tqdm(enumerate(train_loader), desc=f"Stage-II Epoch {epoch+1}/{cfg.EPOCHS_S2}", total=len(train_loader))
        
        for i, batch in progress_bar:
            if batch is None:
                print("Attenzione: saltato un batch di training vuoto.")
                continue
            
            # Prepara i dati
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images_256 = batch['image'].to(device)  # Immagini reali 256x256
            batch_size = real_images_256.size(0)
            
            # --- Fase 1: Addestramento del Discriminatore Stage-II ---
            netD_s2.zero_grad()
            
            # Estrai embedding testuali
            with torch.no_grad():
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
            
            # Genera immagini 64x64 con Stage-I (frozen)
            with torch.no_grad():
                noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
                stage1_images, stage1_mu = netG_s1(cls_embedding, hidden_states, noise)
            
            # Loss su immagini reali 256x256
            labels_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_real = netD_s2(real_images_256, cls_embedding.detach())
            loss_d_real = adversarial_loss(output_real, labels_real)
            
            # Genera immagini 256x256 con Stage-II
            fake_images_256, stage2_mu = netG_s2(stage1_images.detach(), cls_embedding.detach(), stage1_mu.detach())
            
            # Loss su immagini false 256x256
            labels_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD_s2(fake_images_256.detach(), cls_embedding.detach())
            loss_d_fake = adversarial_loss(output_fake, labels_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD_s2.step()
            
            # --- Fase 2: Addestramento del Generatore Stage-II (e dell'Encoder) ---
            netG_s2.zero_grad()
            text_encoder.zero_grad()
            
            # L'encoder ora calcola i gradienti
            cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
            
            # Rigenera immagini Stage-I (con gradienti per l'encoder)
            with torch.no_grad():
                noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
                stage1_images, stage1_mu = netG_s1(cls_embedding.detach(), hidden_states.detach(), noise)
            
            # Genera immagini 256x256 con Stage-II
            fake_images_256, stage2_mu = netG_s2(stage1_images, cls_embedding, stage1_mu)
            
            # L'obiettivo del generatore è che il discriminatore classifichi le sue immagini come reali
            labels_gen = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_g = netD_s2(fake_images_256, cls_embedding)
            
            loss_g_adv = adversarial_loss(output_g, labels_gen)
            loss_g_l1 = l1_loss(fake_images_256, real_images_256) * cfg.LAMBDA_L1_S2
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimizerG_s2.step()
            
            # Aggiorna le loss totali e la progress bar
            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            progress_bar.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())
            
            # Scrivi nel file di log
            log_writer.writerow([epoch + 1, i + 1, loss_d.item(), loss_g.item(), loss_g_adv.item(), loss_g_l1.item()])
        
        avg_loss_d = total_loss_d / len(train_loader)
        avg_loss_g = total_loss_g / len(train_loader)
        print(f"Stage-II Epoch {epoch+1}/{cfg.EPOCHS_S2}, Avg Loss D: {avg_loss_d:.4f}, Avg Loss G: {avg_loss_g:.4f}")
        
        # Salvataggio immagini generate Stage-II
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            text_encoder.eval()
            netG_s2.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader), None)
                if val_batch is None:
                    print("Attenzione: Nessun batch valido nel validation loader.")
                    continue
                
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                noise = torch.randn(input_ids.size(0), cfg.Z_DIM, device=device)
                
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
                
                # Pipeline completo Stage-I -> Stage-II
                stage1_images, stage1_mu = netG_s1(cls_embedding, hidden_states, noise)
                stage2_images, _ = netG_s2(stage1_images, cls_embedding, stage1_mu)
                
                # Salva immagini a tutte le risoluzioni per confronto
                save_image(val_batch['image'], os.path.join(stage2_generated_dir, f"real_images_256_epoch_{epoch+1}.png"), normalize=True)
                save_image(stage1_images, os.path.join(stage2_generated_dir, f"stage1_images_64_epoch_{epoch+1}.png"), normalize=True)
                save_image(stage2_images, os.path.join(stage2_generated_dir, f"stage2_images_256_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini Stage-II salvate per l'epoca {epoch+1}")
        
        # Salvataggio checkpoint Stage-II
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            # Salva i checkpoint specifici dell'epoca
            torch.save(netG_s2.state_dict(), os.path.join(stage2_checkpoint_dir, f"netG_s2_epoch_{epoch+1}.pth"))
            torch.save(netD_s2.state_dict(), os.path.join(stage2_checkpoint_dir, f"netD_s2_epoch_{epoch+1}.pth"))
            torch.save(text_encoder.state_dict(), os.path.join(stage2_checkpoint_dir, f"text_encoder_s2_epoch_{epoch+1}.pth"))
            
            # Salva il checkpoint finale del generatore Stage-II
            generator_s2_path = os.path.join(stage2_checkpoint_dir, "generator_s2.pth")
            torch.save(netG_s2.state_dict(), generator_s2_path)
            
            print(f"Checkpoint Stage-II salvato per l'epoca {epoch+1}")
            print(f"-> Checkpoint finale Stage-II salvato in: {generator_s2_path}")
    
    print("Addestramento Stage-II completato.")
    log_file.close()

if __name__ == '__main__':
    train_stage2(config)
