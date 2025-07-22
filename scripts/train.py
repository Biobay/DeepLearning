import os
import sys
import csv # Importa il modulo csv

# Aggiungi la root del progetto al path di Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from itertools import chain # Per combinare i parametri degli ottimizzatori

# Importa i moduli del progetto
import src.config as config
from src.data.dataset import create_dataloaders
from src.models.encoder import TextEncoder # Importa l'encoder corretto
from src.models.decoder import GeneratorS1
from src.models.discriminator import DiscriminatorS1

def train(cfg):
    """Funzione principale per l'addestramento del modello GAN (Stage-I)."""
    
    # Setup
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    
    # Crea le directory di output
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)

    # Prepara il file di log per le loss
    log_file_path = os.path.join(cfg.LOG_DIR, "loss_log.csv")
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'loss_d', 'loss_g', 'loss_g_adv', 'loss_g_l1'])


    # Dataloader
    train_loader, val_loader, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )
    print("Dataloaders creati con successo.")

    # Modelli
    text_encoder = TextEncoder(model_name=cfg.ENCODER_MODEL_NAME, fine_tune=cfg.FINE_TUNE_ENCODER).to(device)
    netG = GeneratorS1(config=cfg).to(device) # CORREZIONE: Passa l'oggetto config
    netD = DiscriminatorS1(config=cfg).to(device) # CORREZIONE: Passa l'oggetto config
    print("Encoder, Generatore (GeneratorS1) e Discriminatore (DiscriminatorS1) creati.")

    # Ottimizzatori
    # I parametri dell'encoder vengono ottimizzati insieme a quelli del generatore
    params_g = chain(text_encoder.parameters(), netG.parameters())
    optimizerG = optim.Adam(params_g, lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))

    # Loss
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    # Etichette per la loss avversaria
    real_label = 1.
    fake_label = 0.

    # Ciclo di addestramento
    print("Inizio dell'addestramento GAN (Stage-I)...")
    for epoch in range(cfg.EPOCHS):
        text_encoder.train()
        netG.train()
        netD.train()
        
        total_loss_d = 0
        total_loss_g = 0
        
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{cfg.EPOCHS}", total=len(train_loader))

        for i, batch in progress_bar:
            if batch is None:
                print("Attenzione: saltato un batch di training vuoto.")
                continue

            # Prepara i dati
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device)
            batch_size = real_images.size(0)

            # --- Fase 1: Addestramento del Discriminatore ---
            netD.zero_grad()
            
            # Estrai embedding testuali (senza calcolare gradiente per l'encoder)
            with torch.no_grad():
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)

            # Loss su immagini reali
            labels_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_real = netD(real_images, cls_embedding.detach())
            loss_d_real = adversarial_loss(output_real, labels_real)
            
            # Loss su immagini false
            noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
            fake_images, _ = netG(cls_embedding.detach(), hidden_states.detach(), noise)
            
            labels_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)
            output_fake = netD(fake_images.detach(), cls_embedding.detach())
            loss_d_fake = adversarial_loss(output_fake, labels_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizerD.step()
            
            # --- Fase 2: Addestramento del Generatore (e dell'Encoder) ---
            netG.zero_grad()
            text_encoder.zero_grad()
            
            # L'encoder ora calcola i gradienti
            cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
            
            # Genera immagini false
            noise = torch.randn(batch_size, cfg.Z_DIM, device=device)
            fake_images, _ = netG(cls_embedding, hidden_states, noise)

            # L'obiettivo del generatore è che il discriminatore classifichi le sue immagini come reali
            labels_gen = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output_g = netD(fake_images, cls_embedding)
            
            loss_g_adv = adversarial_loss(output_g, labels_gen)
            loss_g_l1 = l1_loss(fake_images, real_images) * cfg.LAMBDA_L1
            loss_g = loss_g_adv + loss_g_l1
            loss_g.backward()
            optimizerG.step()

            # Aggiorna le loss totali e la progress bar
            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            progress_bar.set_postfix(Loss_D=loss_d.item(), Loss_G=loss_g.item())

            # Scrivi nel file di log
            log_writer.writerow([epoch + 1, i + 1, loss_d.item(), loss_g.item(), loss_g_adv.item(), loss_g_l1.item()])


        avg_loss_d = total_loss_d / len(train_loader)
        avg_loss_g = total_loss_g / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}, Avg Loss D: {avg_loss_d:.4f}, Avg Loss G: {avg_loss_g:.4f}")

        # Salvataggio immagini generate
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            text_encoder.eval()
            netG.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader), None)
                if val_batch is None:
                    print("Attenzione: Nessun batch valido nel validation loader.")
                    continue

                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                noise = torch.randn(input_ids.size(0), cfg.Z_DIM, device=device)
                
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
                generated_images, _ = netG(cls_embedding, hidden_states, noise)
                
                save_image(val_batch['image'], os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), normalize=True)
                save_image(generated_images, os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"), normalize=True)
            print(f"Immagini di esempio salvate per l'epoca {epoch+1}")

        # Salvataggio checkpoint
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            # Salva i checkpoint specifici dell'epoca (utile per il debug)
            torch.save(netG.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"netG_epoch_{epoch+1}.pth"))
            torch.save(netD.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"netD_epoch_{epoch+1}.pth"))
            torch.save(text_encoder.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"text_encoder_epoch_{epoch+1}.pth"))

            # --- CORREZIONE CHIAVE ---
            # Salva il checkpoint del generatore con il nome fisso richiesto dalla Fase II.
            # Questo file verrà sovrascritto ad ogni epoca di salvataggio, garantendo
            # che la Fase II parta sempre dall'ultimo checkpoint disponibile della Fase I.
            generator_s1_path = os.path.join(cfg.CHECKPOINT_DIR, "generator_s1.pth")
            torch.save(netG.state_dict(), generator_s1_path)
            
            print(f"Checkpoint salvato per l'epoca {epoch+1}")
            print(f"-> Checkpoint per Stage-II salvato in: {generator_s1_path}")

    print("Addestramento completato.")
    log_file.close() # Chiudi il file di log

if __name__ == '__main__':
    train(config)
