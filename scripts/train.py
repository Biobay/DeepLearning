# scripts/train.py (MODIFICATO PER DIAGNOSTICA)

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
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    train_loader, _, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )

    model = PikaPikaGen(cfg).to(device)
    generator = model
    discriminator = model.discriminator

    opt_gen = optim.Adam(list(generator.encoder.parameters()) + list(generator.decoder.parameters()), lr=cfg.LEARNING_RATE_GEN, betas=(cfg.BETA1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=cfg.LEARNING_RATE_DISC, betas=(cfg.BETA1, 0.999))
    
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    print("\n--- INIZIO ESECUZIONE DI DIAGNOSTICA (1 solo batch) ---")
    
    # Eseguiamo solo per un'epoca
    for epoch in range(1):
        generator.train()
        discriminator.train()
        
        # Prendiamo solo il primo batch
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                print("Primo batch non valido, passo al successivo...")
                continue
            
            print(f"\n--- Processing Batch {batch_idx+1} ---")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images_full = batch['image'].to(device)
            
            real_images = F.interpolate(real_images_full, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
            fake_images, _ = generator.forward_generator(input_ids, attention_mask)

            # FASE 1: Discriminatore
            opt_disc.zero_grad()
            disc_real_pred = discriminator(real_images, real_images)
            real_labels = torch.full_like(disc_real_pred, cfg.REAL_LABEL_SMOOTHING, device=device)
            loss_disc_real = bce_loss(disc_real_pred, real_labels)
            disc_fake_pred = discriminator(fake_images.detach(), real_images)
            fake_labels = torch.zeros_like(disc_fake_pred, device=device)
            loss_disc_fake = bce_loss(disc_fake_pred, fake_labels)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            loss_disc.backward()
            opt_disc.step()

            # FASE 2: Generatore
            opt_gen.zero_grad()
            disc_pred_for_gen = discriminator(fake_images, real_images)
            gan_labels = torch.ones_like(disc_pred_for_gen, device=device)
            loss_gen_gan = bce_loss(disc_pred_for_gen, gan_labels)
            loss_gen_l1 = l1_loss(fake_images, real_images) * cfg.LAMBDA_L1
            loss_gen = loss_gen_gan + loss_gen_l1
            
            # Calcolo dei gradienti per la diagnostica
            loss_gen.backward()
            
            # =====================================================================
            # ## INIZIO BLOCCO DI DIAGNOSTICA GRADIENTI ##
            # =====================================================================
            print("\n--- INIZIO DIAGNOSTICA GRADIENTI ---")
            grad_norm_encoder = sum(p.grad.data.norm(2).item() for p in generator.encoder.parameters() if p.grad is not None)
            params_encoder = sum(1 for p in generator.encoder.parameters() if p.grad is not None)
            print(f"Norma L2 Totale dei Gradienti dell'Encoder: {grad_norm_encoder}")
            print(f"Numero di parametri con gradiente nell'Encoder: {params_encoder}")

            grad_norm_decoder = sum(p.grad.data.norm(2).item() for p in generator.decoder.parameters() if p.grad is not None)
            params_decoder = sum(1 for p in generator.decoder.parameters() if p.grad is not None)
            print(f"Norma L2 Totale dei Gradienti del Decoder: {grad_norm_decoder}")
            print(f"Numero di parametri con gradiente nel Decoder: {params_decoder}")
            print("--- FINE DIAGNOSTICA GRADIENTI ---\n")
            # =====================================================================
            
            opt_gen.step()
            
            # =====================================================================
            # ## INIZIO TEST DI CONDIZIONAMENTO ##
            # =====================================================================
            print("\n--- INIZIO TEST DI CONDIZIONAMENTO ---")
            if input_ids.size(0) >= 2:
                desc_1, desc_2 = batch['description'][0], batch['description'][1]
                print(f"Descrizione 1: '{desc_1[:80]}...'")
                print(f"Descrizione 2: '{desc_2[:80]}...'")

                generator.eval()
                with torch.no_grad():
                    fake_image_1, _ = generator.forward_generator(input_ids[0:1], attention_mask[0:1])
                    fake_image_2, _ = generator.forward_generator(input_ids[1:2], attention_mask[1:2])
                
                l1_diff = torch.nn.functional.l1_loss(fake_image_1, fake_image_2)
                print(f"Differenza L1 tra le due immagini generate: {l1_diff.item():.6f}")

                save_image(fake_image_1, "test_image_1.png", normalize=True)
                save_image(fake_image_2, "test_image_2.png", normalize=True)
                print("Immagini di test salvate come 'test_image_1.png' e 'test_image_2.png'")
            else:
                print("Batch troppo piccolo per il test di condizionamento (serve size >= 2).")
            print("--- FINE TEST DI CONDIZIONAMENTO ---\n")
            # =====================================================================

            # Ferma il training dopo il primo batch valido
            print("\n--- ESECUZIONE DI DIAGNOSTICA TERMINATA ---")
            return None # Esce dalla funzione train
            
    return None # Nel caso il dataloader fosse vuoto

if __name__ == '__main__':
    train(config)