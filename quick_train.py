#!/usr/bin/env python3
"""
🚀 Training Rapido StackGAN Stage-I
Versione semplificata per test veloce del modello
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
from datetime import datetime

# Aggiunge la directory principale al path
sys.path.append(os.path.abspath('.'))

# Import moduli progetto
import src.config as config
from src.data.dataset import create_dataloaders
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1
from src.models.discriminator import DiscriminatorS1

def quick_train_stage1(num_epochs=10):
    """Training rapido Stage-I per test"""
    
    print("🚀 === TRAINING RAPIDO STAGE-I ===")
    print(f"📊 Epoche: {num_epochs}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📌 Device: {device}")
    
    # Seed per riproducibilità
    torch.manual_seed(42)
    
    # Carica dati
    print("📊 Caricamento dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=config.CSV_PATH,
        img_dir=config.IMAGE_DIR,
        splits_dir=config.SPLITS_DIR,
        config=config,
        img_size=config.STAGE1_IMAGE_SIZE,
        use_augmentation=True
    )
    
    # Inizializza modelli
    print("🧠 Inizializzazione modelli...")
    text_encoder = TextEncoder(fine_tune=config.FINE_TUNE_ENCODER).to(device)
    generator = GeneratorS1(config=config).to(device)
    discriminator = DiscriminatorS1(config=config).to(device)
    
    # Ottimizzatori
    optimizer_g = optim.Adam(
        list(text_encoder.parameters()) + list(generator.parameters()),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, 0.999)
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, 0.999)
    )
    
    # Loss functions
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()
    
    # Directory risultati
    results_dir = "results/quick_training"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    
    # File di log
    log_file = os.path.join(results_dir, "training_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'batch', 'loss_d', 'loss_g', 'loss_g_adv', 'loss_g_l1'])
    
    print("✅ Setup completato, inizio training...")
    
    # Training loop
    for epoch in range(num_epochs):
        text_encoder.train()
        generator.train()
        discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoca {epoch+1}/{num_epochs}"
        )
        
        for batch_idx, batch in progress_bar:
            # Dati
            real_images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = real_images.size(0)
            
            # Encoding testo
            with torch.no_grad():
                cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
            
            # === TRAINING DISCRIMINATORE ===
            discriminator.zero_grad()
            
            # Immagini false
            noise = torch.randn(batch_size, config.Z_DIM, device=device)
            with torch.no_grad():
                fake_images, _ = generator(cls_embedding, hidden_states, noise)
            
            # Predizioni
            real_preds = discriminator(real_images, cls_embedding)
            fake_preds = discriminator(fake_images.detach(), cls_embedding)
            
            # Loss discriminatore (con label smoothing)
            d_loss_real = 0
            for pred in real_preds:
                real_labels = torch.full_like(pred, 0.9, device=device)  # Label smoothing
                d_loss_real += adversarial_loss(pred, real_labels)
            
            d_loss_fake = 0
            for pred in fake_preds:
                fake_labels = torch.full_like(pred, 0.1, device=device)  # Label smoothing
                d_loss_fake += adversarial_loss(pred, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # === TRAINING GENERATORE ===
            generator.zero_grad()
            
            # Re-encoding per gradienti
            cls_embedding, hidden_states = text_encoder(input_ids, attention_mask)
            
            # Nuove immagini
            noise = torch.randn(batch_size, config.Z_DIM, device=device)
            fake_images, _ = generator(cls_embedding, hidden_states, noise)
            gen_preds = discriminator(fake_images, cls_embedding)
            
            # Loss generatore
            g_loss_adv = 0
            for pred in gen_preds:
                real_labels_for_g = torch.full_like(pred, 0.9, device=device)
                g_loss_adv += adversarial_loss(pred, real_labels_for_g)
            
            g_loss_l1 = l1_loss(fake_images, real_images) * config.LAMBDA_L1
            g_loss = g_loss_adv + g_loss_l1
            
            g_loss.backward()
            optimizer_g.step()
            
            # Tracking
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            
            # Log dettagliato
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, batch_idx + 1,
                    d_loss.item(), g_loss.item(),
                    g_loss_adv.item(), g_loss_l1.item()
                ])
            
            # Update progress
            progress_bar.set_postfix(
                D_loss=f"{d_loss.item():.4f}",
                G_loss=f"{g_loss.item():.4f}"
            )
        
        # Media epoca
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        print(f"Epoca {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
        
        # Salva checkpoint finale
        if epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'text_encoder_state_dict': text_encoder.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'config': vars(config)
            }
            
            torch.save(checkpoint, os.path.join(results_dir, "checkpoints", "final_model.pth"))
            print(f"💾 Modello salvato: {results_dir}/checkpoints/final_model.pth")
    
    print("✅ Training rapido completato!")
    print(f"📂 Risultati in: {results_dir}")
    return results_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Training rapido StackGAN Stage-I')
    parser.add_argument('--epochs', type=int, default=10, help='Numero di epoche (default: 10)')
    args = parser.parse_args()
    
    try:
        results_dir = quick_train_stage1(args.epochs)
        print("\n🎮 Ora puoi testare il modello con:")
        print("   python3 gradio_demo.py")
        print(f"   (usando checkpoint in {results_dir})")
        
    except Exception as e:
        print(f"❌ Errore durante il training: {e}")
        raise
