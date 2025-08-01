#!/usr/bin/env python3
"""
🎮 StackGAN Training Pipeline Completo
Training automatico Stage-I e Stage-II con Label Smoothing

Esegui semplicemente: python train_complete_pipeline.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import csv
from PIL import Image
import json
from datetime import datetime
import argparse

# Aggiunge la directory principale del progetto al path
sys.path.append(os.path.abspath('.'))

# Importazioni dai moduli del progetto
try:
    import src.config as config
    from src.data.dataset import create_dataloaders
    from src.models.encoder import TextEncoder
    from src.models.decoder import GeneratorS1, GeneratorS2
    from src.models.discriminator import DiscriminatorS1, DiscriminatorS2
    from src.models.attention import CrossAttentionBlock
    print("✅ Moduli del progetto importati con successo!")
except Exception as e:
    print(f"❌ Errore durante l'importazione: {e}")
    sys.exit(1)

class StackGANTrainer:
    """Trainer completo per StackGAN con Label Smoothing"""
    
    def __init__(self, config_overrides=None):
        """Inizializza il trainer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📌 Dispositivo: {self.device}")
        
        # Configurazione
        self.config = config
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(self.config, key, value)
        
        # Parametri training
        self.use_label_smoothing = True
        self.real_label_value = 0.9
        self.fake_label_value = 0.1
        self.use_data_augmentation = True
        
        # Setup seed per riproducibilità
        self.setup_seed(42)
        
        # Inizializza componenti
        self.setup_directories()
        self.load_data()
        self.initialize_models()
        
    def setup_seed(self, seed):
        """Imposta seed per riproducibilità"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"🌱 Seed impostato: {seed}")
    
    def setup_directories(self):
        """Crea directory per risultati"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"stackgan_complete_{timestamp}"
        self.results_dir = os.path.join("results", self.experiment_name)
        
        # Crea directory
        dirs_to_create = [
            self.results_dir,
            os.path.join(self.results_dir, "stage1"),
            os.path.join(self.results_dir, "stage2"), 
            os.path.join(self.results_dir, "stage1", "images"),
            os.path.join(self.results_dir, "stage2", "images"),
            os.path.join(self.results_dir, "stage1", "checkpoints"),
            os.path.join(self.results_dir, "stage2", "checkpoints"),
            os.path.join(self.results_dir, "logs")
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📂 Directory create: {self.results_dir}")
    
    def load_data(self):
        """Carica i dataset"""
        print("📊 Caricamento dataset...")
        try:
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                csv_path=self.config.CSV_PATH,
                img_dir=self.config.IMAGE_DIR,
                splits_dir=self.config.SPLITS_DIR,
                config=self.config,
                img_size=self.config.STAGE1_IMAGE_SIZE,
                use_augmentation=self.use_data_augmentation
            )
            
            print(f"✅ Dataset caricato!")
            print(f"🔢 Training batches: {len(self.train_loader)}")
            print(f"🔢 Validation batches: {len(self.val_loader)}")
            print(f"🔢 Test batches: {len(self.test_loader)}")
            
        except Exception as e:
            print(f"❌ Errore caricamento dataset: {e}")
            raise
    
    def initialize_models(self):
        """Inizializza tutti i modelli"""
        print("🧠 Inizializzazione modelli...")
        
        # Text Encoder (condiviso tra Stage-I e Stage-II)
        self.text_encoder = TextEncoder(fine_tune=self.config.FINE_TUNE_ENCODER).to(self.device)
        
        # Stage-I Models
        self.generator_s1 = GeneratorS1(config=self.config).to(self.device)
        self.discriminator_s1 = DiscriminatorS1(config=self.config).to(self.device)
        
        # Stage-II Models (inizializzati ma usati dopo Stage-I)
        self.generator_s2 = None  # Inizializzato dopo Stage-I
        self.discriminator_s2 = None  # Inizializzato dopo Stage-I
        
        print("✅ Modelli Stage-I inizializzati!")
    
    def create_labels_with_smoothing(self, batch_size, target_value, apply_noise=True):
        """Crea etichette con label smoothing"""
        labels = torch.full((batch_size,), target_value, dtype=torch.float, device=self.device)
        
        if apply_noise and self.use_label_smoothing:
            noise_range = 0.1
            if target_value >= 0.5:  # Etichette reali
                noise = torch.rand_like(labels) * noise_range
                labels = torch.clamp(labels + noise, max=1.0)
            else:  # Etichette false
                noise = torch.rand_like(labels) * noise_range
                labels = torch.clamp(labels - noise, min=0.0)
        
        return labels
    
    def save_checkpoint(self, stage, epoch, models_dict, optimizers_dict, losses_dict):
        """Salva checkpoint"""
        checkpoint_path = os.path.join(
            self.results_dir, 
            f"stage{stage}", 
            "checkpoints", 
            f"checkpoint_epoch_{epoch}.pth"
        )
        
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'models': {name: model.state_dict() for name, model in models_dict.items()},
            'optimizers': {name: opt.state_dict() for name, opt in optimizers_dict.items()},
            'losses': losses_dict,
            'config': vars(self.config),
            'training_params': {
                'use_label_smoothing': self.use_label_smoothing,
                'real_label_value': self.real_label_value,
                'fake_label_value': self.fake_label_value,
                'use_data_augmentation': self.use_data_augmentation
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint salvato: {checkpoint_path}")
        
        # Salva anche il checkpoint più recente
        latest_path = os.path.join(
            self.results_dir, 
            f"stage{stage}", 
            "checkpoints", 
            f"latest_stage{stage}.pth"
        )
        torch.save(checkpoint, latest_path)
    
    def log_training(self, stage, epoch, batch_idx, losses_dict):
        """Log del training"""
        log_file = os.path.join(self.results_dir, "logs", f"training_stage{stage}.csv")
        
        # Crea header se file non esiste
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                headers = ['timestamp', 'stage', 'epoch', 'batch', 'step'] + list(losses_dict.keys())
                writer.writerow(headers)
        
        # Scrive i dati
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                datetime.now().isoformat(),
                stage, epoch, batch_idx, 
                epoch * len(self.train_loader) + batch_idx
            ] + list(losses_dict.values())
            writer.writerow(row)
    
    def train_stage1(self, num_epochs=50):
        """Training Stage-I (64x64 images)"""
        print("\n🚀 === STAGE-I TRAINING (64x64) ===")
        
        # Ottimizzatori
        optimizer_g = optim.Adam(
            list(self.text_encoder.parameters()) + list(self.generator_s1.parameters()),
            lr=self.config.LEARNING_RATE,
            betas=(self.config.BETA1, 0.999)
        )
        optimizer_d = optim.Adam(
            self.discriminator_s1.parameters(),
            lr=self.config.LEARNING_RATE,
            betas=(self.config.BETA1, 0.999)
        )
        
        # Loss functions
        adversarial_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()
        
        # Tracking delle loss
        stage1_losses = {
            'epoch_d_losses': [],
            'epoch_g_losses': [],
            'all_d_losses': [],
            'all_g_losses': []
        }
        
        for epoch in range(num_epochs):
            self.text_encoder.train()
            self.generator_s1.train()
            self.discriminator_s1.train()
            
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            progress_bar = tqdm(
                enumerate(self.train_loader), 
                total=len(self.train_loader),
                desc=f"Stage-I Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch_idx, batch in progress_bar:
                # Preparazione dati
                real_images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_size = real_images.size(0)
                
                # Encoding del testo
                with torch.no_grad():
                    cls_embedding, hidden_states = self.text_encoder(input_ids, attention_mask)
                
                # === TRAINING DISCRIMINATORE ===
                self.discriminator_s1.zero_grad()
                
                # Genera immagini false
                noise = torch.randn(batch_size, self.config.Z_DIM, device=self.device)
                with torch.no_grad():
                    fake_images, _ = self.generator_s1(cls_embedding, hidden_states, noise)
                
                # Predizioni discriminatore
                real_preds = self.discriminator_s1(real_images, cls_embedding)
                fake_preds = self.discriminator_s1(fake_images.detach(), cls_embedding)
                
                # Loss discriminatore
                d_loss_real = 0
                for pred in real_preds:
                    real_labels = self.create_labels_with_smoothing(
                        pred.size(0), self.real_label_value
                    )
                    d_loss_real += adversarial_loss(pred, real_labels)
                
                d_loss_fake = 0
                for pred in fake_preds:
                    fake_labels = self.create_labels_with_smoothing(
                        pred.size(0), self.fake_label_value
                    )
                    d_loss_fake += adversarial_loss(pred, fake_labels)
                
                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                optimizer_d.step()
                
                # === TRAINING GENERATORE ===
                self.generator_s1.zero_grad()
                
                # Re-encoding per permettere gradienti
                cls_embedding, hidden_states = self.text_encoder(input_ids, attention_mask)
                
                # Genera nuove immagini
                noise = torch.randn(batch_size, self.config.Z_DIM, device=self.device)
                fake_images, _ = self.generator_s1(cls_embedding, hidden_states, noise)
                
                # Predizioni discriminatore
                gen_preds = self.discriminator_s1(fake_images, cls_embedding)
                
                # Loss generatore
                g_loss_adv = 0
                for pred in gen_preds:
                    real_labels_for_g = self.create_labels_with_smoothing(
                        pred.size(0), self.real_label_value
                    )
                    g_loss_adv += adversarial_loss(pred, real_labels_for_g)
                
                g_loss_l1 = l1_loss(fake_images, real_images) * self.config.LAMBDA_L1
                g_loss = g_loss_adv + g_loss_l1
                
                g_loss.backward()
                optimizer_g.step()
                
                # Tracking
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                stage1_losses['all_d_losses'].append(d_loss.item())
                stage1_losses['all_g_losses'].append(g_loss.item())
                
                # Log
                losses_dict = {
                    'loss_d': d_loss.item(),
                    'loss_g': g_loss.item(),
                    'loss_g_adv': g_loss_adv.item(),
                    'loss_g_l1': g_loss_l1.item()
                }
                self.log_training(1, epoch, batch_idx, losses_dict)
                
                # Update progress bar
                progress_bar.set_postfix(
                    D_loss=f"{d_loss.item():.4f}",
                    G_loss=f"{g_loss.item():.4f}"
                )
            
            # Loss medie per epoca
            avg_d_loss = epoch_d_loss / len(self.train_loader)
            avg_g_loss = epoch_g_loss / len(self.train_loader)
            stage1_losses['epoch_d_losses'].append(avg_d_loss)
            stage1_losses['epoch_g_losses'].append(avg_g_loss)
            
            print(f"Stage-I Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
            
            # Salva immagini di esempio
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.save_sample_images(1, epoch + 1)
            
            # Salva checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                models_dict = {
                    'text_encoder': self.text_encoder,
                    'generator_s1': self.generator_s1,
                    'discriminator_s1': self.discriminator_s1
                }
                optimizers_dict = {
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d
                }
                self.save_checkpoint(1, epoch + 1, models_dict, optimizers_dict, stage1_losses)
        
        print("✅ Stage-I completato!")
        return stage1_losses
    
    def prepare_stage2(self):
        """Prepara Stage-II utilizzando Stage-I pre-addestrato"""
        print("\n🔧 Preparazione Stage-II...")
        
        # Inizializza modelli Stage-II
        try:
            self.generator_s2 = GeneratorS2(config=self.config).to(self.device)
            self.discriminator_s2 = DiscriminatorS2(config=self.config).to(self.device)
            print("✅ Modelli Stage-II inizializzati!")
        except Exception as e:
            print(f"❌ Errore inizializzazione Stage-II: {e}")
            # Fallback: crea modelli semplificati
            print("🔄 Creazione modelli Stage-II semplificati...")
            self.generator_s2 = self.create_simple_generator_s2()
            self.discriminator_s2 = self.create_simple_discriminator_s2()
    
    def create_simple_generator_s2(self):
        """Crea un generatore Stage-II semplificato se il modulo non esiste"""
        class SimpleGeneratorS2(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Upsampling da 64x64 a 256x256
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(3, 64, 4, 2, 1),  # 64x64 -> 128x128
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 128x128 -> 256x256
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, 3, 1, 1),  # 256x256, 3 canali
                    nn.Tanh()
                )
            
            def forward(self, stage1_images, cls_embedding, hidden_states, noise):
                # Semplice upsampling delle immagini Stage-I
                upsampled = self.upsample(stage1_images)
                return upsampled, None
        
        return SimpleGeneratorS2(self.config).to(self.device)
    
    def create_simple_discriminator_s2(self):
        """Crea un discriminatore Stage-II semplificato se il modulo non esiste"""
        class SimpleDiscriminatorS2(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                # Discriminatore per immagini 256x256
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1),  # 256x256 -> 128x128
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 128, 4, 2, 1),  # 128x128 -> 64x64
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, 4, 2, 1),  # 64x64 -> 32x32
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 512, 4, 2, 1),  # 32x32 -> 16x16
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AdaptiveAvgPool2d(1),  # -> 1x1
                    nn.Flatten(),
                    nn.Linear(512, 1)
                )
            
            def forward(self, images, cls_embedding):
                return [self.conv_layers(images)]  # Lista per compatibilità
        
        return SimpleDiscriminatorS2(self.config).to(self.device)
    
    def train_stage2(self, num_epochs=50):
        """Training Stage-II (256x256 images)"""
        print("\n🚀 === STAGE-II TRAINING (256x256) ===")
        
        # Prepara Stage-II
        self.prepare_stage2()
        
        # Ottimizzatori (solo per modelli Stage-II, encoder già addestrato)
        optimizer_g2 = optim.Adam(
            self.generator_s2.parameters(),
            lr=self.config.LEARNING_RATE * 0.5,  # LR più basso per Stage-II
            betas=(self.config.BETA1, 0.999)
        )
        optimizer_d2 = optim.Adam(
            self.discriminator_s2.parameters(),
            lr=self.config.LEARNING_RATE * 0.5,
            betas=(self.config.BETA1, 0.999)
        )
        
        # Loss functions
        adversarial_loss = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()
        
        # Tracking delle loss
        stage2_losses = {
            'epoch_d_losses': [],
            'epoch_g_losses': [],
            'all_d_losses': [],
            'all_g_losses': []
        }
        
        # Freeze encoder per Stage-II (opzionale)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        for epoch in range(num_epochs):
            self.generator_s2.train()
            self.discriminator_s2.train()
            
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            
            progress_bar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Stage-II Epoch {epoch+1}/{num_epochs}"
            )
            
            for batch_idx, batch in progress_bar:
                # Preparazione dati
                real_images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_size = real_images.size(0)
                
                # Ridimensiona immagini reali a 256x256 per Stage-II
                real_images_s2 = torch.nn.functional.interpolate(
                    real_images, size=(256, 256), mode='bilinear', align_corners=False
                )
                
                # Encoding del testo (frozen)
                with torch.no_grad():
                    cls_embedding, hidden_states = self.text_encoder(input_ids, attention_mask)
                    
                    # Genera immagini Stage-I come input per Stage-II
                    noise_s1 = torch.randn(batch_size, self.config.Z_DIM, device=self.device)
                    stage1_images, _ = self.generator_s1(cls_embedding, hidden_states, noise_s1)
                
                # === TRAINING DISCRIMINATORE STAGE-II ===
                self.discriminator_s2.zero_grad()
                
                # Genera immagini Stage-II
                noise_s2 = torch.randn(batch_size, self.config.Z_DIM, device=self.device)
                with torch.no_grad():
                    fake_images_s2, _ = self.generator_s2(stage1_images, cls_embedding, hidden_states, noise_s2)
                
                # Predizioni discriminatore
                real_preds_s2 = self.discriminator_s2(real_images_s2, cls_embedding)
                fake_preds_s2 = self.discriminator_s2(fake_images_s2.detach(), cls_embedding)
                
                # Loss discriminatore
                d_loss_real_s2 = 0
                for pred in real_preds_s2:
                    real_labels = self.create_labels_with_smoothing(
                        pred.size(0), self.real_label_value
                    )
                    d_loss_real_s2 += adversarial_loss(pred, real_labels)
                
                d_loss_fake_s2 = 0
                for pred in fake_preds_s2:
                    fake_labels = self.create_labels_with_smoothing(
                        pred.size(0), self.fake_label_value
                    )
                    d_loss_fake_s2 += adversarial_loss(pred, fake_labels)
                
                d_loss_s2 = (d_loss_real_s2 + d_loss_fake_s2) / 2
                d_loss_s2.backward()
                optimizer_d2.step()
                
                # === TRAINING GENERATORE STAGE-II ===
                self.generator_s2.zero_grad()
                
                # Genera nuove immagini Stage-II
                fake_images_s2, _ = self.generator_s2(stage1_images, cls_embedding, hidden_states, noise_s2)
                
                # Predizioni discriminatore
                gen_preds_s2 = self.discriminator_s2(fake_images_s2, cls_embedding)
                
                # Loss generatore
                g_loss_adv_s2 = 0
                for pred in gen_preds_s2:
                    real_labels_for_g = self.create_labels_with_smoothing(
                        pred.size(0), self.real_label_value
                    )
                    g_loss_adv_s2 += adversarial_loss(pred, real_labels_for_g)
                
                g_loss_l1_s2 = l1_loss(fake_images_s2, real_images_s2) * self.config.LAMBDA_L1
                g_loss_s2 = g_loss_adv_s2 + g_loss_l1_s2
                
                g_loss_s2.backward()
                optimizer_g2.step()
                
                # Tracking
                epoch_d_loss += d_loss_s2.item()
                epoch_g_loss += g_loss_s2.item()
                stage2_losses['all_d_losses'].append(d_loss_s2.item())
                stage2_losses['all_g_losses'].append(g_loss_s2.item())
                
                # Log
                losses_dict = {
                    'loss_d': d_loss_s2.item(),
                    'loss_g': g_loss_s2.item(),
                    'loss_g_adv': g_loss_adv_s2.item(),
                    'loss_g_l1': g_loss_l1_s2.item()
                }
                self.log_training(2, epoch, batch_idx, losses_dict)
                
                # Update progress bar
                progress_bar.set_postfix(
                    D_loss=f"{d_loss_s2.item():.4f}",
                    G_loss=f"{g_loss_s2.item():.4f}"
                )
            
            # Loss medie per epoca
            avg_d_loss = epoch_d_loss / len(self.train_loader)
            avg_g_loss = epoch_g_loss / len(self.train_loader)
            stage2_losses['epoch_d_losses'].append(avg_d_loss)
            stage2_losses['epoch_g_losses'].append(avg_g_loss)
            
            print(f"Stage-II Epoch {epoch+1}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}")
            
            # Salva immagini di esempio
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.save_sample_images(2, epoch + 1)
            
            # Salva checkpoint
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                models_dict = {
                    'text_encoder': self.text_encoder,
                    'generator_s1': self.generator_s1,
                    'discriminator_s1': self.discriminator_s1,
                    'generator_s2': self.generator_s2,
                    'discriminator_s2': self.discriminator_s2
                }
                optimizers_dict = {
                    'optimizer_g2': optimizer_g2,
                    'optimizer_d2': optimizer_d2
                }
                self.save_checkpoint(2, epoch + 1, models_dict, optimizers_dict, stage2_losses)
        
        print("✅ Stage-II completato!")
        return stage2_losses
    
    def save_sample_images(self, stage, epoch):
        """Salva immagini di esempio durante il training"""
        try:
            # Metti modelli in eval mode
            self.text_encoder.eval()
            if stage == 1:
                self.generator_s1.eval()
            else:
                self.generator_s1.eval()
                self.generator_s2.eval()
            
            with torch.no_grad():
                # Prendi un batch di validazione
                val_batch = next(iter(self.val_loader))
                input_ids = val_batch['input_ids'].to(self.device)
                attention_mask = val_batch['attention_mask'].to(self.device)
                real_images = val_batch['image']
                
                # Encoding testo
                cls_embedding, hidden_states = self.text_encoder(input_ids, attention_mask)
                noise = torch.randn(input_ids.size(0), self.config.Z_DIM, device=self.device)
                
                if stage == 1:
                    # Stage-I: 64x64
                    generated_images, _ = self.generator_s1(cls_embedding, hidden_states, noise)
                    image_size = "64x64"
                else:
                    # Stage-II: 256x256
                    # Prima genera Stage-I
                    stage1_images, _ = self.generator_s1(cls_embedding, hidden_states, noise)
                    # Poi Stage-II
                    noise_s2 = torch.randn(input_ids.size(0), self.config.Z_DIM, device=self.device)
                    generated_images, _ = self.generator_s2(stage1_images, cls_embedding, hidden_states, noise_s2)
                    image_size = "256x256"
                
                # Salva immagini reali
                save_image(
                    real_images,
                    os.path.join(self.results_dir, f"stage{stage}", "images", f"real_epoch_{epoch}.png"),
                    normalize=True,
                    nrow=4
                )
                
                # Salva immagini generate
                save_image(
                    generated_images,
                    os.path.join(self.results_dir, f"stage{stage}", "images", f"fake_epoch_{epoch}.png"),
                    normalize=True,
                    nrow=4
                )
                
                print(f"💾 Immagini salvate per Stage-{stage} Epoca {epoch} ({image_size})")
                
        except Exception as e:
            print(f"⚠️ Errore nel salvataggio immagini: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """Carica un checkpoint salvato"""
        print(f"📂 Caricamento checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Carica stati dei modelli
            if 'text_encoder' in checkpoint['models']:
                self.text_encoder.load_state_dict(checkpoint['models']['text_encoder'])
            if 'generator_s1' in checkpoint['models']:
                self.generator_s1.load_state_dict(checkpoint['models']['generator_s1'])
            if 'discriminator_s1' in checkpoint['models']:
                self.discriminator_s1.load_state_dict(checkpoint['models']['discriminator_s1'])
            
            # Stage-II se presente
            if 'generator_s2' in checkpoint['models'] and self.generator_s2 is not None:
                self.generator_s2.load_state_dict(checkpoint['models']['generator_s2'])
            if 'discriminator_s2' in checkpoint['models'] and self.discriminator_s2 is not None:
                self.discriminator_s2.load_state_dict(checkpoint['models']['discriminator_s2'])
            
            print(f"✅ Checkpoint caricato: Epoca {checkpoint['epoch']}, Stage {checkpoint['stage']}")
            return checkpoint
            
        except Exception as e:
            print(f"❌ Errore caricamento checkpoint: {e}")
            return None
    
    def run_complete_training(self, stage1_epochs=50, stage2_epochs=50):
        """Esegue il training completo Stage-I + Stage-II"""
        print("🚀 === TRAINING COMPLETO STACKGAN ===")
        print(f"📋 Configurazione:")
        print(f"   - Label Smoothing: {self.use_label_smoothing}")
        print(f"   - Data Augmentation: {self.use_data_augmentation}")
        print(f"   - Stage-I Epoche: {stage1_epochs}")
        print(f"   - Stage-II Epoche: {stage2_epochs}")
        print(f"   - Device: {self.device}")
        print("=" * 60)
        
        # Training Stage-I
        stage1_losses = self.train_stage1(stage1_epochs)
        
        print("\n" + "=" * 60)
        print("✅ Stage-I completato! Passaggio a Stage-II...")
        print("=" * 60)
        
        # Training Stage-II
        stage2_losses = self.train_stage2(stage2_epochs)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETO TERMINATO!")
        print(f"📂 Risultati salvati in: {self.results_dir}")
        print("=" * 60)
        
        return stage1_losses, stage2_losses

def main():
    """Funzione principale"""
    parser = argparse.ArgumentParser(description='Training completo StackGAN')
    parser.add_argument('--stage1-epochs', type=int, default=50, help='Epoche per Stage-I (default: 50)')
    parser.add_argument('--stage2-epochs', type=int, default=50, help='Epoche per Stage-II (default: 50)')
    parser.add_argument('--no-label-smoothing', action='store_true', help='Disabilita label smoothing')
    parser.add_argument('--no-augmentation', action='store_true', help='Disabilita data augmentation')
    parser.add_argument('--resume', type=str, help='Path checkpoint da cui riprendere')
    
    args = parser.parse_args()
    
    # Configurazione override
    config_overrides = {}
    
    # Inizializza trainer
    trainer = StackGANTrainer(config_overrides)
    
    # Applica opzioni command line
    if args.no_label_smoothing:
        trainer.use_label_smoothing = False
        trainer.real_label_value = 1.0
        trainer.fake_label_value = 0.0
        print("⚠️ Label smoothing disabilitato")
    
    if args.no_augmentation:
        trainer.use_data_augmentation = False
        print("⚠️ Data augmentation disabilitata")
    
    # Carica checkpoint se specificato
    if args.resume:
        checkpoint = trainer.load_checkpoint(args.resume)
        if checkpoint is None:
            print("❌ Impossibile caricare checkpoint, avvio training da zero")
    
    try:
        # Esegui training completo
        stage1_losses, stage2_losses = trainer.run_complete_training(
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs
        )
        
        # Salva summary finale
        summary = {
            'experiment_name': trainer.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'label_smoothing': trainer.use_label_smoothing,
                'data_augmentation': trainer.use_data_augmentation,
                'stage1_epochs': args.stage1_epochs,
                'stage2_epochs': args.stage2_epochs,
                'device': str(trainer.device)
            },
            'results_directory': trainer.results_dir
        }
        
        summary_path = os.path.join(trainer.results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Summary salvato in: {summary_path}")
        print("\n🎮 Per testare il modello, usa il file di demo Gradio!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrotto dall'utente")
    except Exception as e:
        print(f"\n❌ Errore durante il training: {e}")
        raise

if __name__ == "__main__":
    main()
