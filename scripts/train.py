import os
import sys

# Aggiungi la root del progetto al path di Python per risolvere i problemi di importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Importa i moduli del progetto
import src.config as config
from src.data.dataset import PokemonDataset, create_dataloaders
from src.models.model import PikaPikaGen
from src.models.loss import PerceptualLoss, CombinedLoss

def train(cfg):
    """Funzione principale per l'addestramento del modello."""
    
    # Setup
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    
    # Crea le directory di output se non esistono
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)

    # Dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME),
        img_dir=cfg.IMAGE_DIR, # CORREZIONE: Il percorso corretto è direttamente alla radice
        splits_dir=cfg.SPLITS_DIR,
        config=cfg
    )
    print("Dataloaders creati con successo.")

    # Modello
    model = PikaPikaGen(config).to(device)
    print("Modello PikaPikaGen creato con successo.")

    # Ottimizzatore e Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Seleziona la funzione di loss in base alla configurazione
    if cfg.LOSS_FUNCTION == 'Combined':
        criterion = CombinedLoss(l1_weight=cfg.L1_LOSS_WEIGHT, device=device).to(device)
        print(f"Utilizzo della Combined Loss con peso L1: {cfg.L1_LOSS_WEIGHT}")
    elif cfg.LOSS_FUNCTION == 'Perceptual':
        criterion = PerceptualLoss(device=device).to(device)
        print("Utilizzo della Perceptual Loss.")
    else:
        criterion = nn.MSELoss() # Fallback a MSE
        print("Utilizzo della MSE Loss.")

    # Ciclo di addestramento
    print("Inizio dell'addestramento...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")

        for batch in progress_bar:
            if batch is None:
                # Questo batch è stato saltato da collate_fn perché probabilmente tutte le immagini non sono state trovate
                print("Attenzione: saltato un batch di training vuoto.")
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device)

            # Forward pass
            generated_images, _ = model(input_ids, attention_mask)
            
            # Calcolo della loss
            loss = criterion(generated_images, real_images)
            
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.EPOCHS}, Average Loss: {avg_loss:.4f}")

        # Salvataggio immagini generate
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            model.eval()
            with torch.no_grad():
                # Cerca un batch di validazione valido, poiché alcuni potrebbero essere None
                val_batch = None
                for batch in val_loader:
                    if batch is not None:
                        val_batch = batch
                        break
                
                if val_batch is None:
                    print("Attenzione: Nessun batch valido trovato nel validation loader. Salto il salvataggio delle immagini.")
                    continue

                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                
                generated_images, _ = model(input_ids, attention_mask)
                
                # Salva le immagini reali e generate per confronto
                save_image(
                    val_batch['image'], 
                    os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_images_epoch_{epoch+1}.png"), 
                    normalize=True
                )
                save_image(
                    generated_images, 
                    os.path.join(cfg.GENERATED_IMAGE_DIR, f"generated_images_epoch_{epoch+1}.png"),
                    normalize=True
                )
            print(f"Immagini di esempio salvate per l'epoca {epoch+1}")

        # Salvataggio checkpoint
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")

    print("Addestramento completato.")

if __name__ == '__main__':
    # Utilizziamo direttamente la configurazione importata
    train(config)
