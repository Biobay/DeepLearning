# src/data/dataset.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, default_collate
from transformers import BertTokenizer
# Importiamo 'transforms' con l'alias T per chiarezza
from torchvision import transforms as T

def collate_fn(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None
    return default_collate(valid_batch)


class PokemonDataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform, max_seq_len):
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRORE CRITICO: File CSV non trovato: {csv_path}")
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.missing_images_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            national_number = int(row['national_number'])
            description = str(row.iloc[-1])
        except (ValueError, KeyError):
            return None

        inputs = self.tokenizer(
            description, return_tensors='pt', max_length=self.max_seq_len,
            padding='max_length', truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        img_filename = f"{national_number:03d}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert("RGBA")
            background = Image.new('RGBA', image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image).convert('RGB')
            # La trasformazione (con augmentation) viene applicata qui
            image = self.transform(alpha_composite)
        except FileNotFoundError:
            self.missing_images_count += 1
            return None 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "description": description
        }


def create_dataloaders(csv_path, img_dir, splits_dir, config):
    print("-" * 50)
    print("Inizio creazione Dataloaders con DATA AUGMENTATION...")
    
    # =============================================================================
    # ## MODIFICA CHIAVE QUI: PIPELINE DI TRASFORMAZIONE ##
    # =============================================================================
    # Creiamo due pipeline di trasformazioni separate: una per il training
    # (con augmentation) e una per la validazione/test (senza).
    
    # Trasformazioni per il training set
    train_transform = T.Compose([
        T.Resize((config.MODEL_INTERNAL_SIZE, config.MODEL_INTERNAL_SIZE)),
        # Aggiungi qui le tue augmentation
        T.RandomHorizontalFlip(p=0.5), # Ribalta orizzontalmente il 50% delle immagini
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Varia leggermente i colori
        # T.RandomRotation(degrees=10), # Puoi aggiungere anche piccole rotazioni
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Trasformazioni per validation e test set (nessuna augmentation)
    val_test_transform = T.Compose([
        T.Resize((config.MODEL_INTERNAL_SIZE, config.MODEL_INTERNAL_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # =============================================================================
    
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    # Creiamo istanze separate del dataset con le rispettive trasformazioni
    # Nota: questo è inefficiente, sarebbe meglio passare la transform al Subset.
    # Per semplicità, creiamo un'istanza per ogni split.
    
    # Gestione degli split
    # ... (il tuo codice per caricare o creare train_indices, val_indices, etc. va qui)
    # ...
    # Assumiamo che train_indices, val_indices, test_indices siano stati caricati
    train_indices = np.load(os.path.join(splits_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(splits_dir, 'val_indices.npy'))
    # ...

    # Crea un dataset per ogni split, passando la transform corretta
    train_dataset = PokemonDataset(csv_path, img_dir, tokenizer, train_transform, 128)
    val_dataset = PokemonDataset(csv_path, img_dir, tokenizer, val_test_transform, 128)
    test_dataset = PokemonDataset(csv_path, img_dir, tokenizer, val_test_transform, 128)
    
    # Usa Subset per selezionare gli indici corretti per ogni dataset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    # test_subset = Subset(test_dataset, test_indices) # Se ti serve il test set
    
    print("\n--- Riepilogo Suddivisione Dataset ---")
    print(f"  -> Training set:   {len(train_subset)} campioni")
    print(f"  -> Validation set: {len(val_subset)} campioni")
    print("-" * 50)

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    # test_loader = DataLoader(test_subset, ...)

    return train_loader, val_loader, None # Restituisce None per test_loader per ora