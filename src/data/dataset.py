# src/data/dataset.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, default_collate
from transformers import BertTokenizer
from torchvision import transforms as T # Usiamo l'alias T per chiarezza

# La funzione collate_fn rimane invariata
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

# La classe PokemonDataset rimane invariata
class PokemonDataset(Dataset):
    def __init__(self, csv_path, img_dir, tokenizer, transform, max_seq_len):
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"File CSV non trovato al percorso: {csv_path}")
        self.img_dir, self.tokenizer, self.transform, self.max_seq_len = img_dir, tokenizer, transform, max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            num = int(row['national_number'])
            desc = str(row.iloc[-1])
        except (KeyError, ValueError): return None

        inputs = self.tokenizer(desc, return_tensors='pt', max_length=self.max_seq_len, padding='max_length', truncation=True)
        img_path = os.path.join(self.img_dir, f"{num:03d}.png")
        
        try:
            image = Image.open(img_path).convert("RGBA")
            bg = Image.new('RGBA', image.size, (255, 255, 255))
            image = Image.alpha_composite(bg, image).convert('RGB')
            image = self.transform(image)
        except FileNotFoundError:
            return None 

        return {"input_ids": inputs['input_ids'].squeeze(0), "attention_mask": inputs['attention_mask'].squeeze(0), "image": image, "description": desc}

# --- FUNZIONE create_dataloaders MODIFICATA ---
def create_dataloaders(csv_path, img_dir, splits_dir, config, img_size=None, use_augmentation=True):
    """
    Crea i DataLoader, applicando la Data Augmentation solo al training set.
    
    Args:
        csv_path: Percorso al file CSV
        img_dir: Directory delle immagini
        splits_dir: Directory degli split
        config: Oggetto di configurazione
        img_size: Dimensione target delle immagini (opzionale)
        use_augmentation: Se True, applica data augmentation al training set
    """
    target_size = img_size if img_size is not None else config.STAGE1_IMAGE_SIZE
    
    # --- PIPELINE DI TRASFORMAZIONE CON AUGMENTATION ---
    if use_augmentation:
        train_transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.RandomHorizontalFlip(p=0.5), # Ribalta il 50% delle immagini
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Varia i colori
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # Trasformazioni base senza augmentation per il training
        train_transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # --- PIPELINE DI TRASFORMAZIONE SENZA AUGMENTATION ---
    val_test_transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    # Creiamo due istanze di dataset, una per il training e una per la validazione
    train_full_dataset = PokemonDataset(csv_path, img_dir, tokenizer, train_transform, 128)
    val_test_full_dataset = PokemonDataset(csv_path, img_dir, tokenizer, val_test_transform, 128)

    # Carica o crea gli indici per gli split
    train_indices_path = os.path.join(splits_dir, 'train_indices.npy')
    val_indices_path = os.path.join(splits_dir, 'val_indices.npy')
    test_indices_path = os.path.join(splits_dir, 'test_indices.npy') # Aggiunto per completezza

    if os.path.exists(train_indices_path):
        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        test_indices = np.load(test_indices_path)
    else:
        # Codice per creare gli split (se non esistono)
        dataset_size = len(pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python'))
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size:]
        os.makedirs(splits_dir, exist_ok=True)
        np.save(train_indices_path, train_indices)
        np.save(val_indices_path, val_indices)
        np.save(test_indices_path, test_indices)

    # Crea i Subset usando il dataset corretto per ogni split
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_test_full_dataset, val_indices)
    test_dataset = Subset(val_test_full_dataset, test_indices)

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader