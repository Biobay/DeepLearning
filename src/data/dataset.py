# dataset.py (Versione Migliorata con Debug Integrato)

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, default_collate
from transformers import BertTokenizer
from torchvision import transforms

# Funzione collate definita a livello di modulo per essere "picklable"
def collate_fn(batch):
    # Filtra gli elementi None che sono il risultato di immagini non trovate
    valid_batch = [b for b in batch if b is not None]
    if not valid_batch:
        return None # Se l'intero batch è invalido, restituisce None
    return default_collate(valid_batch)

class PokemonDataset(Dataset):
    """Dataset per caricare descrizioni testuali e immagini di Pokémon."""
    def __init__(self, csv_path, img_dir, tokenizer, transform, max_seq_len):
        try:
            # NOTA: il tuo CSV potrebbe richiedere opzioni di parsing specifiche
            self.data = pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRORE CRITICO: File CSV non trovato al percorso: {csv_path}")
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # Contatore per le immagini non trovate (per il debug)
        self.missing_images_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        national_number = row['national_number']
        description = row.iloc[-1] # Prende l'ultima colonna come descrizione

        if not isinstance(description, str):
            description = str(description)

        inputs = self.tokenizer(
            description,
            return_tensors='pt',
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        img_filename = f"{int(national_number):03d}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            image = Image.open(img_path).convert("RGBA")
            background = Image.new('RGBA', image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image).convert('RGB')
            image = self.transform(alpha_composite)
        except FileNotFoundError:
            self.missing_images_count += 1
            # Restituisce None, che verrà gestito da collate_fn
            return None 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "description": description
        }

def create_dataloaders(csv_path, img_dir, splits_dir, config):
    """
    Crea e restituisce i DataLoader per training, validazione e test.
    """
    # =============================================================================
    # ## DEBUG INIZIALE ##
    # =============================================================================
    print("-" * 50)
    print("Inizio creazione Dataloaders...")
    print(f"  -> Path CSV principale: {csv_path}")
    print(f"  -> Path cartella immagini: {img_dir}")
    print(f"  -> Path cartella degli split: {splits_dir}")
    if not os.path.exists(img_dir):
        print(f"!!! ATTENZIONE !!! La cartella delle immagini '{img_dir}' non esiste.")
    print("-" * 50)
    # =============================================================================
    
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    full_dataset = PokemonDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_seq_len=128
    )

    train_indices_path = os.path.join(splits_dir, 'train_indices.npy')
    val_indices_path = os.path.join(splits_dir, 'val_indices.npy')

    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path):
        print("Caricamento degli indici di split esistenti...")
        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        
        all_indices = set(range(len(full_dataset)))
        train_set_indices = set(train_indices)
        val_set_indices = set(val_indices)
        test_indices = np.array(list(all_indices - train_set_indices - val_set_indices))
    else:
        print("Creazione di nuovi split di dati (80/10/10)...")
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - val_size
        
        train_split, val_split, test_split = random_split(dataset, [train_size, val_size, test_size])
        train_indices = train_split.indices
        val_indices = val_split.indices
        test_indices = test_split.indices
        
        os.makedirs(splits_dir, exist_ok=True)
        np.save(os.path.join(splits_dir, 'train_indices.npy'), train_indices)
        np.save(os.path.join(splits_dir, 'val_indices.npy'), val_indices)
        np.save(os.path.join(splits_dir, 'test_indices.npy'), test_indices)

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print("\n--- Riepilogo Suddivisione Dataset ---")
    print(f"  -> Training set:   {len(train_dataset)} campioni")
    print(f"  -> Validation set: {len(val_dataset)} campioni")
    print(f"  -> Test set:         {len(test_dataset)} campioni")
    print("-" * 50)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn
    )
    
    # Stampa un riepilogo finale delle immagini mancanti
    if full_dataset.missing_images_count > 0:
        print(f"\n!!! RIASSUNTO DEBUG: Trovate {full_dataset.missing_images_count} immagini mancanti su {len(full_dataset)} totali.")
        print("!!! Controllare che il path in 'config.py' e i nomi dei file siano corretti.")
        print("-" * 50)

    return train_loader, val_loader, test_loader