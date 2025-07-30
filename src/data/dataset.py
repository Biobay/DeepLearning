import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset, default_collate
from transformers import BertTokenizer
from torchvision import transforms

# Funzione collate definita a livello di modulo.
# Questo è necessario perché le funzioni devono essere "picklable" (serializzabili)
# per funzionare con multiprocessing (num_workers > 0).
def collate_fn(batch):
    """
    Funzione di collazione personalizzata che gestisce campioni 'None'.
    Se un'immagine non viene trovata, PokemonDataset restituisce None.
    Questa funzione filtra questi None prima di creare il batch.
    """
    # Filtra tutti gli elementi che sono None
    valid_batch = [item for item in batch if item is not None]
    
    # Se dopo il filtraggio il batch è vuoto, restituisce None.
    # Questo verrà gestito nel ciclo di training.
    if not valid_batch:
        return None
        
    # Usa la funzione di collazione di default di PyTorch sul batch pulito.
    return default_collate(valid_batch)


class PokemonDataset(Dataset):
    """Dataset per caricare descrizioni testuali e immagini di Pokémon."""
    def __init__(self, csv_path, img_dir, tokenizer, transform, max_seq_len):
        try:
            # NOTA: Mantenute le opzioni di parsing specifiche per il tuo CSV.
            # Se il CSV cambia, queste opzioni potrebbero dover essere adattate.
            self.data = pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRORE CRITICO: File CSV non trovato al percorso: {csv_path}")
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # Contatore per il debug, per sapere quante immagini non sono state trovate.
        self.missing_images_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Estrazione dei dati dalla riga del CSV
        try:
            national_number = int(row['national_number'])
            description = row.iloc[-1] # Prende l'ultima colonna come descrizione
        except (ValueError, KeyError) as e:
            print(f"Attenzione: Errore nel leggere la riga {idx} del CSV. Salto il campione. Dettagli: {e}")
            return None

        # Assicura che la descrizione sia una stringa
        if not isinstance(description, str):
            description = str(description)

        # Tokenizzazione del testo
        inputs = self.tokenizer(
            description,
            return_tensors='pt',
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # Costruzione del path dell'immagine
        img_filename = f"{national_number:03d}.png" # Formatta il numero con zeri iniziali (es. 1 -> 001.png)
        img_path = os.path.join(self.img_dir, img_filename)
        
        try:
            # Caricamento e processing dell'immagine
            image = Image.open(img_path).convert("RGBA")
            # Gestisce la trasparenza componendo l'immagine su uno sfondo bianco
            background = Image.new('RGBA', image.size, (255, 255, 255))
            alpha_composite = Image.alpha_composite(background, image).convert('RGB')
            image = self.transform(alpha_composite)
        except FileNotFoundError:
            # Se l'immagine non viene trovata, incrementa il contatore e restituisce None.
            # Sarà gestito da `collate_fn`.
            self.missing_images_count += 1
            return None 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "description": description # Utile per il debug
        }


def create_dataloaders(csv_path, img_dir, splits_dir, config):
    """
    Crea e restituisce i DataLoader per training, validazione e test.
    Gestisce la creazione o il caricamento degli split del dataset.
    """
    print("-" * 50)
    print("Inizio creazione Dataloaders...")
    print(f"  -> Path CSV principale: {csv_path}")
    print(f"  -> Path cartella immagini: {img_dir}")
    print(f"  -> Path cartella degli split: {splits_dir}")
    if not os.path.exists(img_dir):
        print(f"!!! ATTENZIONE !!! La cartella delle immagini '{img_dir}' non esiste.")
    print("-" * 50)
    
    # --- MODIFICA CRUCIALE: USA LA DIMENSIONE INTERNA DEL MODELLO ---
    # Il dataset viene preparato alla dimensione che la U-Net si aspetta (es. 256x256),
    # non alla dimensione finale dell'output (215x215).
    transform = transforms.Compose([
        transforms.Resize((config.MODEL_INTERNAL_SIZE, config.MODEL_INTERNAL_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    full_dataset = PokemonDataset(
        csv_path=csv_path,
        img_dir=img_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_seq_len=128 # Valore standard, potrebbe essere messo in config
    )

    # Gestione degli split: carica se esistono, altrimenti crea e salva
    train_indices_path = os.path.join(splits_dir, 'train_indices.npy')
    val_indices_path = os.path.join(splits_dir, 'val_indices.npy')

    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path):
        print("Caricamento degli indici di split esistenti...")
        train_indices = np.load(train_indices_path)
        val_indices = np.load(val_indices_path)
        
        all_indices = set(range(len(full_dataset)))
        train_set_indices = set(train_indices)
        val_set_indices = set(val_indices)
        test_indices = np.array(list(all_indices - train_set_indices - val_set_indices), dtype=np.int64)
    else:
        print("Creazione di nuovi split di dati (70/10/20) e salvataggio...")
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.7)
        val_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - val_size
        
        # CORREZIONE: `random_split` va chiamato su `full_dataset`
        train_split, val_split, test_split = random_split(full_dataset, [train_size, val_size, test_size])
        train_indices = train_split.indices
        val_indices = val_split.indices
        test_indices = test_split.indices
        
        os.makedirs(splits_dir, exist_ok=True)
        np.save(os.path.join(splits_dir, 'train_indices.npy'), np.array(train_indices))
        np.save(os.path.join(splits_dir, 'val_indices.npy'), np.array(val_indices))
        np.save(os.path.join(splits_dir, 'test_indices.npy'), np.array(test_indices))

    # Creazione dei Subset basati sugli indici
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print("\n--- Riepilogo Suddivisione Dataset ---")
    print(f"  -> Training set:   {len(train_dataset)} campioni")
    print(f"  -> Validation set: {len(val_dataset)} campioni")
    print(f"  -> Test set:         {len(test_dataset)} campioni")
    print("-" * 50)

    # Creazione dei DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    
    # Stampa un riepilogo finale delle immagini mancanti (se ce ne sono)
    if full_dataset.missing_images_count > 0:
        print(f"\n!!! RIASSUNTO DEBUG: Trovate {full_dataset.missing_images_count} immagini mancanti su {len(full_dataset)} totali.")
        print("!!! Controllare che il path in `config.py` e i nomi dei file siano corretti.")
        print("-" * 50)

    return train_loader, val_loader, test_loader