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
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

class PokemonDataset(Dataset):
    """Dataset per caricare descrizioni testuali e immagini di Pokémon."""
    def __init__(self, csv_path, img_dir, tokenizer, transform, max_seq_len):
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-16-le', sep='\t', engine='python')
        except FileNotFoundError:
            raise FileNotFoundError(f"File CSV non trovato al percorso: {csv_path}")
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        national_number = row['national_number']
        description = row.iloc[-1]

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
            # print(f"Attenzione: Immagine non trovata: {img_path}. Verrà saltata.")
            return None 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "description": description
        }

def create_dataloaders(csv_path, img_dir, splits_dir, config, img_size=None):
    """
    Crea e restituisce i DataLoader per training, validazione e test.
    
    Args:
        img_size: Se specificato, usa questa dimensione invece di STAGE1_IMAGE_SIZE
    """
    # Determina la dimensione dell'immagine
    target_size = img_size if img_size is not None else config.STAGE1_IMAGE_SIZE
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    dataset = PokemonDataset(
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
        
        all_indices = set(range(len(dataset)))
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_indices = np.array(list(all_indices - train_set - val_set))

        train_dataset = Subset(dataset, train_indices.tolist())
        val_dataset = Subset(dataset, val_indices.tolist())
        test_dataset = Subset(dataset, test_indices.tolist())
    else:
        print("Creazione di nuovi split di dati...")
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        val_size = int(dataset_size * 0.1)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        os.makedirs(splits_dir, exist_ok=True)
        np.save(os.path.join(splits_dir, 'train_indices.npy'), train_dataset.indices)
        np.save(os.path.join(splits_dir, 'val_indices.npy'), val_dataset.indices)
        np.save(os.path.join(splits_dir, 'test_indices.npy'), test_dataset.indices)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
