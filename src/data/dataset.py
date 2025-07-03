import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from transformers import AutoTokenizer
from torchvision import transforms

class PokemonDataset(Dataset):
    """
    Dataset custom per caricare descrizioni testuali e immagini di Pokémon.

    Args:
        csv_path (str): Percorso al file CSV contenente i metadati.
        images_dir (str): Directory contenente le immagini dei Pokémon.
        tokenizer_name (str): Nome del tokenizer pre-addestrato da usare (es. 'prajjwal1/bert-mini').
        max_length (int): Lunghezza massima per il padding delle sequenze di testo.
        image_size (int): Dimensione a cui ridimensionare le immagini (altezza e larghezza).
    """
    def __init__(self, csv_path, images_dir, tokenizer_name='prajjwal1/bert-mini', max_length=128, image_size=215):
        # --- MODIFICA CHIAVE ---
        # Implementazione basata sul feedback dell'utente.
        # Il file CSV ha un'intestazione, è separato da tab e richiede la codifica 'utf-16 LE'.
        # Usiamo 'usecols' per caricare solo le colonne necessarie in modo efficiente.
        try:
            self.df = pd.read_csv(
                csv_path,
                delimiter='\t',
                encoding='utf-16 LE',
                usecols=['national_number', 'description']
            )
        except FileNotFoundError:
            raise ValueError(f"File CSV non trovato al percorso: {csv_path}")
        except KeyError as e:
            raise ValueError(f"Una delle colonne richieste non è stata trovata nel CSV. Assicurati che il file contenga 'national_number' e 'description'. Errore: {e}")
        except Exception as e:
            # Forniamo un messaggio di errore più utile per altri problemi
            raise ValueError(
                f"Impossibile leggere il file CSV in '{csv_path}'. "
                f"Verifica il formato, la codifica ('utf-16 LE') e il separatore (tab). Errore originale: {e}"
            )
        # --- FINE MODIFICA ---

        self.images_dir = images_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # Definisci le trasformazioni per le immagini
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), # Converte in tensore e normalizza i pixel a [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizza in [-1, 1]
        ])

        # Filtra i dati per cui esiste un'immagine corrispondente
        self._filter_available_data()

    def _filter_available_data(self):
        """Filtra il DataFrame per mantenere solo le righe con un'immagine corrispondente."""
        available_images = {f for f in os.listdir(self.images_dir) if f.endswith('.png')}
        
        # Costruisce il nome del file immagine dal numero nazionale del Pokémon.
        self.df['image_filename'] = self.df['national_number'].apply(lambda n: f'{n:03d}.png')

        self.df = self.df[self.df['image_filename'].isin(available_images)]
        print(f"Trovati {len(self.df)} Pokémon con immagini e descrizioni corrispondenti.")

    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.df)

    def _convert_rgba_to_rgb(self, image):
        """Converte un'immagine da RGBA a RGB usando uno sfondo bianco."""
        if image.mode == 'RGBA':
            # Aggiungiamo # type: ignore per sopprimere il falso positivo del linter.
            # Il codice è corretto: 'RGB' richiede una tupla di 3 valori per il colore.
            background = Image.new('RGB', image.size, (255, 255, 255)) # type: ignore
            background.paste(image, mask=image.split()[3]) # Usa il canale alpha come maschera
            return background
        return image.convert('RGB')

    def __getitem__(self, idx):
        """
        Recupera un campione (testo e immagine) dal dataset all'indice specificato.
        """
        row = self.df.iloc[idx]
        
        # --- Preprocessing del Testo ---
        description = row['description']
        inputs = self.tokenizer(description, 
                                max_length=self.max_length, 
                                padding='max_length', 
                                truncation=True, 
                                return_tensors='pt')
        
        input_ids = inputs['input_ids'].squeeze(0) # Rimuove la dimensione del batch
        attention_mask = inputs['attention_mask'].squeeze(0)

        # --- Preprocessing dell'Immagine ---
        img_path = os.path.join(self.images_dir, row['image_filename'])
        image = Image.open(img_path)
        image = self._convert_rgba_to_rgb(image)
        image_tensor = self.image_transform(image)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image': image_tensor
        }

def create_dataloaders(csv_path, images_dir, batch_size, train_split=0.8, val_split=0.1):
    """Crea e restituisce i DataLoader per training, validazione e test."""
    dataset = PokemonDataset(csv_path=csv_path, images_dir=images_dir)

    # Calcola le dimensioni degli split
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Esegui lo split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) # Per la riproducibilità
    )

    # Crea i DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"DataLoader creati:")
    print(f"  - Training: {len(train_dataset)} campioni")
    print(f"  - Validazione: {len(val_dataset)} campioni")
    print(f"  - Test: {len(test_dataset)} campioni")

    return train_loader, val_loader, test_loader, dataset.tokenizer
