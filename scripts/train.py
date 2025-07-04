import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel

# --- 1. Impostazioni e Iperparametri ---

def get_args():
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(description="Addestra un modello Text-to-Image per generare Pokémon.")
    
    parser.add_argument('--data_dir', type=str, default='data', help='Directory contenente pokemon.csv e la cartella delle immagini.')
    parser.add_argument('--image_dir', type=str, default='small_images', help='Directory contenente le immagini dei Pokémon.')
    parser.add_argument('--output_dir', type=str, default='../models', help='Directory dove salvare i modelli e gli output.')
    parser.add_argument('--epochs', type=int, default=100, help='Numero di epoche di addestramento.')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate per l\'ottimizzatore Adam.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 per l\'ottimizzatore Adam.')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimensione del vettore di rumore casuale.')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimensione degli embedding del testo (BERT-mini).')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Lunghezza massima delle sequenze di testo.')
    parser.add_argument('--img_size', type=int, default=215, help='Dimensione delle immagini generate (altezza e larghezza).')
    parser.add_argument('--save_interval', type=int, default=10, help='Ogni quante epoche salvare un\'immagine di esempio.')

    return parser.parse_args()

# --- 2. Dataset e Pre-processing ---

class PokemonDataset(Dataset):
    """Dataset per caricare descrizioni testuali e immagini di Pokémon."""
    def __init__(self, csv_file, img_dir, tokenizer, max_seq_len, transform):
        # Rimuoviamo header=None per leggere i nomi delle colonne dalla prima riga
        self.data = pd.read_csv(csv_file, encoding='utf-16-le', sep='\t', engine='python')
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- Selezione Colonne come da richiesta ---
        # Usa la colonna 'national_number' per l'ID e l'ultima colonna per la descrizione
        national_number = row['national_number']
        description = row.iloc[-1] 

        # Preprocessing Testo
        if not isinstance(description, str):
            description = str(description) # Assicura che la descrizione sia una stringa

        inputs = self.tokenizer(
            description,
            return_tensors='pt',
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # Preprocessing Immagine
        # Formatta il national_number a 3 cifre con zeri (es. 1 -> "001")
        img_filename = f"{int(national_number):03d}.png"
        img_name = os.path.join(self.img_dir, img_filename)
        
        image = Image.open(img_name).convert("RGBA")

        # Gestione trasparenza: blend con sfondo bianco
        background = Image.new('RGBA', image.size, (255, 255, 255))
        alpha_composite = Image.alpha_composite(background, image).convert('RGB')
        
        image = self.transform(alpha_composite)

        return input_ids, attention_mask, image

# --- 3. Architettura del Modello ---

class TextEncoder(nn.Module):
    """Encoder basato su BERT per estrarre feature dal testo."""
    def __init__(self, model_name='prajjwal1/bert-mini', fine_tune=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state # [batch_size, seq_len, embedding_dim]

class ImageDecoder(nn.Module):
    """Decoder CNN (Generatore) per creare un'immagine da un vettore di contesto."""
    def __init__(self, context_dim, noise_dim, img_channels=3):
        super().__init__()
        self.context_dim = context_dim
        self.noise_dim = noise_dim
        input_dim = context_dim + noise_dim

        # Proiezione e Rimodellamento iniziale
        self.project = nn.Sequential(
            nn.Linear(input_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )

        # Blocchi di Upsampling con ConvTranspose2d
        self.decoder = nn.Sequential(
            # Input: 512 x 4 x 4
            self._make_block(512, 256, kernel_size=4, stride=2, padding=1), # -> 256 x 8 x 8
            self._make_block(256, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 16 x 16
            self._make_block(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 32 x 32
            self._make_block(64, 32, kernel_size=4, stride=2, padding=1)    # -> 32 x 64 x 64
        )
        
        # Layer finale per raggiungere la dimensione esatta e i canali corretti
        self.final_layers = nn.Sequential(
            nn.Upsample(scale_factor=3.36, mode='bilinear', align_corners=False), # Trucco per avvicinarsi a 215
            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Mappa i pixel nell'intervallo [-1, 1]
        )

    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, context_vector, noise):
        x = torch.cat([context_vector, noise], dim=1)
        x = self.project(x)
        x = x.view(-1, 512, 4, 4) # Rimodellamento in una feature map 3D
        x = self.decoder(x)
        x = self.final_layers(x)
        # Assicura la dimensione finale esatta con un crop o pad se necessario
        x = nn.functional.interpolate(x, size=(215, 215), mode='bilinear', align_corners=False)
        return x

class Attention(nn.Module):
    """Meccanismo di Attention per pesare gli hidden states dell'encoder."""
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.W = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.v = nn.Linear(decoder_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        # Semplice attention che calcola una media pesata degli output dell'encoder
        # In un modello più complesso, l'hidden state del decoder influenzerebbe i pesi
        scores = self.v(torch.tanh(self.W(encoder_outputs)))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context_vector

class TextToImageModel(nn.Module):
    """Modello completo che combina Encoder, Attention e Decoder."""
    def __init__(self, text_encoder, image_decoder, attention):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_decoder = image_decoder
        self.attention = attention

    def forward(self, input_ids, attention_mask, noise):
        encoder_outputs = self.text_encoder(input_ids, attention_mask)
        context_vector = self.attention(encoder_outputs)
        generated_image = self.image_decoder(context_vector, noise)
        return generated_image

# --- 4. Ciclo di Addestramento ---

def train(args):
    """Funzione principale per l'addestramento del modello."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    # Dataset e DataLoader
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizza a [-1, 1]
    ])
    
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
    
    dataset = PokemonDataset(
        csv_file=os.path.join(args.data_dir, 'pokemon.csv'),
        img_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Modello, Loss e Ottimizzatore
    text_encoder = TextEncoder(fine_tune=True).to(device)
    attention = Attention(args.embedding_dim, args.embedding_dim).to(device)
    image_decoder = ImageDecoder(args.embedding_dim, args.noise_dim).to(device)
    
    model = TextToImageModel(text_encoder, image_decoder, attention).to(device)
    
    criterion = nn.L1Loss() # Mean Absolute Error, come suggerito
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Ciclo di addestramento
    fixed_noise = torch.randn(args.batch_size, args.noise_dim, device=device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (input_ids, attention_mask, real_images) in enumerate(progress_bar):
            input_ids, attention_mask, real_images = \
                input_ids.to(device), attention_mask.to(device), real_images.to(device)

            # Genera rumore per questo batch
            noise = torch.randn(real_images.size(0), args.noise_dim, device=device)

            # Forward pass
            optimizer.zero_grad()
            fake_images = model(input_ids, attention_mask, noise)
            
            # Calcolo della loss
            loss = criterion(fake_images, real_images)
            
            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'L1 Loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average L1 Loss: {avg_loss:.4f}")

        # Salva immagini di esempio
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Usa lo stesso batch di testo per coerenza
                sample_input_ids, sample_attention_mask, _ = next(iter(dataloader))
                sample_input_ids = sample_input_ids.to(device)
                sample_attention_mask = sample_attention_mask.to(device)
                
                fake_samples = model(sample_input_ids, sample_attention_mask, fixed_noise[:sample_input_ids.size(0)])
                
                # Denormalizza e salva
                fake_samples = fake_samples * 0.5 + 0.5
                # Crea una trasformazione separata solo per il salvataggio
                save_transform = transforms.ToPILImage()
                save_transform(fake_samples[0].cpu()).save(
                    os.path.join(args.output_dir, 'images', f'epoch_{epoch+1}.png')
                )
            print(f"Saved sample image for epoch {epoch+1}.")

    # Salva il modello finale
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'pikikagen_model.pth'))
    print("Training complete. Model saved.")


# --- 5. Esecuzione ---

if __name__ == '__main__':
    args = get_args()
    train(args)
