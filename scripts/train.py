
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer
import argparse
import os

# Assicurati che i percorsi dei moduli siano corretti
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PokemonDataset, create_dataloaders
from src.models.encoder import TextEncoder
from src.models.decoder import ImageDecoder
from src.models.pikapikaGen import PikaPikaGen
from src.training.trainer import Trainer

def main(args):
    """
    Funzione principale per l'addestramento del modello PikaPikaGen.
    """
    # 1. Impostazioni preliminari
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Utilizzo del dispositivo: {device}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 3. Creazione dei DataLoader
    print("Creazione dei DataLoader...")
    train_loader, val_loader = create_dataloaders(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers
    )
    print("DataLoader creati con successo.")

    # 4. Inizializzazione del Modello
    print("Inizializzazione del modello...")
    text_encoder = TextEncoder(model_name=args.bert_model, fine_tune=args.fine_tune_encoder)
    image_decoder = ImageDecoder(latent_dim=text_encoder.bert.config.hidden_size, output_size=args.image_size)
    
    model = PikaPikaGen(text_encoder, image_decoder).to(device)
    print("Modello inizializzato.")

    # 5. Ottimizzatore e Funzione di Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.L1Loss() # Mean Absolute Error (L1 Loss)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir
    )

    # 7. Avvio dell'addestramento
    print("Avvio dell'addestramento...")
    trainer.train()
    print("Addestramento completato.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Addestra il modello PikaPikaGen per generare sprite di Pokémon.')
    
    # Argomenti relativi ai dati
    parser.add_argument('--csv_file', type=str, default='data/pokemon.csv', help='Percorso del file CSV con le descrizioni.')
    parser.add_argument('--img_dir', type=str, default='small_images/', help='Directory con le immagini dei Pokémon.')
    parser.add_argument('--image_size', type=int, default=215, help='Dimensione desiderata per le immagini (altezza e larghezza).')

    # Argomenti relativi al modello
    parser.add_argument('--bert_model', type=str, default='prajjwal1/bert-mini', help='Nome del modello BERT da usare come encoder.')
    parser.add_argument('--fine_tune_encoder', action='store_true', help='Se impostato, il text encoder verrà fine-tuned durante l'addestramento.')

    # Argomenti relativi all'addestramento
    parser.add_argument('--epochs', type=int, default=50, help='Numero di epoche di addestramento.')
    parser.add_argument('--batch_size', type=int, default=16, help='Dimensione del batch.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate per l'ottimizzatore.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Frazione del dataset da usare per la validazione.')
    parser.add_argument('--num_workers', type=int, default=0, help='Numero di worker per il DataLoader.')
    parser.add_argument('--use_cuda', action='store_true', help='Se impostato, usa la GPU per l'addestramento (se disponibile).')

    # Argomenti relativi ai checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints', help='Directory dove salvare i checkpoint del modello.')

    args = parser.parse_args()
    main(args)
