import torch
import argparse
import os
import sys
from torchvision.utils import save_image
from transformers import BertTokenizer

# Aggiungi la directory src al path per importare i moduli custom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config import (
    ENCODER_MODEL_NAME,
    DEVICE,
    CONTEXT_DIM,
    DECODER_DIM,
    ATTENTION_DIM,
    NGF,
    OUTPUT_CHANNELS,
    GENERATED_IMAGE_DIR
)
from models.model import PikaPikaGen

def find_latest_checkpoint(checkpoint_dir):
    """Trova il checkpoint più recente in una directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def generate(args):
    """
    Funzione principale per generare un'immagine da un testo.
    """
    # --- Controllo e preparazione ---
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        print("Nessun checkpoint specificato, cerco il più recente in 'results/checkpoints'...")
        checkpoint_path = find_latest_checkpoint("results/checkpoints")

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Errore: Nessun checkpoint trovato. Assicurati di aver addestrato il modello e che i checkpoint siano in 'results/checkpoints'.")
        return

    print(f"Utilizzo del checkpoint: {checkpoint_path}")

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata directory di output: {output_dir}")

    # --- Caricamento del modello ---
    print("Caricamento del modello...")
    model = PikaPikaGen(
        encoder_model_name=ENCODER_MODEL_NAME,
        context_dim=CONTEXT_DIM,
        decoder_dim=DECODER_DIM,
        attention_dim=ATTENTION_DIM,
        ngf=NGF,
        output_channels=OUTPUT_CHANNELS,
        fine_tune_encoder=False
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Modello caricato con successo.")

    # --- Preparazione del testo ---
    print("Preparazione del testo di input...")
    tokenizer = BertTokenizer.from_pretrained(ENCODER_MODEL_NAME)
    
    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    # --- Generazione dell'immagine ---
    print("Generazione dell'immagine in corso...")
    with torch.no_grad():
        generated_image, _ = model(input_ids, attention_mask)

    # --- Salvataggio dell'immagine ---
    save_image(
        generated_image, 
        args.output_path,
        normalize=True,
        nrow=1
    )
    print(f"Immagine generata e salvata in '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un'immagine di un Pokémon da una descrizione testuale.")
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="La descrizione testuale del Pokémon da generare."
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Percorso del checkpoint del modello (.pth). Se non specificato, usa il più recente."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(GENERATED_IMAGE_DIR, "generated_pokemon.png"),
        help="Percorso dove salvare l'immagine generata."
    )

    args = parser.parse_args()
    generate(args)
