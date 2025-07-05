import torch
import argparse
import os
import sys
from torchvision.utils import save_image
from torchvision import transforms
from transformers import DistilBertTokenizer, DistilBertModel

# Aggiungi la directory src al path per importare i moduli custom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    DEVICE,
    Z_DIM,
    TEXT_EMBEDDING_DIM,
    CHECKPOINT_DIR_S1,
    CHECKPOINT_DIR_S2,
    GENERATED_IMAGE_DIR,
    DECODER_BASE_CHANNELS
)
from src.models.decoder import GeneratorS1
from src.models.generator_s2 import GeneratorS2

def find_latest_checkpoint(checkpoint_dir):
    """Trova il checkpoint del generatore più recente in una directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    # Filtra per i file del generatore
    checkpoints = [
        os.path.join(checkpoint_dir, f) 
        for f in os.listdir(checkpoint_dir) 
        if f.startswith("generator") and f.endswith(".pth")
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def generate(args):
    """
    Funzione principale per generare un'immagine da un testo utilizzando la pipeline StackGAN a 2 stadi.
    """
    # --- Controllo e preparazione delle directory ---
    os.makedirs(GENERATED_IMAGE_DIR, exist_ok=True)
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata directory di output: {output_dir}")

    # --- Trova i checkpoint ---
    checkpoint_s1_path = args.checkpoint_s1 or find_latest_checkpoint(CHECKPOINT_DIR_S1)
    checkpoint_s2_path = args.checkpoint_s2 or find_latest_checkpoint(CHECKPOINT_DIR_S2)

    if not checkpoint_s1_path or not os.path.exists(checkpoint_s1_path):
        print(f"Errore: Checkpoint per la Fase I non trovato in '{CHECKPOINT_DIR_S1}'.")
        print("Assicurati di aver addestrato il modello della Fase I.")
        return
    if not checkpoint_s2_path or not os.path.exists(checkpoint_s2_path):
        print(f"Errore: Checkpoint per la Fase II non trovato in '{CHECKPOINT_DIR_S2}'.")
        print("Assicurati di aver addestrato il modello della Fase II.")
        return

    print(f"Utilizzo checkpoint Fase I: {checkpoint_s1_path}")
    print(f"Utilizzo checkpoint Fase II: {checkpoint_s2_path}")

    # --- Caricamento dei modelli ---
    print("Caricamento dei modelli...")
    # Generatore Fase I
    gen_s1 = GeneratorS1(z_dim=Z_DIM, text_embedding_dim=TEXT_EMBEDDING_DIM, g_base_channels=DECODER_BASE_CHANNELS).to(DEVICE)
    gen_s1.load_state_dict(torch.load(checkpoint_s1_path, map_location=DEVICE))
    gen_s1.eval()

    # Generatore Fase II
    gen_s2 = GeneratorS2(text_embedding_dim=TEXT_EMBEDDING_DIM, g_base_channels=DECODER_BASE_CHANNELS).to(DEVICE)
    gen_s2.load_state_dict(torch.load(checkpoint_s2_path, map_location=DEVICE))
    gen_s2.eval()
    
    # Encoder di testo
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE)
    text_encoder.eval()
    print("Modelli caricati con successo.")

    # --- Preparazione del testo ---
    print("Preparazione del testo di input...")
    with torch.no_grad():
        inputs = tokenizer(
            args.text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        ).to(DEVICE)
        text_embedding = text_encoder(**inputs).last_hidden_state[:, 0, :] # Estratto [CLS] token

    # --- Generazione dell'immagine ---
    print("Generazione dell'immagine in corso...")
    with torch.no_grad():
        # Vettore di rumore
        noise = torch.randn(1, Z_DIM, device=DEVICE)
        
        # Fase I: Genera immagine a bassa risoluzione
        low_res_image = gen_s1(noise, text_embedding)
        
        # Fase II: Genera immagine ad alta risoluzione
        high_res_image = gen_s2(low_res_image, text_embedding)

    # --- Ridimensionamento e Salvataggio ---
    print(f"Ridimensionamento dell'immagine a {args.size}x{args.size}...")
    resize_transform = transforms.Resize((args.size, args.size))
    final_image = resize_transform(high_res_image)

    save_image(
        final_image, 
        args.output_path,
        normalize=True,
        nrow=1
    )
    print(f"Immagine generata e salvata in '{args.output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera un'immagine di un Pokémon da una descrizione testuale usando StackGAN.")
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="La descrizione testuale del Pokémon da generare."
    )
    
    parser.add_argument(
        "--checkpoint_s1",
        type=str,
        default=None,
        help=f"Percorso del checkpoint del generatore S1. Se non specificato, usa il più recente da '{CHECKPOINT_DIR_S1}'."
    )

    parser.add_argument(
        "--checkpoint_s2",
        type=str,
        default=None,
        help=f"Percorso del checkpoint del generatore S2. Se non specificato, usa il più recente da '{CHECKPOINT_DIR_S2}'."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(GENERATED_IMAGE_DIR, "generated_pokemon.png"),
        help="Percorso dove salvare l'immagine generata."
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=215,
        help="Dimensione finale dell'immagine (lato)."
    )

    args = parser.parse_args()
    generate(args)
