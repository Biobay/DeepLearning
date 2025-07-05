import torch
from torchvision.utils import save_image
from transformers import BertTokenizer, BertModel # Modificato per usare BERT
import os
import argparse
import sys

# Aggiungi la root del progetto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.models.decoder import GeneratorS1
from src.models.generator_s2 import GeneratorS2

def generate_image(text_prompt, output_filename):
    """
    Genera un'immagine ad alta risoluzione (256x256) da un prompt testuale
    utilizzando la pipeline a due stadi (Stage-I e Stage-II).
    """
    # --- Setup ---
    device = config.DEVICE
    os.makedirs(config.GENERATED_IMAGE_DIR, exist_ok=True)

    # --- Caricamento Modelli ---
    # Tokenizer e Text Encoder
    tokenizer = BertTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    text_encoder = BertModel.from_pretrained(config.ENCODER_MODEL_NAME).to(device)
    text_encoder.eval()

    # Generatore Fase I
    gen_s1 = GeneratorS1(config).to(device)
    checkpoint_s1_path = os.path.join(config.CHECKPOINT_DIR_S1, 'generator.pth')
    if not os.path.exists(checkpoint_s1_path):
        raise FileNotFoundError(f"Checkpoint S1 non trovato in {checkpoint_s1_path}")
    gen_s1.load_state_dict(torch.load(checkpoint_s1_path, map_location=device))
    gen_s1.eval()

    # Generatore Fase II
    gen_s2 = GeneratorS2(config).to(device)
    # Cerca l'ultimo checkpoint per il generatore S2
    try:
        checkpoints = [f for f in os.listdir(config.CHECKPOINT_DIR_S2) if f.startswith('generator_s2_epoch_')]
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint_s2 = checkpoints[-1]
        checkpoint_s2_path = os.path.join(config.CHECKPOINT_DIR_S2, latest_checkpoint_s2)
        print(f"Caricamento checkpoint S2 da: {checkpoint_s2_path}")
        gen_s2.load_state_dict(torch.load(checkpoint_s2_path, map_location=device))
    except (FileNotFoundError, IndexError):
         raise FileNotFoundError(f"Nessun checkpoint S2 trovato in {config.CHECKPOINT_DIR_S2}. Addestra prima la Fase II.")
    gen_s2.eval()

    # --- Generazione ---
    with torch.no_grad():
        # 1. Prepara il testo
        inputs = tokenizer(text_prompt, return_tensors='pt', padding=True, truncation=True, max_length=config.MAX_SEQ_LEN).to(device)
        text_embedding = text_encoder(**inputs).last_hidden_state[:, 0, :] # [CLS] token

        # 2. Genera immagine a bassa risoluzione (Fase I)
        noise = torch.randn(1, config.Z_DIM, device=device)
        low_res_image = gen_s1(noise, text_embedding)

        # 3. Genera immagine ad alta risoluzione (Fase II)
        high_res_image = gen_s2(low_res_image, text_embedding)

        # 4. Salva l'immagine
        output_path = os.path.join(config.GENERATED_IMAGE_DIR, output_filename)
        save_image(high_res_image, output_path, normalize=True)
        print(f"Immagine salvata in: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un'immagine da un prompt testuale usando StackGAN.')
    parser.add_argument('--text', type=str, required=True, help='Il prompt testuale per la generazione dell'immagine.')
    parser.add_argument('--output', type=str, default='generated_image.png', help='Il nome del file per l'immagine di output.')
    args = parser.parse_args()

    generate_image(args.text, args.output)
