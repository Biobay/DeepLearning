import sys
import os
import torch

# Aggiunge la cartella radice del progetto (DeepLearning) al path di Python.
# Questo permette di usare importazioni assolute a partire da 'src', 
# che è una pratica standard e viene compresa meglio dagli editor.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.data.dataset import PokemonDataset

def test_tokenizer_and_dataset():
    """
    Testa il caricamento del dataset e il processo di tokenizzazione.
    """
    print("Avvio del test del dataset...")
    
    # --- MODIFICA: Costruisci percorsi assoluti ---
    # Questo rende lo script eseguibile da qualsiasi posizione.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, 'data', 'pokemon.csv')
    images_dir = os.path.join(project_root, 'small_images')
    # --- FINE MODIFICA ---

    # Controlla se i file esistono
    if not os.path.exists(csv_path):
        print(f"Errore: File CSV non trovato in {csv_path}")
        return
    if not os.path.exists(images_dir):
        print(f"Errore: Directory immagini non trovata in {images_dir}")
        return

    print("Creazione dell'istanza di PokemonDataset...")
    try:
        dataset = PokemonDataset(csv_path=csv_path, images_dir=images_dir)
    except Exception as e:
        print(f"Errore durante la creazione del dataset: {e}")
        return

    print(f"\nDataset caricato con successo. Numero di campioni: {len(dataset)}")

    # Prendi il primo campione
    sample = dataset[1]
    
    print("\n--- Analisi del primo campione ---")
    
    # Recupera la descrizione originale
    original_description = dataset.df.iloc[0]['description']
    print(f"Descrizione Originale: {original_description}")

    # Analizza i tensori
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    
    print(f"\nForma di input_ids: {input_ids.shape}")
    print(f"Forma di attention_mask: {attention_mask.shape}")
    
    print(f"\nPrimi 30 token ID: {input_ids[:30].tolist()}")
    
    # Decodifica i token per vedere il risultato
    decoded_text = dataset.tokenizer.decode(input_ids)
    print(f"\nTesto decodificato dai token ID (con token speciali):")
    print(decoded_text)

    # Verifica il padding
    num_padding_tokens = (input_ids == dataset.tokenizer.pad_token_id).sum().item()
    print(f"\nNumero di token di padding aggiunti: {num_padding_tokens}")
    print(f"Lunghezza totale della sequenza: {len(input_ids)}")

    print("\nTest completato con successo!")

if __name__ == "__main__":
    test_tokenizer_and_dataset()
