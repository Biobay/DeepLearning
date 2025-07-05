import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Aggiungi la root del progetto al path di Python per risolvere i problemi di importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa i moduli del progetto
try:
    import src.config as config
except ImportError as e:
    print(f"Errore durante l'importazione dei moduli: {e}")
    print("Assicurati di eseguire lo script dalla cartella principale del progetto o che il path sia corretto.")
    sys.exit(1)

def analyze_dataset(cfg):
    """
    Analizza il dataset di immagini e stampa statistiche su dimensioni e distribuzione dei colori.
    """
    print("\n--- Inizio Analisi del Dataset ---")
    
    csv_path = os.path.join(cfg.DATA_DIR, cfg.CSV_NAME)
    img_dir = cfg.IMAGE_DIR
    results_dir = cfg.RESULTS_DIR

    # Assicurati che la cartella dei risultati esista
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"ERRORE: File CSV non trovato in {csv_path}")
        return
        
    if not os.path.exists(img_dir):
        print(f"ERRORE: Cartella delle immagini non trovata in {img_dir}")
        return

    df = pd.read_csv(csv_path)
    # Costruisci i percorsi completi e verifica l'esistenza
    image_paths = [os.path.join(img_dir, f"{row['Name']}.png") for index, row in df.iterrows()]
    image_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not image_paths:
        print("ERRORE: Nessuna immagine trovata nella cartella specificata.")
        return

    print(f"Trovate {len(image_paths)} immagini da analizzare.")

    widths, heights = [], []
    # Usiamo una lista per accumulare i pixel per l'analisi dei colori
    # per evitare di tenere tutte le immagini in memoria.
    pixel_samples = []

    for img_path in tqdm(image_paths, desc="Analisi Immagini"):
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
                
                w, h = img_rgb.size
                widths.append(w)
                heights.append(h)
                
                # Normalizza i pixel a [0, 1] e aggiungili ai campioni
                pixel_samples.append(np.array(img_rgb) / 255.0)
        except Exception as e:
            print(f"\nATTENZIONE: Impossibile processare l'immagine {img_path}. Errore: {e}")

    if not widths: # Se nessuna immagine è stata processata con successo
        print("ERRORE: Nessuna immagine è stata processata correttamente.")
        return

    # --- Statistiche sulle Dimensioni ---
    print("\n--- Statistiche Dimensioni Immagini ---")
    print(f"Larghezza | Min: {np.min(widths):>4} | Max: {np.max(widths):>4} | Media: {np.mean(widths):>7.2f}")
    print(f"Altezza   | Min: {np.min(heights):>4} | Max: {np.max(heights):>4} | Media: {np.mean(heights):>7.2f}")

    # --- Statistiche sui Colori ---
    print("\n--- Statistiche Colori (Canali R, G, B) ---")
    # Calcola media e std su tutti i pixel campionati
    all_pixels = np.concatenate([p.reshape(-1, 3) for p in pixel_samples], axis=0)
    color_mean = np.mean(all_pixels, axis=0)
    color_std = np.std(all_pixels, axis=0)
    print(f"Media (R, G, B):       {color_mean}")
    print(f"Deviazione Std (R, G, B): {color_std}")
    print("(Questi valori sono ideali da usare per la normalizzazione dei dati)")

    # --- Creazione Istogramma Dimensioni ---
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(widths, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribuzione Larghezze Immagini')
        plt.xlabel('Larghezza (pixel)')
        plt.ylabel('Frequenza')

        plt.subplot(1, 2, 2)
        plt.hist(heights, bins=30, color='salmon', edgecolor='black')
        plt.title('Distribuzione Altezze Immagini')
        plt.xlabel('Altezza (pixel)')
        plt.ylabel('Frequenza')

        histogram_path = os.path.join(results_dir, "dimensions_histogram.png")
        plt.tight_layout()
        plt.savefig(histogram_path)
        plt.close()
        print(f"\nIstogramma delle dimensioni salvato in: {histogram_path}")
    except Exception as e:
        print(f"\nATTENZIONE: Impossibile creare l'istogramma. Errore: {e}")

    print("\n--- Analisi del Dataset Completata ---")

if __name__ == '__main__':
    analyze_dataset(config)
