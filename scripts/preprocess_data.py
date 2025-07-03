import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(
    csv_path, 
    output_dir, 
    train_size=0.6, 
    val_size=0.2, 
    test_size=0.2, 
    random_state=42
):
    """
    Carica un file CSV, lo mescola e lo suddivide in set di training, 
    validazione e test, salvandoli in file separati.

    Args:
        csv_path (str): Percorso del file CSV di input.
        output_dir (str): Directory dove salvare i file CSV splittati.
        train_size (float): Proporzione del dataset per il training.
        val_size (float): Proporzione del dataset per la validazione.
        test_size (float): Proporzione del dataset per il test.
        random_state (int): Seed per la riproducibilità dello split.
    """
    # --- 1. Validazione degli input ---
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Le proporzioni di train, val e test devono sommare a 1.")
    
    print(f"Caricamento dati da: {csv_path}")
    
    # --- 2. Caricamento Dati ---
    # Usiamo le stesse impostazioni del PokemonDataset per coerenza
    try:
        df = pd.read_csv(csv_path, delimiter='\t', encoding='utf-16 LE')
    except Exception as e:
        print(f"Errore durante la lettura del file CSV: {e}")
        return

    print(f"Dataset caricato con successo. Numero totale di righe: {len(df)}")

    # --- 3. Suddivisione dei Dati ---
    # Prima dividiamo in training e "resto" (validazione + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )

    # Calcoliamo la proporzione del test set rispetto al "resto"
    # Esempio: se val=0.2 e test=0.2, il "resto" è 0.4. Il test è 0.2/0.4 = 50% del resto.
    relative_test_size = test_size / (val_size + test_size)

    # Dividiamo il "resto" in validazione e test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        shuffle=True
    )

    print("\nSplit completato:")
    print(f"  - Campioni di Training:   {len(train_df)} ({len(train_df)/len(df):.0%})")
    print(f"  - Campioni di Validazione: {len(val_df)} ({len(val_df)/len(df):.0%})")
    print(f"  - Campioni di Test:        {len(test_df)} ({len(test_df)/len(df):.0%})")

    # --- 4. Salvataggio dei File ---
    # Creiamo la directory di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    # Salviamo i file con la stessa codifica e separatore, senza l'indice di pandas
    train_df.to_csv(train_path, sep='\t', encoding='utf-16 LE', index=False)
    val_df.to_csv(val_path, sep='\t', encoding='utf-16 LE', index=False)
    test_df.to_csv(test_path, sep='\t', encoding='utf-16 LE', index=False)

    print(f"\nFile salvati in '{output_dir}'")


if __name__ == '__main__':
    # Definiamo i percorsi relativi alla radice del progetto
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    INPUT_CSV = os.path.join(PROJECT_ROOT, 'data', 'pokemon.csv')
    OUTPUT_SPLITS_DIR = os.path.join(PROJECT_ROOT, 'data', 'splits')

    # Eseguiamo la funzione con le proporzioni richieste (60-20-20)
    preprocess_and_split_data(
        csv_path=INPUT_CSV,
        output_dir=OUTPUT_SPLITS_DIR,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2
    )
