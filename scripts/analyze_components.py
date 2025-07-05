import os
import sys
import torch

# Aggiungi la root del progetto al path di Python per risolvere i problemi di importazione
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa i moduli del progetto
try:
    import src.config as config
    from src.models.model import PikaPikaGen
except ImportError as e:
    print(f"Errore durante l'importazione dei moduli: {e}")
    print("Assicurati di eseguire lo script dalla cartella principale del progetto o che il path sia corretto.")
    sys.exit(1)

def analyze_model_parameters(model, model_name="Modello"):
    """
    Analizza e stampa i parametri di un modello PyTorch in una tabella formattata.
    """
    print(f"\n--- Analisi dei Parametri per: {model_name} ---")
    total_params = 0
    trainable_params = 0
    
    print(f"{'Layer':<70} {'Shape':<25} {'Parametri':<15}")
    print("="*110)
    
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        
        params = parameter.numel()
        shape = str(list(parameter.shape))
        
        print(f"{name:<70} {shape:<25} {params:<15,}")
        
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
            
    print("="*110)
    print(f"Parametri totali: {total_params:,}")
    print(f"Parametri addestrabili: {trainable_params:,}")
    print(f"--- Fine Analisi per: {model_name} ---\n")
    return total_params

if __name__ == '__main__':
    print("Creazione del modello PikaPikaGen per l'analisi dei componenti...")
    
    try:
        # Creiamo il modello completo
        full_model = PikaPikaGen(config)
        
        # 1. Analisi dell'Encoder Testuale
        # Accediamo al sottomodello 'encoder' e lo analizziamo
        if hasattr(full_model, 'encoder'):
            analyze_model_parameters(full_model.encoder, "Text Encoder")
        else:
            print("ATTENZIONE: Il modello principale non ha un attributo 'encoder'.")

        # 2. Analisi del Decoder di Immagini (Generatore)
        # Accediamo al sottomodello 'decoder' e lo analizziamo
        if hasattr(full_model, 'decoder'):
            analyze_model_parameters(full_model.decoder, "Image Decoder (GeneratorS1)")
        else:
            print("ATTENZIONE: Il modello principale non ha un attributo 'decoder'.")

    except Exception as e:
        print(f"Si è verificato un errore durante la creazione o l'analisi dei modelli: {e}")
