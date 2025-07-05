import os
import sys
import torch

# Aggiungi la root del progetto al path di Python per risolvere i problemi di importazione
# Questo assicura che possiamo importare moduli da 'src'
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

    Args:
        model (torch.nn.Module): Il modello da analizzare.
        model_name (str): Il nome del modello da usare nell'output.
    """
    print(f"\n--- Analisi dei Parametri per: {model_name} ---")
    total_params = 0
    trainable_params = 0
    
    # Intestazioni della tabella
    print(f"{'Layer':<70} {'Shape':<25} {'Parametri':<15}")
    print("="*110)
    
    for name, parameter in model.named_parameters():
        # Salta i parametri non addestrabili (es. buffer)
        if not parameter.requires_grad:
            continue
        
        params = parameter.numel()
        shape = str(list(parameter.shape))
        
        # Stampa la riga della tabella formattata
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
    print("Creazione del modello Generatore per l'analisi...")
    
    # Analisi del Generatore (il tuo modello principale)
    try:
        # Nota: non è necessario spostare il modello su GPU (.to(device)) per questa analisi
        generator = PikaPikaGen(config)
        analyze_model_parameters(generator, "Generatore (PikaPikaGen / Stage-I)")
    except Exception as e:
        print(f"Errore durante la creazione o analisi del Generatore: {e}")
