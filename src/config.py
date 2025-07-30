# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_SIZE = 215
BATCH_SIZE = 16
NUM_WORKERS = 2

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True

# Decoder e Attenzione
NUM_HEADS = 4
DECODER_DIM = 256
CONTEXT_DIM = ENCODER_DIM
NGF = 64
OUTPUT_CHANNELS = 3

# =============================================================================
# ## PARAMETRI MODIFICATI PER IL NUOVO ESPERIMENTO ##
# =============================================================================

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100 # Puoi ridurlo a 30-50 per test più rapidi

# Learning Rate iniziale. Manteniamolo per ora, ma lo scheduler lo gestirà.
LEARNING_RATE = 1e-4 

# Weight decay per l'ottimizzatore Adam
WEIGHT_DECAY = 1e-5

# 1. Tasso di Dropout ridotto per non distruggere le feature.
#    Iniziamo con un valore basso per vedere se l'informazione passa.
DROPOUT_RATE = 0.2 

# 2. Peso per la loss LPIPS (lambda).
#    Iniziamo con un valore molto più basso per evitare che domini sulla L1.
#    L'obiettivo è bilanciare le due loss.
LAMBDA_LPIPS = 0.1

# =============================================================================

# --- Parametri per lo Scheduler (NUOVO) ---
# Questi parametri verranno usati per creare lo scheduler `ReduceLROnPlateau`
SCHEDULER_PATIENCE = 10 # Epoche di pazienza prima di ridurre il LR
SCHEDULER_FACTOR = 0.1  # Fattore di riduzione del LR (es. LR * 0.1)


# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
LOG_INTERVAL = 10
SAVE_IMAGE_EPOCHS = 5 # Salviamo le immagini meno spesso per non riempire il disco
CHECKPOINT_SAVE_EPOCHS = 5 # Idem per i checkpoint