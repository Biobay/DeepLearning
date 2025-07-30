# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_SIZE = 215 # Manteniamo la risoluzione originale
BATCH_SIZE = 8   # Ridotto per accomodare la U-Net che usa più memoria
NUM_WORKERS = 0  # IMPOSTATO A 0 PER EVITARE ERRORI DI SHARED MEMORY

IMAGE_OUTPUT_SIZE = 215

# Dimensione interna usata dalla U-Net per un funzionamento stabile
# Deve essere una potenza di 2 (es. 128, 256)
MODEL_INTERNAL_SIZE = 256

# --- Parametri del Modello ---
# Encoder (invariato)
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True

# Decoder (NUOVA ARCHITETTURA U-NET)
NUM_HEADS = 4       # Per l'Attention
CONTEXT_DIM = ENCODER_DIM # Il contesto è sempre l'output dell'encoder

# Canali per ogni livello della U-Net. Es: 64 -> 128 -> 256 -> 512 (bottleneck) -> 256 -> ...
UNET_CHANNELS = (64, 128, 256, 512)
OUTPUT_CHANNELS = 3

# --- Parametri di Addestramento (PIANO SEMPLIFICATO) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50 # Iniziamo con 50 epoche per vedere se impara

LEARNING_RATE = 2e-4 # Un buon punto di partenza per Adam
WEIGHT_DECAY = 1e-5

# Disabilitiamo temporaneamente le tecniche avanzate per debuggare l'architettura
DROPOUT_RATE = 0.0
LAMBDA_LPIPS = 0.0 # Usiamo solo L1 Loss per ora

# Parametri per lo Scheduler
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5 # Riduciamo il LR in modo meno aggressivo

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_unet_v1" # Nuova cartella per non sovrascrivere i vecchi esperimenti
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 5
CHECKPOINT_SAVE_EPOCHS = 5