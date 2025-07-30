# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"

IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256

BATCH_SIZE = 4   # Le GAN usano molta più memoria, riduciamo il batch size
NUM_WORKERS = 0

# --- Parametri del Modello ---
# Encoder (invariato)
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True

# Decoder U-Net
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
UNET_CHANNELS = (64, 128, 256, 512)
OUTPUT_CHANNELS = 3
DROPOUT_RATE = 0.5 # Possiamo riattivare il dropout, è utile nelle GAN

# --- Parametri di Addestramento (GAN) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150 # Le GAN richiedono più epoche per convergere

# Usiamo learning rate e beta1 standard per le GAN
# In src/config.py

LEARNING_RATE_GEN = 2e-4  # Lascia questo invariato
LEARNING_RATE_DISC = 5e-5   # Riduci di 4 volte (o anche 1e-5, 10 volte)

BETA1 = 0.5 # Parametro per l'ottimizzatore Adam

# Peso per la L1 Loss nella loss totale del Generatore
LAMBDA_L1 = 100

# Parametri per lo Scheduler
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_gan_v1" # Nuova cartella per l'esperimento GAN
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10 # Salviamo meno spesso, il training è più lungo
CHECKPOINT_SAVE_EPOCHS = 10