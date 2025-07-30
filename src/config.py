# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0

# --- Parametri del Modello ---
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
UNET_CHANNELS = (64, 128, 256, 512)
OUTPUT_CHANNELS = 3
DROPOUT_RATE = 0.5

# =============================================================================
# ## PARAMETRI DI TRAINING GAN RIVISTI PER LA STABILITÀ ##
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150

# Learning Rates
LEARNING_RATE_GEN = 2e-4  # Manteniamo il LR del Generatore
LEARNING_RATE_DISC = 2e-5   # RIDOTTO DRASTICAMENTE per rallentare il Discriminatore

# Parametro Adam standard per le GAN
BETA1 = 0.5

# Peso per la L1 Loss nella loss totale del Generatore
LAMBDA_L1 = 50

# NUOVO: Parametri per il Label Smoothing
# Usiamo label "morbide" per il Discriminatore per rendergli il compito più difficile
# Es. 0.9 invece di 1.0 per le immagini reali
REAL_LABEL_SMOOTHING = 0.9

# Parametri per lo Scheduler (invariati)
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_gan_v2_stabilized" # Nuova cartella per non confondere gli esperimenti
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10