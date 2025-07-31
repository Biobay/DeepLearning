# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"

IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256 # La U-Net lavora a questa dimensione

BATCH_SIZE = 8 # Un batch size leggermente più grande è ok senza la GAN
NUM_WORKERS = 0

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True

# Decoder U-Net
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
UNET_CHANNELS = (64, 128, 256, 512)

# --- NUOVI PARAMETRI PER L'INPUT DEL DECODER ---
# Canali del rumore da concatenare alla mappa di contesto
NUM_NOISE_CHANNELS = 4 
# Il primo layer del decoder prenderà in input il vettore di testo + il rumore
DECODER_IN_CHANNELS = ENCODER_DIM + NUM_NOISE_CHANNELS

OUTPUT_CHANNELS = 3

# --- Parametri di Addestramento (SEMPLIFICATI PER TESTARE L'ARCHITETTURA) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50 # 50 epoche sono sufficienti per vedere se le forme emergono

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

# Disabilitiamo la GAN per ora. Useremo solo L1 Loss.
# Questi parametri non verranno usati, ma li lasciamo per riferimento.
# LEARNING_RATE_DISC = 2e-5
# BETA1 = 0.5
# LAMBDA_L1 = 100
# REAL_LABEL_SMOOTHING = 0.9

# Parametri per lo Scheduler
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_structured_input_v1" # Nuova cartella per l'esperimento
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 5
CHECKPOINT_SAVE_EPOCHS = 5