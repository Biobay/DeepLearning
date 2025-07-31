# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"

IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256

BATCH_SIZE = 4   # Le GAN usano più memoria, manteniamo un batch size basso
NUM_WORKERS = 0

# --- Parametri del Modello ---
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
OUTPUT_CHANNELS = 3

# Parametri per l'input strutturato del decoder
NUM_NOISE_CHANNELS = 4
DECODER_IN_CHANNELS = ENCODER_DIM + NUM_NOISE_CHANNELS

# =============================================================================
# ## PARAMETRI FINALI PER IL TRAINING DELLA cGAN ##
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200 # Le GAN richiedono più epoche per convergere e affinare i dettagli

# Learning Rates e Beta1 per Adam (standard per Pix2Pix/GANs)
LEARNING_RATE = 2e-4
BETA1 = 0.5

# Peso per la L1 Loss nella loss totale del Generatore. Valore standard Pix2Pix.
LAMBDA_L1 = 100

# Parametro per il Label Smoothing (aiuta a stabilizzare il Discriminatore)
REAL_LABEL_SMOOTHING = 0.9

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_final_cgan" # Cartella per l'esperimento finale
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10 # Salviamo meno spesso per non riempire il disco
CHECKPOINT_SAVE_EPOCHS = 10