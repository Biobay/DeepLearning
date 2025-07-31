# src/config.py
import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0

# =============================================================================
# ## PARAMETRI COMPLETI DEL MODELLO ##
# =============================================================================

# --- Encoder ---
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True

# --- Attention ---
# CONTEXT_DIM è la dimensione dell'output dell'encoder, che l'attention usa.
CONTEXT_DIM = ENCODER_DIM
NUM_HEADS = 4

# --- Decoder (U-Net) ---
OUTPUT_CHANNELS = 3

# =============================================================================

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150

# --- Learning Rates Differenziati ---
LEARNING_RATE_ENCODER = 1e-4
LEARNING_RATE_DECODER = 2e-5
WEIGHT_DECAY = 1e-5

# --- Parametri per lo Scheduler ---
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_differential_lr_v2" # v2 per non confondere
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10