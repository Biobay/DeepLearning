# src/config.py
import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256
BATCH_SIZE = 8 # Aumentiamo un po', il modello StyleGAN è efficiente
NUM_WORKERS = 0

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
OUTPUT_CHANNELS = 3

# Parametri per il Generatore StyleGAN-inspired
LATENT_DIM = ENCODER_DIM
STYLE_DIM = 512
MAPPING_NETWORK_DEPTH = 4

# --- Parametri di Addestramento (Test con L1 Loss) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100 # Diamo al modello un po' di tempo per imparare

LEARNING_RATE = 2e-4 # Un LR standard per Adam
WEIGHT_DECAY = 1e-5

# Parametri per lo Scheduler
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_stylegan_cls_token" # Nuova cartella per l'esperimento
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10