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
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
OUTPUT_CHANNELS = 3

# --- PARAMETRI PER IL GENERATORE STYLEGAN-INSPIRED ---
LATENT_DIM = ENCODER_DIM  # Il testo codificato è il nostro vettore latente
STYLE_DIM = 512          # Dimensione dello spazio di stile intermedio 'w'
MAPPING_NETWORK_DEPTH = 4 # Profondità del MLP per la Mapping Network

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200 # Questo modello potrebbe richiedere più tempo per convergere
LEARNING_RATE = 1e-4 # Un LR più basso è più stabile
WEIGHT_DECAY = 1e-5
LAMBDA_L1 = 10 # Peso per la L1 Loss

# Parametri per lo Scheduler
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_stylegan_v1"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10