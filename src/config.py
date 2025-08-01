# src/config.py

import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
CSV_PATH = f"{DATA_DIR}/{CSV_NAME}"  # Percorso completo al file CSV
SPLITS_DIR = "data/splits"

IMAGE_OUTPUT_SIZE = 215
IMAGE_SIZE = IMAGE_OUTPUT_SIZE  # Alias per compatibilità con i notebook
STAGE1_IMAGE_SIZE = 64  # Dimensione delle immagini per lo Stage-I del StackGAN
MODEL_INTERNAL_SIZE = 256

BATCH_SIZE = 4   # Manteniamo un batch size basso
NUM_WORKERS = 0

# --- Parametri del Modello ---
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
TEXT_EMBEDDING_DIM = 256  # Aggiunto per compatibilità
FINE_TUNE_ENCODER = True

# Decoder U-Net con Cross-Attention
NUM_HEADS = 8 # La Cross-Attention beneficia di più teste
CONTEXT_DIM = ENCODER_DIM
UNET_CHANNELS = (64, 128, 256, 512)
OUTPUT_CHANNELS = 3
DROPOUT_RATE = 0.5 # Dropout nei blocchi UpBlock

# Parametri GAN
Z_DIM = 100  # Dimensione del vettore noise

# --- Parametri di Addestramento (GAN) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200

LEARNING_RATE = 2e-4
BETA1 = 0.5
LAMBDA_L1 = 100
REAL_LABEL_SMOOTHING = 0.9

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results_cross_attention_gan"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10