# src/config.py
import torch

# Dataset & Dataloader
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0

# Modello
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
CONTEXT_DIM = ENCODER_DIM # Dimensione dell'output di BERT
OUTPUT_CHANNELS = 3

# Parametri per la U-Net con Cross-Attention
NUM_HEADS = 4
UNET_CHANNELS = (64, 128, 256, 512) # Canali della U-Net

# Training (SOLO L1 LOSS PER DEBUG)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5

# Logging & Checkpoints
RESULTS_DIR = "results_cross_attention_v1"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10