# src/config.py
import torch

# Dataset & Dataloader
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_OUTPUT_SIZE = 215
MODEL_INTERNAL_SIZE = 256
BATCH_SIZE = 4
NUM_WORKERS = 0

# Modello
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256
FINE_TUNE_ENCODER = True
NUM_HEADS = 4
CONTEXT_DIM = ENCODER_DIM
OUTPUT_CHANNELS = 3

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 200
LEARNING_RATE = 2e-4
BETA1 = 0.5
LAMBDA_L1 = 100

# Logging & Checkpoints
RESULTS_DIR = "results_final_cgan"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10