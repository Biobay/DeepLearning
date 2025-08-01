# src/config.py
import torch, os

# Percorsi Assoluti
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "small_images")
CSV_PATH = os.path.join(DATA_DIR, "pokemon.csv")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

# Dataset
BATCH_SIZE = 16 # Possiamo tornare a 16 per Stage-I
NUM_WORKERS = 0
MAX_SEQ_LEN = 128

# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
FINE_TUNE_ENCODER = True
TEXT_EMBEDDING_DIM = 256
NUM_HEADS = 4

# Stage-I
Z_DIM = 100
DECODER_BASE_CHANNELS = 64
DISCRIMINATOR_BASE_CHANNELS = 64
STAGE1_IMAGE_SIZE = 64

# Stage-II
STAGE2_IMAGE_SIZE = 215

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stage-I Training (Parametri per la stabilità)
EPOCHS = 150 # Diamo più tempo per convergere
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 1e-4 # Leggermente più basso per il Discriminatore
LAMBDA_L1 = 50       # RIDOTTO per dare più peso alla loss avversaria

# Stage-II Training
EPOCHS_S2 = 150
LEARNING_RATE_S2 = 2e-4
LAMBDA_L1_S2 = 20 # Ancora più basso per Stage-II

# Logging e Checkpoints
RESULTS_DIR = os.path.join(BASE_DIR, "results_multiscale_gan")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
GENERATED_IMAGE_DIR = os.path.join(RESULTS_DIR, "generated_images")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
SAVE_IMAGE_EPOCHS = 10
CHECKPOINT_SAVE_EPOCHS = 10