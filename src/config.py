import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_SIZE = 215
BATCH_SIZE = 16  # Ridurre se si esaurisce la memoria della GPU
NUM_WORKERS = 0  # Numero di processi per il caricamento dei dati

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
FINE_TUNE_ENCODER = True # Se fare il fine-tuning dell'encoder

# Dimensioni Fondamentali
# La dimensione dell'embedding del testo è determinata dall'encoder scelto.
TEXT_EMBEDDING_DIM = 256  # Dimensione di output di BERT-mini (corrisponde a ENCODER_MODEL_NAME)
Z_DIM = 100               # Dimensione del vettore di rumore latente

# Architettura
NUM_HEADS = 8             # Numero di teste per la Multi-Head Attention
DECODER_BASE_CHANNELS = 128 # Controlla la larghezza/potenza del generatore
DISCRIMINATOR_BASE_CHANNELS = 64 # Controlla la larghezza/potenza del discriminatore

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150 # Numero di epoche per l'addestramento
# Ridotto per evitare tempi di addestramento troppo lunghi
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # Per la regolarizzazione L2 sull'ottimizzatore
LAMBDA_L1 = 100 # Peso per la loss di ricostruzione L1 nel generatore

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
LOG_DIR = f"{RESULTS_DIR}/logs" # Directory per i file di log
LOG_INTERVAL = 10 # Ogni quanti batch stampare le informazioni sulla loss
SAVE_IMAGE_EPOCHS = 1 # Ogni quante epoche salvare un batch di immagini generate
CHECKPOINT_SAVE_EPOCHS = 1 # Ogni quante epoche salvare un checkpoint del modello

# --- Parametri Aggiuntivi ---
STAGE1_IMAGE_SIZE = 64  # Dimensione delle immagini per la Fase I (Stage-I GAN)
