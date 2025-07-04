import torch

# --- Parametri del Dataset e dei Dataloader ---
DATA_DIR = "data"
IMAGE_DIR = "small_images"
CSV_NAME = "pokemon.csv"
SPLITS_DIR = "data/splits"
IMAGE_SIZE = 215
BATCH_SIZE = 16  # Ridurre se si esaurisce la memoria della GPU
NUM_WORKERS = 2  # Numero di processi per il caricamento dei dati

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
ENCODER_DIM = 256  # Dimensione di output di BERT-mini
FINE_TUNE_ENCODER = True # Se fare il fine-tuning dell'encoder

# Decoder e Attenzione
DECODER_DIM = 256  # Dimensione dello stato nascosto dell'LSTM nel decoder
ATTENTION_HEADS = 8 # Numero di teste nel meccanismo di Multi-Head Attention
CONTEXT_DIM = ENCODER_DIM # Il contesto per il generatore di immagini è l'output dell'encoder
NGF = 64 # Numero di feature nel generatore
OUTPUT_CHANNELS = 3 # Canali RGB

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5 # Numero di epoche per l'addestramento
# Ridotto per evitare tempi di addestramento troppo lunghi
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # Per la regolarizzazione L2 sull'ottimizzatore

# --- Parametri per il Logging e i Checkpoint ---
RESULTS_DIR = "results"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
GENERATED_IMAGE_DIR = f"{RESULTS_DIR}/generated_images"
LOG_INTERVAL = 10 # Ogni quanti batch stampare le informazioni sulla loss
SAVE_IMAGE_EPOCHS = 1 # Ogni quante epoche salvare un batch di immagini generate
CHECKPOINT_SAVE_EPOCHS = 1 # Ogni quante epoche salvare un checkpoint del modello
