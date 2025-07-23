import torch
import os

# --- Percorso di Base Assoluto ---
# Calcola il percorso assoluto della directory principale del progetto (es. "DeepLearning")
# __file__ è il percorso di questo file (config.py)
# os.path.dirname(__file__) è la directory 'src'
# os.path.join(..., '..') sale di un livello, arrivando alla root del progetto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# --- Parametri del Dataset e dei Dataloader (con percorsi assoluti) ---
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "small_images") 
CSV_NAME = "pokemon.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
IMAGE_SIZE = 215
BATCH_SIZE = 16  # Ridurre se si esaurisce la memoria della GPU
NUM_WORKERS = 0  # Numero di processi per il caricamento dei dati
MAX_SEQ_LEN = 128 # Lunghezza massima delle sequenze di testo
MAX_TEXT_LENGTH = 128 # Lunghezza massima per la tokenizzazione (stesso valore di MAX_SEQ_LEN)

# --- Parametri del Modello ---
# Encoder
ENCODER_MODEL_NAME = "prajjwal1/bert-mini" 
FINE_TUNE_ENCODER = False # L'encoder non viene addestrato in questo setup

# Dimensioni Fondamentali
# La dimensione dell'embedding del testo è determinata dall'encoder scelto.
TEXT_EMBEDDING_DIM = 256  # Dimensione di output di BERT-mini
Z_DIM = 100               # Dimensione del vettore di rumore latente

# Architettura
NUM_HEADS = 8             # Numero di teste per la Multi-Head Attention
DECODER_BASE_CHANNELS = 128 # Controlla la larghezza/potenza del generatore
DISCRIMINATOR_BASE_CHANNELS = 64 # Controlla la larghezza/potenza del discriminatore

# --- Parametri di Addestramento ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100 # Numero di epoche per l'addestramento Stage-I
# Ridotto per evitare tempi di addestramento troppo lunghi
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5 # Per la regolarizzazione L2 sull'ottimizzatore
LAMBDA_L1 = 100 # Peso per la loss di ricostruzione L1 nel generatore

# --- Parametri per il Logging e i Checkpoint (con percorsi assoluti) ---
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
GENERATED_IMAGE_DIR = os.path.join(RESULTS_DIR, "generated_images")
LOG_DIR = os.path.join(RESULTS_DIR, "logs") # Directory per i file di log
LOG_INTERVAL = 10 # Ogni quanti batch stampare le informazioni sulla loss
SAVE_IMAGE_EPOCHS = 1 # Ogni quante epoche salvare un batch di immagini generate
CHECKPOINT_SAVE_EPOCHS = 1 # Ogni quante epoche salvare un checkpoint del modello

# --- Parametri Aggiuntivi ---
STAGE1_IMAGE_SIZE = 64  # Dimensione delle immagini per la Fase I (Stage-I GAN)

# --- Parametri per la Fase II (StackGAN Stage-II) (con percorsi assoluti) ---
STAGE2_IMAGE_SIZE = 215 # Dimensione delle immagini ad alta risoluzione (come da requisiti del progetto)
EPOCHS_S2 = 100 # Numero di epoche per la Fase II
LEARNING_RATE_S2 = 1e-4 # Learning rate per gli ottimizzatori della Fase II
LAMBDA_L1_S2 = 100 # Peso per la loss L1 dello Stage-II
CHECKPOINT_DIR_S1 = CHECKPOINT_DIR # Ora punta già alla cartella corretta e assoluta
CHECKPOINT_DIR_S2 = os.path.join(RESULTS_DIR, "checkpoints_s2")
GENERATED_IMAGE_DIR_S2 = os.path.join(RESULTS_DIR, "generated_images_s2")
