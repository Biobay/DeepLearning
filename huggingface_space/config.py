# Configurazione StackGAN per Hugging Face
import torch

class Config:
    # Dimensioni
    Z_DIM = 100
    TEXT_DIM = 768  # BERT hidden size
    STAGE1_IMAGE_SIZE = 64
    STAGE2_IMAGE_SIZE = 215
    
    # Text encoder
    ENCODER_MODEL_NAME = "prajjwal1/bert-mini"
    MAX_TEXT_LENGTH = 128
    
    # Training (per referenza)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0002
    EPOCHS = 50
    EPOCHS_S2 = 30
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Loss weights
    LAMBDA_L1_S2 = 10.0
    
    # Paths (aggiustati per HF Spaces)
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
