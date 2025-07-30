# src/models/model.py

import torch.nn as nn
from .encoder import TextEncoder
from .decoder import UNetDecoder
from .discriminator import Discriminator # <-- NUOVO IMPORT

class PikaPikaGen(nn.Module):
    """
    Modello completo che ora include Generatore e Discriminatore per il training GAN.
    """
    def __init__(self, config):
        super().__init__()
        
        # Il Generatore è composto da Encoder e Decoder U-Net
        self.encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME,
            fine_tune=config.FINE_TUNE_ENCODER
        )
        self.decoder = UNetDecoder(config)
        
        # Aggiungiamo il Discriminatore
        self.discriminator = Discriminator(in_channels=config.OUTPUT_CHANNELS)

    def forward_generator(self, input_ids, attention_mask):
        """Esegue solo il forward pass del generatore."""
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        generated_image, attention_weights = self.decoder(text_features)
        return generated_image, attention_weights