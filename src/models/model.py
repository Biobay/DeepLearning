# src/models/model.py

import torch.nn as nn
from .encoder import TextEncoder
from .decoder import UNetDecoder # <-- CORREZIONE: Usa UNetDecoder, non StyleBasedGenerator
from .discriminator import MultiScaleDiscriminator

class PikaPikaGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME, fine_tune=config.FINE_TUNE_ENCODER)
        
        # Usa la nuova classe UNetDecoder
        self.decoder = UNetDecoder(config)
        
        # Il discriminatore è ancora definito, pronto per essere riattivato in futuro
        self.discriminator = MultiScaleDiscriminator(in_channels=config.OUTPUT_CHANNELS)

    def forward_generator(self, input_ids, attention_mask):
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Il decoder ora si aspetta l'intera sequenza di text_features
        generated_image, _ = self.decoder(text_features)
        return generated_image, None