# src/models/model.py
import torch.nn as nn
from .encoder import TextEncoder
from .decoder import UNetDecoder
from .discriminator import MultiScaleDiscriminator

class PikaPikaGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TextEncoder(model_name=config.ENCODER_MODEL_NAME, fine_tune=config.FINE_TUNE_ENCODER)
        self.decoder = UNetDecoder(config)
        self.discriminator = MultiScaleDiscriminator(in_channels=config.OUTPUT_CHANNELS)
    def forward_generator(self, input_ids, attention_mask):
        text_features = self.encoder(input_ids, attention_mask)
        return self.decoder(text_features)