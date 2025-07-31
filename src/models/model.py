# src/models/model.py
import torch.nn as nn
from .encoder import TextEncoder
from .decoder import StyleBasedGenerator
from .discriminator import Discriminator

class PikaPikaGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME, fine_tune=config.FINE_TUNE_ENCODER)
        self.decoder = StyleBasedGenerator(config)
        self.discriminator = Discriminator(in_channels=config.OUTPUT_CHANNELS)

    def forward_generator(self, input_ids, attention_mask):
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        generated_image, _ = self.decoder(text_features)
        return generated_image, None