# src/models/model.py (versione CORRETTA)

import torch.nn as nn
from src.models.encoder import TextEncoder
from src.models.decoder import UNetDecoder # <-- 1. CAMBIA NOME NELL'IMPORT

class PikaPikaGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME,
            fine_tune=config.FINE_TUNE_ENCODER
        )
        
        # 2. USA LA NUOVA CLASSE E PASSA IL CONFIG COMPLETO
        self.decoder = UNetDecoder(config)

    def forward(self, input_ids, attention_mask):
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Il nuovo decoder si aspetta solo le feature del testo
        generated_image, attention_weights = self.decoder(text_features)
        return generated_image, attention_weights