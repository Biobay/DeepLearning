import torch
import torch.nn as nn
from src.models.encoder import TextEncoder
from src.models.decoder import ImageDecoder

class PikaPikaGen(nn.Module):
    """
    Modello end-to-end che combina l'encoder di testo e il decoder di immagini.
    """
    def __init__(self, config):
        """
        Args:
            config: Oggetto o dizionario con i parametri di configurazione.
        """
        super().__init__()
        
        self.encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME,
            fine_tune=config.FINE_TUNE_ENCODER
        )
        
        self.decoder = ImageDecoder(
            text_embed_dim=config.ENCODER_DIM,
            num_heads=config.NUM_HEADS,
            output_channels=config.OUTPUT_CHANNELS,
            ngf=config.NGF
        )

    def forward(self, input_ids, attention_mask):
        """
        Passaggio forward del modello.

        Args:
            input_ids (torch.Tensor): ID dei token di input.
            attention_mask (torch.Tensor): Maschera di attenzione.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Immagine generata e pesi di attenzione.
        """
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        generated_image, attention_weights = self.decoder(text_features)
        return generated_image, attention_weights
