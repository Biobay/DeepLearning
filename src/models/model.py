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
        super(PikaPikaGen, self).__init__()
        self.config = config
        self.encoder = TextEncoder(config)
        # Passa il numero di teste di attenzione al decoder
        self.decoder = ImageDecoder(config, num_heads=config.NUM_HEADS)

    def forward(self, input_ids, attention_mask, z=None):
        """
        Passaggio forward del modello.

        Args:
            input_ids (torch.Tensor): ID dei token di input.
            attention_mask (torch.Tensor): Maschera di attenzione.
            z (torch.Tensor, optional): Vettore di rumore latente. Defaults to None.

        Returns:
            Tuple[torch.Tensor, None]: Immagini generate e un placeholder.
        """
        cls_embedding, all_text_features = self.encoder(input_ids, attention_mask)
        
        # Passa tutti gli output dell'encoder e il rumore z al decoder
        generated_images = self.decoder(
            cls_embedding=cls_embedding, 
            encoder_hidden_states=all_text_features, 
            z=z
        )
        
        # La tupla è per mantenere la coerenza con l'output atteso nel ciclo di training
        return generated_images, None
