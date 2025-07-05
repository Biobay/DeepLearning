import torch
import torch.nn as nn
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1

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
        
        self.decoder = GeneratorS1(config)

    def forward(self, input_ids, attention_mask, z_noise=None):
        """
        Passaggio forward del modello.

        Args:
            input_ids (torch.Tensor): ID dei token di input.
            attention_mask (torch.Tensor): Maschera di attenzione.
            z_noise (torch.Tensor, opzionale): Vettore di rumore latente.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Immagine generata e pesi di attenzione.
        """
        # L'encoder restituisce sia [CLS] che hidden states
        cls_embedding, hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if z_noise is None:
            batch_size = input_ids.size(0)
            device = input_ids.device
            z_dim = self.decoder.z_dim
            z_noise = torch.randn(batch_size, z_dim, device=device)
        generated_image, attention_weights = self.decoder(cls_embedding, hidden_states, z_noise)
        return generated_image, attention_weights
