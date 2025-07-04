import torch
import torch.nn as nn
from src.models.encoder import TextEncoder
from src.models.decoder import DecoderWithAttention

class PikaPikaGen(nn.Module):
    """
    Modello end-to-end che combina l'encoder di testo e il decoder di immagini con attenzione.
    Prende in input una descrizione testuale e genera un'immagine corrispondente.
    """
    def __init__(self, encoder_model_name, encoder_dim, decoder_dim, attention_dim, context_dim, output_channels=3, ngf=64, fine_tune_encoder=True):
        """
        Args:
            encoder_model_name (str): Nome del modello per l'encoder (es. 'prajjwal1/bert-mini').
            encoder_dim (int): Dimensione dell'output dell'encoder.
            decoder_dim (int): Dimensione dello stato nascosto del decoder (LSTM).
            attention_dim (int): Dimensione interna del meccanismo di attenzione.
            context_dim (int): Dimensione del vettore di contesto per l'ImageDecoder.
            output_channels (int): Canali dell'immagine di output (3 per RGB).
            ngf (int): Numero di feature nel generatore.
            fine_tune_encoder (bool): Se fare il fine-tuning dei pesi dell'encoder.
        """
        super().__init__()
        
        # 1. Encoder di Testo
        self.encoder = TextEncoder(
            model_name=encoder_model_name,
            fine_tune=fine_tune_encoder
        )
        
        # 2. Decoder di Immagini con Attenzione
        self.decoder = DecoderWithAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            context_dim=context_dim,
            output_channels=output_channels,
            ngf=ngf
        )

    def forward(self, input_ids, attention_mask):
        """
        Passaggio forward del modello completo.

        Args:
            input_ids (torch.Tensor): Tensor degli ID dei token dal tokenizer.
                                      Dim: (batch_size, seq_len)
            attention_mask (torch.Tensor): Maschera di attenzione dal tokenizer.
                                           Dim: (batch_size, seq_len)

        Returns:
            torch.Tensor: Immagine generata. Dim: (batch_size, C, H, W)
            torch.Tensor: Pesi di attenzione. Dim: (batch_size, seq_len)
        """
        # Passa il testo attraverso l'encoder per ottenere gli hidden states
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # Passa gli hidden states al decoder per generare l'immagine
        generated_image, attention_weights = self.decoder(encoder_output)
        
        return generated_image, attention_weights
