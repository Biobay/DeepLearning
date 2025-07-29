# In src/models/decoder.py

import torch
import torch.nn as nn
from src.models.attention import MultiHeadCrossAttention

class ImageDecoder(nn.Module):
    """
    Decoder basato su CNN (Generatore) per creare un'immagine.
    Utilizza la cross-attention per condizionare la generazione dell'immagine
    sull'output dell'encoder di testo.
    ## MODIFICA: Aggiunto Dropout2d per regolarizzazione spaziale.
    """
    
    def __init__(self, text_embed_dim, num_heads, output_channels=3, ngf=64, output_size=215, dropout_rate=0.5):
        super().__init__()
        
        self.init_projection = nn.Linear(text_embed_dim, ngf * 8 * 4 * 4)
        self.attention = MultiHeadCrossAttention(embed_dim=text_embed_dim, num_heads=num_heads)
        
        # Rete generativa CNN (basata su DCGAN)
        self.main = nn.Sequential(
            # Input: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # ## CORREZIONE: Usare Dropout2d per input 4D (immagini/feature maps) ##
            nn.Dropout2d(dropout_rate), 

            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate), 

            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),

            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Nota: Non si mette il dropout subito prima del layer di output
            
            # State size. (ngf) x 64 x 64
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf, output_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, text_features):
        batch_size = text_features.size(0)
        context_vector = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        x = self.init_projection(conditioned_vector)
        x = x.view(batch_size, -1, 4, 4)
        
        generated_image = self.main(x)
        
        return generated_image, attn_weights