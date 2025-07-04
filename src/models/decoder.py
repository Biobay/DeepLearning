import torch
import torch.nn as nn
from src.models.attention import MultiHeadCrossAttention

class ImageDecoder(nn.Module):
    """
    Decoder basato su CNN (Generatore) per creare un'immagine.
    Utilizza la cross-attention per condizionare la generazione dell'immagine
    sull'output dell'encoder di testo.
    """
    def __init__(self, text_embed_dim, num_heads, output_channels=3, ngf=64, output_size=215):
        super().__init__()
        
        # Proiezione per il vettore condizionato
        self.init_projection = nn.Linear(text_embed_dim, ngf * 8 * 4 * 4)

        # Modulo di Attention
        self.attention = MultiHeadCrossAttention(embed_dim=text_embed_dim, num_heads=num_heads)
        
        # Rete generativa CNN (basata su DCGAN)
        self.main = nn.Sequential(
            # Input: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size. (ngf) x 64 x 64
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf, output_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
            # Final size. (output_channels) x 64 x 64
        )

    def forward(self, text_features):
        """
        Args:
            text_features (torch.Tensor): Output dell'encoder di testo.
                                          Dim: (batch_size, seq_len, text_embed_dim)

        Returns:
            torch.Tensor: Immagine generata. Dim: (batch_size, 3, 64, 64)
        """
        batch_size = text_features.size(0)
        
        # 1. Aggrega le feature del testo in un singolo vettore di contesto
        #    Usiamo la media, ma si potrebbe usare anche l'output di [CLS] di BERT
        context_vector = text_features.mean(dim=1).unsqueeze(1) # (batch, 1, embed_dim)

        # 2. Applica l'attenzione
        #    La query è il contesto aggregato, K/V sono le feature complete
        attn_output, attn_weights = self.attention(query=context_vector, key_value=text_features)
        
        # Rimuoviamo la dimensione "sequence" che era 1
        conditioned_vector = attn_output.squeeze(1) # (batch, embed_dim)

        # 3. Genera l'immagine a partire dal vettore condizionato
        #    Invece di partire da rumore casuale, partiamo da una proiezione
        #    del nostro vettore condizionato per guidare la generazione.
        #    (Questa è una semplificazione, un approccio più avanzato come in StyleGAN
        #     inietterebbe l'informazione a più livelli del generatore)
        
        # Proiettiamo il vettore condizionato alla dimensione iniziale del generatore
        x = self.init_projection(conditioned_vector)
        x = x.view(batch_size, -1, 4, 4) # Reshape a (batch, ngf*8, 4, 4)
        
        # Passa attraverso la rete generativa
        generated_image = self.main(x)
        
        return generated_image, attn_weights
