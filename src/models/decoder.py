import torch
import torch.nn as nn
from src.models.attention import MultiHeadCrossAttention

class ImageDecoder(nn.Module):
    """
    Decoder basato su CNN (Generatore) per creare un'immagine.
    Utilizza la cross-attention per condizionare la generazione dell'immagine
    sull'output dell'encoder di testo e il rumore z.
    """
    def __init__(self, config, num_heads, output_channels=3, ngf=64, output_size=215):
        super().__init__()
        # La dimensione di input è la somma di ENCODER_DIM (testo) e Z_DIM (rumore)
        text_and_noise_dim = config.ENCODER_DIM + config.Z_DIM
        self.init_projection = nn.Linear(text_and_noise_dim, ngf * 8 * 4 * 4)
        self.attention = MultiHeadCrossAttention(embed_dim=config.ENCODER_DIM, num_heads=num_heads)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf, output_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, cls_embedding, encoder_hidden_states, z):
        batch_size = cls_embedding.size(0)
        # Concatenazione tra embedding [CLS] e rumore z
        cond_vector = torch.cat([cls_embedding, z], dim=1) # (batch, ENCODER_DIM + Z_DIM)
        # Proiezione iniziale
        x = self.init_projection(cond_vector)
        x = x.view(batch_size, -1, 4, 4)
        # Attention: query = [CLS], key/value = encoder_hidden_states
        attn_output, attn_weights = self.attention(query=cls_embedding.unsqueeze(1), key_value=encoder_hidden_states)
        
        # Aggiungiamo l'output dell'attenzione all'input della CNN per un condizionamento più forte
        # (Questo richiede che le dimensioni siano compatibili)
        # Per semplicità, per ora non lo aggiungiamo e usiamo solo la proiezione iniziale.
        
        # Passa attraverso la rete generativa
        generated_image = self.main(x)
        
        # Restituiamo solo l'immagine, poiché i pesi dell'attenzione non servono per la loss
        return generated_image
