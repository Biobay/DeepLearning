import torch
import torch.nn as nn
from src.config import TEXT_EMBEDDING_DIM, DISCRIMINATOR_BASE_CHANNELS

class DiscriminatorS2(nn.Module):
    """
    Discriminatore per la Fase II di StackGAN (DiscriminatorS2).
    Questo discriminatore lavora con immagini a risoluzione 256x256.
    Prende in input un'immagine e un embedding testuale e determina se la coppia
    immagine-testo è reale o generata.
    L'architettura è progettata per catturare dettagli fini nelle immagini ad alta risoluzione.
    """
    def __init__(self, text_embedding_dim=TEXT_EMBEDDING_DIM, d_base_channels=DISCRIMINATOR_BASE_CHANNELS):
        super(DiscriminatorS2, self).__init__()
        
        self.image_encoder = nn.Sequential(
            # Input: 256x256x3
            nn.Conv2d(3, d_base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128x128x64
            nn.Conv2d(d_base_channels, d_base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64x64x128
            nn.Conv2d(d_base_channels * 2, d_base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 32x32x256
            nn.Conv2d(d_base_channels * 4, d_base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 16x16x512
            nn.Conv2d(d_base_channels * 8, d_base_channels * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 8x8x1024
            nn.Conv2d(d_base_channels * 16, d_base_channels * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 4x4x2048
        )

        self.text_compressor = nn.Sequential(
            nn.Linear(text_embedding_dim, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.combined_classifier = nn.Sequential(
            # Input: (2048 + 128) x 4 x 4
            nn.Conv2d(d_base_channels * 32 + 128, d_base_channels * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(d_base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 1024 x 4 x 4
            nn.Conv2d(d_base_channels * 16, 1, 4, 1, 0, bias=False),
            # Output: 1x1x1
            nn.Sigmoid()
        )

    def forward(self, image, text_embedding):
        # 1. Codifica l'immagine
        image_features = self.image_encoder(image)
        
        # 2. Comprimi e prepara l'embedding del testo
        text_features = self.text_compressor(text_embedding)
        # Replicare l'embedding del testo per concatenarlo con le feature dell'immagine
        replicated_text_features = text_features.unsqueeze(-1).unsqueeze(-1)
        replicated_text_features = replicated_text_features.repeat(1, 1, 4, 4)

        # 3. Concatena le feature dell'immagine e del testo
        combined_features = torch.cat([image_features, replicated_text_features], dim=1)
        
        # 4. Classifica
        output = self.combined_classifier(combined_features)
        
        return output.view(-1)
