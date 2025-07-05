import torch
import torch.nn as nn
from src.config import TEXT_EMBEDDING_DIM, DISCRIMINATOR_BASE_CHANNELS

class DiscriminatorS1(nn.Module):
    """
    Discriminatore per la Fase I di StackGAN (64x64).
    Determina se un'immagine è reale o falsa, condizionata da un embedding di testo.
    """
    def __init__(self, text_embedding_dim=TEXT_EMBEDDING_DIM, d_base_channels=DISCRIMINATOR_BASE_CHANNELS):
        """
        Args:
            text_embedding_dim (int): Dimensione dell'embedding del testo.
            d_base_channels (int): Numero di canali base per i layer convoluzionali.
        """
        super().__init__()

        # Layer per proiettare l'embedding del testo a una dimensione adeguata
        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Blocco convoluzionale principale per l'immagine
        self.conv_block = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, d_base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: d_base_channels x 32 x 32

            nn.Conv2d(d_base_channels, d_base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*2) x 16 x 16

            nn.Conv2d(d_base_channels * 2, d_base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*4) x 8 x 8

            nn.Conv2d(d_base_channels * 4, d_base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*8) x 4 x 4
        )

        # Blocco finale che combina le feature e classifica
        self.final_block = nn.Sequential(
            # Input: (d_base_channels*8 + d_base_channels*4) x H x W
            nn.Conv2d(d_base_channels * 8 + d_base_channels * 4, d_base_channels * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Pooling adattivo per gestire qualsiasi dimensione spaziale della feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Layer lineare finale per la classificazione
        self.final_classifier = nn.Linear(d_base_channels * 8, 1)

    def forward(self, image, text_embedding):
        """
        Passaggio forward.

        Args:
            image (torch.Tensor): Immagine di input. Dim: (batch_size, 3, H, W)
            text_embedding (torch.Tensor): Embedding del testo [CLS]. Dim: (batch_size, text_embedding_dim)

        Returns:
            torch.Tensor: Logit di output. Dim: (batch_size)
        """
        # 1. Estrai feature dall'immagine
        image_features = self.conv_block(image)

        # 2. Proietta l'embedding del testo
        text_features = self.text_projection(text_embedding)
        
        # 3. Prepara le feature del testo per la concatenazione in modo dinamico
        h, w = image_features.shape[2], image_features.shape[3]
        text_features = text_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)

        # 4. Concatena le feature dell'immagine e del testo
        combined_features = torch.cat([image_features, text_features], dim=1)

        # 5. Passa attraverso il blocco convoluzionale finale
        x = self.final_block(combined_features)

        # 6. Applica il pooling adattivo per ridurre la dimensione spaziale a 1x1
        x = self.adaptive_pool(x)

        # 7. Appiattisci il tensore per il layer lineare
        x = x.view(x.size(0), -1)

        # 8. Ottieni il logit finale dal classificatore
        output = self.final_classifier(x)

        # Rimuovi le dimensioni singleton per avere (batch_size)
        return output.squeeze()
