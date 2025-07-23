import torch
import torch.nn as nn

class DiscriminatorS1(nn.Module):
    """
    Discriminatore per la Fase I di StackGAN (64x64).
    Determina se un'immagine è reale o falsa, condizionata da un embedding di testo.
    """
    def __init__(self, config):
        """
        Args:
            config: Oggetto di configurazione con TEXT_EMBEDDING_DIM e DISCRIMINATOR_BASE_CHANNELS.
        """
        super().__init__()
        
        self.text_embed_dim = config.TEXT_EMBEDDING_DIM
        self.d_base_channels = config.DISCRIMINATOR_BASE_CHANNELS

        # Layer per proiettare l'embedding del testo a una dimensione adeguata
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Blocco convoluzionale principale per l'immagine
        self.conv_block = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, self.d_base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: d_base_channels x 32 x 32

            nn.Conv2d(self.d_base_channels, self.d_base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*2) x 16 x 16

            nn.Conv2d(self.d_base_channels * 2, self.d_base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*4) x 8 x 8

            nn.Conv2d(self.d_base_channels * 4, self.d_base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (d_base_channels*8) x 4 x 4
        )

        # Blocco finale che combina le feature e classifica
        self.final_block = nn.Sequential(
            # Input: (d_base_channels*8 + d_base_channels*4) x H x W
            nn.Conv2d(self.d_base_channels * 8 + self.d_base_channels * 4, self.d_base_channels * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Pooling adattivo per gestire qualsiasi dimensione spaziale della feature map
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Layer lineare finale per la classificazione
        self.final_classifier = nn.Linear(self.d_base_channels * 8, 1)

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


class DiscriminatorS2(nn.Module):
    """Discriminatore per Stage-II (256x256 images)"""
    
    def __init__(self, config):
        super(DiscriminatorS2, self).__init__()
        
        self.d_base_channels = config.DISCRIMINATOR_BASE_CHANNELS
        self.text_embed_dim = config.TEXT_EMBEDDING_DIM
        
        # Rete convoluzionale per le immagini (dimensione flessibile)
        self.conv_block = nn.Sequential(
            # Input: 3 x H x W (dove H, W possono essere 256 o 215)
            nn.Conv2d(3, self.d_base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x H/2 x W/2
            nn.Conv2d(self.d_base_channels, self.d_base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x H/4 x W/4
            nn.Conv2d(self.d_base_channels * 2, self.d_base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x H/8 x W/8
            nn.Conv2d(self.d_base_channels * 4, self.d_base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x H/16 x W/16
            nn.Conv2d(self.d_base_channels * 8, self.d_base_channels * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # 1024 x H/32 x W/32
            nn.Conv2d(self.d_base_channels * 16, self.d_base_channels * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # 2048 x H/64 x W/64
        )
        
        # Pooling adattivo per normalizzare a 4x4
        self.adaptive_conv_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Proiezione per l'embedding del testo
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Blocco finale per la fusione
        self.final_block = nn.Sequential(
            # Input: (2048 + 512) x 4 x 4
            nn.Conv2d(self.d_base_channels * 32 + self.d_base_channels * 8, self.d_base_channels * 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # Ulteriore riduzione
            nn.Conv2d(self.d_base_channels * 16, self.d_base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.d_base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 1 x 1
        )
        
        # Classificatore finale
        self.final_classifier = nn.Linear(self.d_base_channels * 8, 1)
    
    def forward(self, image, text_embedding):
        """
        Forward pass del DiscriminatorS2.
        
        Args:
            image: Immagine HxW [B, 3, H, W] (può essere 256x256 o 215x215)
            text_embedding: Text embedding [B, text_dim]
            
        Returns:
            output: Logit di discriminazione [B]
        """
        # 1. Estrai feature dall'immagine
        image_features = self.conv_block(image)  # [B, 2048, H', W']
        
        # 2. Normalizza le feature dell'immagine a 4x4 usando pooling adattivo
        image_features = self.adaptive_conv_pool(image_features)  # [B, 2048, 4, 4]
        
        # 3. Proietta l'embedding del testo
        text_features = self.text_projection(text_embedding)  # [B, 512]
        
        # 4. Prepara le feature del testo per la concatenazione
        h, w = image_features.shape[2], image_features.shape[3]  # 4, 4
        text_features = text_features.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)  # [B, 512, 4, 4]
        
        # 5. Concatena le feature
        combined_features = torch.cat([image_features, text_features], dim=1)  # [B, 2560, 4, 4]
        
        # 6. Blocco finale
        x = self.final_block(combined_features)  # [B, 512, 1, 1]
        
        # 7. Appiattisci e classifica
        x = x.view(x.size(0), -1)  # [B, 512]
        output = self.final_classifier(x)  # [B, 1]
        
        return output.squeeze()  # [B]
