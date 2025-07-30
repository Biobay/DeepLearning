# src/models/discriminator.py

import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    """Blocco convoluzionale per il Discriminatore."""
    def __init__(self, in_channels, out_channels, stride=2, use_batch_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    """
    Discriminatore PatchGAN condizionato.
    Prende in input sia l'immagine reale che quella generata (o target)
    e determina se la coppia è plausibile.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # L'input sono due immagini concatenate, quindi 3+3 = 6 canali
        self.initial_block = DiscriminatorBlock(in_channels * 2, 64, use_batch_norm=False)
        
        self.model = nn.Sequential(
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            # L'ultimo blocco ha stride 1
            DiscriminatorBlock(256, 512, stride=1),
            # Convoluzione finale per produrre l'output (la "patch" di predizioni)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, generated_or_real_image, target_image):
        """
        Args:
            generated_or_real_image (torch.Tensor): Immagine prodotta dal generatore o dal dataset.
            target_image (torch.Tensor): Immagine reale usata come condizionamento.
        """
        # Concatena le due immagini lungo la dimensione dei canali
        x = torch.cat([generated_or_real_image, target_image], dim=1)
        x = self.initial_block(x)
        return self.model(x)