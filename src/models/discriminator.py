# src/models/discriminator.py
import torch, torch.nn as nn, torch.nn.functional as F

# --- DISCRIMINATORE PATCHGAN DI BASE ---
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 2, 1)
        )
    def forward(self, x): return self.model(x)

# --- DISCRIMINATORE STAGE-I (MULTI-SCALA) ---
class DiscriminatorS1(nn.Module):
    def __init__(self, config, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # Un discriminatore per ogni scala
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(PatchDiscriminator(in_channels=3 + config.TEXT_EMBEDDING_DIM))
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, image, text_embedding):
        outputs = []
        for i in range(self.num_scales):
            disc = self.discriminators[i]
            # Prepara il testo per la concatenazione
            h, w = image.shape[2], image.shape[3]
            text_features = text_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
            # Concatena e discrimina
            input_tensor = torch.cat([image, text_features], dim=1)
            outputs.append(disc(input_tensor))
            # Riduci la risoluzione dell'immagine per la scala successiva
            if i != self.num_scales - 1:
                image = self.downsample(image)
        return outputs

# --- DISCRIMINATORE STAGE-II (SINGOLA SCALA) ---
class DiscriminatorS2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Riusiamo il PatchDiscriminator per coerenza
        self.main = PatchDiscriminator(
            in_channels=3 + config.TEXT_EMBEDDING_DIM,
            base_channels=config.DISCRIMINATOR_BASE_CHANNELS
        )
    def forward(self, image, text_embedding):
        h, w = image.shape[2], image.shape[3]
        text_features = text_embedding.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        input_tensor = torch.cat([image, text_features], dim=1)
        return self.main(input_tensor)