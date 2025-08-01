# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention

# =============================================================================
# ## GENERATORE STAGE I (64x64) - INVARIATO ##
# =============================================================================
class GeneratorS1(nn.Module):
    """Generatore dello Stage-I. Produce immagini 64x64."""
    def __init__(self, config):
        super().__init__()
        self.text_embed_dim = config.TEXT_EMBEDDING_DIM
        self.z_dim = config.Z_DIM
        self.base_channels = config.DECODER_BASE_CHANNELS
        
        self.init_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim + self.z_dim, self.base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(self.base_channels * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.attention = MultiHeadCrossAttention(embed_dim=self.text_embed_dim, num_heads=config.NUM_HEADS)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, cls_embedding, hidden_states, z_noise):
        attn_output, attn_weights = self.attention(query=cls_embedding.unsqueeze(1), key_value=hidden_states)
        conditioned_vector = attn_output.squeeze(1)
        combined_input = torch.cat([conditioned_vector, z_noise], dim=1)
        x = self.init_projection(combined_input)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x), attn_weights


# --- BLOCCHI COSTITUTIVI PER LA U-NET DI STAGE II ---
class UNetDown(nn.Module):
    """Blocco di discesa per la U-Net: Conv -> InstanceNorm -> LeakyReLU."""
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    """Blocco di risalita per la U-Net: ConvTranspose -> InstanceNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# =============================================================================
# ## GENERATORE STAGE-II (ARCHITETTURA U-NET CORRETTA) ##
# =============================================================================
class GeneratorS2(nn.Module):
    """Generatore Stage-II basato su architettura U-Net per il raffinamento."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        ngf = config.DECODER_BASE_CHANNELS
        
        # U-Net Encoder (percorso di discesa per l'immagine 64x64)
        self.down1 = UNetDown(3, ngf, normalize=False)  # 64->32
        self.down2 = UNetDown(ngf, ngf * 2)             # 32->16
        self.down3 = UNetDown(ngf * 2, ngf * 4)           # 16->8
        self.down4 = UNetDown(ngf * 4, ngf * 8)           # 8->4
        
        # Bottleneck
        self.bottleneck = UNetDown(ngf * 8, ngf * 8, normalize=False) # 4->2
        
        # Proiezione del testo nel bottleneck
        self.text_projection = nn.Linear(config.TEXT_EMBEDDING_DIM, ngf * 8 * 2 * 2)

        # U-Net Decoder (percorso di risalita)
        # I canali di input tengono conto della concatenazione con la skip connection
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5) # Input dal bottleneck
        self.up2 = UNetUp(ngf * 8 * 2, ngf * 4, dropout=0.5)
        self.up3 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.up4 = UNetUp(ngf * 2 * 2, ngf)
        
        # Layer finale
        self.final_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, stage1_img, text_embedding, _=None): # Accetta un terzo argomento ma lo ignora
        # Percorso di discesa
        d1 = self.down1(stage1_img)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Bottleneck e iniezione del testo
        b = self.bottleneck(d4)
        text_features = self.text_projection(text_embedding).view(-1, self.config.DECODER_BASE_CHANNELS * 8, 2, 2)
        b = b + text_features
        
        # Percorso di risalita
        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        
        out_128 = self.final_up(u4)
        
        # Upsampling finale alla dimensione richiesta
        final_img = F.interpolate(
            out_128, 
            size=(self.config.STAGE2_IMAGE_SIZE, self.config.STAGE2_IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        return final_img, None