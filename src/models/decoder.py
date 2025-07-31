# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# Usiamo di nuovo la versione semplice dell'attention per creare il vettore di contesto
from .attention import MultiHeadCrossAttention 

# --- I blocchi DownBlock e UpBlock rimangono invariati ---
class DownBlock(nn.Module):
    """Blocco di discesa: Conv -> BatchNorm -> LeakyReLU."""
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Blocco di risalita: Upsample -> Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        if use_dropout: self.conv.append(nn.Dropout(dropout_rate))
    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


# --- Il Decoder U-Net con Input Strutturato ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        channels = cfg.UNET_CHANNELS

        # Attention per creare il vettore di contesto iniziale
        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # --- NUOVO BLOCCO DI INPUT ---
        # Questo prende la "tela" di contesto+rumore e la proietta nei canali della U-Net
        self.input_conv = nn.Conv2d(cfg.DECODER_IN_CHANNELS, channels[0], kernel_size=3, padding=1)

        # --- Percorso di Discesa ---
        # Il primo blocco ora prende 'channels[0]' come input
        self.down1 = DownBlock(channels[0], channels[1])
        self.down2 = DownBlock(channels[1], channels[2])
        self.down3 = DownBlock(channels[2], channels[3])
        
        # Bottleneck (semplificato, senza ulteriore downsampling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- Percorso di Risalita ---
        self.up1 = UpBlock(channels[3], channels[2])
        self.up2 = UpBlock(channels[2], channels[1])
        self.up3 = UpBlock(channels[1], channels[0])
        
        # Layer finale
        self.final_conv = nn.Conv2d(channels[0], cfg.OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # 1. Crea il vettore di contesto aggregato dal testo
        context_vector_query = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector_query, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1) # Dim: (B, CONTEXT_DIM)

        # 2. CREA LA "TELA" DI PARTENZA STRUTTURATA
        batch_size = text_features.size(0)
        
        # Espandi il vettore di contesto per creare una mappa di feature spaziale
        context_map = conditioned_vector.unsqueeze(-1).unsqueeze(-1)
        context_map = context_map.repeat(1, 1, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE)
        
        # Crea una mappa di rumore per aggiungere variabilità stocastica
        noise_map = torch.randn(
            batch_size, 
            self.cfg.NUM_NOISE_CHANNELS, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            device=text_features.device
        )
        
        # Concatena la mappa di contesto e quella di rumore per formare l'input
        x = torch.cat([context_map, noise_map], dim=1) # Dim: (B, 256 + 4, 256, 256)

        # 3. Passa la "tela" al blocco di input per iniziare il percorso della U-Net
        x = self.input_conv(x) # Dim: (B, channels[0], 256, 256)

        # 4. Percorso di discesa
        d1 = self.down1(x)   # 256 -> 128
        d2 = self.down2(d1)  # 128 -> 64
        d3 = self.down3(d2)  # 64 -> 32
        
        # 5. Bottleneck
        b = self.bottleneck(d3) # Dim: 32x32

        # 6. Percorso di risalita
        u1 = self.up1(b, d3)   # 32 -> 64
        u2 = self.up2(u1, d2)  # 64 -> 128
        u3 = self.up3(u2, d1)  # 128 -> 256
        
        # 7. Layer finale per mappare ai canali RGB
        x = self.final_conv(u3) # Dim: (B, 3, 256, 256)
        
        # 8. Output alla dimensione richiesta (215x215)
        final_image = F.interpolate(
            x, 
            size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        return self.final_act(final_image), attn_weights