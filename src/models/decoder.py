# src/models/decoder.py (Versione CORRETTA e DEFINITIVA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention

# --- Blocchi Costitutivi della U-Net ---

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
    """
    Blocco di risalita CORRETTO: Upsample -> Conv -> BatchNorm -> ReLU.
    Questo approccio è più stabile per le dimensioni rispetto a ConvTranspose.
    """
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        # Usiamo Upsample + Conv2d invece di ConvTranspose2d per un miglior controllo
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            # L'input channel è doppio per via della concatenazione
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        if use_dropout:
            self.conv.append(nn.Dropout(dropout_rate))

    def forward(self, x, skip_connection):
        x = self.up(x)
        # La concatenazione avviene DOPO l'upsampling
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


# --- Il Decoder U-Net ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        channels = cfg.UNET_CHANNELS # es. (64, 128, 256, 512)

        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.text_projection = nn.Linear(cfg.CONTEXT_DIM, channels[-1] * 2) # Proiettiamo per la modulazione

        # Percorso di Discesa
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, channels[0], use_batch_norm=False)
        self.down2 = DownBlock(channels[0], channels[1])
        self.down3 = DownBlock(channels[1], channels[2])
        self.down4 = DownBlock(channels[2], channels[3])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=4, stride=2, padding=1), # Da 16x16 a 8x8
            nn.ReLU()
        )
        
        # Percorso di Risalita
        self.up1 = UpBlock(channels[3], channels[2])
        self.up2 = UpBlock(channels[2], channels[1])
        self.up3 = UpBlock(channels[1], channels[0])
        self.up4 = UpBlock(channels[0], channels[0])
        
        # Layer finale
        self.final_conv = nn.Conv2d(channels[0], cfg.OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # 1. Crea il vettore di contesto
        context_vector = text_features.mean(dim=1).unsqueeze(1)
        attn_output, _ = self.attention(query=context_vector, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        # 2. Inizia con rumore
        batch_size = text_features.size(0)
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        
        # 3. Percorso di discesa
        d1 = self.down1(x)   # 256 -> 128
        d2 = self.down2(d1) # 128 -> 64
        d3 = self.down3(d2) # 64 -> 32
        d4 = self.down4(d3) # 32 -> 16
        
        # 4. Bottleneck
        b = self.bottleneck(d4) # 16 -> 8
        
        # 5. Iniezione del Testo nel Bottleneck (Modulazione FiLM-like)
        text_proj = self.text_projection(conditioned_vector)
        # Dividiamo il vettore proiettato in due parti: una per la scala (gamma) e una per lo shift (beta)
        gamma, beta = torch.chunk(text_proj, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        # Applichiamo la modulazione
        b = gamma * b + beta

        # 6. Percorso di risalita
        u1 = self.up1(b, d4)   # 8 -> 16 (concatena con d4 che è 16x16)
        u2 = self.up2(u1, d3)  # 16 -> 32 (concatena con d3 che è 32x32)
        u3 = self.up3(u2, d2)  # 32 -> 64 (concatena con d2 che è 64x64)
        u4 = self.up4(u3, d1)  # 64 -> 128 (concatena con d1 che è 128x128)

        # 7. Layer finale per mappare ai canali RGB
        # Upsampliamo fino alla dimensione finale e poi applichiamo l'ultima convoluzione
        u4 = F.interpolate(u4, size=self.cfg.MODEL_INTERNAL_SIZE, mode='bilinear', align_corners=False)
        x = self.final_conv(u4)
        
        # 8. Output alla dimensione richiesta (215x215)
        final_image = F.interpolate(
            x, 
            size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        return self.final_act(final_image), _ # Restituisci anche i pesi di attention