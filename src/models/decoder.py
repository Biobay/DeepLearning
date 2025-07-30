# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock # Importiamo il nuovo blocco di attenzione

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
    """Blocco di risalita: Upsample -> Conv -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            # L'input channel è doppio per la skip connection
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        if use_dropout:
            self.conv.append(nn.Dropout(dropout_rate))

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = torch.cat([x, skip_connection], dim=1)
        return self.conv(x)


# --- Il Decoder U-Net con Cross-Attention ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        channels = cfg.UNET_CHANNELS # es. (64, 128, 256, 512)

        # Percorso di Discesa (Encoder della U-Net)
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, channels[0], use_batch_norm=False)
        self.down2 = DownBlock(channels[0], channels[1])
        self.down3 = DownBlock(channels[1], channels[2])
        self.down4 = DownBlock(channels[2], channels[3])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[3], channels[3], kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Percorso di Risalita (Decoder della U-Net)
        reversed_channels = list(reversed(channels))
        
        self.up1 = UpBlock(reversed_channels[0], reversed_channels[1])
        self.up2 = UpBlock(reversed_channels[1], reversed_channels[2])
        self.up3 = UpBlock(reversed_channels[2], reversed_channels[3])
        self.up4 = UpBlock(reversed_channels[3], reversed_channels[3]) # Output ha canali = channels[0]
        
        # --- Blocchi di Cross-Attention ---
        # Un blocco di attenzione per ogni livello di risalita
        self.attn1 = CrossAttentionBlock(query_dim=reversed_channels[1], context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=reversed_channels[2], context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn3 = CrossAttentionBlock(query_dim=reversed_channels[3], context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn4 = CrossAttentionBlock(query_dim=reversed_channels[3], context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # Layer finale
        self.final_conv = nn.Conv2d(reversed_channels[3], cfg.OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # text_features ha dimensioni (Batch, SeqLen, ContextDim)

        # 1. Inizia con rumore casuale alla dimensione interna del modello
        batch_size = text_features.size(0)
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        
        # 2. Percorso di discesa, salvando le skip connections
        d1 = self.down1(x)   # Dim: 128x128
        d2 = self.down2(d1)  # Dim: 64x64
        d3 = self.down3(d2)  # Dim: 32x32
        d4 = self.down4(d3)  # Dim: 16x16
        
        # 3. Bottleneck
        b = self.bottleneck(d4) # Dim: 8x8

        # 4. Percorso di risalita con iniezione di testo a ogni passo
        # Funzione helper per applicare l'attenzione
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            # Prepara le feature dell'immagine: (B, C, H, W) -> (B, H*W, C)
            img_features = x.view(B, C, H * W).permute(0, 2, 1)
            # Applica l'attenzione
            attn_features = attn_block(img_features, context)
            # Riporta le feature alla forma di immagine: (B, H*W, C) -> (B, C, H, W)
            return attn_features.permute(0, 2, 1).view(B, C, H, W)

        x = self.up1(b, d4)   # Risale a 16x16
        x = apply_attention(x, self.attn1, text_features)

        x = self.up2(x, d3)   # Risale a 32x32
        x = apply_attention(x, self.attn2, text_features)
        
        x = self.up3(x, d2)   # Risale a 64x64
        x = apply_attention(x, self.attn3, text_features)
        
        x = self.up4(x, d1)   # Risale a 128x128
        x = apply_attention(x, self.attn4, text_features)

        # 5. Layer finale per mappare ai canali RGB e alla dimensione corretta
        x = F.interpolate(x, size=self.cfg.MODEL_INTERNAL_SIZE, mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        
        # 6. Output alla dimensione richiesta dalla traccia (215x215)
        final_image = F.interpolate(
            x, 
            size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Ritorniamo None per i pesi di attenzione perché ora sono distribuiti su più layer
        return self.final_act(final_image), None