# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# --- Blocchi Costitutivi della U-Net ---

class DownBlock(nn.Module):
    """Blocco di discesa: Conv -> BatchNorm -> LeakyReLU."""
    # CORREZIONE: Aggiunto 'use_batch_norm' come argomento
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        layers = [
            # Il bias è spesso disattivato quando si usa la BatchNorm
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_batch_norm)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Blocco di risalita: ConvTranspose -> BatchNorm -> ReLU. Usa skip connections."""
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        # Il Dropout viene applicato dopo l'attivazione per regolarizzare
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        self.block = nn.Sequential(*layers)
            
    def forward(self, x, skip_connection):
        # La concatenazione avviene PRIMA del blocco, unendo l'input upsamplato e la skip
        x = torch.cat([x, skip_connection], dim=1)
        return self.block(x)

# --- Il Decoder U-Net con Cross-Attention ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Numero di feature base del generatore (standard Pix2Pix)

        # Percorso di Discesa (Encoder della U-Net)
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, ngf, use_batch_norm=False) # 256 -> 128
        self.down2 = DownBlock(ngf, ngf * 2)       # 128 -> 64
        self.down3 = DownBlock(ngf * 2, ngf * 4)     # 64 -> 32
        self.down4 = DownBlock(ngf * 4, ngf * 8)     # 32 -> 16
        self.down5 = DownBlock(ngf * 8, ngf * 8)     # 16 -> 8
        self.down6 = DownBlock(ngf * 8, ngf * 8)     # 8 -> 4
        
        # Bottleneck
        self.bottleneck = DownBlock(ngf * 8, ngf * 8) # 4 -> 2
        
        # Percorso di Risalita (Decoder della U-Net)
        self.up1 = UpBlock(ngf * 8, ngf * 8, use_dropout=True, dropout_rate=cfg.DROPOUT_RATE)
        self.up2 = UpBlock(ngf * 8 * 2, ngf * 8, use_dropout=True, dropout_rate=cfg.DROPOUT_RATE)
        self.up3 = UpBlock(ngf * 8 * 2, ngf * 8)
        self.up4 = UpBlock(ngf * 8 * 2, ngf * 4)
        self.up5 = UpBlock(ngf * 4 * 2, ngf * 2)
        self.up6 = UpBlock(ngf * 2 * 2, ngf)
        
        # --- Blocchi di Cross-Attention a diverse risoluzioni ---
        # Li applicheremo nei punti più critici per la formazione della struttura
        self.attn1 = CrossAttentionBlock(query_dim=ngf * 8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf * 4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        
        # Layer Finale
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        batch_size = text_features.size(0)
        # Si parte da rumore puro, la struttura verrà imposta dall'attention
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)

        # Helper per applicare l'attention in modo pulito
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            img_features = x.view(B, C, H * W).permute(0, 2, 1) # (B, H*W, C)
            attn_features = attn_block(img_features, context)
            return attn_features.permute(0, 2, 1).view(B, C, H, W)

        # --- Percorso di discesa ---
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        b = self.bottleneck(d6)

        # --- Percorso di risalita con Cross-Attention ---
        u1 = self.up1(b, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        
        # Applica l'attention a una risoluzione intermedia (16x16 -> 32x32)
        u3_attn = apply_attention(u3, self.attn1, text_features)
        
        u4 = self.up4(u3_attn, d3)
        
        # Applica l'attention a una risoluzione più alta (32x32 -> 64x64)
        u4_attn = apply_attention(u4, self.attn2, text_features)
        
        u5 = self.up5(u4_attn, d2)
        u6 = self.up6(u5, d1)
        
        # --- Output ---
        out = self.final_up(u6) # Output a 256x256
        
        # Ridimensiona alla dimensione richiesta dalla traccia
        final_image = F.interpolate(out, size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), mode='bilinear', align_corners=False)
        
        return self.final_act(final_image), None