
# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention

# --- Blocchi Costitutivi (Semplificati e Robusti) ---

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_batch_norm)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
    def forward(self, x, skip_connection):
        x = self.relu(self.bn(self.conv_transpose(x)))
        x = self.dropout(x)
        # La concatenazione avviene DOPO l'upsampling
        x = torch.cat([x, skip_connection], dim=1)
        return x

# --- Il Decoder U-Net con Input Strutturato ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Usiamo la versione semplice dell'attention per creare il vettore di contesto
        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # Il primo layer della U-Net prenderà in input il testo e il rumore
        # e produrrà una mappa di feature con 'ngf * 8' canali.
        # NGF (Number of Generator Features) è un parametro comune in DCGAN/Pix2Pix
        ngf = 64 

        # --- Percorso di Discesa (Encoder della U-Net) ---
        # Input: (B, C_in, 256, 256)
        self.down1 = DownBlock(cfg.DECODER_IN_CHANNELS, ngf, use_batch_norm=False) # Out: (B, ngf, 128, 128)
        self.down2 = DownBlock(ngf, ngf * 2)       # Out: (B, ngf*2, 64, 64)
        self.down3 = DownBlock(ngf * 2, ngf * 4)     # Out: (B, ngf*4, 32, 32)
        self.down4 = DownBlock(ngf * 4, ngf * 8)     # Out: (B, ngf*8, 16, 16)
        self.down5 = DownBlock(ngf * 8, ngf * 8)     # Out: (B, ngf*8, 8, 8)
        self.down6 = DownBlock(ngf * 8, ngf * 8)     # Out: (B, ngf*8, 4, 4)
        
        # --- Bottleneck ---
        self.bottleneck = DownBlock(ngf * 8, ngf * 8) # Out: (B, ngf*8, 2, 2)
        
        # --- Percorso di Risalita (Decoder della U-Net) ---
        self.up1 = UpBlock(ngf * 8, ngf * 8, use_dropout=True) # In: (B, ngf*8*2, 4, 4)
        self.up2 = UpBlock(ngf * 8 * 2, ngf * 8, use_dropout=True)
        self.up3 = UpBlock(ngf * 8 * 2, ngf * 8)
        self.up4 = UpBlock(ngf * 8 * 2, ngf * 4)
        self.up5 = UpBlock(ngf * 4 * 2, ngf * 2)
        self.up6 = UpBlock(ngf * 2 * 2, ngf)
        
        # --- Layer Finale ---
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # 1. Crea il vettore di contesto aggregato dal testo
        context_vector_query = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector_query, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        # 2. Crea la "tela" di partenza strutturata
        batch_size = text_features.size(0)
        context_map = conditioned_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE)
        noise_map = torch.randn(batch_size, self.cfg.NUM_NOISE_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        x = torch.cat([context_map, noise_map], dim=1)

        # 3. Percorso di discesa
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        b = self.bottleneck(d6)
        
        # 4. Percorso di risalita
        u1 = self.up1(b, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        
        # 5. Output
        out = self.final_up(u6) # out è (B, 3, 256, 256)
        
        # 6. Ridimensiona alla dimensione richiesta
        final_image = F.interpolate(out, size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), mode='bilinear', align_corners=False)
        
        return self.final_act(final_image), attn_weights