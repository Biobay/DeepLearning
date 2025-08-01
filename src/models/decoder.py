# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# --- Blocchi Costitutivi ---
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_bn),
            nn.BatchNorm2d(out_c) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity()
        )
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

# --- Il Decoder U-Net con Cross-Attention Integrata ---
class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Base number of generator features

        # --- Percorso di Discesa ---
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, ngf, use_bn=False) # In: (B, 3, 256, 256) -> Out: (B, 64, 128, 128)
        self.down2 = DownBlock(ngf, ngf * 2)       # -> (B, 128, 64, 64)
        self.down3 = DownBlock(ngf * 2, ngf * 4)     # -> (B, 256, 32, 32)
        self.down4 = DownBlock(ngf * 4, ngf * 8)     # -> (B, 512, 16, 16)
        
        # --- Bottleneck ---
        self.bottleneck = DownBlock(ngf * 8, ngf * 8) # -> (B, 512, 8, 8)
        
        # --- Percorso di Risalita e Blocchi di Attenzione ---
        self.up1 = UpBlock(ngf * 8, ngf * 8, use_dropout=True, dropout_rate=cfg.DROPOUT_RATE) # In: (B, 512*2, 16, 16)
        self.attn1 = CrossAttentionBlock(query_dim=ngf*8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        self.up2 = UpBlock(ngf * 8 * 2, ngf * 4, use_dropout=True, dropout_rate=cfg.DROPOUT_RATE)
        self.attn2 = CrossAttentionBlock(query_dim=ngf*4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        self.up3 = UpBlock(ngf * 4 * 2, ngf * 2)
        self.attn3 = CrossAttentionBlock(query_dim=ngf*2, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        self.up4 = UpBlock(ngf * 2 * 2, ngf)
        self.attn4 = CrossAttentionBlock(query_dim=ngf, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        
        # --- Layer Finale ---
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def apply_attention(self, x, attn_block, context):
        B, C, H, W = x.shape
        img_features = x.view(B, C, H * W).permute(0, 2, 1)
        attn_features = attn_block(img_features, context)
        return attn_features.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, text_features):
        # 1. Inizia con rumore casuale
        batch_size = text_features.size(0)
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        
        # 2. Percorso di discesa
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        b = self.bottleneck(d4)
        
        # 3. Percorso di risalita con iniezione di testo a ogni passo
        u1 = self.up1(b, d4)
        u1 = self.apply_attention(u1, self.attn1, text_features)

        u2 = self.up2(u1, d3)
        u2 = self.apply_attention(u2, self.attn2, text_features)

        u3 = self.up3(u2, d2)
        u3 = self.apply_attention(u3, self.attn3, text_features)

        u4 = self.up4(u3, d1)
        u4 = self.apply_attention(u4, self.attn4, text_features)
        
        # 4. Output
        out = self.final_up(u4)
        
        final_image = F.interpolate(out, size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), mode='bilinear', align_corners=False)
        
        return self.final_act(final_image), None # Non abbiamo più una singola attention map