# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# --- ARCHITETTURA U-NET ROBUSTA (ISPIRATA A PIX2PIXHD) ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Numero di feature base

        # --- BLOCCHI DI DISCESA (ENCODER) ---
        self.down1 = nn.Conv2d(cfg.OUTPUT_CHANNELS, ngf, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2))
        self.down3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1), nn.BatchNorm2d(ngf * 4))
        self.down4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1), nn.BatchNorm2d(ngf * 8))
        self.down5 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1), nn.BatchNorm2d(ngf * 8))
        
        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1), nn.ReLU())
        
        # --- BLOCCHI DI RISALITA (DECODER) ---
        self.up1 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1), nn.BatchNorm2d(ngf * 8))
        self.up2 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1), nn.BatchNorm2d(ngf * 8))
        self.up3 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1), nn.BatchNorm2d(ngf * 4))
        self.up4 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1), nn.BatchNorm2d(ngf * 2))
        self.up5 = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf))
        
        # --- LAYER FINALE ---
        self.final_up = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, 4, 2, 1), nn.Tanh())
        
        # --- BLOCCHI DI ATTENZIONE ---
        self.attn1 = CrossAttentionBlock(query_dim=ngf * 8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf * 4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

    def forward(self, text_features):
        batch_size = text_features.size(0)
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            img_feat = x.view(B, C, H * W).permute(0, 2, 1)
            attn_feat = attn_block(img_feat, context)
            return attn_feat.permute(0, 2, 1).view(B, C, H, W)

        # --- PERCORSO DI DISCESA ---
        d1 = self.down1(x)    # 256 -> 128
        d2 = self.down2(d1)   # 128 -> 64
        d3 = self.down3(d2)   # 64 -> 32
        d4 = self.down4(d3)   # 32 -> 16
        d5 = self.down5(d4)   # 16 -> 8
        
        b = self.bottleneck(d5) # 8 -> 4
        
        # --- PERCORSO DI RISALITA ---
        u1 = self.up1(b)                 # 4 -> 8
        u1 = torch.cat([u1, d5], dim=1)  # Concatena con d5 (8x8)
        u1 = apply_attention(u1, self.attn1, text_features) # Applica attention qui
        
        u2 = self.up2(u1)                # 8 -> 16
        u2 = torch.cat([u2, d4], dim=1)  # Concatena con d4 (16x16)
        
        u3 = self.up3(u2)                # 16 -> 32
        u3 = torch.cat([u3, d3], dim=1)  # Concatena con d3 (32x32)
        u3 = apply_attention(u3, self.attn2, text_features) # Applica attention qui
        
        u4 = self.up4(u3)                # 32 -> 64
        u4 = torch.cat([u4, d2], dim=1)  # Concatena con d2 (64x64)
        
        u5 = self.up5(u4)                # 64 -> 128
        u5 = torch.cat([u5, d1], dim=1)  # Concatena con d1 (128x128)
        
        out = self.final_up(u5)          # 128 -> 256
        
        final_image = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return final_image, None