# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Numero di feature base

        # --- Percorso di Discesa (Encoder) ---
        # Ogni blocco dimezza la risoluzione
        self.d1 = nn.Conv2d(cfg.OUTPUT_CHANNELS, ngf, 4, 2, 1)                  # 256 -> 128
        self.d2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2))  # 128 -> 64
        self.d3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf*2, ngf*4, 4, 2, 1), nn.BatchNorm2d(ngf*4))# 64 -> 32
        self.d4 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf*4, ngf*8, 4, 2, 1), nn.BatchNorm2d(ngf*8))# 32 -> 16
        
        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf*8, ngf*8, 4, 2, 1), nn.ReLU()) # 16 -> 8

        # --- Percorso di Risalita (Decoder) ---
        # Ogni blocco raddoppia la risoluzione. I canali di input tengono conto della concatenazione
        self.u1 = nn.Sequential(nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1), nn.BatchNorm2d(ngf*8), nn.ReLU())
        self.u2 = nn.Sequential(nn.ConvTranspose2d(ngf*8*2, ngf*4, 4, 2, 1), nn.BatchNorm2d(ngf*4), nn.ReLU())
        self.u3 = nn.Sequential(nn.ConvTranspose2d(ngf*4*2, ngf*2, 4, 2, 1), nn.BatchNorm2d(ngf*2), nn.ReLU())
        self.u4 = nn.Sequential(nn.ConvTranspose2d(ngf*2*2, ngf, 4, 2, 1), nn.BatchNorm2d(ngf), nn.ReLU())
        
        # --- Blocchi di Attenzione ---
        self.attn1 = CrossAttentionBlock(query_dim=ngf*8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf*4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # --- Layer Finale ---
        self.final_up = nn.Sequential(nn.ConvTranspose2d(ngf*2, cfg.OUTPUT_CHANNELS, 4, 2, 1), nn.Tanh())

    def forward(self, text_features):
        x = torch.randn(
            text_features.size(0), 
            self.cfg.OUTPUT_CHANNELS, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            device=text_features.device
        )
        
        def apply_attention(x_img, attn_block, context):
            B, C, H, W = x_img.shape
            img_feat = x_img.view(B, C, H * W).permute(0, 2, 1)
            attn_feat = attn_block(img_feat, context)
            return attn_feat.permute(0, 2, 1).view(B, C, H, W)

        # --- Percorso di Discesa (salva le skip connections) ---
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        b = self.bottleneck(d4)
        
        # --- Percorso di Risalita (con ordine delle operazioni corretto) ---
        
        # Primo passo di risalita
        u1_out = self.u1(b)
        # Applica l'attention PRIMA della concatenazione
        u1_attn = apply_attention(u1_out, self.attn1, text_features)
        # POI concatena con la skip connection
        u1_cat = torch.cat([u1_attn, d4], 1)
        
        # Secondo passo
        u2_out = self.u2(u1_cat)
        u2_attn = apply_attention(u2_out, self.attn2, text_features)
        u2_cat = torch.cat([u2_attn, d3], 1)

        # Passi successivi (senza attention)
        u3_out = self.u3(u2_cat)
        u3_cat = torch.cat([u3_out, d2], 1)

        u4_out = self.u4(u3_cat)
        u4_cat = torch.cat([u4_out, d1], 1)
        
        out = self.final_up(u4_cat)
        
        final_image = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return final_image, None