# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# --- Architettura U-Net Robusta ---

class UNetDownBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64

        # --- Percorso di Discesa (Encoder) ---
        self.down1 = UNetDownBlock(cfg.OUTPUT_CHANNELS, ngf, norm=False)
        self.down2 = UNetDownBlock(ngf, ngf * 2)
        self.down3 = UNetDownBlock(ngf * 2, ngf * 4)
        self.down4 = UNetDownBlock(ngf * 4, ngf * 8)
        self.down5 = UNetDownBlock(ngf * 8, ngf * 8)
        
        # --- Bottleneck ---
        self.bottleneck = UNetDownBlock(ngf * 8, ngf * 8)
        
        # --- Percorso di Risalita (Decoder) ---
        # Nota: in_channels ora tiene conto della concatenazione
        self.up1 = UNetUpBlock(ngf * 8, ngf * 8)
        self.up2 = UNetUpBlock(ngf * 8 * 2, ngf * 4)
        self.up3 = UNetUpBlock(ngf * 4 * 2, ngf * 2)
        self.up4 = UNetUpBlock(ngf * 2 * 2, ngf)
        self.up5 = UNetUpBlock(ngf * 2, ngf) # Aggiunto per simmetria
        
        # --- Blocchi di Attenzione ---
        self.attn1 = CrossAttentionBlock(query_dim=ngf * 8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf * 4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # --- Layer Finale ---
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, 4, 2, 1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        batch_size = text_features.size(0)
        x = torch.randn(
            batch_size, 
            self.cfg.OUTPUT_CHANNELS, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            self.cfg.MODEL_INTERNAL_SIZE, 
            device=text_features.device
        )
        
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            img_feat = x.view(B, C, H * W).permute(0, 2, 1)
            attn_feat = attn_block(img_feat, context)
            return attn_feat.permute(0, 2, 1).view(B, C, H, W)

        # --- Percorso di Discesa ---
        d1 = self.down1(x)    # Out: 128x128, ngf
        d2 = self.down2(d1)   # Out: 64x64, ngf*2
        d3 = self.down3(d2)   # Out: 32x32, ngf*4
        d4 = self.down4(d3)   # Out: 16x16, ngf*8
        d5 = self.down5(d4)   # Out: 8x8, ngf*8
        
        b = self.bottleneck(d5) # Out: 4x4, ngf*8
        
        # --- Percorso di Risalita con Ordine Corretto ---
        
        u1_out = self.up1(b)                 # 4x4 -> 8x8
        u1_attn = apply_attention(u1_out, self.attn1, text_features)
        u1_cat = torch.cat([u1_attn, d5], dim=1)
        
        u2_out = self.up2(u1_cat)            # 8x8 -> 16x16
        u2_attn = apply_attention(u2_out, self.attn2, text_features)
        u2_cat = torch.cat([u2_attn, d4], dim=1)
        
        u3_out = self.up3(u2_cat)            # 16x16 -> 32x32
        u3_cat = torch.cat([u3_out, d3], dim=1)
        
        u4_out = self.up4(u3_cat)            # 32x32 -> 64x64
        u4_cat = torch.cat([u4_out, d2], dim=1)
        
        u5_out = self.up5(u4_cat)            # 64x64 -> 128x128
        u5_cat = torch.cat([u5_out, d1], dim=1)

        # --- Layer Finale ---
        out = self.final_up(u5_cat) # 128x128 -> 256x256
        
        final_image = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return self.final_act(final_image), None 