# src/models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

class UNetDownBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)
        return self.model(x)

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64

        # Discesa
        self.d1 = UNetDownBlock(cfg.OUTPUT_CHANNELS, ngf, norm=False) # 128
        self.d2 = UNetDownBlock(ngf, ngf * 2)             # 64
        self.d3 = UNetDownBlock(ngf * 2, ngf * 4)           # 32
        self.d4 = UNetDownBlock(ngf * 4, ngf * 8)           # 16
        
        # Bottleneck
        self.bottleneck = UNetDownBlock(ngf * 8, ngf * 8) # 8

        # Risalita
        self.u1 = UNetUpBlock(ngf * 8, ngf * 8)
        self.u2 = UNetUpBlock(ngf * 8 * 2, ngf * 4)
        self.u3 = UNetUpBlock(ngf * 4 * 2, ngf * 2)
        self.u4 = UNetUpBlock(ngf * 2 * 2, ngf)
        
        # Attention
        self.attn1 = CrossAttentionBlock(ngf * 8, cfg.CONTEXT_DIM, cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(ngf * 4, cfg.CONTEXT_DIM, cfg.NUM_HEADS)

        # Output
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, 4, 2, 1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        B, C, H, W = text_features.shape[0], self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE
        x = torch.randn(B, C, H, W, device=text_features.device)
        
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            img_feat = x.view(B, C, H * W).permute(0, 2, 1)
            attn_feat = attn_block(img_feat, context)
            return attn_feat.permute(0, 2, 1).view(B, C, H, W)
        
        # Discesa
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        
        b = self.bottleneck(d4)

        # Risalita con attention
        u1 = self.u1(b, d4)
        u1 = apply_attention(u1, self.attn1, text_features)
        
        u2 = self.u2(u1, d3)
        u2 = apply_attention(u2, self.attn2, text_features)
        
        u3 = self.u3(u2, d2)
        u4 = self.u4(u3, d1)

        out = self.final_up(u4)
        final_image = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        
        return self.final_act(final_image), None