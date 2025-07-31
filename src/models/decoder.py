# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# --- Blocchi Costitutivi (Semplificati e Robusti) ---

class UNetDownBlock(nn.Module):
    """Blocco di discesa: Conv -> Norma -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)) # Usiamo InstanceNorm, spesso più stabile
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUpBlock(nn.Module):
    """Blocco di risalita: ConvTranspose -> Norma -> ReLU -> Dropout"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# --- Il Decoder U-Net con Architettura Corretta ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Numero di feature base

        # --- Percorso di Discesa (Encoder) ---
        self.down1 = UNetDownBlock(cfg.OUTPUT_CHANNELS, ngf, normalize=False)
        self.down2 = UNetDownBlock(ngf, ngf * 2)
        self.down3 = UNetDownBlock(ngf * 2, ngf * 4)
        self.down4 = UNetDownBlock(ngf * 4, ngf * 8)
        self.down5 = UNetDownBlock(ngf * 8, ngf * 8)
        self.down6 = UNetDownBlock(ngf * 8, ngf * 8)
        self.down7 = UNetDownBlock(ngf * 8, ngf * 8)
        self.bottleneck = UNetDownBlock(ngf * 8, ngf * 8, normalize=False)

        # --- Percorso di Risalita (Decoder) ---
        # I canali di input tengono conto della concatenazione con la skip connection
        self.up1 = UNetUpBlock(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUpBlock(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up3 = UNetUpBlock(ngf * 8 * 2, ngf * 8, dropout=0.5)
        self.up4 = UNetUpBlock(ngf * 8 * 2, ngf * 8)
        self.up5 = UNetUpBlock(ngf * 8 * 2, ngf * 4)
        self.up6 = UNetUpBlock(ngf * 4 * 2, ngf * 2)
        self.up7 = UNetUpBlock(ngf * 2 * 2, ngf)

        # --- Layer Finale ---
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, cfg.OUTPUT_CHANNELS, 3, padding=1),
            nn.Tanh()
        )
        
        # --- Blocchi di Attenzione ---
        # Applichiamo l'attention solo dove serve di più (risoluzioni intermedie)
        self.attn1 = CrossAttentionBlock(query_dim=ngf * 8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf * 4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)


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
        d1 = self.down1(x)    # 128
        d2 = self.down2(d1)   # 64
        d3 = self.down3(d2)   # 32
        d4 = self.down4(d3)   # 16
        d5 = self.down5(d4)   # 8
        d6 = self.down6(d5)   # 4
        d7 = self.down7(d6)   # 2
        b = self.bottleneck(d7) # 1

        # --- Percorso di Risalita ---
        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        
        # Applica attention
        u3_attn = apply_attention(u3, self.attn1, text_features)
        
        u4 = self.up4(u3_attn, d4)
        u5 = self.up5(u4, d3)
        
        # Applica attention
        u5_attn = apply_attention(u5, self.attn2, text_features)
        
        u6 = self.up6(u5_attn, d2)
        u7 = self.up7(u6, d1)

        # --- Output ---
        out = self.final_layer(u7)
        
        final_image = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        return final_image, None