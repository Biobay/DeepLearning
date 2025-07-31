# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrossAttentionBlock

# I blocchi DownBlock e UpBlock rimangono gli stessi che abbiamo definito l'ultima volta
class DownBlock(nn.Module):
    # ... (codice identico)
    pass

class UpBlock(nn.Module):
    # ... (codice identico)
    pass


class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ngf = 64 # Numero di feature base

        # Percorso di Discesa
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, ngf, use_batch_norm=False)
        self.down2 = DownBlock(ngf, ngf * 2)
        self.down3 = DownBlock(ngf * 2, ngf * 4)
        self.down4 = DownBlock(ngf * 4, ngf * 8)
        
        # Bottleneck
        self.bottleneck_conv1 = DownBlock(ngf * 8, ngf * 8)
        self.bottleneck_conv2 = DownBlock(ngf * 8, ngf * 8)

        # Percorso di Risalita
        self.up1 = UpBlock(ngf * 8, ngf * 8)
        self.up2 = UpBlock(ngf * 8 * 2, ngf * 4)
        self.up3 = UpBlock(ngf * 4 * 2, ngf * 2)
        self.up4 = UpBlock(ngf * 2 * 2, ngf)
        
        # --- Blocchi di Cross-Attention a diverse risoluzioni ---
        self.attn1 = CrossAttentionBlock(query_dim=ngf * 8, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn2 = CrossAttentionBlock(query_dim=ngf * 4, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        self.attn3 = CrossAttentionBlock(query_dim=ngf * 2, context_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        
        # Layer Finale
        self.final_up = nn.ConvTranspose2d(ngf * 2, cfg.OUTPUT_CHANNELS, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        batch_size = text_features.size(0)
        # Si parte da rumore puro, la struttura verrà imposta dall'attention
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)

        # Helper per applicare l'attention
        def apply_attention(x, attn_block, context):
            B, C, H, W = x.shape
            img_features = x.view(B, C, H * W).permute(0, 2, 1)
            attn_features = attn_block(img_features, context)
            return attn_features.permute(0, 2, 1).view(B, C, H, W)

        # Percorso di discesa
        d1 = self.down1(x)  # 128
        d2 = self.down2(d1) # 64
        d3 = self.down3(d2) # 32
        d4 = self.down4(d3) # 16
        
        b = self.bottleneck_conv1(d4) # 8
        b = self.bottleneck_conv2(b) # 4

        # Percorso di risalita con attention
        u1 = self.up1(b, d4) # 8
        u1 = apply_attention(u1, self.attn1, text_features)

        u2 = self.up2(u1, d3) # 16
        u2 = apply_attention(u2, self.attn2, text_features)
        
        u3 = self.up3(u2, d2) # 32
        u3 = apply_attention(u3, self.attn3, text_features)
        
        u4 = self.up4(u3, d1) # 64
        # Possiamo aggiungere un blocco di attention anche qui se vogliamo
        
        # Output
        out = self.final_up(u4) # 128
        out = F.interpolate(out, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        
        return self.final_act(out), None