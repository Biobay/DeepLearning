import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        return self.model(x)

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)
        
        # Input Layer
        self.input_conv = nn.Conv2d(cfg.DECODER_IN_CHANNELS, 64, kernel_size=3, padding=1)

        # Downsampling
        self.d1 = DownBlock(64, 128)
        self.d2 = DownBlock(128, 256)
        self.d3 = DownBlock(256, 512)
        self.d4 = DownBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU()) # 8x8

        # Upsampling
        self.u1 = UpBlock(512 * 2, 512)
        self.u2 = UpBlock(512 * 2, 256)
        self.u3 = UpBlock(256 * 2, 128)
        self.u4 = UpBlock(128 * 2, 64)
        
        # Output Layer
        self.output_conv = nn.ConvTranspose2d(64 * 2, cfg.OUTPUT_CHANNELS, 4, 2, 1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        context_vector_query = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector_query, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        batch_size = text_features.size(0)
        context_map = conditioned_vector.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE)
        noise_map = torch.randn(batch_size, self.cfg.NUM_NOISE_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        x = torch.cat([context_map, noise_map], dim=1)
        
        x0 = self.input_conv(x) # 256
        d1 = self.d1(x0) # 128
        d2 = self.d2(d1) # 64
        d3 = self.d3(d2) # 32
        d4 = self.d4(d3) # 16
        
        b = self.bottleneck(d4) # 8
        
        u1 = self.u1(b, d4) # 16
        u2 = self.u2(u1, d3) # 32
        u3 = self.u3(u2, d2) # 64
        u4 = self.u4(u3, d1) # 128
        
        out = self.output_conv(torch.cat((u4, x0), 1)) # 256
        
        final_image = F.interpolate(out, size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), mode='bilinear', align_corners=False)
        return self.final_act(final_image), attn_weights