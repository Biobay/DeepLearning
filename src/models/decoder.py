# src/models/decoder.py

import torch
import torch.nn as nn
from .attention import MultiHeadCrossAttention

# --- Blocchi Costitutivi della U-Net ---

class DownBlock(nn.Module):
    """Blocco di discesa: Conv -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Blocco di risalita: Upsample -> Conv -> BatchNorm -> ReLU. Usa skip connections."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # L'input channel è doppio per via della concatenazione della skip connection
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1) # Concatena lungo l'asse dei canali
        return self.block(x)

# --- Il Decoder U-Net ---

class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = cfg.UNET_CHANNELS
        
        # Proiezione del testo per iniziare
        self.text_projection = nn.Linear(cfg.CONTEXT_DIM, channels[0] * 8 * 8)
        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # Percorso di Discesa (Encoder della U-Net)
        self.down1 = nn.Conv2d(3, channels[0], kernel_size=4, stride=2, padding=1) # Da 215 -> ~107
        self.down_blocks = nn.ModuleList([
            DownBlock(channels[i], channels[i+1]) for i in range(len(channels)-1)
        ])

        # Bottleneck - il punto più profondo
        self.bottleneck = DownBlock(channels[-1], channels[-1])
        
        # Percorso di Risalita (Decoder della U-Net)
        self.up_blocks = nn.ModuleList([
            UpBlock(channels[i], channels[i-1]) for i in reversed(range(1, len(channels)))
        ])
        
        # Layer finale
        self.final_up = UpBlock(channels[0], channels[0])
        self.final_conv = nn.Conv2d(channels[0] * 2, cfg.OUTPUT_CHANNELS, kernel_size=3, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # 1. Crea il vettore di contesto dal testo
        context_vector = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        # 2. Crea un rumore iniziale (o un'immagine nera)
        batch_size = text_features.size(0)
        noise_input = torch.randn(batch_size, 3, 215, 215, device=text_features.device)
        
        # 3. Percorso di discesa con skip connections
        skips = []
        x = self.down1(noise_input)
        skips.append(x)
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
            
        # 4. Bottleneck + Iniezione del Testo
        x = self.bottleneck(x)
        text_proj = self.text_projection(conditioned_vector)
        # Rimodella il testo proiettato per aggiungerlo come canale
        text_proj = text_proj.view(batch_size, -1, 8, 8) 
        # NOTA: Qui si potrebbe fare di meglio, ma per iniziare va bene
        # x = torch.cat([x, text_proj], dim=1) --> richiede che le dimensioni spaziali corrispondano

        # 5. Percorso di risalita
        skips = list(reversed(skips))
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[i])
            
        x = self.final_up(x, skips[-1])
        
        # 6. Immagine finale
        x = self.final_conv(x)
        
        # Assicuriamoci che l'output sia 215x215
        x = nn.functional.interpolate(x, size=215, mode='bilinear', align_corners=False)
        
        return self.final_act(x), attn_weights