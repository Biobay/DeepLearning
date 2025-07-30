# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadCrossAttention

# --- Blocchi Costitutivi della U-Net ---

class DownBlock(nn.Module):
    """Blocco di discesa: Conv -> BatchNorm -> LeakyReLU."""
    def __init__(self, in_channels, out_channels, use_batch_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    """Blocco di risalita: ConvTranspose -> BatchNorm -> ReLU. Usa skip connections."""
    def __init__(self, in_channels, out_channels, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        layers = [
            # L'input channel è doppio per via della concatenazione della skip connection
            nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate)) # Dropout standard dopo l'attivazione
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        return self.block(x)


# --- Il Decoder U-Net ---

class UNetDecoder(nn.Module):
    """
    Decoder U-Net condizionato dal testo. Lavora internamente a una dimensione
    potenza di 2 e restituisce un'immagine alla dimensione finale richiesta.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        channels = cfg.UNET_CHANNELS # es. (64, 128, 256, 512)

        # Meccanismo di attenzione per il condizionamento testuale
        self.attention = MultiHeadCrossAttention(embed_dim=cfg.CONTEXT_DIM, num_heads=cfg.NUM_HEADS)

        # Percorso di Discesa (Encoder della U-Net)
        # Il primo blocco non ha BatchNorm
        self.down1 = DownBlock(cfg.OUTPUT_CHANNELS, channels[0], use_batch_norm=False)
        
        down_blocks_layers = []
        for i in range(len(channels) - 1):
            down_blocks_layers.append(DownBlock(channels[i], channels[i+1]))
        self.down_blocks = nn.ModuleList(down_blocks_layers)
        
        # Bottleneck: il punto più profondo, dove inietteremo il testo
        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Proiezione del testo per farlo corrispondere ai canali del bottleneck
        self.text_projection = nn.Linear(cfg.CONTEXT_DIM, bottleneck_channels)
        
        # Percorso di Risalita (Decoder della U-Net)
        up_blocks_layers = []
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels) - 1):
            # Aggiungiamo dropout solo nei layer più profondi per regolarizzare
            use_dropout = i < 3 
            up_blocks_layers.append(
                UpBlock(reversed_channels[i], reversed_channels[i+1], use_dropout=use_dropout, dropout_rate=cfg.DROPOUT_RATE)
            )
        self.up_blocks = nn.ModuleList(up_blocks_layers)
        
        # Layer finale
        self.final_conv = nn.ConvTranspose2d(channels[0] * 2, cfg.OUTPUT_CHANNELS, kernel_size=4, stride=2, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        # 1. Crea il vettore di contesto dal testo
        context_vector = text_features.mean(dim=1).unsqueeze(1)
        attn_output, attn_weights = self.attention(query=context_vector, key_value=text_features)
        conditioned_vector = attn_output.squeeze(1)

        # 2. Inizia con un'immagine di rumore alla dimensione interna del modello
        batch_size = text_features.size(0)
        x = torch.randn(batch_size, self.cfg.OUTPUT_CHANNELS, self.cfg.MODEL_INTERNAL_SIZE, self.cfg.MODEL_INTERNAL_SIZE, device=text_features.device)
        
        # 3. Percorso di discesa, salvando le skip connections
        skips = []
        x = self.down1(x)
        skips.append(x)
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        
        # 4. Bottleneck
        x = self.bottleneck(x)
        
        # 5. Iniezione del Testo (Modulazione)
        text_proj = self.text_projection(conditioned_vector)
        # Aggiunge due dimensioni (altezza, larghezza) al tensore del testo
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1)
        # "Modula" le feature nel bottleneck con l'informazione del testo
        x = x * text_proj

        # 6. Percorso di risalita, usando le skip connections
        skips = list(reversed(skips))
        for i, block in enumerate(self.up_blocks):
            x = block(x, skips[i])
        
        # 7. Layer finale per produrre l'immagine
        x = self.final_conv(x)
        
        # 8. Output alla dimensione richiesta dalla traccia (215x215)
        final_image = F.interpolate(
            x, 
            size=(self.cfg.IMAGE_OUTPUT_SIZE, self.cfg.IMAGE_OUTPUT_SIZE), 
            mode='bilinear', 
            align_corners=False
        )
        
        return self.final_act(final_image), attn_weights