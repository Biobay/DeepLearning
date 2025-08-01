# src/models/decoder.py
import torch, torch.nn as nn
from .attention import MultiHeadCrossAttention

# --- GENERATORE STAGE-I (INVARIATO) ---
class GeneratorS1(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... (Il codice di GeneratorS1 rimane esattamente lo stesso di prima)
        self.text_embed_dim = config.TEXT_EMBEDDING_DIM
        self.z_dim = config.Z_DIM
        self.base_channels = config.DECODER_BASE_CHANNELS
        self.init_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim + self.z_dim, self.base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(self.base_channels * 8 * 4 * 4), nn.ReLU(True))
        self.attention = MultiHeadCrossAttention(embed_dim=self.text_embed_dim, num_heads=config.NUM_HEADS)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels), nn.ReLU(True),
            nn.ConvTranspose2d(self.base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh())
    def forward(self, cls_embedding, hidden_states, z_noise):
        # ... (Il forward di GeneratorS1 rimane esattamente lo stesso di prima)
        attn_output, attn_weights = self.attention(query=cls_embedding.unsqueeze(1), key_value=hidden_states)
        conditioned_vector = attn_output.squeeze(1)
        combined_input = torch.cat([conditioned_vector, z_noise], dim=1)
        x = self.init_projection(combined_input)
        x = x.view(x.size(0), -1, 4, 4)
        return self.main(x), attn_weights

# --- BLOCCHI COSTITUTIVI PER LA U-NET ---
class UNetDown(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm: layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)]
        if dropout: layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], 1)

# =============================================================================
# ## GENERATORE STAGE-II (NUOVA ARCHITETTURA U-NET) ##
# =============================================================================
class GeneratorS2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ngf = config.DECODER_BASE_CHANNELS # Usiamo la stessa base di canali

        # U-Net Encoder (percorso di discesa)
        self.down1 = UNetDown(3, ngf, norm=False) # 64 -> 32
        self.down2 = UNetDown(ngf, ngf * 2)        # 32 -> 16
        self.down3 = UNetDown(ngf * 2, ngf * 4)      # 16 -> 8
        self.down4 = UNetDown(ngf * 4, ngf * 8)      # 8 -> 4
        
        self.bottleneck = UNetDown(ngf * 8, ngf * 8, normalize=False) # 4 -> 2

        # Proiezione del testo nel bottleneck
        self.text_projection = nn.Linear(config.TEXT_EMBEDDING_DIM, ngf * 8 * 2 * 2)

        # U-Net Decoder (percorso di risalita)
        self.up1 = UNetUp(ngf * 8, ngf * 8, dropout=0.5)
        self.up2 = UNetUp(ngf * 8 * 2, ngf * 4)
        self.up3 = UNetUp(ngf * 4 * 2, ngf * 2)
        self.up4 = UNetUp(ngf * 2 * 2, ngf)
        
        # Layer finale
        self.final_up = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, stage1_img, text_embedding, _): # Ignoriamo 'stage1_mu'
        # Percorso di discesa (encoding dell'immagine 64x64)
        d1 = self.down1(stage1_img)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Bottleneck e iniezione del testo
        b = self.bottleneck(d4)
        text_features = self.text_projection(text_embedding).view(-1, self.config.DECODER_BASE_CHANNELS * 8, 2, 2)
        b = b + text_features # Fusione additiva
        
        # Percorso di risalita con skip connections
        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        
        out_128 = self.final_up(u4)
        
        # Upsampling finale alla dimensione richiesta
        final_img = nn.functional.interpolate(
            out_128, 
            size=(self.config.STAGE2_IMAGE_SIZE, self.config.STAGE2_IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        )
        
        # Ritorna None per mu, per coerenza con la firma originale
        return final_img, None