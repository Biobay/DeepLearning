# src/models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Layer Chiave: AdaIN (Adaptive Instance Normalization) ---
class AdaIN(nn.Module):
    def __init__(self, style_dim, num_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels)
        self.style_transform = nn.Linear(style_dim, num_channels * 2)

    def forward(self, image_features, style_vector):
        normalized_features = self.norm(image_features)
        style = self.style_transform(style_vector).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        return gamma * normalized_features + beta

# --- Blocco di Sintesi (Upsampling + Convoluzioni + AdaIN) ---
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain1 = AdaIN(style_dim, out_channels)
        self.relu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adain2 = AdaIN(style_dim, out_channels)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, style_vector):
        x = self.up(x)
        x = self.conv1(x)
        x = self.adain1(x, style_vector)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.adain2(x, style_vector)
        x = self.relu2(x)
        return x

# --- Il Decoder/Generatore Completo ---
class StyleBasedGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        style_dim = cfg.STYLE_DIM
        
        # 1. Mapping Network (MLP) - Trasforma il testo in stile
        mapping_layers = [nn.Linear(cfg.LATENT_DIM, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(cfg.MAPPING_NETWORK_DEPTH - 1):
            mapping_layers.extend([nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)])
        self.mapping_network = nn.Sequential(*mapping_layers)

        # 2. Input Costante - Il punto di partenza che il modello impara
        self.initial_constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # 3. Blocco Iniziale e Blocchi di Sintesi
        self.initial_adain1 = AdaIN(style_dim, 512)
        self.initial_adain2 = AdaIN(style_dim, 512)
        self.initial_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Blocchi per upsampling da 4x4 a 256x256
        # 4x4 -> 8x8
        self.synth_block1 = SynthesisBlock(512, 512, style_dim) 
        # 8x8 -> 16x16
        self.synth_block2 = SynthesisBlock(512, 512, style_dim) 
        # 16x16 -> 32x32
        self.synth_block3 = SynthesisBlock(512, 256, style_dim) 
        # 32x32 -> 64x64
        self.synth_block4 = SynthesisBlock(256, 128, style_dim) 
        # 64x64 -> 128x128
        self.synth_block5 = SynthesisBlock(128, 64, style_dim)  
        # 128x128 -> 256x256
        self.synth_block6 = SynthesisBlock(64, 32, style_dim)   
        
        # 4. Layer di Output
        self.to_rgb = nn.Conv2d(32, cfg.OUTPUT_CHANNELS, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, text_features):
        batch_size = text_features.size(0)
        
        # 1. Aggrega e trasforma il testo in un vettore di stile 'w'
        latent_vector = text_features.mean(dim=1) # (B, LATENT_DIM)
        style_vector = self.mapping_network(latent_vector) # (B, STYLE_DIM)
        
        # 2. Inizia dal tensore costante e applica il primo stile
        x = self.initial_constant.repeat(batch_size, 1, 1, 1)
        x = self.initial_adain1(x, style_vector)
        x = self.initial_conv(x)
        x = self.initial_adain2(x, style_vector)
        
        # 3. Passa attraverso i blocchi di sintesi, modulando con lo stile a ogni passo
        x = self.synth_block1(x, style_vector)
        x = self.synth_block2(x, style_vector)
        x = self.synth_block3(x, style_vector)
        x = self.synth_block4(x, style_vector)
        x = self.synth_block5(x, style_vector)
        x = self.synth_block6(x, style_vector)

        # 4. Converte in immagine RGB
        x = self.to_rgb(x) # x è (B, 3, 256, 256)
        
        # 5. Ridimensiona all'output finale richiesto
        final_image = F.interpolate(x, size=self.cfg.IMAGE_OUTPUT_SIZE, mode='bilinear', align_corners=False)
        
        return self.final_act(final_image), None # Non ci sono più pesi di attention da restituire