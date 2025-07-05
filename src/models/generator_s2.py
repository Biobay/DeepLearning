import torch
import torch.nn as nn

# --- Blocco Residuale ---
def conv3x3(in_planes, out_planes, stride=1):
    """Convoluzione 3x3 con padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

# --- Blocco di Upsampling ---
def upsample_block(in_channels, out_channels):
    """Blocco per aumentare la risoluzione spaziale."""
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )
    return block

# --- Generatore di Fase II ---
class GeneratorS2(nn.Module):
    def __init__(self, config):
        super(GeneratorS2, self).__init__()
        self.config = config
        self.text_embedding_dim = config.TEXT_EMBEDDING_DIM
        self.s1_img_size = config.STAGE1_IMAGE_SIZE # 64
        self.s2_img_size = config.STAGE2_IMAGE_SIZE # 256
        
        # Aumenta la dimensionalità dell'embedding testuale per il condizionamento
        self.ca_net = nn.Sequential(
            nn.Linear(self.text_embedding_dim, self.text_embedding_dim * 2),
            nn.ReLU()
        )

        # Fase di processamento iniziale dell'immagine a bassa risoluzione
        self.initial_conv = nn.Sequential(
            conv3x3(3, config.DECODER_BASE_CHANNELS * 4), # 3 canali RGB -> 512
            nn.ReLU(True)
        )

        # Modulo di condizionamento: combina immagine e testo
        self.concat_conv = nn.Sequential(
            conv3x3(config.DECODER_BASE_CHANNELS * 4 + self.text_embedding_dim * 2, config.DECODER_BASE_CHANNELS * 4),
            nn.ReLU(True)
        )

        # Blocchi residuali per il refinement dei dettagli
        self.residual_blocks = nn.Sequential(
            ResidualBlock(config.DECODER_BASE_CHANNELS * 4),
            ResidualBlock(config.DECODER_BASE_CHANNELS * 4),
            ResidualBlock(config.DECODER_BASE_CHANNELS * 4),
            ResidualBlock(config.DECODER_BASE_CHANNELS * 4)
        )

        # Blocchi di upsampling per aumentare la risoluzione
        self.upsample1 = upsample_block(config.DECODER_BASE_CHANNELS * 4, config.DECODER_BASE_CHANNELS * 2) # 64 -> 128
        self.upsample2 = upsample_block(config.DECODER_BASE_CHANNELS * 2, config.DECODER_BASE_CHANNELS)     # 128 -> 256
        
        # Convoluzione finale per generare l'immagine RGB
        self.final_conv = nn.Sequential(
            conv3x3(config.DECODER_BASE_CHANNELS, 3),
            nn.Tanh()
        )

    def forward(self, low_res_image, text_embedding):
        # 1. Prepara l'embedding testuale
        ca_embedding = self.ca_net(text_embedding)
        ca_embedding_spatial = ca_embedding.unsqueeze(-1).unsqueeze(-1) # Aggiunge dimensioni spaziali
        ca_embedding_spatial = ca_embedding_spatial.repeat(1, 1, self.s1_img_size, self.s1_img_size) # Ripete per tutta la griglia

        # 2. Processa l'immagine a bassa risoluzione
        x = self.initial_conv(low_res_image)

        # 3. Concatena le feature dell'immagine e del testo
        x = torch.cat([x, ca_embedding_spatial], dim=1)
        x = self.concat_conv(x)

        # 4. Applica i blocchi residuali per il refinement
        x = self.residual_blocks(x)

        # 5. Esegui l'upsampling
        x = self.upsample1(x)
        x = self.upsample2(x)

        # 6. Genera l'immagine finale
        high_res_image = self.final_conv(x)

        return high_res_image
