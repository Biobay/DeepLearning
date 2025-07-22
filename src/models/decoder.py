import torch
import torch.nn as nn
from src.models.attention import MultiHeadCrossAttention

class GeneratorS1(nn.Module):
    """
    Generatore dello Stage-I, basato su un'architettura CNN (DCGAN-like).
    Crea un'immagine a bassa risoluzione (64x64) partendo da:
    1. Un embedding testuale (processato tramite attention).
    2. Un vettore di rumore latente z.
    """
    def __init__(self, config):
        super().__init__()
        
        self.text_embed_dim = config.TEXT_EMBEDDING_DIM
        self.z_dim = config.Z_DIM
        self.base_channels = config.DECODER_BASE_CHANNELS

        # La proiezione iniziale ora accetta sia l'embedding del testo che il rumore z
        self.init_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim + self.z_dim, self.base_channels * 8 * 4 * 4),
            nn.BatchNorm1d(self.base_channels * 8 * 4 * 4), # Aggiunta per stabilità
            nn.ReLU(True)
        )

        # Modulo di Attention per condizionare il testo
        self.attention = MultiHeadCrossAttention(
            embed_dim=self.text_embed_dim, 
            num_heads=config.NUM_HEADS
        )
        
        # Rete generativa CNN, ora più "larga" grazie a base_channels
        self.main = nn.Sequential(
            # Input: (base_channels * 8) x 4 x 4
            nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 4),
            nn.ReLU(True),
            # State: (base_channels * 4) x 8 x 8
            nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels * 2),
            nn.ReLU(True),
            # State: (base_channels * 2) x 16 x 16
            nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(True),
            # State: (base_channels) x 32 x 32
            nn.ConvTranspose2d(self.base_channels, 3, 4, 2, 1, bias=False),
            # Output: 3 (RGB) x 64 x 64
            nn.Tanh() # Normalizza l'output tra -1 e 1
        )

    def forward(self, cls_embedding, hidden_states, z_noise):
        """
        Args:
            cls_embedding (torch.Tensor): Vettore [CLS] di BERT. Dim: (batch, embed_dim)
            hidden_states (torch.Tensor): Output dell'ultimo layer di BERT. Dim: (batch, seq_len, embed_dim)
            z_noise (torch.Tensor): Vettore di rumore. Dim: (batch, z_dim)

        Returns:
            torch.Tensor: Immagine generata. Dim: (batch, 3, 64, 64)
            torch.Tensor: Pesi dell'attenzione. Dim: (batch, num_heads, 1, seq_len)
        """
        batch_size = hidden_states.size(0)
        
        # 1. Applica l'attenzione per ottenere un vettore di contesto dal testo
        #    La query è il [CLS] embedding, Key e Value sono gli hidden states
        attn_output, attn_weights = self.attention(
            query=cls_embedding.unsqueeze(1), # Aggiunge la dimensione per la sequenza (len=1)
            key_value=hidden_states
        )
        conditioned_vector = attn_output.squeeze(1) # Rimuove la dimensione della sequenza

        # 2. Concatena il contesto testuale e il rumore
        combined_input = torch.cat([conditioned_vector, z_noise], dim=1)
        
        # 3. Proietta l'input combinato nella dimensione iniziale della CNN
        x = self.init_projection(combined_input)
        x = x.view(batch_size, -1, 4, 4) # Reshape a (batch, base_channels*8, 4, 4)
        
        # 4. Passa attraverso la rete generativa per creare l'immagine
        generated_image = self.main(x)
        
        return generated_image, attn_weights


class GeneratorS2(nn.Module):
    """Generatore Stage-II: raffina immagini 64x64 a 256x256"""
    
    def __init__(self, config):
        super(GeneratorS2, self).__init__()
        self.config = config
        self.text_dim = config.TEXT_EMBEDDING_DIM
        self.base_channels = config.DECODER_BASE_CHANNELS
        
        # Encoder per le immagini 64x64 del Stage-I
        self.img_encoder = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(3, self.base_channels//2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32 x 32
            nn.Conv2d(self.base_channels//2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16
            nn.Conv2d(self.base_channels, self.base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8
            nn.Conv2d(self.base_channels*2, self.base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
        )
        
        # Proiezione del testo condizionato
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_dim, self.base_channels*4),
            nn.BatchNorm1d(self.base_channels*4),
            nn.ReLU(True)
        )
        
        # Decoder principale per upsampling a 256x256
        self.main = nn.Sequential(
            # Input: (256 + 256) x 4 x 4 = 512 x 4 x 4
            nn.ConvTranspose2d(self.base_channels*8, self.base_channels*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels*4),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(self.base_channels*4, self.base_channels*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels*2),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(self.base_channels*2, self.base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(self.base_channels, self.base_channels//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels//2),
            nn.ReLU(True),
            # 32 x 64 x 64
            nn.ConvTranspose2d(self.base_channels//2, self.base_channels//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_channels//4),
            nn.ReLU(True),
            # 16 x 128 x 128
            nn.ConvTranspose2d(self.base_channels//4, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 256 x 256
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializza i pesi della rete."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, stage1_img, text_embedding, stage1_mu):
        """
        Forward pass del GeneratorS2.
        
        Args:
            stage1_img: Immagini 64x64 generate dal Stage-I [B, 3, 64, 64]
            text_embedding: Text embedding [B, text_dim]
            stage1_mu: Parametri mu del Stage-I (per condizionamento)
            
        Returns:
            stage2_img: Immagini raffinate 256x256 [B, 3, 256, 256]
            mu: Parametri di condizionamento per Stage-II
        """
        batch_size = stage1_img.size(0)
        
        # 1. Codifica l'immagine Stage-I
        img_features = self.img_encoder(stage1_img)  # [B, 256, 4, 4]
        
        # 2. Proietta il text embedding
        text_features = self.text_projection(text_embedding)  # [B, 256]
        text_features = text_features.view(batch_size, -1, 1, 1)  # [B, 256, 1, 1]
        text_features = text_features.expand(-1, -1, 4, 4)  # [B, 256, 4, 4]
        
        # 3. Concatena features immagine e testo
        combined_features = torch.cat([img_features, text_features], dim=1)  # [B, 512, 4, 4]
        
        # 4. Genera immagine 256x256
        stage2_img = self.main(combined_features)
        
        # 5. Calcola mu per Stage-II (per consistenza con l'architettura)
        mu = stage1_mu  # Riusa i parametri del Stage-I
        
        return stage2_img, mu
