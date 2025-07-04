import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    """
    Modulo di Cross-Attention basato su Multi-Head Attention di PyTorch.
    Permette a una sequenza di query (dal decoder) di "prestare attenzione"
    a una sequenza di key/value (dal testo codificato).
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True  # Si aspetta input come (batch, seq, feature)
        )

    def forward(self, query, key_value):
        """
        Args:
            query (torch.Tensor): Lo stato del decoder. Dim: (batch_size, 1, embed_dim)
            key_value (torch.Tensor): L'output dell'encoder di testo. Dim: (batch_size, seq_len, embed_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: L'output dell'attenzione e i pesi.
        """
        # Per la cross-attention, la query, la chiave e il valore sono diversi.
        # Query: stato del decoder
        # Key & Value: output dell'encoder di testo
        attn_output, attn_weights = self.attention(
            query=query, 
            key=key_value, 
            value=key_value, 
            need_weights=True
        )
        return attn_output, attn_weights

class ImageDecoder(nn.Module):
    """
    Decoder basato su CNN (Generatore) per creare un'immagine da un vettore di contesto.
    Utilizza strati di Convoluzione Trasposta per aumentare le dimensioni spaziali.

    Args:
        context_dim (int): Dimensione del vettore di contesto in input (dall'encoder).
        output_channels (int): Canali dell'immagine di output (3 per RGB).
        ngf (int): Numero di feature nel generatore, controlla la "larghezza" della rete.
    """
    def __init__(self, context_dim=256, output_channels=3, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            # Input: vettore di contesto (context_dim) -> proiettato e rimodellato
            # Proiettiamo il contesto in uno spazio ad alta dimensionalità
            nn.Linear(context_dim, ngf * 8 * 4 * 4),
            nn.ReLU(True),
            # Reshape a (ngf * 8) x 4 x 4
            nn.Unflatten(1, (ngf * 8, 4, 4)),

            # Strato 1: 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Strato 2: 8x8 -> 16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Strato 3: 16x16 -> 32x32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Strato 4: 32x32 -> 64x64
            nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Aggiungiamo strati per arrivare a 215x215. Non è una potenza di 2, quindi richiede un po' di aritmetica.
            # Strato 5: 64x64 -> 128x128
            nn.ConvTranspose2d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # Strato 6: 128x128 -> 215x215
            # Usiamo un kernel e padding calcolati per ottenere la dimensione esatta
            # output_size = (input_size - 1) * stride - 2 * padding + kernel_size
            # 215 = (128 - 1) * 1 - 2 * padding + kernel_size -> non funziona con stride 1
            # Proviamo con stride 2: 215 = (128-1)*2 - 2p + k -> 215 = 254 - 2p + k -> 2p-k = 39
            # Se k=5, 2p=44, p=22. Se k=7, 2p=46, p=23. Usiamo k=7, s=2, p=23
            # No, questo non funziona. Usiamo un approccio più semplice: upsample + conv
            nn.Upsample(size=215, mode='bilinear', align_corners=False),
            nn.Conv2d(ngf // 2, output_channels, kernel_size=3, stride=1, padding=1, bias=False),

            # Funzione di attivazione finale per mappare i pixel in [-1, 1]
            nn.Tanh()
        )

    def forward(self, context_vector):
        """
        Passaggio forward del generatore.

        Args:
            context_vector (torch.Tensor): Vettore di contesto dall'encoder.

        Returns:
            torch.Tensor: Immagine generata di dimensione (N, output_channels, 215, 215).
        """
        return self.main(context_vector)

class DecoderWithAttention(nn.Module):
    """
    Decoder completo che orchestra l'attenzione e la generazione dell'immagine.

    Questo modulo prende l'output di un encoder di testo, usa un meccanismo di attenzione
    per creare un vettore di contesto focalizzato, e poi usa un ImageDecoder
    per generare un'immagine da quel contesto.
    """
    def __init__(self, encoder_dim, decoder_dim, num_attention_heads, context_dim, output_channels=3, ngf=64):
        """
        Args:
            encoder_dim (int): Dimensione degli output dell'encoder di testo.
            decoder_dim (int): Dimensione dello stato nascosto del decoder (LSTM).
            num_attention_heads (int): Numero di teste per la Multi-Head Attention.
            context_dim (int): Dimensione del vettore di contesto atteso dall'ImageDecoder.
            output_channels (int): Canali dell'immagine di output (es. 3 per RGB).
            ngf (int): Feature map size per il generatore.
        """
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        # --- MODIFICA CHIAVE: Sostituzione dell'Attenzione ---
        # Meccanismo di Multi-Head Cross-Attention
        self.attention = MultiHeadCrossAttention(embed_dim=encoder_dim, num_heads=num_attention_heads)

        # LSTM per generare lo stato nascosto (query) per l'attenzione
        self.decode_step = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)

        # Layer per inizializzare lo stato nascosto (h) e di cella (c) dell'LSTM
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Layer per proiettare l'output dell'attenzione alla dimensione del contesto se necessario
        # In questo caso, l'output dell'attenzione ha già la dimensione corretta (encoder_dim == context_dim)
        # self.project_context = nn.Linear(encoder_dim, context_dim)

        # Generatore di immagini
        self.image_decoder = ImageDecoder(context_dim, output_channels, ngf)

    def init_hidden_state(self, encoder_out):
        """
        Inizializza gli stati h e c dell'LSTM basandosi sull'output dell'encoder.
        Usiamo l'output medio dell'encoder come rappresentazione iniziale della frase.
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out):
        """
        Passaggio forward.

        Args:
            encoder_out (torch.Tensor): Output dell'encoder di testo.
                                       Dim: (batch_size, seq_len, encoder_dim)

        Returns:
            torch.Tensor: Immagine generata. Dim: (batch_size, C, H, W)
            torch.Tensor: Pesi di attenzione (alpha). Dim: (batch_size, 1, seq_len)
        """
        # 1. Inizializza lo stato nascosto dell'LSTM
        h, c = self.init_hidden_state(encoder_out)

        # 2. Esegui un singolo passo dell'LSTM per ottenere la query per l'attenzione
        mean_encoder_out = encoder_out.mean(dim=1)
        h, c = self.decode_step(mean_encoder_out, (h, c))

        # 3. Calcola il vettore di contesto usando la Multi-Head Cross-Attention
        # Lo stato 'h' del decoder (la nostra query) deve avere una dimensione di sequenza.
        query = h.unsqueeze(1)  # Shape: (batch_size, 1, decoder_dim)
        
        # L'output dell'encoder è la nostra key e value
        context_vector, alpha = self.attention(query=query, key_value=encoder_out)
        
        # L'output dell'attenzione ha shape (batch_size, 1, embed_dim), lo riduciamo
        context_vector = context_vector.squeeze(1) # Shape: (batch_size, embed_dim)

        # 4. Genera l'immagine dal vettore di contesto
        generated_image = self.image_decoder(context_vector)

        return generated_image, alpha
