import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    Meccanismo di attenzione per pesare gli output dell'encoder.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

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

class Decoder(nn.Module):
    """
    Decoder per generare l'immagine.
    """
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=768, dropout=0.5):
        super(Decoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, decoder_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.fc.out_features

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                              h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
