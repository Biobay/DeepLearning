import torch.nn as nn

class PikaPikaGen(nn.Module):
    """
    Modello completo che combina l'Encoder e il Decoder.
    """
    def __init__(self, encoder, decoder):
        super(PikaPikaGen, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(encoder_out, captions, caption_lengths)
        return scores, caps_sorted, decode_lengths, alphas, sort_ind
