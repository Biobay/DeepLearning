import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Encoder di testo basato su un modello Transformer pre-addestrato.
    Estrae le feature dal testo di input.
    """
    def __init__(self, config):
        """
        Args:
            config: Oggetto di configurazione con i parametri del modello.
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(config.ENCODER_MODEL_NAME)
        
        if not config.FINE_TUNE_ENCODER:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Passaggio forward.

        Args:
            input_ids (torch.Tensor): Tensor degli ID dei token.
                                      Dim: (batch_size, seq_len)
            attention_mask (torch.Tensor): Maschera di attenzione.
                                           Dim: (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - cls_embedding (torch.Tensor): Embedding del token [CLS] per il contesto.
                                                Dim: (batch_size, encoder_dim)
                - last_hidden_state (torch.Tensor): Hidden states per l'attention.
                                                    Dim: (batch_size, seq_len, encoder_dim)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # Estraiamo l'embedding del token [CLS] (il primo token della sequenza)
        # Questo vettore è un riassunto denso di significato dell'intera frase
        cls_embedding = last_hidden_state[:, 0]

        # Restituiamo sia il riassunto [CLS] per il contesto principale,
        # sia gli stati completi che serviranno all'attention per i dettagli.
        return cls_embedding, last_hidden_state
