import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Encoder di testo basato su un modello Transformer pre-addestrato.
    Estrae le feature dal testo di input.
    """
    def __init__(self, model_name='prajjwal1/bert-mini', fine_tune=True):
        """
        Args:
            model_name (str): Nome del modello da Hugging Face.
            fine_tune (bool): Se fare il fine-tuning dei pesi del modello.
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        
        if not fine_tune:
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
            - cls_embedding (torch.Tensor): Embedding del token [CLS] (riassunto globale).
                                            Dim: (batch_size, encoder_dim)
            - last_hidden_state (torch.Tensor): Hidden states dall'ultimo layer (contesto per parola).
                                                Dim: (batch_size, seq_len, encoder_dim)
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Estraiamo l'output completo dell'ultimo layer
        last_hidden_state = outputs.last_hidden_state
        
        # L'embedding [CLS] è l'output del primo token della sequenza
        cls_embedding = last_hidden_state[:, 0]
        
        # Restituiamo sia l'embedding [CLS] (per il contesto globale) 
        # che gli hidden states completi (per l'attention)
        return cls_embedding, last_hidden_state
