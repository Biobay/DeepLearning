import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Encoder per il testo basato su un modello Transformer pre-addestrato.

    Args:
        model_name (str): Nome del modello da Hugging Face (es. 'prajjwal1/bert-mini').
        fine_tune (bool): Se True, i pesi del modello verranno aggiornati durante l'addestramento.
    """
    def __init__(self, model_name='prajjwal1/bert-mini', fine_tune=True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        # Se fine_tune è False, congela i parametri del trasformer
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Passaggio forward dell'encoder.

        Args:
            input_ids (torch.Tensor): Tensor degli ID dei token di input.
            attention_mask (torch.Tensor): Maschera di attenzione per ignorare il padding.

        Returns:
            torch.Tensor: Output del modello transformer (last_hidden_state).
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Restituiamo l'ultimo stato nascosto
        return outputs.last_hidden_state
