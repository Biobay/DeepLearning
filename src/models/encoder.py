import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-mini', fine_tune=True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state