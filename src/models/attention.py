import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, query, key_value):
        attn_output, attn_weights = self.attention(
            query=query, 
            key=key_value, 
            value=key_value, 
            need_weights=True
        )
        return attn_output, attn_weights