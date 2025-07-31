# src/models/attention.py
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=num_heads, kdim=context_dim,
            vdim=context_dim, batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)
    
    def forward(self, query, context):
        attn_output, _ = self.attention(query, context, context)
        return self.norm(query + attn_output)