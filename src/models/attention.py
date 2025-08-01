# src/models/attention.py

import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    Blocco di Cross-Attention AVANZATO, ispirato ai Transformer.
    Permette alle feature di un'immagine (query) di "prestare attenzione"
    alle feature del testo (context/key/value) per arricchirsi semanticamente.
    """
    def __init__(self, query_dim, context_dim, num_heads, inner_dim=None):
        super().__init__()
        inner_dim = inner_dim if inner_dim is not None else query_dim
        
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=num_heads, 
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, inner_dim * 4),
            nn.GELU(),
            nn.Linear(inner_dim * 4, query_dim)
        )

    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): Feature dell'immagine nel formato (B, Sequence, Features).
            context (torch.Tensor): Feature del testo (output di BERT). Dim: (B, SeqLen, C).
        """
        residual = query
        attn_output, _ = self.attention(query=query, key=context, value=context)
        query = self.norm1(residual + attn_output)
        
        residual = query
        ffn_output = self.ffn(query)
        query = self.norm2(residual + ffn_output)
        
        return query