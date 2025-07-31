# src/models/attention.py

import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    """
    Blocco di Cross-Attention che permette alle feature dell'immagine (query)
    di prestare attenzione alle feature del testo (context).
    """
    def __init__(self, query_dim, context_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=num_heads, 
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): Feature dell'immagine (B, H*W, C_query).
            context (torch.Tensor): Feature del testo (B, SeqLen, C_context).
        """
        attn_output, _ = self.attention(query=query, key=context, value=context)
        # Connessione residua e normalizzazione
        query = self.norm(query + attn_output)
        return query