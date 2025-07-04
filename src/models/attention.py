import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    """
    Modulo di Cross-Attention basato su Multi-Head Attention di PyTorch.
    Permette a una sequenza di query (dal decoder) di "prestare attenzione"
    a una sequenza di key/value (dal testo codificato).
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True  # Si aspetta input come (batch, seq, feature)
        )

    def forward(self, query, key_value):
        """
        Args:
            query (torch.Tensor): Lo stato del decoder o una rappresentazione aggregata del testo.
                                  Dim: (batch_size, 1, embed_dim)
            key_value (torch.Tensor): L'output dell'encoder di testo.
                                      Dim: (batch_size, seq_len, embed_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: L'output dell'attenzione (vettore di contesto)
                                               e i pesi di attenzione.
        """
        # Per la cross-attention, la query, la chiave e il valore sono diversi.
        # Query: stato del decoder / rappresentazione aggregata
        # Key & Value: output dell'encoder di testo
        attn_output, attn_weights = self.attention(
            query=query, 
            key=key_value, 
            value=key_value, 
            need_weights=True
        )
        return attn_output, attn_weights
