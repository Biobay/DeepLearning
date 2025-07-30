# src/models/attention.py

import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    """
    Versione SEMPLICE di Cross-Attention.
    Usa una rappresentazione aggregata del testo (es. la media) come query
    per "guardare" l'intera sequenza di parole.
    Utile per creare un singolo vettore di contesto.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

    def forward(self, query, key_value):
        """
        Args:
            query (torch.Tensor): Rappresentazione aggregata (es. media). Dim: (B, 1, C)
            key_value (torch.Tensor): Sequenza completa delle parole. Dim: (B, SeqLen, C)
        """
        attn_output, attn_weights = self.attention(
            query=query, 
            key=key_value, 
            value=key_value, 
            need_weights=True
        )
        return attn_output, attn_weights


class CrossAttentionBlock(nn.Module):
    """
    Blocco di Cross-Attention AVANZATO, ispirato ai Transformer.
    Permette alle feature di un'immagine (query) di "prestare attenzione"
    alle feature del testo (context/key/value) per arricchirsi semanticamente.
    Questa è la versione usata nei modelli text-to-image moderni.
    """
    def __init__(self, query_dim, context_dim, num_heads, inner_dim=None):
        """
        Args:
            query_dim (int): Dimensione delle feature dell'immagine (i canali).
            context_dim (int): Dimensione delle feature del testo (es. 256 da BERT).
            num_heads (int): Numero di teste di attenzione.
            inner_dim (int, optional): Dimensione interna per le proiezioni. 
                                     Se None, usa query_dim.
        """
        super().__init__()
        inner_dim = inner_dim if inner_dim is not None else query_dim
        
        # Multi-Head Attention layer che gestisce dimensioni diverse per query e key/value
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=num_heads, 
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True
        )
        
        # Layer di normalizzazione e feed-forward, come in un blocco Transformer
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        
        # Feed-Forward Network per elaborare ulteriormente le feature
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, inner_dim * 4),
            nn.GELU(), # GELU è un'attivazione moderna ed efficace
            nn.Linear(inner_dim * 4, query_dim)
        )

    def forward(self, query, context):
        """
        Args:
            query (torch.Tensor): Feature dell'immagine. 
                                  Devono essere nel formato (Batch, Sequence, Features).
                                  Per un'immagine, questo significa (B, H*W, C).
            context (torch.Tensor): Feature del testo (output di BERT).
                                    Dim: (B, SeqLen, context_dim).
        Returns:
            torch.Tensor: Feature dell'immagine arricchite dal contesto testuale.
        """
        # La query originale viene usata per la connessione residua (skip connection)
        residual = query
        
        # Applica l'attenzione: l'immagine (query) "guarda" al testo (context)
        attn_output, _ = self.attention(query=query, key=context, value=context)
        
        # Prima connessione residua e normalizzazione
        query = self.norm1(residual + attn_output)
        
        # La query aggiornata viene usata per la seconda connessione residua
        residual = query
        
        # Passa attraverso la rete Feed-Forward
        ffn_output = self.ffn(query)
        
        # Seconda connessione residua e normalizzazione
        query = self.norm2(residual + ffn_output)
        
        return query