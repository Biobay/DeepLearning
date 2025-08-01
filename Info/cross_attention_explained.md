# Cross-Attention nella Generazione di Immagini da Testo

## Introduzione

La Cross-Attention è un meccanismo fondamentale nei modelli di text-to-image come il nostro StackGAN per Pokémon. Questo documento spiega come funziona, perché è importante e come la abbiamo implementata.

## Cos'è la Cross-Attention?

La Cross-Attention è un meccanismo che consente a un modello di "focalizzare l'attenzione" su parti rilevanti di una sequenza di input mentre elabora un'altra sequenza. Nel nostro caso:

- **Query**: Le feature dell'immagine che stiamo generando
- **Key/Value**: Le feature del testo di input (la descrizione del Pokémon)

## Come Funziona la Cross-Attention

1. **Multi-Head Attention**: Dividiamo l'attenzione in più "teste" (nel nostro caso 8), ognuna specializzata in diversi aspetti della relazione testo-immagine.

2. **Calcolo dell'Attenzione**:
   ```
   Attention(Q, K, V) = softmax(QK^T / √d_k) · V
   ```
   Dove:
   - Q = Query (feature dell'immagine)
   - K = Key (feature del testo)
   - V = Value (feature del testo)
   - d_k = dimensione della feature

3. **Passaggi del Processo**:
   - La query dell'immagine viene confrontata con tutte le key del testo
   - I punteggi di attenzione vengono normalizzati con softmax
   - Le feature del testo (value) vengono pesate dai punteggi di attenzione
   - Il risultato è un insieme di feature arricchite dalle informazioni testuali

## Vantaggi della Cross-Attention

1. **Allineamento Semantico**: Collega direttamente parti dell'immagine alle parole che le descrivono.

2. **Generazione Condizionata**: Consente al generatore di focalizzarsi sugli attributi menzionati nel testo.

3. **Controllo Fine**: Diversamente da una semplice concatenazione dell'embedding testuale, l'attenzione permette un controllo più granulare.

4. **Flessibilità**: Può dare più o meno importanza a diverse parti del testo a seconda del contesto visivo.

## La Nostra Implementazione

Nel nostro modello, la Cross-Attention è implementata in `src/models/attention.py`:

```python
class CrossAttentionBlock(nn.Module):
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
```

Questa implementazione include:

- **Multi-head attention** con `num_heads=8`
- **Layer normalization** per stabilizzare l'addestramento
- **Feed-forward network** per trasformare ulteriormente le feature

## Visualizzazione dell'Attenzione

Per visualizzare quali parole influenzano quali parti dell'immagine:

1. Estrarre le matrici di attenzione durante la generazione
2. Sovrapporre i pesi di attenzione all'immagine generata
3. Creare mappe di calore che mostrano come diverse parole influenzano diverse regioni

## Miglioramenti Possibili

1. **Attention Refinement**: Aggiungere più livelli di attenzione per raffinare iterativamente il condizionamento.

2. **Self-Attention**: Combinare cross-attention con self-attention per migliorare la coerenza interna delle immagini.

3. **Adaptive Attention**: Modificare dinamicamente i pesi di attenzione basandosi sulla qualità della generazione.

4. **Fine-tuning dell'Encoder**: Addestrare specificamente l'encoder testuale per produrre embedding più adatti alla generazione di Pokémon.

## Conclusioni

La Cross-Attention è il "ponte semantico" che collega il linguaggio naturale alle immagini generate. È un componente cruciale che consente al nostro modello di interpretare correttamente le descrizioni testuali e tradurle in caratteristiche visive coerenti.
