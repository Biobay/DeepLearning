import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

def save_image_batch(tensor, output_dir, epoch, batch_idx, normalize=True):
    """
    Salva un batch di immagini generate in una griglia.

    Args:
        tensor (torch.Tensor): Batch di immagini da salvare (B, C, H, W).
        output_dir (str): Directory dove salvare l'immagine.
        epoch (int): Epoca corrente.
        batch_idx (int): Indice del batch corrente.
        normalize (bool): Se normalizzare l'immagine nell'intervallo [0, 1].
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Crea una griglia di immagini
    grid = vutils.make_grid(tensor, padding=2, normalize=normalize, scale_each=True)
    
    # Salva l'immagine
    filename = os.path.join(output_dir, f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png")
    try:
        vutils.save_image(grid, filename)
        print(f"Immagini generate salvate in: {filename}")
    except Exception as e:
        print(f"ATTENZIONE: Impossibile salvare il batch di immagini a causa di un errore: {e}")
        print("L'addestramento continua...")

def denormalize_and_save_image(tensor, file_path):
    """
    Denormalizza un tensore di immagine e lo salva come file.
    """
    # Denormalizza l'immagine. L'output del generatore è in [-1, 1] (tanh).
    # Dobbiamo riportarlo in [0, 1] per la visualizzazione/salvataggio.
    image = (tensor.cpu().detach() + 1) / 2.0
    
    # Rimuove la dimensione del batch se presente
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)
        
    # Converte in PIL Image
    transform = ToPILImage()
    pil_image = transform(image)
    
    # Salva l'immagine
    pil_image.save(file_path)
    print(f"Immagine salvata in: {file_path}")

def plot_attention(attention_weights, tokens, image, output_path):
    """
    Visualizza e salva i pesi di attenzione su un'immagine.
    Gestisce sia l'attenzione additiva (2D) che la Multi-Head (4D).

    Args:
        attention_weights (torch.Tensor): Pesi di attenzione.
        tokens (list): Lista di token corrispondenti ai pesi.
        image (torch.Tensor): Immagine generata (C, H, W).
        output_path (str): Path dove salvare il plot.
    """
    # --- MODIFICA CHIAVE: Gestione dei pesi da Multi-Head Attention ---
    # I pesi da MultiHeadAttention hanno shape (batch, heads, query_len, key_len)
    # Nel nostro caso, per il primo elemento del batch, sarà (num_heads, 1, seq_len)
    if attention_weights.dim() == 3:
        # Calcoliamo la media dei pesi attraverso le teste
        attention_weights = attention_weights.mean(dim=0)

    # Pulisce e prepara i dati
    attention_weights = attention_weights.squeeze().cpu().detach().numpy()
    
    # Tronca i pesi di attenzione e i token alla lunghezza effettiva (rimuovendo il padding)
    num_tokens = len(tokens)
    attention_weights = attention_weights[:num_tokens]

    image = image.permute(1, 2, 0).cpu().detach().numpy()
    # Normalizza l'immagine se non è già in [0, 1]
    if image.min() < 0 or image.max() > 1:
        image = (image + 1) / 2

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    # Mostra l'immagine
    ax[0].imshow(image)
    ax[0].set_title("Immagine Generata")
    ax[0].axis("off")

    # Mostra i pesi di attenzione
    ax[1].barh(np.arange(len(tokens)), attention_weights, align='center')
    ax[1].set_yticks(np.arange(len(tokens)))
    ax[1].set_yticklabels(tokens)
    ax[1].invert_yaxis()  # Per avere il primo token in alto
    ax[1].set_xlabel("Peso di Attenzione")
    ax[1].set_title("Attenzione sul Testo")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot dell'attenzione salvato in: {output_path}")
