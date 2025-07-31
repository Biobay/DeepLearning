# scripts/train.py (MODALITÀ TEST DI SANITÀ)

import os, sys, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.utils import save_image
from tqdm import tqdm

import src.config as cfg
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def train(cfg):
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    
    train_loader, _, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME), img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR, config=cfg)

    model = PikaPikaGen(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # --- TEST DI SANITÀ: PRENDIAMO UN SOLO BATCH ---
    print("\n--- INIZIO TEST DI SANITÀ (OVERFIT SU UN SINGOLO BATCH) ---")
    try:
        single_batch = next(iter(train_loader))
        print("Batch di test caricato con successo.")
    except StopIteration:
        print("ERRORE: Dataloader è vuoto. Impossibile eseguire il test.")
        return

    # Spostiamo il batch sul device UNA SOLA VOLTA
    ids = single_batch['input_ids'].to(device)
    mask = single_batch['attention_mask'].to(device)
    real_full = single_batch['image'].to(device)
    real = F.interpolate(real_full, size=cfg.IMAGE_OUTPUT_SIZE)
    
    # Salviamo l'immagine target per riferimento
    save_image(real, os.path.join(cfg.GENERATED_IMAGE_DIR, "TARGET_IMAGE.png"), normalize=True)
    print("Immagine target salvata in 'TARGET_IMAGE.png'")
    
    # --- CICLO DI OVERFITTING ---
    # Addestriamo per molte iterazioni sullo stesso identico batch
    iterations = 500
    for i in tqdm(range(iterations), desc="Overfitting su un batch"):
        model.train()
        
        gen_img, _ = model.forward_generator(ids, mask)
        loss = criterion(gen_img, real)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 50 == 0:
            print(f"Iterazione {i+1}/{iterations}, Loss: {loss.item():.4f}")
            # Salviamo l'output per vedere i progressi
            save_image(gen_img, os.path.join(cfg.GENERATED_IMAGE_DIR, f"overfit_iter_{i+1}.png"), normalize=True)

    print("\n--- TEST DI SANITÀ COMPLETATO ---")
    print("Controlla la cartella dei risultati per vedere se l'immagine generata converge a quella target.")
    return {}

if __name__ == '__main__':
    train(cfg)