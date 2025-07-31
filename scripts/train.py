# scripts/train.py
import os, sys, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.utils import save_image
from tqdm import tqdm
import src.config as cfg
from src.data.dataset import create_dataloaders
from src.models.model import PikaPikaGen

def train(cfg):
    device = torch.device(cfg.DEVICE)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True); os.makedirs(cfg.GENERATED_IMAGE_DIR, exist_ok=True)
    train_loader, _, _ = create_dataloaders(
        csv_path=os.path.join(cfg.DATA_DIR, cfg.CSV_NAME), img_dir=cfg.IMAGE_DIR,
        splits_dir=cfg.SPLITS_DIR, config=cfg)

    model = PikaPikaGen(cfg).to(device)
    gen, disc = model, model.discriminator
    
    opt_gen = optim.Adam(list(gen.encoder.parameters()) + list(gen.decoder.parameters()), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, 0.999))
    
    bce_loss, l1_loss = nn.BCEWithLogitsLoss(), nn.L1Loss()
    history = {'gen_loss': [], 'disc_loss': []}

    print("\nInizio addestramento finale con cGAN Multi-Scala e Cross-Attention...")
    for epoch in range(cfg.EPOCHS):
        gen.train(); disc.train()
        total_g_loss, total_d_loss = 0.0, 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}"):
            if batch is None: continue
            ids, mask, real_full = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['image'].to(device)
            real = F.interpolate(real_full, cfg.IMAGE_OUTPUT_SIZE)
            fake, _ = gen.forward_generator(ids, mask)

            # Train Discriminator
            opt_disc.zero_grad()
            disc_real_preds, disc_fake_preds = disc(real, real), disc(fake.detach(), real)
            loss_d_real, loss_d_fake = 0, 0
            for pred in disc_real_preds: loss_d_real += bce_loss(pred, torch.ones_like(pred))
            for pred in disc_fake_preds: loss_d_fake += bce_loss(pred, torch.zeros_like(pred))
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_disc.step()

            # Train Generator
            opt_gen.zero_grad()
            disc_gen_preds = disc(fake, real)
            loss_g_gan = 0
            for pred in disc_gen_preds: loss_g_gan += bce_loss(pred, torch.ones_like(pred))
            loss_g_l1 = l1_loss(fake, real) * cfg.LAMBDA_L1
            loss_g = loss_g_gan + loss_g_l1
            loss_g.backward()
            opt_gen.step()
            
            total_g_loss += loss_g.item(); total_d_loss += loss_d.item()
        
        avg_g_loss, avg_d_loss = total_g_loss / len(train_loader), total_d_loss / len(train_loader)
        history['gen_loss'].append(avg_g_loss); history['disc_loss'].append(avg_d_loss)
        print(f"-> Gen Loss: {avg_g_loss:.4f} | Disc Loss: {avg_d_loss:.4f}")
        
        if (epoch + 1) % cfg.SAVE_IMAGE_EPOCHS == 0:
            save_image(real, os.path.join(cfg.GENERATED_IMAGE_DIR, f"real_e{epoch+1}.png"), normalize=True)
            save_image(fake, os.path.join(cfg.GENERATED_IMAGE_DIR, f"gen_e{epoch+1}.png"), normalize=True)
            print("Immagini salvate.")
        if (epoch + 1) % cfg.CHECKPOINT_SAVE_EPOCHS == 0:
            torch.save(gen.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f"gen_e{epoch+1}.pth"))
            print("Checkpoint salvato.")
            
    return history

if __name__ == '__main__': train(cfg)