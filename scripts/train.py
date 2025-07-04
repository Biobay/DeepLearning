import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoTokenizer
import os
import sys
from tqdm import tqdm

# Aggiunge la root del progetto al path per permettere l'import dei moduli src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import create_dataloader
from src.models.model import PikaPikaGen
import src.config as config
from src.utils import save_image_batch, plot_attention

def main():
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.
    """
    # --- 1. Setup Iniziale ---
    device = torch.device(config.DEVICE)
    print(f"Utilizzo del dispositivo: {device}")

    # Crea le directory per i risultati se non esistono
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.GENERATED_IMAGE_DIR, exist_ok=True)

    # --- 2. Tokenizer e Dataloaders ---
    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
    
    train_csv_path = os.path.join(config.SPLITS_DIR, "train.csv")
    val_csv_path = os.path.join(config.SPLITS_DIR, "val.csv")

    print("Creazione dei DataLoader per training e validazione...")
    train_loader = create_dataloader(
        csv_path=train_csv_path,
        images_dir=config.IMAGE_DIR,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        max_length=128, # Questo potrebbe essere aggiunto a config
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        csv_path=val_csv_path,
        images_dir=config.IMAGE_DIR,
        tokenizer=tokenizer,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        max_length=128, # Questo potrebbe essere aggiunto a config
        num_workers=config.NUM_WORKERS,
        shuffle=False
    )
    print("DataLoader creati.")

    # --- 3. Modello, Loss e Ottimizzatore ---
    print("Inizializzazione del modello...")
    model = PikaPikaGen(
        encoder_model_name=config.ENCODER_MODEL_NAME,
        encoder_dim=config.ENCODER_DIM,
        decoder_dim=config.DECODER_DIM,
        attention_dim=config.ATTENTION_DIM,
        context_dim=config.CONTEXT_DIM,
        output_channels=config.OUTPUT_CHANNELS,
        ngf=config.NGF,
        fine_tune_encoder=config.FINE_TUNE_ENCODER
    ).to(device)
    print("Modello inizializzato.")

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # --- 4. Ciclo di Addestramento e Validazione ---
    print("Avvio dell'addestramento...")
    for epoch in range(config.EPOCHS):
        print(f"--- Epoca {epoch+1}/{config.EPOCHS} ---")
        
        # --- Fase di Addestramento ---
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for i, batch in enumerate(train_progress_bar):
            real_images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            generated_images, _ = model(input_ids, attention_mask)
            loss = criterion(generated_images, real_images)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % config.LOG_INTERVAL == 0:
                train_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        print(f"Loss di Addestramento Media: {avg_train_loss:.4f}")

        # --- Fase di Validazione ---
        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for i, batch in enumerate(val_progress_bar):
                real_images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                generated_images, attention_weights = model(input_ids, attention_mask)
                loss = criterion(generated_images, real_images)
                
                val_loss += loss.item()

                # Salva un batch di immagini generate e il grafico dell'attenzione
                if i == 0 and (epoch + 1) % config.SAVE_IMAGE_EPOCHS == 0:
                    try:
                        # Salva le immagini generate
                        save_image_batch(
                            tensor=generated_images.cpu(),
                            output_dir=config.GENERATED_IMAGE_DIR,
                            epoch=epoch + 1,
                            batch_idx=0
                        )
                        
                        # Prepara i dati per il plot dell'attenzione (solo per la prima immagine del batch)
                        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu(), skip_special_tokens=True)
                        
                        # Pulisce i token da eventuali caratteri speciali di BERT
                        cleaned_tokens = [t.replace('##', '') for t in tokens]

                        # Plotta l'attenzione
                        plot_attention(
                            attention_weights[0].cpu(), 
                            cleaned_tokens, 
                            generated_images[0].cpu(), 
                            output_path=os.path.join(config.GENERATED_IMAGE_DIR, f"attention_epoch_{epoch+1}.png")
                        )
                    except Exception as e:
                        print(f"\n[ATTENZIONE] Impossibile salvare l'immagine o il plot dell'attenzione all'epoca {epoch+1}. Errore: {e}")
                        # L'addestramento continua comunque

        avg_val_loss = val_loss / len(val_loader)
        print(f"Loss di Validazione Media: {avg_val_loss:.4f}")

        # --- Salvataggio Checkpoint ---
        if (epoch + 1) % config.CHECKPOINT_SAVE_EPOCHS == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"pikapikagen_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint salvato in: {checkpoint_path}")

    print("Addestramento completato.")

if __name__ == '__main__':
    main()
