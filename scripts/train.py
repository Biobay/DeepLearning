# In scripts/train.py

# ...
import torch
# ...

def train(cfg):
    # ... (tutto il setup)

    print("\nINIZIO SESSIONE DI DEBUG...")
    for epoch in range(1): # ESEGUIAMO UNA SOLA EPOCA
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None: continue
            if batch_idx > 5: break # ESEGUIAMO SOLO 5 BATCH PER VELOCIZZARE

            print(f"\n--- DEBUG BATCH {batch_idx} ---")

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            real_images = batch['image'].to(device)

            # 1. CONTROLLO DATI DI INPUT
            print(f"Shape input_ids: {input_ids.shape}")
            print(f"Shape real_images: {real_images.shape}")
            print(f"Valore medio real_images: {real_images.mean().item():.4f}")
            if torch.isnan(real_images).any() or torch.isinf(real_images).any():
                print("!!!!!! ERRORE: NaN o Inf nelle immagini reali !!!!!!")
                return

            # --- ESEGUIAMO IL FORWARD PASS DEL GENERATORE ---
            text_features = model.encoder(input_ids, attention_mask)
            
            # 2. CONTROLLO OUTPUT ENCODER
            print(f"Shape text_features: {text_features.shape}")
            print(f"Valore medio text_features: {text_features.mean().item():.4f}")
            if torch.isnan(text_features).any() or torch.isinf(text_features).any():
                print("!!!!!! ERRORE: NaN o Inf nell'output dell'encoder !!!!!!")
                return
            
            generated_images, _ = model.decoder(text_features)
            
            # 3. CONTROLLO OUTPUT GENERATORE
            print(f"Shape generated_images: {generated_images.shape}")
            print(f"Valore medio generated_images: {generated_images.mean().item():.4f}")
            
            real_images_resized = F.interpolate(real_images, size=(cfg.IMAGE_OUTPUT_SIZE, cfg.IMAGE_OUTPUT_SIZE))
            loss = criterion(generated_images, real_images_resized)
            
            print(f"Loss Iniziale: {loss.item():.4f}")

            # --- BACKWARD PASS E CONTROLLO GRADIENTI ---
            optimizer.zero_grad()
            loss.backward()
            
            # 4. CONTROLLO GRADIENTI
            total_norm_encoder = 0
            for p in model.encoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_encoder += param_norm.item() ** 2
            total_norm_encoder = total_norm_encoder ** 0.5
            print(f"Norma L2 dei gradienti dell'Encoder: {total_norm_encoder:.4f}")

            total_norm_decoder = 0
            for p in model.decoder.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_decoder += param_norm.item() ** 2
            total_norm_decoder = total_norm_decoder ** 0.5
            print(f"Norma L2 dei gradienti del Decoder: {total_norm_decoder:.4f}")
            
            if total_norm_encoder == 0 or total_norm_decoder == 0:
                print("!!!!!! ERRORE: Gradiente nullo! Il modello non sta imparando. !!!!!!")

            optimizer.step()
            
    print("\n--- DEBUG COMPLETATO ---")
    return {} # Terminiamo dopo il debug