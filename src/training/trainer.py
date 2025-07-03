import torch
import torch.nn as nn
from tqdm import tqdm
import os

class Trainer:
    """
    Classe per gestire il training e la validazione del modello PikaPikaGen.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, epochs, checkpoint_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.best_val_loss = float('inf')

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _train_one_epoch(self):
        """Esegue un'epoca di training."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.epochs} [Training]")

        for batch in progress_bar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Forward pass
            generated_images = self.model(input_ids, attention_mask)
            
            # Calcola la loss (es. L1 Loss tra immagine generata e reale)
            loss = self.criterion(generated_images, images)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self):
        """Esegue un'epoca di validazione."""
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1}/{self.epochs} [Validation]")

        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                generated_images = self.model(input_ids, attention_mask)
                
                # Calcola la loss
                loss = self.criterion(generated_images, images)
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.val_loader)

    def train(self):
        """Avvia il ciclo di training completo."""
        for epoch in range(self.epochs):
            self.epoch = epoch

            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Salva il checkpoint se la validation loss migliora
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_model.pth')

            # Salva l'ultimo checkpoint
            self._save_checkpoint('last_model.pth')

    def _save_checkpoint(self, filename):
        """Salva lo stato del modello."""
        checkpoint = {
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        print(f"Checkpoint salvato in {os.path.join(self.checkpoint_dir, filename)}")
