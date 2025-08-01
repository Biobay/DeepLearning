# 🎮 Guida Completa al Training StackGAN

## 🚀 Modi per Addestrare il Modello

Hai **3 opzioni** per addestrare il tuo modello StackGAN:

### Opzione 1: Training Interattivo (CONSIGLIATO)
```bash
./start_training.sh
```
- **Più facile**: Interfaccia guidata interattiva
- **Configurabile**: Scegli epoche, opzioni, etc.
- **Sicuro**: Verifica automatica dipendenze
- **Robusto**: Gestione errori e fallback

### Opzione 2: Training Completo Diretto
```bash
python3 train_complete_pipeline.py --stage1-epochs 50 --stage2-epochs 50
```
- **Controllo completo**: Stage-I + Stage-II
- **Parametri personalizzabili**: Vedi opzioni sotto
- **Professionale**: Log dettagliati e checkpoint

### Opzione 3: Training Rapido per Test
```bash
python3 quick_train.py --epochs 10
```
- **Veloce**: Solo 10 epoche per test
- **Semplice**: Solo Stage-I
- **Ideale per**: Verificare che tutto funzioni

---

## 📋 Parametri Disponibili

### Training Completo (`train_complete_pipeline.py`)

#### Opzioni Base:
```bash
--stage1-epochs 50      # Epoche per Stage-I (default: 50)
--stage2-epochs 50      # Epoche per Stage-II (default: 50)
```

#### Opzioni Avanzate:
```bash
--no-label-smoothing    # Disabilita label smoothing
--no-augmentation       # Disabilita data augmentation
--resume CHECKPOINT     # Riprendi da checkpoint
```

#### Esempi:
```bash
# Training completo standard
python3 train_complete_pipeline.py

# Training veloce
python3 train_complete_pipeline.py --stage1-epochs 20 --stage2-epochs 20

# Solo Stage-I
python3 train_complete_pipeline.py --stage1-epochs 50 --stage2-epochs 0

# Senza label smoothing
python3 train_complete_pipeline.py --no-label-smoothing

# Riprendi training
python3 train_complete_pipeline.py --resume results/experiment_123/stage1/checkpoints/latest_stage1.pth
```

### Training Rapido (`quick_train.py`)

```bash
python3 quick_train.py --epochs 10    # Training rapido con 10 epoche
```

---

## 🔧 Setup Iniziale

### 1. Installa Dipendenze
```bash
# Opzione A: Automatic (se hai requirements.txt)
pip3 install -r requirements.txt

# Opzione B: Manuale
pip3 install torch torchvision transformers tqdm matplotlib pillow numpy pandas
```

### 2. Verifica Dataset
Assicurati di avere:
```
data/
├── pokemon.csv          # Dataset Pokémon con descrizioni
├── splits/              # Split train/val/test
└── images/              # Immagini Pokémon
```

### 3. Verifica Configurazione
Controlla `src/config.py` per:
- Percorsi dataset corretti
- Parametri modello appropriati
- Impostazioni hardware (GPU/CPU)

---

## 📊 Durante il Training

### Cosa Aspettarsi:
1. **Verifica Ambiente**: Controllo dipendenze e file
2. **Caricamento Dataset**: Preparazione dati con augmentation
3. **Inizializzazione Modelli**: Setup encoder, generator, discriminator
4. **Stage-I Training**: Generazione immagini 64x64
5. **Stage-II Training**: Upscaling a 256x256
6. **Salvataggio**: Checkpoint e immagini di esempio

### Output durante Training:
```
🚀 === STAGE-I TRAINING (64x64) ===
Stage-I Epoch 1/50: 100%|██████| 123/123 [05:42<00:00, D_loss=0.6543, G_loss=2.1234]
Stage-I Epoch 1: D_loss=0.6543, G_loss=2.1234
💾 Checkpoint salvato: results/experiment_20250801_143022/stage1/checkpoints/checkpoint_epoch_10.pth
```

### Tempi Stimati:
- **GPU CUDA**: ~1-2 ore per 50 epoche Stage-I
- **CPU**: ~8-12 ore per 50 epoche Stage-I
- **Training Rapido**: ~10-20 minuti

---

## 📂 Struttura Risultati

Dopo il training troverai:
```
results/stackgan_complete_TIMESTAMP/
├── stage1/
│   ├── images/           # Immagini generate durante training
│   │   ├── real_epoch_1.png
│   │   ├── fake_epoch_1.png
│   │   └── ...
│   └── checkpoints/      # Modelli salvati
│       ├── checkpoint_epoch_10.pth
│       ├── checkpoint_epoch_50.pth
│       └── latest_stage1.pth
├── stage2/
│   ├── images/           # Immagini Stage-II (256x256)
│   └── checkpoints/      # Checkpoint Stage-II
├── logs/
│   ├── training_stage1.csv
│   └── training_stage2.csv
└── training_summary.json
```

---

## 🎮 Dopo il Training

### 1. Testa il Modello
```bash
# Demo Gradio interattiva
python3 gradio_demo.py

# Demo semplificata (se problemi)
python3 gradio_demo_simple.py
```

### 2. Analizza Risultati
- **Immagini**: Guarda `results/*/stage*/images/`
- **Loss**: Apri file CSV in `results/*/logs/`
- **Checkpoint**: Usa per inference o resume training

### 3. Condividi Demo
```bash
# Demo locale
python3 gradio_demo.py

# Demo pubblica (link condivisibile)
GRADIO_MODE=demo python3 gradio_demo_advanced.py
```

---

## ⚡ Training Rapido per Principianti

Se sei nuovo o vuoi solo testare:

```bash
# 1. Training super veloce (5 minuti)
python3 quick_train.py --epochs 5

# 2. Testa subito il risultato
python3 gradio_demo_simple.py

# 3. Se funziona, prova training completo
./start_training.sh
```

---

## 🛠️ Risoluzione Problemi

### Errore "CUDA out of memory"
```bash
# Riduci batch size in src/config.py
BATCH_SIZE = 8  # invece di 16

# Oppure usa CPU
export CUDA_VISIBLE_DEVICES=""
```

### Errore "Dataset not found"
```bash
# Verifica percorsi in src/config.py
CSV_PATH = "data/pokemon.csv"
IMAGE_DIR = "data/images"
SPLITS_DIR = "data/splits"
```

### Training troppo lento
```bash
# Usa training rapido
python3 quick_train.py --epochs 10

# Oppure riduci epoche
python3 train_complete_pipeline.py --stage1-epochs 20 --stage2-epochs 20
```

### Checkpoint corrotto
```bash
# Riavvia senza resume
python3 train_complete_pipeline.py  # senza --resume

# Oppure usa checkpoint precedente
python3 train_complete_pipeline.py --resume results/*/stage1/checkpoints/checkpoint_epoch_40.pth
```

---

## 📈 Ottimizzazione Performance

### Per Training Veloce:
1. **Usa GPU** se disponibile
2. **Riduci epoche** inizialmente (20+20)
3. **Batch size** ottimale per la tua GPU
4. **Mixed precision** (automatico in PyTorch moderni)

### Per Qualità Migliore:
1. **Più epoche** (100+ per Stage-I)
2. **Learning rate scheduling**
3. **Data augmentation** avanzata
4. **Label smoothing** fine-tuned

### Per Debugging:
1. **Training rapido** (10 epoche)
2. **Batch size piccolo** (4-8)
3. **Log verbosi** abilitati
4. **Checkpoint frequenti**

---

## 🎯 Prossimi Passi

1. **Avvia il training**:
   ```bash
   ./start_training.sh
   ```

2. **Monitora i progressi**:
   - Guarda le loss che diminuiscono
   - Controlla le immagini generate

3. **Testa il modello**:
   ```bash
   python3 gradio_demo.py
   ```

4. **Condividi i risultati**:
   - Demo web interattiva
   - Immagini generate
   - Analisi delle performance

**Buon training! 🚀**
