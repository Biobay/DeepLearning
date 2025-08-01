# 🎨 StackGAN Demo con Gradio

Demo interattiva per il modello StackGAN che genera immagini di Pokémon da descrizioni testuali.

## 🚀 Avvio Rapido

### 1. Installa le dipendenze
```bash
# Installa Gradio e dipendenze
./install_demo_deps.sh

# O manualmente:
pip install gradio pillow numpy
```

### 2. Testa la configurazione (opzionale)
```bash
python test_demo.py
```

### 3. Avvia la demo
```bash
python gradio_demo.py
```

La demo sarà disponibile su: **http://127.0.0.1:7860**

## 📋 Requisiti

### Dipendenze Python
- `torch` e `torchvision`
- `transformers` 
- `gradio` 
- `pillow`
- `numpy`

### Modelli Allenati
La demo cerca i checkpoint dei modelli in:
- **Stage-I**: `results/checkpoints/generator_s1_final.pth`
- **Stage-II**: `results/checkpoints_s2/generator_s2_final.pth`

**Nota**: Se i checkpoint non esistono, la demo userà pesi casuali (per test).

## 🎯 Come Usare

### Interfaccia Web
1. **Inserisci una descrizione** del Pokémon che vuoi generare
2. **Imposta un seed** (opzionale) per risultati riproducibili
3. **Clicca "Genera Immagine"** e attendi il risultato
4. **Visualizza** entrambe le versioni: Stage-I (64x64) e Stage-II (215x215)

### Esempi di Prompt
```
"a small bird with a red head and black wings"
"a large blue bird with white stripes on its wings" 
"a yellow electric mouse pokemon with red cheeks"
"a fire dragon with orange scales and blue eyes"
"a small green grass type pokemon with a flower on its head"
```

## 🏗️ Architettura

### Pipeline di Generazione
```
Testo → [Text Encoder] → Text Embedding
                              ↓
Rumore Z → [Generator S1] → Immagine 64x64
                              ↓
[Generator S2] + Text Embedding → Immagine 215x215
```

### Componenti
- **Text Encoder**: BERT-mini per encoding del testo
- **Stage-I Generator**: CNN che genera immagini 64x64 
- **Stage-II Generator**: CNN con blocchi residuali per upscaling a 215x215

## 📂 Struttura dei File

```
├── gradio_demo.py           # Demo principale con interfaccia Gradio
├── test_demo.py            # Script di test per verificare la configurazione
├── install_demo_deps.sh    # Script per installare le dipendenze
├── DEMO_README.md          # Questo file
└── results/
    ├── checkpoints/        # Checkpoint Stage-I
    ├── checkpoints_s2/     # Checkpoint Stage-II  
    └── generated_images/   # Immagini generate dalla demo
```

## 🔧 Risoluzione Problemi

### Errore: "ModuleNotFoundError: No module named 'gradio'"
```bash
pip install gradio
```

### Errore: "CUDA out of memory"
La demo usa automaticamente CPU se CUDA non è disponibile.

### Checkpoint non trovati
La demo funziona anche senza checkpoint (usa pesi casuali), ma per risultati migliori:
1. Completa l'allenamento Stage-I e Stage-II  
2. Verifica i percorsi dei checkpoint in `src/config.py`

### Immagini di bassa qualità
Se i checkpoint non sono allenati, le immagini saranno casuali. Allena prima i modelli:
```bash
# Allena Stage-I
python scripts/train.py

# Allena Stage-II  
python scripts/train_stage2.py
```

## 📊 Monitoraggio

### Storia delle Generazioni
La demo mantiene una storia delle generazioni recenti accessibile tramite la tab "Storia".

### Salvataggio Automatico
Tutte le immagini generate vengono salvate automaticamente in:
- `results/generated_images/generated_TIMESTAMP_XX_hr.png` (alta risoluzione)
- `results/generated_images/generated_TIMESTAMP_XX_lr.png` (bassa risoluzione)

## 🌐 Condivisione

Per condividere la demo pubblicamente, modifica in `gradio_demo.py`:
```python
interface.launch(
    share=True,  # Abilita condivisione pubblica
    server_name="0.0.0.0",
    server_port=7860
)
```

## 📈 Performance

### Tempi di Generazione Tipici
- **CPU**: ~10-30 secondi per immagine
- **GPU**: ~2-5 secondi per immagine

### Memoria Richiesta
- **CPU**: ~2-4 GB RAM
- **GPU**: ~2-6 GB VRAM (dipende dalla GPU)

## 🔍 Debug

Per debug avanzato, modifica `debug=True` in:
```python
interface.launch(debug=True)
```

---

💡 **Suggerimento**: Per migliori risultati, usa descrizioni dettagliate dei Pokémon includendo colori, caratteristiche fisiche e tipo.
