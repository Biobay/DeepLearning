# PikaPikaGen: Generatore di Pokémon Text-to-Image

Questo progetto implementa un modello di deep learning per generare sprite di Pokémon a partire da descrizioni testuali. Utilizza un'architettura Encoder-Decoder con un meccanismo di attenzione per tradurre il testo in immagini.

## Struttura del Progetto

```
DeepLearning/
│
├── README.md                 # Documentazione principale
├── requirements.txt          # Dipendenze Python
├── setup.py                 # Setup del package
├── .gitignore              # File da ignorare in Git
│
├── data/                   # Dataset e dati
│   ├── pokemon.csv         # Dataset principale (già presente)
│   ├── processed/          # Dati preprocessati
│   └── splits/             # Split train/val/test
│
├── small_images/           # Immagini Pokémon (già presente)
│
├── src/                    # Codice sorgente principale
│   ├── __init__.py
│   ├── data/               # Gestione dei dati
│   │   ├── __init__.py
│   │   ├── dataset.py      # Dataset PyTorch custom
│   │   ├── preprocessing.py # Preprocessing dati
│   │   └── transforms.py   # Trasformazioni immagini
│   │
│   ├── models/             # Architetture dei modelli
│   │   ├── __init__.py
│   │   ├── encoder.py      # Text Encoder (BERT-mini)
│   │   ├── decoder.py      # Image Decoder
│   │   ├── attention.py    # Meccanismo di attenzione
│   │   └── pikapikaGen.py  # Modello principale
│   │
│   ├── training/           # Training e validazione
│   │   ├── __init__.py
│   │   ├── trainer.py      # Classe Trainer
│   │   ├── losses.py       # Loss functions
│   │   └── metrics.py      # Metriche di valutazione
│   │
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── config.py       # Configurazioni
│       ├── visualization.py # Plot e visualizzazioni
│       └── checkpoint.py   # Salvataggio modelli
│
├── notebooks/              # Jupyter notebooks per analisi
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── scripts/                # Script eseguibili
│   ├── train.py           # Script principale di training
│   ├── evaluate.py        # Script di valutazione
│   ├── generate.py        # Script per generazione
│   └── preprocess_data.py # Preprocessing batch
│
├── tests/                  # Test unitari
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_training.py
│
├── results/                # Risultati esperimenti
│   ├── checkpoints/        # Modelli salvati
│   ├── logs/              # Log di training
│   ├── generated_images/   # Immagini generate
│   └── metrics/           # Metriche salvate
│
├── docs/                   # Documentazione aggiuntiva
│   ├── architecture.md    # Documentazione architettura
│   ├── experiments.md     # Log degli esperimenti
│   └── report.md          # Report finale
│
└── demo/                   # Demo e interfacce
    ├── gradio_app.py      # Interfaccia Gradio
    ├── streamlit_app.py   # Interfaccia Streamlit (opzionale)
    └── assets/            # Asset per demo
```

## Setup

1.  **Clona il repository:**
    ```bash
    git clone https://github.com/Biobay/DeepLearning.git
    cd DeepLearning
    ```

2.  **Crea un ambiente virtuale:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Su Windows: venv\Scripts\activate
    ```

3.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Installa il pacchetto in modalità editabile**
    ```bash
    pip install -e .
    ```

## Utilizzo

### Training
Per avviare il training del modello, esegui lo script `train.py`:
```bash
python scripts/train.py --data_dir data/ --batch_size 16 --epochs 100
```

### Valutazione
Per valutare un modello salvato:
```bash
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.pth
```

### Generazione
Per generare nuove immagini da un testo:
```bash
python scripts/generate.py --checkpoint results/checkpoints/best_model.pth --text "Un pokémon di tipo fuoco con le ali"
```
