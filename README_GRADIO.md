# 🎮 Pokémon Generator - Demo Gradio

Questo progetto fornisce un'interfaccia web interattiva per generare sprite Pokémon basati su descrizioni testuali utilizzando il modello StackGAN addestrato.

## 📋 Panoramica del Progetto

Il **Pokémon Generator** è un sistema di Text-to-Image synthesis che:
- Prende in input una descrizione testuale di un Pokémon
- Genera sprite 2D corrispondenti alla descrizione
- Utilizza l'architettura StackGAN con meccanismi di Cross-Attention
- Implementa Label Smoothing per stabilizzare l'addestramento GAN

## 🚀 Avvio Rapido

### Prerequisiti
- Python 3.8+
- pip3

### Installazione e Avvio

1. **Installa le dipendenze**:
   ```bash
   pip install -r requirements_gradio.txt
   ```

2. **Avvia la demo** (scegli una delle opzioni):

   **Opzione A - Demo completa** (richiede modelli addestrati):
   ```bash
   python gradio_demo.py
   ```

   **Opzione B - Demo semplificata** (funziona senza modelli):
   ```bash
   python gradio_demo_simple.py
   ```

   **Opzione C - Script automatico**:
   ```bash
   chmod +x start_demo.sh
   ./start_demo.sh
   ```

3. **Apri il browser** su: http://127.0.0.1:7860

## 📁 File della Demo

### `gradio_demo.py`
Demo completa che utilizza i modelli StackGAN addestrati:
- Carica automaticamente i checkpoint dei modelli
- Genera immagini reali utilizzando il generatore addestrato
- Richiede modelli pre-addestrati nella cartella `results/label_smoothing_experiment/checkpoints/`

### `gradio_demo_simple.py`
Demo semplificata per scopi didattici:
- Funziona senza modelli addestrati
- Genera immagini placeholder basate sulla descrizione
- Perfetta per testare l'interfaccia e comprendere il flusso di lavoro

### `requirements_gradio.txt`
Lista delle dipendenze necessarie per eseguire la demo Gradio.

### `start_demo.sh`
Script di avvio automatico che:
- Installa le dipendenze
- Avvia la demo
- Gestisce eventuali errori comuni

## 🎯 Come Usare la Demo

### 1. Inserimento della Descrizione
Scrivi una descrizione dettagliata del Pokémon che vuoi generare. Esempi efficaci:

```
"A small electric mouse Pokemon with yellow fur, red cheeks, and a lightning bolt-shaped tail"

"A blue turtle Pokemon with a hard shell and water abilities"

"A fire-type dragon Pokemon with orange scales and large wings"

"A grass-type Pokemon with a bulb on its back and green skin"
```

### 2. Configurazione dei Parametri
- **Numero di immagini**: Scegli quante varianti generare (1-4)
- **Seed**: Opzionale, per risultati riproducibili
- **Usa seed fisso**: Abilita per risultati deterministici

### 3. Generazione
Clicca "Genera Pokémon" e attendi i risultati!

## 💡 Suggerimenti per Descrizioni Efficaci

### ✅ Includi sempre:
- **Colori specifici**: "giallo", "blu scuro", "rosso brillante"
- **Tipo del Pokémon**: "elettrico", "acqua", "fuoco", "erba", "psichico"
- **Caratteristiche fisiche**: "coda lunga", "ali grandi", "guscio duro"
- **Dimensioni**: "piccolo", "grande", "massiccio", "minuscolo"

### ✅ Esempi di caratteristiche utili:
- **Elementi corporei**: "artigli affilati", "antenna", "corna", "cresta"
- **Texture**: "pelliccia morbida", "scaglie lucenti", "pelle lisa"
- **Accessori**: "collare", "gemma sul petto", "segni distintivi"

### ❌ Evita:
- Descrizioni troppo vaghe: "un Pokémon carino"
- Troppi dettagli contraddittori
- Riferimenti a Pokémon specifici esistenti

## 🧠 Architettura Tecnica

### Modelli Utilizzati
- **Text Encoder**: BERT-mini per elaborazione del linguaggio naturale
- **Generator**: U-Net con Cross-Attention per la sintesi delle immagini
- **Discriminator**: Multi-scale per valutazione realistica delle immagini

### Pipeline di Generazione
1. **Tokenizzazione**: La descrizione viene convertita in token
2. **Encoding**: Estrazione di embedding testuali semantici
3. **Cross-Attention**: Collegamento tra testo e caratteristiche visive
4. **Generazione**: Sintesi dell'immagine 64x64 pixel
5. **Post-processing**: Upscaling a 256x256 per visualizzazione

### Tecniche Avanzate
- **Label Smoothing**: Stabilizza l'addestramento GAN
- **Data Augmentation**: Migliora la robustezza del modello
- **Multi-scale Discrimination**: Valutazione a più livelli di dettaglio

## 📊 Dataset e Training

### Dataset
- **Fonte**: Sprite Pokémon ufficiali con descrizioni del Pokédex
- **Dimensione**: Centinaia di coppie testo-immagine
- **Preprocessing**: Normalizzazione e augmentation dei dati

### Training
Per addestrare il modello:
1. Utilizza il notebook `LabelSmoothing_Experiment.ipynb`
2. Segui il processo di training completo
3. I checkpoint verranno salvati automaticamente
4. Copia i checkpoint nella cartella corretta per la demo

## 🔧 Risoluzione Problemi

### La demo non si avvia
```bash
# Verifica Python
python3 --version

# Reinstalla dipendenze
pip3 install --upgrade -r requirements_gradio.txt

# Prova la demo semplificata
python3 gradio_demo_simple.py
```

### Errore "Checkpoint non trovato"
- Usa `gradio_demo_simple.py` per testare l'interfaccia
- Addestra il modello prima di usare `gradio_demo.py`
- Verifica il percorso dei checkpoint

### Prestazioni lente
- Usa GPU se disponibile
- Riduci il numero di immagini generate
- Chiudi altre applicazioni pesanti

## 🌐 Condivisione della Demo

### Accesso Locale (Default)
- URL: http://127.0.0.1:7860
- Accessibile solo dal tuo computer

### Condivisione Pubblica (Opzionale)
Modifica `gradio_demo.py` e cambia:
```python
demo.launch(share=True)  # Crea un link pubblico temporaneo
```

⚠️ **Attenzione**: I link pubblici sono temporanei e espongono la tua demo su internet.

## 🔬 Aspetti Educativi

### Concetti Dimostrati
- **Text-to-Image Synthesis**: Conversione da linguaggio naturale a immagini
- **Generative Adversarial Networks**: Competizione tra generatore e discriminatore
- **Cross-Attention Mechanisms**: Collegamento semantico tra modalità diverse
- **Transfer Learning**: Utilizzo di modelli pre-addestrati (BERT)

### Applicazioni Pratiche
- Design automatico di personaggi
- Prototipazione rapida di concept art
- Generazione di contenuti per videogiochi
- Strumenti creativi per artisti

## 📈 Possibili Miglioramenti

### Tecnici
- **Stage-II StackGAN**: Generazione ad alta risoluzione (256x256)
- **StyleGAN Integration**: Controllo più fine dello stile
- **Classifier Guidance**: Miglior controllo dei tipi Pokémon

### Interfaccia
- **Editing Interattivo**: Modifica delle immagini generate
- **Batch Generation**: Generazione di molte varianti
- **Style Transfer**: Applicazione di stili artistici

### Dataset
- **Descrizioni Multiple**: Più varianti per ogni Pokémon
- **Attributi Strutturati**: Controllo preciso di caratteristiche
- **Generazione Condizionale**: Controllo di tipo, colore, dimensione

## 📄 Licenza e Crediti

Questo progetto è sviluppato per scopi educativi nel contesto di un corso di Deep Learning.

### Tecnologie Utilizzate
- **PyTorch**: Framework di deep learning
- **Transformers**: Modelli di linguaggio pre-addestrati
- **Gradio**: Framework per interfacce ML interattive
- **Pillow**: Elaborazione delle immagini

### Ispirazione
- Paper originale StackGAN
- Tecniche di Text-to-Image synthesis
- Dataset Pokémon della community

---

🎮 **Buon divertimento con la generazione di Pokémon!** 🎮
