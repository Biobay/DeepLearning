mente! Ecco un piccolo riassunto con le principali specifiche del progetto "PikaPikaGen":
Il progetto, denominato PikaPikaGen, è un'esercitazione di Deep Learning proposta dall'Instructor Vito Walter Anelli del Politecnico di Bari.
Obiettivo Principale:
•
Sviluppare un modello di sintesi Text-to-Image che generi sprite Pokémon 2D a partire da descrizioni testuali dettagliate (come quelle del Pokédex). Questo richiede una profonda comprensione semantica sia del testo che del dominio visivo.
Architettura del Modello: Il modello principale sarà un'architettura Encoder-Decoder con Meccanismo di Attenzione.
•
Encoder Testuale: Sarà un Transformer-based Encoder.
◦
Il layer di embedding verrà inizializzato con vettori pre-trained bert-mini (256 dimensioni) e sarà fine-tuned durante l'addestramento per adattarsi al dominio specifico dei Pokémon.
◦
Il ruolo dell'encoder è produrre una sequenza di stati nascosti contestuali per ogni token di input.
•
Decoder Immagine (Generatore): Sarà una Convolutional Neural Network (CNN).
◦
Genererà l'immagine partendo da un vettore di rumore casuale concatenato con un vettore di contesto derivato dal testo.
◦
Utilizzerà una serie di strati di Convoluzione Trasposta (Transposed Convolution) per aumentare le dimensioni spaziali fino all'immagine finale di 215x215 pixel.
◦
Ogni strato di convoluzione trasposta sarà seguito da un Normalization Layer e una funzione di attivazione.
•
Meccanismo di Attenzione: Colla l'encoder al decoder, permettendo al decoder di "concentrarsi" selettivamente sulle parole più rilevanti della descrizione durante la generazione di diverse parti dell'immagine.
Dataset:
•
Verrà utilizzato "The Pokémon Dataset", disponibile pubblicamente.
•
Contiene: Descrizioni Testuali (Pokédex entries), Immagini Sprite di alta qualità (215x215 pixel con sfondi trasparenti), e Metadati (tipo, nome, statistiche).
Pre-processing dei Dati:
•
Testo: Parsare il file CSV, pulire e tokenizzare il testo usando bert-mini. Mappare i token a indici interi e opzionalmente padare le sequenze a una lunghezza fissa.
•
Immagini: Caricare le sprite 215x215. Opzionalmente, normalizzare i valori dei pixel (es. a [-1, 1]) e gestire il canale alfa (trasparenza), ad esempio mescolandolo con uno sfondo bianco.
Addestramento e Valutazione:
•
Il modello sarà addestrato end-to-end.
•
La funzione di perdita (loss function) suggerita è una Reconstruction Loss pixel-wise, con la L1 (Mean Absolute Error) preferita rispetto alla L2 per produrre immagini meno sfocate.
•
Monitorare le performance su un set di validazione e fine-tunare gli iperparametri. La valutazione finale sarà eseguita su un test set.
Consegne (Deliverables):
•
Un'implementazione del codice Python ben commentata.
•
Un report dettagliato sull'approccio, inclusi pre-processing dati, architettura, addestramento e risultati di valutazione.
•
Un'analisi delle performance del modello, delle sfide affrontate e raccomandazioni per miglioramenti.
•
Una presentazione per l'esame orale.
•
IMPORTANTE: Deve essere creata una demo interattiva live del modello addestrato usando Gradio, con un'interfaccia web per l'input testuale e la visualizzazione della sprite generata.
Le scadenze del progetto sono: rilascio il 2 luglio 2025 e scadenza il 3 agosto 2025. Si incoraggia l'uso di librerie esistenti come TensorFlow o PyTorch.