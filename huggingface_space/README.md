---
title: StackGAN Pokemon Generator
emoji: 🎮
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# StackGAN Pokemon Generator

Questo è un generatore di Pokemon basato su StackGAN che crea immagini Pokemon da descrizioni testuali.

## Caratteristiche

- **Architettura StackGAN**: Generazione a due stadi (64x64 → 215x215)
- **Text-to-Image**: Genera Pokemon dalle tue descrizioni
- **Modelli pre-allenati**: Su dataset Pokemon con caption testuali

## Come usare

1. Inserisci una descrizione del Pokemon (es: "a red fire dragon pokemon with wings")
2. Scegli un seed per risultati riproducibili
3. Clicca "Genera Pokemon!"

## Tecnologie

- **PyTorch**: Framework deep learning
- **Transformers**: Per encoding del testo (BERT)
- **Gradio**: Interfaccia web interattiva

## Modello

Il modello è basato su StackGAN con le seguenti modifiche:
- Stage-I: Genera immagini base 64x64
- Stage-II: Raffina a 215x215 ad alta risoluzione
- Text Encoder: BERT per embedding testuali
