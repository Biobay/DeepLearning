#!/bin/bash

# Script per installare le dipendenze necessarie per la demo Gradio
echo "🔧 Installazione dipendenze per StackGAN Demo..."

# Installa Gradio se non presente
echo "📦 Installazione Gradio..."
pip install gradio

# Installa altre dipendenze opzionali per la demo
echo "📦 Installazione dipendenze aggiuntive..."
pip install pillow numpy

echo "✅ Installazione completata!"
echo ""
echo "🚀 Per avviare la demo, esegui:"
echo "   python gradio_demo.py"
echo ""
echo "🌐 La demo sarà disponibile su http://127.0.0.1:7860"
