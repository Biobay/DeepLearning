#!/bin/bash
# start_demo.sh - Script per avviare la demo Gradio

echo "🎮 Avvio della demo Pokémon Generator"
echo "=================================="

# Controlla se Python è installato
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non è installato. Installa Python 3.8+ per continuare."
    exit 1
fi

# Controlla se pip è installato
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 non è installato. Installa pip per continuare."
    exit 1
fi

echo "📦 Installazione delle dipendenze Gradio..."
pip3 install -r requirements_gradio.txt

echo "🚀 Avvio dell'interfaccia Gradio..."
python3 gradio_demo.py

echo "✅ Demo completata!"
