#!/bin/bash

# 🎮 Pokémon Generator - Setup Script
# Script di installazione automatica per la demo Gradio

echo "🎮 === POKEMON GENERATOR SETUP === 🎮"
echo ""

# Colori per output colorato
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funzione per stampare messaggi colorati
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica Python
print_status "Verifico versione Python..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 non trovato! Installa Python 3.8+ prima di continuare."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_success "Python $PYTHON_VERSION trovato"

# Verifica pip
print_status "Verifico pip..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 non trovato! Installa pip prima di continuare."
    exit 1
fi
print_success "pip3 trovato"

# Installa dipendenze
print_status "Installo dipendenze Gradio..."
if pip3 install -r requirements_gradio.txt; then
    print_success "Dipendenze installate con successo!"
else
    print_error "Errore durante l'installazione delle dipendenze"
    print_warning "Prova manualmente: pip3 install -r requirements_gradio.txt"
    exit 1
fi

echo ""
print_success "🎉 Setup completato con successo!"
echo ""

# Verifica modelli addestrati
print_status "Verifico presenza modelli addestrati..."
if [ -d "results/label_smoothing_experiment/checkpoints" ]; then
    CHECKPOINT_COUNT=$(find results/label_smoothing_experiment/checkpoints -name "*.pth" 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        print_success "Trovati $CHECKPOINT_COUNT checkpoint! Puoi usare la demo completa."
        DEMO_TYPE="full"
    else
        print_warning "Cartella checkpoint esistente ma vuota."
        DEMO_TYPE="simple"
    fi
else
    print_warning "Nessun modello addestrato trovato."
    DEMO_TYPE="simple"
fi

echo ""
echo "📋 === OPZIONI DISPONIBILI ==="

if [ "$DEMO_TYPE" = "full" ]; then
    echo ""
    echo "✅ Demo Completa (CON modelli addestrati):"
    echo "   python3 gradio_demo.py"
    echo ""
    echo "🔧 Demo Semplificata (SENZA modelli):"
    echo "   python3 gradio_demo_simple.py"
else
    echo ""
    echo "🔧 Demo Semplificata (SENZA modelli):"
    echo "   python3 gradio_demo_simple.py"
    echo ""
    echo "❌ Demo Completa: Non disponibile (mancano modelli addestrati)"
    echo "   Per ottenere modelli: esegui LabelSmoothing_Experiment.ipynb"
fi

echo ""
echo "📚 Per istruzioni dettagliate: leggi README_GRADIO.md"
echo ""

# Chiedi all'utente quale demo avviare
echo "🚀 Vuoi avviare una demo ora? [y/N]"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    if [ "$DEMO_TYPE" = "full" ]; then
        echo "Quale demo vuoi avviare?"
        echo "1) Demo Completa (con modelli addestrati)"
        echo "2) Demo Semplificata (senza modelli)"
        echo "Scegli [1/2]:"
        read -r demo_choice
        
        case $demo_choice in
            1)
                print_status "Avvio demo completa..."
                python3 gradio_demo.py
                ;;
            2)
                print_status "Avvio demo semplificata..."
                python3 gradio_demo_simple.py
                ;;
            *)
                print_status "Avvio demo semplificata (default)..."
                python3 gradio_demo_simple.py
                ;;
        esac
    else
        print_status "Avvio demo semplificata..."
        python3 gradio_demo_simple.py
    fi
else
    echo ""
    print_success "Setup completato! Esegui manualmente:"
    if [ "$DEMO_TYPE" = "full" ]; then
        echo "  python3 gradio_demo.py      # Demo completa"
    fi
    echo "  python3 gradio_demo_simple.py  # Demo semplificata"
    echo ""
fi

print_success "🎮 Buon divertimento con il Pokémon Generator! 🎮"
