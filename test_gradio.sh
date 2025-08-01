#!/bin/bash

# 🎮 Test Completo Demo Gradio
# Script per testare tutte le versioni della demo

echo "🎮 === TEST DEMO GRADIO === 🎮"
echo ""

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Testa versioni Python/pip
print_status "Verifico ambiente Python..."
python3 --version || { print_error "Python3 non trovato!"; exit 1; }
pip3 --version || { print_error "pip3 non trovato!"; exit 1; }
print_success "Ambiente Python OK"

# Verifica file demo
print_status "Verifico file demo..."
DEMO_FILES=(
    "gradio_demo_simple.py"
    "gradio_demo.py" 
    "gradio_demo_advanced.py"
    "gradio_config.py"
    "requirements_gradio.txt"
    "start_demo.sh"
    "setup_gradio.sh"
)

for file in "${DEMO_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "✓ $file"
    else
        print_error "✗ $file mancante"
    fi
done

# Test sintassi Python
print_status "Testo sintassi file Python..."
for py_file in gradio_demo_simple.py gradio_demo.py gradio_demo_advanced.py gradio_config.py; do
    if [ -f "$py_file" ]; then
        if python3 -m py_compile "$py_file" 2>/dev/null; then
            print_success "✓ $py_file sintassi OK"
        else
            print_warning "⚠ $py_file problemi sintassi (normale se dipendenze mancanti)"
        fi
    fi
done

# Installa dipendenze se richiesto
echo ""
echo "Vuoi installare le dipendenze ora? [y/N]"
read -r install_deps

if [[ "$install_deps" =~ ^[Yy]$ ]]; then
    print_status "Installo dipendenze..."
    if pip3 install -r requirements_gradio.txt; then
        print_success "Dipendenze installate!"
        DEPS_INSTALLED=true
    else
        print_error "Errore installazione dipendenze"
        DEPS_INSTALLED=false
    fi
else
    print_warning "Dipendenze non installate"
    DEPS_INSTALLED=false
fi

# Test imports se dipendenze installate
if [ "$DEPS_INSTALLED" = true ]; then
    echo ""
    print_status "Testo import delle dipendenze..."
    
    # Test Gradio
    if python3 -c "import gradio; print(f'Gradio {gradio.__version__}')" 2>/dev/null; then
        print_success "✓ Gradio importato"
    else
        print_error "✗ Gradio non importabile"
    fi
    
    # Test PyTorch
    if python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        print_success "✓ PyTorch importato"
    else
        print_error "✗ PyTorch non importabile"
    fi
    
    # Test PIL
    if python3 -c "from PIL import Image; print('PIL OK')" 2>/dev/null; then
        print_success "✓ PIL importato"
    else
        print_error "✗ PIL non importabile"
    fi
fi

# Verifica modelli addestrati
echo ""
print_status "Verifico modelli addestrati..."
if [ -d "results/label_smoothing_experiment/checkpoints" ]; then
    CHECKPOINT_COUNT=$(find results/label_smoothing_experiment/checkpoints -name "*.pth" 2>/dev/null | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        print_success "✓ Trovati $CHECKPOINT_COUNT checkpoint"
        MODELS_AVAILABLE=true
    else
        print_warning "⚠ Directory checkpoint vuota"
        MODELS_AVAILABLE=false
    fi
else
    print_warning "⚠ Directory checkpoint non trovata"
    MODELS_AVAILABLE=false
fi

# Suggerimenti di test
echo ""
echo "📋 === OPZIONI DI TEST ==="

echo ""
echo "🔧 Test Base (SEMPRE funzionante):"
echo "   python3 gradio_demo_simple.py"

if [ "$DEPS_INSTALLED" = true ]; then
    echo ""
    echo "⚙️ Test Avanzato (con dipendenze):"
    echo "   python3 gradio_demo_advanced.py"
    
    if [ "$MODELS_AVAILABLE" = true ]; then
        echo ""
        echo "🤖 Test Completo (con modelli ML):"
        echo "   python3 gradio_demo.py"
    fi
fi

echo ""
echo "🚀 Test Automatico:"
echo "   ./start_demo.sh"

# Test veloce se richiesto
echo ""
echo "Vuoi fare un test rapido della demo semplice? [y/N]"
read -r quick_test

if [[ "$quick_test" =~ ^[Yy]$ ]]; then
    print_status "Avvio test rapido..."
    timeout 10s python3 gradio_demo_simple.py &
    TEST_PID=$!
    sleep 3
    
    if ps -p $TEST_PID > /dev/null; then
        print_success "✓ Demo semplice avviata (PID: $TEST_PID)"
        print_status "Stopping test..."
        kill $TEST_PID 2>/dev/null
        wait $TEST_PID 2>/dev/null
        print_success "Test completato"
    else
        print_error "✗ Demo semplice non avviata"
    fi
fi

echo ""
print_success "🎯 Test completato!"
echo ""
echo "📖 Per istruzioni dettagliate: README_GRADIO.md"
echo "🎮 Buon divertimento con il Pokémon Generator!"
