#!/bin/bash

# 🚀 Script di Avvio Training StackGAN Completo
# Esegue automaticamente tutto il pipeline di training

echo "🎮 === STACKGAN TRAINING PIPELINE === 🎮"
echo ""

# Colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

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

# Verifica ambiente
print_status "Verifico ambiente di training..."

# Verifica Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 non trovato!"
    exit 1
fi

# Verifica file necessari
REQUIRED_FILES=(
    "train_complete_pipeline.py"
    "src/config.py"
    "src/data/dataset.py"
    "src/models/encoder.py"
    "src/models/decoder.py"
    "src/models/discriminator.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "File mancante: $file"
        exit 1
    fi
done

print_success "Ambiente verificato!"

# Configurazione training
echo ""
print_status "Configurazione Training:"
echo "📋 Opzioni disponibili:"
echo "   1) Training completo (Stage-I + Stage-II, 50+50 epoche)"
echo "   2) Training veloce (Stage-I + Stage-II, 20+20 epoche)"
echo "   3) Solo Stage-I (50 epoche)"
echo "   4) Solo Stage-II (richiede Stage-I già addestrato)"
echo "   5) Configurazione personalizzata"
echo ""

read -p "Scegli un'opzione [1-5]: " TRAINING_OPTION

case $TRAINING_OPTION in
    1)
        STAGE1_EPOCHS=50
        STAGE2_EPOCHS=50
        TRAINING_TYPE="completo"
        ;;
    2)
        STAGE1_EPOCHS=20
        STAGE2_EPOCHS=20
        TRAINING_TYPE="veloce"
        ;;
    3)
        STAGE1_EPOCHS=50
        STAGE2_EPOCHS=0
        TRAINING_TYPE="solo Stage-I"
        ;;
    4)
        STAGE1_EPOCHS=0
        STAGE2_EPOCHS=50
        TRAINING_TYPE="solo Stage-II"
        ;;
    5)
        echo ""
        read -p "Epoche Stage-I: " STAGE1_EPOCHS
        read -p "Epoche Stage-II: " STAGE2_EPOCHS
        TRAINING_TYPE="personalizzato"
        ;;
    *)
        print_warning "Opzione non valida, uso configurazione completa"
        STAGE1_EPOCHS=50
        STAGE2_EPOCHS=50
        TRAINING_TYPE="completo (default)"
        ;;
esac

# Opzioni avanzate
echo ""
print_status "Opzioni avanzate:"
echo "🏷️ Usare Label Smoothing? [Y/n]"
read -r USE_LABEL_SMOOTHING
if [[ "$USE_LABEL_SMOOTHING" =~ ^[Nn]$ ]]; then
    LABEL_SMOOTHING_FLAG="--no-label-smoothing"
    LABEL_SMOOTHING_STATUS="❌ Disabilitato"
else
    LABEL_SMOOTHING_FLAG=""
    LABEL_SMOOTHING_STATUS="✅ Abilitato"
fi

echo "🎨 Usare Data Augmentation? [Y/n]"
read -r USE_AUGMENTATION
if [[ "$USE_AUGMENTATION" =~ ^[Nn]$ ]]; then
    AUGMENTATION_FLAG="--no-augmentation"
    AUGMENTATION_STATUS="❌ Disabilitata"
else
    AUGMENTATION_FLAG=""
    AUGMENTATION_STATUS="✅ Abilitata"
fi

# Summary configurazione
echo ""
echo "📋 === RIEPILOGO CONFIGURAZIONE ==="
echo "🎯 Tipo training: $TRAINING_TYPE"
echo "📊 Stage-I epoche: $STAGE1_EPOCHS"
echo "📊 Stage-II epoche: $STAGE2_EPOCHS"
echo "🏷️ Label Smoothing: $LABEL_SMOOTHING_STATUS"
echo "🎨 Data Augmentation: $AUGMENTATION_STATUS"
echo ""

# Conferma
echo "Procedere con il training? [y/N]"
read -r CONFIRM

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    print_warning "Training annullato dall'utente"
    exit 0
fi

# Installa dipendenze se necessario
print_status "Verifico dipendenze Python..."
REQUIRED_PACKAGES=("torch" "torchvision" "transformers" "tqdm" "matplotlib" "pillow" "numpy" "pandas")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    print_warning "Dipendenze mancanti: ${MISSING_PACKAGES[*]}"
    echo "Installare automaticamente? [y/N]"
    read -r INSTALL_DEPS
    
    if [[ "$INSTALL_DEPS" =~ ^[Yy]$ ]]; then
        print_status "Installazione dipendenze..."
        
        # Prova prima con requirements.txt se esiste
        if [ -f "requirements.txt" ]; then
            pip3 install -r requirements.txt
        else
            # Installa pacchetti base
            pip3 install torch torchvision transformers tqdm matplotlib pillow numpy pandas
        fi
    else
        print_error "Dipendenze mancanti. Installa manualmente e riprova."
        exit 1
    fi
fi

print_success "Dipendenze verificate!"

# Crea directory risultati se non esiste
mkdir -p results

# Costruisci comando
COMMAND="python3 train_complete_pipeline.py"

if [ $STAGE1_EPOCHS -gt 0 ]; then
    COMMAND="$COMMAND --stage1-epochs $STAGE1_EPOCHS"
fi

if [ $STAGE2_EPOCHS -gt 0 ]; then
    COMMAND="$COMMAND --stage2-epochs $STAGE2_EPOCHS"
fi

COMMAND="$COMMAND $LABEL_SMOOTHING_FLAG $AUGMENTATION_FLAG"

# Avvia training
echo ""
print_success "🚀 Avvio training StackGAN..."
echo "📝 Comando: $COMMAND"
echo ""
echo "⏱️ Il training potrebbe richiedere diverse ore."
echo "🔄 Puoi interrompere con Ctrl+C e riprendere con --resume"
echo ""
echo "=" * 60

# Salva timestamp inizio
echo "$(date): Training avviato" >> training.log

# Esegui training
if eval "$COMMAND"; then
    print_success "🎉 Training completato con successo!"
    echo "$(date): Training completato" >> training.log
    
    # Mostra risultati
    echo ""
    print_status "📂 Risultati salvati in: results/"
    print_status "🖼️ Immagini generate disponibili nelle sottocartelle"
    print_status "💾 Checkpoint salvati per riuso futuro"
    print_status "📊 Log di training disponibili"
    
    echo ""
    print_success "🎮 Ora puoi usare il modello con la demo Gradio!"
    echo "   python3 gradio_demo.py"
    
else
    print_error "❌ Training fallito"
    echo "$(date): Training fallito" >> training.log
    echo ""
    print_status "🔍 Verifica:"
    print_status "   - Log di errore sopra"
    print_status "   - File training.log"
    print_status "   - Disponibilità GPU/memoria"
    exit 1
fi
