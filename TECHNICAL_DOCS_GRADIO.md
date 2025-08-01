# 🔧 Documentazione Tecnica - Demo Gradio

## Architettura del Sistema

### Panoramica
Il sistema di demo Gradio è strutturato in modo modulare con tre livelli di complessità:

1. **Demo Semplice** (`gradio_demo_simple.py`) - Funziona senza dipendenze ML
2. **Demo Standard** (`gradio_demo.py`) - Richiede modelli pre-addestrati
3. **Demo Avanzata** (`gradio_demo_advanced.py`) - Versione completa con configurazione

### Diagramma dell'Architettura

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  Text Encoder   │    │   Generator     │
│                 │    │   (BERT-mini)   │    │  (StackGAN S1)  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Input Text    │───▶│ • Tokenization  │───▶│ • Cross-Attn    │
│ • Parameters    │    │ • Embeddings    │    │ • U-Net Arch    │
│ • Gallery       │    │ • Context Vec   │    │ • 64x64 Output  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │                                              ▼
┌─────────────────┐                        ┌─────────────────┐
│ Post-Processing │                        │ Noise Vector    │
│                 │                        │   (Random Z)    │
├─────────────────┤                        ├─────────────────┤
│ • Tensor→PIL    │                        │ • Gaussian      │
│ • Upscaling     │                        │ • 100-dim       │
│ • Display       │                        │ • Seed Control  │
└─────────────────┘                        └─────────────────┘
```

## File e Responsabilità

### Core Files

#### `gradio_demo_simple.py`
**Scopo**: Demo base senza dipendenze ML
- **Classe**: `SimplePokemonGenerator`
- **Funzionalità**: Generazione artistica rule-based
- **Dipendenze**: Solo PIL e librerie standard
- **Output**: Sprite stilizzati basati su regole

```python
class SimplePokemonGenerator:
    def generate_pokemon(description, num_images=1):
        # Analisi testuale semplice
        colors = extract_colors(description)
        shapes = extract_shapes(description)
        
        # Generazione artistica
        return create_artistic_sprites(colors, shapes)
```

#### `gradio_demo.py`
**Scopo**: Demo standard con modelli ML
- **Classe**: `PokemonGenerator`
- **Funzionalità**: Generazione con StackGAN addestrato
- **Dipendenze**: PyTorch, Transformers, modelli pre-addestrati
- **Output**: Immagini generate da GAN

```python
class PokemonGenerator:
    def __init__(self):
        self.text_encoder = BERTTextEncoder()
        self.generator = StackGANStage1Generator()
        
    def generate_pokemon(description):
        embeddings = self.text_encoder.encode(description)
        noise = torch.randn(100)
        return self.generator(noise, embeddings)
```

#### `gradio_demo_advanced.py`
**Scopo**: Demo completa con features avanzate
- **Classe**: `AdvancedPokemonGenerator`
- **Funzionalità**: 
  - Fallback intelligente (ML → Artistico)
  - Logging e analytics
  - Configurazione dinamica
  - Error handling robusto

```python
class AdvancedPokemonGenerator:
    def __init__(self):
        self.models_loaded = self._try_load_models()
        self.generation_history = []
        
    def generate_pokemon(self, description, **kwargs):
        if self.models_loaded:
            return self._generate_with_models(description)
        else:
            return self._generate_artistic(description)
```

### Configuration Files

#### `gradio_config.py`
**Scopo**: Configurazione centralizzata
- **Classes**: `GradioConfig`, `DevelopmentConfig`, `ProductionConfig`
- **Responsabilità**: 
  - Parametri UI
  - Configurazione server
  - Stili CSS
  - Esempi predefiniti

```python
class GradioConfig:
    # Server
    HOST = "127.0.0.1"
    PORT = 7860
    
    # UI
    TITLE = "🎮 Pokémon Generator"
    MAX_IMAGES = 4
    
    # Model paths
    CHECKPOINT_DIR = "results/label_smoothing_experiment/checkpoints"
```

### Utility Scripts

#### `setup_gradio.sh`
**Scopo**: Setup automatico con validazione
- Verifica ambiente Python
- Installazione dipendenze
- Rilevamento modelli disponibili
- Guida interattiva

#### `start_demo.sh`
**Scopo**: Launcher rapido
- Selezione automatica demo appropriata
- Gestione errori comuni
- Fallback intelligente

#### `test_gradio.sh`
**Scopo**: Suite di test completa
- Validazione sintassi
- Test import
- Verifica dipendenze
- Test funzionalità

## Flusso di Esecuzione

### 1. Inizializzazione
```python
# Caricamento configurazione
from gradio_config import config

# Inizializzazione generatore
generator = AdvancedPokemonGenerator()

# Setup Gradio interface
demo = create_interface()
```

### 2. Generazione Standard
```
Input Text → Preprocessing → Model Inference → Post-processing → Display
     ↓             ↓              ↓               ↓            ↓
"fire dragon" → Tokenize → Text Embeddings → GAN Generate → PIL Image
```

### 3. Fallback Artistico
```
Input Text → Color Analysis → Shape Detection → Rule-based → Artistic Sprite
     ↓            ↓              ↓              Generation        ↓
"red mouse" → ['#FF0000'] → ['circle'] → Geometric → Stylized PNG
```

## Dipendenze e Requisiti

### Livello 1: Base (Demo Semplice)
```
- Python 3.8+
- PIL/Pillow
- Standard library (random, datetime, json)
```

### Livello 2: Standard (Demo ML)
```
- Livello 1 +
- torch >= 1.9.0
- transformers >= 4.0.0
- gradio >= 3.0.0
```

### Livello 3: Avanzato (Demo Completa)
```
- Livello 2 +
- numpy
- Custom models (src/models/*)
- Pre-trained checkpoints
```

## Struttura Dati

### Input Format
```python
GenerationRequest = {
    'description': str,      # Testo descrittivo
    'num_images': int,       # 1-4 immagini
    'seed': Optional[int],   # Seed per riproducibilità
    'use_fixed_seed': bool   # Flag seed fisso
}
```

### Output Format
```python
GenerationResult = {
    'images': List[PIL.Image],  # Immagini generate
    'status': str,              # Messaggio stato
    'metadata': {               # Metadati generazione
        'model_used': str,
        'generation_time': float,
        'parameters': dict
    }
}
```

### Model Checkpoint Structure
```
results/label_smoothing_experiment/checkpoints/
├── generator_epoch_X.pth
├── discriminator_epoch_X.pth
├── text_encoder_epoch_X.pth
└── optimizer_states_epoch_X.pth

# Checkpoint content:
{
    'epoch': int,
    'generator_state_dict': OrderedDict,
    'discriminator_state_dict': OrderedDict,
    'text_encoder_state_dict': OrderedDict,
    'optimizer_g_state_dict': OrderedDict,
    'optimizer_d_state_dict': OrderedDict,
    'losses': dict,
    'config': dict
}
```

## API Interna

### Text Processing
```python
def extract_colors(description: str) -> List[str]:
    """Estrae colori dal testo usando pattern matching"""
    
def extract_shapes(description: str) -> List[str]:
    """Identifica forme geometriche nella descrizione"""
    
def extract_size_hint(description: str) -> float:
    """Determina scala dimensionale (0.5-2.0)"""
```

### Model Loading
```python
def load_checkpoint(checkpoint_path: str) -> dict:
    """Carica checkpoint con validazione"""
    
def initialize_models(device: str) -> Tuple[TextEncoder, Generator]:
    """Inizializza modelli su device specificato"""
```

### Image Generation
```python
def generate_with_models(
    text_embeddings: torch.Tensor,
    noise: torch.Tensor
) -> torch.Tensor:
    """Generazione ML standard"""
    
def generate_artistic(
    colors: List[str],
    shapes: List[str],
    size_hint: float
) -> PIL.Image:
    """Generazione artistica rule-based"""
```

## Error Handling

### Livelli di Fallback
1. **Model Loading Error** → Fallback a modalità artistica
2. **CUDA Error** → Fallback a CPU
3. **Memory Error** → Riduzione batch size
4. **Import Error** → Demo semplificata

### Exception Management
```python
try:
    # Generazione ML
    result = model_generation(description)
except torch.cuda.OutOfMemoryError:
    # Fallback CPU
    result = cpu_generation(description)
except ImportError:
    # Fallback artistico
    result = artistic_generation(description)
```

## Performance Optimization

### Memory Management
- **Lazy Loading**: Modelli caricati solo quando necessari
- **Gradient Disabled**: `torch.no_grad()` durante inference
- **Garbage Collection**: Pulizia esplicita cache GPU

### Caching Strategy
```python
@lru_cache(maxsize=100)
def encode_text(description: str) -> torch.Tensor:
    """Cache text embeddings for repeated descriptions"""
```

### Batch Processing
```python
def generate_batch(descriptions: List[str]) -> List[PIL.Image]:
    """Generazione batch per efficienza"""
    embeddings = encode_batch(descriptions)
    return decode_batch(model(embeddings))
```

## Deployment Modes

### Development
```bash
GRADIO_MODE=development python gradio_demo_advanced.py
# Host: 127.0.0.1, Debug: True, Share: False
```

### Production
```bash
GRADIO_MODE=production python gradio_demo_advanced.py
# Host: 0.0.0.0, Debug: False, Share: True
```

### Demo
```bash
GRADIO_MODE=demo python gradio_demo_advanced.py
# Host: 0.0.0.0, Max Images: 2, Analytics: True
```

## Monitoring e Analytics

### Generation Logging
```python
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'description_hash': hash(description),
    'num_images': num_images,
    'generation_time': time_elapsed,
    'model_used': 'ml' | 'artistic',
    'success': bool,
    'error': Optional[str]
}
```

### Usage Statistics
- Generazioni totali per sessione
- Ratio ML vs Artistico
- Tempo medio generazione
- Descrizioni più popolari

## Testing Strategy

### Unit Tests
```python
def test_color_extraction():
    assert extract_colors("red dragon") == ['#FF0000']
    
def test_artistic_generation():
    img = generate_artistic(['#FF0000'], ['circle'], 1.0)
    assert isinstance(img, PIL.Image.Image)
```

### Integration Tests
```python
def test_full_pipeline():
    result = generate_pokemon("small blue creature")
    assert result is not None
```

### Load Tests
```python
def test_concurrent_generation():
    # Test multiple simultaneous requests
    pass
```

## Troubleshooting Guide

### Common Issues

#### "ModuleNotFoundError: No module named 'gradio'"
**Soluzione**: `pip install -r requirements_gradio.txt`

#### "CUDA out of memory"
**Soluzione**: Aggiungere `torch.cuda.empty_cache()` o fallback CPU

#### "Checkpoint not found"
**Soluzione**: Usare demo semplice o addestrare modelli

#### "Port already in use"
**Soluzione**: Cambiare porta in configurazione o terminare processo esistente

### Debug Commands
```bash
# Test dipendenze
python -c "import gradio, torch, PIL; print('OK')"

# Test demo semplice
timeout 5s python gradio_demo_simple.py

# Verifica checkpoint
ls -la results/label_smoothing_experiment/checkpoints/

# Test configurazione
python -c "from gradio_config import config; print(config.TITLE)"
```

## Estensioni Future

### Possibili Miglioramenti
1. **StackGAN Stage-II**: Upscaling a 256x256
2. **StyleGAN Integration**: Controllo stile avanzato
3. **Real-time Generation**: WebSocket per streaming
4. **Multi-language Support**: Descrizioni multilingue
5. **Advanced UI**: Editing interattivo immagini

### Plugin Architecture
```python
class GenerationPlugin:
    def process_description(self, text: str) -> str:
        """Pre-processing del testo"""
        
    def post_process_image(self, img: PIL.Image) -> PIL.Image:
        """Post-processing dell'immagine"""
```

---

📚 **Questa documentazione è complementare al README_GRADIO.md per utenti finali**
