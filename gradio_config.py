# 🎮 Configurazione Demo Gradio
# File di configurazione per personalizzare l'interfaccia Gradio

import os

class GradioConfig:
    """Configurazione centralizzata per la demo Gradio"""
    
    # === CONFIGURAZIONE SERVER ===
    HOST = "127.0.0.1"  # Locale: "127.0.0.1", Pubblico: "0.0.0.0"
    PORT = 7860
    SHARE = False  # True per link pubblico temporaneo
    
    # === CONFIGURAZIONE INTERFACCIA ===
    TITLE = "🎮 Pokémon Generator"
    DESCRIPTION = """
    Genera sprite Pokémon personalizzati usando descrizioni testuali!
    
    Scrivi una descrizione dettagliata del Pokémon che vuoi creare:
    - Include colori specifici (es. "giallo brillante", "blu scuro")
    - Specifica il tipo (elettrico, fuoco, acqua, erba, ecc.)
    - Descrivi caratteristiche fisiche (coda lunga, ali grandi, artigli)
    - Aggiungi dettagli unici (gemma sul petto, segni distintivi)
    """
    
    EXAMPLES = [
        "A small electric mouse Pokemon with yellow fur, red cheeks, and a lightning bolt-shaped tail",
        "A blue turtle Pokemon with a hard brown shell, water abilities, and gentle eyes",
        "A fire-type dragon Pokemon with orange scales, large wings, and flames coming from its mouth",
        "A grass-type Pokemon with a large flower bulb on its back, green skin, and vine-like appendages",
        "A psychic-type Pokemon with purple fur, large ears, and a long tail with a bulb at the end",
        "A rock-type Pokemon with gray stone armor, crystalline spikes, and glowing blue eyes"
    ]
    
    # === PARAMETRI DI GENERAZIONE ===
    MAX_IMAGES = 4
    DEFAULT_IMAGES = 1
    DEFAULT_SEED = 42
    USE_FIXED_SEED = False
    
    # === CONFIGURAZIONE MODELLI ===
    CHECKPOINT_DIR = "results/label_smoothing_experiment/checkpoints"
    MODEL_DEVICE = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
    
    # === CONFIGURAZIONE IMMAGINI ===
    IMAGE_SIZE = 64  # Dimensione generazione
    DISPLAY_SIZE = 256  # Dimensione visualizzazione
    IMAGE_FORMAT = "PNG"
    
    # === CONFIGURAZIONE TESTO ===
    MAX_TEXT_LENGTH = 200
    TEXT_PLACEHOLDER = "Scrivi qui la descrizione del tuo Pokémon ideale..."
    
    # === MESSAGGI INTERFACCIA ===
    GENERATE_BUTTON = "🎨 Genera Pokémon"
    CLEAR_BUTTON = "🗑️ Pulisci"
    
    SUCCESS_MESSAGE = "✅ Pokémon generato con successo!"
    ERROR_MESSAGE = "❌ Errore durante la generazione. Riprova."
    LOADING_MESSAGE = "🔄 Generazione in corso..."
    
    # === STILI CSS PERSONALIZZATI ===
    CUSTOM_CSS = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .gr-button-primary {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .gr-button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .gr-textbox textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .gr-textbox textarea:focus {
        border-color: #FF8E53;
        box-shadow: 0 0 10px rgba(255,142,83,0.3);
    }
    
    .output-image {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .output-image:hover {
        transform: scale(1.05);
    }
    
    .gr-accordion {
        border-radius: 10px;
        overflow: hidden;
    }
    """
    
    # === CONFIGURAZIONE AVANZATA ===
    ENABLE_QUEUE = True
    QUEUE_CONCURRENCY = 2
    MAX_QUEUE_SIZE = 10
    
    # Debug mode
    DEBUG = False
    
    # Analytics (opzionale)
    ENABLE_ANALYTICS = False
    ANALYTICS_KEY = None
    
    @classmethod
    def get_launch_kwargs(cls):
        """Restituisce i parametri per demo.launch()"""
        return {
            "server_name": cls.HOST,
            "server_port": cls.PORT,
            "share": cls.SHARE,
            "debug": cls.DEBUG,
            "enable_queue": cls.ENABLE_QUEUE,
            "max_threads": cls.QUEUE_CONCURRENCY,
        }
    
    @classmethod
    def get_interface_kwargs(cls):
        """Restituisce i parametri per gr.Interface()"""
        return {
            "title": cls.TITLE,
            "description": cls.DESCRIPTION,
            "examples": cls.EXAMPLES[:3],  # Primi 3 esempi
            "css": cls.CUSTOM_CSS,
            "analytics_enabled": cls.ENABLE_ANALYTICS,
        }

# === CONFIGURAZIONI PRESET ===

class DevelopmentConfig(GradioConfig):
    """Configurazione per sviluppo locale"""
    DEBUG = True
    SHARE = False
    HOST = "127.0.0.1"
    PORT = 7860

class ProductionConfig(GradioConfig):
    """Configurazione per produzione"""
    DEBUG = False
    SHARE = True  # Link pubblico
    HOST = "0.0.0.0"
    PORT = 7860
    ENABLE_QUEUE = True
    QUEUE_CONCURRENCY = 1  # Più conservativo

class DemoConfig(GradioConfig):
    """Configurazione per demo pubbliche"""
    SHARE = True
    HOST = "0.0.0.0"
    MAX_IMAGES = 2  # Limita per prestazioni
    ENABLE_ANALYTICS = True

# Seleziona configurazione basata su variabile d'ambiente
CONFIG_MODE = os.getenv("GRADIO_MODE", "development").lower()

if CONFIG_MODE == "production":
    config = ProductionConfig()
elif CONFIG_MODE == "demo":
    config = DemoConfig()
else:
    config = DevelopmentConfig()

# Export per facile importazione
__all__ = ["config", "GradioConfig", "DevelopmentConfig", "ProductionConfig", "DemoConfig"]
