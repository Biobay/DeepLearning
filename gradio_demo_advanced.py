#!/usr/bin/env python3
"""
🎮 Pokémon Generator - Demo Gradio Avanzata
Demo interattiva migliorata con configurazione personalizzabile
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os
import sys
from datetime import datetime
import json

# Import configurazione
try:
    from gradio_config import config
except ImportError:
    # Fallback se config non disponibile
    print("⚠️ Configurazione non trovata, uso impostazioni default")
    class config:
        TITLE = "🎮 Pokémon Generator"
        DESCRIPTION = "Genera sprite Pokémon da descrizioni testuali!"
        EXAMPLES = ["A small electric mouse Pokemon with yellow fur"]
        MAX_IMAGES = 4
        DEFAULT_IMAGES = 1
        CUSTOM_CSS = ""
        
        @classmethod
        def get_launch_kwargs(cls):
            return {"server_name": "127.0.0.1", "server_port": 7860}

# Importa modelli se disponibili
try:
    from src.models.stackgan_stage1 import StackGANStage1Generator
    from src.models.text_encoder import BERTTextEncoder
    from src.data.pokemon_dataset import PokemonDataset
    from src.config import Config
    MODELS_AVAILABLE = True
    print("✅ Modelli StackGAN caricati")
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"⚠️ Modelli non disponibili: {e}")

class AdvancedPokemonGenerator:
    """Generatore Pokémon avanzato con fallback intelligente"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_loaded = False
        self.generation_history = []
        
        print(f"🔧 Dispositivo: {self.device}")
        
        if MODELS_AVAILABLE:
            self._load_models()
        else:
            print("🎨 Modalità artistica attivata (senza modelli ML)")
    
    def _load_models(self):
        """Carica i modelli pre-addestrati"""
        try:
            checkpoint_dir = config.CHECKPOINT_DIR
            
            # Cerca i checkpoint più recenti
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                
                if checkpoints:
                    # Ordina per data di modifica
                    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
                    latest_checkpoint = checkpoints[-1]
                    
                    print(f"📂 Carico checkpoint: {latest_checkpoint}")
                    
                    # Carica modelli
                    self.text_encoder = BERTTextEncoder()
                    self.generator = StackGANStage1Generator(
                        text_dim=self.text_encoder.text_dim,
                        noise_dim=100,
                        num_gen_filters=128
                    )
                    
                    # Carica pesi
                    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    
                    if 'generator_state_dict' in checkpoint:
                        self.generator.load_state_dict(checkpoint['generator_state_dict'])
                    if 'text_encoder_state_dict' in checkpoint:
                        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
                    
                    self.generator.to(self.device)
                    self.text_encoder.to(self.device)
                    self.generator.eval()
                    self.text_encoder.eval()
                    
                    self.models_loaded = True
                    print("✅ Modelli caricati con successo!")
                else:
                    print("⚠️ Nessun checkpoint trovato")
            else:
                print("⚠️ Directory checkpoint non trovata")
                
        except Exception as e:
            print(f"❌ Errore caricamento modelli: {e}")
            self.models_loaded = False
    
    def generate_pokemon(self, description, num_images=1, seed=None, use_fixed_seed=False):
        """Genera Pokémon da descrizione"""
        
        # Validazione input
        if not description or len(description.strip()) < 5:
            return self._create_error_image("⚠️ Descrizione troppo breve!\nInserisci almeno 5 caratteri.")
        
        # Gestione seed
        if use_fixed_seed and seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # Log generazione
        self._log_generation(description, num_images, seed, use_fixed_seed)
        
        if self.models_loaded:
            return self._generate_with_models(description, num_images)
        else:
            return self._generate_artistic(description, num_images)
    
    def _generate_with_models(self, description, num_images):
        """Generazione con modelli ML"""
        try:
            with torch.no_grad():
                # Encoding del testo
                text_embeddings = self.text_encoder.encode_text([description] * num_images)
                text_embeddings = text_embeddings.to(self.device)
                
                # Rumore casuale
                noise = torch.randn(num_images, 100, device=self.device)
                
                # Generazione
                fake_images = self.generator(noise, text_embeddings)
                
                # Conversione in PIL
                images = []
                for i in range(num_images):
                    img_tensor = fake_images[i].cpu()
                    img_tensor = (img_tensor + 1) / 2  # Da [-1,1] a [0,1]
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    # Converte in PIL
                    img_array = img_tensor.permute(1, 2, 0).numpy()
                    img_array = (img_array * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    
                    # Ridimensiona per visualizzazione
                    img = img.resize((256, 256), Image.LANCZOS)
                    images.append(img)
                
                return images if len(images) > 1 else images[0]
                
        except Exception as e:
            print(f"❌ Errore generazione ML: {e}")
            return self._create_error_image(f"Errore generazione:\n{str(e)}")
    
    def _generate_artistic(self, description, num_images):
        """Generazione artistica senza modelli"""
        images = []
        
        for i in range(num_images):
            # Analizza descrizione per colori e forme
            colors = self._extract_colors(description)
            shapes = self._extract_shapes(description)
            size_hint = self._extract_size(description)
            
            # Crea immagine artistica
            img = self._create_artistic_pokemon(colors, shapes, size_hint, i)
            images.append(img)
        
        return images if len(images) > 1 else images[0]
    
    def _extract_colors(self, description):
        """Estrae colori dalla descrizione"""
        color_map = {
            'red': '#FF4444', 'blue': '#4444FF', 'green': '#44FF44',
            'yellow': '#FFFF44', 'purple': '#FF44FF', 'orange': '#FF8844',
            'pink': '#FF88CC', 'brown': '#8B4513', 'black': '#333333',
            'white': '#EEEEEE', 'gray': '#888888', 'silver': '#C0C0C0',
            'gold': '#FFD700', 'electric': '#FFFF00', 'fire': '#FF4500',
            'water': '#0077BE', 'grass': '#228B22', 'psychic': '#DA70D6'
        }
        
        found_colors = []
        desc_lower = description.lower()
        
        for color, hex_code in color_map.items():
            if color in desc_lower:
                found_colors.append(hex_code)
        
        return found_colors if found_colors else ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    def _extract_shapes(self, description):
        """Estrae forme dalla descrizione"""
        desc_lower = description.lower()
        shapes = []
        
        if any(word in desc_lower for word in ['round', 'ball', 'sphere']):
            shapes.append('circle')
        if any(word in desc_lower for word in ['long', 'tail', 'snake']):
            shapes.append('line')
        if any(word in desc_lower for word in ['wing', 'triangle']):
            shapes.append('triangle')
        if any(word in desc_lower for word in ['square', 'block']):
            shapes.append('rectangle')
        
        return shapes if shapes else ['circle', 'triangle']
    
    def _extract_size(self, description):
        """Estrae indicazioni di dimensione"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['small', 'tiny', 'little']):
            return 0.7
        elif any(word in desc_lower for word in ['big', 'large', 'huge', 'giant']):
            return 1.5
        else:
            return 1.0
    
    def _create_artistic_pokemon(self, colors, shapes, size_hint, variant):
        """Crea sprite Pokémon artistico"""
        img = Image.new('RGBA', (256, 256), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Colore principale e secondario
        primary_color = colors[0] if colors else '#FF6B6B'
        secondary_color = colors[1] if len(colors) > 1 else '#4ECDC4'
        
        # Dimensioni base scalate
        base_size = int(60 * size_hint)
        
        # Corpo principale (sempre presente)
        body_x = 128 + variant * 10 - 20
        body_y = 140
        draw.ellipse([
            body_x - base_size, body_y - base_size,
            body_x + base_size, body_y + base_size
        ], fill=primary_color, outline='#333333', width=2)
        
        # Testa
        head_x = body_x
        head_y = body_y - base_size + 20
        head_size = int(base_size * 0.8)
        draw.ellipse([
            head_x - head_size, head_y - head_size,
            head_x + head_size, head_y + head_size
        ], fill=primary_color, outline='#333333', width=2)
        
        # Occhi
        eye_size = max(3, int(head_size * 0.15))
        draw.ellipse([
            head_x - 15, head_y - 10,
            head_x - 15 + eye_size, head_y - 10 + eye_size
        ], fill='#000000')
        draw.ellipse([
            head_x + 15 - eye_size, head_y - 10,
            head_x + 15, head_y - 10 + eye_size
        ], fill='#000000')
        
        # Elementi basati su forme estratte
        if 'triangle' in shapes:
            # Orecchie/corna triangolari
            draw.polygon([
                (head_x - 20, head_y - head_size),
                (head_x - 35, head_y - head_size - 20),
                (head_x - 5, head_y - head_size - 5)
            ], fill=secondary_color, outline='#333333')
            draw.polygon([
                (head_x + 20, head_y - head_size),
                (head_x + 35, head_y - head_size - 20),
                (head_x + 5, head_y - head_size - 5)
            ], fill=secondary_color, outline='#333333')
        
        if 'line' in shapes:
            # Coda
            tail_points = [
                (body_x + base_size - 10, body_y + 10),
                (body_x + base_size + 30 + variant * 10, body_y - 20),
                (body_x + base_size + 40 + variant * 10, body_y - 15),
                (body_x + base_size + 35 + variant * 10, body_y + 15),
                (body_x + base_size + 5, body_y + 25)
            ]
            draw.polygon(tail_points, fill=secondary_color, outline='#333333')
        
        # Dettagli aggiuntivi
        if len(colors) > 2:
            # Macchie colorate
            accent_color = colors[2]
            for i in range(2 + variant):
                spot_x = random.randint(body_x - base_size//2, body_x + base_size//2)
                spot_y = random.randint(body_y - base_size//2, body_y + base_size//2)
                spot_size = random.randint(5, 15)
                draw.ellipse([
                    spot_x - spot_size, spot_y - spot_size,
                    spot_x + spot_size, spot_y + spot_size
                ], fill=accent_color)
        
        # Zampe semplici
        paw_size = max(8, int(base_size * 0.3))
        for dx, dy in [(-30, 20), (30, 20), (-20, 45), (20, 45)]:
            draw.ellipse([
                body_x + dx - paw_size, body_y + dy - paw_size,
                body_x + dx + paw_size, body_y + dy + paw_size
            ], fill=primary_color, outline='#333333', width=1)
        
        # Aggiungi testo descrittivo
        try:
            # Prova a caricare font (fallback se non disponibile)
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Scrivi un hint della descrizione
        hint_text = f"Variant {variant + 1}"
        draw.text((10, 10), hint_text, fill='#333333', font=font)
        
        return img
    
    def _create_error_image(self, message):
        """Crea immagine di errore"""
        img = Image.new('RGB', (256, 256), '#FFE6E6')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Centra il testo
        lines = message.split('\n')
        y_start = 128 - (len(lines) * 20) // 2
        
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (256 - text_width) // 2
            draw.text((x, y_start + i * 20), line, fill='#CC0000', font=font)
        
        return img
    
    def _log_generation(self, description, num_images, seed, use_fixed_seed):
        """Log delle generazioni per analytics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'description': description[:100],  # Primi 100 caratteri
            'num_images': num_images,
            'seed': seed if use_fixed_seed else None,
            'models_available': self.models_loaded
        }
        self.generation_history.append(log_entry)
        
        # Mantieni solo le ultime 100 generazioni
        if len(self.generation_history) > 100:
            self.generation_history = self.generation_history[-100:]
    
    def get_generation_stats(self):
        """Statistiche di utilizzo"""
        if not self.generation_history:
            return "Nessuna generazione ancora effettuata."
        
        total = len(self.generation_history)
        with_models = sum(1 for entry in self.generation_history if entry['models_available'])
        
        stats = f"""
        📊 **Statistiche Utilizzo**
        - Generazioni totali: {total}
        - Con modelli ML: {with_models}
        - Modalità artistica: {total - with_models}
        - Ultima generazione: {self.generation_history[-1]['timestamp'][:19]}
        """
        return stats

# Inizializza generatore
generator = AdvancedPokemonGenerator()

def generate_interface(description, num_images, seed, use_fixed_seed):
    """Interfaccia Gradio per generazione"""
    
    if not description.strip():
        return None, "⚠️ Inserisci una descrizione del Pokémon!"
    
    try:
        # Clamp valori
        num_images = max(1, min(num_images, config.MAX_IMAGES))
        
        # Genera immagini
        result = generator.generate_pokemon(
            description=description,
            num_images=num_images,
            seed=seed,
            use_fixed_seed=use_fixed_seed
        )
        
        success_msg = f"✅ Generato{'i' if num_images > 1 else ''} {num_images} Pokémon!"
        if not generator.models_loaded:
            success_msg += " (Modalità artistica)"
        
        return result, success_msg
        
    except Exception as e:
        error_msg = f"❌ Errore durante la generazione: {str(e)}"
        return generator._create_error_image(error_msg), error_msg

def get_stats():
    """Restituisce statistiche di utilizzo"""
    return generator.get_generation_stats()

# === INTERFACCIA GRADIO ===

def create_interface():
    """Crea interfaccia Gradio avanzata"""
    
    with gr.Blocks(
        css=config.CUSTOM_CSS,
        title=config.TITLE,
        theme=gr.themes.Soft()
    ) as demo:
        
        # Header
        gr.Markdown(f"# {config.TITLE}")
        gr.Markdown(config.DESCRIPTION)
        
        # Stato modelli
        model_status = "🤖 **Modelli ML**: Attivi" if generator.models_loaded else "🎨 **Modalità**: Artistica (senza ML)"
        gr.Markdown(model_status)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input principale
                description_input = gr.Textbox(
                    label="📝 Descrizione Pokémon",
                    placeholder=config.TEXT_PLACEHOLDER if hasattr(config, 'TEXT_PLACEHOLDER') else "Scrivi qui...",
                    lines=3,
                    max_lines=5
                )
                
                # Parametri
                with gr.Accordion("⚙️ Parametri Avanzati", open=False):
                    with gr.Row():
                        num_images = gr.Slider(
                            label="🖼️ Numero Immagini",
                            minimum=1,
                            maximum=config.MAX_IMAGES,
                            value=config.DEFAULT_IMAGES,
                            step=1
                        )
                        
                        seed_input = gr.Number(
                            label="🎲 Seed",
                            value=getattr(config, 'DEFAULT_SEED', 42),
                            precision=0
                        )
                    
                    use_fixed_seed = gr.Checkbox(
                        label="📌 Usa seed fisso",
                        value=getattr(config, 'USE_FIXED_SEED', False)
                    )
                
                # Bottoni
                with gr.Row():
                    generate_btn = gr.Button(
                        config.GENERATE_BUTTON if hasattr(config, 'GENERATE_BUTTON') else "Genera",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        config.CLEAR_BUTTON if hasattr(config, 'CLEAR_BUTTON') else "Pulisci",
                        variant="secondary"
                    )
            
            with gr.Column(scale=3):
                # Output
                output_gallery = gr.Gallery(
                    label="🎮 Pokémon Generati",
                    columns=2,
                    height=400,
                    object_fit="contain"
                )
                
                status_output = gr.Textbox(
                    label="📊 Stato",
                    interactive=False,
                    lines=2
                )
        
        # Esempi
        with gr.Accordion("💡 Esempi di Descrizioni", open=False):
            example_texts = config.EXAMPLES if hasattr(config, 'EXAMPLES') else [
                "A small electric mouse Pokemon with yellow fur",
                "A blue turtle Pokemon with water abilities"
            ]
            
            for example in example_texts[:6]:  # Massimo 6 esempi
                gr.Markdown(f"- *{example}*")
        
        # Statistiche (solo se in debug mode)
        if getattr(config, 'DEBUG', False):
            with gr.Accordion("📈 Statistiche Utilizzo", open=False):
                stats_output = gr.Textbox(
                    label="Statistiche",
                    interactive=False,
                    lines=6
                )
                stats_btn = gr.Button("Aggiorna Statistiche")
                stats_btn.click(get_stats, outputs=stats_output)
        
        # Eventi
        generate_btn.click(
            generate_interface,
            inputs=[description_input, num_images, seed_input, use_fixed_seed],
            outputs=[output_gallery, status_output]
        )
        
        clear_btn.click(
            lambda: ("", 1, 42, False, None, ""),
            outputs=[description_input, num_images, seed_input, use_fixed_seed, output_gallery, status_output]
        )
        
        # Esempi cliccabili
        if hasattr(config, 'EXAMPLES') and config.EXAMPLES:
            gr.Examples(
                examples=[[example] for example in config.EXAMPLES[:3]],
                inputs=[description_input],
                label="🎯 Esempi Rapidi"
            )
    
    return demo

if __name__ == "__main__":
    print(f"\n{config.TITLE}")
    print("=" * 50)
    
    # Crea e avvia interfaccia
    demo = create_interface()
    
    # Parametri di lancio
    launch_kwargs = config.get_launch_kwargs() if hasattr(config, 'get_launch_kwargs') else {}
    
    print(f"🚀 Avvio demo su: http://{launch_kwargs.get('server_name', '127.0.0.1')}:{launch_kwargs.get('server_port', 7860)}")
    print("🔗 Premi Ctrl+C per fermare la demo")
    print("=" * 50)
    
    demo.launch(**launch_kwargs)
