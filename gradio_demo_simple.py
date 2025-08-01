# gradio_demo_simple.py
"""
Demo semplificato con Gradio per il generatore di Pokémon
Versione che funziona anche senza modelli pre-addestrati (genera immagini placeholder)
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import os

class SimplePokemonGenerator:
    def __init__(self):
        """Generatore semplificato per demo senza modelli addestrati."""
        self.colors = {
            'electric': (255, 255, 0),   # Giallo
            'water': (0, 150, 255),      # Blu
            'fire': (255, 100, 0),       # Arancione
            'grass': (0, 200, 0),        # Verde
            'psychic': (200, 0, 200),    # Viola
            'flying': (150, 150, 255),   # Azzurro
            'normal': (200, 200, 200),   # Grigio
        }
        
        self.shapes = ['circle', 'oval', 'square', 'triangle']
        
    def generate_pokemon(self, description, num_images=1, seed=None):
        """
        Genera immagini placeholder basate sulla descrizione.
        
        Args:
            description (str): Descrizione del Pokémon
            num_images (int): Numero di immagini da generare
            seed (int): Seed per la riproducibilità
            
        Returns:
            List[PIL.Image]: Lista di immagini generate
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        images = []
        
        for i in range(num_images):
            # Analizza la descrizione per estrarre caratteristiche
            color = self._extract_color(description)
            pokemon_type = self._extract_type(description)
            size = self._extract_size(description)
            
            # Genera un'immagine placeholder
            image = self._create_placeholder_image(description, color, pokemon_type, size)
            images.append(image)
        
        return images
    
    def _extract_color(self, description):
        """Estrae il colore dalla descrizione."""
        description_lower = description.lower()
        
        color_keywords = {
            'yellow': (255, 255, 0),
            'giallo': (255, 255, 0),
            'blue': (0, 150, 255),
            'blu': (0, 150, 255),
            'red': (255, 0, 0),
            'rosso': (255, 0, 0),
            'green': (0, 200, 0),
            'verde': (0, 200, 0),
            'purple': (150, 0, 150),
            'viola': (150, 0, 150),
            'orange': (255, 150, 0),
            'arancione': (255, 150, 0),
            'brown': (150, 75, 0),
            'marrone': (150, 75, 0),
            'black': (50, 50, 50),
            'nero': (50, 50, 50),
            'white': (230, 230, 230),
            'bianco': (230, 230, 230),
        }
        
        for keyword, color in color_keywords.items():
            if keyword in description_lower:
                return color
        
        return (150, 150, 150)  # Grigio di default
    
    def _extract_type(self, description):
        """Estrae il tipo del Pokémon dalla descrizione."""
        description_lower = description.lower()
        
        type_keywords = {
            'electric': 'electric',
            'elettrico': 'electric',
            'water': 'water',
            'acqua': 'water',
            'fire': 'fire',
            'fuoco': 'fire',
            'grass': 'grass',
            'erba': 'grass',
            'psychic': 'psychic',
            'psichico': 'psychic',
            'flying': 'flying',
            'volante': 'flying',
        }
        
        for keyword, pokemon_type in type_keywords.items():
            if keyword in description_lower:
                return pokemon_type
        
        return 'normal'
    
    def _extract_size(self, description):
        """Estrae la dimensione dalla descrizione."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['small', 'piccolo', 'tiny', 'minuscolo']):
            return 'small'
        elif any(word in description_lower for word in ['large', 'grande', 'big', 'massive', 'massiccio']):
            return 'large'
        else:
            return 'medium'
    
    def _create_placeholder_image(self, description, color, pokemon_type, size):
        """Crea un'immagine placeholder stilizzata."""
        # Dimensioni base
        width, height = 256, 256
        
        # Crea l'immagine
        image = Image.new('RGB', (width, height), color=(240, 240, 250))
        draw = ImageDraw.Draw(image)
        
        # Determina la dimensione del Pokémon
        if size == 'small':
            pokemon_size = 60
        elif size == 'large':
            pokemon_size = 120
        else:
            pokemon_size = 90
        
        # Posizione centrale
        center_x, center_y = width // 2, height // 2
        
        # Disegna il corpo principale
        body_rect = [
            center_x - pokemon_size//2,
            center_y - pokemon_size//2,
            center_x + pokemon_size//2,
            center_y + pokemon_size//2
        ]
        draw.ellipse(body_rect, fill=color, outline=(0, 0, 0), width=2)
        
        # Aggiungi caratteristiche basate sul tipo
        if pokemon_type == 'electric':
            # Aggiunge dei "fulmini"
            for i in range(3):
                x = center_x + random.randint(-40, 40)
                y = center_y + random.randint(-40, 40)
                draw.polygon([(x, y), (x+10, y+15), (x+5, y+25)], fill=(255, 255, 100))
        
        elif pokemon_type == 'water':
            # Aggiunge delle "gocce"
            for i in range(4):
                x = center_x + random.randint(-50, 50)
                y = center_y + random.randint(-50, 50)
                draw.ellipse([x, y, x+8, y+12], fill=(100, 200, 255))
        
        elif pokemon_type == 'fire':
            # Aggiunge delle "fiamme"
            for i in range(3):
                x = center_x + random.randint(-30, 30)
                y = center_y - 50 + random.randint(-10, 10)
                draw.polygon([(x, y), (x+8, y-15), (x+16, y), (x+8, y-5)], fill=(255, 200, 0))
        
        # Aggiunge occhi
        eye_size = 8
        left_eye = [center_x - 15, center_y - 10, center_x - 15 + eye_size, center_y - 10 + eye_size]
        right_eye = [center_x + 15 - eye_size, center_y - 10, center_x + 15, center_y - 10 + eye_size]
        draw.ellipse(left_eye, fill=(0, 0, 0))
        draw.ellipse(right_eye, fill=(0, 0, 0))
        
        # Aggiunge una bocca semplice
        mouth_y = center_y + 5
        draw.arc([center_x - 10, mouth_y, center_x + 10, mouth_y + 10], 0, 180, fill=(0, 0, 0), width=2)
        
        # Aggiunge testo con la descrizione troncata
        try:
            # Prova a caricare un font
            font_size = 12
            # Su alcuni sistemi potrebbe non essere disponibile, usa il font di default
            description_short = description[:30] + "..." if len(description) > 30 else description
            
            # Posiziona il testo in basso
            text_y = height - 40
            
            # Calcola la larghezza del testo per centrarlo
            bbox = draw.textbbox((0, 0), description_short)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            # Disegna il testo con sfondo
            draw.rectangle([text_x - 5, text_y - 2, text_x + text_width + 5, text_y + 15], 
                          fill=(255, 255, 255, 200), outline=(0, 0, 0))
            draw.text((text_x, text_y), description_short, fill=(0, 0, 0))
            
        except Exception:
            # Se ci sono problemi con il font, continua senza testo
            pass
        
        return image

# Inizializza il generatore semplificato
pokemon_gen = SimplePokemonGenerator()

def generate_interface(description, num_images, seed, use_seed):
    """Funzione principale per l'interfaccia Gradio."""
    if not description.strip():
        # Immagine vuota se non c'è descrizione
        empty_img = Image.new('RGB', (256, 256), color=(240, 240, 240))
        return [empty_img] * 4
    
    actual_seed = seed if use_seed else None
    images = pokemon_gen.generate_pokemon(description, num_images, actual_seed)
    
    # Riempi con immagini vuote se necessario
    while len(images) < 4:
        empty_img = Image.new('RGB', (256, 256), color=(250, 250, 250))
        images.append(empty_img)
    
    return images[:4]

# Esempi predefiniti
examples = [
    ["A small electric mouse Pokemon with yellow fur and red cheeks", 2, 42, True],
    ["A blue turtle Pokemon with a hard shell and water abilities", 1, 123, True],
    ["A fire-type dragon Pokemon with orange scales", 3, 456, True],
    ["A grass-type Pokemon with green leaves", 2, 789, False],
    ["A large purple psychic Pokemon with strong mental powers", 1, 101, True]
]

# Crea l'interfaccia Gradio
with gr.Blocks(title="🎮 Pokémon Generator Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 Pokémon Sprite Generator (Demo Version)
    
    **Nota**: Questa è una versione dimostrativa che genera immagini placeholder.
    Per ottenere risultati reali, addestra prima il modello StackGAN usando il notebook fornito.
    
    ## 🚀 Come usare:
    1. Inserisci una descrizione del Pokémon
    2. Scegli il numero di immagini da generare
    3. Clicca "Genera Pokémon"
    
    ## 🎨 La demo riconosce:
    - **Colori**: giallo, blu, rosso, verde, viola, arancione, marrone, nero, bianco
    - **Tipi**: elettrico, acqua, fuoco, erba, psichico, volante
    - **Dimensioni**: piccolo, medio, grande
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            description_input = gr.Textbox(
                label="📝 Descrizione del Pokémon",
                placeholder="Es: A small yellow electric mouse Pokemon with red cheeks",
                lines=3
            )
            
            with gr.Row():
                num_images = gr.Slider(1, 4, value=2, step=1, label="🎯 Numero di immagini")
                use_seed = gr.Checkbox(label="🎲 Usa seed fisso", value=False)
            
            seed_input = gr.Number(label="🌱 Seed", value=42, precision=0)
            generate_btn = gr.Button("🎮 Genera Pokémon", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            output_gallery = gr.Gallery(
                label="🖼️ Pokémon Generati (Placeholder)",
                columns=2,
                rows=2,
                height="auto"
            )
    
    with gr.Accordion("ℹ️ Informazioni", open=False):
        gr.Markdown("""
        ### 🔧 Versione Demo
        Questa è una versione semplificata che crea immagini placeholder basate sulla descrizione.
        
        ### 🚀 Per la versione completa:
        1. Addestra il modello StackGAN usando `LabelSmoothing_Experiment.ipynb`
        2. Salva i checkpoint dei modelli addestrati
        3. Usa `gradio_demo.py` con i checkpoint per generazioni reali
        
        ### 📊 Caratteristiche riconosciute:
        - Estrazione automatica di colori, tipi e dimensioni dalla descrizione
        - Generazione di sprite stilizzati con caratteristiche visive appropriate
        - Supporto per seed riproducibili
        """)
    
    gr.Examples(
        examples=examples,
        inputs=[description_input, num_images, seed_input, use_seed],
        outputs=[output_gallery],
        fn=generate_interface
    )
    
    generate_btn.click(
        fn=generate_interface,
        inputs=[description_input, num_images, seed_input, use_seed],
        outputs=[output_gallery]
    )

if __name__ == "__main__":
    print("🚀 Avvio della demo semplificata...")
    print("📱 Interfaccia disponibile su: http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
