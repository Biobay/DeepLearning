# gradio_demo.py
"""
Demo interattivo con Gradio per il generatore di Pokémon StackGAN
Permette agli utenti di inserire descrizioni testuali e generare sprite Pokémon corrispondenti.
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import sys

# Aggiungi il percorso del progetto
sys.path.append(os.path.abspath('.'))

# Import dei moduli del progetto
import src.config as config
from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1
from src.models.discriminator import DiscriminatorS1
from transformers import AutoTokenizer

class PokemonGenerator:
    def __init__(self, checkpoint_path=None):
        """
        Inizializza il generatore di Pokémon con i modelli pre-addestrati.
        
        Args:
            checkpoint_path (str): Percorso ai checkpoint dei modelli addestrati
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Utilizzo dispositivo: {self.device}")
        
        # Inizializza il tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
        
        # Inizializza i modelli
        self.text_encoder = TextEncoder(
            model_name=config.ENCODER_MODEL_NAME, 
            fine_tune=False  # Per l'inferenza non serve il fine-tuning
        ).to(self.device)
        
        self.generator = GeneratorS1(config).to(self.device)
        
        # Carica i checkpoint se disponibili
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            print("⚠️ Nessun checkpoint trovato. Utilizzo modelli con pesi casuali.")
            print("   Per risultati migliori, addestra prima il modello e specifica il percorso del checkpoint.")
        
        # Imposta i modelli in modalità evaluation
        self.text_encoder.eval()
        self.generator.eval()
    
    def load_checkpoint(self, checkpoint_path):
        """
        Carica i pesi dei modelli dai checkpoint salvati.
        
        Args:
            checkpoint_path (str): Percorso alla directory contenente i checkpoint
        """
        try:
            # Percorsi ai checkpoint
            generator_path = os.path.join(checkpoint_path, "netG_with_smoothing.pth")
            
            if os.path.exists(generator_path):
                self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
                print(f"✅ Checkpoint del generatore caricato da: {generator_path}")
            else:
                print(f"❌ Checkpoint del generatore non trovato: {generator_path}")
                
        except Exception as e:
            print(f"❌ Errore nel caricamento dei checkpoint: {e}")
    
    def generate_pokemon(self, description, num_images=1, seed=None):
        """
        Genera immagini di Pokémon basate sulla descrizione testuale.
        
        Args:
            description (str): Descrizione testuale del Pokémon
            num_images (int): Numero di immagini da generare
            seed (int): Seed per la riproducibilità (opzionale)
            
        Returns:
            List[PIL.Image]: Lista di immagini generate
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        try:
            with torch.no_grad():
                # Tokenizza la descrizione
                inputs = self.tokenizer(
                    description, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=config.TEXT_EMBEDDING_DIM
                ).to(self.device)
                
                # Estrai embedding testuali
                cls_embedding, hidden_states = self.text_encoder(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
                
                images = []
                for i in range(num_images):
                    # Genera rumore casuale
                    noise = torch.randn(1, config.Z_DIM, device=self.device)
                    
                    # Genera l'immagine
                    generated_image, _ = self.generator(cls_embedding, hidden_states, noise)
                    
                    # Converti in formato PIL
                    image = self.tensor_to_pil(generated_image[0])
                    images.append(image)
                
                return images
                
        except Exception as e:
            print(f"❌ Errore nella generazione: {e}")
            # Ritorna un'immagine placeholder in caso di errore
            placeholder = Image.new('RGB', (64, 64), color=(255, 100, 100))
            return [placeholder]
    
    def tensor_to_pil(self, tensor):
        """
        Converte un tensore PyTorch in un'immagine PIL.
        
        Args:
            tensor (torch.Tensor): Tensore dell'immagine
            
        Returns:
            PIL.Image: Immagine convertita
        """
        # Denormalizza l'immagine (da [-1, 1] a [0, 1])
        image = (tensor * 0.5 + 0.5).clamp(0, 1)
        
        # Converti in numpy array
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        # Converti in PIL Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        
        # Ridimensiona per una migliore visualizzazione
        image_pil = image_pil.resize((256, 256), Image.NEAREST)
        
        return image_pil

# Inizializza il generatore
print("🔄 Inizializzazione del generatore di Pokémon...")
pokemon_gen = PokemonGenerator(checkpoint_path="results/label_smoothing_experiment/checkpoints")

def generate_interface(description, num_images, seed, use_seed):
    """
    Funzione principale chiamata dall'interfaccia Gradio.
    
    Args:
        description (str): Descrizione del Pokémon
        num_images (int): Numero di immagini da generare
        seed (int): Seed per la riproducibilità
        use_seed (bool): Se utilizzare il seed specificato
        
    Returns:
        List[PIL.Image]: Immagini generate
    """
    if not description.strip():
        return [Image.new('RGB', (256, 256), color=(200, 200, 200))]
    
    actual_seed = seed if use_seed else None
    images = pokemon_gen.generate_pokemon(description, num_images, actual_seed)
    
    # Gradio si aspetta esattamente il numero di immagini specificate nei outputs
    # Riempi con immagini placeholder se necessario
    while len(images) < 4:
        placeholder = Image.new('RGB', (256, 256), color=(220, 220, 220))
        images.append(placeholder)
    
    return images[:4]  # Limita a 4 immagini massimo per l'interfaccia

# Esempi predefiniti di descrizioni
examples = [
    ["A small electric mouse Pokemon with yellow fur, red cheeks, and a lightning bolt-shaped tail", 2, 42, True],
    ["A blue turtle Pokemon with a hard shell and water abilities", 1, 123, True],
    ["A fire-type dragon Pokemon with orange scales and large wings", 3, 456, True],
    ["A grass-type Pokemon with a bulb on its back and green skin", 2, 789, False],
    ["A psychic-type Pokemon with purple fur and a long tail ending in a ball", 1, 101, True],
    ["A flying-type Pokemon with brown feathers and keen eyes", 2, 202, False]
]

# Crea l'interfaccia Gradio
with gr.Blocks(title="🎮 Pokémon Generator - StackGAN Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 Pokémon Sprite Generator
    
    Benvenuto nel generatore di sprite Pokémon basato su StackGAN! 
    Inserisci una descrizione testuale di un Pokémon e il modello genererà sprite corrispondenti.
    
    ## 🚀 Come usare:
    1. **Scrivi una descrizione** dettagliata del Pokémon che vuoi generare
    2. **Scegli quante immagini** generare (1-4)
    3. **Opzionale**: Imposta un seed per risultati riproducibili
    4. **Clicca "Genera Pokémon"** e aspetta il risultato!
    
    ## 💡 Suggerimenti per descrizioni efficaci:
    - Includi **colori** (es. "giallo", "blu", "rosso")
    - Specifica il **tipo** (es. "elettrico", "acqua", "fuoco")
    - Descrivi **caratteristiche fisiche** (es. "coda a fulmine", "guscio duro", "ali grandi")
    - Aggiungi **dimensioni** (es. "piccolo", "grande", "massiccio")
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input controls
            description_input = gr.Textbox(
                label="📝 Descrizione del Pokémon",
                placeholder="Es: Un piccolo Pokémon topo elettrico con pelliccia gialla, guance rosse e una coda a forma di fulmine",
                lines=3,
                max_lines=5
            )
            
            with gr.Row():
                num_images = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="🎯 Numero di immagini da generare"
                )
                
                use_seed = gr.Checkbox(
                    label="🎲 Usa seed fisso",
                    value=False,
                    info="Per risultati riproducibili"
                )
            
            seed_input = gr.Number(
                label="🌱 Seed (se abilitato)",
                value=42,
                precision=0,
                visible=True
            )
            
            generate_btn = gr.Button(
                "🎮 Genera Pokémon",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            # Output gallery
            output_gallery = gr.Gallery(
                label="🖼️ Pokémon Generati",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=2,
                height="auto",
                object_fit="contain"
            )
    
    # Sezione informazioni
    with gr.Accordion("ℹ️ Informazioni Tecniche", open=False):
        gr.Markdown("""
        ### 🧠 Architettura del Modello
        - **Modello**: StackGAN Stage-I con Cross-Attention
        - **Encoder Testuale**: BERT-mini per l'elaborazione del testo
        - **Generatore**: U-Net con meccanismi di Cross-Attention
        - **Risoluzione Output**: 64x64 pixel (upscalata a 256x256 per visualizzazione)
        - **Tecnica**: Label Smoothing per stabilizzare l'addestramento
        
        ### 📊 Dataset
        - **Fonte**: Dataset Pokémon con descrizioni del Pokédex
        - **Dimensione**: Immagini di sprite Pokémon con relative descrizioni testuali
        
        ### ⚙️ Parametri di Generazione
        - **Dimensione Rumore Latente**: 100
        - **Embedding Testuale**: 256 dimensioni
        - **Batch Size**: Ottimizzato per inferenza singola
        """)
    
    # Esempi predefiniti
    gr.Examples(
        examples=examples,
        inputs=[description_input, num_images, seed_input, use_seed],
        outputs=[output_gallery],
        fn=generate_interface,
        cache_examples=False
    )
    
    # Connetti il pulsante alla funzione
    generate_btn.click(
        fn=generate_interface,
        inputs=[description_input, num_images, seed_input, use_seed],
        outputs=[output_gallery]
    )
    
    # Footer
    gr.Markdown("""
    ---
    🔬 **Progetto di Deep Learning - Text-to-Image Synthesis**  
    Implementazione di StackGAN per la generazione di sprite Pokémon da descrizioni testuali.
    """)

# Avvia l'applicazione
if __name__ == "__main__":
    print("🚀 Avvio dell'interfaccia Gradio...")
    print("📱 L'interfaccia sarà disponibile nel browser.")
    print("🔗 URL locale: http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",  # Solo accesso locale per sicurezza
        server_port=7860,
        share=False,  # Cambia a True se vuoi un link pubblico temporaneo
        show_error=True,
        quiet=False
    )
