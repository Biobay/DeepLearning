import gradio as gr
import torch
import numpy as np
from PIL import Image
import os

# Versione semplificata per HF Spaces
def generate_demo_pokemon(text_prompt, seed=42):
    """
    Demo function - sostituisci con il tuo modello StackGAN
    """
    if not text_prompt.strip():
        return None, None, "⚠️ Inserisci una descrizione!"
    
    # Per ora genera immagini dummy - sostituisci con il tuo modello
    np.random.seed(seed)
    
    # Simula Stage-I (64x64)
    stage1_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    stage1_pil = Image.fromarray(stage1_img)
    
    # Simula Stage-II (215x215)
    stage2_img = np.random.randint(0, 255, (215, 215, 3), dtype=np.uint8)
    stage2_pil = Image.fromarray(stage2_img)
    
    info = f"""
    ✅ Pokemon generato (DEMO)!
    📝 Prompt: '{text_prompt}'
    🎲 Seed: {seed}
    
    ⚠️ Questa è una versione demo.
    Per il modello completo, carica i tuoi checkpoint allenati.
    """
    
    return stage1_pil, stage2_pil, info

# Interfaccia Gradio
with gr.Blocks(title="StackGAN Pokemon Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎮 StackGAN Pokemon Generator
    
    **Genera Pokemon dalle tue descrizioni testuali!**
    
    🚀 **Architettura**: StackGAN a due stadi (64x64 → 215x215)  
    🤖 **Modello**: Text-to-Image con BERT encoding  
    🎯 **Dataset**: Pokemon con descrizioni testuali
    
    ---
    
    ⚠️ **Nota**: Questa è una versione demo. 
    Per usare il modello completo, carica i checkpoint allenati nella cartella del progetto.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="📝 Descrizione Pokemon",
                placeholder="Descrivi il Pokemon che vuoi generare...",
                lines=4,
                value="a red fire dragon pokemon with wings and a long tail"
            )
            
            seed_input = gr.Slider(
                minimum=0,
                maximum=1000,
                value=42,
                step=1,
                label="🎲 Seed (per riproducibilità)"
            )
            
            generate_btn = gr.Button(
                "🚀 Genera Pokemon!",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### 💡 Esempi di prompt:
            - `a blue water pokemon with fins`
            - `a yellow electric mouse pokemon`
            - `a green grass pokemon with leaves`
            - `a purple psychic cat pokemon`
            """)
        
        with gr.Column(scale=2):
            with gr.Row():
                stage1_output = gr.Image(
                    label="🎯 Stage-I Output (64x64)",
                    height=300
                )
                stage2_output = gr.Image(
                    label="🚀 Stage-II Output (215x215)",
                    height=300
                )
            
            info_output = gr.Markdown(
                "👆 Inserisci una descrizione e clicca 'Genera Pokemon!' per iniziare!"
            )
    
    # Collega l'evento
    generate_btn.click(
        fn=generate_demo_pokemon,
        inputs=[text_input, seed_input],
        outputs=[stage1_output, stage2_output, info_output]
    )
    
    # Esempi predefiniti
    gr.Examples(
        examples=[
            ["a red fire dragon pokemon with wings", 42],
            ["a blue water pokemon with fins and bubbles", 123],
            ["a yellow electric mouse pokemon with lightning", 456],
            ["a green grass pokemon with leaves and flowers", 789],
            ["a purple psychic cat pokemon with mystical aura", 101]
        ],
        inputs=[text_input, seed_input],
        label="🎯 Prova questi esempi!"
    )
    
    gr.Markdown("""
    ---
    
    ## 📚 Come usare il modello completo:
    
    1. **Clona questo repository**
    2. **Sostituisci** `generate_demo_pokemon()` con il tuo modello StackGAN
    3. **Carica** i checkpoint allenati in `/checkpoints/`
    4. **Installa** le dipendenze: `pip install -r requirements.txt`
    
    ### 🔧 Struttura del progetto:
    ```
    /src/models/decoder.py     # GeneratorS1, GeneratorS2
    /src/models/encoder.py     # TextEncoder
    /checkpoints/generator_s1.pth
    /checkpoints/stage2/generator_s2.pth
    ```
    
    **📖 Documentazione completa**: [GitHub Repository](https://github.com/Biobay/DeepLearning/)
    """)

# Avvia l'app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
