import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import sys

# Aggiungi il percorso del progetto
sys.path.append('.')

from src.models.encoder import TextEncoder
from src.models.decoder import GeneratorS1, GeneratorS2
import src.config as config

class StackGANHuggingFace:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Inizializzazione su: {self.device}")
        
        # Inizializza modelli
        self.text_encoder = TextEncoder(model_name=config.ENCODER_MODEL_NAME, fine_tune=False).to(self.device)
        self.generator_s1 = GeneratorS1(config=config).to(self.device)
        self.generator_s2 = GeneratorS2(config=config).to(self.device)
        
        # Carica checkpoint
        self._load_checkpoints()
        
        # Modalità evaluation
        self.text_encoder.eval()
        self.generator_s1.eval()
        self.generator_s2.eval()
    
    def _load_checkpoints(self):
        """Carica i checkpoint dei modelli"""
        # Stage-I
        s1_path = "results/checkpoints/generator_s1.pth"
        if os.path.exists(s1_path):
            self.generator_s1.load_state_dict(torch.load(s1_path, map_location=self.device))
            self.s1_loaded = True
            print("✅ Stage-I caricato")
        else:
            self.s1_loaded = False
            print("⚠️ Stage-I non trovato")
        
        # Stage-II
        s2_path = "results/checkpoints/stage2/generator_s2.pth"
        if os.path.exists(s2_path):
            try:
                self.generator_s2.load_state_dict(torch.load(s2_path, map_location=self.device))
                self.s2_loaded = True
                print("✅ Stage-II caricato")
            except:
                self.generator_s2.load_state_dict(torch.load(s2_path, map_location=self.device), strict=False)
                self.s2_loaded = True
                print("✅ Stage-II caricato (parziale)")
        else:
            self.s2_loaded = False
            print("⚠️ Stage-II non trovato")
    
    def generate(self, text_prompt, seed=42):
        """Genera Pokemon da testo"""
        if not text_prompt.strip():
            return None, None, "⚠️ Inserisci una descrizione!"
        
        # Imposta seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        try:
            with torch.no_grad():
                # Codifica testo
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL_NAME)
                
                tokens = tokenizer(text_prompt, padding='max_length', 
                                 max_length=config.MAX_TEXT_LENGTH, 
                                 truncation=True, return_tensors='pt')
                
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                
                cls_embedding, hidden_states = self.text_encoder(input_ids, attention_mask)
                
                # Stage-I
                noise = torch.randn(1, config.Z_DIM, device=self.device)
                stage1_img, stage1_mu = self.generator_s1(cls_embedding, hidden_states, noise)
                
                # Stage-II
                if self.s2_loaded:
                    try:
                        stage2_img, _ = self.generator_s2(stage1_img, cls_embedding, stage1_mu)
                    except:
                        import torch.nn.functional as F
                        stage2_img = F.interpolate(stage1_img, size=(215, 215), mode='bilinear')
                else:
                    import torch.nn.functional as F
                    stage2_img = F.interpolate(stage1_img, size=(215, 215), mode='bilinear')
                
                # Converti in PIL
                stage1_pil = self._tensor_to_pil(stage1_img[0])
                stage2_pil = self._tensor_to_pil(stage2_img[0])
                
                info = f"✅ Pokemon generato!\n📝 '{text_prompt}'\n🎲 Seed: {seed}"
                return stage1_pil, stage2_pil, info
        
        except Exception as e:
            return None, None, f"❌ Errore: {str(e)}"
    
    def _tensor_to_pil(self, tensor):
        """Converte tensor in PIL Image"""
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        np_img = tensor.cpu().numpy().transpose(1, 2, 0)
        np_img = (np_img * 255).astype(np.uint8)
        return Image.fromarray(np_img, 'RGB')

# Inizializza modello
model = StackGANHuggingFace()

# Interfaccia Gradio
with gr.Blocks(title="StackGAN Pokemon Generator") as demo:
    gr.Markdown("""
    # 🎮 StackGAN Pokemon Generator
    
    Genera Pokemon unici dalle tue descrizioni testuali!
    
    **Modello:** StackGAN a due stadi (64x64 → 215x215)
    """)
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📝 Descrizione Pokemon",
                placeholder="a red fire dragon pokemon with wings",
                lines=3
            )
            seed_input = gr.Slider(0, 1000, 42, label="🎲 Seed")
            generate_btn = gr.Button("🚀 Genera Pokemon!", variant="primary")
        
        with gr.Column():
            stage1_output = gr.Image(label="Stage-I (64x64)")
            stage2_output = gr.Image(label="Stage-II (215x215)")
            info_output = gr.Markdown()
    
    generate_btn.click(
        model.generate,
        inputs=[text_input, seed_input],
        outputs=[stage1_output, stage2_output, info_output]
    )
    
    # Esempi
    gr.Examples([
        ["a red fire dragon pokemon with wings"],
        ["a blue water pokemon with fins"],
        ["a yellow electric mouse pokemon"],
        ["a green grass pokemon with leaves"]
    ], inputs=[text_input])

if __name__ == "__main__":
    demo.launch()
