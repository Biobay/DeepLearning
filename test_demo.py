#!/usr/bin/env python3
"""
Test rapido per verificare che la demo Gradio funzioni correttamente
====================================================================

Questo script testa i componenti della demo senza avviare l'interfaccia web.
"""

import os
import sys
import torch

# Aggiungi la directory src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Testa che tutti i moduli si importino correttamente"""
    print("🔍 Test degli import...")
    
    try:
        from src.config import *
        print("✅ Config importato")
        
        from src.models.encoder import TextEncoder
        print("✅ TextEncoder importato")
        
        from src.models.decoder import GeneratorS1
        print("✅ GeneratorS1 importato")
        
        from src.models.generator_s2 import GeneratorS2
        print("✅ GeneratorS2 importato")
        
        return True
    except Exception as e:
        print(f"❌ Errore negli import: {e}")
        return False

def test_model_initialization():
    """Testa l'inizializzazione dei modelli"""
    print("\n🏗️ Test inizializzazione modelli...")
    
    try:
        # Configurazione
        class Config:
            def __init__(self):
                self.TEXT_EMBEDDING_DIM = 256
                self.Z_DIM = 100
                self.DECODER_BASE_CHANNELS = 128
                self.NUM_HEADS = 8
                self.STAGE1_IMAGE_SIZE = 64
                self.STAGE2_IMAGE_SIZE = 215
        
        config = Config()
        
        # Text Encoder
        text_encoder = TextEncoder("prajjwal1/bert-mini")
        print("✅ TextEncoder inizializzato")
        
        # Generator S1
        gen_s1 = GeneratorS1(config)
        print("✅ GeneratorS1 inizializzato")
        
        # Generator S2
        gen_s2 = GeneratorS2(config)
        print("✅ GeneratorS2 inizializzato")
        
        return True
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione: {e}")
        return False

def test_forward_pass():
    """Testa un forward pass completo"""
    print("\n🔄 Test forward pass...")
    
    try:
        # Import necessari
        from src.models.encoder import TextEncoder
        from src.models.decoder import GeneratorS1
        from src.models.generator_s2 import GeneratorS2
        
        # Configurazione
        class Config:
            def __init__(self):
                self.TEXT_EMBEDDING_DIM = 256
                self.Z_DIM = 100
                self.DECODER_BASE_CHANNELS = 128
                self.NUM_HEADS = 8
                self.STAGE1_IMAGE_SIZE = 64
                self.STAGE2_IMAGE_SIZE = 215
        
        config = Config()
        device = torch.device("cpu")  # Usa CPU per i test
        
        # Inizializza modelli
        text_encoder = TextEncoder("prajjwal1/bert-mini")
        gen_s1 = GeneratorS1(config)
        gen_s2 = GeneratorS2(config)
        
        # Metti in modalità eval
        text_encoder.eval()
        gen_s1.eval()
        gen_s2.eval()
        
        with torch.no_grad():
            # 1. Codifica testo
            text_prompt = ["a small blue bird pokemon"]
            text_embedding = text_encoder.encode_text(text_prompt)
            print(f"✅ Text embedding shape: {text_embedding.shape}")
            
            # 2. Genera rumore
            z = torch.randn(1, config.Z_DIM)
            print(f"✅ Noise vector shape: {z.shape}")
            
            # 3. Stage-I
            low_res_image = gen_s1(text_embedding, z)
            print(f"✅ Stage-I output shape: {low_res_image.shape}")
            
            # 4. Stage-II
            high_res_image = gen_s2(low_res_image, text_embedding)
            print(f"✅ Stage-II output shape: {high_res_image.shape}")
            
            # Verifica le dimensioni
            expected_lr_shape = (1, 3, 64, 64)
            expected_hr_shape = (1, 3, 215, 215)
            
            assert low_res_image.shape == expected_lr_shape, f"Shape mismatch: {low_res_image.shape} vs {expected_lr_shape}"
            assert high_res_image.shape == expected_hr_shape, f"Shape mismatch: {high_res_image.shape} vs {expected_hr_shape}"
            
            print("✅ Forward pass completato con successo!")
            return True
            
    except Exception as e:
        print(f"❌ Errore nel forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_import():
    """Testa l'import di Gradio"""
    print("\n🎨 Test Gradio...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio {gr.__version__} disponibile")
        return True
    except ImportError:
        print("❌ Gradio non installato. Esegui: pip install gradio")
        return False

def test_checkpoint_paths():
    """Verifica i percorsi dei checkpoint"""
    print("\n📂 Test percorsi checkpoint...")
    
    from src.config import CHECKPOINT_DIR, CHECKPOINT_DIR_S2
    
    s1_path = os.path.join(CHECKPOINT_DIR, "generator_s1_final.pth")
    s2_path = os.path.join(CHECKPOINT_DIR_S2, "generator_s2_final.pth")
    
    print(f"Stage-I checkpoint: {s1_path}")
    if os.path.exists(s1_path):
        print("✅ Checkpoint Stage-I trovato")
    else:
        print("⚠️  Checkpoint Stage-I non trovato (la demo userà pesi casuali)")
    
    print(f"Stage-II checkpoint: {s2_path}")
    if os.path.exists(s2_path):
        print("✅ Checkpoint Stage-II trovato")
    else:
        print("⚠️  Checkpoint Stage-II non trovato (la demo userà pesi casuali)")
    
    return True

def main():
    """Funzione principale di test"""
    print("🧪 Test StackGAN Demo")
    print("=" * 50)
    
    tests = [
        ("Import dei moduli", test_imports),
        ("Inizializzazione modelli", test_model_initialization),
        ("Forward pass", test_forward_pass),
        ("Import Gradio", test_gradio_import),
        ("Percorsi checkpoint", test_checkpoint_paths)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Errore in {test_name}: {e}")
            results.append((test_name, False))
    
    # Riassunto
    print("\n" + "=" * 50)
    print("📊 RIASSUNTO DEI TEST")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Risultato: {passed}/{total} test superati")
    
    if passed == total:
        print("🎉 Tutti i test superati! La demo è pronta per l'uso.")
        print("\n🚀 Per avviare la demo:")
        print("   python gradio_demo.py")
    else:
        print("⚠️  Alcuni test falliti. Verifica le dipendenze e la configurazione.")
        if passed >= 3:  # Se almeno i test base passano
            print("💡 La demo potrebbe comunque funzionare con pesi casuali.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
