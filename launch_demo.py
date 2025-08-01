#!/usr/bin/env python3
"""
Launcher completo per la demo StackGAN
=====================================

Questo script:
1. Verifica e installa le dipendenze necessarie
2. Testa la configurazione
3. Avvia la demo Gradio

Uso: python launch_demo.py
"""

import subprocess
import sys
import os

def install_package(package):
    """Installa un pacchetto Python se non presente"""
    try:
        __import__(package)
        return True
    except ImportError:
        print(f"📦 Installazione {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Errore nell'installazione di {package}")
            return False

def check_and_install_dependencies():
    """Verifica e installa le dipendenze necessarie"""
    print("🔍 Verifica dipendenze...")
    
    required_packages = [
        "gradio",
        "pillow", 
        "numpy"
    ]
    
    all_installed = True
    
    for package in required_packages:
        if not install_package(package):
            all_installed = False
    
    # Verifica pacchetti deep learning (possono essere già installati)
    try:
        import torch
        print("✅ PyTorch disponibile")
    except ImportError:
        print("⚠️  PyTorch non trovato - necessario per la demo")
        all_installed = False
    
    try:
        import transformers
        print("✅ Transformers disponibile")
    except ImportError:
        print("⚠️  Transformers non trovato - necessario per la demo")
        all_installed = False
    
    return all_installed

def run_test():
    """Esegue un test rapido"""
    print("\n🧪 Test rapido della configurazione...")
    try:
        result = subprocess.run([sys.executable, "test_demo.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test superati!")
            return True
        else:
            print("⚠️  Alcuni test falliti, ma la demo potrebbe funzionare comunque")
            print("Output:", result.stdout[-500:])  # Mostra ultimi 500 caratteri
            return True  # Continua comunque
    except subprocess.TimeoutExpired:
        print("⚠️  Test interrotti per timeout")
        return True
    except Exception as e:
        print(f"⚠️  Errore nei test: {e}")
        return True  # Continua comunque

def launch_demo():
    """Avvia la demo Gradio"""
    print("\n🚀 Avvio della demo...")
    print("=" * 60)
    print("🌐 La demo sarà disponibile su:")
    print("   - Local: http://127.0.0.1:7860")
    print("   - Network: http://0.0.0.0:7860")
    print("=" * 60)
    print("💡 Premi Ctrl+C per fermare la demo")
    print("=" * 60)
    
    try:
        # Avvia la demo
        subprocess.run([sys.executable, "gradio_demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo fermata dall'utente")
    except Exception as e:
        print(f"\n❌ Errore nell'avvio della demo: {e}")

def main():
    """Funzione principale"""
    print("🎨 StackGAN Demo Launcher")
    print("=" * 40)
    
    # 1. Verifica dipendenze
    if not check_and_install_dependencies():
        print("\n❌ Impossibile installare tutte le dipendenze necessarie")
        print("💡 Prova a installare manualmente:")
        print("   pip install torch transformers gradio pillow numpy")
        return
    
    # 2. Test opzionale
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-test":
        print("\n⏭️  Test saltati (--skip-test)")
    else:
        run_test()
    
    # 3. Avvia demo
    launch_demo()

if __name__ == "__main__":
    main()
