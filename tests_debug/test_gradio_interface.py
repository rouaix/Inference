#!/usr/bin/env python3
"""
Test de l'interface Gradio sans démarrer le serveur
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test l'import et le chargement initial
def test_gradio_imports():
    """Test que tous les imports nécessaires fonctionnent."""
    print("Testing Gradio interface imports...")
    
    try:
        import gradio as gr
        print("SUCCESS Gradio imported successfully")
        
        from app import build_app, load_model, scan_models
        print("SUCCESS App functions imported successfully")
        
        return True
    except Exception as e:
        print(f"ERROR Import error: {e}")
        traceback.print_exc()
        return False

def test_model_loading():
    """Test le chargement du modèle."""
    print("\nTesting model loading...")
    
    try:
        from app import load_model
        
        info, log = load_model("models/tinyllama_q8_fragments_v2", verbose=False)
        print("SUCCESS Model loaded successfully")
        print(f"Info: {info[:200]}...")
        
        return True
    except Exception as e:
        print(f"ERROR Model loading error: {e}")
        traceback.print_exc()
        return False

def test_app_building():
    """Test la construction de l'application."""
    print("\nTesting app building...")
    
    try:
        from app import build_app
        
        demo = build_app()
        print("SUCCESS App built successfully")
        print(f"App type: {type(demo)}")
        
        return True
    except Exception as e:
        print(f"ERROR App building error: {e}")
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("GRADIO INTERFACE TEST")
    print("="*60)
    
    success = True
    
    # Test imports
    if not test_gradio_imports():
        success = False
    
    # Test model loading
    if not test_model_loading():
        success = False
    
    # Test app building
    if not test_app_building():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("SUCCESS: All tests passed!")
        print("The Gradio interface should work correctly.")
        print("\nTo start the interface, run:")
        print("python app.py --fragments-dir models/tinyllama_q8_fragments_v2")
    else:
        print("ERROR: Some tests failed.")
        print("Check the error messages above.")
    print("="*60)

if __name__ == "__main__":
    main()