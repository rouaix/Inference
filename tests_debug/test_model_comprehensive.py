#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Test complet du modèle pour valider sa robustesse
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from inference.p2p_inference import P2PInferenceEngine
from p2p_bridge import reconstruct_gguf
from llama_cpp import Llama

def test_python_engine(prompts, max_tokens=5):
    """Test the Python engine with multiple prompts."""
    print("="*80)
    print("TESTING PYTHON ENGINE")
    print("="*80)
    
    engine = P2PInferenceEngine(Path("models/tinyllama_q8_fragments_v2"))
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        
        # Generate tokens - the generate method expects text prompt, not tokens
        try:
            generated_tokens = engine.generate(prompt, max_tokens=max_tokens)
            # Convert token IDs to text
            generated_text = engine.tokenizer.decode([int(t) for t in generated_tokens])
            
            print(f"   Generated: '{generated_text}'")
            print(f"   Full output: '{prompt}{generated_text}'")
            
        except Exception as e:
            print(f"   ERROR: {e}")

def test_llama_cpp_engine(prompts, max_tokens=5):
    """Test llama.cpp engine for comparison."""
    print("\n" + "="*80)
    print("TESTING LLAMA.CPP ENGINE (REFERENCE)")
    print("="*80)
    
    # Reconstruct GGUF
    temp_gguf = Path("temp_comprehensive_test.gguf")
    reconstruct_gguf(Path("models/tinyllama_q8_fragments_v2"), temp_gguf)
    
    llm = Llama(model_path=str(temp_gguf), n_ctx=512, n_threads=4, verbose=False)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Testing prompt: '{prompt}'")
        
        try:
            result = llm(prompt, max_tokens=max_tokens, temperature=0.7, echo=False)
            generated_text = result["choices"][0]["text"]
            
            print(f"   Generated: '{generated_text}'")
            print(f"   Full output: '{prompt}{generated_text}'")
            
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Cleanup
    temp_gguf.unlink()

def main():
    # Test prompts covering various scenarios
    test_prompts = [
        "The capital of France is",
        "Hello, how are you",
        "Once upon a time",
        "The quick brown fox",
        "Artificial intelligence is",
        "In the beginning",
        "To be or not to be",
        "The future of technology"
    ]
    
    print("COMPREHENSIVE MODEL TESTING")
    print("="*80)
    print(f"Testing with {len(test_prompts)} different prompts...")
    print(f"Max tokens per generation: 5")
    
    # Test Python engine
    test_python_engine(test_prompts, max_tokens=5)
    
    # Test llama.cpp engine for comparison
    test_llama_cpp_engine(test_prompts, max_tokens=5)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print("\nIf both engines produce similar outputs, the Python implementation is working correctly.")
    print("Minor differences are expected due to temperature sampling.")

if __name__ == "__main__":
    main()