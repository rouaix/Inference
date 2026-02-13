#!/usr/bin/env python3
"""
Layer-by-layer comparison between Python and llama.cpp implementations
This script will help identify exactly where the numerical divergence occurs
"""
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from p2p_inference import P2PInferenceEngine, LlamaLayer, rms_norm
from p2p_bridge import reconstruct_gguf
from llama_cpp import Llama

def get_python_activations(prompt_tokens: List[int], engine: P2PInferenceEngine) -> Dict:
    """Get activations from Python engine at each layer."""
    activations = {}
    
    # Load weights
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.shape[0] == engine.config.dim:
        w_emb = w_emb.T

    # Prefill: process all prompt tokens
    x = w_emb[prompt_tokens]  # [seq_len, dim]
    activations['input_embeddings'] = x.copy()

    # Run through all layers
    for l in range(engine.config.n_layers):
        layer = LlamaLayer(engine, l)
        x, _, _ = layer.forward(x, engine.freqs_cis, None, None, start_pos=0)
        activations[f'layer_{l}_output'] = x.copy()

    return activations

def get_llama_cpp_activations(prompt_tokens: List[int], temp_gguf: Path) -> Dict:
    """Get activations from llama.cpp - this is tricky as llama.cpp doesn't expose internal activations easily."""
    # Note: This is a limitation - we can't easily get internal activations from llama.cpp
    # We'll use a workaround by running inference and capturing the output
    activations = {}
    
    llm = Llama(model_path=str(temp_gguf), n_ctx=512, n_threads=4, verbose=False)
    
    # Convert tokens to text for llama.cpp
    from p2p_inference import P2PInferenceEngine
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    prompt_text = engine.tokenizer.decode(prompt_tokens)
    
    # Run inference to get the next token
    result = llm(prompt_text, max_tokens=1, temperature=0.0, echo=False)
    generated_token = result["choices"][0]["text"]
    
    activations['final_output'] = generated_token
    
    return activations

def compare_layer_by_layer(prompt_tokens: List[int]):
    """Compare activations layer by layer."""
    print("="*80)
    print("LAYER-BY-LAYER COMPARISON")
    print("="*80)
    
    # Get Python activations
    print("\n[1] Getting Python activations...")
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    python_activations = get_python_activations(prompt_tokens, engine)
    
    # Get llama.cpp reference
    print("\n[2] Getting llama.cpp reference...")
    temp_gguf = Path("temp_layer_comparison.gguf")
    reconstruct_gguf(Path("tinyllama_q8_fragments_v2"), temp_gguf)
    llama_activations = get_llama_cpp_activations(prompt_tokens, temp_gguf)
    
    # Compare final outputs
    print("\n[3] Final output comparison:")
    
    # Get Python's final prediction
    w_out = engine.load_tensor("output.weight")
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (engine.config.dim,):
        w_norm = engine.load_tensor("norm.weight")
    
    x_last = python_activations[f'layer_{engine.config.n_layers-1}_output'][-1:]
    x_last = rms_norm(x_last, w_norm, engine.config.norm_eps)
    python_logits = (x_last @ w_out).flatten()
    python_top_token = np.argmax(python_logits)
    python_top_text = engine.tokenizer.decode([int(python_top_token)])
    
    print(f"Python predicts: '{python_top_text}' (token {python_top_token})")
    print(f"llama.cpp predicts: '{llama_activations['final_output']}'")
    
    if python_top_text.strip() == llama_activations['final_output'].strip():
        print("✅ Predictions match!")
    else:
        print("❌ Predictions differ!")
    
    # Analyze layer activations
    print("\n[4] Layer activation analysis:")
    print(f"Input shape: {python_activations['input_embeddings'].shape}")
    print(f"Input stats: mean={python_activations['input_embeddings'].mean():.6f}, "
          f"std={python_activations['input_embeddings'].std():.6f}")
    
    for l in range(engine.config.n_layers):
        layer_output = python_activations[f'layer_{l}_output']
        print(f"\nLayer {l} output:")
        print(f"  Shape: {layer_output.shape}")
        print(f"  Stats: mean={layer_output.mean():.6f}, std={layer_output.std():.6f}")
        print(f"  Min/Max: {layer_output.min():.6f} / {layer_output.max():.6f}")
        
        # Check for NaN or Inf
        if np.any(np.isnan(layer_output)):
            print(f"  ❌ WARNING: NaN values detected!")
        if np.any(np.isinf(layer_output)):
            print(f"  ❌ WARNING: Inf values detected!")
    
    # Cleanup
    temp_gguf.unlink()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def main():
    # Test with [BOS, Hello] - should predict comma
    prompt_tokens = [1, 15043]
    
    print(f"Testing with prompt tokens: {prompt_tokens}")
    
    # Decode tokens for readability
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    prompt_text = engine.tokenizer.decode(prompt_tokens)
    print(f"Prompt text: '{prompt_text}'")
    print(f"Expected next token: ',' (comma)")
    
    compare_layer_by_layer(prompt_tokens)

if __name__ == "__main__":
    main()