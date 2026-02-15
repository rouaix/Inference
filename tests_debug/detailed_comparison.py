#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Detailed numerical comparison between Python and llama.cpp
This script focuses on identifying the exact source of numerical divergence
"""
import numpy as np
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from inference.p2p_inference import P2PInferenceEngine, LlamaLayer, rms_norm, apply_rotary_emb
from p2p_bridge import reconstruct_gguf
from llama_cpp import Llama

def debug_python_forward_pass(prompt_tokens: List[int], engine: P2PInferenceEngine, verbose: bool = True) -> Dict:
    """Debug the Python forward pass step by step."""
    debug_info = {}
    
    # Load weights
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.shape[0] == engine.config.dim:
        w_emb = w_emb.T

    # Prefill: process all prompt tokens
    x = w_emb[prompt_tokens]  # [seq_len, dim]
    debug_info['input_embeddings'] = x.copy()
    
    if verbose:
        print(f"Input embeddings shape: {x.shape}")
        print(f"Input embeddings stats: mean={x.mean():.6f}, std={x.std():.6f}")

    # Run through all layers with detailed debugging
    for l in range(engine.config.n_layers):
        layer = LlamaLayer(engine, l)
        layer_info = {}
        
        # Store input to this layer
        layer_input = x.copy()
        layer_info['input'] = layer_input
        
        # Attention mechanism
        if verbose:
            print(f"\n--- Layer {l}: Attention ---")
        
        # Load attention weights
        w_q = layer.w_q
        w_k = layer.w_k
        w_v = layer.w_v
        w_o = layer.w_o
        
        # Q, K, V projections
        q = (x @ w_q.T)
        k = (x @ w_k.T)
        v = (x @ w_v.T)
        
        layer_info['q_proj'] = q.copy()
        layer_info['k_proj'] = k.copy()
        layer_info['v_proj'] = v.copy()
        
        if verbose:
            print(f"Q shape: {q.shape}, stats: mean={q.mean():.6f}, std={q.std():.6f}")
            print(f"K shape: {k.shape}, stats: mean={k.mean():.6f}, std={k.std():.6f}")
            print(f"V shape: {v.shape}, stats: mean={v.mean():.6f}, std={v.std():.6f}")
        
        # Apply RoPE
        q_rope = apply_rotary_emb(q, engine.freqs_cis, start_pos=0)
        k_rope = apply_rotary_emb(k, engine.freqs_cis, start_pos=0)
        
        layer_info['q_rope'] = q_rope.copy()
        layer_info['k_rope'] = k_rope.copy()
        
        # Attention scores
        scores = (q_rope @ k_rope.transpose(0, 1, 3, 2)) / np.sqrt(q_rope.shape[-1])
        
        # Causal mask
        mask = np.full((1, 1, scores.shape[2], scores.shape[3]), -np.inf)
        mask = np.triu(mask, k=1)
        scores = scores + mask
        
        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
        attn_weights = scores_exp / scores_sum
        
        layer_info['attn_weights'] = attn_weights.copy()
        
        # Apply attention
        output = attn_weights @ v
        
        layer_info['attn_output'] = output.copy()
        
        # Final projection
        x = (output @ w_o.T)
        
        layer_info['attn_final'] = x.copy()
        
        # FFN
        if verbose:
            print(f"\n--- Layer {l}: FFN ---")
        
        # SwiGLU
        gate = x @ layer.w_gate.T
        up = x @ layer.w_up.T
        
        layer_info['ffn_gate'] = gate.copy()
        layer_info['ffn_up'] = up.copy()
        
        # SwiGLU activation
        swiglu_output = gate * (1.0 / (1.0 + np.exp(-gate))) * up
        
        layer_info['ffn_swiglu'] = swiglu_output.copy()
        
        # Final projection
        x = (swiglu_output @ layer.w_down.T)
        
        layer_info['ffn_output'] = x.copy()
        
        debug_info[f'layer_{l}'] = layer_info
        
        if verbose:
            print(f"Layer {l} output shape: {x.shape}")
            print(f"Layer {l} output stats: mean={x.mean():.6f}, std={x.std():.6f}")
    
    return debug_info

def compare_with_llama_cpp(prompt_tokens: List[int]):
    """Compare Python implementation with llama.cpp reference."""
    print("="*80)
    print("DETAILED NUMERICAL COMPARISON")
    print("="*80)
    
    # Debug Python forward pass
    print("\n[1] Debugging Python forward pass...")
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    debug_info = debug_python_forward_pass(prompt_tokens, engine, verbose=True)
    
    # Get final prediction from Python
    print("\n[2] Getting Python final prediction...")
    w_out = engine.load_tensor("output.weight")
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (engine.config.dim,):
        w_norm = engine.load_tensor("norm.weight")
    
    x_last = debug_info[f'layer_{engine.config.n_layers-1}']['ffn_output'][-1:]
    x_last = rms_norm(x_last, w_norm, engine.config.norm_eps)
    python_logits = (x_last @ w_out).flatten()
    python_top_token = np.argmax(python_logits)
    python_top_text = engine.tokenizer.decode([int(python_top_token)])
    
    # Get llama.cpp prediction
    print("\n[3] Getting llama.cpp reference...")
    temp_gguf = Path("temp_detailed_comparison.gguf")
    reconstruct_gguf(Path("tinyllama_q8_fragments_v2"), temp_gguf)
    
    llm = Llama(model_path=str(temp_gguf), n_ctx=512, n_threads=4, verbose=False)
    
    # Convert tokens to text for llama.cpp
    prompt_text = engine.tokenizer.decode(prompt_tokens)
    
    # Run inference to get the next token
    result = llm(prompt_text, max_tokens=1, temperature=0.0, echo=False)
    llama_output = result["choices"][0]["text"]
    
    # Cleanup
    temp_gguf.unlink()
    
    # Compare results
    print("\n[4] COMPARISON RESULTS:")
    print(f"Prompt: '{prompt_text}'")
    print(f"Python predicts: '{python_top_text}' (token {python_top_token})")
    print(f"llama.cpp predicts: '{llama_output}'")
    
    if python_top_text.strip() == llama_output.strip():
        print("✅ Predictions match!")
        return True
    else:
        print("❌ Predictions differ!")
        
        # Show top predictions from Python
        print(f"\n[5] Python top 10 predictions:")
        top10_indices = np.argsort(python_logits)[-10:][::-1]
        for i, idx in enumerate(top10_indices):
            token_str = engine.tokenizer.decode([int(idx)])
            print(f"  {i+1}. Token {idx}: '{token_str}' (logit={python_logits[idx]:.4f})")
        
        return False

def analyze_numerical_stability():
    """Analyze potential numerical stability issues."""
    print("\n" + "="*80)
    print("NUMERICAL STABILITY ANALYSIS")
    print("="*80)
    
    # Test with [BOS, Hello]
    prompt_tokens = [1, 15043]
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    
    print("\n[1] Testing with different precision settings...")
    
    # Test float32 vs float64
    original_dtype = np.float32
    
    # Run with float32
    print("\nTesting with float32...")
    debug_info_f32 = debug_python_forward_pass(prompt_tokens, engine, verbose=False)
    
    # Get final logits
    w_out = engine.load_tensor("output.weight")
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (engine.config.dim,):
        w_norm = engine.load_tensor("norm.weight")
    
    x_last_f32 = debug_info_f32[f'layer_{engine.config.n_layers-1}']['ffn_output'][-1:]
    x_last_f32 = rms_norm(x_last_f32, w_norm, engine.config.norm_eps)
    logits_f32 = (x_last_f32 @ w_out).flatten()
    
    print(f"float32 top prediction: token {np.argmax(logits_f32)}")
    
    # Check for numerical issues
    print("\n[2] Checking for numerical issues...")
    
    # Check each layer for potential problems
    for l in range(engine.config.n_layers):
        layer_info = debug_info_f32[f'layer_{l}']
        
        # Check attention weights
        attn_weights = layer_info['attn_weights']
        if np.any(np.isnan(attn_weights)):
            print(f"❌ Layer {l}: NaN in attention weights!")
        if np.any(np.isinf(attn_weights)):
            print(f"❌ Layer {l}: Inf in attention weights!")
        
        # Check for very large values
        if np.max(np.abs(attn_weights)) > 1e6:
            print(f"⚠️  Layer {l}: Very large attention weights (max={np.max(np.abs(attn_weights)):.2e})")
        
        # Check SwiGLU output
        swiglu_output = layer_info['ffn_swiglu']
        if np.any(np.isnan(swiglu_output)):
            print(f"❌ Layer {l}: NaN in SwiGLU output!")
        if np.any(np.isinf(swiglu_output)):
            print(f"❌ Layer {l}: Inf in SwiGLU output!")
    
    print("\n[3] Analysis complete.")

def main():
    # Test with [BOS, Hello] - should predict comma
    prompt_tokens = [1, 15043]
    
    print(f"Testing with prompt tokens: {prompt_tokens}")
    
    # Decode tokens for readability
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))
    prompt_text = engine.tokenizer.decode(prompt_tokens)
    print(f"Prompt text: '{prompt_text}'")
    print(f"Expected next token: ',' (comma)")
    
    # Run comparison
    match = compare_with_llama_cpp(prompt_tokens)
    
    # Run numerical stability analysis
    analyze_numerical_stability()
    
    print("\n" + "="*80)
    if match:
        print("🎉 SUCCESS: Python and llama.cpp predictions match!")
    else:
        print("🔍 INVESTIGATION NEEDED: Predictions differ")
        print("Next steps:")
        print("1. Check attention mechanism implementation")
        print("2. Verify RoPE implementation")
        print("3. Examine SwiGLU activation")
        print("4. Compare weight loading and transposition")
    print("="*80)

if __name__ == "__main__":
    main()