#!/usr/bin/env python3
"""
Compare our implementation with llama.cpp by checking a reference implementation
Let's try using llama-cpp-python to extract logits and compare
"""
import numpy as np
from pathlib import Path
from llama_cpp import Llama
import sys

sys.path.insert(0, str(Path(__file__).parent))
from p2p_inference import P2PInferenceEngine
from p2p_bridge import reconstruct_gguf

def compare_logits_for_token(fragments_dir: str, token_id: int = 1):
    """Compare logits for a single token between Python and llama.cpp."""
    fragments_path = Path(fragments_dir)

    print("="*60)
    print(f"Comparing logits for token {token_id}")
    print("="*60)

    # 1. Get Python logits
    print("\n[1] Python engine...")
    engine = P2PInferenceEngine(fragments_path)

    # Get embedding
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.shape[0] == engine.config.dim:
        w_emb = w_emb.T

    x = w_emb[token_id].reshape(1, -1)

    # Run through layers
    from p2p_inference import LlamaLayer, rms_norm
    for l in range(engine.config.n_layers):
        layer = LlamaLayer(engine, l)
        x, _, _ = layer.forward(x, engine.freqs_cis, None, None, 0)

    # Final norm
    w_final_norm = engine.load_tensor("output_norm.weight")
    if w_final_norm.shape != (engine.config.dim,):
        w_final_norm = engine.load_tensor("norm.weight")

    x = rms_norm(x, w_final_norm, engine.config.norm_eps)

    # Output projection
    w_out = engine.load_tensor("output.weight")
    python_logits = (x @ w_out)[0]  # [vocab]

    print(f"Python logits shape: {python_logits.shape}")
    print(f"Python logits stats: mean={python_logits.mean():.6f}, std={python_logits.std():.6f}")

    # Top 5
    python_top5 = np.argsort(python_logits)[-5:][::-1]
    print("\nPython top 5:")
    for i, idx in enumerate(python_top5):
        token_str = engine.tokenizer.decode([int(idx)])
        print(f"  {i+1}. Token {idx}: '{token_str}' (logit={python_logits[idx]:.4f})")

    # 2. Get llama.cpp logits (if possible)
    print("\n[2] llama.cpp engine...")
    print("Note: llama-cpp-python doesn't expose raw logits easily.")
    print("We can only compare final token predictions.")

    # Reconstruct and run
    temp_gguf = Path("temp_logits_test.gguf")
    reconstruct_gguf(fragments_path, temp_gguf)

    llm = Llama(model_path=str(temp_gguf), n_ctx=512, n_threads=4, verbose=False, logits_all=True)

    # Generate from BOS
    result = llm("", max_tokens=1, temperature=0.0, echo=False)
    generated_token = result["choices"][0]["text"]

    print(f"\nllama.cpp generated: '{generated_token}'")

    # Cleanup
    temp_gguf.unlink()

    print("\n" + "="*60)
    print("If Python top prediction doesn't match llama.cpp output,")
    print("there's a bug in the forward pass implementation.")
    print("="*60)

if __name__ == "__main__":
    compare_logits_for_token("tinyllama_q8_fragments_v2", token_id=1)
