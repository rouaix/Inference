#!/usr/bin/env python3
"""
Diagnostic tool to compare Python inference engine vs llama.cpp
Identifies where numerical divergence occurs
"""
import numpy as np
import json
from pathlib import Path
from llama_cpp import Llama
import sys

# Import our Python engine
sys.path.insert(0, str(Path(__file__).parent))
from p2p_inference import P2PInferenceEngine, ModelConfig

def compare_embeddings(python_engine, prompt_tokens):
    """Compare embedding outputs."""
    print("\n=== EMBEDDING COMPARISON ===")

    # Get Python embedding
    emb_weight = python_engine.load_tensor("token_embd.weight")
    python_emb = emb_weight[prompt_tokens[0]]

    print(f"Token: {prompt_tokens[0]}")
    print(f"Python embedding shape: {python_emb.shape}")
    print(f"Python embedding stats: mean={python_emb.mean():.6f}, std={python_emb.std():.6f}")
    print(f"Python embedding sample: {python_emb[:10]}")

    # Note: We can't easily extract llama.cpp embeddings without modifying the library
    # So we'll focus on final output comparison

def compare_outputs(python_output, llamacpp_output, prompt):
    """Compare final text outputs."""
    print("\n=== OUTPUT COMPARISON ===")
    print(f"Prompt: '{prompt}'")
    print(f"\nPython output: '{python_output}'")
    print(f"llama.cpp output: '{llamacpp_output}'")

    if python_output.strip() == llamacpp_output.strip():
        print("\n✅ OUTPUTS MATCH!")
        return True
    else:
        print("\n❌ OUTPUTS DIFFER")
        return False

def run_diagnostic(fragments_dir: str, prompt: str = "The capital of France is", max_tokens: int = 5):
    """Run full diagnostic comparison."""
    fragments_path = Path(fragments_dir)

    print("="*60)
    print("DIAGNOSTIC: Python Engine vs llama.cpp")
    print("="*60)

    # 1. Run Python engine
    print("\n[1/3] Running Python inference engine...")
    python_engine = P2PInferenceEngine(fragments_path)
    python_tokens = python_engine.tokenizer.encode(prompt)
    print(f"Encoded tokens: {python_tokens}")

    python_output_tokens = python_engine.generate(prompt, max_tokens=max_tokens)
    python_output = python_engine.tokenizer.decode(python_output_tokens)

    # 2. Run llama.cpp
    print("\n[2/3] Running llama.cpp inference...")
    temp_gguf = Path("temp_diagnostic.gguf")

    # Reconstruct GGUF (reuse bridge logic)
    from p2p_bridge import reconstruct_gguf
    reconstruct_gguf(fragments_path, temp_gguf)

    llm = Llama(
        model_path=str(temp_gguf),
        n_ctx=512,
        n_threads=4,
        verbose=False
    )

    result = llm(prompt, max_tokens=max_tokens, temperature=0.0, echo=False)
    llamacpp_output = result["choices"][0]["text"]

    # 3. Compare
    print("\n[3/3] Comparing outputs...")
    # compare_embeddings(python_engine, python_tokens)  # Skip for now - focus on output
    match = compare_outputs(python_output, llamacpp_output, prompt)

    # Cleanup
    temp_gguf.unlink()

    print("\n" + "="*60)
    if match:
        print("✅ CALIBRATION SUCCESSFUL - Outputs match!")
    else:
        print("❌ CALIBRATION NEEDED - Investigating divergence...")
        print("\nNext steps:")
        print("1. Check RoPE implementation (interleaved vs split)")
        print("2. Verify attention mask and softmax")
        print("3. Compare RMSNorm epsilon values")
        print("4. Check final layer normalization")
    print("="*60)

    return match

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnostic comparison tool")
    parser.add_argument("fragments_dir", help="Directory containing fragments")
    parser.add_argument("--prompt", default="The capital of France is", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=5, help="Max tokens to generate")

    args = parser.parse_args()

    run_diagnostic(args.fragments_dir, args.prompt, args.max_tokens)
