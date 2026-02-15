#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Let's try a MINIMAL test - just check if there's a weight loading issue
Maybe some weights need special handling (dequantization artifacts?)
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference.p2p_inference import P2PInferenceEngine

def check_weight_stats():
    """Check if weight statistics look reasonable."""
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))

    print("="*60)
    print("Checking weight statistics")
    print("="*60)

    weights_to_check = [
        ("token_embd.weight", (32000, 2048)),
        ("blk.0.attn_q.weight", (2048, 2048)),
        ("blk.0.attn_k.weight", (2048, 256)),
        ("blk.0.attn_v.weight", (2048, 256)),
        ("blk.0.attn_output.weight", (2048, 2048)),
        ("blk.0.ffn_gate.weight", (2048, 5632)),
        ("blk.0.ffn_down.weight", (5632, 2048)),
        ("blk.0.ffn_up.weight", (2048, 5632)),
        ("output.weight", (2048, 32000)),
    ]

    for name, expected_shape in weights_to_check:
        w = engine.load_tensor(name)

        # Handle embedding transpose
        if name == "token_embd.weight" and w.shape[0] == 2048:
            w = w.T

        print(f"\n{name}:")
        print(f"  Shape: {w.shape} (expected {expected_shape})")
        print(f"  Dtype: {w.dtype}")
        print(f"  Mean: {w.mean():.6f}, Std: {w.std():.6f}")
        print(f"  Min: {w.min():.6f}, Max: {w.max():.6f}")
        print(f"  Has NaN: {np.isnan(w).any()}")
        print(f"  Has Inf: {np.isinf(w).any()}")

        # Check for suspicious patterns
        if np.abs(w.mean()) > 0.1:
            print(f"  ⚠️  WARNING: Mean is far from zero!")
        if w.std() < 0.001 or w.std() > 1.0:
            print(f"  ⚠️  WARNING: Std is unusual!")

    print("\n" + "="*60)
    print("All weights should have:")
    print("  - Mean close to 0")
    print("  - Std between 0.01 and 0.1")
    print("  - No NaN or Inf values")
    print("="*60)

if __name__ == "__main__":
    check_weight_stats()
