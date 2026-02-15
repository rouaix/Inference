#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Compare our RoPE implementation with reference
Check if freqs_cis is computed correctly
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference.p2p_inference import P2PInferenceEngine, precompute_freqs_cis

def test_rope_freqs():
    """Test RoPE frequency computation."""
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))

    head_dim = engine.config.dim // engine.config.n_heads  # 64

    print("="*60)
    print("Testing RoPE frequency computation")
    print("="*60)

    print(f"\nConfig:")
    print(f"  dim: {engine.config.dim}")
    print(f"  n_heads: {engine.config.n_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  rope_freq_base: {engine.config.rope_freq_base}")

    # Compute freqs_cis for first few positions
    freqs_cis = precompute_freqs_cis(head_dim, 10, engine.config.rope_freq_base)

    print(f"\nfreqs_cis shape: {freqs_cis.shape}")
    print(f"Expected: [positions, head_dim/2] = [10, 32]")

    print(f"\nfreqs_cis[0] (position 0):")
    print(f"  First 5 values: {freqs_cis[0, :5]}")
    print(f"  All should be 1+0j for position 0: {np.allclose(freqs_cis[0], 1+0j)}")

    print(f"\nfreqs_cis[1] (position 1):")
    print(f"  First 5 values: {freqs_cis[1, :5]}")

    # Check the frequency formula
    print(f"\n--- Frequency Formula Check ---")
    theta = engine.config.rope_freq_base
    dim = head_dim

    # Our formula
    freqs_ours = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))

    print(f"Frequencies (first 5): {freqs_ours[:5]}")
    print(f"Frequencies (last 5): {freqs_ours[-5:]}")

    # Alternative formula (sometimes used)
    freqs_alt = 1.0 / (theta ** (np.arange(0, dim // 2).astype(np.float32) * 2 / dim))

    print(f"\nAlternative formula (first 5): {freqs_alt[:5]}")
    print(f"Match? {np.allclose(freqs_ours, freqs_alt)}")

    # Check if engine.freqs_cis matches
    print(f"\n--- Engine freqs_cis ---")
    print(f"Shape: {engine.freqs_cis.shape}")
    print(f"Position 0, first 5: {engine.freqs_cis[0, :5]}")
    print(f"Position 1, first 5: {engine.freqs_cis[1, :5]}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_rope_freqs()
