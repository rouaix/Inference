#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Minimal test - check if the issue is in how we handle single-token attention
For a single token, attention should essentially be identity (attending to itself)
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference.p2p_inference import P2PInferenceEngine, rms_norm

def test_single_token_attention():
    """Test attention for a single token."""
    engine = P2PInferenceEngine(Path("tinyllama_q8_fragments_v2"))

    print("="*60)
    print("Testing single-token attention (should be ~identity)")
    print("="*60)

    # Get BOS embedding
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.shape[0] == engine.config.dim:
        w_emb = w_emb.T

    x = w_emb[1].reshape(1, -1)  # BOS token, shape [1, 2048]
    print(f"\nInput shape: {x.shape}")
    print(f"Input stats: mean={x.mean():.6f}, std={x.std():.6f}")

    # Layer 0 attention
    print("\n--- Layer 0 Attention ---")

    # 1. Norm
    w_attn_norm = engine.load_tensor("blk.0.attn_norm.weight")
    xn = rms_norm(x, w_attn_norm, engine.config.norm_eps)
    print(f"After norm: mean={xn.mean():.6f}, std={xn.std():.6f}")

    # 2. QKV
    wq = engine.load_tensor("blk.0.attn_q.weight")
    wk = engine.load_tensor("blk.0.attn_k.weight")
    wv = engine.load_tensor("blk.0.attn_v.weight")
    wo = engine.load_tensor("blk.0.attn_output.weight")

    xq = xn @ wq  # [1, 2048]
    xk = xn @ wk  # [1, 256]
    xv = xn @ wv  # [1, 256]

    print(f"Q shape: {xq.shape}, mean={xq.mean():.6f}")
    print(f"K shape: {xk.shape}, mean={xk.mean():.6f}")
    print(f"V shape: {xv.shape}, mean={xv.mean():.6f}")

    # 3. Reshape for heads
    head_dim = engine.config.dim // engine.config.n_heads  # 64
    n_kv_heads = 4

    xq = xq.reshape(1, engine.config.n_heads, head_dim)  # [1, 32, 64]
    xk = xk.reshape(1, n_kv_heads, head_dim)  # [1, 4, 64]
    xv = xv.reshape(1, n_kv_heads, head_dim)  # [1, 4, 64]

    print(f"\nAfter reshape:")
    print(f"  Q: {xq.shape}")
    print(f"  K: {xk.shape}")
    print(f"  V: {xv.shape}")

    # 4. RoPE (at position 0)
    from inference.p2p_inference import apply_rotary_emb
    freqs_cis = engine.freqs_cis[0:1]
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

    print(f"\nAfter RoPE:")
    print(f"  Q: mean={xq.mean():.6f}, std={xq.std():.6f}")
    print(f"  K: mean={xk.mean():.6f}, std={xk.std():.6f}")

    # 5. GQA - repeat K,V
    n_rep = engine.config.n_heads // n_kv_heads  # 8
    keys = np.repeat(xk, n_rep, axis=1)  # [1, 32, 64]
    values = np.repeat(xv, n_rep, axis=1)  # [1, 32, 64]

    print(f"\nAfter GQA repeat:")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")

    # 6. Transpose for attention
    xq = xq.transpose(1, 0, 2)  # [32, 1, 64]
    keys = keys.transpose(1, 0, 2)  # [32, 1, 64]
    values = values.transpose(1, 0, 2)  # [32, 1, 64]

    # 7. Attention scores
    scores = np.matmul(xq, keys.transpose(0, 2, 1)) / np.sqrt(head_dim)
    # [32, 1, 64] @ [32, 64, 1] = [32, 1, 1]

    print(f"\nAttention scores:")
    print(f"  Shape: {scores.shape}")
    print(f"  Values: {scores[0, 0, 0]:.6f} (should be same for all heads)")
    print(f"  All heads same? {np.allclose(scores[:, 0, 0], scores[0, 0, 0])}")

    # 8. Softmax (for single token, should be [1.0])
    from inference.p2p_inference import softmax
    probs = softmax(scores)

    print(f"\nAttention weights:")
    print(f"  Shape: {probs.shape}")
    print(f"  Value: {probs[0, 0, 0]:.6f} (should be 1.0 for single token)")
    print(f"  All 1.0? {np.allclose(probs, 1.0)}")

    # 9. Apply attention
    output = np.matmul(probs, values)  # [32, 1, 64]

    print(f"\nAttention output:")
    print(f"  Shape: {output.shape}")
    print(f"  Should equal values (since probs=1.0)")
    print(f"  Match? {np.allclose(output, values)}")

    # 10. Reshape and project
    output = output.transpose(1, 0, 2).reshape(1, engine.config.dim)  # [1, 2048]
    output = output @ wo

    print(f"\nFinal attention output:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean():.6f}, Std: {output.std():.6f}")

    # 11. Residual
    result = x + output

    print(f"\nAfter residual:")
    print(f"  Mean: {result.mean():.6f}, Std: {result.std():.6f}")

    print("\n" + "="*60)
    print("Key checks:")
    print("1. Attention weights should all be 1.0 for single token")
    print("2. Attention output should equal values")
    print("3. If these fail, there's a bug in attention logic")
    print("="*60)

if __name__ == "__main__":
    test_single_token_attention()
