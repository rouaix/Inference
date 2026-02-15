#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Deep diagnostic - Layer-by-layer comparison
Identifies exact divergence point in forward pass
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from inference.p2p_inference import P2PInferenceEngine, LlamaLayer, rms_norm, apply_rotary_emb

def analyze_single_token(fragments_dir: str, token_id: int = 1):
    """Analyze forward pass for a single token."""
    fragments_path = Path(fragments_dir)

    print("="*60)
    print(f"DEEP DIAGNOSTIC: Analyzing token {token_id}")
    print("="*60)

    engine = P2PInferenceEngine(fragments_path)

    # 1. Embedding
    print("\n[1] EMBEDDING")
    w_emb = engine.load_tensor("token_embd.weight")
    print(f"Embedding weight shape: {w_emb.shape}")

    if w_emb.shape[0] == engine.config.dim and w_emb.shape[1] == engine.config.vocab_size:
        w_emb = w_emb.T
        print(f"Transposed to: {w_emb.shape}")

    x = w_emb[token_id].reshape(1, -1)
    print(f"Embedding output shape: {x.shape}")
    print(f"Stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    print(f"Sample: {x[0, :5]}")

    # 2. Layer 0 Analysis
    print("\n[2] LAYER 0 ANALYSIS")
    layer = LlamaLayer(engine, 0)

    # Attention Norm
    print("\n  [2.1] Attention Norm")
    w_attn_norm = engine.load_tensor("blk.0.attn_norm.weight")
    print(f"  Norm weight shape: {w_attn_norm.shape}")
    print(f"  Norm weight stats: mean={w_attn_norm.mean():.6f}, std={w_attn_norm.std():.6f}")

    x_normed = rms_norm(x, w_attn_norm, engine.config.norm_eps)
    print(f"  After norm: mean={x_normed.mean():.6f}, std={x_normed.std():.6f}")
    print(f"  Sample: {x_normed[0, :5]}")

    # Q, K, V projections
    print("\n  [2.2] Q, K, V Projections")
    w_q = engine.load_tensor("blk.0.attn_q.weight")
    w_k = engine.load_tensor("blk.0.attn_k.weight")
    w_v = engine.load_tensor("blk.0.attn_v.weight")

    print(f"  Q weight shape: {w_q.shape}")
    print(f"  K weight shape: {w_k.shape}")
    print(f"  V weight shape: {w_v.shape}")

    xq = x_normed @ w_q
    xk = x_normed @ w_k
    xv = x_normed @ w_v

    print(f"  Q output shape: {xq.shape}, mean={xq.mean():.6f}, std={xq.std():.6f}")
    print(f"  K output shape: {xk.shape}, mean={xk.mean():.6f}, std={xk.std():.6f}")
    print(f"  V output shape: {xv.shape}, mean={xv.mean():.6f}, std={xv.std():.6f}")

    # RoPE
    print("\n  [2.3] RoPE Application")
    head_dim = engine.config.dim // engine.config.n_heads
    n_kv_heads = getattr(engine.config, 'n_kv_heads', engine.config.n_heads)

    xq_reshaped = xq.reshape(1, 1, engine.config.n_heads, head_dim)
    xk_reshaped = xk.reshape(1, 1, n_kv_heads, head_dim)

    print(f"  Q reshaped: {xq_reshaped.shape}")
    print(f"  K reshaped: {xk_reshaped.shape}")

    # Apply RoPE at position 0
    freqs_cis = engine.freqs_cis[0:1]
    print(f"  freqs_cis shape: {freqs_cis.shape}")
    print(f"  freqs_cis sample: {freqs_cis[0, :3]}")

    xq_rope, xk_rope = apply_rotary_emb(xq_reshaped, xk_reshaped, freqs_cis)

    print(f"  After RoPE Q: mean={xq_rope.mean():.6f}, std={xq_rope.std():.6f}")
    print(f"  After RoPE K: mean={xk_rope.mean():.6f}, std={xk_rope.std():.6f}")

    # Attention scores
    print("\n  [2.4] Attention Scores")
    xq_flat = xq_rope.reshape(1, engine.config.n_heads, head_dim)
    xk_flat = xk_rope.reshape(1, n_kv_heads, head_dim)

    # Repeat K for GQA
    n_rep = engine.config.n_heads // n_kv_heads
    if n_rep > 1:
        xk_flat = np.repeat(xk_flat, n_rep, axis=1)

    scores = (xq_flat @ xk_flat.transpose(0, 2, 1)) / np.sqrt(head_dim)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Scores: mean={scores.mean():.6f}, std={scores.std():.6f}")
    print(f"  Scores sample: {scores[0, 0, :]}")

    # Softmax
    scores_exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

    print(f"  Attention weights: mean={attn_weights.mean():.6f}, std={attn_weights.std():.6f}")
    print(f"  Attention weights sample: {attn_weights[0, 0, :]}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nKey things to check:")
    print("1. Are embedding values reasonable? (should be ~[-0.1, 0.1])")
    print("2. Does RMSNorm produce unit variance? (std should be ~1.0)")
    print("3. Are RoPE frequencies correct? (check freqs_cis values)")
    print("4. Do attention scores make sense? (should be small values)")
    print("5. Do attention weights sum to 1.0?")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep layer diagnostic")
    parser.add_argument("fragments_dir", help="Directory containing fragments")
    parser.add_argument("--token", type=int, default=1, help="Token ID to analyze")

    args = parser.parse_args()

    analyze_single_token(args.fragments_dir, args.token)
