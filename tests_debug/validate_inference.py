#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setup_path  # noqa - adds project root to sys.path
import sys, io
# Force UTF-8 sur la console Windows (évite UnicodeEncodeError avec cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
Validation complète pas-à-pas : moteur Python vs llama.cpp.

Stratégie :
  1. Tests unitaires de chaque opération mathématique (RoPE, RMSNorm, SwiGLU)
  2. Inspection des shapes des tenseurs chargés
  3. Comparaison des logits finaux avec llama.cpp (température=0, greedy)
  4. Rapport de divergence avec causes probables

Utilisation :
    python validate_inference.py <fragments_dir> [--gguf path/to/model.gguf]
    python validate_inference.py <fragments_dir> --gguf model.gguf --prompt "Hello" --tokens 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PASS = "[OK] "
FAIL = "[KO] "
WARN = "[!!] "
INFO = "[..] "
DIV  = "=" * 60


# ============================================================
# 1. Tests unitaires des opérations de base
# ============================================================

def test_rms_norm() -> List[str]:
    """Vérifie que RMSNorm normalise correctement."""
    from inference.p2p_inference import rms_norm
    results = []

    # Cas basique : x = ones, w = ones → sortie = 1.0 partout
    x = np.ones((1, 8), dtype=np.float32)
    w = np.ones(8, dtype=np.float32)
    out = rms_norm(x, w, 1e-5)
    expected = np.ones_like(out)
    if np.allclose(out, expected, atol=1e-4):
        results.append(f"  %s rms_norm(ones, ones) = 1.0 (OK)" % PASS)
    else:
        results.append(f"  {FAIL} rms_norm(ones, ones) = {out.mean():.6f} (attendu 1.0)")

    # Cas scale : x = ones, w = 2*ones → sortie = 2.0
    w2 = np.full(8, 2.0, dtype=np.float32)
    out2 = rms_norm(x, w2, 1e-5)
    if np.allclose(out2, 2.0, atol=1e-4):
        results.append(f"  {PASS} rms_norm(ones, 2*ones) = 2.0 (OK)")
    else:
        results.append(f"  {FAIL} rms_norm(ones, 2*ones) = {out2.mean():.6f} (attendu 2.0)")

    # Variance normalisée : std de la sortie doit être w.std() / sqrt(mean(w^2))
    np.random.seed(42)
    x_rand = np.random.randn(1, 64).astype(np.float32)
    w_rand = np.ones(64, dtype=np.float32)
    out_rand = rms_norm(x_rand, w_rand, 1e-5)
    rms = np.sqrt(np.mean(x_rand ** 2))
    expected_rand = x_rand / rms
    if np.allclose(out_rand, expected_rand, atol=1e-4):
        results.append(f"  {PASS} rms_norm(rand, 1) = x / rms(x) (OK)")
    else:
        diff = np.max(np.abs(out_rand - expected_rand))
        results.append(f"  {FAIL} rms_norm(rand, 1) diverge de x/rms(x), max_diff={diff:.2e}")

    return results


def test_softmax() -> List[str]:
    from inference.p2p_inference import softmax
    results = []

    x = np.array([[1.0, 2.0, 3.0]])
    s = softmax(x)

    if abs(s.sum() - 1.0) < 1e-6:
        results.append(f"  {PASS} softmax: somme = {s.sum():.8f} (≈1.0)")
    else:
        results.append(f"  {FAIL} softmax: somme = {s.sum():.8f} (attendu 1.0)")

    if s[0, 0] < s[0, 1] < s[0, 2]:
        results.append(f"  {PASS} softmax: ordre croissant respecté → {s[0]}")
    else:
        results.append(f"  {FAIL} softmax: ordre incorrect → {s[0]}")

    # Stabilité numérique : grands nombres
    x_large = np.array([[1000.0, 1001.0, 1002.0]])
    s_large = softmax(x_large)
    if not np.any(np.isnan(s_large)) and abs(s_large.sum() - 1.0) < 1e-5:
        results.append(f"  {PASS} softmax: stable sur grands nombres (1000..1002)")
    else:
        results.append(f"  {FAIL} softmax: instable sur grands nombres → {s_large}")

    return results


def test_swiglu() -> List[str]:
    """
    Vérifie le bug de SwiGLU.

    Llama FFN : down( silu(gate) * up )
    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Bug actuel dans p2p_inference.py :
        swiglu(x) = x / (1 + exp(-x)) * x   ← FAUX (= silu(x) * x = x^2 * sigmoid(x))

    Correct :
        silu(x) = x / (1 + exp(-x))           ← x * sigmoid(x)
    """
    from inference.p2p_inference import swiglu
    results = []

    x = np.array([1.0, 2.0, -1.0, 0.0], dtype=np.float32)

    # SiLU correct : x * sigmoid(x)
    silu_correct = x / (1.0 + np.exp(-x))
    # Ce que le code fait actuellement
    silu_code = swiglu(x)

    max_diff = np.max(np.abs(silu_code - silu_correct))

    if max_diff < 1e-5:
        results.append(f"  {PASS} swiglu = silu correct (max_diff={max_diff:.2e})")
    else:
        results.append(
            f"  {FAIL} swiglu BUGUÉ : max_diff={max_diff:.4f}\n"
            f"       Code actuel : {silu_code}\n"
            f"       SiLU correct: {silu_correct}\n"
            f"       Ratio        : {silu_code / (silu_correct + 1e-10)}\n"
            f"       → La fonction retourne silu(x)*x au lieu de silu(x)"
        )

    # Vérification du signe sur x>0
    x_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = swiglu(x_pos)
    correct_out = x_pos / (1.0 + np.exp(-x_pos))
    ratio = out / (correct_out + 1e-10)
    results.append(
        f"  {INFO} Pour x=[1,2,3] : code={out}, correct={correct_out}\n"
        f"       Ratio (doit être 1.0) : {np.round(ratio, 3)}"
    )

    return results


def test_rope_implementation() -> List[str]:
    """
    Vérifie RoPE.
    Deux conventions :
      A) Paires adjacentes : (x0, x1), (x2, x3), ... → complexe
      B) Moitiés : première moitié = réel, deuxième = imaginaire
    GGUF (llama.cpp) utilise la convention (A) avec les poids déjà permutés.
    Notre code utilise (A) → devrait être correct.
    """
    from inference.p2p_inference import precompute_freqs_cis, apply_rotary_emb
    results = []

    head_dim = 64
    n_heads = 2
    seq_len = 4

    freqs = precompute_freqs_cis(head_dim, seq_len * 2)

    xq = np.random.randn(seq_len, n_heads, head_dim).astype(np.float32)
    xk = np.random.randn(seq_len, n_heads, head_dim).astype(np.float32)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

    # 1. Formes conservées
    if xq_out.shape == xq.shape and xk_out.shape == xk.shape:
        results.append(f"  {PASS} RoPE : formes conservées {xq.shape} → {xq_out.shape}")
    else:
        results.append(f"  {FAIL} RoPE : formes modifiées {xq.shape} → {xq_out.shape}")

    # 2. Norme conservée (rotation = isométrie)
    norm_in = np.linalg.norm(xq.reshape(-1))
    norm_out = np.linalg.norm(xq_out.reshape(-1))
    rel_diff = abs(norm_in - norm_out) / (norm_in + 1e-10)
    if rel_diff < 1e-4:
        results.append(f"  {PASS} RoPE : isométrie — norme conservée (diff={rel_diff:.2e})")
    else:
        results.append(f"  {FAIL} RoPE : norme modifiée ({norm_in:.4f} → {norm_out:.4f}, diff={rel_diff:.3%})")

    # 3. Position 0 → fréquences nulles → rotation identité
    xq_pos0 = np.ones((1, 1, head_dim), dtype=np.float32)
    xk_pos0 = np.ones((1, 1, head_dim), dtype=np.float32)
    xq_r0, _ = apply_rotary_emb(xq_pos0, xk_pos0, freqs[:1])
    # À position 0, freqs_cis[0] = exp(0) = 1+0j → rotation nulle
    if np.allclose(xq_r0, xq_pos0, atol=1e-5):
        results.append(f"  {PASS} RoPE : position 0 = identité (rotation nulle)")
    else:
        max_d = np.max(np.abs(xq_r0 - xq_pos0))
        results.append(f"  {WARN} RoPE : position 0 ≠ identité (max_diff={max_d:.4f})")

    # 4. Différentes positions → rotations différentes
    xq_p1 = np.ones((1, 1, head_dim), dtype=np.float32)
    xq_r1, _ = apply_rotary_emb(xq_p1, xq_p1, freqs[1:2])
    diff_01 = np.max(np.abs(xq_r0 - xq_r1))
    if diff_01 > 1e-4:
        results.append(f"  {PASS} RoPE : positions différentes → rotations différentes (diff={diff_01:.4f})")
    else:
        results.append(f"  {FAIL} RoPE : positions 0 et 1 donnent le même résultat !")

    return results


def test_attention_mask() -> List[str]:
    """Vérifie le masque causal de l'attention."""
    from inference.p2p_inference import softmax
    results = []

    seq_len = 4
    # Simulation de scores
    scores = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Masque causal
    mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    masked = scores + mask

    probs = softmax(masked)

    # La première ligne doit être [1, 0, 0, 0]
    expected_row0 = np.array([1.0, 0.0, 0.0, 0.0])
    # La dernière ligne doit être [0.25, 0.25, 0.25, 0.25]
    expected_rowN = np.full(seq_len, 1.0 / seq_len)

    if np.allclose(probs[0], expected_row0, atol=1e-5):
        results.append(f"  {PASS} Masque causal : ligne 0 = [1, 0, 0, 0]")
    else:
        results.append(f"  {FAIL} Masque causal : ligne 0 = {probs[0]} (attendu [1,0,0,0])")

    if np.allclose(probs[-1], expected_rowN, atol=1e-5):
        results.append(f"  {PASS} Masque causal : ligne N = [1/4, 1/4, 1/4, 1/4]")
    else:
        results.append(f"  {FAIL} Masque causal : ligne N = {probs[-1]} (attendu uniforme)")

    # Vérification des NaN (problème fréquent avec -inf)
    if np.any(np.isnan(probs)):
        results.append(f"  {FAIL} Masque causal : NaN détectés dans softmax(mask) !")
    else:
        results.append(f"  {PASS} Masque causal : aucun NaN")

    return results


# ============================================================
# 2. Inspection des shapes des tenseurs
# ============================================================

def inspect_tensor_shapes(engine) -> List[str]:
    """Vérifie que les shapes des tenseurs chargés sont cohérentes avec la config."""
    cfg = engine.config
    results = []

    key_tensors = {
        "token_embd.weight": {
            "expected_shapes": [
                (cfg.vocab_size, cfg.dim),
                (cfg.dim, cfg.vocab_size),
            ],
            "desc": f"embedding [{cfg.vocab_size}, {cfg.dim}] ou [{cfg.dim}, {cfg.vocab_size}]",
        },
        "output.weight": {
            "expected_shapes": [
                (cfg.vocab_size, cfg.dim),
                (cfg.dim, cfg.vocab_size),
            ],
            "desc": f"LM head",
        },
        "output_norm.weight": {
            "expected_shapes": [(cfg.dim,)],
            "desc": f"norm finale [{cfg.dim}]",
        },
        "norm.weight": {
            "expected_shapes": [(cfg.dim,)],
            "desc": f"norm finale (alt) [{cfg.dim}]",
        },
    }

    # Tenseurs par couche (couche 0)
    head_dim = cfg.dim // cfg.n_heads
    kv_dim = cfg.n_kv_heads * head_dim

    layer_tensors = {
        "blk.0.attn_norm.weight": {
            "expected_shapes": [(cfg.dim,)],
            "desc": "attention norm",
        },
        "blk.0.ffn_norm.weight": {
            "expected_shapes": [(cfg.dim,)],
            "desc": "FFN norm",
        },
        "blk.0.attn_q.weight": {
            "expected_shapes": [
                (cfg.dim, cfg.dim),
                (cfg.n_heads * head_dim, cfg.dim),
            ],
            "desc": f"Q proj [{cfg.n_heads*head_dim}, {cfg.dim}] ou transposé",
        },
        "blk.0.attn_k.weight": {
            "expected_shapes": [
                (kv_dim, cfg.dim),
                (cfg.dim, kv_dim),
            ],
            "desc": f"K proj GQA [{kv_dim}, {cfg.dim}]",
        },
        "blk.0.attn_v.weight": {
            "expected_shapes": [
                (kv_dim, cfg.dim),
                (cfg.dim, kv_dim),
            ],
            "desc": f"V proj GQA [{kv_dim}, {cfg.dim}]",
        },
        "blk.0.attn_output.weight": {
            "expected_shapes": [(cfg.dim, cfg.dim)],
            "desc": f"O proj [{cfg.dim}, {cfg.dim}]",
        },
        "blk.0.ffn_gate.weight": {
            "expected_shapes": [
                (cfg.hidden_dim, cfg.dim),
                (cfg.dim, cfg.hidden_dim),
            ],
            "desc": f"FFN gate [{cfg.hidden_dim}, {cfg.dim}]",
        },
        "blk.0.ffn_up.weight": {
            "expected_shapes": [
                (cfg.hidden_dim, cfg.dim),
                (cfg.dim, cfg.hidden_dim),
            ],
            "desc": f"FFN up [{cfg.hidden_dim}, {cfg.dim}]",
        },
        "blk.0.ffn_down.weight": {
            "expected_shapes": [
                (cfg.dim, cfg.hidden_dim),
                (cfg.hidden_dim, cfg.dim),
            ],
            "desc": f"FFN down [{cfg.dim}, {cfg.hidden_dim}]",
        },
    }

    all_tensors = {**key_tensors, **layer_tensors}

    for name, info in all_tensors.items():
        try:
            w = engine.load_tensor(name)
            shape = w.shape
            ok = any(shape == tuple(s) for s in info["expected_shapes"])
            symbol = PASS if ok else WARN
            results.append(
                f"  {symbol} {name:<45} shape={shape}  ({info['desc']})"
            )
            if not ok:
                results.append(
                    f"       Shapes attendues : {info['expected_shapes']}"
                )
        except Exception as e:
            results.append(f"  {FAIL} {name:<45} ERREUR: {e}")

    return results


def detect_matmul_order(engine) -> List[str]:
    """
    Détermine si les poids sont stockés en [out, in] ou [in, out].
    Pour une projection linéaire : y = x @ w  (si w=[in,out]) ou y = x @ w.T (si w=[out,in])

    Méthode : on vérifie si `x @ w` produit la bonne dimension de sortie
    pour la projection Q (qui doit donner [seq, n_heads*head_dim]).
    """
    cfg = engine.config
    head_dim = cfg.dim // cfg.n_heads
    results = []

    try:
        w_q = engine.load_tensor("blk.0.attn_q.weight")
        x = np.ones((1, cfg.dim), dtype=np.float32)

        # Test direct : x @ wq → [1, ?]
        try:
            out_direct = x @ w_q
            results.append(
                f"  {INFO} x @ wq : [{x.shape}] @ [{w_q.shape}] → [{out_direct.shape}]"
            )
            if out_direct.shape[-1] == cfg.n_heads * head_dim:
                results.append(
                    f"  {PASS} Multiplication directe (x @ w) correcte → dim={out_direct.shape[-1]}"
                )
            else:
                results.append(
                    f"  {WARN} Multiplication directe donne dim={out_direct.shape[-1]}, "
                    f"attendu {cfg.n_heads * head_dim}"
                )
        except Exception as e:
            results.append(f"  {FAIL} x @ wq impossible : {e}")

        # Test transposé : x @ wq.T
        try:
            out_T = x @ w_q.T
            results.append(
                f"  {INFO} x @ wq.T : [{x.shape}] @ [{w_q.T.shape}] → [{out_T.shape}]"
            )
            if out_T.shape[-1] == cfg.n_heads * head_dim:
                results.append(
                    f"  {PASS} Multiplication transposée (x @ w.T) correcte → dim={out_T.shape[-1]}"
                )
        except Exception as e:
            results.append(f"  {INFO} x @ wq.T impossible : {e}")

    except Exception as e:
        results.append(f"  {FAIL} Impossible de charger wq : {e}")

    return results


# ============================================================
# 3. Comparaison avec llama.cpp
# ============================================================

def get_llamacpp_logits(
    gguf_path: str,
    prompt_tokens: List[int],
    n_ctx: int = 512,
) -> Optional[np.ndarray]:
    """
    Obtient les logits du prochain token via llama.cpp.
    Retourne un vecteur [vocab_size] ou None si llama_cpp non disponible.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return None

    print(f"  Chargement llama.cpp : {gguf_path}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=4,
        logits_all=True,
        verbose=False,
    )

    # eval + extraction des logits du dernier token
    llm.eval(prompt_tokens)
    logits = np.array(llm.eval_logits[-1], dtype=np.float32)
    return logits


def compare_logits(
    python_logits: np.ndarray,
    ref_logits: np.ndarray,
    tokenizer,
    top_n: int = 10,
) -> List[str]:
    """Compare les top-N tokens prédits par les deux moteurs."""
    results = []

    # Cosine similarity
    dot = float(np.dot(python_logits, ref_logits))
    norm_p = float(np.linalg.norm(python_logits))
    norm_r = float(np.linalg.norm(ref_logits))
    cos_sim = dot / (norm_p * norm_r + 1e-10)

    # L2 relative error
    l2_err = float(np.linalg.norm(python_logits - ref_logits)) / (norm_r + 1e-10)

    # Top-N tokens
    top_python = np.argsort(-python_logits)[:top_n]
    top_ref    = np.argsort(-ref_logits)[:top_n]

    top1_match = top_python[0] == top_ref[0]
    overlap = len(set(top_python.tolist()) & set(top_ref.tolist()))

    results.append(f"  Cosine similarity : {cos_sim:.6f}  (1.0 = parfait)")
    results.append(f"  L2 relative error : {l2_err:.6f}  (0.0 = parfait)")
    results.append(f"  Top-1 match       : {'✅ OUI' if top1_match else '❌ NON'}")
    results.append(f"  Overlap top-{top_n}   : {overlap}/{top_n}")
    results.append("")

    # Tableau comparatif
    results.append(f"  {'Rang':<5} {'Python':<30} {'llama.cpp':<30} {'Match'}")
    results.append(f"  {'-'*4} {'-'*29} {'-'*29} {'-'*5}")

    for i in range(top_n):
        tid_p = int(top_python[i])
        tid_r = int(top_ref[i])
        try:
            w_p = tokenizer.decode([tid_p])[:20]
        except Exception:
            w_p = f"<{tid_p}>"
        try:
            w_r = tokenizer.decode([tid_r])[:20]
        except Exception:
            w_r = f"<{tid_r}>"

        lp = float(python_logits[tid_p])
        lr = float(ref_logits[tid_r])
        match_sym = "✅" if tid_p == tid_r else "  "
        results.append(
            f"  {i+1:<5} {repr(w_p):<20} {lp:+8.3f}    "
            f"{repr(w_r):<20} {lr:+8.3f}   {match_sym}"
        )

    return results, cos_sim, top1_match


# ============================================================
# 4. Forward pass Python pas-à-pas
# ============================================================

def python_forward_single(engine, token_id: int) -> Tuple[np.ndarray, dict]:
    """
    Exécute un forward pass pour un seul token.
    Retourne (logits, stats_par_couche).
    """
    from inference.p2p_inference import LlamaLayer, rms_norm

    stats = {}
    cfg = engine.config

    # Embedding
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.ndim == 2 and w_emb.shape[0] == cfg.dim and w_emb.shape[1] == cfg.vocab_size:
        w_emb = w_emb.T
    x = w_emb[token_id].reshape(1, -1) if token_id < w_emb.shape[0] else np.zeros((1, cfg.dim), dtype=np.float32)

    stats["embedding"] = {
        "shape": x.shape,
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "sample": x[0, :5].tolist(),
    }

    # Layers
    for l in range(cfg.n_layers):
        layer = LlamaLayer(engine, l)
        x_prev = x.copy()
        x, _, _ = layer.forward(x, engine.freqs_cis, None, None, 0)

        if l < 3 or l == cfg.n_layers - 1:
            delta = float(np.max(np.abs(x - x_prev)))
            stats[f"layer_{l}"] = {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "max_delta": delta,
                "has_nan": bool(np.any(np.isnan(x))),
                "has_inf": bool(np.any(np.isinf(x))),
            }

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            stats[f"layer_{l}_EXPLODED"] = True
            print(f"  {FAIL} NaN/Inf détecté à la couche {l} !")
            break

    # Norme finale
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (cfg.dim,):
        w_norm = engine.load_tensor("norm.weight")
    x = rms_norm(x, w_norm, cfg.norm_eps)

    stats["final_norm"] = {
        "mean": float(x.mean()),
        "std": float(x.std()),
    }

    # LM head
    w_out = engine.load_tensor("output.weight")
    logits = x @ w_out
    logits = logits.flatten()

    stats["logits"] = {
        "shape": logits.shape[0],
        "mean": float(logits.mean()),
        "std": float(logits.std()),
        "min": float(logits.min()),
        "max": float(logits.max()),
        "top1_id": int(np.argmax(logits)),
    }

    return logits, stats


def print_forward_stats(stats: dict) -> None:
    """Affiche les statistiques de chaque étape du forward pass."""
    for key, val in stats.items():
        if key == "logits":
            print(
                f"  {INFO} logits     : vocab={val['shape']}, "
                f"mean={val['mean']:+.4f}, std={val['std']:.4f}, "
                f"range=[{val['min']:+.1f}, {val['max']:+.1f}], "
                f"top1_id={val['top1_id']}"
            )
        elif key == "embedding":
            print(
                f"  {INFO} embedding  : mean={val['mean']:+.4f}, std={val['std']:.4f}, "
                f"sample={[round(v, 4) for v in val['sample']]}"
            )
        elif key.startswith("layer_") and not key.endswith("EXPLODED"):
            layer_id = key.split("_")[1]
            nan_warn = f" {FAIL} NaN!" if val["has_nan"] else ""
            inf_warn = f" {FAIL} Inf!" if val["has_inf"] else ""
            print(
                f"  {INFO} couche {layer_id:<3} : mean={val['mean']:+.4f}, "
                f"std={val['std']:.4f}, delta={val['max_delta']:.4f}"
                f"{nan_warn}{inf_warn}"
            )
        elif key == "final_norm":
            print(
                f"  {INFO} final_norm : mean={val['mean']:+.4f}, std={val['std']:.4f}"
            )


# ============================================================
# 5. Rapport principal
# ============================================================

def run_full_validation(
    fragments_dir: str,
    gguf_path: Optional[str] = None,
    prompt: str = "The capital of France is",
    max_tokens: int = 1,
) -> None:
    from inference.p2p_inference import P2PInferenceEngine

    print(DIV)
    print("  VALIDATION COMPLÈTE — Moteur Python vs Référence")
    print(DIV)
    print()

    # ── Chargement du moteur Python ──────────────────────────
    print("[ CHARGEMENT DU MOTEUR PYTHON ]")
    engine = P2PInferenceEngine(fragments_dir, verbose=False)
    cfg = engine.config
    print(f"  Config : {cfg.n_layers}L, dim={cfg.dim}, heads={cfg.n_heads} "
          f"(KV={cfg.n_kv_heads}), vocab={cfg.vocab_size}, "
          f"ffn={cfg.hidden_dim}")
    print()

    # ── 1. Tests unitaires ───────────────────────────────────
    print(DIV)
    print("[ 1. TESTS UNITAIRES ]")
    print()

    print("  RMSNorm :")
    for r in test_rms_norm():
        print(r)
    print()

    print("  Softmax :")
    for r in test_softmax():
        print(r)
    print()

    print("  SwiGLU (vérification bug) :")
    for r in test_swiglu():
        print(r)
    print()

    print("  RoPE :")
    for r in test_rope_implementation():
        print(r)
    print()

    print("  Masque d'attention causal :")
    for r in test_attention_mask():
        print(r)
    print()

    # ── 2. Shapes des tenseurs ───────────────────────────────
    print(DIV)
    print("[ 2. SHAPES DES TENSEURS CHARGÉS ]")
    print()
    for r in inspect_tensor_shapes(engine):
        print(r)
    print()

    # ── 3. Ordre des multiplications ────────────────────────
    print(DIV)
    print("[ 3. ORDRE DES MULTIPLICATIONS MATRICIELLES ]")
    print()
    for r in detect_matmul_order(engine):
        print(r)
    print()

    # ── 4. Forward pass Python ───────────────────────────────
    print(DIV)
    print("[ 4. FORWARD PASS PYTHON — TOKEN PAR TOKEN ]")
    print()

    tokens = engine.tokenizer.encode(prompt)
    if not tokens or tokens[0] != 1:
        tokens = [1] + tokens
    print(f"  Prompt : '{prompt}'")
    print(f"  Tokens : {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    print()

    t0 = time.time()
    python_logits, fwd_stats = python_forward_single(engine, tokens[-1])
    t_fwd = time.time() - t0
    print(f"  Forward pass en {t_fwd:.2f}s")
    print()
    print_forward_stats(fwd_stats)
    print()

    # Top 5 Python
    top5 = np.argsort(-python_logits)[:5]
    print("  Top-5 Python :")
    for i, tid in enumerate(top5):
        try:
            word = engine.tokenizer.decode([int(tid)])
        except Exception:
            word = f"<{tid}>"
        print(f"    {i+1}. id={tid:<6} logit={python_logits[tid]:+8.3f}  '{word}'")
    print()

    # ── 5. Comparaison llama.cpp ─────────────────────────────
    print(DIV)
    print("[ 5. COMPARAISON AVEC LLAMA.CPP ]")
    print()

    if gguf_path is None:
        print(f"  {WARN} Pas de fichier GGUF fourni — comparaison llama.cpp ignorée.")
        print(f"       Utilisez --gguf path/to/model.gguf pour activer la comparaison.")
    else:
        gguf_path = str(gguf_path)
        if not Path(gguf_path).exists():
            print(f"  {FAIL} Fichier GGUF introuvable : {gguf_path}")
        else:
            try:
                print(f"  Récupération des logits llama.cpp…")
                t0 = time.time()
                ref_logits = get_llamacpp_logits(gguf_path, tokens)
                t_ref = time.time() - t0

                if ref_logits is None:
                    print(f"  {WARN} llama_cpp Python non installé.")
                    print(f"       pip install llama-cpp-python")
                else:
                    print(f"  llama.cpp en {t_ref:.2f}s")
                    print()
                    cmp_lines, cos_sim, top1_match = compare_logits(
                        python_logits, ref_logits, engine.tokenizer
                    )
                    for line in cmp_lines:
                        print(line)

                    # Diagnostic de la divergence
                    print()
                    print(f"  --- DIAGNOSTIC ---")
                    if cos_sim > 0.999 and top1_match:
                        print(f"  {PASS} Les logits sont quasi-identiques → moteur correct !")
                    elif cos_sim > 0.95:
                        print(f"  {WARN} Légère divergence (cos={cos_sim:.4f}).")
                        print(f"       Causes probables : epsilon norm, ordre GQA, dtype intermédiaire")
                    elif cos_sim > 0.5:
                        print(f"  {FAIL} Divergence modérée (cos={cos_sim:.4f}).")
                        print(f"       Causes probables : bug SwiGLU, transpose des poids, RoPE incorrect")
                    else:
                        print(f"  {FAIL} Divergence forte (cos={cos_sim:.4f}) — sortie incohérente.")
                        print(f"       Causes probables : multiplication transposée inversée, NaN, shape mismatch")

            except Exception as e:
                import traceback
                print(f"  {FAIL} Erreur llama.cpp : {e}")
                traceback.print_exc()

    # ── 6. Résumé & suggestions ──────────────────────────────
    print()
    print(DIV)
    print("[ 6. BUGS CONNUS ET SUGGESTIONS DE CORRECTION ]")
    print()

    # SwiGLU — vérification dynamique
    from inference.p2p_inference import swiglu
    x_test = np.array([1.0, 2.0], dtype=np.float32)
    silu_correct = x_test / (1.0 + np.exp(-x_test))
    silu_code = swiglu(x_test)
    if not np.allclose(silu_code, silu_correct, atol=1e-5):
        print(f"  {FAIL} BUG ACTIF — swiglu() retourne silu(x)*x au lieu de silu(x)")
        print(f"       Fix : return x / (1.0 + np.exp(-x))")
        print()
    else:
        print(f"  {PASS} swiglu() = SiLU correct")
        print()

    # RoPE — cohérence de la freq base
    print(f"  {INFO} RoPE freq base : {cfg.rope_freq_base}")
    if cfg.rope_freq_base == 10000.0:
        print(f"       Standard LLaMA-1/2. Llama-3 utilise 500000, Mistral 1000000.")
        print(f"       Verifiez que la valeur correspond a celle du fichier GGUF.")
    print()

    # Norm eps
    print(f"  {INFO} Norm epsilon : {cfg.norm_eps}")
    if cfg.norm_eps < 1e-7:
        print(f"       Tres petit — peut causer des instabilites numeriques.")
    print()

    # Résumé des points à vérifier
    print("  POINTS A VERIFIER (par ordre de probabilite d'impact) :")
    print()
    print(f"  1. {FAIL} Prefill manquant — IMPACT FORT")
    print(f"       Le code n'utilise que le DERNIER token du prompt, pas tout le contexte.")
    print(f"       Consequence : le modele repond sans avoir lu le prompt.")
    print(f"       Fix : faire passer x = w_emb[tokens] (tous les tokens) dans les couches.")
    print()
    print(f"  2. {WARN} Verifier le transpose des poids — IMPACT FORT si concerne")
    print(f"       GGUF stocke les poids en [out, in]. Le code fait x @ w.")
    print(f"       Si w=[out,in], il faut x @ w.T pour avoir la bonne dimension.")
    print(f"       Voir section 3 ci-dessus pour le diagnostic de votre modele.")
    print()
    print(f"  3. {WARN} KV cache absent — IMPACT MOYEN (correctif, pas un bug)")
    print(f"       Sans KV cache, chaque token est traite independamment → O(n²).")
    print(f"       Les resultats sont equivalents mais tres lents.")
    print()
    print(f"  4. {INFO} GQA : verification que n_kv_heads est bien lu depuis le manifest")
    print(f"       Config actuelle : n_heads={cfg.n_heads}, n_kv_heads={cfg.n_kv_heads}")
    if cfg.n_kv_heads == cfg.n_heads:
        print(f"       [!!] n_kv_heads == n_heads : GQA peut-etre non detecte dans le manifest")
    print()
    print(DIV)


# (section 6 réservée pour de futurs correctifs automatiques)


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validation complète du moteur d'inférence Python"
    )
    parser.add_argument(
        "fragments_dir",
        help="Répertoire contenant manifest.json et les .dat",
    )
    parser.add_argument(
        "--gguf",
        default=None,
        help="Chemin vers le fichier .gguf de référence (pour comparaison llama.cpp)",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt de test",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="Nombre de tokens à générer pour la comparaison",
    )
    parser.add_argument(
        "--units-only",
        action="store_true",
        help="Exécute uniquement les tests unitaires (rapide, sans chargement modèle)",
    )
    args = parser.parse_args()

    if args.units_only:
        # Tests unitaires seulement
        print(DIV)
        print("  TESTS UNITAIRES RAPIDES")
        print(DIV)
        for label, fn in [
            ("RMSNorm", test_rms_norm),
            ("Softmax", test_softmax),
            ("SwiGLU", test_swiglu),
            ("RoPE", test_rope_implementation),
            ("Masque causal", test_attention_mask),
        ]:
            print(f"\n  {label} :")
            for r in fn():
                print(r)
    else:
        run_full_validation(
            fragments_dir=args.fragments_dir,
            gguf_path=args.gguf,
            prompt=args.prompt,
            max_tokens=args.tokens,
        )

