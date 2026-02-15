"""
kernels_numba.py
================
Kernels critiques récrits avec numba.njit(parallel=True) pour remplacer
les versions NumPy pur de p2p_inference.py.

Chaque kernel est accompagné d'un wrapper Python qui gère le pré-parsing
des données brutes (non supporté dans @njit) et la transposition du layout
physique GGUF.

Warm-up : appeler warmup_kernels() au démarrage pour déclencher la
compilation AOT et éviter une pause pendant l'inférence.
"""

import numpy as np
import numba


# ---------------------------------------------------------------------------
# 1. Q8_0 dequantization
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _q8_0_kernel(scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    Kernel JIT Q8_0.

    scales : float32[n_blocks]
    qs     : float32[n_blocks, 32]
    retour : float32[n_blocks * 32]  (mise à plat, sans transposition)
    """
    n_blocks = scales.shape[0]
    out = np.empty(n_blocks * 32, dtype=np.float32)
    for i in numba.prange(n_blocks):
        s = scales[i]
        for j in range(32):
            out[i * 32 + j] = s * qs[i, j]
    return out


def dequantize_q8_0(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantise un tenseur Q8_0 GGUF.

    Paramètres
    ----------
    data  : octets bruts du fragment (n_blocks × 34 octets par bloc)
    shape : tuple — shape LOGIQUE du tenseur (depuis manifest.json)

    Retourne
    --------
    np.ndarray float32 avec transposition du layout physique GGUF :
      layout physique [out_dim, in_dim]  →  layout logique [in_dim, out_dim]

    Cf. CLAUDE.md § "GGUF Transposed Physical Layout".
    """
    dt = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))])
    blocks = np.frombuffer(data, dtype=dt)
    scales = blocks['d'].astype(np.float32)         # [n_blocks]
    qs     = blocks['qs'].astype(np.float32)         # [n_blocks, 32]
    decoded = _q8_0_kernel(scales, qs)               # [n_blocks * 32]

    # Transposition layout physique → logique (identique à p2p_inference.py:load_tensor)
    if len(shape) == 2:
        out_dim = shape[-1]   # 2e dim logique = nombre de lignes physiques
        in_dim  = shape[0]
        return decoded.reshape(out_dim, in_dim).T.astype(np.float32)
    return decoded.reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. RMS Norm
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def rms_norm_numba(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    RMSNorm token par token.

    x      : float32[seq_len, dim]
    weight : float32[dim]
    retour : float32[seq_len, dim]

    Formule : x * (1 / sqrt(mean(x²) + eps)) * weight
    """
    seq_len, dim = x.shape
    out = np.empty_like(x)
    for i in numba.prange(seq_len):
        ss = numba.float32(0.0)
        for j in range(dim):
            ss += x[i, j] * x[i, j]
        ss = numba.float32(1.0) / np.sqrt(ss / numba.float32(dim) + numba.float32(eps))
        for j in range(dim):
            out[i, j] = x[i, j] * ss * weight[j]
    return out


# ---------------------------------------------------------------------------
# 3. Softmax
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def softmax_numba(x: np.ndarray) -> np.ndarray:
    """
    Softmax sur la dernière dimension, numériquement stable (soustraction du max).

    x      : float32[n_heads, seq_len, seq_len]
    retour : float32[n_heads, seq_len, seq_len]
    """
    n_heads, seq, seq2 = x.shape
    out = np.empty_like(x)
    for h in numba.prange(n_heads):
        for i in range(seq):
            max_val = x[h, i, 0]
            for j in range(1, seq2):
                if x[h, i, j] > max_val:
                    max_val = x[h, i, j]
            s = numba.float32(0.0)
            for j in range(seq2):
                out[h, i, j] = np.exp(x[h, i, j] - max_val)
                s += out[h, i, j]
            for j in range(seq2):
                out[h, i, j] /= s
    return out


# ---------------------------------------------------------------------------
# 4. SwiGLU (SiLU activation)
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def swiglu_numba(x: np.ndarray) -> np.ndarray:
    """
    SiLU : x / (1 + exp(-x))

    x      : float32[seq_len, hidden_dim]
    retour : float32[seq_len, hidden_dim]
    """
    n, m = x.shape
    out = np.empty_like(x)
    for i in numba.prange(n):
        for j in range(m):
            out[i, j] = x[i, j] / (numba.float32(1.0) + np.exp(-x[i, j]))
    return out


# ---------------------------------------------------------------------------
# 5. Utilitaire contiguïté
# ---------------------------------------------------------------------------

def ensure_contiguous(a: np.ndarray) -> np.ndarray:
    """Force un array en C-contiguous float32 pour BLAS optimal."""
    if not a.flags['C_CONTIGUOUS'] or a.dtype != np.float32:
        return np.ascontiguousarray(a, dtype=np.float32)
    return a


# ---------------------------------------------------------------------------
# Warm-up JIT
# ---------------------------------------------------------------------------

def warmup_kernels():
    """
    Déclenche la compilation AOT de tous les kernels avec des données factices.
    À appeler une seule fois au démarrage, avant la première inférence.
    Avec cache=True, la compilation n'a lieu qu'une seule fois (résultat mis en cache disque).
    """
    _s = np.ones(1, dtype=np.float32)
    _q = np.ones((1, 32), dtype=np.float32)
    _q8_0_kernel(_s, _q)

    _x = np.ones((1, 64), dtype=np.float32)
    _w = np.ones(64, dtype=np.float32)
    rms_norm_numba(_x, _w, np.float32(1e-5))

    _sc = np.ones((1, 1, 1), dtype=np.float32)
    softmax_numba(_sc)

    _g = np.ones((1, 64), dtype=np.float32)
    swiglu_numba(_g)

    # Warm-up du kernel GEMV Q4_K si disponible
    try:
        from dequantize.Q4_K_GGUF import warmup_q4k_gemv
        warmup_q4k_gemv()
    except ImportError:
        pass

    print("[kernels_numba] Warm-up JIT termine.")


# ---------------------------------------------------------------------------
# Tests unitaires inline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ajouter le dossier racine au PYTHONPATH si lancé depuis tests_debug/
    sys.path.insert(0, str(Path(__file__).parent))

    print("=== Warm-up JIT ===")
    warmup_kernels()

    print("\n=== Tests unitaires ===")

    # -- rms_norm_numba --
    from p2p_inference import rms_norm
    x = np.random.randn(8, 2048).astype(np.float32)
    w = np.random.randn(2048).astype(np.float32)
    ref_norm = rms_norm(x, w, 1e-5)
    res_norm = rms_norm_numba(x, w, np.float32(1e-5))
    assert np.allclose(ref_norm, res_norm, atol=1e-5), \
        f"rms_norm mismatch: max_diff={np.max(np.abs(ref_norm - res_norm)):.2e}"
    print("[OK] rms_norm_numba")

    # -- softmax_numba --
    from p2p_inference import softmax
    s = np.random.randn(4, 8, 8).astype(np.float32)
    ref_s = softmax(s)
    res_s = softmax_numba(s)
    assert np.allclose(ref_s, res_s, atol=1e-5), \
        f"softmax mismatch: max_diff={np.max(np.abs(ref_s - res_s)):.2e}"
    print("[OK] softmax_numba")

    # -- swiglu_numba --
    from p2p_inference import swiglu
    g = np.random.randn(4, 5632).astype(np.float32)
    ref_g = swiglu(g)
    res_g = swiglu_numba(g)
    assert np.allclose(ref_g, res_g, atol=1e-5), \
        f"swiglu mismatch: max_diff={np.max(np.abs(ref_g - res_g)):.2e}"
    print("[OK] swiglu_numba")

    # -- dequantize_q8_0 (nécessite des fragments Q8_0) --
    q8_fragments = Path("models/tinyllama_q8_fragments_v2")
    if q8_fragments.exists():
        from distribution.local import LocalFragmentLoader
        loader = LocalFragmentLoader(str(q8_fragments))
        tname = "blk.0.attn_q.weight"
        ref = loader.load_tensor(tname)
        frags = loader.fragments_map[tname]
        raw = b"".join(loader.load_raw(f["fragment_id"]) for f in frags)
        result = dequantize_q8_0(raw, tuple(frags[0]["shape"]))
        assert np.allclose(ref, result, atol=1e-5), \
            f"dequantize_q8_0 mismatch: max_diff={np.max(np.abs(ref - result)):.2e}"
        print("[OK] dequantize_q8_0")
    else:
        print("[SKIP] dequantize_q8_0 — fragments TinyLlama Q8_0 non disponibles")

    print("\nTous les kernels valides.")
