"""
Q4_K_GGUF.py — Q4_K dequantization for GGUF tensors.

Block layout (144 bytes / 256 elements):
    d        f16[1]     super-block scale
    dmin     f16[1]     super-block min
    scales   u8[12]     8 × (scale, min) pairs packed as 6-bit
    qs       u8[128]    256 × 4-bit values (2 per byte, lo nibble first)

Formula per sub-group g of 32 elements:
    x[i] = d * scale[g] * q4[i] - dmin * min[g]

GGUF layout convention for 2-D weight matrices:
    Logical shape in manifest : [in_dim, out_dim]
    Physical data order       : [out_dim, in_dim]
    → reshape to [out_dim, in_dim] then .T → [in_dim, out_dim]

Fused GEMV (q4k_gemv):
    For decode (seq_len=1), instead of:
        dequantize 640 MB float32 → matmul
    We compute directly in physical layout:
        y[c] = sum_i  x[i] * W_phys[c, i]
    Never materialises the float32 weight matrix → ~20x faster for large FFN tensors.
"""

import numpy as np
import numba

QK_K        = 256
BLOCK_BYTES = 144  # 2 + 2 + 12 + 128


# ---------------------------------------------------------------------------
# Numba JIT kernel — compiles une seule fois, mis en cache sur disque
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _q4k_dequant_kernel(d_arr, dmin_arr, sc_arr, mn_arr, qs_arr, out):
    """
    Kernel compilé Q4_K.

    Args:
        d_arr   : (n_blocks,) float32   super-block scales
        dmin_arr: (n_blocks,) float32   super-block mins
        sc_arr  : (n_blocks, 8) uint8   sous-bloc scales (6-bit)
        mn_arr  : (n_blocks, 8) uint8   sous-bloc mins   (6-bit)
        qs_arr  : (n_blocks, 128) uint8 valeurs 4-bit (2 par octet)
        out     : (n_blocks, 256) float32  résultat (pre-alloué)
    """
    nb = d_arr.shape[0]
    for b in numba.prange(nb):
        db    = d_arr[b]
        dminb = dmin_arr[b]
        # 4 groupes de 64 éléments (2 sous-blocs × 32 éléments chacun)
        for g in range(4):
            s0 = numba.float32(sc_arr[b, g * 2])     * db
            s1 = numba.float32(sc_arr[b, g * 2 + 1]) * db
            m0 = numba.float32(mn_arr[b, g * 2])     * dminb
            m1 = numba.float32(mn_arr[b, g * 2 + 1]) * dminb
            base_q  = g * 32
            base_lo = g * 64
            base_hi = g * 64 + 32
            for l in range(32):
                q = qs_arr[b, base_q + l]
                out[b, base_lo + l] = s0 * numba.float32(q & numba.uint8(0x0F)) - m0
                out[b, base_hi + l] = s1 * numba.float32(q >> numba.uint8(4))   - m1


def _unpack_scales(scales_raw: np.ndarray):
    """
    Decode 8 (scale, min) pairs from 12 packed bytes (6 bits each).
    (identique à l'original — appelé côté Python, pas dans le kernel)
    """
    n  = len(scales_raw)
    sc = np.empty((n, 8), dtype=np.uint8)
    mn = np.empty((n, 8), dtype=np.uint8)

    sc[:, 0:4] = scales_raw[:, 0:4] & 0x3F
    mn[:, 0:4] = scales_raw[:, 4:8] & 0x3F

    sc[:, 4:8] = (scales_raw[:, 8:12] & 0x0F) | ((scales_raw[:, 0:4] >> 6) << 4)
    mn[:, 4:8] = (scales_raw[:, 8:12] >> 4)   | ((scales_raw[:, 4:8] >> 6) << 4)

    return sc, mn


def dequantize(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantize raw Q4_K bytes to float32 with GGUF transpose correction.
    Premier appel : JIT compilation (~5-10s). Appels suivants : binaire cache.
    """
    n_elements = int(np.prod(shape))
    n_blocks   = n_elements // QK_K
    assert len(data) == n_blocks * BLOCK_BYTES, (
        f"Q4_K size mismatch: expected {n_blocks * BLOCK_BYTES} bytes, got {len(data)}"
    )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)

    d    = raw[:, 0:2].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    dmin = raw[:, 2:4].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    sc, mn = _unpack_scales(raw[:, 4:16])   # (n_blocks, 8) uint8 chacun
    qs = raw[:, 16:144]                     # (n_blocks, 128) uint8

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    _q4k_dequant_kernel(d, dmin, sc, mn, qs, out)

    out = out.reshape(n_elements)

    if len(shape) == 2:
        out = out.reshape(shape[1], shape[0]).T.astype(np.float32)
    else:
        out = out.reshape(shape)

    return out


# ---------------------------------------------------------------------------
# Fused Q4_K GEMV — decode fast path (seq_len = 1)
# ---------------------------------------------------------------------------
# Évite de matérialiser la matrice float32 complète (640 MB pour FFN).
# Travaille directement dans le layout physique GGUF [out_dim, in_dim].
# Résultat : y[c] = sum_i  x[i] * W_phys[c, i]
#          = x @ W_logical  (car W_logical = W_phys.T)

@numba.njit(parallel=True, cache=True)
def _q4k_gemv_kernel(
    x: np.ndarray,           # float32[in_dim]
    d_arr: np.ndarray,       # float32[n_blocks]
    dmin_arr: np.ndarray,    # float32[n_blocks]
    sc_arr: np.ndarray,      # uint8[n_blocks, 8]
    mn_arr: np.ndarray,      # uint8[n_blocks, 8]
    qs_arr: np.ndarray,      # uint8[n_blocks, 128]
    out_dim: int,
    blocks_per_row: int,
) -> np.ndarray:
    """
    GEMV fusionné Q4_K en layout physique [out_dim, in_dim].

    Chaque thread calcule un scalaire y[c] = sum_i x[i] * W_phys[c, i].
    Jamais d'allocation float32 intermédiaire pour W.
    """
    result = np.zeros(out_dim, dtype=np.float32)
    for c in numba.prange(out_dim):
        acc = numba.float32(0.0)
        for b in range(blocks_per_row):
            bi = c * blocks_per_row + b          # indice global du bloc
            d    = d_arr[bi]
            dmin = dmin_arr[bi]
            inp_base = b * 256
            for g in range(4):
                s0 = numba.float32(sc_arr[bi, g * 2])     * d
                m0 = numba.float32(mn_arr[bi, g * 2])     * dmin
                s1 = numba.float32(sc_arr[bi, g * 2 + 1]) * d
                m1 = numba.float32(mn_arr[bi, g * 2 + 1]) * dmin
                base_q = g * 32
                for l in range(32):
                    q  = qs_arr[bi, base_q + l]
                    lo = numba.float32(q & numba.uint8(0x0F))
                    hi = numba.float32(q >> numba.uint8(4))
                    # Éléments du sous-bloc : lo half puis hi half
                    acc += x[inp_base + g * 64 + l]      * (s0 * lo - m0)
                    acc += x[inp_base + g * 64 + 32 + l] * (s1 * hi - m1)
        result[c] = acc
    return result


def q4k_gemv(x: np.ndarray, data: bytes, logical_shape: tuple) -> np.ndarray:
    """
    Fused Q4_K GEMV : calcule  y = x @ W  sans matérialiser W en float32.

    Paramètres
    ----------
    x             : float32[in_dim]  ou  float32[1, in_dim]
    data          : octets bruts Q4_K (format GGUF)
    logical_shape : (in_dim, out_dim) — shape LOGIQUE depuis manifest.json

    Retourne
    --------
    float32[out_dim]  (1D)

    Conditions
    ----------
    - Uniquement pour des tenseurs 2-D Q4_K
    - in_dim doit être un multiple de 256
    - Appeler warmup_q4k_gemv() une fois au démarrage pour la compilation JIT
    """
    in_dim, out_dim = logical_shape
    assert in_dim % QK_K == 0, f"in_dim={in_dim} doit être un multiple de {QK_K}"

    n_blocks       = in_dim * out_dim // QK_K
    blocks_per_row = in_dim // QK_K    # blocs par ligne physique (= par neurone de sortie)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)

    d    = raw[:, 0:2].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    dmin = raw[:, 2:4].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    sc, mn = _unpack_scales(raw[:, 4:16])
    qs = raw[:, 16:144]

    x_flat = np.ascontiguousarray(x.flatten(), dtype=np.float32)

    return _q4k_gemv_kernel(x_flat, d, dmin, sc, mn, qs, out_dim, blocks_per_row)


def warmup_q4k_gemv():
    """Déclenche la compilation JIT du kernel GEMV avec des données factices."""
    _x   = np.ones(256, dtype=np.float32)
    _d   = np.ones(1,   dtype=np.float32)
    _dm  = np.zeros(1,  dtype=np.float32)
    _sc  = np.ones((1, 8), dtype=np.uint8)
    _mn  = np.zeros((1, 8), dtype=np.uint8)
    _qs  = np.zeros((1, 128), dtype=np.uint8)
    _q4k_gemv_kernel(_x, _d, _dm, _sc, _mn, _qs, 1, 1)
