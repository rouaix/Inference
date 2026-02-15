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
