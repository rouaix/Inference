"""
Q6_K_GGUF.py — Q6_K dequantization for GGUF tensors.

Block layout (210 bytes / 256 elements):
    ql       u8[128]    256 × 4-bit low bits  (2 per byte)
    qh       u8[64]     256 × 2-bit high bits (4 per byte)
    scales   i8[16]     16 × 8-bit signed scales (1 scale per 16 elements)
    d        f16[1]     super-block scale

6-bit value assembly:
    q6 = (4-bit from ql) | (2-bit from qh) << 4   →  range 0..63
    q6 -= 32                                        →  range -32..31  (signed)

Formula per element i:
    x[i] = d * scales[i // 16] * q6[i]

GGUF layout convention for 2-D weight matrices:
    Logical shape in manifest : [in_dim, out_dim]
    Physical data order       : [out_dim, in_dim]
    → reshape to [out_dim, in_dim] then .T → [in_dim, out_dim]
"""

import numpy as np
import numba

QK_K        = 256
BLOCK_BYTES = 210  # 128 + 64 + 16 + 2


# ---------------------------------------------------------------------------
# Numba JIT kernel
# ---------------------------------------------------------------------------

@numba.njit(parallel=True, cache=True)
def _q6k_dequant_kernel(d_arr, sc_arr, ql_arr, qh_arr, out):
    """
    Kernel compilé Q6_K.

    Args:
        d_arr  : (n_blocks,) float32   super-block scales
        sc_arr : (n_blocks, 16) int8   scales signées
        ql_arr : (n_blocks, 128) uint8 4 bits bas des valeurs 6-bit
        qh_arr : (n_blocks, 64)  uint8 2 bits hauts des valeurs 6-bit
        out    : (n_blocks, 256) float32  résultat (pre-alloué)
    """
    nb = d_arr.shape[0]
    for b in numba.prange(nb):
        db = d_arr[b]
        # 2 demi-blocs de 128 éléments
        for h in range(2):
            ql_off = h * 64
            qh_off = h * 32
            sc_off = h * 8
            el     = h * 128
            # l = 0..31 dans chaque demi-bloc
            for l in range(32):
                ql_a = ql_arr[b, ql_off + l]
                ql_b = ql_arr[b, ql_off + 32 + l]
                qh_v = qh_arr[b, qh_off + l]

                # Reconstruction 6-bit → int8 centré sur 0
                q1 = numba.int8((ql_a & numba.uint8(0x0F)) | (((qh_v >> numba.uint8(0)) & numba.uint8(3)) << numba.uint8(4))) - numba.int8(32)
                q2 = numba.int8((ql_b & numba.uint8(0x0F)) | (((qh_v >> numba.uint8(2)) & numba.uint8(3)) << numba.uint8(4))) - numba.int8(32)
                q3 = numba.int8((ql_a >> numba.uint8(4))   | (((qh_v >> numba.uint8(4)) & numba.uint8(3)) << numba.uint8(4))) - numba.int8(32)
                q4 = numba.int8((ql_b >> numba.uint8(4))   | (((qh_v >> numba.uint8(6)) & numba.uint8(3)) << numba.uint8(4))) - numba.int8(32)

                # Scale index : is = l // 16  (0 pour l=0..15, 1 pour l=16..31)
                is_idx = l >> 4  # équivalent l // 16
                s1 = numba.float32(sc_arr[b, sc_off + is_idx + 0]) * db
                s2 = numba.float32(sc_arr[b, sc_off + is_idx + 2]) * db
                s3 = numba.float32(sc_arr[b, sc_off + is_idx + 4]) * db
                s4 = numba.float32(sc_arr[b, sc_off + is_idx + 6]) * db

                out[b, el +  0 + l] = s1 * numba.float32(q1)
                out[b, el + 32 + l] = s2 * numba.float32(q2)
                out[b, el + 64 + l] = s3 * numba.float32(q3)
                out[b, el + 96 + l] = s4 * numba.float32(q4)


def dequantize(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantize raw Q6_K bytes to float32 with GGUF transpose correction.
    Premier appel : JIT compilation (~5-10s). Appels suivants : binaire cache.
    """
    n_elements = int(np.prod(shape))
    n_blocks   = n_elements // QK_K
    assert len(data) == n_blocks * BLOCK_BYTES, (
        f"Q6_K size mismatch: expected {n_blocks * BLOCK_BYTES} bytes, got {len(data)}"
    )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)

    ql = raw[:, 0:128]                                              # (n_blocks, 128)
    qh = raw[:, 128:192]                                            # (n_blocks, 64)
    sc = raw[:, 192:208].copy().view(np.int8).reshape(n_blocks, 16) # (n_blocks, 16) int8
    d  = raw[:, 208:210].copy().view(np.float16).reshape(n_blocks).astype(np.float32)

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    _q6k_dequant_kernel(d, sc, ql, qh, out)

    out = out.reshape(n_elements)

    if len(shape) == 2:
        out = out.reshape(shape[1], shape[0]).T.astype(np.float32)
    else:
        out = out.reshape(shape)

    return out
