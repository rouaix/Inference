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

QK_K        = 256
BLOCK_BYTES = 210  # 128 + 64 + 16 + 2


def dequantize(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantize raw Q6_K bytes to float32 with GGUF transpose correction.

    Args:
        data  : Raw bytes from a .dat fragment (or concatenated shards).
        shape : Logical tensor shape from manifest.

    Returns:
        np.ndarray float32, shape corrected (2-D tensors are transposed).
    """
    n_elements = int(np.prod(shape))
    n_blocks   = n_elements // QK_K
    assert len(data) == n_blocks * BLOCK_BYTES, (
        f"Q6_K size mismatch: expected {n_blocks * BLOCK_BYTES} bytes, got {len(data)}"
    )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)

    ql = raw[:, 0:128]                                          # (n_blocks, 128)
    qh = raw[:, 128:192]                                        # (n_blocks, 64)
    sc = raw[:, 192:208].copy().view(np.int8).reshape(n_blocks, 16)  # (n_blocks, 16)
    d  = raw[:, 208:210].copy().view(np.float16).reshape(n_blocks).astype(np.float32)

    # Scale index for l = 0..31 :  is = l // 16  →  0 for l=0..15, 1 for l=16..31
    is_idx = np.arange(32) // 16   # (32,) ∈ {0, 1}

    out = np.empty((n_blocks, QK_K), dtype=np.float32)

    # Two outer halves:  h=0 → elements 0..127,  h=1 → elements 128..255
    for h in range(2):
        ql_h = ql[:, h*64 : h*64+64]                           # (n_blocks, 64)
        qh_h = qh[:, h*32 : h*32+32]                           # (n_blocks, 32)
        sc_h = sc[:, h*8  : h*8+8 ].astype(np.float32)         # (n_blocks, 8)
        el   = h * 128

        ql_lo = ql_h[:, 0:32]    # ql[l]       for l = 0..31
        ql_hi = ql_h[:, 32:64]   # ql[l + 32]

        # Assemble 6-bit values, centre at 0 (subtract 32)
        # q1 → output positions  el+0  .. el+31
        # q2 → output positions  el+32 .. el+63
        # q3 → output positions  el+64 .. el+95
        # q4 → output positions  el+96 .. el+127
        q1 = ((ql_lo & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.int16) - 32
        q2 = ((ql_hi & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.int16) - 32
        q3 = ((ql_lo >>   4) | (((qh_h >> 4) & 3) << 4)).astype(np.int16) - 32
        q4 = ((ql_hi >>   4) | (((qh_h >> 6) & 3) << 4)).astype(np.int16) - 32

        # Scale arrays  (n_blocks, 32) — is_idx selects the right scale per element
        s0 = sc_h[:, is_idx + 0]   # for q1
        s2 = sc_h[:, is_idx + 2]   # for q2
        s4 = sc_h[:, is_idx + 4]   # for q3
        s6 = sc_h[:, is_idx + 6]   # for q4

        d_col = d[:, None]
        out[:, el+0  : el+32 ] = d_col * s0 * q1.astype(np.float32)
        out[:, el+32 : el+64 ] = d_col * s2 * q2.astype(np.float32)
        out[:, el+64 : el+96 ] = d_col * s4 * q3.astype(np.float32)
        out[:, el+96 : el+128] = d_col * s6 * q4.astype(np.float32)

    out = out.reshape(n_elements)

    if len(shape) == 2:
        out = out.reshape(shape[1], shape[0]).T.astype(np.float32)
    else:
        out = out.reshape(shape)

    return out
