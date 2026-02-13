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

QK_K       = 256
BLOCK_BYTES = 144  # 2 + 2 + 12 + 128


def _unpack_scales(scales_raw: np.ndarray):
    """
    Decode 8 (scale, min) pairs from 12 packed bytes (6 bits each).

    Encoding (ggml get_scale_min_k4):
      j = 0..3 : scale = bytes[j]   & 0x3F
                 min   = bytes[j+4] & 0x3F
      j = 4..7 : scale = (bytes[j+4] & 0x0F) | ((bytes[j-4] >> 6) << 4)
                 min   = (bytes[j+4] >> 4)   | ((bytes[j  ] >> 6) << 4)

    Args:
        scales_raw: (n_blocks, 12) uint8

    Returns:
        sc, mn : each (n_blocks, 8) uint8
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

    Args:
        data  : Raw bytes from a .dat fragment (or concatenated shards).
        shape : Logical tensor shape from manifest, e.g. (in_dim, out_dim).

    Returns:
        np.ndarray float32, shape corrected (2-D tensors are transposed).
    """
    n_elements = int(np.prod(shape))
    n_blocks   = n_elements // QK_K
    assert len(data) == n_blocks * BLOCK_BYTES, (
        f"Q4_K size mismatch: expected {n_blocks * BLOCK_BYTES} bytes, got {len(data)}"
    )

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, BLOCK_BYTES)

    # Super-block scales  (.copy() ensures contiguity before .view())
    d    = raw[:, 0:2].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    dmin = raw[:, 2:4].copy().view(np.float16).reshape(n_blocks).astype(np.float32)

    sc, mn = _unpack_scales(raw[:, 4:16])   # (n_blocks, 8) each

    qs = raw[:, 16:144]                      # (n_blocks, 128)
    lo = (qs & 0x0F).astype(np.float32)     # lower nibble
    hi = (qs >>   4).astype(np.float32)     # upper nibble

    # Reshape for fully-vectorised group operations.
    # 4 outer groups × [lo, hi] × 32 elements = 256 elements per block.
    # Output order matches the C dequant loop:
    #   [g0_lo×32, g0_hi×32, g1_lo×32, g1_hi×32, g2_lo×32, g2_hi×32, g3_lo×32, g3_hi×32]
    lo_g  = lo.reshape(n_blocks, 4, 32)
    hi_g  = hi.reshape(n_blocks, 4, 32)
    sc_g  = sc.reshape(n_blocks, 4, 2).astype(np.float32)
    mn_g  = mn.reshape(n_blocks, 4, 2).astype(np.float32)
    d4    = d[:, None, None]
    dmin4 = dmin[:, None, None]

    out = np.empty((n_blocks, 4, 2, 32), dtype=np.float32)
    out[:, :, 0, :] = d4 * sc_g[:, :, 0:1] * lo_g - dmin4 * mn_g[:, :, 0:1]
    out[:, :, 1, :] = d4 * sc_g[:, :, 1:2] * hi_g - dmin4 * mn_g[:, :, 1:2]

    out = out.reshape(n_elements)

    if len(shape) == 2:
        out = out.reshape(shape[1], shape[0]).T.astype(np.float32)
    else:
        out = out.reshape(shape)

    return out
