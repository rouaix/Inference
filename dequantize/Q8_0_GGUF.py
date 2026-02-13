"""
Q8_0_GGUF.py — Q8_0 dequantization for GGUF tensors.

Block layout (34 bytes / 32 elements):
    d    f16[1]    block scale
    qs   i8[32]    32 × 8-bit signed quantized values

Formula per element i:
    x[i] = d * qs[i]

GGUF layout convention for 2-D weight matrices:
    Logical shape in manifest : [in_dim, out_dim]
    Physical data order       : [out_dim, in_dim]
    → reshape to [out_dim, in_dim] then .T → [in_dim, out_dim]

Note: This is the format used by TinyLlama Q8_0.
      The transpose fix was the root cause of the ">>" token loop bug
      (systematic prediction of token 5099) before it was identified.
"""

import numpy as np

QK      = 32    # Elements per block (Q8_0 uses QK=32, not QK_K=256)
BLOCK_BYTES = 34   # 2 + 32


def dequantize(data: bytes, shape: tuple) -> np.ndarray:
    """
    Dequantize raw Q8_0 bytes to float32 with GGUF transpose correction.

    Args:
        data  : Raw bytes from a .dat fragment (or concatenated shards).
        shape : Logical tensor shape from manifest.

    Returns:
        np.ndarray float32, shape corrected (2-D tensors are transposed).
    """
    n_elements = int(np.prod(shape))
    n_blocks   = n_elements // QK
    assert len(data) == n_blocks * BLOCK_BYTES, (
        f"Q8_0 size mismatch: expected {n_blocks * BLOCK_BYTES} bytes, got {len(data)}"
    )

    dt  = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))])
    blocks = np.frombuffer(data, dtype=dt)

    # Dequantize: x = d * qs
    d   = blocks['d'].astype(np.float32)   # (n_blocks,)
    qs  = blocks['qs'].astype(np.float32)  # (n_blocks, 32)
    out = (d[:, None] * qs).reshape(n_elements)

    # GGUF transpose correction for 2-D weight matrices
    if len(shape) == 2:
        out_dim = shape[1]   # 2nd logical dim = physical row count
        in_dim  = shape[0]
        out = out.reshape(out_dim, in_dim).T.astype(np.float32)
    else:
        out = out.reshape(shape)

    return out
