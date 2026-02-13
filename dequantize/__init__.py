"""
dequantize/ — GGUF dequantization package.

Each module handles one quantization format:
    Q4_K_GGUF.py   →  Q4_K_M / Q4_K_S
    Q6_K_GGUF.py   →  Q6_K
    Q8_0_GGUF.py   →  Q8_0
    F32_GGUF.py    →  F32, F16

Usage:
    from dequantize import dequantize
    weights = dequantize(raw_bytes, tensor_type, shape)

Or import a specific module directly:
    from dequantize.Q4_K_GGUF import dequantize as dequantize_q4k
"""

import numpy as np

from dequantize.Q4_K_GGUF import dequantize as _q4k
from dequantize.Q6_K_GGUF import dequantize as _q6k
from dequantize.Q8_0_GGUF import dequantize as _q8_0
from dequantize.F32_GGUF  import dequantize_f32, dequantize_f16


def dequantize(data: bytes, tensor_type: str, shape: tuple) -> np.ndarray:
    """
    Dispatch raw fragment bytes to the correct dequantizer.

    Args:
        data        : Raw bytes from a .dat fragment (or concatenated shards).
        tensor_type : String from manifest["tensor_type"], e.g. "Q4_K", "Q6_K".
        shape       : Logical tensor shape from manifest["shape"].

    Returns:
        np.ndarray float32.

    Raises:
        NotImplementedError for unknown quantization types.
    """
    if tensor_type in ("Q4_K", "Q4_K_M", "Q4_K_S"):
        return _q4k(data, shape)

    elif tensor_type == "Q6_K":
        return _q6k(data, shape)

    elif tensor_type == "Q8_0":
        return _q8_0(data, shape)

    elif tensor_type == "F32":
        return dequantize_f32(data, shape)

    elif tensor_type == "F16":
        return dequantize_f16(data, shape)

    else:
        raise NotImplementedError(
            f"Unsupported tensor type '{tensor_type}' — add a new module in dequantize/"
        )
