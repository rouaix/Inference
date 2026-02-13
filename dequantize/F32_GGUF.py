"""
F32_GGUF.py — F32 and F16 passthrough for GGUF tensors.

F32 : raw float32, stored as-is (no quantization, no transpose).
F16 : raw float16, upcasted to float32 on load.

These types are used for 1-D tensors (RMSNorm weights, biases, etc.)
and are never transposed because they carry their correct shape directly.
"""

import numpy as np


def dequantize_f32(data: bytes, shape: tuple) -> np.ndarray:
    """
    Load F32 tensor bytes as float32.

    Args:
        data  : Raw bytes from a .dat fragment.
        shape : Logical tensor shape from manifest.

    Returns:
        np.ndarray float32.
    """
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


def dequantize_f16(data: bytes, shape: tuple) -> np.ndarray:
    """
    Load F16 tensor bytes and upcast to float32.

    Args:
        data  : Raw bytes from a .dat fragment.
        shape : Logical tensor shape from manifest.

    Returns:
        np.ndarray float32.
    """
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)
