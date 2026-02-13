#!/usr/bin/env python3
"""
Test RoPE implementation
"""
import numpy as np

# Test the current RoPE reshape logic
def test_rope_reshape():
    # Create a simple test case
    xq = np.arange(8, dtype=np.float32).reshape(1, 1, 8)  # [seq=1, heads=1, dim=8]
    print("Original xq:", xq)

    # Current implementation
    xq_r = xq.reshape(*xq.shape[:-1], -1, 2)  # [1, 1, 4, 2]
    print("Reshaped to pairs:", xq_r)

    xq_c = xq_r[..., 0] + 1j * xq_r[..., 1]  # [1, 1, 4]
    print("As complex:", xq_c)

    # Apply a simple rotation (multiply by 1+0j, no change)
    freqs = np.ones((1, 4), dtype=np.complex64)
    freqs_broadcast = freqs.reshape(1, 1, -1)
    xq_out_c = xq_c * freqs_broadcast
    print("After rotation:", xq_out_c)

    # Convert back - CURRENT METHOD
    xq_out_current = np.stack([xq_out_c.real, xq_out_c.imag], axis=-1).flatten().reshape(xq.shape)
    print("Back to real (current):", xq_out_current)

    # Convert back - CORRECT METHOD
    xq_out_correct = np.stack([xq_out_c.real, xq_out_c.imag], axis=-1).reshape(xq.shape)
    print("Back to real (correct):", xq_out_correct)

    print("\nMatch?", np.allclose(xq_out_current, xq))
    print("Correct match?", np.allclose(xq_out_correct, xq))

if __name__ == "__main__":
    test_rope_reshape()
