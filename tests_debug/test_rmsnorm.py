#!/usr/bin/env python3
"""
Test RMSNorm implementation
"""
import numpy as np

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    # Current implementation
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(var + eps)
    return x_normed * weight

def test_rmsnorm():
    # Create test input
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    weight = np.ones(4, dtype=np.float32)
    eps = 1e-5

    print("Input:", x)
    print("Weight:", weight)

    # Apply RMSNorm
    result = rms_norm(x, weight, eps)

    print("\nAfter RMSNorm:")
    print("Output:", result)
    print("Mean:", result.mean())
    print("Std:", result.std())
    print("Variance:", np.var(result))

    # Check if variance is ~1.0
    var_check = np.mean(result ** 2)
    print(f"\nMean of squares (should be ~1.0): {var_check:.6f}")

    # Manual calculation
    var_manual = np.mean(x ** 2)
    x_normed_manual = x / np.sqrt(var_manual + eps)
    print(f"\nManual calculation:")
    print(f"  Variance of input: {var_manual:.6f}")
    print(f"  Normalized: {x_normed_manual}")
    print(f"  Mean of squares: {np.mean(x_normed_manual ** 2):.6f}")

if __name__ == "__main__":
    test_rmsnorm()
