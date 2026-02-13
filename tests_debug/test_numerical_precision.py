#!/usr/bin/env python3
"""
Test si le problème vient de l'ordre des opérations ou de la précision numérique
Comparaison directe avec une implémentation de référence simple
"""
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from p2p_inference import P2PInferenceEngine, LlamaLayer, rms_norm, swiglu

def test_numerical_precision():
    """Test la précision numérique des opérations."""
    engine = P2PInferenceEngine(Path("../models/tinyllama_q8_fragments_v2"))

    print("="*60)
    print("Test de Précision Numérique")
    print("="*60)

    # Test 1: Vérifier que float32 vs float64 fait une différence
    x = np.random.randn(10).astype(np.float32)

    # Calcul en float32
    result32 = x / (1.0 + np.exp(-x))

    # Calcul en float64
    x64 = x.astype(np.float64)
    result64 = x64 / (1.0 + np.exp(-x64))
    result64_32 = result64.astype(np.float32)

    diff = np.abs(result32 - result64_32).max()
    print(f"\nTest Float32 vs Float64:")
    print(f"  Max difference: {diff}")
    print(f"  Relative error: {diff / np.abs(result32).max()}")

    # Test 2: Accumuler les erreurs sur 22 couches
    print(f"\n--- Simulation d'accumulation d'erreurs ---")

    x = np.random.randn(1, 2048).astype(np.float32)
    x_original = x.copy()

    for i in range(22):
        # Simuler une opération par couche
        x = x + 0.01 * (x / (1.0 + np.exp(-x)))

    print(f"Après 22 couches:")
    print(f"  Changement: {np.abs(x - x_original).mean()}")

    # Test 3: Vérifier si les poids Q8_0 ont des artéfacts
    print(f"\n--- Test Poids Q8_0 ---")

    w = engine.load_tensor("blk.0.attn_q.weight")
    print(f"Shape: {w.shape}")
    print(f"Dtype: {w.dtype}")
    print(f"Min: {w.min():.6f}, Max: {w.max():.6f}")
    print(f"Mean: {w.mean():.6f}, Std: {w.std():.6f}")

    # Vérifier s'il y a des valeurs quantifiées visibles
    unique_values = np.unique(w.flatten())
    print(f"Nombre de valeurs uniques: {len(unique_values)}")

    if len(unique_values) < 1000:
        print(f"⚠️  Peu de valeurs uniques - quantification visible")
        print(f"Premières valeurs: {unique_values[:10]}")

    print("\n" + "="*60)

if __name__ == "__main__":
    test_numerical_precision()
