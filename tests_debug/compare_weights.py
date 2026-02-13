#!/usr/bin/env python3
"""
Vérifier si les poids chargés par Python sont identiques à ceux de llama.cpp
"""
import numpy as np
from pathlib import Path
import sys
import gguf

sys.path.insert(0, str(Path(__file__).parent.parent))
from p2p_inference import P2PInferenceEngine
from p2p_bridge import reconstruct_gguf

def compare_weights():
    """Comparer les poids Python vs GGUF reconstruit."""
    print("="*60)
    print("Comparaison des Poids: Python vs GGUF")
    print("="*60)

    # Load Python engine
    engine = P2PInferenceEngine(Path("../models/tinyllama_q8_fragments_v2"))

    # Reconstruct GGUF
    temp_gguf = Path("../temp_weights_compare.gguf")
    reconstruct_gguf(Path("../models/tinyllama_q8_fragments_v2"), temp_gguf)

    # Load GGUF
    reader = gguf.GGUFReader(str(temp_gguf))

    # Build tensor dict from GGUF
    gguf_tensors = {}
    for tensor in reader.tensors:
        name = str(tensor.name)
        data = tensor.data
        gguf_tensors[name] = data

    # Compare key tensors
    tensors_to_compare = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output.weight",
    ]

    all_match = True

    for tensor_name in tensors_to_compare:
        print(f"\n[{tensor_name}]")

        # Load from Python
        py_tensor = engine.load_tensor(tensor_name)

        # Get from GGUF
        if tensor_name not in gguf_tensors:
            print(f"  ❌ Not found in GGUF!")
            all_match = False
            continue

        gguf_tensor = gguf_tensors[tensor_name]

        # Compare shapes
        if py_tensor.shape != gguf_tensor.shape:
            print(f"  ❌ Shape mismatch!")
            print(f"     Python: {py_tensor.shape}")
            print(f"     GGUF: {gguf_tensor.shape}")
            all_match = False
            continue

        # Compare values
        diff = np.abs(py_tensor - gguf_tensor)
        max_diff = diff.max()
        mean_diff = diff.mean()

        if max_diff > 1e-6:
            print(f"  ❌ Values differ!")
            print(f"     Max diff: {max_diff}")
            print(f"     Mean diff: {mean_diff}")
            print(f"     Python sample: {py_tensor.flatten()[:5]}")
            print(f"     GGUF sample: {gguf_tensor.flatten()[:5]}")
            all_match = False
        else:
            print(f"  ✅ Match (max diff: {max_diff:.2e})")

    # Cleanup
    temp_gguf.unlink()

    print("\n" + "="*60)
    if all_match:
        print("✅ Tous les poids correspondent!")
        print("Le bug n'est PAS dans le chargement des poids.")
    else:
        print("❌ Certains poids diffèrent!")
        print("Le bug pourrait être dans le chargement des poids.")
    print("="*60)

if __name__ == "__main__":
    compare_weights()
