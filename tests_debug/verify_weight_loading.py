#!/usr/bin/env python3
"""
import setup_path  # noqa - adds project root to sys.path
Vérifier si les poids sont correctement chargés depuis les fragments
en les comparant directement avec le GGUF source
"""
import numpy as np
from pathlib import Path
import sys
import gguf

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.p2p_inference import P2PInferenceEngine

def compare_with_source_gguf():
    """Comparer les poids Python avec le GGUF source."""
    print("="*60)
    print("Comparaison: Fragments Python vs GGUF Source")
    print("="*60)

    # Load Python engine from fragments
    engine = P2PInferenceEngine(Path("../models/tinyllama_q8_fragments_v2"))

    # Load source GGUF
    source_gguf = Path("../models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
    if not source_gguf.exists():
        print(f"❌ Source GGUF not found: {source_gguf}")
        return

    reader = gguf.GGUFReader(str(source_gguf))

    # Build tensor dict from source GGUF
    gguf_tensors = {}
    for tensor in reader.tensors:
        name = str(tensor.name)
        data = tensor.data
        gguf_tensors[name] = data

    print(f"\nGGUF contient {len(gguf_tensors)} tenseurs")

    # Compare key tensors
    tensors_to_compare = [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_norm.weight",
        "output.weight",
        "output_norm.weight",
    ]

    all_match = True

    for tensor_name in tensors_to_compare:
        print(f"\n[{tensor_name}]")

        # Load from Python fragments
        try:
            py_tensor = engine.load_tensor(tensor_name)
        except Exception as e:
            print(f"  ❌ Erreur chargement Python: {e}")
            all_match = False
            continue

        # Get from source GGUF
        if tensor_name not in gguf_tensors:
            print(f"  ⚠️  Pas dans GGUF source (nom différent?)")
            # Try alternative names
            alt_names = [
                tensor_name.replace("output_norm", "norm"),
                tensor_name.replace("norm", "output_norm"),
            ]
            found = False
            for alt_name in alt_names:
                if alt_name in gguf_tensors:
                    print(f"  → Trouvé sous: {alt_name}")
                    gguf_tensor = gguf_tensors[alt_name]
                    found = True
                    break
            if not found:
                all_match = False
                continue
        else:
            gguf_tensor = gguf_tensors[tensor_name]

        # Compare shapes
        print(f"  Shape Python: {py_tensor.shape}")
        print(f"  Shape GGUF: {gguf_tensor.shape}")

        if py_tensor.shape != gguf_tensor.shape:
            # Try transpose
            if py_tensor.T.shape == gguf_tensor.shape:
                print(f"  ⚠️  Shapes correspondent après transpose")
                py_tensor = py_tensor.T
            else:
                print(f"  ❌ Shapes incompatibles!")
                all_match = False
                continue

        # Compare values
        diff = np.abs(py_tensor - gguf_tensor)
        max_diff = diff.max()
        mean_diff = diff.mean()

        # Check if values are identical or very close
        if max_diff < 1e-6:
            print(f"  ✅ Identiques (max diff: {max_diff:.2e})")
        elif max_diff < 1e-3:
            print(f"  ⚠️  Très proches (max diff: {max_diff:.2e})")
        else:
            print(f"  ❌ Différents!")
            print(f"     Max diff: {max_diff:.6f}")
            print(f"     Mean diff: {mean_diff:.6f}")
            print(f"     Python [0:5]: {py_tensor.flatten()[:5]}")
            print(f"     GGUF [0:5]: {gguf_tensor.flatten()[:5]}")
            all_match = False

    print("\n" + "="*60)
    if all_match:
        print("✅ TOUS LES POIDS CORRESPONDENT!")
        print("→ Le bug N'EST PAS dans le chargement des poids")
        print("→ Le bug EST dans le calcul du forward pass")
    else:
        print("❌ CERTAINS POIDS DIFFÈRENT!")
        print("→ Le bug POURRAIT ÊTRE dans le chargement des poids")
    print("="*60)

    return all_match

if __name__ == "__main__":
    compare_with_source_gguf()
