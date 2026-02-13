#!/usr/bin/env python3
"""Test simple de détection d'architecture."""

import json
from pathlib import Path
import gguf
import numpy as np

# Chemin vers le modèle
model_path = Path("models/Magistral-Small-2509-Q4_K_M.gguf")

print(f"Analyse de {model_path}...")

# Charger le lecteur GGUF
reader = gguf.GGUFReader(model_path)

# Extraire les métadonnées de base
metadata = {}
for key, field in reader.fields.items():
    values = []
    for idx in field.data:
        val = field.parts[idx]
        if isinstance(val, (bytes, bytearray)):
            try:
                val = val.decode('utf-8')
            except:
                val = str(val)
        values.append(val)
    metadata[key] = values[0] if len(values) == 1 else values

print(f"Configuration du modèle:")
print(f"  dim: {metadata.get('llama.embedding_length', 'N/A')}")
print(f"  hidden_dim: {metadata.get('llama.feed_forward_length', 'N/A')}")
print(f"  n_layers: {metadata.get('llama.block_count', 'N/A')}")
print(f"  n_heads: {metadata.get('llama.attention.head_count', 'N/A')}")
print(f"  n_kv_heads: {metadata.get('llama.attention.head_count_kv', 'N/A')}")

# Analyser les tenseurs
wq = next((t for t in reader.tensors if t.name == "blk.0.attn_q.weight"), None)
wk = next((t for t in reader.tensors if t.name == "blk.0.attn_k.weight"), None)

if wq and wk:
    print(f"\nDimensions des tenseurs d'attention:")
    print(f"  Q: {wq.data.shape}")
    print(f"  K: {wk.data.shape}")
    
    # Détecter l'architecture
    if wq.data.shape[1] == 4096 and wk.data.shape[1] == 1024:
        arch = "mistral_small"
    elif wq.data.shape[1] == wq.data.shape[0] and wk.data.shape[1] == wk.data.shape[0]:
        arch = "standard_llama"
    else:
        arch = "custom"
    
    print(f"\nArchitecture détectée: {arch}")

# Créer un manifest simple
def get_int_value(value, default=0):
    """Convertir une valeur en entier."""
    if isinstance(value, (np.integer, np.uint64)):
        return int(value)
    elif isinstance(value, list) and len(value) > 0:
        return get_int_value(value[0], default)
    elif isinstance(value, (int, float)):
        return int(value)
    return default

manifest = {
    "model_name": str(model_path.stem),
    "architecture": arch,
    "config": {
        "dim": get_int_value(metadata.get('llama.embedding_length'), 4096),
        "hidden_dim": get_int_value(metadata.get('llama.feed_forward_length'), 11008),
        "n_layers": get_int_value(metadata.get('llama.block_count'), 32),
        "n_heads": get_int_value(metadata.get('llama.attention.head_count'), 32),
        "n_kv_heads": get_int_value(metadata.get('llama.attention.head_count_kv'), 32),
    }
}

if wq and wk:
    manifest["tensor_specifics"] = {
        "attention": {
            "q_dim": int(wq.data.shape[1]),
            "k_dim": int(wk.data.shape[1])
        }
    }

# Sauvegarder
output_path = model_path.with_suffix(".arch_manifest.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nManifest sauvegardé dans {output_path}")
