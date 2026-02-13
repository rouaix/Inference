#!/usr/bin/env python3
"""Test rapide de détection d'architecture sans fragmentation complète."""

import json
from pathlib import Path
import gguf

def detect_architecture(reader):
    """Détecte l'architecture du modèle basé sur les tenseurs d'attention."""
    try:
        wq = None
        wk = None
        
        for tensor in reader.tensors:
            if tensor.name == "blk.0.attn_q.weight":
                wq = tensor
            elif tensor.name == "blk.0.attn_k.weight":
                wk = tensor
            
            if wq and wk:
                break
        
        if not wq or not wk:
            return "unknown"
        
        wq_data = wq.data
        wk_data = wk.data
        
        if wq_data.shape[1] == 4096 and wk_data.shape[1] == 1024:
            return "mistral_small"
        elif wq_data.shape[1] == wq_data.shape[0] and wk_data.shape[1] == wk_data.shape[0]:
            return "standard_llama"
        else:
            return "custom"
            
    except Exception as e:
        print(f"⚠️  Impossible de détecter l'architecture: {e}")
        return "standard_llama"

def extract_model_config(metadata):
    """Extraire la configuration du modèle à partir des métadonnées GGUF."""
    def get_scalar(value, default):
        """Extraire une valeur scalaire d'une liste ou valeur unique."""
        if isinstance(value, list) and len(value) > 0:
            return value[0]
        return value if value is not None else default
    
    config = {
        "dim": get_scalar(metadata.get("llama.embedding_length"), 4096),
        "hidden_dim": get_scalar(metadata.get("llama.feed_forward_length"), 11008),
        "n_layers": get_scalar(metadata.get("llama.block_count", metadata.get("llms.count")), 32),
        "n_heads": get_scalar(metadata.get("llama.attention.head_count"), 32),
        "n_kv_heads": get_scalar(metadata.get("llama.attention.head_count_kv", metadata.get("llama.attention.head_count")), 32),
        "vocab_size": get_scalar(metadata.get("llama.vocab_size"), 32000),
        "norm_eps": get_scalar(metadata.get("llama.attention.layer_norm_rms_epsilon"), 1e-5),
        "rope_freq_base": get_scalar(metadata.get("llama.rope.freq_base"), 10000.0)
    }
    return config

def extract_tensor_specifics(reader):
    """Extraire les dimensions spécifiques des tenseurs."""
    specifics = {"attention": {}, "ffn": {}}
    
    try:
        wq = next((t for t in reader.tensors if t.name == "blk.0.attn_q.weight"), None)
        wk = next((t for t in reader.tensors if t.name == "blk.0.attn_k.weight"), None)
        wv = next((t for t in reader.tensors if t.name == "blk.0.attn_v.weight"), None)
        wo = next((t for t in reader.tensors if t.name == "blk.0.attn_output.weight"), None)

        if wq and wk and wv and wo:
            specifics["attention"] = {
                "q_dim": wq.data.shape[1],
                "k_dim": wk.data.shape[1],
                "v_dim": wv.data.shape[1],
                "output_dim": wo.data.shape[1]
            }

        w_gate = next((t for t in reader.tensors if t.name == "blk.0.ffn_gate.weight"), None)
        w_up = next((t for t in reader.tensors if t.name == "blk.0.ffn_up.weight"), None)
        w_down = next((t for t in reader.tensors if t.name == "blk.0.ffn_down.weight"), None)

        if w_gate and w_up and w_down:
            specifics["ffn"] = {
                "gate_dim": w_gate.data.shape[1],
                "up_dim": w_up.data.shape[1],
                "down_dim": w_down.data.shape[1]
            }

    except Exception as e:
        print(f"⚠️  Impossible d'extraire les dimensions spécifiques: {e}")

    return specifics

def extract_metadata(reader):
    """Extraire les métadonnées GGUF."""
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
        
        if len(values) == 1:
            metadata[key] = values[0]
        else:
            metadata[key] = values
    return metadata

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python test_architecture_detection.py <path_to_gguf_file>")
        sys.exit(1)
    
    gguf_path = Path(sys.argv[1])
    print(f"Analyse de {gguf_path}...")
    
    reader = gguf.GGUFReader(gguf_path)
    
    # Détecter l'architecture
    arch = detect_architecture(reader)
    print(f"Architecture détectée: {arch}")
    
    # Extraire les métadonnées
    metadata = extract_metadata(reader)
    print(f"Configuration du modèle:")
    config = extract_model_config(metadata)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Extraire les dimensions spécifiques
    specifics = extract_tensor_specifics(reader)
    print(f"Dimensions spécifiques:")
    print(f"  Attention: {specifics['attention']}")
    print(f"  FFN: {specifics['ffn']}")
    
    # Créer un manifest minimal (sans les données memmap)
    manifest = {
        "model_name": str(gguf_path.stem),
        "architecture": arch,
        "config": config,
        "tensor_specifics": specifics,
        "metadata": {k: str(v) if isinstance(v, (bytes, bytearray)) else v for k, v in metadata.items()}
    }
    
    output_path = gguf_path.with_suffix(".manifest.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\nManifest sauvegardé dans {output_path}")
