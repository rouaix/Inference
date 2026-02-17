#!/usr/bin/env python3
"""Test du chargement avec le nouveau manifest."""

import setup_path  # noqa - adds project root to sys.path
import json
from pathlib import Path
import sys
import os

# Ajouter le chemin pour importer p2p_inference
sys.path.insert(0, str(Path(__file__).parent))

from inference.p2p_inference import P2PInferenceEngine

# Chemin vers les fragments
fragments_dir = Path("models/Magistral-Small-2509-Q4_K_M_fragments")

print(f"Test de chargement depuis {fragments_dir}...")

try:
    # Charger le manifest
    manifest_path = fragments_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Erreur: Manifest non trouve: {manifest_path}")
        sys.exit(1)
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    print(f"Manifest chargé:")
    print(f"  Architecture: {manifest.get('architecture', 'N/A')}")
    print(f"  Config: {manifest.get('config', 'N/A')}")
    print(f"  Tensor specifics: {manifest.get('tensor_specifics', 'N/A')}")
    
    # Tester le chargement avec P2PInferenceEngine
    print(f"\nChargement avec P2PInferenceEngine...")
    engine = P2PInferenceEngine(str(fragments_dir), verbose=True)
    
    print(f"\nSucces ! Configuration chargee:")
    print(f"  dim: {engine.config.dim}")
    print(f"  n_heads: {engine.config.n_heads}")
    print(f"  n_kv_heads: {engine.config.n_kv_heads}")
    print(f"  Tensor specifics: {engine.tensor_specifics}")
    
    # Tester le chargement d'un tenseur
    print(f"\nTest de chargement d'un tenseur...")
    wq = engine.load_tensor("blk.0.attn_q.weight")
    print(f"  blk.0.attn_q.weight shape: {wq.shape}")
    
    wk = engine.load_tensor("blk.0.attn_k.weight")
    print(f"  blk.0.attn_k.weight shape: {wk.shape}")
    
    # Tester les dimensions spécifiques
    attn_dims = engine.get_attention_dims(0)
    print(f"\nDimensions spécifiques pour l'attention:")
    print(f"  {attn_dims}")
    
except Exception as e:
    print(f"\nErreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\nTous les tests passes !")
