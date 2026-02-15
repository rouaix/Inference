#!/usr/bin/env python3
"""
Génère un manifest.json cohérent pour des fragments GGUF existants.
Analyse les noms de fichiers et crée un manifest compatible.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any

class FragmentInfo:
    def __init__(self, fragment_id: str, tensor_name: str, shard_index: int, 
                 file_size: int, file_hash: str, tensor_type: str = "Q6_K"):
        self.fragment_id = fragment_id
        self.tensor_name = tensor_name
        self.shard_index = shard_index
        self.file_size = file_size
        self.file_hash = file_hash
        self.tensor_type = tensor_type
        # Shape sera déterminée par l'architecture
        self.shape = []
        self.dtype = "uint8"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "tensor_name": self.tensor_name,
            "shard_index": self.shard_index,
            "file_size": self.file_size,
            "hash": self.file_hash,
            "tensor_type": self.tensor_type,
            "shape": self.shape,
            "dtype": self.dtype
        }

def analyze_fragments(fragments_dir: Path) -> Dict[str, List[FragmentInfo]]:
    """Analyse les fragments et retourne un dictionnaire tenseur -> fragments"""
    
    # Patterns pour les différents types de noms de fichiers
    patterns = [
        # Pattern principal: capture L-2_lm_head ou L0_attn_q
        r'Magistral-Small-2509-Q4_K_M_L([^_]+)_([^_]+)_([^_]+)_S(\d+)_([0-9a-f]+)\.dat',
        # Pattern alternatif pour les cas spéciaux
        r'Magistral-Small-2509-Q4_K_M_L([^_]+)_([^_]+)_S(\d+)_([0-9a-f]+)\.dat',
        # Pattern pour embedding/lm_head (2 parties seulement)
        r'Magistral-Small-2509-Q4_K_M_L-(\d+)_([^_]+)_S(\d+)_([0-9a-f]+)\.dat'
    ]
    
    tensor_fragments = {}
    unmatched_files = 0
    processed_files = 0
    
    for frag_file in fragments_dir.glob('*.dat'):
        matched = False
        
        for pattern in patterns:
            match = re.match(pattern, frag_file.name)
            if match:
                matched = True
                
                # Extraire les groupes - le pattern peut varier
                if len(match.groups()) == 5:
                    # Pattern principal: L0_attn_q_S0_hash (tenseurs de blocs)
                    layer_part = match.group(1)  # ex: "0", "1", ..., "39"
                    part2 = match.group(2)       # ex: "attn", "ffn"
                    part3 = match.group(3)       # ex: "q", "k", "up", "gate"
                    shard_idx = int(match.group(4))
                    file_hash = match.group(5)

                    layer_idx = layer_part.lstrip('-')

                    # Déterminer le nom du tenseur
                    tensor_name = f'blk.{layer_idx}.{part2}_{part3}.weight'

                    # Shape selon le type
                    shape = [5120, 5120]  # Valeur par défaut
                    if part2 == 'attn':
                        if part3 in ('q', 'output'):
                            shape = [5120, 5120]
                        elif part3 in ('k', 'v'):
                            shape = [5120, 1024]
                        elif part3 == 'norm':
                            shape = [5120]
                    elif part2 == 'ffn':
                        if part3 in ('up', 'gate'):
                            shape = [5120, 32768]
                        elif part3 == 'down':
                            shape = [32768, 5120]
                        elif part3 == 'norm':
                            shape = [5120]

                elif len(match.groups()) == 4:
                    # Pattern secondaire: L-2_lm_S0_hash ou L-1_embedding_S0_hash
                    layer_part = match.group(1)  # ex: "-2", "-1", "0"
                    part2 = match.group(2)       # ex: "lm", "embedding", "output"
                    shard_idx = int(match.group(3))
                    file_hash = match.group(4)

                    layer_idx = layer_part.lstrip('-')

                    # Déterminer le nom du tenseur
                    if part2 == 'embedding':
                        tensor_name = 'token_embd.weight'
                        shape = [5120, 131072]
                    elif part2 == 'lm':
                        tensor_name = 'output.weight'
                        shape = [5120, 131072]
                    elif part2 == 'output':
                        tensor_name = 'output_norm.weight'
                        shape = [5120]
                    else:
                        tensor_name = f'blk.{layer_idx}.{part2}.weight'
                        shape = [5120, 5120]

                elif len(match.groups()) == 3:
                    # Ancien pattern: L-1_embedding_S0_hash
                    layer_idx = match.group(1)
                    tensor_part = match.group(2)
                    
                    # Extraire shard_idx et hash du nom de fichier
                    shard_match = re.search(r'_S(\d+)_', frag_file.name)
                    if shard_match:
                        shard_idx = int(shard_match.group(1))
                    else:
                        shard_idx = 0
                        
                    hash_match = re.search(r'([0-9a-f]+)\.dat$', frag_file.name)
                    if hash_match:
                        file_hash = hash_match.group(1)
                    else:
                        file_hash = "unknown"
                    
                    # Déterminer le nom du tenseur (ancienne logique pour compatibilité)
                    if tensor_part == 'embedding':
                        tensor_name = 'token_embd.weight'
                        shape = [5120, 131072]
                    elif tensor_part == 'lm_head':
                        tensor_name = 'output.weight'
                        shape = [5120, 131072]
                    elif tensor_part == 'output_norm':
                        tensor_name = 'output_norm.weight'
                        shape = [5120]
                    else:
                        tensor_name = f'blk.{layer_idx}.{tensor_part}.weight'
                        if tensor_part in ['attn_q', 'attn_output']:
                            shape = [5120, 5120]
                        elif tensor_part in ['attn_k', 'attn_v']:
                            shape = [5120, 1024]
                        elif tensor_part in ['ffn_up', 'ffn_gate']:
                            shape = [5120, 32768]
                        elif tensor_part == 'ffn_down':
                            shape = [32768, 5120]
                        elif tensor_part in ['attn_norm', 'ffn_norm']:
                            shape = [5120]
                        else:
                            shape = [5120, 5120]
                else:
                    continue
                
                frag_info = FragmentInfo(
                    fragment_id=frag_file.stem,
                    tensor_name=tensor_name,
                    shard_index=shard_idx,
                    file_size=frag_file.stat().st_size,
                    file_hash=file_hash,
                    tensor_type="Q6_K"  # Supposons Q6_K pour Mistral
                )
                frag_info.shape = shape
                
                if tensor_name not in tensor_fragments:
                    tensor_fragments[tensor_name] = []
                tensor_fragments[tensor_name].append(frag_info)
                break
        
        if not matched:
            unmatched_files += 1
            # print(f"[DEBUG] Fichier non matched: {frag_file.name}")
        else:
            processed_files += 1
    
    if unmatched_files > 0:
        print(f"[INFO] {unmatched_files} fichiers non reconnus (sur {len(list(fragments_dir.glob('*.dat')))})")
    print(f"[INFO] {processed_files} fichiers traités")
    
    return tensor_fragments

def generate_manifest(fragments_dir: Path, output_path: Path):
    """Génère un manifest.json pour les fragments existants"""
    
    print(f"[INFO] Analyse des fragments dans {fragments_dir}...")
    tensor_fragments = analyze_fragments(fragments_dir)
    
    # Créer la structure du manifest
    manifest = {
        "model_name": "Magistral-Small-2509-Q4_K_M",
        "architecture": "mistral_small",
        "chunk_size": 10485760,  # 10 Mo
        "header_size": 0,  # À déterminer
        "total_fragments": sum(len(frags) for frags in tensor_fragments.values()),
        "config": {
            "dim": 5120,
            "hidden_dim": 32768,
            "n_layers": 40,
            "n_heads": 32,
            "n_kv_heads": 8,
            "vocab_size": 131072,
            "norm_eps": 1e-05,
            "rope_freq_base": 10000.0
        },
        "tensor_specifics": {},
        "fragments": [],
        "metadata": {
            "GGUF.version": 3,
            "llama.vocab_size": 131072,
            "llama.embedding_length": 5120,
            "llama.feed_forward_length": 32768,
            "llama.block_count": 40,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8
        }
    }
    
    # Ajouter tous les fragments
    for tensor_name, frag_list in tensor_fragments.items():
        # Trier par shard_index
        frag_list.sort(key=lambda x: x.shard_index)
        for frag in frag_list:
            manifest["fragments"].append(frag.to_dict())
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[SUCCESS] Manifest généré: {output_path}")
    print(f"   - {len(tensor_fragments)} tenseurs uniques")
    print(f"   - {manifest['total_fragments']} fragments totaux")
    print(f"   - Tenseurs: {', '.join(list(tensor_fragments.keys())[:5])}...")

def integrate_with_fragmenter(output_dir: Path):
    """
    Fonction pour être appelée depuis fragmenter.py après la fragmentation.
    Génère automatiquement un manifest.json cohérent.
    """
    print("\n[POST-FRAGMENTATION] Génération du manifest cohérent...")
    output_path = output_dir / "manifest.json"
    generate_manifest(output_dir, output_path)
    print(f"[POST-FRAGMENTATION] Manifest généré: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Génère un manifest.json pour des fragments GGUF existants")
    parser.add_argument("fragments_dir", help="Dossier contenant les fragments .dat")
    parser.add_argument("--output", default="manifest.json", help="Fichier de sortie manifest")
    parser.add_argument("--integrate", action="store_true", help="Mode intégration avec fragmenter.py")
    
    args = parser.parse_args()
    
    fragments_dir = Path(args.fragments_dir)
    output_path = fragments_dir / args.output
    
    if args.integrate:
        integrate_with_fragmenter(fragments_dir)
    else:
        generate_manifest(fragments_dir, output_path)
