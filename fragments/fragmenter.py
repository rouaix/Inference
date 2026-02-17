import os
import json
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from enum import Enum
import gguf

# ============================================================
# Constants
# ============================================================

CHUNK_SIZE = 10 * 1024 * 1024  # 10 Mo par fragment

# ============================================================
# Fragment Types & Metadata
# ============================================================

class FragmentType(Enum):
    EMBEDDING = "embedding"
    ATTENTION = "attention"        # Q, K, V, O projections
    EXPERT = "expert"              # Un expert FFN individuel
    ROUTER = "router"              # Le routeur qui choisit les experts
    SHARED = "shared"              # Couches partagées (RMSNorm, etc.)
    LM_HEAD = "lm_head"
    UNKNOWN = "unknown"

@dataclass
class FragmentMeta:
    """Métadonnées d'un fragment de 10 Mo."""
    fragment_id: str
    model_name: str
    fragment_type: str             # FragmentType value
    layer_index: int               # -1 = embedding, -2 = lm_head
    component: str                 # "q_proj", "expert_42", "router", etc.
    expert_index: Optional[int]    # Index de l'expert (None si pas MoE)
    shard_index: int               # Index du shard dans ce composant
    total_shards: int
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    tensor_name: str               # Original GGUF tensor name
    tensor_type: str               # GGMLQuantizationType name (e.g. Q4_K)
    checksum: str

    # MoE routing info
    is_always_active: bool = True

    # Reconstruction info
    data_offset: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d

# ============================================================
# Real GGUF Fragmenter
# ============================================================

class RealGGUFFragmenter:
    """
    Découpe un fichier GGUF réel en fragments de 10 Mo.
    Utilise la bibliothèque `gguf` pour lire la structure.
    """

    def __init__(self, gguf_path: str, chunk_size: int = CHUNK_SIZE):
        self.gguf_path = Path(gguf_path)
        self.chunk_size = chunk_size
        self.fragments: List[FragmentMeta] = []
        self.stats = {
            "total_bytes": 0,
            "tensor_count": 0,
            "fragment_count": 0
        }

    def fragment(self, output_dir: str):
        """Procédure principale de fragmentation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Lecture de : {self.gguf_path}")
        reader = gguf.GGUFReader(self.gguf_path)

        # Récupération du nom du modèle (s'il existe dans les metadata KV)
        model_name = self.gguf_path.stem

        # Détecter l'architecture du modèle
        self.detected_arch = self.detect_architecture(reader)
        print(f"Architecture detectee: {self.detected_arch}")

        print(f"Debut de la fragmentation (Chunk size: {self.chunk_size/1024/1024:.0f} Mo)...")

        # 1. Calculate Header Size (min data_offset)
        min_offset = float('inf')
        for tensor in reader.tensors:
            if tensor.data_offset < min_offset:
                min_offset = tensor.data_offset

        header_size = min_offset
        print(f"   Header size: {header_size} bytes")

        # 2. Save Header
        with open(self.gguf_path, "rb") as f_in:
            header_data = f_in.read(header_size)

        header_path = output_dir / "gguf_header.dat"
        with open(header_path, "wb") as f_out:
            f_out.write(header_data)
        print(f"   Saved header to {header_path.name}")

        # 3. Process Tensors
        for tensor in reader.tensors:
            self._process_tensor(tensor, model_name, output_dir)

        self._save_manifest(output_dir / "manifest.json", model_name, reader, header_size)

        # Post-fragmentation : extraction du tokenizer depuis le GGUF
        try:
            import sys as _sys
            _fragments_dir = str(Path(__file__).parent)
            if _fragments_dir not in _sys.path:
                _sys.path.insert(0, _fragments_dir)
            from generate_tokenizer_model import extract_tokenizer
            extract_tokenizer(reader, output_dir)
        except ImportError as e:
            print(f"[WARN] Module generate_tokenizer_model introuvable: {e}")
        except Exception as e:
            print(f"[WARN] Extraction du tokenizer échouée: {e}")

        print(f"\n[OK] Fragmentation terminee !")
        print(f"   Fragments crees : {self.stats['fragment_count']}")
        print(f"   Volume total : {self.stats['total_bytes'] / (1024**3):.2f} Go")

    def _process_tensor(self, tensor: Any, model_name: str, output_dir: str):
        """Traite un tenseur : classification, découpage, sauvegarde."""
        # DEBUG: Check attributes
        # print(f"DEBUG Tensor attrs: {dir(tensor)}")
        # print(f"DEBUG Tensor offset: {getattr(tensor, 'offset', 'N/A')}")
        # print(f"DEBUG Tensor data_offset: {getattr(tensor, 'data_offset', 'N/A')}")

        name = tensor.name
        shape = tuple(tensor.shape)
        dtype = str(tensor.data.dtype)
        tensor_type = tensor.tensor_type.name
        data_offset = tensor.data_offset

        # Classification
        ftype, layer_idx, component, expert_idx, always_active = self._classify_tensor(name)

        # Lecture des données (Attention: charge en mémoire)
        # Pour les très gros modèles, il faudrait utiliser memmap ou lire par blocs si gguf le permet.
        # reader.tensors data attribute is a memmap if valid, or a numpy array.
        data = tensor.data
        data_bytes = data.tobytes()
        total_size = len(data_bytes)

        # Calcul du nombre de shards
        total_shards = max(1, -(-total_size // self.chunk_size))

        for shard_idx in range(total_shards):
            start = shard_idx * self.chunk_size
            end = min(start + self.chunk_size, total_size)
            chunk_data = data_bytes[start:end]

            checksum = hashlib.sha256(chunk_data).hexdigest()[:16]
            fid = f"{model_name}_L{layer_idx}_{component}_S{shard_idx}_{checksum}"

            # Sauvegarde fichier binaire
            filename = f"{fid}.dat"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(chunk_data)

            # Métadonnées
            meta = FragmentMeta(
                fragment_id=fid,
                model_name=model_name,
                fragment_type=ftype.value,
                layer_index=layer_idx,
                component=component,
                expert_index=expert_idx,
                shard_index=shard_idx,
                total_shards=total_shards,
                shape=shape,
                dtype=dtype,
                size_bytes=len(chunk_data),
                tensor_name=name,
                tensor_type=tensor_type,
                checksum=checksum,
                is_always_active=always_active,
                data_offset=data_offset
            )
            self.fragments.append(meta)
            self.stats["total_bytes"] += len(chunk_data)
            self.stats["fragment_count"] += 1

    def _classify_tensor(self, name: str) -> Tuple[FragmentType, int, str, Optional[int], bool]:
        """
        Classifie le tenseur d'après son nom (conventions GGUF/llama.cpp).
        Retourne : (Type, Layer, ComponentName, ExpertIdx, AlwaysActive)
        """
        # Exemples de noms GGUF :
        # token_embd.weight
        # output.weight
        # blk.0.attn_q.weight
        # blk.0.ffn_gate.weight (ou ffn_gate_inp pour MoE ?)

        parts = name.split('.')

        if name == "token_embd.weight":
            return FragmentType.EMBEDDING, -1, "embedding", None, True

        if name == "output.weight":
            return FragmentType.LM_HEAD, -2, "lm_head", None, True

        if name.startswith("blk."):
            # blk.0.attn_q.weight
            try:
                layer_idx = int(parts[1])
            except ValueError:
                layer_idx = -3 # Unknown layer

            component_part = parts[2] # attn_q, ffn_gate, etc.

            # Détection MoE (basique pour l'instant)
            # Dans Mixtral GGUF : blk.0.ffn_gate_inp.weight (router)
            # Les experts sont souvent dans ffn_gate, ffn_down, ffn_up mais avec une dimension expert
            # OU splités : blk.0.ffn_gate.0.weight

            if "ffn_gate_inp" in component_part:
                return FragmentType.ROUTER, layer_idx, "router", None, True

            if "attn" in component_part or "attn_norm" in component_part or "ffn_norm" in component_part:
                return FragmentType.ATTENTION, layer_idx, component_part, None, True

            # Pour les experts, c'est plus complexe selon le packing.
            # Supposons par défaut que tout le reste est "Expert" ou "FeedForward"
            # Si c'est un MoE, on devrait voir des patterns spécifiques.
            # Pour l'instant, traitons les FFN classiques comme "Toujours Actifs" sauf si on détecte explicitement "expert"

            if "expert" in name:
                # Si le nom contient "expert" (ex: blk.0.expert_0.weight - pas standard GGUF mais possible)
                return FragmentType.EXPERT, layer_idx, component_part, 0, False

            # Par défaut, on considère actif (modèle dense ou structure non reconnue)
            return FragmentType.SHARED, layer_idx, component_part, None, True

        return FragmentType.UNKNOWN, -3, name, None, True

    def _save_manifest(self, output_path: Path, model_name: str, reader: gguf.GGUFReader, header_size: int):
        # Helper to convert numpy types to native types
        def json_encoder(obj):
            if isinstance(obj, (np.integer, np.uint64)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, bytes):
                return str(obj) # Handle bytes (unlikely in KV values but safely)
            return obj

        # Extract metadata (KV pairs)
        # We don't really rely on this JSON metadata for reconstruction (we have the binary header),
        # but it's good for debugging.
        metadata = self._extract_metadata(reader)

        # Extraire les dimensions spécifiques du modèle
        model_config = self._extract_model_config(metadata)
        tensor_specifics = self._extract_tensor_specifics(reader)

        manifest = {
            "model_name": model_name,
            "architecture": self.detected_arch,
            "chunk_size": self.chunk_size,
            "header_size": header_size,
            "total_fragments": len(self.fragments),
            "config": model_config,
            "tensor_specifics": tensor_specifics,
            "fragments": [m.to_dict() for m in self.fragments],
            "metadata": metadata
        }
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, default=json_encoder)

    def detect_architecture(self, reader: gguf.GGUFReader) -> str:
        """Détecte l'architecture du modèle basé sur les tenseurs d'attention."""
        try:
            # Trouver les tenseurs Q et K de la première couche
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
            
            # Charger les données pour vérifier les dimensions
            wq_data = wq.data
            wk_data = wk.data
            
            # Mistral-Small a des dimensions réduites pour Q/K/V
            if wq_data.shape[1] == 4096 and wk_data.shape[1] == 1024:
                return "mistral_small"
            # Architecture standard LLaMA/Mistral
            elif wq_data.shape[1] == wq_data.shape[0] and wk_data.shape[1] == wk_data.shape[0]:
                return "standard_llama"
            # Autres architectures
            else:
                return "custom"
                
        except Exception as e:
            print(f"⚠️  Impossible de détecter l'architecture: {e}")
            return "standard_llama"

    def _extract_metadata(self, reader: gguf.GGUFReader) -> Dict[str, Any]:
        """Extracts KV pairs from GGUFReader."""
        metadata = {}
        for key, field in reader.fields.items():
            # We need to get the actual value.
            # In current gguf python, valid way is usually specific.
            # Let's try accessing `field.parts` via data indices is the raw way.
            # But `field.data` is List[int] (indices).

            values = []
            for idx in field.data:
                # Byte objects might need decoding if they are strings
                val = field.parts[idx]
                if isinstance(val, (bytes, bytearray)):
                    try:
                        val = val.decode('utf-8')
                    except:
                        val = str(val) # Fallback
                values.append(val)

            if len(values) == 1:
                metadata[key] = values[0]
            else:
                metadata[key] = values

        return metadata

    def _extract_model_config(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extraire la configuration du modèle à partir des métadonnées GGUF."""
        config = {
            "dim": metadata.get("llama.embedding_length", 4096),
            "hidden_dim": metadata.get("llama.feed_forward_length", 11008),
            "n_layers": metadata.get("llama.block_count", metadata.get("llms.count", 32)),
            "n_heads": metadata.get("llama.attention.head_count", 32),
            "n_kv_heads": metadata.get("llama.attention.head_count_kv", metadata.get("llama.attention.head_count", 32)),
            "vocab_size": metadata.get("llama.vocab_size", 32000),
            "norm_eps": metadata.get("llama.attention.layer_norm_rms_epsilon", 1e-5),
            "rope_freq_base": metadata.get("llama.rope.freq_base", 10000.0)
        }
        return config

    def _extract_tensor_specifics(self, reader: gguf.GGUFReader) -> Dict[str, Any]:
        """Extraire les dimensions spécifiques des tenseurs pour les architectures non-standard."""
        specifics = {
            "attention": {},
            "ffn": {}
        }

        try:
            # Analyser les tenseurs de la première couche pour détecter les dimensions spécifiques
            wq = self._find_tensor(reader, "blk.0.attn_q.weight")
            wk = self._find_tensor(reader, "blk.0.attn_k.weight")
            wv = self._find_tensor(reader, "blk.0.attn_v.weight")
            wo = self._find_tensor(reader, "blk.0.attn_output.weight")

            if wq and wk and wv and wo:
                specifics["attention"] = {
                    "q_dim": wq.data.shape[1],
                    "k_dim": wk.data.shape[1],
                    "v_dim": wv.data.shape[1],
                    "output_dim": wo.data.shape[1]
                }

            # Analyser les tenseurs FFN
            w_gate = self._find_tensor(reader, "blk.0.ffn_gate.weight")
            w_up = self._find_tensor(reader, "blk.0.ffn_up.weight")
            w_down = self._find_tensor(reader, "blk.0.ffn_down.weight")

            if w_gate and w_up and w_down:
                specifics["ffn"] = {
                    "gate_dim": w_gate.data.shape[1],
                    "up_dim": w_up.data.shape[1],
                    "down_dim": w_down.data.shape[1]
                }

        except Exception as e:
            print(f"⚠️  Impossible d'extraire les dimensions spécifiques: {e}")

        return specifics

    def _find_tensor(self, reader: gguf.GGUFReader, name: str):
        """Trouver un tenseur par son nom."""
        for tensor in reader.tensors:
            if tensor.name == name:
                return tensor
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real GGUF Fragmenter")
    parser.add_argument("gguf_file", help="Path to input GGUF file")
    parser.add_argument("--output", default="fragments", help="Output directory")
    args = parser.parse_args()

    frag = RealGGUFFragmenter(args.gguf_file)
    frag.fragment(args.output)
