"""
Model Fragmenter v2
Cible : Mistral Large 3 (675B MoE, 41B actifs) en Q4
Chunks de 10 Mo — optimisé pour le calcul distribué MoE.

L'avantage du MoE pour le P2P :
- 675B params total, mais seulement 41B actifs par token
- Chaque token n'active que ~2 experts sur ~128 par couche
- On n'a donc besoin que d'un SOUS-ENSEMBLE de nœuds par requête
- Réduit drastiquement la coordination réseau
"""

import os
import json
import hashlib
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from enum import Enum


# ============================================================
# Constants
# ============================================================

CHUNK_SIZE = 10 * 1024 * 1024  # 10 Mo par fragment


# ============================================================
# Fragment Types — MoE-aware
# ============================================================

class FragmentType(Enum):
    EMBEDDING = "embedding"
    ATTENTION = "attention"        # Q, K, V, O projections
    EXPERT = "expert"              # Un expert FFN individuel
    ROUTER = "router"              # Le routeur qui choisit les experts
    SHARED = "shared"              # Couches partagées (RMSNorm, etc.)
    LM_HEAD = "lm_head"


@dataclass
class FragmentMeta:
    """Métadonnées d'un fragment de 10 Mo — MoE aware."""
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
    checksum: str

    # MoE routing info
    is_always_active: bool = True  # True = requis pour chaque token
    # False = activé seulement quand l'expert est sélectionné

    def to_dict(self) -> dict:
        d = asdict(self)
        d['shape'] = list(d['shape'])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'FragmentMeta':
        d['shape'] = tuple(d['shape'])
        return cls(**d)


# ============================================================
# Mistral Large 3 Architecture (réaliste)
# ============================================================

@dataclass
class MistralLarge3Config:
    """
    Configuration basée sur Mistral Large 3 réel.
    MoE granulaire : 675B total, 41B actifs.
    """
    model_name: str = "mistral-large-3-675b"

    # Architecture globale
    num_layers: int = 88              # Nombre de couches transformer
    hidden_dim: int = 6144            # Dimension cachée
    num_attention_heads: int = 48     # Têtes d'attention
    num_kv_heads: int = 8             # Grouped Query Attention
    head_dim: int = 128
    vocab_size: int = 131072          # Tokenizer Tekken

    # MoE config
    num_experts: int = 128            # Experts par couche
    num_active_experts: int = 2       # Experts activés par token
    expert_intermediate_dim: int = 4096  # Dim FFN par expert
    router_dim: int = 128             # Petite couche de routage

    # Quantization
    dtype: str = "int4"               # Q4 quantization
    bits_per_weight: float = 4.5      # ~4.5 bits effectifs en Q4_K_M

    def estimate_total_size_gb(self) -> float:
        """Estime la taille totale du modèle quantifié."""
        bytes_per_param = self.bits_per_weight / 8

        # Embedding + LM Head
        embed_params = self.vocab_size * self.hidden_dim * 2

        # Par couche :
        # Attention: Q, K, V, O
        attn_params_per_layer = (
            self.hidden_dim * self.num_attention_heads * self.head_dim +  # Q
            self.hidden_dim * self.num_kv_heads * self.head_dim +         # K
            self.hidden_dim * self.num_kv_heads * self.head_dim +         # V
            self.num_attention_heads * self.head_dim * self.hidden_dim    # O
        )

        # Router par couche
        router_params_per_layer = self.hidden_dim * self.num_experts

        # Experts FFN par couche (gate + up + down pour chaque expert)
        expert_params = (
            self.hidden_dim * self.expert_intermediate_dim +   # gate
            self.hidden_dim * self.expert_intermediate_dim +   # up
            self.expert_intermediate_dim * self.hidden_dim     # down
        )
        all_experts_per_layer = expert_params * self.num_experts

        # Shared (RMSNorm etc.) — petit
        shared_per_layer = self.hidden_dim * 4

        total_per_layer = (attn_params_per_layer + router_params_per_layer +
                          all_experts_per_layer + shared_per_layer)

        total_params = embed_params + (total_per_layer * self.num_layers)
        total_bytes = total_params * bytes_per_param

        return total_bytes / (1024**3)

    def estimate_active_size_gb(self) -> float:
        """Estime la taille des paramètres ACTIFS par token (41B)."""
        bytes_per_param = self.bits_per_weight / 8

        embed_params = self.vocab_size * self.hidden_dim * 2

        attn_params_per_layer = (
            self.hidden_dim * self.num_attention_heads * self.head_dim +
            self.hidden_dim * self.num_kv_heads * self.head_dim +
            self.hidden_dim * self.num_kv_heads * self.head_dim +
            self.num_attention_heads * self.head_dim * self.hidden_dim
        )

        router_params_per_layer = self.hidden_dim * self.num_experts

        # Seulement 2 experts actifs sur 128
        active_expert_params = (
            self.hidden_dim * self.expert_intermediate_dim +
            self.hidden_dim * self.expert_intermediate_dim +
            self.expert_intermediate_dim * self.hidden_dim
        ) * self.num_active_experts

        shared_per_layer = self.hidden_dim * 4

        active_per_layer = (attn_params_per_layer + router_params_per_layer +
                           active_expert_params + shared_per_layer)

        total_active = embed_params + (active_per_layer * self.num_layers)
        return total_active * bytes_per_param / (1024**3)


# ============================================================
# Simulated MoE Model
# ============================================================

class SimulatedMoEModel:
    """
    Modèle MoE simulé avec poids aléatoires.
    Respecte la structure réelle de Mistral Large 3.
    """

    # Configs pré-définies
    CONFIGS = {
        "micro-moe": MistralLarge3Config(
            model_name="micro-moe",
            num_layers=4,
            hidden_dim=256,
            num_attention_heads=8,
            num_kv_heads=2,
            head_dim=32,
            vocab_size=1000,
            num_experts=8,
            num_active_experts=2,
            expert_intermediate_dim=512,
        ),
        "mini-moe": MistralLarge3Config(
            model_name="mini-moe",
            num_layers=12,
            hidden_dim=1024,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=64,
            vocab_size=10000,
            num_experts=16,
            num_active_experts=2,
            expert_intermediate_dim=2048,
        ),
        "mistral-large-3": MistralLarge3Config(),  # Config réelle
    }

    def __init__(self, config_name: str = "micro-moe"):
        if config_name not in self.CONFIGS:
            raise ValueError(f"Config inconnue: {config_name}. Choix: {list(self.CONFIGS.keys())}")

        self.config = self.CONFIGS[config_name]
        self.tensors: Dict[str, np.ndarray] = {}
        self._build()

    def _build(self):
        cfg = self.config
        print(f"\n🔨 Construction du modèle MoE simulé '{cfg.model_name}'")
        print(f"   {cfg.num_layers} couches, {cfg.num_experts} experts/couche, "
              f"{cfg.num_active_experts} actifs/token")
        print(f"   hidden={cfg.hidden_dim}, heads={cfg.num_attention_heads}")

        # Embedding
        self.tensors["embedding"] = self._rand(cfg.vocab_size, cfg.hidden_dim)

        for layer in range(cfg.num_layers):
            prefix = f"layer_{layer}"

            # Attention (toujours actif)
            self.tensors[f"{prefix}.q_proj"] = self._rand(
                cfg.hidden_dim, cfg.num_attention_heads * cfg.head_dim)
            self.tensors[f"{prefix}.k_proj"] = self._rand(
                cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim)
            self.tensors[f"{prefix}.v_proj"] = self._rand(
                cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim)
            self.tensors[f"{prefix}.o_proj"] = self._rand(
                cfg.num_attention_heads * cfg.head_dim, cfg.hidden_dim)

            # Router (toujours actif, très petit)
            self.tensors[f"{prefix}.router"] = self._rand(
                cfg.hidden_dim, cfg.num_experts)

            # RMSNorm (toujours actif, très petit)
            self.tensors[f"{prefix}.input_norm"] = self._rand(1, cfg.hidden_dim)
            self.tensors[f"{prefix}.post_attn_norm"] = self._rand(1, cfg.hidden_dim)

            # Experts FFN (activés conditionnellement)
            for exp in range(cfg.num_experts):
                self.tensors[f"{prefix}.expert_{exp}.gate"] = self._rand(
                    cfg.hidden_dim, cfg.expert_intermediate_dim)
                self.tensors[f"{prefix}.expert_{exp}.up"] = self._rand(
                    cfg.hidden_dim, cfg.expert_intermediate_dim)
                self.tensors[f"{prefix}.expert_{exp}.down"] = self._rand(
                    cfg.expert_intermediate_dim, cfg.hidden_dim)

        # LM Head
        self.tensors["lm_head"] = self._rand(cfg.hidden_dim, cfg.vocab_size)

        # Stats
        total_bytes = sum(t.nbytes for t in self.tensors.values())
        print(f"\n   📊 Statistiques du modèle simulé:")
        print(f"   Taille simulée: {total_bytes / (1024**3):.2f} Go")
        print(f"   Nombre de tenseurs: {len(self.tensors)}")

        # Estimations pour le vrai modèle
        est_total = cfg.estimate_total_size_gb()
        est_active = cfg.estimate_active_size_gb()
        print(f"\n   📐 Estimations pour le vrai {cfg.model_name} en Q4:")
        print(f"   Taille totale:  ~{est_total:.0f} Go → {int(est_total * 1024 / 10)} fragments de 10 Mo")
        print(f"   Taille active:  ~{est_active:.0f} Go → {int(est_active * 1024 / 10)} fragments actifs/token")
        print(f"   Ratio actif/total: {est_active/est_total*100:.1f}%")

    def _rand(self, rows: int, cols: int) -> np.ndarray:
        """Génère des poids aléatoires int8 (simulation de Q4)."""
        return np.random.randint(-128, 127, size=(rows, cols), dtype=np.int8)


# ============================================================
# MoE-Aware Fragmenter
# ============================================================

class MoEFragmenter:
    """
    Fragmenteur adapté au MoE.
    Distingue les fragments "toujours actifs" des fragments "experts conditionnels".
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.fragments: List[Tuple[FragmentMeta, np.ndarray]] = []
        self.stats = {
            "always_active": 0,
            "conditional": 0,
            "total_bytes": 0,
            "active_bytes": 0,
        }

    def fragment_model(self, model: SimulatedMoEModel) -> List[Tuple[FragmentMeta, np.ndarray]]:
        """Découpe le modèle MoE en fragments de 10 Mo."""
        self.fragments = []
        cfg = model.config

        print(f"\n✂️  Découpage MoE en fragments de {self.chunk_size / (1024*1024):.0f} Mo...")

        for tensor_name, tensor_data in model.tensors.items():
            # Déterminer le type et si c'est toujours actif
            ftype, layer_idx, component, expert_idx, always_active = \
                self._classify_tensor(tensor_name, cfg)

            self._fragment_tensor(
                tensor=tensor_data,
                model_name=cfg.model_name,
                fragment_type=ftype,
                layer_index=layer_idx,
                component=component,
                expert_index=expert_idx,
                always_active=always_active,
            )

        # Stats
        self.stats["always_active"] = sum(
            1 for m, _ in self.fragments if m.is_always_active)
        self.stats["conditional"] = sum(
            1 for m, _ in self.fragments if not m.is_always_active)
        self.stats["total_bytes"] = sum(m.size_bytes for m, _ in self.fragments)
        self.stats["active_bytes"] = sum(
            m.size_bytes for m, _ in self.fragments if m.is_always_active)

        print(f"\n   📊 Résultat du découpage:")
        print(f"   Total fragments:    {len(self.fragments)}")
        print(f"   Toujours actifs:    {self.stats['always_active']} "
              f"({self.stats['active_bytes']/(1024**2):.0f} Mo)")
        print(f"   Experts conditionnels: {self.stats['conditional']} "
              f"({(self.stats['total_bytes'] - self.stats['active_bytes'])/(1024**2):.0f} Mo)")
        print(f"   Ratio actif/total:  "
              f"{self.stats['active_bytes']/max(1,self.stats['total_bytes'])*100:.1f}%")

        return self.fragments

    def _classify_tensor(self, name: str, cfg) -> Tuple[str, int, str, Optional[int], bool]:
        """Classifie un tenseur : type, couche, composant, expert, toujours_actif."""
        if name == "embedding":
            return FragmentType.EMBEDDING.value, -1, "embedding", None, True
        elif name == "lm_head":
            return FragmentType.LM_HEAD.value, -2, "lm_head", None, True

        parts = name.split(".")
        layer_idx = int(parts[0].split("_")[1])

        if "expert_" in name:
            expert_idx = int(parts[1].split("_")[1])
            component = parts[2]  # gate, up, down
            return (FragmentType.EXPERT.value, layer_idx,
                    f"expert_{expert_idx}.{component}", expert_idx, False)
        elif "router" in name:
            return FragmentType.ROUTER.value, layer_idx, "router", None, True
        elif "norm" in name:
            return FragmentType.SHARED.value, layer_idx, parts[1], None, True
        else:
            return FragmentType.ATTENTION.value, layer_idx, parts[1], None, True

    def _fragment_tensor(self, tensor: np.ndarray, model_name: str,
                         fragment_type: str, layer_index: int,
                         component: str, expert_index: Optional[int],
                         always_active: bool):
        """Découpe un tenseur en chunks."""
        data = tensor.tobytes()
        total_shards = max(1, -(-len(data) // self.chunk_size))  # ceil division

        for shard_idx in range(total_shards):
            start = shard_idx * self.chunk_size
            end = min(start + self.chunk_size, len(data))
            chunk_data = data[start:end]
            chunk_array = np.frombuffer(chunk_data, dtype=tensor.dtype).copy()

            checksum = hashlib.sha256(chunk_data).hexdigest()[:16]
            fid = f"{model_name}_L{layer_index}_{component}_S{shard_idx}_{checksum}"

            meta = FragmentMeta(
                fragment_id=fid,
                model_name=model_name,
                fragment_type=fragment_type,
                layer_index=layer_index,
                component=component,
                expert_index=expert_index,
                shard_index=shard_idx,
                total_shards=total_shards,
                shape=chunk_array.shape,
                dtype=str(tensor.dtype),
                size_bytes=len(chunk_data),
                checksum=checksum,
                is_always_active=always_active,
            )
            self.fragments.append((meta, chunk_array))

    def save_manifest(self, output_path: str):
        """Sauvegarde le manifeste (sans les données — trop gros)."""
        manifest = {
            "model_name": self.fragments[0][0].model_name if self.fragments else "",
            "chunk_size": self.chunk_size,
            "total_fragments": len(self.fragments),
            "always_active_fragments": self.stats["always_active"],
            "conditional_fragments": self.stats["conditional"],
            "total_size_bytes": self.stats["total_bytes"],
            "active_size_bytes": self.stats["active_bytes"],
            "fragments": [m.to_dict() for m, _ in self.fragments],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n💾 Manifeste sauvegardé: {output_path}")


    def generate_virtual_fragments(self, config) -> List[Tuple[FragmentMeta, Optional[np.ndarray]]]:
        """Génère des fragments virtuels (sans données) pour les gros modèles."""
        self.fragments = []
        cfg = config
        print(f"\n⚡ Génération de fragments VIRTUELS pour {cfg.model_name}...")

        # Helper to generate fragments for a theoretical tensor size
        def add_virtual_tensor(name, shape, dtype="int8"):
            # Calculate size
            size_bytes = 1  # int8 = 1 byte
            for dim in shape:
                size_bytes *= dim

            ftype, layer_idx, component, expert_idx, always_active = \
                self._classify_tensor(name, cfg)

            total_shards = max(1, -(-size_bytes // self.chunk_size))

            for shard_idx in range(total_shards):
                # Calculate exact size of this shard
                remaining = size_bytes - (shard_idx * self.chunk_size)
                this_chunk_size = min(self.chunk_size, remaining)

                fid = f"{cfg.model_name}_L{layer_idx}_{component}_S{shard_idx}_virtual"

                meta = FragmentMeta(
                    fragment_id=fid,
                    model_name=cfg.model_name,
                    fragment_type=ftype,
                    layer_index=layer_idx,
                    component=component,
                    expert_index=expert_idx,
                    shard_index=shard_idx,
                    total_shards=total_shards,
                    shape=shape, # This is tensor shape, not chunk shape
                    dtype=dtype,
                    size_bytes=this_chunk_size,
                    checksum="virtual",
                    is_always_active=always_active,
                )
                # No data
                self.fragments.append((meta, None))

        # Replicate the structure from SimulatedMoEModel._build

        # Embedding
        add_virtual_tensor("embedding", (cfg.vocab_size, cfg.hidden_dim))

        for layer in range(cfg.num_layers):
            prefix = f"layer_{layer}"

            # Attention
            add_virtual_tensor(f"{prefix}.q_proj", (cfg.hidden_dim, cfg.num_attention_heads * cfg.head_dim))
            add_virtual_tensor(f"{prefix}.k_proj", (cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim))
            add_virtual_tensor(f"{prefix}.v_proj", (cfg.hidden_dim, cfg.num_kv_heads * cfg.head_dim))
            add_virtual_tensor(f"{prefix}.o_proj", (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_dim))

            # Router
            add_virtual_tensor(f"{prefix}.router", (cfg.hidden_dim, cfg.num_experts))

            # RMSNorm
            add_virtual_tensor(f"{prefix}.input_norm", (1, cfg.hidden_dim))
            add_virtual_tensor(f"{prefix}.post_attn_norm", (1, cfg.hidden_dim))

            # Experts
            for exp in range(cfg.num_experts):
                add_virtual_tensor(f"{prefix}.expert_{exp}.gate", (cfg.hidden_dim, cfg.expert_intermediate_dim))
                add_virtual_tensor(f"{prefix}.expert_{exp}.up", (cfg.hidden_dim, cfg.expert_intermediate_dim))
                add_virtual_tensor(f"{prefix}.expert_{exp}.down", (cfg.expert_intermediate_dim, cfg.hidden_dim))

        # LM Head
        add_virtual_tensor("lm_head", (cfg.hidden_dim, cfg.vocab_size))

        # Stats
        self.stats["always_active"] = sum(1 for m, _ in self.fragments if m.is_always_active)
        self.stats["conditional"] = sum(1 for m, _ in self.fragments if not m.is_always_active)
        self.stats["total_bytes"] = sum(m.size_bytes for m, _ in self.fragments)
        self.stats["active_bytes"] = sum(m.size_bytes for m, _ in self.fragments if m.is_always_active)

        print(f"\n   📊 Résultat virtuel:")
        print(f"   Total fragments:    {len(self.fragments)}")
        print(f"   Toujours actifs:    {self.stats['always_active']}")
        return self.fragments


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE Fragmenter v2")
    parser.add_argument("--model", default="mistral-large-3",
                        choices=["micro-moe", "mini-moe", "mistral-large-3"],
                        help="Modèle simulé")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="Taille des fragments en octets (défaut: 10 Mo)")
    parser.add_argument("--save-manifest", type=str, default=None,
                        help="Chemin pour sauvegarder le manifeste")
    parser.add_argument("--stats-only", action="store_true",
                        help="Afficher seulement les stats sans fragmenter")
    args = parser.parse_args()

    if args.stats_only or args.model == "mistral-large-3":
        # Pour mistral-large-3, on montre seulement les stats
        # (trop gros pour simuler en mémoire)
        cfg = SimulatedMoEModel.CONFIGS[args.model]
        print("=" * 60)
        print(f"Stats pour {cfg.model_name}")
        print("=" * 60)

        chunk_mb = args.chunk_size / (1024 * 1024)
        est_total = cfg.estimate_total_size_gb()
        est_active = cfg.estimate_active_size_gb()

        total_fragments = int(est_total * 1024 / chunk_mb)
        active_fragments = int(est_active * 1024 / chunk_mb)
        expert_fragments = total_fragments - active_fragments

        print(f"\n📐 Architecture:")
        print(f"   Couches:           {cfg.num_layers}")
        print(f"   Dimension cachée:  {cfg.hidden_dim}")
        print(f"   Experts/couche:    {cfg.num_experts}")
        print(f"   Experts actifs:    {cfg.num_active_experts}")
        print(f"   Vocabulaire:       {cfg.vocab_size}")

        print(f"\n📊 Tailles en Q4_K_M:")
        print(f"   Total:             ~{est_total:.0f} Go")
        print(f"   Actif par token:   ~{est_active:.0f} Go ({est_active/est_total*100:.1f}%)")

        print(f"\n🧩 Fragments de {chunk_mb:.0f} Mo:")
        print(f"   Total:             {total_fragments:,}")
        print(f"   Toujours actifs:   {active_fragments:,}")
        print(f"   Experts (cond.):   {expert_fragments:,}")

        for rep in [3, 5]:
            print(f"\n🌐 Réseau avec réplication ×{rep}:")
            print(f"   Nœuds total:       {total_fragments * rep:,}")
            print(f"   Nœuds actifs/req:  ~{active_fragments * rep:,}")
            print(f"   Bande passante/token: ~{est_active * 1024:.0f} Mo "
                  f"(activations inter-couches)")

    else:
        print("=" * 60)
        print(f"MoE Fragmenter v2")
        print("=" * 60)

        model = SimulatedMoEModel(config_name=args.model)
        fragmenter = MoEFragmenter(chunk_size=args.chunk_size)
        fragments = fragmenter.fragment_model(model)

        if args.save_manifest:
            fragmenter.save_manifest(args.save_manifest)
