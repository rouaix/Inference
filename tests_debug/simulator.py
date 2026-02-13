import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class FragmentMeta:
    id: str
    type: str
    layer: int
    component: str
    shard_index: int
    total_shards: int
    tensor_name: str
    size_bytes: int

class FunctionalSimulator:
    """
    Simulateur fonctionnel qui exécute la logique d'inférence (boucle couches/tokens)
    en chargeant réellement les fragments depuis le disque.
    """
    def __init__(self, fragments_dir: str):
        self.fragments_dir = Path(fragments_dir)
        self.manifest_path = self.fragments_dir / "manifest.json"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")

        print(f"📖 Chargement du manifeste : {self.manifest_path} ...")
        with open(self.manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.model_name = self.manifest["model_name"]
        self.metadata = self.manifest.get("metadata", {})

        # Extraction des hyperparamètres clés
        def get_int(key, default):
            val = self.metadata.get(key, default)
            if isinstance(val, list):
                if len(val) > 0: return int(val[0])
                return default
            return int(val)

        self.n_layers = get_int("llms.count" if "llms.count" in self.metadata else "llama.block_count", 22)
        self.n_embd = get_int("llama.embedding_length", 2048)
        self.vocab_size = get_int("llama.vocab_size", 32000)

        print(f"🤖 Modèle : {self.model_name}")
        print(f"   Couches : {self.n_layers}")
        print(f"   Dim : {self.n_embd}")
        print(f"   Vocab : {self.vocab_size}")

        self.fragments_index = self._index_fragments()
        print(f"✅ Indexation terminée : {len(self.manifest['fragments'])} fragments connus.\n")

    def _index_fragments(self) -> Dict[str, List[FragmentMeta]]:
        """Index fragments by a key useful for retrieval (e.g. 'layer_idx')"""
        index = {}
        # On indexe par layer_index pour un accès rapide "get_layer_fragments(i)"
        # et aussi des entrées spéciales pour "embedding" (-1) et "output" (-2 ou autre code)

        for f in self.manifest["fragments"]:
            # On convertit le dict en objet propre
            meta = FragmentMeta(
                id=f["fragment_id"],
                type=f["fragment_type"],
                layer=f["layer_index"],
                component=f["component"],
                shard_index=f["shard_index"],
                total_shards=f["total_shards"],
                tensor_name=f["tensor_name"],
                size_bytes=f["size_bytes"]
            )

            # Key: layer index (int)
            lid = meta.layer
            if lid not in index:
                index[lid] = []
            index[lid].append(meta)

        return index

    def _load_fragment_data(self, fragment: FragmentMeta) -> bytes:
        """Simulate loading data. Returns bytes."""
        path = self.fragments_dir / f"{fragment.id}.dat"
        if not path.exists():
            print(f"⚠️ Fragment manquant : {path}")
            return b""

        # Lecture réelle pour prouver l'IO
        size = path.stat().st_size
        with open(path, "rb") as f:
            # On lit tout ou juste un header pour aller vite ?
            # Pour le "Vrai" simulateur, on lit tout pour voir le débit disque.
            data = f.read()
            return data

    def generate(self, prompt: str, max_tokens: int = 5):
        print(f"🚀 Début de l'inférence pour le prompt : \"{prompt}\"")

        # 1. Tokenization (Dummy)
        tokens = prompt.split()
        if not tokens: tokens = ["<start>"]
        print(f"🔤 Tokens initiaux ({len(tokens)}) : {tokens}")

        # Simulation de la prédiction token par token
        # On commence après le prompt

        start_time = time.time()

        for i in range(max_tokens):
            token_step_start = time.time()
            print(f"\n⚡ Génération token {i+1}/{max_tokens} ...")

            # A. Embedding Lookup
            # Layer -1 ou "embedding"
            self._process_layer(-1, "Embedding")

            # B. Layers Transformer
            for layer_idx in range(self.n_layers):
                # On simule le passage dans chaque couche
                # En P2P, on téléchargerait/chargerait les fragments de cette couche.
                self._process_layer(layer_idx, f"Block {layer_idx}")

            # C. Output Head
            # Layer -2 ou "lm_head" / "output"
            # Souvent layer_index dans mon fragmenter pour lm_head est max_layer ou -2?
            # Vérifions dans l'index.
            # On suppose qu'il est indexé. Sinon on cherche.
            self._process_layer(-2, "LM Head")

            # Fin du token
            dt = time.time() - token_step_start
            print(f"   ⏱️  Token généré en {dt:.2f}s")

        total_time = time.time() - start_time
        print(f"\n🏁 Inférence terminée. Temps total : {total_time:.2f}s ({max_tokens/total_time:.2f} tokens/s)")

    def _process_layer(self, layer_idx: int, label: str):
        """Traite une couche : récupère les fragments et simule le chargement."""
        fragments = self.fragments_index.get(layer_idx, [])
        if not fragments:
            # Essai de fallback pour lm_head qui peut être mal indexé si logic fragmenter a changé
            # Si layer_idx correspond à 'output'
            return

        # Calcul du volume de données "utilisé"
        total_bytes = 0
        loaded_count = 0

        t0 = time.time()
        for frag in fragments:
            # Dans un vrai système P2P MoE, on ne chargerait que les experts actifs.
            # Ici on simule un modèle dense ou on charge tout (pire cas).

            # Si c'est un expert, on simule une activation éparse (ex: 2 experts sur 8)
            if "expert" in frag.component or "ffn" in frag.component:
                # Simplification: on charge tout pour ce test fonctionnel de DISQUE
                pass

            _ = self._load_fragment_data(frag)
            total_bytes += frag.size_bytes
            loaded_count += 1

        dt = time.time() - t0
        mb = total_bytes / (1024*1024)
        print(f"   [{label}] {loaded_count} fragments chargés ({mb:.1f} MB) en {dt*1000:.1f}ms -> {mb/dt if dt>0 else 0:.1f} MB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Functional Simulator using Real Fragments")
    parser.add_argument("fragments_dir", help="Path to fragments directory")
    parser.add_argument("--prompt", type=str, default="Hello AI", help="Prompt to simulate")
    parser.add_argument("--tokens", type=int, default=3, help="Number of tokens to generate")

    args = parser.parse_args()

    sim = FunctionalSimulator(args.fragments_dir)
    sim.generate(args.prompt, max_tokens=args.tokens)
