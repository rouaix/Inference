"""
Distributed MoE Inference Simulator v2
Simule l'inférence distribuée d'un modèle MoE (Mistral Large 3).

Avantage clé du MoE distribué :
Pour chaque token, seuls ~2 experts sur 128 sont activés.
→ On ne mobilise qu'une fraction du réseau à chaque étape.
→ Le reste des nœuds est en veille (économie de bande passante + CPU).
"""
# pip install gguf numpy

import time
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict

from fragmenter_v2 import (
    FragmentMeta, FragmentType, MoEFragmenter,
    SimulatedMoEModel, MistralLarge3Config, CHUNK_SIZE
)


# ============================================================
# Node
# ============================================================

@dataclass
class Node:
    """Nœud P2P hébergeant un fragment de 10 Mo."""
    node_id: str
    fragment_meta: FragmentMeta
    fragment_data: Optional[np.ndarray]
    is_alive: bool = True
    latency_ms: float = 50.0
    compute_time_ms: float = 0.0   # Temps de calcul simulé
    total_computations: int = 0

    def compute_partial(self, activations: np.ndarray) -> np.ndarray:
        """Calcul partiel sur le fragment."""
        if not self.is_alive:
            raise ConnectionError(f"Node {self.node_id} offline")

        self.total_computations += 1

        # Mode virtuel (pas de données)
        if self.fragment_data is None:
            # Simulation de latence de calcul si besoin
            return activations

        # Simulation : produit element-wise tronqué
        min_len = min(len(activations), len(self.fragment_data))
        result = (activations[:min_len].astype(np.float32) *
                  self.fragment_data[:min_len].astype(np.float32))
        return result

    def heartbeat(self) -> bool:
        return self.is_alive


# ============================================================
# MoE-Aware DHT
# ============================================================

class MoEDHT:
    """
    DHT optimisée pour le MoE.
    Sépare les nœuds "toujours actifs" des nœuds "experts".
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

        # Index rapide par type
        self.always_active: Dict[Tuple[int, str, int], List[str]] = defaultdict(list)
        self.expert_nodes: Dict[Tuple[int, int, str, int], List[str]] = defaultdict(list)

        # Stats
        self.total_registered = 0

    def register_node(self, node: Node):
        meta = node.fragment_meta
        self.nodes[node.node_id] = node
        self.total_registered += 1

        if meta.is_always_active:
            key = (meta.layer_index, meta.component, meta.shard_index)
            self.always_active[key].append(node.node_id)
        else:
            key = (meta.layer_index, meta.expert_index, meta.component, meta.shard_index)
            self.expert_nodes[key].append(node.node_id)

    def find_alive_for_active(self, layer: int, component: str,
                               shard: int) -> Optional[Node]:
        """Trouve un nœud vivant pour un fragment toujours actif."""
        key = (layer, component, shard)
        for nid in self.always_active.get(key, []):
            node = self.nodes.get(nid)
            if node and node.is_alive:
                return node
        return None

    def find_alive_for_expert(self, layer: int, expert_idx: int,
                               component: str, shard: int) -> Optional[Node]:
        """Trouve un nœud vivant pour un expert spécifique."""
        key = (layer, expert_idx, component, shard)
        for nid in self.expert_nodes.get(key, []):
            node = self.nodes.get(nid)
            if node and node.is_alive:
                return node
        return None

    def get_expert_shards(self, layer: int, expert_idx: int,
                          component: str) -> int:
        """Nombre de shards pour un composant d'un expert."""
        max_shard = -1
        for (l, e, c, s) in self.expert_nodes:
            if l == layer and e == expert_idx and c == component:
                max_shard = max(max_shard, s)
        return max_shard + 1 if max_shard >= 0 else 0

    def get_active_shards(self, layer: int, component: str) -> int:
        """Nombre de shards pour un composant toujours actif."""
        max_shard = -1
        for (l, c, s) in self.always_active:
            if l == layer and c == component:
                max_shard = max(max_shard, s)
        return max_shard + 1 if max_shard >= 0 else 0

    def get_layers(self) -> List[int]:
        """Liste ordonnée de toutes les couches."""
        layers = set()
        for (l, _, _) in self.always_active:
            layers.add(l)
        for (l, _, _, _) in self.expert_nodes:
            layers.add(l)
        return sorted(layers)

    def get_layer_active_components(self, layer: int) -> List[str]:
        """Composants toujours actifs d'une couche."""
        comps = set()
        for (l, c, s) in self.always_active:
            if l == layer:
                comps.add(c)
        return sorted(comps)

    def get_layer_experts(self, layer: int) -> List[int]:
        """Experts disponibles pour une couche."""
        experts = set()
        for (l, e, c, s) in self.expert_nodes:
            if l == layer:
                experts.add(e)
        return sorted(experts)

    def get_expert_components(self, layer: int, expert_idx: int) -> List[str]:
        """Composants d'un expert."""
        comps = set()
        for (l, e, c, s) in self.expert_nodes:
            if l == layer and e == expert_idx:
                comps.add(c)
        return sorted(comps)

    def health_check(self) -> dict:
        total = len(self.nodes)
        alive = sum(1 for n in self.nodes.values() if n.is_alive)

        # Fragments actifs orphelins (critique)
        active_orphaned = 0
        for key, nids in self.always_active.items():
            if not any(self.nodes[nid].is_alive for nid in nids if nid in self.nodes):
                active_orphaned += 1

        # Experts orphelins (moins critique — on peut rerouter)
        expert_orphaned = 0
        for key, nids in self.expert_nodes.items():
            if not any(self.nodes[nid].is_alive for nid in nids if nid in self.nodes):
                expert_orphaned += 1

        return {
            "total_nodes": total,
            "alive": alive,
            "dead": total - alive,
            "health_pct": (alive / total * 100) if total > 0 else 0,
            "active_orphaned": active_orphaned,
            "expert_orphaned": expert_orphaned,
            "can_run_inference": active_orphaned == 0,
            "degraded": expert_orphaned > 0,
        }


# ============================================================
# MoE Network Simulator
# ============================================================

class MoENetworkSimulator:
    """
    Simule le réseau P2P pour un modèle MoE.

    Logique d'inférence :
    1. Embedding (toujours actif)
    2. Pour chaque couche :
       a. Attention Q/K/V/O (toujours actif)
       b. Router choisit 2 experts
       c. Seuls ces 2 experts sont contactés
       d. Agrégation pondérée
    3. LM Head → token
    """

    def __init__(self, replication_factor: int = 3):
        self.dht = MoEDHT()
        self.replication = replication_factor
        self.model_config: Optional[MistralLarge3Config] = None

    def load_fragments(self, fragments: List[Tuple[FragmentMeta, np.ndarray]],
                       config: MistralLarge3Config):
        self.model_config = config
        node_count = 0

        print(f"\n🌐 Création du réseau MoE P2P (réplication ×{self.replication})...")

        for meta, data in fragments:
            for r in range(self.replication):
                node = Node(
                    node_id=f"node_{node_count:06d}",
                    fragment_meta=meta,
                    fragment_data=data.copy() if data is not None else None,
                    is_alive=True,
                    latency_ms=random.uniform(10, 300),
                )
                self.dht.register_node(node)
                node_count += 1

        active_frags = sum(1 for m, _ in fragments if m.is_always_active)
        expert_frags = sum(1 for m, _ in fragments if not m.is_always_active)

        print(f"   → {node_count} nœuds créés")
        print(f"   → {active_frags} fragments actifs × {self.replication} = "
              f"{active_frags * self.replication} nœuds critiques")
        print(f"   → {expert_frags} fragments experts × {self.replication} = "
              f"{expert_frags * self.replication} nœuds experts")

    def kill_random_nodes(self, kill_rate: float) -> int:
        nodes = list(self.dht.nodes.values())
        num_kill = int(len(nodes) * kill_rate)
        victims = random.sample(nodes, min(num_kill, len(nodes)))
        for n in victims:
            n.is_alive = False
        return len(victims)

    def revive_all(self):
        for n in self.dht.nodes.values():
            n.is_alive = True

    def run_inference(self, input_tokens: List[int], max_new_tokens: int = 3,
                      verbose: bool = True) -> dict:
        """
        Inférence MoE distribuée.
        Retourne des stats détaillées sur la requête.
        """
        health = self.dht.health_check()
        stats = {
            "success": False,
            "tokens_generated": 0,
            "total_nodes_contacted": 0,
            "expert_selections": [],
            "failovers": 0,
            "errors": 0,
        }

        if not health["can_run_inference"]:
            if verbose:
                print(f"❌ {health['active_orphaned']} fragments actifs orphelins")
            return stats

        layers = self.dht.get_layers()
        # Filtrer embedding (-1) et lm_head (-2) des couches transformer
        transformer_layers = [l for l in layers if l >= 0]

        generated = list(input_tokens)
        total_contacted = 0
        total_failovers = 0

        if verbose:
            print(f"\n🚀 Inférence MoE distribuée")
            print(f"   Couches transformer: {len(transformer_layers)}")
            n_experts = self.model_config.num_experts if self.model_config else "?"
            n_active = self.model_config.num_active_experts if self.model_config else "?"
            print(f"   Experts: {n_active} actifs / {n_experts} total par couche")

        for step in range(max_new_tokens):
            activation = np.random.randn(128).astype(np.float32)
            step_contacted = 0
            step_experts = []

            # ---- Embedding ----
            contacted, fo = self._compute_component(-1, "embedding", activation)
            step_contacted += contacted
            total_failovers += fo

            # ---- Transformer layers ----
            for layer_idx in transformer_layers:
                # Attention (toujours actif)
                for comp in self.dht.get_layer_active_components(layer_idx):
                    contacted, fo = self._compute_component(layer_idx, comp, activation)
                    step_contacted += contacted
                    total_failovers += fo

                # Router → sélection des experts
                available_experts = self.dht.get_layer_experts(layer_idx)
                if available_experts:
                    n_active = self.model_config.num_active_experts if self.model_config else 2
                    # Simuler la sélection du routeur
                    selected = random.sample(
                        available_experts,
                        min(n_active, len(available_experts))
                    )
                    step_experts.extend([(layer_idx, e) for e in selected])

                    # Calculer seulement les experts sélectionnés
                    for exp_idx in selected:
                        for comp in self.dht.get_expert_components(layer_idx, exp_idx):
                            contacted, fo = self._compute_expert(
                                layer_idx, exp_idx, comp, activation)
                            step_contacted += contacted
                            total_failovers += fo

            # ---- LM Head ----
            contacted, fo = self._compute_component(-2, "lm_head", activation)
            step_contacted += contacted

            # Générer token
            next_token = random.randint(0, 32000)
            generated.append(next_token)
            total_contacted += step_contacted

            if verbose:
                n_expert_layers = len(set(l for l, _ in step_experts))
                print(f"   Token {step+1}: {step_contacted} nœuds contactés, "
                      f"{len(step_experts)} experts activés sur {n_expert_layers} couches")

        stats["success"] = True
        stats["tokens_generated"] = max_new_tokens
        stats["total_nodes_contacted"] = total_contacted
        stats["expert_selections"] = []  # Trop verbeux pour le résumé
        stats["failovers"] = total_failovers

        if verbose:
            print(f"\n   ✅ {max_new_tokens} tokens générés")
            print(f"   📡 {total_contacted} nœuds contactés au total")
            print(f"   🔄 {total_failovers} failovers")
            nodes_per_token = total_contacted / max_new_tokens
            print(f"   📊 ~{nodes_per_token:.0f} nœuds/token en moyenne")

        return stats

    def _compute_component(self, layer: int, component: str,
                           activation: np.ndarray) -> Tuple[int, int]:
        """Calcul sur un composant toujours actif. Retourne (contacted, failovers)."""
        num_shards = self.dht.get_active_shards(layer, component)
        contacted = 0
        failovers = 0

        for shard in range(num_shards):
            node = self.dht.find_alive_for_active(layer, component, shard)
            if node:
                try:
                    node.compute_partial(activation)
                    contacted += 1
                except ConnectionError:
                    failovers += 1

        return contacted, failovers

    def _compute_expert(self, layer: int, expert_idx: int,
                        component: str, activation: np.ndarray) -> Tuple[int, int]:
        """Calcul sur un expert. Retourne (contacted, failovers)."""
        num_shards = self.dht.get_expert_shards(layer, expert_idx, component)
        contacted = 0
        failovers = 0

        for shard in range(num_shards):
            node = self.dht.find_alive_for_expert(layer, expert_idx, component, shard)
            if node:
                try:
                    node.compute_partial(activation)
                    contacted += 1
                except ConnectionError:
                    failovers += 1
            else:
                # Expert non disponible — en MoE, on peut dégrader gracieusement
                # en donnant plus de poids aux autres experts
                failovers += 1

        return contacted, failovers


# ============================================================
# Fault Tolerance Tester
# ============================================================

class MoEFaultTester:
    """Teste la résilience du réseau MoE."""

    def __init__(self, sim: MoENetworkSimulator):
        self.sim = sim

    def run_test(self, kill_rates: List[float] = None):
        if kill_rates is None:
            kill_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        input_tokens = [1, 42, 100]
        results = []

        print("\n" + "=" * 70)
        print("🧪 TEST DE TOLÉRANCE AUX PANNES — MoE")
        print("=" * 70)

        for rate in kill_rates:
            self.sim.revive_all()
            killed = self.sim.kill_random_nodes(rate)
            health = self.sim.dht.health_check()
            inf = self.sim.run_inference(input_tokens, max_new_tokens=2, verbose=False)

            result = {
                "kill_rate": rate,
                "killed": killed,
                "alive_pct": health["health_pct"],
                "active_orphaned": health["active_orphaned"],
                "expert_orphaned": health["expert_orphaned"],
                "success": inf["success"],
                "nodes_contacted": inf["total_nodes_contacted"],
                "degraded": health["degraded"],
            }
            results.append(result)

            icon = "✅" if inf["success"] and not health["degraded"] else \
                   "⚡" if inf["success"] and health["degraded"] else "❌"
            status = "OK" if not health["degraded"] else \
                     f"dégradé ({health['expert_orphaned']} experts perdus)" \
                     if inf["success"] else "ÉCHOUÉ"

            print(f"\n{icon} Kill {rate*100:.0f}% ({killed} nœuds)")
            print(f"   Vivants: {health['alive']}/{health['total_nodes']} "
                  f"({health['health_pct']:.1f}%)")
            print(f"   Actifs orphelins: {health['active_orphaned']}  "
                  f"Experts orphelins: {health['expert_orphaned']}")
            print(f"   Statut: {status}")

        # Résumé
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ MoE")
        print("=" * 70)
        print(f"{'Kill%':>6} {'Vivants%':>9} {'Actifs⚠':>9} {'Experts⚠':>10} {'Statut':>12}")
        print("-" * 50)
        for r in results:
            icon = "✅" if r["success"] and not r["degraded"] else \
                   "⚡ dégradé" if r["success"] else "❌ FAIL"
            print(f"{r['kill_rate']*100:>5.0f}% {r['alive_pct']:>8.1f}% "
                  f"{r['active_orphaned']:>9} {r['expert_orphaned']:>10} {icon:>12}")

        max_ok = max((r["kill_rate"] for r in results
                      if r["success"] and not r["degraded"]), default=0)
        max_degraded = max((r["kill_rate"] for r in results
                           if r["success"]), default=0)

        print(f"\n🛡️  Max sans dégradation: {max_ok*100:.0f}%")
        print(f"⚡  Max en mode dégradé:  {max_degraded*100:.0f}%")
        print(f"   (réplication ×{self.sim.replication})")

        return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoE Simulator v2")
    parser.add_argument("--model", default="mistral-large-3",
                        choices=["micro-moe", "mini-moe", "mistral-large-3"])
    parser.add_argument("--replication", type=int, default=3)
    parser.add_argument("--fault-test", action="store_true")
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--kill-rate", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 70)
    print("MoE Distributed Inference Simulator v2")
    print("=" * 70)

    # Build model
    if args.model == "mistral-large-3":
        print(f"⚠️ Mode virtuel pour {args.model} (trop gros pour la RAM)")
        cfg = SimulatedMoEModel.CONFIGS[args.model]
        # On ne construit pas le modèle (SimulatedMoEModel), juste la config
        model_config = cfg
        fragmenter = MoEFragmenter(chunk_size=CHUNK_SIZE)
        fragments = fragmenter.generate_virtual_fragments(cfg)
    else:
        model = SimulatedMoEModel(config_name=args.model)
        model_config = model.config
        fragmenter = MoEFragmenter(chunk_size=CHUNK_SIZE)
        fragments = fragmenter.fragment_model(model)

    # Build network
    sim = MoENetworkSimulator(replication_factor=args.replication)
    sim.load_fragments(fragments, model_config)

    if args.fault_test:
        tester = MoEFaultTester(sim)
        tester.run_test()
    else:
        if args.kill_rate > 0:
            killed = sim.kill_random_nodes(args.kill_rate)
            print(f"\n💀 {killed} nœuds tués ({args.kill_rate*100:.0f}%)")

        health = sim.dht.health_check()
        print(f"\n📊 Réseau: {health['alive']}/{health['total_nodes']} nœuds "
              f"({health['health_pct']:.1f}%)")

        sim.run_inference([1, 42, 100], max_new_tokens=args.tokens)
