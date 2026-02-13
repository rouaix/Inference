"""
distribution/local.py
=====================
Chargement LOCAL des fragments — stratégie par défaut.

Les fragments sont lus directement depuis le système de fichiers local,
dans le dossier contenant le manifest.json et les fichiers .dat.

C'est la stratégie utilisée dans p2p_inference.py (P2PInferenceEngine.load_tensor).
Ce module la factorise en une classe réutilisable indépendante du moteur.

Usage
-----
    from distribution.local import LocalFragmentLoader

    loader = LocalFragmentLoader("models/tinyllama_q8_fragments_v2")
    tensor = loader.load_tensor("blk.0.attn_q.weight")
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np


# ---------------------------------------------------------------------------
# Interface commune (contrat attendu par tous les loaders)
# ---------------------------------------------------------------------------

class BaseFragmentLoader:
    """
    Interface de base que chaque stratégie de distribution doit respecter.

    Toute sous-classe doit implémenter :
      - load_raw(fragment_id)  → bytes
      - load_tensor(tensor_name) → np.ndarray
    """

    def load_raw(self, fragment_id: str) -> bytes:
        raise NotImplementedError

    def load_tensor(self, tensor_name: str) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implémentation locale
# ---------------------------------------------------------------------------

class LocalFragmentLoader(BaseFragmentLoader):
    """
    Charge les fragments depuis un dossier local.

    Paramètres
    ----------
    fragments_dir : str | Path
        Dossier contenant manifest.json et les fichiers .dat.
    verbose : bool
        Affiche des informations de débogage lors du chargement.
    """

    def __init__(self, fragments_dir, verbose: bool = False):
        self.fragments_dir = Path(fragments_dir)
        self.verbose = verbose

        manifest_path = self.fragments_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest introuvable : {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        # Index : tensor_name → liste de fragments triés par shard_index
        self.fragments_map: Dict[str, List[dict]] = {}
        for frag in self.manifest.get("fragments", []):
            tname = frag.get("tensor_name")
            if tname:
                self.fragments_map.setdefault(tname, []).append(frag)

        for tname in self.fragments_map:
            self.fragments_map[tname].sort(key=lambda x: x["shard_index"])

        if self.verbose:
            print(f"[LocalFragmentLoader] {len(self.fragments_map)} tenseurs indexés depuis {self.fragments_dir}")

    # ------------------------------------------------------------------
    # Méthode bas niveau : lecture brute d'un fragment
    # ------------------------------------------------------------------

    def load_raw(self, fragment_id: str) -> bytes:
        """
        Retourne les octets bruts du fichier .dat correspondant à fragment_id.

        Paramètres
        ----------
        fragment_id : str
            Identifiant du fragment (sans extension .dat).

        Retourne
        --------
        bytes
            Contenu binaire du fragment.
        """
        path = self.fragments_dir / f"{fragment_id}.dat"
        if not path.exists():
            raise FileNotFoundError(f"Fragment manquant : {path}")
        with open(path, "rb") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Méthode haut niveau : reconstitution + dequantisation d'un tenseur
    # ------------------------------------------------------------------

    def load_tensor(self, tensor_name: str) -> np.ndarray:
        """
        Reconstitue et dequantise un tenseur à partir de ses fragments locaux.

        Gère les formats :
          - float32 / float16 / int32  → conversion directe
          - Q8_0 (GGUF)               → dequantisation par blocs de 32

        Convention de layout Q8_0 (GGUF)
        ---------------------------------
        Les données physiques sont stockées en [out_dim, in_dim].
        La shape logique dans le manifest est [in_dim, out_dim].
        On reshape en [out_dim, in_dim] puis on transpose → [in_dim, out_dim].
        Cf. CLAUDE.md § "Q8_0 Dequantization — Transposed Physical Layout".

        Paramètres
        ----------
        tensor_name : str
            Nom GGUF du tenseur (ex : "blk.0.attn_q.weight").

        Retourne
        --------
        np.ndarray  dtype=float32
            Tenseur dequantisé prêt à l'emploi.
        """
        fragments = self.fragments_map.get(tensor_name)

        if not fragments:
            if self.verbose:
                print(f"[LocalFragmentLoader] ⚠️  Tenseur manquant : {tensor_name} — retour aléatoire")
            return np.random.normal(0, 0.01, size=(64, 64)).astype(np.float32)

        if self.verbose:
            print(f"[LocalFragmentLoader] 📂 '{tensor_name}' — {len(fragments)} fragment(s)")

        # --- Reconstitution des octets bruts ---
        raw = bytearray()
        for frag in fragments:
            raw.extend(self.load_raw(frag["fragment_id"]))
        data = bytes(raw)

        # --- Métadonnées du premier fragment ---
        frag0 = fragments[0]
        dtype_str = frag0["dtype"]
        shape = tuple(frag0["shape"])
        tensor_type = frag0.get("tensor_type", "")

        # --- Décodage selon le type ---
        if "float" in dtype_str or "int32" in dtype_str:
            arr = np.frombuffer(data, dtype=dtype_str).reshape(shape)
            return arr.astype(np.float32)

        if "Q8_0" in tensor_type:
            return self._dequantize_q8_0(data, shape)

        # Format non géré : retourner des zéros de la bonne forme
        if self.verbose:
            print(f"[LocalFragmentLoader] ⚠️  Type non géré '{tensor_type}' pour '{tensor_name}'")
        return np.zeros(shape, dtype=np.float32)

    # ------------------------------------------------------------------
    # Dequantisation Q8_0
    # ------------------------------------------------------------------

    def _dequantize_q8_0(self, data: bytes, shape: tuple) -> np.ndarray:
        """
        Dequantise des données Q8_0 GGUF.

        Structure d'un bloc Q8_0 (34 octets) :
          - 2 octets : delta (float16)
          - 32 octets : 32 entiers int8

        La valeur dequantisée de chaque élément est : delta * int8_value
        """
        if len(data) % 34 != 0:
            if self.verbose:
                print(f"[LocalFragmentLoader] ❌ Taille de données Q8_0 invalide ({len(data)} octets)")
            return np.zeros(shape, dtype=np.float32)

        dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
        blocks = np.frombuffer(data, dtype=dt)

        d = blocks["d"].astype(np.float32)[:, None]   # [n_blocks, 1]
        qs = blocks["qs"].astype(np.float32)           # [n_blocks, 32]
        decoded = (d * qs).flatten()

        # Correction du layout transposé GGUF Q8_0
        if len(shape) == 2:
            out_dim = shape[-1]
            in_dim = shape[0]
            return decoded.reshape([out_dim, in_dim]).T.astype(np.float32)

        return decoded.reshape(shape).astype(np.float32)

    # ------------------------------------------------------------------
    # Informations utilitaires
    # ------------------------------------------------------------------

    def list_tensors(self) -> List[str]:
        """Retourne la liste de tous les noms de tenseurs disponibles."""
        return list(self.fragments_map.keys())

    def tensor_info(self, tensor_name: str) -> Optional[dict]:
        """Retourne les métadonnées du premier fragment d'un tenseur, ou None."""
        frags = self.fragments_map.get(tensor_name)
        return frags[0] if frags else None


# ---------------------------------------------------------------------------
# CLI de test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test du chargement local de fragments")
    parser.add_argument("fragments_dir", help="Dossier contenant manifest.json et les .dat")
    parser.add_argument("--tensor", default="blk.0.attn_q.weight", help="Tenseur à charger")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    loader = LocalFragmentLoader(args.fragments_dir, verbose=args.verbose)
    print(f"Tenseurs disponibles : {len(loader.list_tensors())}")

    t = loader.load_tensor(args.tensor)
    print(f"Tenseur '{args.tensor}' : shape={t.shape} dtype={t.dtype}")
    print(f"  Min={t.min():.4f}  Max={t.max():.4f}  Mean={t.mean():.4f}  Std={t.std():.4f}")
