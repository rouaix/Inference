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
    cache_raw : bool
        Si True, conserve les octets bruts de chaque tenseur en mémoire après
        le premier chargement (évite les relectures disque pour le decode).
        Beaucoup moins coûteux en RAM que cache_weights=True sur P2PInferenceEngine
        (4.5x plus compact que float32 pour Q4_K).
    """

    def __init__(self, fragments_dir, verbose: bool = False, cache_raw: bool = False):
        self.fragments_dir = Path(fragments_dir)
        self.verbose = verbose
        # Cache des octets bruts par tenseur (évite la relecture disque)
        self._raw_cache: Optional[Dict[str, bytes]] = {} if cache_raw else None

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
                print(f"[LocalFragmentLoader] [WARN] Tenseur manquant : {tensor_name} — retour aléatoire")
            return np.random.normal(0, 0.01, size=(64, 64)).astype(np.float32)

        if self.verbose:
            print(f"[LocalFragmentLoader] [FILE] '{tensor_name}' — {len(fragments)} fragment(s)")

        # --- Métadonnées du premier fragment ---
        frag0 = fragments[0]
        dtype_str = frag0["dtype"]
        shape = tuple(frag0["shape"])
        tensor_type = frag0.get("tensor_type", "")

        # --- Reconstitution des octets bruts (avec cache optionnel) ---
        if self._raw_cache is not None and tensor_name in self._raw_cache:
            data = self._raw_cache[tensor_name]
        else:
            data = b''.join(self.load_raw(frag["fragment_id"]) for frag in fragments)
            if self._raw_cache is not None:
                self._raw_cache[tensor_name] = data

        # --- Décodage selon le type ---
        if "float" in dtype_str or "int32" in dtype_str:
            arr = np.frombuffer(data, dtype=dtype_str).reshape(shape)
            return arr.astype(np.float32)

        # Utiliser la déquantization centralisée pour tous les formats quantifiés
        return self._dequantize_tensor(data, tensor_type, shape)

    # ------------------------------------------------------------------
    # Chargement brut (sans dequantisation) — pour le GEMV fusionné
    # ------------------------------------------------------------------

    def load_raw_tensor(self, tensor_name: str):
        """
        Retourne (raw_bytes, tensor_type, logical_shape) sans dequantiser.

        Utilisé par FragmentExecutor pour le GEMV fusionné Q4_K (seq_len=1).
        Si cache_raw=True, les octets sont mis en cache après le premier appel.
        """
        fragments = self.fragments_map.get(tensor_name)
        if not fragments:
            return None, "", ()

        frag0 = fragments[0]
        shape = tuple(frag0["shape"])
        tensor_type = frag0.get("tensor_type", "")

        if self._raw_cache is not None and tensor_name in self._raw_cache:
            data = self._raw_cache[tensor_name]
        else:
            data = b''.join(self.load_raw(frag["fragment_id"]) for frag in fragments)
            if self._raw_cache is not None:
                self._raw_cache[tensor_name] = data

        return data, tensor_type, shape

    # ------------------------------------------------------------------
    # Dequantisation (centralisée via module dequantize)
    # ------------------------------------------------------------------

    def _dequantize_tensor(self, data: bytes, tensor_type: str, shape: tuple) -> np.ndarray:
        """
        Dequantise des données en utilisant le module dequantize centralisé.
        Gère tous les formats : Q4_K, Q6_K, Q8_0, F32, F16.
        """
        try:
            from dequantize import dequantize
            return dequantize(data, tensor_type, shape)
        except ImportError:
            if self.verbose:
                print(f"[LocalFragmentLoader] [WARN] Module dequantize non disponible, retour à l'ancienne méthode")
            # Fallback to old Q8_0 implementation for backward compatibility
            if tensor_type == "Q8_0":
                return self._dequantize_q8_0_legacy(data, shape)
            else:
                return np.zeros(shape, dtype=np.float32)
        except NotImplementedError as e:
            if self.verbose:
                print(f"[LocalFragmentLoader] [WARN] Format non supporté: {e}")
            return np.zeros(shape, dtype=np.float32)
        except Exception as e:
            if self.verbose:
                print(f"[LocalFragmentLoader] [ERROR] Erreur de déquantization: {e}")
            return np.zeros(shape, dtype=np.float32)

    def _dequantize_q8_0_legacy(self, data: bytes, shape: tuple) -> np.ndarray:
        """
        Dequantisation Q8_0 (méthode legacy pour fallback).
        À supprimer une fois que tout le monde utilise le module dequantize.
        """
        if len(data) % 34 != 0:
            if self.verbose:
                print(f"[LocalFragmentLoader] [ERROR] Taille de données Q8_0 invalide ({len(data)} octets)")
            return np.zeros(shape, dtype=np.float32)

        dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
        blocks = np.frombuffer(data, dtype=dt)

        d = blocks["d"].astype(np.float32)[:, None]
        qs = blocks["qs"].astype(np.float32)
        decoded = (d * qs).flatten()

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
