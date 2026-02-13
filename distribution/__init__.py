"""
distribution/
=============
Ce package définit les différentes stratégies de chargement des fragments
pour le moteur d'inférence P2P.

Modules disponibles :
  - local   : chargement depuis le système de fichiers local (implémenté)
  - reseau  : chargement depuis un serveur HTTP/API centralisé (à coder)
  - p2p     : chargement distribué via un réseau pair-à-pair (à coder)

Interface commune (BaseFragmentLoader)
---------------------------------------
Chaque module expose une classe qui hérite de BaseFragmentLoader et
implémente les deux méthodes suivantes :

    load_raw(fragment_id: str) -> bytes
        Récupère les octets bruts d'un fragment identifié par son fragment_id.

    load_tensor(tensor_name: str, fragments_map: dict, fragments_dir_or_context) -> np.ndarray
        Reconstitue et dequantise un tenseur à partir de ses fragments.

Usage typique
-------------
    from distribution.local import LocalFragmentLoader
    loader = LocalFragmentLoader("models/tinyllama_q8_fragments_v2")
    data = loader.load_raw("tinyllama_L0_attn_q_S0_abc123")
"""

from .local import LocalFragmentLoader

__all__ = ["LocalFragmentLoader"]
