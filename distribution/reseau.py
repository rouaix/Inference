"""
distribution/reseau.py
======================
Exécution RÉSEAU des couches — inférence distribuée via HTTP.

STATUT : À CODER — Ce module est un stub documenté.
         Les interfaces sont définies ; l'implémentation est vide.

Concept corrigé
---------------
Les fragments ne sont PAS téléchargés côté client.
Chaque nœud distant conserve ses propres fragments et exécute les calculs.
Seules les ACTIVATIONS (hidden states, ~dim × seq_len × 4 octets) voyagent sur le réseau.

Ancien concept (FAUX)         Concept correct
─────────────────────         ──────────────────────────────────────────
Client télécharge les .dat    Client envoie le hidden state
→ dequantise localement       → Nœud exécute forward() avec ses fragments
→ calcule en local            → Nœud retourne le hidden state résultat

Pourquoi c'est important :
  - Poids d'une couche Magistral Q4_K : ~230 MB déquantisés → inacceptable à transmettre
  - Activations d'une couche : dim × seq_len × 4 octets = 5120 × 1 × 4 = 20 KB → négligeable
  - Les fragments restent distribués : ni le client ni aucun nœud n'a le modèle complet
  - La confidentialité des poids est garantie par construction

Architecture réseau
-------------------

    p2p_inference.py (moteur client)
         │
         │  Pour chaque couche i :
         │  POST http://node_i/execute_layer
         │  Body : { layer_idx, hidden_state, pos, cache_k, cache_v }
         ▼
    Nœud distant (RemoteNodeServer — futur module server.py)
         │  ← charge ses fragments locaux (LocalFragmentLoader ou FragmentExecutor)
         │  ← exécute LlamaLayer.forward(x, cache_k, cache_v, pos)
         │  → sérialise output + new_k + new_v
         ▼
    Client reçoit { output, new_k, new_v }
         │
         │  → envoie output au nœud suivant (couche i+1)
         │  ...

Protocole HTTP cible
--------------------

  POST /execute_layer
    Body JSON :
      {
        "layer_idx": 0,
        "hidden_state": [[...float...]],  # float32, shape [seq_len, dim]
        "pos":  42,                        # position dans la séquence (pour RoPE + masque)
        "cache_k": null | [[...]],         # float32, shape [past_len, n_kv_heads, head_dim]
        "cache_v": null | [[...]]
      }
    Réponse JSON (200 OK) :
      {
        "output": [[...]],    # float32, shape [seq_len, dim]
        "new_k":  [[...]],    # float32, shape [seq_len + past_len, n_kv_heads, head_dim]
        "new_v":  [[...]]
      }

  GET /status
    Réponse JSON :
      {
        "node_id": "node-abc123",
        "layers":  [0, 1, 2],     # indices de couches gérés par ce nœud
        "model":   "Magistral-Small-2509-Q4_K_M",
        "ready":   true
      }

  GET /manifest
    Réponse JSON : manifest.json tel que généré par fragments/fragmenter.py
    (métadonnées uniquement, sans les octets des tenseurs)

Optimisations futures
---------------------
  1. Sérialisation binaire (numpy tobytes + Content-Type: application/octet-stream)
     au lieu de JSON pour réduire latence et CPU de sérialisation
  2. Pipeline asynchrone : envoyer la couche i+1 pendant que i est en cours
     (asyncio + aiohttp)
  3. Compression zstd des activations pour connexions lentes
  4. Authentification : token Bearer dans le header Authorization
  5. Retry avec backoff exponentiel sur timeout réseau
  6. Routage automatique : le client interroge un nœud coordinateur
     pour découvrir quel nœud héberge quelle couche

Dépendances à ajouter dans requirements.txt
--------------------------------------------
  requests>=2.31          # HTTP synchrone (client)
  aiohttp>=3.9            # HTTP asynchrone (optionnel, pipeline)
  fastapi>=0.110          # Serveur REST sur le nœud distant (futur server.py)
  uvicorn>=0.29           # Serveur ASGI pour FastAPI
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import base64

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False
    zstd = None


# ---------------------------------------------------------------------------
# Interface commune pour l'exécution d'une couche (locale ou distante)
# ---------------------------------------------------------------------------

class BaseLayerExecutor:
    """
    Interface abstraite : exécute une couche du transformer.

    Implémentations :
      - LocalLayerExecutor   → charge les fragments localement, calcule en local
                               (à créer dans distribution/local.py ou inference/)
      - RemoteLayerExecutor  → envoie les activations à un nœud distant via HTTP
                               (ce module)
      - P2PLayerRouter       → route vers le bon nœud selon l'index de couche
                               (futur distribution/p2p.py)
    """

    def execute_layer(
        self,
        layer_idx: int,
        hidden_state: np.ndarray,
        pos: int,
        cache_k: Optional[np.ndarray] = None,
        cache_v: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exécute la passe forward d'une couche du transformer.

        Paramètres
        ----------
        layer_idx : int
            Index de la couche (0..n_layers-1).
        hidden_state : np.ndarray
            Tenseur d'activation entrant, shape [seq_len, dim], dtype float32.
        pos : int
            Position dans la séquence (offset pour RoPE et masque causal).
        cache_k : np.ndarray | None
            Cache des clés jusqu'à ce token, shape [past_len, n_kv_heads, head_dim].
        cache_v : np.ndarray | None
            Cache des valeurs jusqu'à ce token, même shape.

        Retourne
        --------
        output : np.ndarray
            Activation sortante, shape [seq_len, dim], dtype float32.
        new_k : np.ndarray
            Cache clés mis à jour, shape [past_len + seq_len, n_kv_heads, head_dim].
        new_v : np.ndarray
            Cache valeurs mis à jour, même shape.
        """


# ---------------------------------------------------------------------------
# Proxy client HTTP — exécution sur nœud distant
# ---------------------------------------------------------------------------

class RemoteLayerExecutor(BaseLayerExecutor):
    """
    Proxy client : délègue l'exécution d'une couche à un nœud HTTP distant.

    Le nœud distant possède les fragments de la couche et exécute le calcul.
    Seules les activations (hidden states) transitent sur le réseau.

    Paramètres
    ----------
    node_url : str
        URL de base du nœud (ex : "http://192.168.1.10:8000").
    layers : list[int] | None
        Indices de couches que ce nœud peut exécuter.
        None = interroger /status pour auto-découverte.
    auth_token : str | None
        Token Bearer optionnel.
    timeout : float
        Timeout en secondes par requête.
    use_binary : bool
        Si True, envoie les tenseurs en binaire (numpy bytes) au lieu de JSON.
        Plus rapide, à activer quand le serveur le supporte.
    verbose : bool
        Affiche les requêtes effectuées.

    Exemple d'utilisation (une fois implémenté)
    -------------------------------------------
        from distribution.reseau import RemoteLayerExecutor

        node = RemoteLayerExecutor("http://192.168.1.10:8000", layers=[0, 1, 2])
        x = np.zeros((1, 5120), dtype=np.float32)
        output, new_k, new_v = node.execute_layer(0, x, pos=0)
    """

    def __init__(
        self,
        node_url: str,
        layers: Optional[List[int]] = None,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
        use_binary: bool = False,
        use_compression: bool = False,
        verbose: bool = False,
    ):
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "Le module 'requests' est requis pour RemoteLayerExecutor.\n"
                "  .venv\\Scripts\\python.exe -m pip install requests"
            )
        self.node_url = node_url.rstrip("/")
        self.layers = layers
        self.auth_token = auth_token
        self.timeout = timeout
        self.use_binary = use_binary
        self.use_compression = use_compression and _ZSTD_AVAILABLE
        self.verbose = verbose
        
        if self.verbose:
            print(f"[RemoteLayerExecutor] Initialized with use_binary={use_binary}, use_compression={use_compression}")
        
        if self.use_compression and not _ZSTD_AVAILABLE:
            print("[WARN] zstandard non disponible, la compression est désactivée")

    # ------------------------------------------------------------------
    # À implémenter
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        """
        Construit les en-têtes HTTP pour les requêtes.

        Retourne : dict avec Authorization et Content-Type/Accept appropriés.
        """
        headers = {"Content-Type": "application/json"}
        
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        if self.use_binary:
            # Indicate we can accept binary responses
            headers["Accept"] = "application/json, application/octet-stream"
        
        return headers

    def _serialize_array(self, arr: Optional[np.ndarray]):
        """
        Sérialise un tableau numpy pour transmission HTTP.
        
        Mode JSON    → arr.tolist()  (simple, lent)
        Mode binaire → arr.tobytes() + encoding hex (compatible JSON)
        Mode compressé → zstd compression + base64 encoding
        
        Retourne : dict avec {"__binary__": ...} ou {"__binary_zstd__": ...} ou list (JSON)
        """
        if arr is None:
            return None
            
        if not self.use_binary:
            # Mode JSON (fallback)
            return arr.tolist()
        
        # Mode binaire
        if self.use_compression and _ZSTD_AVAILABLE:
            try:
                # Compress with zstd
                compressed = zstd.ZstdCompressor().compress(arr.tobytes())
                # Encode as base64 for JSON compatibility
                encoded = base64.b64encode(compressed).decode('ascii')
                
                if self.verbose:
                    original_size = arr.nbytes
                    compressed_size = len(compressed)
                    ratio = compressed_size / original_size * 100
                    print(f"[COMPRESS] {original_size} bytes -> {compressed_size} bytes ({ratio:.1f}%)")
                
                return {
                    "__binary_zstd__": True,
                    "data": encoded,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype)
                }
            except Exception as e:
                print(f"[WARN] Compression failed, falling back to binary: {e}")
                # Fall back to binary mode
        
        # Binary mode (no compression)
        # Use hex encoding for binary data to ensure JSON compatibility
        hex_data = arr.tobytes().hex()
        
        if self.verbose:
            print(f"[BINARY] Serialized {arr.nbytes} bytes as hex string")
        
        return {
            "__binary__": True,
            "data": hex_data,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype)
        }

    def _deserialize_array(self, data, shape: tuple, dtype=np.float32) -> np.ndarray:
        """
        Désérialise un tableau numpy depuis la réponse HTTP.
        
        Gère les formats JSON (list), binaire (hex), et compressé (zstd + base64).
        """
        if data is None:
            return None
            
        if isinstance(data, dict):
            if data.get("__binary_zstd__"):
                # Compressed format: base64 encoded zstd compressed data
                try:
                    import base64
                    compressed_data = base64.b64decode(data["data"])
                    decompressed = zstd.ZstdDecompressor().decompress(compressed_data)
                    return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
                except ImportError:
                    raise RuntimeError("zstandard requis pour décompresser les données")
                except Exception as e:
                    raise RuntimeError(f"Échec de la décompression: {e}")
            
            elif data.get("__binary__"):
                # Binary format: hex encoded bytes
                import binascii
                bytes_data = binascii.unhexlify(data["data"])
                return np.frombuffer(bytes_data, dtype=dtype).reshape(shape)
        
        elif isinstance(data, list):
            # JSON format (fallback)
            return np.array(data, dtype=dtype)
        
        raise ValueError(f"Format de données non reconnu: {type(data)}")

    def execute_layer(
        self,
        layer_idx: int,
        hidden_state: np.ndarray,
        pos: int,
        cache_k: Optional[np.ndarray] = None,
        cache_v: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO : envoie une requête POST /execute_layer au nœud distant.

        Étapes :
          1. Construire le body JSON (ou binaire) avec hidden_state, pos, cache_k, cache_v
          2. POST {node_url}/execute_layer avec timeout + retry
          3. Désérialiser la réponse : output, new_k, new_v
          4. Retourner (output, new_k, new_v)

        Gestion des erreurs :
          - HTTPError → lever RemoteExecutionError
          - Timeout   → retry avec backoff exponentiel (max 3 essais)
        """

    def get_status(self) -> dict:
        """
        TODO : GET {node_url}/status → dict avec node_id, layers, model, ready.
        """


# ---------------------------------------------------------------------------
# Routeur multi-nœuds — distribue les couches entre plusieurs nœuds
# ---------------------------------------------------------------------------

class P2PLayerRouter(BaseLayerExecutor):
    """
    Route chaque couche vers le nœud distant qui en est responsable.

    Construit une table layer_idx → RemoteLayerExecutor à partir d'une
    liste de nœuds découverts dynamiquement (via /status) ou configurés
    statiquement.

    Exemple d'utilisation (une fois implémenté)
    -------------------------------------------
        from distribution.reseau import P2PLayerRouter

        router = P2PLayerRouter([
            "http://node0.local:8000",   # couches 0-9
            "http://node1.local:8000",   # couches 10-19
            "http://node2.local:8000",   # couches 20-29
            "http://node3.local:8000",   # couches 30-39
        ])
        output, new_k, new_v = router.execute_layer(5, x, pos=0)
    """

    def __init__(self, node_urls: List[str], verbose: bool = False):
        self.node_urls = node_urls
        self.verbose = verbose

        # Table de routage : layer_idx → RemoteLayerExecutor
        self._routing_table: Dict[int, RemoteLayerExecutor] = {}

    def _build_routing_table(self) -> None:
        """
        TODO : interroge /status sur chaque nœud pour construire la table
        layer_idx → executor.

        Lève une erreur si deux nœuds revendiquent la même couche,
        ou si des couches sont manquantes.
        """

    def execute_layer(
        self,
        layer_idx: int,
        hidden_state: np.ndarray,
        pos: int,
        cache_k: Optional[np.ndarray] = None,
        cache_v: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TODO : résout le nœud responsable de layer_idx et délègue.
        """


# ---------------------------------------------------------------------------
# Exceptions spécifiques
# ---------------------------------------------------------------------------

class RemoteExecutionError(RuntimeError):
    """Levée quand un nœud distant retourne une erreur ou est injoignable."""
    pass


class LayerNotFoundError(KeyError):
    """Levée quand aucun nœud ne peut exécuter la couche demandée."""
    pass
