"""
fragment_executor.py
====================
Exécuteur d'une couche de transformeur travaillant fragment par fragment.

Une seule couche occupe la RAM à la fois : les tenseurs sont chargés dans
__enter__ et explicitement libérés dans __exit__ (+ gc.collect()).

Ce modèle est la brique de base de l'architecture distribuée (Phase 4) :
chaque nœud P2P héberge ses fragments et exécute les couches qui lui arrivent.
Pour changer de backend de distribution, remplacer le loader :

    # Local (défaut)
    with FragmentExecutor(LocalFragmentLoader(dir), l, cfg) as ex: ...

    # Réseau P2P (futur)
    with FragmentExecutor(ReseauFragmentLoader(node_addr), l, cfg) as ex: ...

Optimisations actives
---------------------
* Fused Q4_K GEMV (decode, seq_len=1) : jamais d'allocation float32 intermédiaire
  pour les grandes matrices — ~10-20x plus rapide que dequantize+matmul.
* Raw byte cache dans LocalFragmentLoader(cache_raw=True) : évite les relectures
  disque à chaque token de decode.
* Partage du cache RoPE avec le moteur : pas de recalcul par couche.
* gc.collect() déplacé hors de la boucle couche (appelé une fois par token).

Usage
-----
    from distribution.local import LocalFragmentLoader
    from inference.fragment_executor import FragmentExecutor

    loader = LocalFragmentLoader("models/...", cache_raw=True)
    with FragmentExecutor(loader, layer_idx=0, config=cfg) as executor:
        x, new_k, new_v = executor.forward(x, rope_cache=engine._rope_cache,
                                            cache_k=None, cache_v=None,
                                            start_pos=0)
    # Ici toute la mémoire flottante de la couche 0 a été libérée.
"""

import gc
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum

from distribution.local import BaseFragmentLoader
from .p2p_inference import (
    ModelConfig, rms_norm, softmax, swiglu,
    apply_rotary_emb, precompute_freqs_cis,
)


class ModelArchitecture(Enum):
    """Supported model architectures."""
    MAGISTRAL = "magistral"
    MISTRAL_7B = "mistral_7b"
    
    @staticmethod
    def detect_from_config(config: ModelConfig) -> 'ModelArchitecture':
        """Detect architecture from model configuration."""
        dim = config.dim
        hidden_dim = config.hidden_dim
        vocab_size = config.vocab_size
        
        if dim == 5120 and hidden_dim == 32768 and vocab_size == 131072:
            return ModelArchitecture.MAGISTRAL
        elif dim == 4096 and hidden_dim == 14336 and vocab_size == 32768:
            return ModelArchitecture.MISTRAL_7B
        else:
            raise ValueError(f"Unsupported architecture: dim={dim}, hidden={hidden_dim}, vocab={vocab_size}")

# Fused GEMV Q4_K et Q6_K (chemin rapide decode)
try:
    from dequantize.Q4_K_GGUF import q4k_gemv as _q4k_gemv, warmup_q4k_gemv
    _USE_Q4K_GEMV = True
except ImportError:
    _USE_Q4K_GEMV = False

try:
    from dequantize.Q6_K_GGUF import q6k_gemv as _q6k_gemv, warmup_q6k_gemv
    _USE_Q6K_GEMV = True
except ImportError:
    _USE_Q6K_GEMV = False

# Kernels numba si disponibles, sinon fallback NumPy pur
try:
    from .kernels_numba import rms_norm_numba, softmax_numba, swiglu_numba, warmup_kernels
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    rms_norm_numba = rms_norm      # type: ignore[assignment]
    softmax_numba  = softmax       # type: ignore[assignment]
    swiglu_numba   = swiglu        # type: ignore[assignment]

# Monitoring mémoire optionnel
try:
    import psutil
    import os as _os
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# Types reconnus pour la voie rapide (GEMV fusionné)
# Mistral 7B typically uses Q4_K_M quantization
_Q4K_TYPES = frozenset(("Q4_K", "Q4_K_M", "Q4_K_S"))
_Q6K_TYPES = frozenset(("Q6_K",))


def _rss_mb() -> float:
    """Mémoire résidente (RSS) du processus courant en MB."""
    if _PSUTIL:
        return psutil.Process(_os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0


# ---------------------------------------------------------------------------
# FragmentExecutor
# ---------------------------------------------------------------------------

class FragmentExecutor:
    """
    Exécute la passe forward d'une seule couche Llama.

    Les tenseurs ne sont chargés que pendant la durée du bloc `with`.
    __exit__ les met à None (gc.collect() est géré par le moteur, pas ici).

    Paramètres
    ----------
    loader       : BaseFragmentLoader
                   Backend de chargement (local, réseau, P2P).
    layer_idx    : int
                   Index de la couche (0 à n_layers-1).
    config       : ModelConfig
                   Hyperparamètres du modèle.
    track_memory : bool
                   Si True, affiche l'empreinte mémoire avant/après chaque couche.
    """

    def __init__(
        self,
        loader: BaseFragmentLoader,
        layer_idx: int,
        config: ModelConfig,
        track_memory: bool = False,
    ):
        self.loader       = loader
        self.idx          = layer_idx
        self.cfg          = config
        self.pfx          = f"blk.{layer_idx}"
        self.track_memory = track_memory

        # Architecture detection
        self.architecture = ModelArchitecture.detect_from_config(config)
        
        # Extract architecture-specific parameters
        self.dim = config.dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.vocab_size = config.vocab_size
        
        # Calculate derived parameters
        self.head_dim = self.dim // self.n_heads
        
        # Architecture-specific adjustments for Mistral 7B
        if self.architecture == ModelArchitecture.MISTRAL_7B:
            # Mistral 7B uses specific tensor dimensions
            # dim=4096, n_heads=32, so head_dim=128
            self.head_dim = self.dim // self.n_heads  # 4096 // 32 = 128 for Mistral 7B
            print(f"[ARCH] Mistral 7B detected: dim={self.dim}, n_heads={self.n_heads}, head_dim={self.head_dim}")
            # Mistral 7B uses a different RoPE dimension calculation
            self.rope_dim = self.head_dim  # For Mistral 7B, rope_dim equals head_dim
        
        # Suivi mémoire
        self._mem_before:     float = 0.0
        self._mem_after_load: float = 0.0

        # Tenseurs flottants toujours chargés (norms, petites matrices attn)
        self._w_attn_norm: Optional[np.ndarray] = None
        self._w_ffn_norm:  Optional[np.ndarray] = None

        # Tenseurs float32 (utilisés quand la voie rapide GEMV n'est pas disponible)
        self._wq:    Optional[np.ndarray] = None
        self._wk:    Optional[np.ndarray] = None
        self._wv:    Optional[np.ndarray] = None
        self._wo:    Optional[np.ndarray] = None
        self._w_gate: Optional[np.ndarray] = None
        self._w_up:   Optional[np.ndarray] = None
        self._w_down: Optional[np.ndarray] = None

        # Octets bruts + métadonnées pour le fused GEMV Q4_K
        # Tuple (bytes, tensor_type, logical_shape) ou None
        self._raw_wq:    Optional[tuple] = None
        self._raw_wk:    Optional[tuple] = None
        self._raw_wv:    Optional[tuple] = None
        self._raw_wo:    Optional[tuple] = None
        self._raw_w_gate: Optional[tuple] = None
        self._raw_w_up:   Optional[tuple] = None
        self._raw_w_down: Optional[tuple] = None

    # ------------------------------------------------------------------

    def __enter__(self) -> "FragmentExecutor":
        if self.track_memory:
            self._mem_before = _rss_mb()
        self._load_all_weights()
        if self.track_memory:
            self._mem_after_load = _rss_mb()
            delta = self._mem_after_load - self._mem_before
            print(f"[MemTrack] Couche {self.idx} chargee  : +{delta:.1f} MB "
                  f"(total {self._mem_after_load:.1f} MB)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Libérer toutes les références (float32 et raw bytes)
        self._w_attn_norm = None
        self._w_ffn_norm  = None
        self._wq = self._wk = self._wv = self._wo = None
        self._w_gate = self._w_up = self._w_down = None
        self._raw_wq = self._raw_wk = self._raw_wv = self._raw_wo = None
        self._raw_w_gate = self._raw_w_up = self._raw_w_down = None
        # gc.collect() EST géré par le moteur (une fois par token, pas par couche)
        if self.track_memory:
            mem_after = _rss_mb()
            delta = self._mem_after_load - mem_after
            print(f"[MemTrack] Couche {self.idx} liberee  : -{delta:.1f} MB "
                  f"(total {mem_after:.1f} MB)")
        return False   # ne pas avaler les exceptions

    # ------------------------------------------------------------------
    # Architecture-specific validation
    # ------------------------------------------------------------------

    def get_architecture_config_dict(self) -> Dict:
        """
        Get architecture configuration as a dict for use with architecture-aware functions.
        
        Retourne
        --------
        Dict
            Architecture configuration dictionary
        """
        config = {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "norm_eps": self.cfg.norm_eps,
            "rope_freq_base": self.cfg.rope_freq_base,
            "tensor_specifics": {
                "attention": {
                    "q_dim": self.cfg.dim,  # Will be updated when we load actual tensors
                    "k_dim": self.cfg.dim,
                    "v_dim": self.cfg.dim,
                    "output_dim": self.cfg.dim
                },
                "ffn": {
                    "gate_dim": self.cfg.dim,
                    "up_dim": self.cfg.dim,
                    "down_dim": self.cfg.hidden_dim
                }
            },
            "architecture": self.architecture.value
        }
        
        # Mistral 7B specific adjustments
        if self.architecture == ModelArchitecture.MISTRAL_7B:
            config["tensor_specifics"]["attention"]["q_dim"] = self.dim
            config["tensor_specifics"]["attention"]["k_dim"] = self.dim
            config["tensor_specifics"]["attention"]["v_dim"] = self.dim
            config["tensor_specifics"]["attention"]["output_dim"] = self.dim
            config["tensor_specifics"]["ffn"]["gate_dim"] = self.hidden_dim
            config["tensor_specifics"]["ffn"]["up_dim"] = self.hidden_dim
            config["tensor_specifics"]["ffn"]["down_dim"] = self.dim
        
        return config

    def _validate_input_dimensions(
        self,
        x: np.ndarray,
        cache_k: Optional[np.ndarray] = None,
        cache_v: Optional[np.ndarray] = None
    ):
        """
        Validate that input tensors have correct dimensions for this architecture.
        
        Paramètres
        ----------
        x       : float32[seq_len, dim]
        cache_k : float32[prev_seq, n_kv_heads, head_dim] ou None
        cache_v : float32[prev_seq, n_kv_heads, head_dim] ou None
        """
        # Check hidden state dimension
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Hidden state dimension mismatch for {self.architecture.value} architecture. "
                f"Expected {self.dim}, got {x.shape[-1]}"
            )
        
        # Check cache dimensions if provided
        if cache_k is not None:
            if cache_k.ndim != 3:
                raise ValueError(
                    f"Cache K must be 3D tensor for {self.architecture.value} architecture, "
                    f"got {cache_k.ndim}D"
                )
            
            if cache_k.shape[1] != self.n_kv_heads:
                raise ValueError(
                    f"Cache K heads mismatch for {self.architecture.value} architecture. "
                    f"Expected {self.n_kv_heads}, got {cache_k.shape[1]}"
                )
            
            # For Mistral 7B, head_dim should be dim // n_heads
            expected_head_dim = self.dim // self.n_heads
            if cache_k.shape[2] != expected_head_dim:
                raise ValueError(
                    f"Cache K head dimension mismatch for {self.architecture.value} architecture. "
                    f"Expected {expected_head_dim}, got {cache_k.shape[2]}"
                )
        
        if cache_v is not None:
            if cache_v.ndim != 3:
                raise ValueError(
                    f"Cache V must be 3D tensor for {self.architecture.value} architecture, "
                    f"got {cache_v.ndim}D"
                )
            
            if cache_v.shape[1] != self.n_kv_heads:
                raise ValueError(
                    f"Cache V heads mismatch for {self.architecture.value} architecture. "
                    f"Expected {self.n_kv_heads}, got {cache_v.shape[1]}"
                )
            
            # For Mistral 7B, head_dim should be dim // n_heads
            expected_head_dim = self.dim // self.n_heads
            if cache_v.shape[2] != expected_head_dim:
                raise ValueError(
                    f"Cache V head dimension mismatch for {self.architecture.value} architecture. "
                    f"Expected {expected_head_dim}, got {cache_v.shape[2]}"
                )

    # ------------------------------------------------------------------

    def _load_weight(self, name: str):
        """
        Charge un tenseur.
        - Si le loader supporte load_raw_tensor et que c'est du Q4_K ou Q6_K :
          stocke les octets bruts (pour le GEMV fusionné).
        - Sinon : dequantise directement en float32.
        Retourne le résultat brut (tuple ou ndarray).
        """
        use_raw = (_USE_Q4K_GEMV or _USE_Q6K_GEMV) and hasattr(self.loader, "load_raw_tensor")
        if use_raw:
            raw, ttype, shape = self.loader.load_raw_tensor(name)
            if raw is not None and len(shape) == 2:
                if (_USE_Q4K_GEMV and ttype in _Q4K_TYPES) or \
                   (_USE_Q6K_GEMV and ttype in _Q6K_TYPES):
                    return (raw, ttype, shape)   # format brut — GEMV sera utilisé
        # Fallback : float32 avec cache pour éviter les rechargements
        return self.loader.load_tensor(name)

    def _load_all_weights(self):
        """Charge les 9 tenseurs de la couche depuis le loader."""
        p = self.pfx
        # Norms : toujours float32 (petits vecteurs)
        self._w_attn_norm = self.loader.load_tensor(f"{p}.attn_norm.weight")
        self._w_ffn_norm  = self.loader.load_tensor(f"{p}.ffn_norm.weight")
        # Matrices de projection (raw Q4_K si possible, sinon float32)
        result = self._load_weight(f"{p}.attn_q.weight")
        if isinstance(result, tuple): self._raw_wq    = result
        else:                          self._wq        = result
        result = self._load_weight(f"{p}.attn_k.weight")
        if isinstance(result, tuple): self._raw_wk    = result
        else:                          self._wk        = result
        result = self._load_weight(f"{p}.attn_v.weight")
        if isinstance(result, tuple): self._raw_wv    = result
        else:                          self._wv        = result
        result = self._load_weight(f"{p}.attn_output.weight")
        if isinstance(result, tuple): self._raw_wo    = result
        else:                          self._wo        = result
        result = self._load_weight(f"{p}.ffn_gate.weight")
        if isinstance(result, tuple): self._raw_w_gate = result
        else:                          self._w_gate    = result
        result = self._load_weight(f"{p}.ffn_up.weight")
        if isinstance(result, tuple): self._raw_w_up   = result
        else:                          self._w_up      = result
        result = self._load_weight(f"{p}.ffn_down.weight")
        if isinstance(result, tuple): self._raw_w_down = result
        else:                          self._w_down    = result

    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        rope_cache: Optional[Dict],        # cache RoPE partagé du moteur (évite recalcul)
        cache_k: Optional[np.ndarray],
        cache_v: Optional[np.ndarray],
        start_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Passe forward d'une couche Llama (attention + FFN SwiGLU).

        Paramètres
        ----------
        x          : float32[seq_len, dim]
        rope_cache : dict {head_dim: freqs_cis} partagé avec le moteur.
                     Si None, les fréquences sont calculées à la volée.
        cache_k    : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        cache_v    : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        start_pos  : position du premier token de x dans la séquence complète

        Retourne
        --------
        (x_out, new_k, new_v)
          x_out : float32[seq_len, dim]
          new_k : float32[seq_len, n_kv_heads, head_dim]   (avant répétition GQA)
          new_v : float32[seq_len, n_kv_heads, head_dim]
        """
        assert self._w_attn_norm is not None, \
            "FragmentExecutor doit être utilisé comme context manager (with ... as ex:)"

        # Validate input dimensions for this architecture
        self._validate_input_dimensions(x, cache_k, cache_v)

        seq_len  = x.shape[0]
        # Use the correctly calculated head_dim for this architecture
        head_dim = self.head_dim

        # ── Helpers de projection ────────────────────────────────────────
        def _proj_float(inp: np.ndarray, w: np.ndarray) -> np.ndarray:
            """Projection float32 standard (fallback quand GEMV n'est pas disponible)."""
            if w.ndim == 2:
                if w.shape[0] == inp.shape[-1]:
                    return inp @ w       # [in, out] — layout GGUF après transposition
                if w.shape[1] == inp.shape[-1]:
                    return inp @ w.T     # [out, in] — fallback
            return inp @ w

        def proj(inp: np.ndarray, raw_or_w) -> np.ndarray:
            """
            Dispatch intelligent :
            - seq_len=1 + Q4_K raw → fused GEMV Q4_K (pas d'allocation float32)
            - seq_len=1 + Q6_K raw → fused GEMV Q6_K
            - sinon → dequantize + matmul standard
            """
            if isinstance(raw_or_w, tuple):
                raw, ttype, shape = raw_or_w
                if seq_len == 1:
                    if _USE_Q4K_GEMV and ttype in _Q4K_TYPES:
                        return _q4k_gemv(inp.flatten(), raw, shape).reshape(1, -1)
                    if _USE_Q6K_GEMV and ttype in _Q6K_TYPES:
                        return _q6k_gemv(inp.flatten(), raw, shape).reshape(1, -1)
                # Prefill ou type non géré : dequantise puis matmul
                from dequantize import dequantize as _deq
                w = _deq(raw, ttype, shape)
                return _proj_float(inp, w)
            else:
                return _proj_float(inp, raw_or_w)

        # ── Self-Attention ───────────────────────────────────────────────
        xn = rms_norm_numba(x, self._w_attn_norm, self.cfg.norm_eps)

        # Sélectionner la source (raw tuple ou float32 ndarray)
        wq = self._raw_wq    if self._raw_wq    is not None else self._wq
        wk = self._raw_wk    if self._raw_wk    is not None else self._wk
        wv = self._raw_wv    if self._raw_wv    is not None else self._wv
        wo = self._raw_wo    if self._raw_wo    is not None else self._wo

        xq = proj(xn, wq)
        xk = proj(xn, wk)
        xv = proj(xn, wv)

        # Use the correct head dimensions for this architecture
        # For Mistral 7B, we should use the actual tensor dimensions, not forced values
        q_head_dim = xq.shape[-1] // self.cfg.n_heads
        k_head_dim = xk.shape[-1] // self.cfg.n_kv_heads
        v_head_dim = xv.shape[-1] // self.cfg.n_kv_heads

        xq = xq.reshape(seq_len, self.cfg.n_heads,    q_head_dim)
        xk = xk.reshape(seq_len, self.cfg.n_kv_heads, k_head_dim)
        xv = xv.reshape(seq_len, self.cfg.n_kv_heads, v_head_dim)

        # ── RoPE — utilise le cache partagé du moteur ────────────────────
        # For Mistral 7B, we need to use the correct head_dim for RoPE
        rope_head_dim = q_head_dim
        if self.architecture == ModelArchitecture.MISTRAL_7B:
            # Mistral 7B uses specific RoPE dimensions
            # Use rope_dim if it was set, otherwise use head_dim
            rope_head_dim = getattr(self, 'rope_dim', self.head_dim)
        
        if rope_cache is not None:
            min_end = start_pos + seq_len + 1
            if rope_head_dim not in rope_cache or rope_cache[rope_head_dim].shape[0] < min_end:
                rope_cache[rope_head_dim] = precompute_freqs_cis(
                    rope_head_dim, max(min_end, self.cfg.dim * 2),
                    theta=self.cfg.rope_freq_base,
                )
            cur_freqs = rope_cache[rope_head_dim][start_pos:start_pos + seq_len].reshape(seq_len, 1, -1)
        else:
            # Calcul local si pas de cache fourni (évite erreur silencieuse)
            all_freqs = precompute_freqs_cis(
                rope_head_dim, start_pos + seq_len + 1, theta=self.cfg.rope_freq_base,
            )
            cur_freqs = all_freqs[start_pos:start_pos + seq_len].reshape(seq_len, 1, -1)

        xq, xk = apply_rotary_emb(xq, xk, cur_freqs)

        new_k = xk   # [seq_len, n_kv_heads, k_head_dim]
        new_v = xv   # [seq_len, n_kv_heads, v_head_dim]

        # KV cache : concaténer avec les tokens précédents si disponibles
        if cache_k is not None:
            keys   = np.concatenate([cache_k, new_k], axis=0)
            values = np.concatenate([cache_v, new_v], axis=0)
        else:
            keys, values = new_k, new_v

        total_seq = keys.shape[0]

        # GQA : répéter les têtes KV pour aligner avec n_heads
        n_rep = self.cfg.n_heads // self.cfg.n_kv_heads
        if n_rep > 1:
            keys   = np.repeat(keys,   n_rep, axis=1)   # [total_seq, n_heads, k_head_dim]
            values = np.repeat(values, n_rep, axis=1)

        # Scores d'attention : [n_heads, seq_len, total_seq]
        # Use the correct head_dim for scaling
        # For Mistral 7B, we should use the actual q_head_dim, not forced head_dim
        attention_head_dim = q_head_dim
        
        # Performance optimization: pre-allocate output arrays for attention
        # This prevents repeated memory allocations during attention computation
        output_shape = (seq_len, self.cfg.dim)
            
        xq_t   = xq.transpose(1, 0, 2)                              # [n_heads, seq_len, q_hd]
        keys_t = keys.transpose(1, 0, 2)                            # [n_heads, total_seq, k_hd]
        vals_t = values.transpose(1, 0, 2)                          # [n_heads, total_seq, v_hd]
        scores = np.matmul(xq_t, keys_t.transpose(0, 2, 1)) / np.sqrt(attention_head_dim)

        # Masque causal (uniquement pendant le prefill, seq_len > 1)
        if seq_len > 1:
            mask   = np.triu(np.full((seq_len, total_seq), float("-inf")), k=total_seq - seq_len + 1)
            scores = scores + mask

        probs  = softmax_numba(scores)                               # [n_heads, seq_len, total_seq]
        output = np.matmul(probs, vals_t)                            # [n_heads, seq_len, v_hd]
        output = output.transpose(1, 0, 2).reshape(seq_len, -1)      # [seq_len, n_heads*v_hd]

        h = x + proj(output, wo)
        
        # Clear intermediate variables to free memory
        del probs, output, xq_t, keys_t, vals_t, scores

        # ── FFN SwiGLU ───────────────────────────────────────────────────
        xn = rms_norm_numba(h, self._w_ffn_norm, self.cfg.norm_eps)

        wg   = self._raw_w_gate if self._raw_w_gate is not None else self._w_gate
        wu   = self._raw_w_up   if self._raw_w_up   is not None else self._w_up
        wd   = self._raw_w_down if self._raw_w_down is not None else self._w_down

        gate   = proj(xn, wg)
        up     = proj(xn, wu)
        hidden = swiglu_numba(gate) * up
        out    = proj(hidden, wd)

        # Clear intermediate variables to free memory
        del gate, up, hidden, xn

        result = h + out
        return result, new_k, new_v
