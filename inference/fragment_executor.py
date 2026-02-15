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

Usage
-----
    from distribution.local import LocalFragmentLoader
    from inference.fragment_executor import FragmentExecutor

    loader = LocalFragmentLoader("models/Magistral-Small-2509-Q4_K_M_fragments")
    with FragmentExecutor(loader, layer_idx=0, config=cfg) as executor:
        x, new_k, new_v = executor.forward(x, freqs_cis=None,
                                            cache_k=None, cache_v=None,
                                            start_pos=0)
    # Ici toute la mémoire de la couche 0 a été libérée.
"""

import gc
import numpy as np
from typing import Optional, Tuple

from distribution.local import BaseFragmentLoader
from .p2p_inference import (
    ModelConfig, rms_norm, softmax, swiglu,
    apply_rotary_emb, precompute_freqs_cis,
)

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
    __exit__ les met à None et appelle gc.collect() pour garantir la libération.

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

        # Suivi mémoire
        self._mem_before:     float = 0.0
        self._mem_after_load: float = 0.0

        # Tenseurs (None hors du bloc with)
        self._w_attn_norm: Optional[np.ndarray] = None
        self._wq:          Optional[np.ndarray] = None
        self._wk:          Optional[np.ndarray] = None
        self._wv:          Optional[np.ndarray] = None
        self._wo:          Optional[np.ndarray] = None
        self._w_ffn_norm:  Optional[np.ndarray] = None
        self._w_gate:      Optional[np.ndarray] = None
        self._w_up:        Optional[np.ndarray] = None
        self._w_down:      Optional[np.ndarray] = None

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
        self._w_attn_norm = None
        self._wq          = None
        self._wk          = None
        self._wv          = None
        self._wo          = None
        self._w_ffn_norm  = None
        self._w_gate      = None
        self._w_up        = None
        self._w_down      = None
        gc.collect()
        if self.track_memory:
            mem_after = _rss_mb()
            delta = self._mem_after_load - mem_after
            print(f"[MemTrack] Couche {self.idx} liberee  : -{delta:.1f} MB "
                  f"(total {mem_after:.1f} MB)")
        return False   # ne pas avaler les exceptions

    # ------------------------------------------------------------------

    def _load_all_weights(self):
        """Charge les 9 tenseurs de la couche depuis le loader."""
        p = self.pfx
        self._w_attn_norm = self.loader.load_tensor(f"{p}.attn_norm.weight")
        self._wq          = self.loader.load_tensor(f"{p}.attn_q.weight")
        self._wk          = self.loader.load_tensor(f"{p}.attn_k.weight")
        self._wv          = self.loader.load_tensor(f"{p}.attn_v.weight")
        self._wo          = self.loader.load_tensor(f"{p}.attn_output.weight")
        self._w_ffn_norm  = self.loader.load_tensor(f"{p}.ffn_norm.weight")
        self._w_gate      = self.loader.load_tensor(f"{p}.ffn_gate.weight")
        self._w_up        = self.loader.load_tensor(f"{p}.ffn_up.weight")
        self._w_down      = self.loader.load_tensor(f"{p}.ffn_down.weight")

    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        freqs_cis: Optional[np.ndarray],   # accepté pour compat API, RoPE calculé en interne
        cache_k: Optional[np.ndarray],
        cache_v: Optional[np.ndarray],
        start_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Passe forward d'une couche Llama (attention + FFN SwiGLU).

        Paramètres
        ----------
        x         : float32[seq_len, dim]
        freqs_cis : ignoré (RoPE calculé internement) — conservé pour compat API
        cache_k   : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        cache_v   : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        start_pos : position du premier token de x dans la séquence complète

        Retourne
        --------
        (x_out, new_k, new_v)
          x_out : float32[seq_len, dim]
          new_k : float32[seq_len, n_kv_heads, head_dim]   (avant répétition GQA)
          new_v : float32[seq_len, n_kv_heads, head_dim]
        """
        assert self._wq is not None, \
            "FragmentExecutor doit être utilisé comme context manager (with ... as ex:)"

        seq_len  = x.shape[0]
        head_dim = self.cfg.dim // self.cfg.n_heads

        # ── Projection helper ────────────────────────────────────────────
        def proj(inp: np.ndarray, w: np.ndarray) -> np.ndarray:
            """inp @ w (standard [in, out]) avec fallback [out, in]."""
            if w.ndim == 2:
                if w.shape[0] == inp.shape[-1]:
                    return inp @ w              # [in, out] — cas normal après transposition GGUF
                if w.shape[1] == inp.shape[-1]:
                    return inp @ w.T            # [out, in] — fallback
            return inp @ w

        # ── Self-Attention ───────────────────────────────────────────────
        xn = rms_norm_numba(x, self._w_attn_norm, self.cfg.norm_eps)

        xq = proj(xn, self._wq)
        xk = proj(xn, self._wk)
        xv = proj(xn, self._wv)

        q_head_dim = xq.shape[-1] // self.cfg.n_heads
        k_head_dim = xk.shape[-1] // self.cfg.n_kv_heads
        v_head_dim = xv.shape[-1] // self.cfg.n_kv_heads

        xq = xq.reshape(seq_len, self.cfg.n_heads,    q_head_dim)
        xk = xk.reshape(seq_len, self.cfg.n_kv_heads, k_head_dim)
        xv = xv.reshape(seq_len, self.cfg.n_kv_heads, v_head_dim)

        # RoPE (calculé à chaque appel — pas de cache ici, optimisable si nécessaire)
        all_freqs = precompute_freqs_cis(
            q_head_dim,
            start_pos + seq_len + 1,
            theta=self.cfg.rope_freq_base,
        )
        cur_freqs = all_freqs[start_pos:start_pos + seq_len].reshape(seq_len, 1, -1)
        xq, xk    = apply_rotary_emb(xq, xk, cur_freqs)

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
        xq_t    = xq.transpose(1, 0, 2)                             # [n_heads, seq_len, q_hd]
        keys_t  = keys.transpose(1, 0, 2)                           # [n_heads, total_seq, k_hd]
        vals_t  = values.transpose(1, 0, 2)                         # [n_heads, total_seq, v_hd]
        scores  = np.matmul(xq_t, keys_t.transpose(0, 2, 1)) / np.sqrt(head_dim)

        # Masque causal (uniquement pendant le prefill, seq_len > 1)
        if seq_len > 1:
            mask   = np.triu(np.full((seq_len, total_seq), float("-inf")), k=total_seq - seq_len + 1)
            scores = scores + mask

        probs  = softmax_numba(scores)                               # [n_heads, seq_len, total_seq]
        output = np.matmul(probs, vals_t)                            # [n_heads, seq_len, v_hd]
        output = output.transpose(1, 0, 2).reshape(seq_len, -1)      # [seq_len, n_heads*v_hd]

        h = x + proj(output, self._wo)

        # ── FFN SwiGLU ───────────────────────────────────────────────────
        xn     = rms_norm_numba(h, self._w_ffn_norm, self.cfg.norm_eps)
        gate   = proj(xn, self._w_gate)
        up     = proj(xn, self._w_up)
        hidden = swiglu_numba(gate) * up
        out    = proj(hidden, self._w_down)

        return h + out, new_k, new_v
