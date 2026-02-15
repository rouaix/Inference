# SPEC — Fragment Executor (Solution 2)

> Rédigé le 2026-02-14
> Basé sur `PROPOSAL_fragment_executor.md` — Solution 2, Option C (Python + numba)
> Implémentation recommandée : **moyen terme**, sans changement d'architecture des fragments

---

## Objectif

Transformer le moteur actuel (`p2p_inference.py`) en un système où **une seule couche de transformeur occupe la RAM à la fois**.
Chaque couche est chargée, calculée, puis explicitement libérée avant de passer à la suivante.

Ce modèle est la brique de base de la future architecture distribuée (Phase 4 du roadmap) : chaque nœud P2P héberge quelques couches et exécute les requêtes à la demande.

**État actuel** :
- `LlamaLayer.forward()` charge les tenseurs d'une couche à la volée
- Mais Python ne libère pas la mémoire immédiatement (GC non déterministe)
- Aucune mesure de mémoire, aucune garantie de libération entre couches
- Vitesse : ~14s/token (NumPy pur, pas de BLAS parallèle)

**État cible après ce spec** :
- `FragmentExecutor` : classe dédiée par couche, avec context manager `with` garantissant la libération
- Kernels critiques récrits avec `numba.njit(parallel=True)`
- Gain estimé : ×10 à ×20 sur les matmuls
- RAM pic : ≈ 10–15 MB (une couche TinyLlama) + activation courante

---

## Dépendances

```
numba>=0.59.0   # JIT compiler
numpy>=1.24     # déjà présent
psutil>=5.9     # monitoring mémoire (optionnel)
```

Ajouter dans `requirements.txt`.

---

## Vue d'ensemble des étapes

| Étape | Fichier(s) créé/modifié | Dépend de | Durée estimée |
|-------|------------------------|-----------|---------------|
| **1** — Kernels numba | `kernels_numba.py` (nouveau) | — | 3–4h |
| **2** — FragmentExecutor | `fragment_executor.py` (nouveau) | Étape 1 | 2–3h |
| **3** — Refactor generate() | `p2p_inference.py` (modif) | Étape 2 | 1–2h |
| **4** — Memory tracker | `fragment_executor.py` (modif) | Étape 2 | 1h |
| **5** — Tests & validation | `tests_debug/test_fragment_executor.py` (nouveau) | Étapes 1–4 | 2–3h |

Chaque étape est **indépendante** et testable séparément.
L'ordre recommandé est 1 → 2 → 3 → 4 → 5, mais les étapes 4 et 5 peuvent être faites en parallèle.

---

## Étape 1 — Kernels numba (`kernels_numba.py`)

### But

Réécrire les 5 fonctions mathématiques critiques avec `numba.njit(parallel=True)` pour remplacer les versions NumPy pur de `p2p_inference.py`.

### Fichier à créer : `kernels_numba.py`

```
P:\Projets\Inference\kernels_numba.py
```

### Contenu détaillé

#### 1.1 — `dequantize_q8_0`

Actuellement dans `p2p_inference.py:load_tensor()` (lignes 541–556) et dans `distribution/local.py:_dequantize_q8_0_legacy()` (lignes 198–220).

**Signature cible** :
```python
@numba.njit(parallel=True)
def dequantize_q8_0(raw: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Entrée  : raw uint8 de taille n_blocks * 34 octets
              Format de chaque bloc : [scale: float16 (2 octets)] [qs: int8 x 32 (32 octets)]
    Sortie  : float32 de taille n_blocks * 32 (valeurs dequantisées, mise à plat)

    NB: le layout physique Q8_0 est [out_dim, in_dim] dans GGUF.
    La transposition [out_dim, in_dim] → [in_dim, out_dim] est faite
    dans FragmentExecutor.load_tensor(), PAS ici.
    """
```

**Implémentation** :
- Parser le buffer `raw` octet par octet en évitant `np.frombuffer` (non supporté par numba)
- Extraire le scale `d` (float16 encodé en 2 octets little-endian) avec `np.frombuffer` **avant** l'entrée dans `@njit`, le passer en argument
- Dans le kernel : boucle `numba.prange(n_blocks)` sur les blocs
- Variante 2 (plus simple) : pré-parser les scales et les qs avec NumPy AVANT le kernel, puis passer les deux arrays au kernel JIT

**Variante recommandée** (évite les limitations numba sur les dtypes struct) :

```python
def dequantize_q8_0(data: bytes, shape: tuple) -> np.ndarray:
    """Wrapper Python : parse le header, appelle le kernel JIT."""
    dt = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))])
    blocks = np.frombuffer(data, dtype=dt)
    scales = blocks['d'].astype(np.float32)   # [n_blocks]
    qs = blocks['qs'].astype(np.float32)       # [n_blocks, 32]
    decoded = _q8_0_kernel(scales, qs)          # appel JIT
    # Layout transposition (cf. CLAUDE.md § Q8_0 Transposed Physical Layout)
    if len(shape) == 2:
        out_dim, in_dim = shape[-1], shape[0]
        return decoded.reshape(out_dim, in_dim).T.astype(np.float32)
    return decoded.reshape(shape).astype(np.float32)

@numba.njit(parallel=True)
def _q8_0_kernel(scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """
    scales : float32[n_blocks]
    qs     : float32[n_blocks, 32]
    retour : float32[n_blocks * 32]
    """
    n_blocks = scales.shape[0]
    out = np.empty(n_blocks * 32, dtype=np.float32)
    for i in numba.prange(n_blocks):
        s = scales[i]
        for j in range(32):
            out[i * 32 + j] = s * qs[i, j]
    return out
```

#### 1.2 — `rms_norm_numba`

Actuellement dans `p2p_inference.py:rms_norm()` (lignes 357–361).

```python
@numba.njit(parallel=True)
def rms_norm_numba(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    x      : float32[seq_len, dim]
    weight : float32[dim]
    retour : float32[seq_len, dim]

    Normalisation PAR TOKEN (une variance par ligne).
    Formule : x * (1 / sqrt(mean(x²) + eps)) * weight
    """
    seq_len, dim = x.shape
    out = np.empty_like(x)
    for i in numba.prange(seq_len):
        ss = 0.0
        for j in range(dim):
            ss += x[i, j] * x[i, j]
        ss = 1.0 / np.sqrt(ss / dim + eps)
        for j in range(dim):
            out[i, j] = x[i, j] * ss * weight[j]
    return out
```

#### 1.3 — `softmax_numba`

Actuellement dans `p2p_inference.py:softmax()` (lignes 363–367).

```python
@numba.njit(parallel=True)
def softmax_numba(x: np.ndarray) -> np.ndarray:
    """
    x      : float32[n_heads, seq_len, seq_len]
    retour : float32[n_heads, seq_len, seq_len]

    Softmax sur la dernière dimension, avec soustraction du max pour stabilité numérique.
    """
    n_heads, seq, _ = x.shape
    out = np.empty_like(x)
    for h in numba.prange(n_heads):
        for i in range(seq):
            max_val = x[h, i, 0]
            for j in range(1, seq):
                if x[h, i, j] > max_val:
                    max_val = x[h, i, j]
            s = 0.0
            for j in range(seq):
                out[h, i, j] = np.exp(x[h, i, j] - max_val)
                s += out[h, i, j]
            for j in range(seq):
                out[h, i, j] /= s
    return out
```

#### 1.4 — `swiglu_numba`

Actuellement dans `p2p_inference.py:swiglu()` (ligne 370–372).

```python
@numba.njit(parallel=True)
def swiglu_numba(x: np.ndarray) -> np.ndarray:
    """
    SiLU : x / (1 + exp(-x))
    x      : float32[seq_len, hidden_dim]
    retour : float32[seq_len, hidden_dim]
    """
    out = np.empty_like(x)
    n, m = x.shape
    for i in numba.prange(n):
        for j in range(m):
            out[i, j] = x[i, j] / (1.0 + np.exp(-x[i, j]))
    return out
```

#### 1.5 — `matmul_f32` (wrapper BLAS)

Pour les matmuls, numba utilise OpenBLAS/MKL en interne quand `np.dot` est appelé dans un contexte `@njit`. Il suffit de s'assurer que les arrays sont **contiguous** (C-order, float32) avant d'appeler `np.dot`.

```python
def ensure_contiguous(a: np.ndarray) -> np.ndarray:
    """Force un array en C-contiguous float32 pour BLAS optimal."""
    if not a.flags['C_CONTIGUOUS'] or a.dtype != np.float32:
        return np.ascontiguousarray(a, dtype=np.float32)
    return a
```

> Note : `np.matmul` avec numba utilise automatiquement OpenBLAS si disponible dans le venv.
> Il n'est PAS nécessaire de réécrire le matmul manuellement — numba dispatche vers BLAS.

#### 1.6 — Warm-up JIT au démarrage

Numba compile les kernels au **premier appel**. Ajouter une fonction de warm-up pour que la compilation se fasse au chargement du module, pas pendant l'inférence :

```python
def warmup_kernels():
    """
    Appelle chaque kernel JIT avec des données factices pour déclencher
    la compilation AOT. À appeler une seule fois au démarrage.
    """
    _dummy_scales = np.ones(1, dtype=np.float32)
    _dummy_qs = np.ones((1, 32), dtype=np.float32)
    _q8_0_kernel(_dummy_scales, _dummy_qs)

    _x = np.ones((1, 64), dtype=np.float32)
    _w = np.ones(64, dtype=np.float32)
    rms_norm_numba(_x, _w, 1e-5)

    _s = np.ones((1, 1, 1), dtype=np.float32)
    softmax_numba(_s)

    _g = np.ones((1, 64), dtype=np.float32)
    swiglu_numba(_g)

    print("[kernels_numba] Warm-up JIT terminé.")
```

### Tests unitaires (inline dans le fichier)

Ajouter un bloc `if __name__ == "__main__"` qui vérifie chaque kernel contre la référence NumPy :

```python
if __name__ == "__main__":
    # Test dequantize_q8_0 vs legacy
    from distribution.local import LocalFragmentLoader
    loader = LocalFragmentLoader("models/tinyllama_q8_fragments_v2")
    tensor_name = "blk.0.attn_q.weight"
    ref = loader.load_tensor(tensor_name)
    # Charger les données brutes
    frags = loader.fragments_map[tensor_name]
    raw = b"".join(loader.load_raw(f["fragment_id"]) for f in frags)
    result = dequantize_q8_0(raw, frags[0]["shape"])
    assert np.allclose(ref, result, atol=1e-5), f"dequantize mismatch: max_diff={np.max(np.abs(ref - result))}"
    print("✓ dequantize_q8_0")

    # Test rms_norm
    x = np.random.randn(4, 64).astype(np.float32)
    w = np.random.randn(64).astype(np.float32)
    from p2p_inference import rms_norm
    ref_norm = rms_norm(x, w, 1e-5)
    res_norm = rms_norm_numba(x, w, 1e-5)
    assert np.allclose(ref_norm, res_norm, atol=1e-5), "rms_norm mismatch"
    print("✓ rms_norm_numba")

    # Test softmax
    s = np.random.randn(4, 8, 8).astype(np.float32)
    from p2p_inference import softmax
    ref_s = softmax(s)
    res_s = softmax_numba(s)
    assert np.allclose(ref_s, res_s, atol=1e-5), "softmax mismatch"
    print("✓ softmax_numba")

    print("Tous les kernels validés.")
```

---

## Étape 2 — FragmentExecutor (`fragment_executor.py`)

### But

Classe responsable d'une seule couche. Charge les tenseurs, calcule, libère.
Remplace `LlamaLayer` dans `p2p_inference.py`.

### Fichier à créer : `fragment_executor.py`

```
P:\Projets\Inference\fragment_executor.py
```

### Architecture de la classe

```
FragmentExecutor
├── __init__(loader, layer_idx, config)   # ne charge PAS les tenseurs
├── __enter__()                           # charge TOUS les tenseurs de la couche en RAM
├── __exit__(...)                         # libère explicitement tous les tenseurs
├── forward(x, freqs_cis, cache_k, cache_v, start_pos) → (x_out, new_k, new_v)
└── _load_all_weights()                   # appelé par __enter__
```

### Contenu détaillé

#### 2.1 — Imports et signature

```python
"""
fragment_executor.py
====================
Exécuteur d'une couche de transformeur travaillant fragment par fragment.

Usage
-----
    with FragmentExecutor(loader, layer_idx=0, config=cfg) as executor:
        x, new_k, new_v = executor.forward(x, freqs_cis, cache_k, cache_v, start_pos=0)
    # Ici, toute la mémoire de la couche 0 a été libérée.
"""

import gc
import numpy as np
from typing import Optional, Tuple
from distribution.local import BaseFragmentLoader
from p2p_inference import ModelConfig, rms_norm, softmax, swiglu, apply_rotary_emb, precompute_freqs_cis

# Import kernels numba si disponibles, sinon fallback NumPy
try:
    from kernels_numba import rms_norm_numba, softmax_numba, swiglu_numba, warmup_kernels
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    rms_norm_numba = rms_norm
    softmax_numba  = softmax
    swiglu_numba   = swiglu
```

#### 2.2 — `__init__`

```python
class FragmentExecutor:
    def __init__(
        self,
        loader: BaseFragmentLoader,
        layer_idx: int,
        config: ModelConfig,
    ):
        """
        Paramètres
        ----------
        loader     : BaseFragmentLoader
                     Interface de chargement des fragments (local, réseau, P2P).
        layer_idx  : int
                     Index de la couche (0 à n_layers-1).
        config     : ModelConfig
                     Configuration du modèle.

        NB : aucun tenseur n'est chargé ici. Utiliser comme context manager :
             with FragmentExecutor(loader, idx, cfg) as ex:
                 ...
        """
        self.loader    = loader
        self.idx       = layer_idx
        self.cfg       = config
        self.pfx       = f"blk.{layer_idx}"

        # Tenseurs chargés dans __enter__, mis à None dans __exit__
        self._w_attn_norm: Optional[np.ndarray] = None
        self._wq:          Optional[np.ndarray] = None
        self._wk:          Optional[np.ndarray] = None
        self._wv:          Optional[np.ndarray] = None
        self._wo:          Optional[np.ndarray] = None
        self._w_ffn_norm:  Optional[np.ndarray] = None
        self._w_gate:      Optional[np.ndarray] = None
        self._w_up:        Optional[np.ndarray] = None
        self._w_down:      Optional[np.ndarray] = None
```

#### 2.3 — `__enter__` / `__exit__`

```python
    def __enter__(self) -> "FragmentExecutor":
        """Charge tous les tenseurs de la couche en mémoire."""
        self._load_all_weights()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Libère explicitement tous les tenseurs et force le GC."""
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
        return False  # ne pas supprimer les exceptions

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
```

#### 2.4 — `forward`

Même logique que `LlamaLayer.forward()` dans `p2p_inference.py` (lignes 228–351), mais :
- Utilise `self._wq`, `self._wk`, etc. au lieu de `self.engine.load_tensor()`
- Utilise les kernels numba si disponibles
- La fonction `proj()` reste identique

```python
    def forward(
        self,
        x: np.ndarray,
        freqs_cis: np.ndarray,
        cache_k: Optional[np.ndarray],
        cache_v: Optional[np.ndarray],
        start_pos: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Passe forward d'une couche Llama.

        Paramètres
        ----------
        x         : float32[seq_len, dim]
        freqs_cis : complex64[max_seq, head_dim/2]
        cache_k   : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        cache_v   : float32[prev_seq, n_kv_heads, head_dim] ou None (prefill)
        start_pos : int — position du premier token dans la séquence courante

        Retourne
        --------
        (x_out, new_k, new_v) — activation de sortie + KV à cacher
        new_k : float32[seq_len, n_kv_heads, head_dim]
        new_v : float32[seq_len, n_kv_heads, head_dim]
        """
        assert self._wq is not None, "FragmentExecutor doit être utilisé comme context manager"

        seq_len  = x.shape[0]
        head_dim = self.cfg.dim // self.cfg.n_heads

        def proj(inp, w):
            if w.ndim == 2:
                if w.shape[0] == inp.shape[1] and w.shape[1] != inp.shape[1]:
                    return inp @ w
                elif w.shape[1] == inp.shape[1] and w.shape[0] != inp.shape[1]:
                    return inp @ w.T
            return inp @ w

        # ── Attention ──────────────────────────────────────────
        xn = rms_norm_numba(x, self._w_attn_norm, self.cfg.norm_eps)

        xq = proj(xn, self._wq)
        xk = proj(xn, self._wk)
        xv = proj(xn, self._wv)

        q_head_dim = xq.shape[1] // self.cfg.n_heads
        k_head_dim = xk.shape[1] // self.cfg.n_kv_heads
        v_head_dim = xv.shape[1] // self.cfg.n_kv_heads

        xq = xq.reshape(seq_len, self.cfg.n_heads,    q_head_dim)
        xk = xk.reshape(seq_len, self.cfg.n_kv_heads, k_head_dim)
        xv = xv.reshape(seq_len, self.cfg.n_kv_heads, v_head_dim)

        # RoPE
        rope_dim    = q_head_dim
        all_freqs   = precompute_freqs_cis(rope_dim, start_pos + seq_len, theta=self.cfg.rope_freq_base)
        cur_freqs   = all_freqs[start_pos:start_pos + seq_len].reshape(seq_len, 1, -1)
        xq, xk      = apply_rotary_emb(xq, xk, cur_freqs)

        new_k = xk
        new_v = xv

        # KV cache
        if cache_k is not None:
            keys   = np.concatenate([cache_k, new_k], axis=0)
            values = np.concatenate([cache_v, new_v], axis=0)
        else:
            keys, values = new_k, new_v

        # GQA
        n_rep = self.cfg.n_heads // self.cfg.n_kv_heads
        if n_rep > 1:
            keys   = np.repeat(keys,   n_rep, axis=1)
            values = np.repeat(values, n_rep, axis=1)

        # Attention scores
        xq_t  = xq.transpose(1, 0, 2)
        keys_t = keys.transpose(1, 0, 2)
        vals_t = values.transpose(1, 0, 2)

        scores = np.matmul(xq_t, keys_t.transpose(0, 2, 1)) / np.sqrt(head_dim)

        if seq_len > 1:
            mask   = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
            scores = scores + mask

        probs  = softmax_numba(scores)
        output = np.matmul(probs, vals_t)
        output = output.transpose(1, 0, 2).reshape(seq_len, -1)

        h = x + proj(output, self._wo)

        # ── FFN SwiGLU ─────────────────────────────────────────
        xn     = rms_norm_numba(h, self._w_ffn_norm, self.cfg.norm_eps)
        gate   = proj(xn, self._w_gate)
        up     = proj(xn, self._w_up)
        hidden = swiglu_numba(gate) * up
        out    = proj(hidden, self._w_down)

        return h + out, new_k, new_v
```

---

## Étape 3 — Refactor de `generate()` dans `p2p_inference.py`

### But

Remplacer les instanciations de `LlamaLayer` dans `generate()` par `FragmentExecutor` avec context managers.
Les poids embedding et output restent chargés une seule fois (ils ne font pas partie d'une "couche").

### Modifications dans `p2p_inference.py`

#### 3.1 — Nouvel import en tête de fichier

```python
# Ligne à ajouter après les imports existants
try:
    from fragment_executor import FragmentExecutor
    _USE_FRAGMENT_EXECUTOR = True
except ImportError:
    _USE_FRAGMENT_EXECUTOR = False
```

#### 3.2 — Modification de `P2PInferenceEngine.generate()`

La signature ne change pas. Seul le corps de la boucle change.

**Phase 1 — Prefill (à modifier) :**

```python
# AVANT (ligne 619–622) :
for l in range(self.config.n_layers):
    layer = LlamaLayer(self, l)
    x, new_k, new_v = layer.forward(x, self.freqs_cis, None, None, start_pos=0)
    kv_cache.append((new_k, new_v))

# APRÈS :
for l in range(self.config.n_layers):
    if _USE_FRAGMENT_EXECUTOR:
        with FragmentExecutor(self._loader, l, self.config) as ex:
            x, new_k, new_v = ex.forward(x, self.freqs_cis, None, None, start_pos=0)
    else:
        layer = LlamaLayer(self, l)
        x, new_k, new_v = layer.forward(x, self.freqs_cis, None, None, start_pos=0)
    kv_cache.append((new_k, new_v))
```

**Phase 2 — Decode (à modifier) :**

```python
# AVANT (lignes 658–666) :
for l in range(self.config.n_layers):
    layer = LlamaLayer(self, l)
    ck, cv = kv_cache[l]
    x, new_k, new_v = layer.forward(x, self.freqs_cis, ck, cv, start_pos=start_pos)
    new_kv.append((...))

# APRÈS :
for l in range(self.config.n_layers):
    ck, cv = kv_cache[l]
    if _USE_FRAGMENT_EXECUTOR:
        with FragmentExecutor(self._loader, l, self.config) as ex:
            x, new_k, new_v = ex.forward(x, self.freqs_cis, ck, cv, start_pos=start_pos)
    else:
        layer = LlamaLayer(self, l)
        x, new_k, new_v = layer.forward(x, self.freqs_cis, ck, cv, start_pos=start_pos)
    new_kv.append((
        np.concatenate([ck, new_k], axis=0),
        np.concatenate([cv, new_v], axis=0)
    ))
```

#### 3.3 — Ajout d'un `_loader` dans `P2PInferenceEngine.__init__`

`FragmentExecutor` attend un `BaseFragmentLoader`. Il faut exposer le loader depuis l'engine.

```python
# Dans P2PInferenceEngine.__init__(), après la construction de fragments_map :
from distribution.local import LocalFragmentLoader
self._loader = LocalFragmentLoader(self.fragments_dir, verbose=self.verbose)
```

> Note : `LocalFragmentLoader` relit le manifest — c'est redondant avec ce que fait déjà `P2PInferenceEngine.__init__()`. À terme, refactoriser pour que l'engine délègue entièrement au loader. Pour l'instant, la duplication est acceptable.

#### 3.4 — Warm-up numba au démarrage de l'engine

```python
# Dans P2PInferenceEngine.__init__(), après self._loader = ... :
if _USE_FRAGMENT_EXECUTOR:
    try:
        from kernels_numba import warmup_kernels
        warmup_kernels()
    except ImportError:
        pass
```

---

## Étape 4 — Memory Tracker

### But

Mesurer et journaliser l'empreinte mémoire réelle à chaque étape pour vérifier que l'objectif "une couche en RAM à la fois" est atteint.

### Modification de `fragment_executor.py`

#### 4.1 — Import conditionnel de psutil

```python
try:
    import psutil, os
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

def _rss_mb() -> float:
    """Retourne la mémoire résidente (RSS) du processus en MB."""
    if _PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return 0.0
```

#### 4.2 — Paramètre `track_memory` dans `FragmentExecutor`

```python
class FragmentExecutor:
    def __init__(self, loader, layer_idx, config, track_memory: bool = False):
        ...
        self.track_memory = track_memory
        self._mem_before: float = 0.0
        self._mem_after_load: float = 0.0
        self._mem_after_free: float = 0.0

    def __enter__(self):
        if self.track_memory:
            self._mem_before = _rss_mb()
        self._load_all_weights()
        if self.track_memory:
            self._mem_after_load = _rss_mb()
            delta = self._mem_after_load - self._mem_before
            print(f"[MemTrack] Couche {self.idx} chargée : +{delta:.1f} MB (total: {self._mem_after_load:.1f} MB)")
        return self

    def __exit__(self, *args):
        # ... (libération comme avant)
        if self.track_memory:
            self._mem_after_free = _rss_mb()
            delta = self._mem_after_load - self._mem_after_free
            print(f"[MemTrack] Couche {self.idx} libérée : -{delta:.1f} MB (total: {self._mem_after_free:.1f} MB)")
        return False
```

#### 4.3 — Argument CLI dans `p2p_inference.py`

```python
# Dans le bloc argparse :
parser.add_argument("--track-memory", action="store_true", help="Afficher l'empreinte mémoire par couche")

# Dans engine.generate() — passer track_memory=args.track_memory à FragmentExecutor
```

---

## Étape 5 — Tests & Validation (`tests_debug/test_fragment_executor.py`)

### But

Vérifier que :
1. Chaque kernel numba est numériquement identique à la référence NumPy (atol=1e-5)
2. `FragmentExecutor.forward()` produit les mêmes logits que `LlamaLayer.forward()`
3. `generate()` avec `FragmentExecutor` produit les mêmes tokens qu'avec `LlamaLayer`
4. La mémoire est correctement libérée (test avec `--track-memory`)

### Fichier à créer : `tests_debug/test_fragment_executor.py`

```python
"""
tests_debug/test_fragment_executor.py
======================================
Validation du Fragment Executor contre la référence LlamaLayer.

Usage
-----
    # Test rapide (kernels unitaires seulement)
    .venv\Scripts\python.exe tests_debug/test_fragment_executor.py --units-only

    # Test complet (nécessite le dossier de fragments)
    .venv\Scripts\python.exe tests_debug/test_fragment_executor.py \
        models/tinyllama_q8_fragments_v2

    # Avec comparaison token par token
    .venv\Scripts\python.exe tests_debug/test_fragment_executor.py \
        models/tinyllama_q8_fragments_v2 --compare-tokens
"""
```

#### 5.1 — Tests unitaires des kernels

```python
def test_rms_norm():
    """rms_norm_numba vs rms_norm numpy, atol=1e-5."""
    x = np.random.randn(8, 2048).astype(np.float32)
    w = np.random.randn(2048).astype(np.float32)
    ref = rms_norm(x, w, 1e-5)
    res = rms_norm_numba(x, w, 1e-5)
    assert np.allclose(ref, res, atol=1e-5), f"max_diff={np.max(np.abs(ref-res))}"
    print("✓ rms_norm_numba")

def test_softmax():
    x = np.random.randn(32, 16, 16).astype(np.float32)
    ref = softmax(x)
    res = softmax_numba(x)
    assert np.allclose(ref, res, atol=1e-5)
    print("✓ softmax_numba")

def test_swiglu():
    x = np.random.randn(4, 5632).astype(np.float32)
    ref = swiglu(x)
    res = swiglu_numba(x)
    assert np.allclose(ref, res, atol=1e-5)
    print("✓ swiglu_numba")

def test_dequantize_q8_0(fragments_dir: str):
    """dequantize_q8_0 numba vs LocalFragmentLoader legacy."""
    from distribution.local import LocalFragmentLoader
    from kernels_numba import dequantize_q8_0
    loader = LocalFragmentLoader(fragments_dir)
    for tname in ["blk.0.attn_q.weight", "blk.0.ffn_gate.weight"]:
        ref = loader.load_tensor(tname)
        frags = loader.fragments_map[tname]
        raw = b"".join(loader.load_raw(f["fragment_id"]) for f in frags)
        res = dequantize_q8_0(raw, frags[0]["shape"])
        assert np.allclose(ref, res, atol=1e-5), f"{tname}: max_diff={np.max(np.abs(ref-res))}"
    print("✓ dequantize_q8_0")
```

#### 5.2 — Test d'activation couche par couche

```python
def test_layer_output(fragments_dir: str):
    """
    Compare la sortie de FragmentExecutor.forward() vs LlamaLayer.forward()
    sur l'activation réelle de la couche 0, pour un batch de 4 tokens.
    Tolérance : atol=1e-4 (légères différences numériques float32 acceptables).
    """
    from p2p_inference import P2PInferenceEngine, LlamaLayer, precompute_freqs_cis
    from fragment_executor import FragmentExecutor
    from distribution.local import LocalFragmentLoader

    engine = P2PInferenceEngine(fragments_dir)
    loader = LocalFragmentLoader(fragments_dir)
    cfg    = engine.config

    x         = np.random.randn(4, cfg.dim).astype(np.float32)
    freqs_cis = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.dim * 2)

    # Référence : LlamaLayer
    ref_layer = LlamaLayer(engine, 0)
    ref_x, ref_k, ref_v = ref_layer.forward(x, freqs_cis, None, None, start_pos=0)

    # Résultat : FragmentExecutor
    with FragmentExecutor(loader, 0, cfg) as ex:
        res_x, res_k, res_v = ex.forward(x, freqs_cis, None, None, start_pos=0)

    assert np.allclose(ref_x, res_x, atol=1e-4), f"activation mismatch: max_diff={np.max(np.abs(ref_x-res_x))}"
    assert np.allclose(ref_k, res_k, atol=1e-4), "cache_k mismatch"
    assert np.allclose(ref_v, res_v, atol=1e-4), "cache_v mismatch"
    print("✓ FragmentExecutor.forward() == LlamaLayer.forward()")
```

#### 5.3 — Test end-to-end de génération

```python
def test_generation_tokens(fragments_dir: str, prompt: str = "Hello"):
    """
    Génère 5 tokens avec le moteur original et le moteur Fragment Executor.
    Vérifie que les tokens générés sont identiques (temperature=0 pour déterminisme).
    """
    import p2p_inference

    # Désactiver temporairement le fragment executor
    p2p_inference._USE_FRAGMENT_EXECUTOR = False
    engine_ref = p2p_inference.P2PInferenceEngine(fragments_dir)
    tokens_ref = engine_ref.generate(prompt, max_tokens=5, temperature=0.0)

    # Activer le fragment executor
    p2p_inference._USE_FRAGMENT_EXECUTOR = True
    engine_new = p2p_inference.P2PInferenceEngine(fragments_dir)
    tokens_new = engine_new.generate(prompt, max_tokens=5, temperature=0.0)

    assert tokens_ref == tokens_new, f"Token mismatch!\nRef: {tokens_ref}\nNew: {tokens_new}"
    print(f"✓ Génération identique : {tokens_ref}")
```

#### 5.4 — Test de libération mémoire

```python
def test_memory_release(fragments_dir: str):
    """
    Vérifie que la mémoire augmente lors du chargement d'une couche
    et diminue après __exit__.
    """
    try:
        import psutil, os
    except ImportError:
        print("[SKIP] psutil non disponible — test mémoire ignoré")
        return

    from distribution.local import LocalFragmentLoader
    from fragment_executor import FragmentExecutor
    from p2p_inference import P2PInferenceEngine

    engine = P2PInferenceEngine(fragments_dir)
    loader = LocalFragmentLoader(fragments_dir)
    proc   = psutil.Process(os.getpid())

    mem_before = proc.memory_info().rss
    with FragmentExecutor(loader, 0, engine.config, track_memory=True) as ex:
        mem_loaded = proc.memory_info().rss
        # La couche 0 de TinyLlama fait ~11 MB
        delta_load = (mem_loaded - mem_before) / 1024 / 1024
        assert delta_load > 5, f"Chargement insuffisant: +{delta_load:.1f} MB (attendu >5 MB)"

    mem_after = proc.memory_info().rss
    delta_free = (mem_loaded - mem_after) / 1024 / 1024
    assert delta_free > 3, f"Libération insuffisante: -{delta_free:.1f} MB (attendu >3 MB)"
    print(f"✓ Mémoire libérée après __exit__ : -{delta_free:.1f} MB")
```

---

## Fichiers touchés — récapitulatif

| Fichier | Action | Étape |
|---------|--------|-------|
| `kernels_numba.py` | **Créer** | 1 |
| `fragment_executor.py` | **Créer** | 2 + 4 |
| `p2p_inference.py` | **Modifier** — imports + `generate()` + `__init__` | 3 |
| `requirements.txt` | **Modifier** — ajouter `numba>=0.59.0`, `psutil>=5.9` | 1 |
| `tests_debug/test_fragment_executor.py` | **Créer** | 5 |

---

## Compatibilité avec l'architecture distribuée (Phase 4)

Le `FragmentExecutor` est conçu pour être nativement distribuable. Quand `reseau.py` et `p2p.py` seront implémentés, il suffira de remplacer :

```python
# Local
with FragmentExecutor(LocalFragmentLoader(fragments_dir), l, cfg) as ex:
    ...

# Réseau (futur)
with FragmentExecutor(ReseauFragmentLoader(node_address), l, cfg) as ex:
    ...

# P2P (futur)
with FragmentExecutor(P2PFragmentLoader(dht_key), l, cfg) as ex:
    ...
```

Aucun changement dans `generate()` ni dans la logique de calcul.

---

## Commandes de validation finale

```bash
# 1. Installer les nouvelles dépendances
.venv\Scripts\python.exe -m pip install numba psutil

# 2. Valider les kernels unitaires (pas besoin du modèle)
.venv\Scripts\python.exe kernels_numba.py

# 3. Valider le Fragment Executor vs LlamaLayer (couche 0)
.venv\Scripts\python.exe tests_debug/test_fragment_executor.py \
    models/tinyllama_q8_fragments_v2 --units-only

# 4. Valider la génération end-to-end
.venv\Scripts\python.exe tests_debug/test_fragment_executor.py \
    models/tinyllama_q8_fragments_v2 --compare-tokens

# 5. Run avec tracking mémoire
.venv\Scripts\python.exe p2p_inference.py models/tinyllama_q8_fragments_v2 \
    --prompt "Hello" --max-tokens 5 --track-memory
```
