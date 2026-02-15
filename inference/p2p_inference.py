# python app.py --fragments-dir models/tinyllama_q8_fragments_v2

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Flag FragmentExecutor — import différé dans generate() pour éviter l'import circulaire
# (fragment_executor.py importe depuis p2p_inference, donc l'import au top-level échouerait)
def _try_import_fragment_executor():
    """Retourne la classe FragmentExecutor ou None si non disponible."""
    try:
        from .fragment_executor import FragmentExecutor
        return FragmentExecutor
    except ImportError:
        return None

# ==========================================
# Configuration & Utils
# ==========================================

@dataclass
class ModelConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    norm_eps: float = 1e-5
    rope_freq_base: float = 10000.0

    @staticmethod
    def from_manifest(manifest: dict) -> 'ModelConfig':
        # Nouveau format: utiliser la section "config" si disponible, sinon metadata
        config_section = manifest.get("config", {})
        meta = manifest.get("metadata", {})

        # Helper to safely get int/float
        def get_val(key, default, cast=int):
            # Priorité à la section config
            val = config_section.get(key, meta.get(key, default))
            if isinstance(val, list) and len(val) > 0: val = val[0]
            try:
                return cast(val)
            except:
                return default

        # Mapping des clés GGUF vers notre configuration (pour compatibilité GGUF)
        gguf_key_mapping = {
            'llama.embedding_length': 'dim',
            'llama.feed_forward_length': 'hidden_dim',
            'llama.block_count': 'n_layers',
            'llama.attention.head_count': 'n_heads',
            'llama.attention.head_count_kv': 'n_kv_heads',
            'llama.vocab_size': 'vocab_size',
        }
        
        # Extraire les valeurs des clés GGUF si elles existent
        for gguf_key, config_key in gguf_key_mapping.items():
            if gguf_key in meta:
                val = meta[gguf_key]
                if isinstance(val, list) and len(val) > 0:
                    val = val[0]
                config_section[config_key] = config_section.get(config_key, val)

        # Fix #3 : lire n_heads avant n_kv_heads pour que le fallback soit correct
        n_heads = get_val("n_heads", 32)
        n_kv_heads = get_val("n_kv_heads", n_heads)  # fallback = n_heads, pas 32

        return ModelConfig(
            dim=get_val("dim", 4096),
            hidden_dim=get_val("hidden_dim", 11008),
            n_layers=get_val("n_layers", 32),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=get_val("vocab_size", 32000),
            norm_eps=get_val("norm_eps", 1e-5, float),
            rope_freq_base=get_val("rope_freq_base", 10000.0, float)
        )

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from tokenizers import Tokenizer as HFTokenizer
    _hf_tokenizers = True
except ImportError:
    _hf_tokenizers = False

class Tokenizer:
    """Charge tokenizer.json (HuggingFace) ou tokenizer.model (SentencePiece)."""

    def __init__(self, path: str):
        path = str(path)
        if path.endswith(".json"):
            if not _hf_tokenizers:
                raise ImportError("tokenizers (HuggingFace) non installe")
            self._hf = HFTokenizer.from_file(path)
            self._spm = None
            # Récupérer les ids spéciaux
            vocab = self._hf.get_vocab()
            self.bos_id = vocab.get("<s>", vocab.get("[BOS]", 1))
            self.eos_id = vocab.get("</s>", vocab.get("[EOS]", 2))
        else:
            if not spm:
                raise ImportError("sentencepiece non installe")
            self._spm = spm.SentencePieceProcessor(model_file=path)
            self._hf = None
            self.bos_id = self._spm.bos_id()
            self.eos_id = self._spm.eos_id()

    def encode(self, text: str) -> List[int]:
        if self._hf:
            return self._hf.encode(text).ids
        return self._spm.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self._hf:
            return self._hf.decode(ids)
        return self._spm.decode(ids)

class SimpleTokenizer:
    """Minimal tokenizer extracting vocabulary from GGUF metadata (Fallback)."""
    def __init__(self, manifest: dict):
        self.vocab = []
        self.token_to_id = {}

        meta = manifest.get("metadata", {})

        # GGUF keys: tokenizer.ggml.tokens
        tokens = meta.get("tokenizer.ggml.tokens", [])

        if not tokens:
            # Fallback: dummy vocab
            print("[WARN] No vocabulary found in manifest! Using dummy.")
            self.vocab = ["<unk>", "<s>", "</s>"] + [f"token_{i}" for i in range(32000)]
        else:
            self.vocab = tokens

        # Build reverse map
        for i, t in enumerate(self.vocab):
            # Tokens might be strings or weird internal GGUF repr
            # We treat them as is for now
            self.token_to_id[str(t)] = i

        self.bos_id = self.token_to_id.get("<s>", 1)
        self.eos_id = self.token_to_id.get("</s>", 2)

    def encode(self, text: str) -> List[int]:
        # Extremely naive whitespace splitting for POC
        # Hack: map words if they exist, else unk
        words = text.split()
        ids = [self.bos_id]
        for w in words:
            # Try with leading space (Llama convention)
            w_sp = " " + w
            if w_sp in self.token_to_id:
                ids.append(self.token_to_id[w_sp])
            elif w in self.token_to_id:
                ids.append(self.token_to_id[w])
            else:
                pass
        return ids

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                token = self.vocab[i]
                # Replace standard sentencepiece placeholder
                if isinstance(token, str):
                    t = token.replace(" ", " ")
                    res.append(t)
                else:
                    res.append(str(token))
        return "".join(res)

# ==========================================
# Rope & Utils
# ==========================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> np.ndarray:
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end).astype(np.float32)
    freqs = np.outer(t, freqs)
    freqs_cis = np.exp(1j * freqs).astype(np.complex64)
    return freqs_cis

def apply_rotary_emb(xq: np.ndarray, xk: np.ndarray, freqs_cis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # xq: [seq_len, n_heads, head_dim]
    # GGUF models have weights permuted so that pairs are adjacent (0,1), (2,3)...
    # So we simply reshape last dim to (-1, 2) and treat as complex.

    # Reshape to separate real/imag parts: [..., head_dim/2, 2]
    xq_r = xq.astype(np.float32).reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.astype(np.float32).reshape(*xk.shape[:-1], -1, 2)

    # Construct complex: [..., head_dim/2]
    xq_c = xq_r[..., 0] + 1j * xq_r[..., 1]
    xk_c = xk_r[..., 0] + 1j * xk_r[..., 1]

    # Broadcast freqs_cis [seq_len, head_dim/2] to [seq_len, 1, head_dim/2]
    # freqs_cis matches head_dim/2
    freqs = freqs_cis[:xq.shape[0]].reshape(xq.shape[0], 1, -1)

    # Multiply
    xq_out_c = xq_c * freqs
    xk_out_c = xk_c * freqs

    # Convert back to real
    # Stack real/imag: [..., head_dim/2, 2]
    xq_out = np.stack([xq_out_c.real, xq_out_c.imag], axis=-1).reshape(xq.shape)
    xk_out = np.stack([xk_out_c.real, xk_out_c.imag], axis=-1).reshape(xk.shape)

    return xq_out, xk_out

# ==========================================
# Model Logic
# ==========================================

class LlamaLayer:
    def __init__(self, engine, layer_idx: int):
        self.engine = engine
        self.idx = layer_idx
        self.cfg = engine.config

        # Load weights names map
        # GGUF tensor naming convention:
        # blk.N.attn_q.weight, blk.N.attn_norm.weight ...
        self.pfx = f"blk.{layer_idx}"

    def forward(self, x: np.ndarray, freqs_cis: np.ndarray, cache_k, cache_v, start_pos: int):
        # x: [seq_len, dim]
        seq_len = x.shape[0]
        head_dim = self.cfg.dim // self.cfg.n_heads
        kv_dim   = self.cfg.n_kv_heads * head_dim

        # Fix #2 : projection avec détection automatique de l'orientation du poids.
        # GGUF stocke généralement en [in, out] → x @ w correct.
        # Mais certains modèles exportent en [out, in] (style PyTorch) → x @ w.T nécessaire.
        # Fix #3 : utiliser les dimensions réelles des tenseurs plutôt que cfg.dim
        def proj(inp, w):
            # Détecter l'orientation du poids
            if w.ndim == 2:
                if w.shape[0] == inp.shape[1] and w.shape[1] != inp.shape[1]:
                    return inp @ w         # poids en [in, out] → direct
                elif w.shape[1] == inp.shape[1] and w.shape[0] != inp.shape[1]:
                    return inp @ w.T       # poids en [out, in] → transposer
            return inp @ w

        # 1. Attention
        w_norm = self.engine.load_tensor(f"{self.pfx}.attn_norm.weight")
        xn = rms_norm(x, w_norm, self.cfg.norm_eps)

        wq = self.engine.load_tensor(f"{self.pfx}.attn_q.weight")
        wk = self.engine.load_tensor(f"{self.pfx}.attn_k.weight")
        wv = self.engine.load_tensor(f"{self.pfx}.attn_v.weight")
        wo = self.engine.load_tensor(f"{self.pfx}.attn_output.weight")

        # Utiliser les dimensions réelles des tenseurs
        xq = proj(xn, wq)          # [seq, wq.shape[1]]
        xk = proj(xn, wk)          # [seq, wk.shape[1]]
        xv = proj(xn, wv)          # [seq, wv.shape[1]]

        # Calculer les dimensions réelles des têtes
        # Pour les architectures custom, le nombre de têtes peut être différent
        q_head_dim = xq.shape[1] // self.cfg.n_heads
        k_head_dim = xk.shape[1] // self.cfg.n_kv_heads
        v_head_dim = xv.shape[1] // self.cfg.n_kv_heads
        
        # Vérifier que les dimensions sont cohérentes
        if xq.shape[1] % self.cfg.n_heads != 0:
            # Architecture custom: recalculer le nombre de têtes
            actual_n_heads = xq.shape[1] // q_head_dim if q_head_dim > 0 else self.cfg.n_heads
            print(f"[WARN] Architecture custom détectée: n_heads={actual_n_heads} (config: {self.cfg.n_heads})")
        
        xq = xq.reshape(seq_len, self.cfg.n_heads,    q_head_dim)
        xk = xk.reshape(seq_len, self.cfg.n_kv_heads, k_head_dim)
        xv = xv.reshape(seq_len, self.cfg.n_kv_heads, v_head_dim)

        # RoPE — cache par head_dim réel (évite O(n_layers×n_tokens) recalculs)
        min_end = start_pos + seq_len + 1
        rope_cache = self.engine._rope_cache
        if q_head_dim not in rope_cache or rope_cache[q_head_dim].shape[0] < min_end:
            rope_cache[q_head_dim] = precompute_freqs_cis(
                q_head_dim, max(min_end, self.engine.config.dim * 2),
                theta=self.engine.config.rope_freq_base
            )
        current_freqs = rope_cache[q_head_dim][start_pos:start_pos + seq_len].reshape(seq_len, 1, -1)
        xq, xk = apply_rotary_emb(xq, xk, current_freqs)

        # KV cache : conserver les nouvelles clés/valeurs (avant répétition GQA)
        new_k = xk  # [seq_len, n_kv_heads, head_dim]
        new_v = xv  # [seq_len, n_kv_heads, head_dim]

        # Concaténer avec le cache des tokens précédents si disponible
        if cache_k is not None:
            keys   = np.concatenate([cache_k, new_k], axis=0)  # [total_seq, n_kv_heads, head_dim]
            values = np.concatenate([cache_v, new_v], axis=0)
        else:
            keys   = new_k
            values = new_v

        # GQA: répéter les têtes KV pour aligner avec n_heads
        n_rep = self.cfg.n_heads // self.cfg.n_kv_heads
        if n_rep > 1:
            keys   = np.repeat(keys,   n_rep, axis=1)
            values = np.repeat(values, n_rep, axis=1)

        # Attention scores : [n_heads, seq, seq]
        xq    = xq.transpose(1, 0, 2)
        keys   = keys.transpose(1, 0, 2)
        values = values.transpose(1, 0, 2)

        scores = np.matmul(xq, keys.transpose(0, 2, 1)) / np.sqrt(head_dim)

        # Masque causal — mis en cache pour éviter la réallocation à chaque couche
        if seq_len > 1:
            if seq_len not in self.engine._attn_mask_cache:
                self.engine._attn_mask_cache[seq_len] = np.triu(
                    np.full((seq_len, seq_len), float("-inf")), k=1
                )
            scores = scores + self.engine._attn_mask_cache[seq_len]

        probs  = softmax(scores)                                          # [n_heads, seq, seq]
        output = np.matmul(probs, values)                                 # [n_heads, seq, head_dim]
        # output a la forme [n_heads, seq, head_dim], nous devons le reshaper
        # La dimension totale après transpose sera seq_len * n_heads * head_dim
        output = output.transpose(1, 0, 2)  # [seq, n_heads, head_dim]
        # Utiliser la dimension réelle plutôt que cfg.dim
        actual_output_dim = output.shape[1] * output.shape[2]  # n_heads * head_dim
        output = output.reshape(seq_len, actual_output_dim)  # [seq, n_heads * head_dim]

        # Projection de sortie - wo a la forme [out_dim, in_dim]
        # La nouvelle fonction proj détecte automatiquement l'orientation
        h = x + proj(output, wo)

        # 2. FFN SwiGLU
        w_ffn_norm = self.engine.load_tensor(f"{self.pfx}.ffn_norm.weight")
        xn = rms_norm(h, w_ffn_norm, self.cfg.norm_eps)

        w_gate = self.engine.load_tensor(f"{self.pfx}.ffn_gate.weight")
        w_up   = self.engine.load_tensor(f"{self.pfx}.ffn_up.weight")
        w_down = self.engine.load_tensor(f"{self.pfx}.ffn_down.weight")

        # SwiGLU : silu(gate) * up
        # La nouvelle fonction proj détecte automatiquement l'orientation
        gate   = proj(xn, w_gate)  # [seq, hidden]
        up     = proj(xn, w_up)    # [seq, hidden]
        hidden = swiglu(gate) * up  # [seq, hidden]
        out    = proj(hidden, w_down)  # [seq, dim]

        # Retourner uniquement les nouvelles clés/valeurs (sans répétition GQA).
        # generate() se charge de les concaténer avec le cache des couches.
        return h + out, new_k, new_v

# ==========================================
# Model Layers (NumPy)
# ==========================================

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    # x: [seq, dim] ou [dim]
    # axis=-1 : normalisation PAR TOKEN (une variance par ligne, pas globale)
    ss = np.mean(x ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(ss + eps)) * weight

def softmax(x: np.ndarray) -> np.ndarray:
    # x: [..., seq]
    # sub max for stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def swiglu(x: np.ndarray) -> np.ndarray:
    # SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0 + np.exp(-x))

# ==========================================
# Sampling
# ==========================================

def _sample_logits(logits: np.ndarray, temperature: float,
                   top_k: int = 0, top_p: float = 1.0) -> int:
    """Temperature / top-k / top-p sampling sur un vecteur de logits."""
    logits = logits.flatten().astype(np.float32)
    logits /= max(temperature, 1e-8)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    if top_k > 0:
        k = min(top_k, len(probs))
        top_idx = np.argpartition(probs, -k)[-k:]
        mask = np.zeros_like(probs)
        mask[top_idx] = 1.0
        probs = probs * mask
        probs /= probs.sum()

    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, top_p)
        probs[sorted_idx[cutoff + 1:]] = 0.0
        probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


# ==========================================
# Inference Engine
# ==========================================

class P2PInferenceEngine:
    def __init__(self, fragments_dir: str, verbose: bool = False, cache_weights: bool = False):
        self.fragments_dir = Path(fragments_dir)
        # Cache des tenseurs dequantises. Desactive par defaut : sur Magistral-Small
        # les poids float32 representent ~80 GB, ce qui provoque du swap sur PC standard.
        # Activer uniquement sur machines avec suffisamment de RAM libre.
        self._weight_cache: Optional[Dict[str, np.ndarray]] = {} if cache_weights else None
        self.verbose = verbose

        # Load Manifest
        manifest_path = self.fragments_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("Manifest not found")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.config = ModelConfig.from_manifest(self.manifest)

        # Tokenizer setup — chercher tokenizer.json puis tokenizer.model
        # dans le dossier fragments, son parent, puis le répertoire courant
        tokenizer_path = None
        for candidate_dir in [self.fragments_dir, self.fragments_dir.parent, Path(".")]:
            for filename in ["tokenizer.json", "tokenizer.model"]:
                p = candidate_dir / filename
                if p.exists():
                    tokenizer_path = p
                    break
            if tokenizer_path:
                break

        if tokenizer_path:
            try:
                self.tokenizer = Tokenizer(str(tokenizer_path))
                print(f"[INFO] Tokenizer charge : {tokenizer_path}")
            except ImportError as e:
                print(f"[WARN] {e} — fallback SimpleTokenizer")
                self.tokenizer = SimpleTokenizer(self.manifest)
        else:
            print("[WARN] Aucun tokenizer trouve — fallback SimpleTokenizer (qualite reduite)")
            self.tokenizer = SimpleTokenizer(self.manifest)

        print(f"Loaded config: {self.config}")

        # Extraire les dimensions spécifiques des tenseurs si disponibles
        self.tensor_specifics = self.manifest.get("tensor_specifics", {})
        print(f"Tensor specifics: {self.tensor_specifics}")

        # Weight Index
        self.fragments_map = {} # tensor_name -> list of fragment info
        for f in self.manifest["fragments"]:
            tname = f.get("tensor_name")
            if tname:
                if tname not in self.fragments_map:
                    self.fragments_map[tname] = []
                self.fragments_map[tname].append(f)

        # Sort shards
        for tname in self.fragments_map:
            self.fragments_map[tname].sort(key=lambda x: x["shard_index"])

        # Cache RoPE par head_dim (calculé une seule fois par dimension rencontrée)
        self._rope_cache: Dict[int, np.ndarray] = {}
        # Précalcul pour le head_dim nominal (optimisation si l'archi est standard)
        _nom_hd = self.config.dim // self.config.n_heads
        self._rope_cache[_nom_hd] = precompute_freqs_cis(
            _nom_hd, self.config.dim * 2, theta=self.config.rope_freq_base
        )
        self.freqs_cis = self._rope_cache[_nom_hd]  # compat ancienne API
        # Cache du masque causal (clé = seq_len, valeur = matrice triu)
        self._attn_mask_cache: Dict[int, np.ndarray] = {}

        # Loader dédié pour FragmentExecutor (import différé pour éviter l'import circulaire)
        _FE = _try_import_fragment_executor()
        if _FE is not None:
            from distribution.local import LocalFragmentLoader
            # cache_raw=True : conserve les octets bruts en RAM après le premier prefill.
            # Évite de relire le disque à chaque token de decode.
            # Coût RAM : ~4.5x moins que float32 (Q4_K 4-bit vs float32 32-bit).
            self._loader = LocalFragmentLoader(self.fragments_dir, verbose=False, cache_raw=True)
            # Warm-up JIT des kernels numba (compilation une seule fois, cache disque ensuite)
            try:
                from .kernels_numba import warmup_kernels
                warmup_kernels()
            except ImportError:
                pass
            # Warm-up des kernels GEMV Q4_K et Q6_K
            active = []
            try:
                from dequantize.Q4_K_GGUF import warmup_q4k_gemv
                warmup_q4k_gemv()
                active.append("Q4_K")
            except ImportError:
                pass
            try:
                from dequantize.Q6_K_GGUF import warmup_q6k_gemv
                warmup_q6k_gemv()
                active.append("Q6_K")
            except ImportError:
                pass
            if active:
                print(f"[fragment_executor] Fused GEMV actif : {', '.join(active)}")
        else:
            self._loader = None

    def get_attention_dims(self, layer_idx: int) -> Dict[str, int]:
        """Obtenir les dimensions spécifiques pour les tenseurs d'attention."""
        defaults = {
            "q_dim": self.config.dim,
            "k_dim": self.config.dim,
            "v_dim": self.config.dim,
            "output_dim": self.config.dim
        }
        
        if "attention" in self.tensor_specifics:
            defaults.update(self.tensor_specifics["attention"])
        
        return defaults

    def get_ffn_dims(self, layer_idx: int) -> Dict[str, int]:
        """Obtenir les dimensions spécifiques pour les tenseurs FFN."""
        defaults = {
            "gate_dim": self.config.hidden_dim,
            "up_dim": self.config.hidden_dim,
            "down_dim": self.config.dim
        }
        
        if "ffn" in self.tensor_specifics:
            defaults.update(self.tensor_specifics["ffn"])
        
        return defaults

    def load_tensor(self, tensor_name: str) -> np.ndarray:
        # Verifier le cache : les poids sont identiques pour tous les tokens
        if self._weight_cache is not None and tensor_name in self._weight_cache:
            return self._weight_cache[tensor_name]

        fragments = self.fragments_map.get(tensor_name)
        if not fragments:
             # print(f"⚠️ Tensor missing: {tensor_name}")
             # Return random for stability
             return np.random.normal(0, 0.01, size=(64, 64)).astype(np.float32)

        if self.verbose:
            print(f"[FILE] [P2P] Loading '{tensor_name}' from {len(fragments)} fragments")
            # print(f"   └── Files: {[f['fragment_id'] + '.dat' for f in fragments]}")

        # Reassemble data from shards
        full_data = bytearray()
        for frag in fragments:
             path = self.fragments_dir / f"{frag['fragment_id']}.dat"
             if self.verbose:
                 print(f"    [READ] Reading fragment: {path.name}")

             if not path.exists():
                  raise FileNotFoundError(f"Fragment file missing: {path}")
             with open(path, "rb") as f:
                  full_data.extend(f.read())

        data = bytes(full_data)

        # Use metadata from first fragment
        frag = fragments[0]
        dtype_str = frag["dtype"]
        shape = tuple(frag["shape"])

        res = None
        if "float" in dtype_str or "int32" in dtype_str:
            arr = np.frombuffer(data, dtype=dtype_str).reshape(shape)
            res = arr.astype(np.float32)
        else:
            # Use the centralized dequantize module for all quantized formats
            tensor_type = frag.get("tensor_type", "")
            try:
                from dequantize import dequantize
                res = dequantize(data, tensor_type, shape)
            except ImportError:
                if self.verbose:
                    print(f"[WARN] Module dequantize non disponible, retour à l'ancienne méthode")
                # Fallback to old Q8_0 implementation for backward compatibility
                if "Q8_0" in tensor_type:
                    # Q8_0 Dequantization (legacy fallback)
                    dt = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))])
                    if len(data) % 34 != 0:
                        if self.verbose: print(f"[ERROR] Error: {tensor_name} data size mismatch")
                        res = np.zeros(shape, dtype=np.float32)
                    else:
                        blocks = np.frombuffer(data, dtype=dt)
                        d = blocks['d'].astype(np.float32)[:, None]
                        qs = blocks['qs'].astype(np.float32)
                        decoded = (d * qs).flatten()
                        if len(shape) == 2:
                            out_dim = shape[-1]
                            in_dim  = shape[0]
                            res = decoded.reshape([out_dim, in_dim]).T.astype(np.float32)
                        else:
                            res = decoded.reshape(shape).astype(np.float32)
                else:
                    res = np.zeros(shape, dtype=np.float32)
            except NotImplementedError as e:
                if self.verbose:
                    print(f"[WARN] Format de quantification non supporté: {e}")
                res = np.zeros(shape, dtype=np.float32)
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Erreur lors de la déquantization: {e}")
                res = np.zeros(shape, dtype=np.float32)

        # Debug Stats
        if self.verbose:
            print(f"    [STATS] Mean={np.mean(res):.4f} Std={np.std(res):.4f} Range=[{np.min(res):.4f}, {np.max(res):.4f}]")

        # Stocker dans le cache pour eviter de recharger/redequantiser aux prochains tokens
        if self._weight_cache is not None:
            self._weight_cache[tensor_name] = res

        return res

    def generate(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                 track_memory: bool = False):
        """
        Génère max_tokens tokens à partir du prompt avec KV cache.

        Phase 1 — Prefill  : traite tous les tokens du prompt en une seule passe O(n).
                              Construit le cache K/V pour chaque couche.
        Phase 2 — Decode   : génère un token à la fois O(1) en réutilisant le cache.
                              Coût total : O(n + t) au lieu de O((n+t)²).
        """
        tokens = self.tokenizer.encode(prompt)
        if not (len(tokens) > 0 and tokens[0] == 1):
            tokens = [1] + tokens

        print(f"Prompt : {len(tokens)} token(s) — {tokens}")

        # === Chargement des poids fixes (une seule fois) ===
        w_emb = self.load_tensor("token_embd.weight")
        if w_emb.ndim == 2 and w_emb.shape == (self.config.dim, self.config.vocab_size):
            w_emb = w_emb.T  # [dim, vocab] → [vocab, dim]
        elif w_emb.ndim == 2 and w_emb.shape != (self.config.vocab_size, self.config.dim):
            print(f"WARN: embedding shape inattendue {w_emb.shape}")

        w_out = self.load_tensor("output.weight")
        if w_out.ndim == 2 and w_out.shape == (self.config.vocab_size, self.config.dim):
            w_out = w_out.T  # [vocab, dim] → [dim, vocab]
        elif w_out.ndim == 2 and w_out.shape != (self.config.dim, self.config.vocab_size):
            print(f"WARN: output.weight shape inattendue {w_out.shape}")

        w_norm = self.load_tensor("output_norm.weight")
        if w_norm.shape != (self.config.dim,):
            w_norm = self.load_tensor("norm.weight")

        eos_id = getattr(self.tokenizer, "eos_id", 2)

        # =====================================================
        # Phase 1 : Prefill — toute la séquence du prompt
        # =====================================================
        t_prefill = time.time()
        valid_prompt = [t for t in tokens if 0 <= t < w_emb.shape[0]]
        x = w_emb[valid_prompt]  # [n_prompt, dim]
        n_prompt = len(valid_prompt)

        # kv_cache[l] = (k, v) de forme [seq_so_far, n_kv_heads, head_dim]
        kv_cache: List[Tuple[np.ndarray, np.ndarray]] = []

        _FE = _try_import_fragment_executor()
        if _FE is not None and self._loader is not None:
            # FragmentExecutor : une couche en RAM à la fois, libération explicite entre couches.
            # rope_cache partagé → pas de recalcul des fréquences RoPE à chaque couche.
            for l in range(self.config.n_layers):
                with _FE(self._loader, l, self.config, track_memory=track_memory) as ex:
                    x, new_k, new_v = ex.forward(x, self._rope_cache, None, None, start_pos=0)
                kv_cache.append((new_k, new_v))
            import gc; gc.collect()   # une seule fois après tout le prefill
        else:
            # Fallback : LlamaLayer (lazy-load via load_tensor du moteur)
            layers = [LlamaLayer(self, l) for l in range(self.config.n_layers)]
            for layer in layers:
                x, new_k, new_v = layer.forward(x, self.freqs_cis, None, None, start_pos=0)
                kv_cache.append((new_k, new_v))

        # Prédire le premier token depuis la sortie du dernier token du prompt
        x_last = rms_norm(x[-1:], w_norm, self.config.norm_eps)
        logits  = (x_last @ w_out).flatten()
        if temperature <= 0.0:
            first_token = int(np.argmax(logits))
        else:
            first_token = _sample_logits(logits, temperature, top_k, top_p)

        dt_prefill = time.time() - t_prefill
        print(f"Prefill ({n_prompt} tokens) en {dt_prefill:.2f}s")
        print(f"  Token 1 : '{self.tokenizer.decode([first_token])}' (id={first_token})")

        generated: List[int] = [first_token]
        start_pos = n_prompt  # position courante dans la séquence

        if first_token == eos_id or max_tokens <= 1:
            full_tokens = tokens + generated
            print(f"\n{'='*30}\nREPONSE :\n{self.tokenizer.decode(full_tokens)}\n{'='*30}\n")
            return full_tokens

        # Pré-allouer les buffers KV pour éviter np.concatenate à chaque token (O(n²) → O(1))
        _fk = kv_cache[0][0]
        _kv_h, _kv_d = _fk.shape[1], _fk.shape[2]
        _max_seq = n_prompt + max_tokens + 1
        kv_k_buf = [np.empty((_max_seq, _kv_h, _kv_d), dtype=np.float32)
                    for _ in range(self.config.n_layers)]
        kv_v_buf = [np.empty((_max_seq, _kv_h, _kv_d), dtype=np.float32)
                    for _ in range(self.config.n_layers)]
        for l_i, (ck, cv) in enumerate(kv_cache):
            kv_k_buf[l_i][:n_prompt] = ck
            kv_v_buf[l_i][:n_prompt] = cv

        # =====================================================
        # Phase 2 : Decode — un token à la fois avec KV cache
        # =====================================================
        for i in range(1, max_tokens):
            t0 = time.time()

            # Embedder uniquement le dernier token généré : [1, dim]
            last_id = generated[-1]
            if not (0 <= last_id < w_emb.shape[0]):
                break
            x = w_emb[[last_id]]  # [1, dim]

            # Passe dans toutes les couches — écriture directe dans les buffers pré-alloués
            if _FE is not None and self._loader is not None:
                for l_i in range(self.config.n_layers):
                    ck = kv_k_buf[l_i][:start_pos]
                    cv = kv_v_buf[l_i][:start_pos]
                    with _FE(self._loader, l_i, self.config, track_memory=track_memory) as ex:
                        x, new_k, new_v = ex.forward(x, self._rope_cache, ck, cv, start_pos=start_pos)
                    kv_k_buf[l_i][start_pos] = new_k[0]
                    kv_v_buf[l_i][start_pos] = new_v[0]
                import gc; gc.collect()   # une seule fois par token, pas par couche
            else:
                for l_i, layer in enumerate(layers):
                    ck = kv_k_buf[l_i][:start_pos]
                    cv = kv_v_buf[l_i][:start_pos]
                    x, new_k, new_v = layer.forward(x, self.freqs_cis, ck, cv, start_pos=start_pos)
                    kv_k_buf[l_i][start_pos] = new_k[0]
                    kv_v_buf[l_i][start_pos] = new_v[0]
            start_pos += 1

            # Logits depuis la sortie du token courant (x a déjà la forme [1, dim])
            x_last = rms_norm(x, w_norm, self.config.norm_eps)
            logits  = (x_last @ w_out).flatten()

            if temperature <= 0.0:
                next_token = int(np.argmax(logits))
            else:
                next_token = _sample_logits(logits, temperature, top_k, top_p)

            generated.append(next_token)
            word = self.tokenizer.decode([next_token])
            dt = time.time() - t0
            print(f"  Token {i+1} : '{word}' (id={next_token}) en {dt:.2f}s")

            if next_token == eos_id:
                break

        full_tokens = tokens + generated
        full_text = self.tokenizer.decode(full_tokens)
        print(f"\n{'='*30}\nREPONSE :\n{full_text}\n{'='*30}\n")
        return full_tokens


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pure Python Inference Engine")
    parser.add_argument("fragments_dir", help="Directory containing manifest.json and .dat fragments")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=5, help="Nombre de tokens a generer")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=0, help="Top-K (0 = desactive)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-P nucleus sampling")
    parser.add_argument("--verbose", action="store_true", help="Show fragment loading details")
    parser.add_argument("--no-cache", action="store_true", help="Desactiver le cache des poids (economise la RAM)")
    parser.add_argument("--track-memory", action="store_true",
                        help="Afficher l'empreinte memoire RSS par couche (necessite psutil)")
    args = parser.parse_args()

    engine = P2PInferenceEngine(args.fragments_dir, verbose=args.verbose, cache_weights=not args.no_cache)
    engine.generate(args.prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    track_memory=args.track_memory)
