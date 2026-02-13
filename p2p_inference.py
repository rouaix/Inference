# python app.py --fragments-dir models/tinyllama_q8_fragments_v2

import argparse
import json
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
        meta = manifest.get("metadata", {})

        # Helper to safely get int/float
        def get_val(key, default, cast=int):
            val = meta.get(key, default)
            if isinstance(val, list) and len(val) > 0: val = val[0]
            try:
                return cast(val)
            except:
                return default

        # Fix #3 : lire n_heads avant n_kv_heads pour que le fallback soit correct
        n_heads = get_val("llama.attention.head_count", 32)
        n_kv_heads = get_val("llama.attention.head_count_kv", n_heads)  # fallback = n_heads, pas 32

        return ModelConfig(
            dim=get_val("llama.embedding_length", 4096),
            hidden_dim=get_val("llama.feed_forward_length", 11008),
            n_layers=get_val("llms.count" if "llms.count" in meta else "llama.block_count", 32),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=get_val("llama.vocab_size", 32000),
            norm_eps=get_val("llama.attention.layer_norm_rms_epsilon", 1e-5, float),
            rope_freq_base=get_val("llama.rope.freq_base", 10000.0, float)
        )

try:
    import sentencepiece as spm
except ImportError:
    spm = None

class Tokenizer:
    def __init__(self, model_path: str):
        if not spm:
             raise ImportError("sentencepiece not installed")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

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
            print("⚠️ No vocabulary found in manifest! Using dummy.")
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
    xq_out = np.stack([xq_out_c.real, xq_out_c.imag], axis=-1).flatten().reshape(xq.shape)
    xk_out = np.stack([xk_out_c.real, xk_out_c.imag], axis=-1).flatten().reshape(xk.shape)

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
        def proj(inp, w, out_dim):
            if w.ndim == 2 and w.shape[0] == out_dim and w.shape[1] != out_dim:
                return inp @ w.T   # poids en [out, in] → transposer
            return inp @ w         # poids en [in, out] → direct

        # 1. Attention
        w_norm = self.engine.load_tensor(f"{self.pfx}.attn_norm.weight")
        xn = rms_norm(x, w_norm, self.cfg.norm_eps)

        wq = self.engine.load_tensor(f"{self.pfx}.attn_q.weight")
        wk = self.engine.load_tensor(f"{self.pfx}.attn_k.weight")
        wv = self.engine.load_tensor(f"{self.pfx}.attn_v.weight")
        wo = self.engine.load_tensor(f"{self.pfx}.attn_output.weight")

        xq = proj(xn, wq, self.cfg.dim)          # [seq, dim]
        xk = proj(xn, wk, kv_dim)                 # [seq, kv_dim]
        xv = proj(xn, wv, kv_dim)                 # [seq, kv_dim]

        xq = xq.reshape(seq_len, self.cfg.n_heads,    head_dim)
        xk = xk.reshape(seq_len, self.cfg.n_kv_heads, head_dim)
        xv = xv.reshape(seq_len, self.cfg.n_kv_heads, head_dim)

        # RoPE
        current_freqs = freqs_cis[start_pos : start_pos + seq_len]
        xq, xk = apply_rotary_emb(xq, xk, current_freqs)

        keys   = xk
        values = xv

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

        # Masque causal (actif dès seq_len > 1, c'est-à-dire pendant le prefill)
        if seq_len > 1:
            mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
            scores = scores + mask

        probs  = softmax(scores)                                          # [n_heads, seq, seq]
        output = np.matmul(probs, values)                                 # [n_heads, seq, head_dim]
        output = output.transpose(1, 0, 2).reshape(seq_len, self.cfg.dim) # [seq, dim]

        h = x + proj(output, wo, self.cfg.dim)

        # 2. FFN SwiGLU
        w_ffn_norm = self.engine.load_tensor(f"{self.pfx}.ffn_norm.weight")
        xn = rms_norm(h, w_ffn_norm, self.cfg.norm_eps)

        w_gate = self.engine.load_tensor(f"{self.pfx}.ffn_gate.weight")
        w_up   = self.engine.load_tensor(f"{self.pfx}.ffn_up.weight")
        w_down = self.engine.load_tensor(f"{self.pfx}.ffn_down.weight")

        # SwiGLU : silu(gate) * up
        gate   = proj(xn, w_gate, self.cfg.hidden_dim)  # [seq, hidden]
        up     = proj(xn, w_up,   self.cfg.hidden_dim)  # [seq, hidden]
        hidden = swiglu(gate) * up                       # [seq, hidden]
        out    = proj(hidden, w_down, self.cfg.dim)      # [seq, dim]

        return h + out, keys, values

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
    logits = logits.flatten().astype(np.float64)
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
    def __init__(self, fragments_dir: str, verbose: bool = False):
        self.fragments_dir = Path(fragments_dir)
        self.verbose = verbose

        # Load Manifest
        manifest_path = self.fragments_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("Manifest not found")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.config = ModelConfig.from_manifest(self.manifest)

        # Tokenizer setup — chercher dans le dossier de fragments, son parent, ou le répertoire courant
        tokenizer_path = self.fragments_dir / "tokenizer.model"
        if not tokenizer_path.exists():
            tokenizer_path = self.fragments_dir.parent / "tokenizer.model"
        if not tokenizer_path.exists():
            tokenizer_path = Path("tokenizer.model")

        if tokenizer_path.exists() and spm:
             print(f"DEBUG: Using SentencePiece tokenizer from {tokenizer_path}")
             self.tokenizer = Tokenizer(str(tokenizer_path))
        else:
             print("DEBUG: Using SimpleTokenizer (fallback)")
             self.tokenizer = SimpleTokenizer(self.manifest)

        print(f"Loaded config: {self.config}")

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

        # Precompute RoPE
        self.freqs_cis = precompute_freqs_cis(self.config.dim // self.config.n_heads, self.config.dim * 2)

    def load_tensor(self, tensor_name: str) -> np.ndarray:
        fragments = self.fragments_map.get(tensor_name)
        if not fragments:
             # print(f"⚠️ Tensor missing: {tensor_name}")
             # Return random for stability
             return np.random.normal(0, 0.01, size=(64, 64)).astype(np.float32)

        if self.verbose:
            print(f"📂 [P2P] Loading '{tensor_name}' from {len(fragments)} fragments")
            # print(f"   └── Files: {[f['fragment_id'] + '.dat' for f in fragments]}")

        # Reassemble data from shards
        full_data = bytearray()
        for frag in fragments:
             path = self.fragments_dir / f"{frag['fragment_id']}.dat"
             if self.verbose:
                 print(f"    └── Reading fragment: {path.name}")

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
        elif "Q8_0" in frag["tensor_type"]:
            # Q8_0 Dequantization
            # Block size: 32
            # Structure:
            # - delta (float16)
            # - 32 x int8
            # Total bytes per block: 2 + 32 = 34 bytes

            block_size = 32
            block_bytes = 34

            # GGUF tensor shape is usually [n_cols, n_rows] (transposed) or flattened
            # We trust 'shape' from manifest.
            # num_elements = prod(shape)
            # num_blocks = num_elements // block_size

            # Data is sequence of blocks.
            # We can use numpy structured array or stride tricks.

            # Define block dtype
            # d (float16), qs (32 * int8)
            dt = np.dtype([('d', '<f2'), ('qs', 'i1', (32,))])

            if len(data) % 34 != 0:
                 if self.verbose: print(f"❌ Error: {tensor_name} data size mismatch")
                 # Fallback?
                 res = np.zeros(shape, dtype=np.float32)
            else:
                # Read blocks
                blocks = np.frombuffer(data, dtype=dt)

                # Dequantize: x = d * qs
                # d is [num_blocks], qs is [num_blocks, 32]
                # expand d to [num_blocks, 1]
                d = blocks['d'].astype(np.float32)[:, None]
                qs = blocks['qs'].astype(np.float32)

                decoded = (d * qs).flatten()

                # FIX FONDAMENTAL — convention GGUF Q8_0 :
                # Les données physiques sont stockées en [out_dim, in_dim] (ligne = une unité de sortie)
                # La shape logique dans le manifest est [in_dim, out_dim] (transposée du physique)
                # → reshape vers la shape physique [out_dim, in_dim], puis transposer pour avoir [in_dim, out_dim]
                if len(shape) == 2:
                    out_dim = shape[-1]  # 2ème dim logique = out_dim = nb lignes physiques
                    in_dim  = shape[0]   # 1ère dim logique = in_dim  = nb éléments/ligne physique
                    res = decoded.reshape([out_dim, in_dim]).T.astype(np.float32)
                else:
                    res = decoded.reshape(shape).astype(np.float32)
        else:
            # Other quantized formats: Return Zeros/Random of correct shape
            # We read the data so I/O is simulated.
            res = np.zeros(shape, dtype=np.float32)

        # Debug Stats
        if self.verbose:
            print(f"    🔎 Stats: Mean={np.mean(res):.4f} Std={np.std(res):.4f} Range=[{np.min(res):.4f}, {np.max(res):.4f}]")

        return res

    def generate(self, prompt: str, max_tokens: int = 5,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        """
        Génère max_tokens tokens à partir du prompt.

        Fix #1 — Prefill : à chaque step, on intègre TOUTE la séquence
        (prompt + tokens générés) avant de prédire le prochain token.
        C'est O(n²) en longueur mais mathématiquement correct.
        """
        tokens = self.tokenizer.encode(prompt)
        if not (len(tokens) > 0 and tokens[0] == 1):
            tokens = [1] + tokens

        print(f"Prompt : {len(tokens)} token(s) — {tokens}")

        # === Chargement une seule fois ===
        w_emb = self.load_tensor("token_embd.weight")
        if w_emb.ndim == 2 and w_emb.shape[0] == self.config.dim and w_emb.shape[1] == self.config.vocab_size:
            w_emb = w_emb.T  # → [vocab, dim]
        if w_emb.shape[0] != self.config.vocab_size:
            print(f"WARN: embedding shape {w_emb.shape}, attendu [{self.config.vocab_size}, {self.config.dim}]")

        w_out = self.load_tensor("output.weight")

        w_norm = self.load_tensor("output_norm.weight")
        if w_norm.shape != (self.config.dim,):
            w_norm = self.load_tensor("norm.weight")

        eos_id = getattr(self.tokenizer, "eos_id", 2)
        generated: List[int] = []

        print("Debut de la generation...")
        for i in range(max_tokens):
            t0 = time.time()

            # Fix #1 : embed TOUTE la séquence (prompt + tokens générés)
            all_tokens = tokens + generated
            valid = [t for t in all_tokens if 0 <= t < w_emb.shape[0]]
            x = w_emb[valid]  # [seq_len, dim]

            # Passe dans toutes les couches (start_pos=0 : RoPE depuis position 0)
            for l in range(self.config.n_layers):
                layer = LlamaLayer(self, l)
                x, _, _ = layer.forward(x, self.freqs_cis, None, None, start_pos=0)

            # Prendre uniquement la sortie du DERNIER token pour prédire le suivant
            x_last = x[-1:]                                      # [1, dim]
            x_last = rms_norm(x_last, w_norm, self.config.norm_eps)
            logits  = (x_last @ w_out).flatten()                 # [vocab]

            # Sampling
            if temperature <= 0.0:
                next_token = int(np.argmax(logits))
            else:
                next_token = _sample_logits(logits, temperature, top_k, top_p)

            generated.append(next_token)
            word = self.tokenizer.decode([next_token])
            dt = time.time() - t0
            print(f"  Token {i+1}: '{word}' (id={next_token}) en {dt:.2f}s")

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
    args = parser.parse_args()

    engine = P2PInferenceEngine(args.fragments_dir, verbose=args.verbose)
    engine.generate(args.prompt, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
