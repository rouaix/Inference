"""
P2P Inference — Interface Gradio
Interface de gestion des modèles fragmentés, inférence et tests.
"""
# python app.py --fragments-dir models/tinyllama_q8_fragments_v2

import io
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# État global de l'application
# ============================================================

class AppState:
    def __init__(self):
        self.engine = None          # P2PInferenceEngine chargé
        self.fragments_dir = None   # Répertoire actif

state = AppState()


# ============================================================
# Utilitaires
# ============================================================

class StdoutCapture:
    """Capture sys.stdout dans un buffer."""
    def __init__(self):
        self._buf = io.StringIO()
        self._old = None

    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self._old

    def write(self, text):
        self._buf.write(text)

    def flush(self):
        pass

    def getvalue(self) -> str:
        return self._buf.getvalue()


def scan_fragment_dirs(base_dir: str) -> List[Dict]:
    """Retourne la liste des répertoires contenant un manifest.json."""
    results = []
    base = Path(base_dir) if base_dir else Path(".")
    if not base.exists():
        return results
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        manifest_path = item / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path) as f:
                m = json.load(f)
            model_name = m.get("model_name", item.name)
            n_frags = m.get("total_fragments", len(m.get("fragments", [])))
            chunk_mb = m.get("chunk_size", 0) / (1024 ** 2)
            total_bytes = sum(
                fp.stat().st_size for fp in item.glob("*.dat") if fp.is_file()
            )
            results.append({
                "path": str(item),
                "name": model_name,
                "fragments": n_frags,
                "size_mb": total_bytes / (1024 ** 2),
                "chunk_mb": chunk_mb,
            })
        except Exception as e:
            results.append({
                "path": str(item),
                "name": item.name,
                "fragments": "?",
                "size_mb": 0,
                "chunk_mb": 0,
                "error": str(e),
            })
    return results


def format_dirs_table(dirs: List[Dict]) -> str:
    if not dirs:
        return "_Aucun répertoire de fragments trouvé._"
    lines = [
        "| Nom | Fragments | Taille totale | Chunk | Chemin |",
        "|-----|-----------|---------------|-------|--------|",
    ]
    for d in dirs:
        lines.append(
            f"| **{d['name']}** | {d['fragments']} | {d['size_mb']:.1f} Mo"
            f" | {d['chunk_mb']:.0f} Mo | `{d['path']}` |"
        )
    return "\n".join(lines)


# ============================================================
# Onglet 1 — Modèle
# ============================================================

def load_model(fragments_dir: str, verbose: bool) -> Tuple[str, str]:
    """Charge un P2PInferenceEngine depuis un répertoire de fragments."""
    global state

    fragments_dir = fragments_dir.strip()
    if not fragments_dir or not Path(fragments_dir).exists():
        return "ERROR Répertoire invalide ou introuvable.", ""

    try:
        with StdoutCapture() as cap:
            from p2p_inference import P2PInferenceEngine
            state.engine = P2PInferenceEngine(fragments_dir, verbose=verbose)
            state.fragments_dir = fragments_dir

        cfg = state.engine.config
        info = f"""SUCCESS **Modèle chargé** depuis `{fragments_dir}`

| Paramètre | Valeur |
|-----------|--------|
| Dimensions | {cfg.dim} |
| Couches | {cfg.n_layers} |
| Têtes attention | {cfg.n_heads} (KV : {cfg.n_kv_heads}) |
| Vocabulaire | {cfg.vocab_size} |
| FFN dim | {cfg.hidden_dim} |
| RoPE base | {cfg.rope_freq_base} |
| Norm eps | {cfg.norm_eps} |"""
        return info, cap.getvalue()

    except Exception as e:
        tb = traceback.format_exc()
        return f"ERROR Erreur : {e}", tb


def scan_models(base_dir: str) -> str:
    dirs = scan_fragment_dirs(base_dir.strip() or ".")
    return format_dirs_table(dirs)


# ============================================================
# Onglet 2 — Fragmentation
# ============================================================

def run_fragmentation(gguf_path: str, output_dir: str, chunk_mb: float, progress=None):
    """Fragmente un fichier GGUF en morceaux de chunk_mb Mo."""
    gguf_path = gguf_path.strip()
    if not gguf_path or not Path(gguf_path).exists():
        yield "ERROR Fichier GGUF introuvable.", ""
        return

    output_dir = output_dir.strip()
    if not output_dir:
        output_dir = str(Path(gguf_path).parent / (Path(gguf_path).stem + "_fragments"))

    chunk_bytes = int(chunk_mb * 1024 * 1024)
    log_lines = []

    try:
        log_lines.append(f"INFO Fichier source : {gguf_path}")
        log_lines.append(f"INFO Répertoire de sortie : {output_dir}")
        log_lines.append(f"INFO Taille des chunks : {chunk_mb:.0f} Mo")
        yield "INFO Initialisation...", "\n".join(log_lines)

        from fragmenter import RealGGUFFragmenter
        frag = RealGGUFFragmenter(gguf_path, chunk_size=chunk_bytes)

        log_lines.append("INFO Lancement de la fragmentation...")
        yield "INFO Fragmentation en cours (peut prendre plusieurs minutes)...", "\n".join(log_lines)

        with StdoutCapture() as cap:
            frag.fragment(output_dir)

        log_lines.append(cap.getvalue())
        stats = frag.stats

        summary = (
            f"SUCCESS **Fragmentation terminée !**\n\n"
            f"- Fragments créés : **{stats['fragment_count']}**\n"
            f"- Volume total : **{stats['total_bytes'] / (1024**3):.3f} Go**\n"
            f"- Taille par chunk : **{chunk_mb:.0f} Mo**\n"
            f"- Répertoire : `{output_dir}`"
        )
        yield summary, "\n".join(log_lines)

    except Exception as e:
        tb = traceback.format_exc()
        log_lines.append(f"\nERROR ERREUR :\n{tb}")
        yield f"ERROR Erreur : {e}", "\n".join(log_lines)


# ============================================================
# Onglet 3 — Nettoyage
# ============================================================

def list_sets(base_dir: str):
    import gradio as gr
    dirs = scan_fragment_dirs(base_dir.strip() or ".")
    choices = [d["path"] for d in dirs]
    table = format_dirs_table(dirs)
    return gr.update(choices=choices, value=[]), table


def delete_selected(selected: List[str], confirmed: bool) -> str:
    if not selected:
        return "WARN Aucun répertoire sélectionné."
    if not confirmed:
        return "WARN Cochez la case de confirmation avant de supprimer."
    msgs = []
    for path in selected:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
            msgs.append(f"🗑️ Supprimé : `{path}`")
        else:
            msgs.append(f"WARN Introuvable : `{path}`")
    return "\n\n".join(msgs)


# ============================================================
# Templates de chat par famille de modèle
# ============================================================

CHAT_TEMPLATES = {
    "TinyLlama": {
        "label": "TinyLlama 1.1B Chat",
        "template": (
            "<|system|>\nYou are a helpful assistant.</s>\n"
            "<|user|>\n{prompt}</s>\n"
            "<|assistant|>\n"
        ),
        "eos_ids": [2],
        "bos": True,
    },
    "Llama 2": {
        "label": "Llama 2 / CodeLlama",
        "template": (
            "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST]"
        ),
        "eos_ids": [2],
        "bos": True,
    },
    "Llama 3": {
        "label": "Llama 3.x (1B / 3B / 8B / 70B / 405B)",
        "template": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant.<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            "{prompt}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        ),
        "eos_ids": [128009, 128001],  # <|eot_id|> et <|end_of_text|>
        "bos": False,  # déjà dans le template
    },
    "Mistral": {
        "label": "Mistral / Nemo / Large",
        "template": "[INST] {prompt} [/INST]",
        "eos_ids": [2],
        "bos": True,
    },
}

CHAT_FAMILY_CHOICES = [v["label"] for v in CHAT_TEMPLATES.values()]
_LABEL_TO_KEY = {v["label"]: k for k, v in CHAT_TEMPLATES.items()}


# ============================================================
# Onglet 4 — Chat / Inférence
# ============================================================


def run_chat(
    prompt: str,
    history: list,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    verbose: bool,
    model_family: str,
):
    """Génère une réponse token par token (streaming via yield).
    Utilise le prefill complet : toute la séquence est passée à chaque step.
    """
    global state

    if state.engine is None:
        new_hist = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "ERROR Aucun modèle chargé. Allez dans l'onglet **Modèle**."},
        ]
        yield new_hist, ""
        return

    if not prompt.strip():
        yield history, ""
        return

    from p2p_inference import LlamaLayer, rms_norm, _sample_logits

    engine = state.engine
    engine.verbose = verbose

    # Sélection du template selon la famille de modèle
    family_key = _LABEL_TO_KEY.get(model_family, "TinyLlama")
    tpl_cfg = CHAT_TEMPLATES[family_key]
    formatted_prompt = tpl_cfg["template"].format(prompt=prompt)

    # Encodage du prompt
    tokens = engine.tokenizer.encode(formatted_prompt)
    if tpl_cfg["bos"] and (not tokens or tokens[0] != 1):
        tokens = [1] + tokens

    new_hist = history + [{"role": "user", "content": prompt}]

    # Chargement des poids fixes (une seule fois)
    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.ndim == 2 and w_emb.shape[0] == engine.config.dim and w_emb.shape[1] == engine.config.vocab_size:
        w_emb = w_emb.T  # → [vocab, dim]

    w_out = engine.load_tensor("output.weight")
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (engine.config.dim,):
        w_norm_alt = engine.load_tensor("norm.weight")
        if w_norm_alt.shape == (engine.config.dim,):
            w_norm = w_norm_alt

    generated_tokens: List[int] = []
    generated_text = ""
    log_lines = [
        f"Famille : {model_family} | Prompt : {len(tokens)} token(s) | "
        f"Max : {max_tokens} | T={temperature} K={top_k} P={top_p}"
    ]
    # EOS : liste de tokens d'arrêt selon la famille (ex. Llama 3 en a deux)
    eos_ids = set(tpl_cfg["eos_ids"])
    # Ajouter l'eos du tokenizer si disponible
    tok_eos = getattr(engine.tokenizer, "eos_id", 2)
    eos_ids.add(tok_eos)

    for i in range(max_tokens):
        t0 = time.time()

        # Fix prefill : embed TOUTE la séquence (prompt + tokens générés)
        all_tokens = tokens + generated_tokens
        valid = [t for t in all_tokens if 0 <= t < w_emb.shape[0]]
        x = w_emb[valid]  # [seq_len, dim]

        for l in range(engine.config.n_layers):
            layer = LlamaLayer(engine, l)
            x, _, _ = layer.forward(x, engine.freqs_cis, None, None, start_pos=0)

        # Prédiction sur le dernier token de la séquence
        x_last = rms_norm(x[-1:], w_norm, engine.config.norm_eps)
        logits = (x_last @ w_out).flatten()

        next_token = _sample_logits(logits, temperature, top_k, top_p)

        if next_token in eos_ids:
            log_lines.append(f"Token {i+1}: <EOS> (id={next_token}) — arrêt.")
            break

        generated_tokens.append(next_token)
        generated_text = engine.tokenizer.decode(generated_tokens)

        dt = time.time() - t0
        word = engine.tokenizer.decode([next_token])
        log_lines.append(f"Token {i+1}: '{word}' (id={next_token}) en {dt:.2f}s")

        # Mise à jour streaming
        stream_hist = new_hist + [{"role": "assistant", "content": generated_text + "▌"}]
        yield stream_hist, "\n".join(log_lines)

    final_hist = new_hist + [{"role": "assistant", "content": generated_text or "_(réponse vide)_"}]
    yield final_hist, "\n".join(log_lines)


# ============================================================
# Onglet 5 — Tests
# ============================================================

def run_system_tests() -> str:
    """Vérifie que tous les composants sont fonctionnels."""
    global state
    results = []
    ok = True

    def check(label, fn):
        nonlocal ok
        try:
            fn()
            results.append(f"SUCCESS {label}")
        except Exception as e:
            results.append(f"ERROR {label} — {e}")
            ok = False

    def warn(label, fn):
        try:
            fn()
            results.append(f"SUCCESS {label}")
        except Exception as e:
            results.append(f"WARN {label} — {e} _(optionnel)_")

    check("Python & numpy", lambda: np.zeros(1))

    warn("Module `gguf`", lambda: __import__("gguf"))
    warn("Module `sentencepiece`", lambda: __import__("sentencepiece"))

    import gradio
    check(f"Gradio {gradio.__version__}", lambda: None)

    check("p2p_inference importable", lambda: __import__("p2p_inference"))
    check("fragmenter importable", lambda: __import__("fragmenter"))
    check("recombiner importable", lambda: __import__("recombiner"))

    def test_rope():
        from p2p_inference import precompute_freqs_cis, apply_rotary_emb
        freqs = precompute_freqs_cis(64, 128)
        assert freqs.shape == (128, 32), f"Shape inattendue : {freqs.shape}"
        xq = np.random.randn(4, 8, 64).astype(np.float32)
        xk = np.random.randn(4, 8, 64).astype(np.float32)
        xq_r, xk_r = apply_rotary_emb(xq, xk, freqs)
        assert xq_r.shape == xq.shape

    check("RoPE (precompute + apply)", test_rope)

    def test_softmax():
        from p2p_inference import softmax
        x = np.array([[1.0, 2.0, 3.0]])
        s = softmax(x)
        assert abs(s.sum() - 1.0) < 1e-5

    check("Softmax", test_softmax)

    def test_rms_norm():
        from p2p_inference import rms_norm
        x = np.ones((1, 64), dtype=np.float32)
        w = np.ones(64, dtype=np.float32)
        out = rms_norm(x, w, 1e-5)
        assert out.shape == x.shape

    check("RMS Norm", test_rms_norm)

    if state.engine is not None:
        cfg = state.engine.config
        check(f"Modèle chargé ({cfg.n_layers}L dim={cfg.dim})", lambda: None)

        def test_tokenizer():
            tok = state.engine.tokenizer
            ids = tok.encode("Hello world")
            decoded = tok.decode(ids)
            results.append(f"    → 'Hello world' → ids={ids[:5]}… → '{decoded[:30]}'")

        check("Tokenizer encode/decode", test_tokenizer)

        def test_embedding():
            w = state.engine.load_tensor("token_embd.weight")
            assert w.ndim == 2, f"Embedding doit être 2D, obtenu {w.ndim}D"
            results.append(f"    → shape={w.shape} dtype={w.dtype}")

        check("Chargement embedding", test_embedding)
    else:
        results.append("⏭️ Tests modèle : skipped (aucun modèle chargé)")

    banner = "SUCCESS **Tous les tests réussis !**" if ok else "WARN **Certains tests ont échoué.**"
    return banner + "\n\n" + "\n\n".join(results)


def run_quality_test(custom_prompt: str) -> str:
    """Génère quelques tokens sur des prompts de référence pour évaluer la qualité."""
    global state

    if state.engine is None:
        return "ERROR Aucun modèle chargé."

    from p2p_inference import LlamaLayer, rms_norm

    engine = state.engine
    test_cases = [
        ("Hello, my name is", "Texte libre"),
        ("1 + 1 =", "Arithmétique"),
        ("The capital of France is", "Connaissance"),
    ]
    if custom_prompt.strip():
        test_cases.append((custom_prompt.strip(), "Personnalisé"))

    w_emb = engine.load_tensor("token_embd.weight")
    if w_emb.ndim == 2 and w_emb.shape[0] == engine.config.dim and w_emb.shape[1] == engine.config.vocab_size:
        w_emb = w_emb.T

    w_out = engine.load_tensor("output.weight")
    w_norm = engine.load_tensor("output_norm.weight")
    if w_norm.shape != (engine.config.dim,):
        w_norm_alt = engine.load_tensor("norm.weight")
        if w_norm_alt.shape == (engine.config.dim,):
            w_norm = w_norm_alt

    # Nombre de couches limité pour la vitesse des tests de qualité
    n_layers_test = min(engine.config.n_layers, 4)
    eos_id = getattr(engine.tokenizer, "eos_id", 2)

    output_parts = []
    for prompt, label in test_cases:
        try:
            tokens = engine.tokenizer.encode(prompt)
            if not tokens or tokens[0] != 1:
                tokens = [1] + tokens

            generated = []

            for _ in range(15):
                # Prefill : toute la séquence
                all_tokens = tokens + generated
                valid = [t for t in all_tokens if 0 <= t < w_emb.shape[0]]
                x = w_emb[valid]
                for l in range(n_layers_test):
                    layer = LlamaLayer(engine, l)
                    x, _, _ = layer.forward(x, engine.freqs_cis, None, None, 0)
                x_last = rms_norm(x[-1:], w_norm, engine.config.norm_eps)
                logits = (x_last @ w_out).flatten()
                next_token = int(np.argmax(logits))
                if next_token == eos_id:
                    break
                generated.append(next_token)

            text = engine.tokenizer.decode(generated)
            output_parts.append(f"**{label}** — `{prompt}` → `{text}`")
        except Exception as e:
            output_parts.append(f"**{label}** — `{prompt}` → ERROR {e}")

    return "\n\n".join(output_parts)


# ============================================================
# Onglet 6 — Paramètres
# ============================================================

def apply_params(
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    verbose: bool,
    rope_base: float,
    norm_eps: float,
) -> str:
    global state

    msgs = ["SUCCESS Paramètres enregistrés."]

    if state.engine is not None:
        cfg = state.engine.config
        cfg.rope_freq_base = float(rope_base)
        cfg.norm_eps = float(norm_eps)
        state.engine.verbose = verbose

        # Recalcul RoPE
        from p2p_inference import precompute_freqs_cis
        state.engine.freqs_cis = precompute_freqs_cis(
            cfg.dim // cfg.n_heads,
            cfg.dim * 2,
            theta=float(rope_base),
        )
        msgs.append(f"🔄 RoPE recalculé (base={rope_base:.0f}).")
        msgs.append(f"🔧 Modèle mis à jour : {cfg.n_layers}L dim={cfg.dim}.")
    else:
        msgs.append("_(Aucun modèle chargé — les valeurs seront appliquées au prochain chargement)_")

    return "\n\n".join(msgs)


# ============================================================
# Construction de l'interface Gradio
# ============================================================

def build_app():
    import gradio as gr

    with gr.Blocks(title="P2P Inference UI") as demo:

        gr.Markdown("# 🌐 P2P Inference — Interface de gestion")
        gr.Markdown(
            "Chargez, fragmentez et interrogez des modèles IA distribués en peer-to-peer."
        )

        with gr.Tabs():

            # ─────────────────────────────────────────────
            # ONGLET 1 : MODÈLE
            # ─────────────────────────────────────────────
            with gr.Tab("MODELE Modèle"):
                gr.Markdown("## Charger un modèle fragmenté")

                with gr.Row():
                    with gr.Column(scale=3):
                        frag_dir_box = gr.Textbox(
                            label="Répertoire des fragments",
                            placeholder="Ex : ./tinyllama_fragments",
                            info="Répertoire contenant manifest.json et les .dat",
                        )
                        verbose_cb = gr.Checkbox(label="Verbose", value=False)
                        load_btn = gr.Button("🚀 Charger le modèle", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("**Modèles disponibles**")
                        scan_dir_box = gr.Textbox(label="Dossier à scanner", value=".", placeholder=".")
                        scan_btn = gr.Button("🔍 Scanner", size="sm")
                        scan_result = gr.Markdown("_Cliquez sur Scanner_")

                model_info = gr.Markdown("_Aucun modèle chargé_")
                load_log_box = gr.Textbox(
                    label="Logs", lines=8, interactive=False, elem_classes="mono"
                )

                load_btn.click(load_model, [frag_dir_box, verbose_cb], [model_info, load_log_box])
                scan_btn.click(scan_models, [scan_dir_box], [scan_result])

                gr.Markdown("""
---
## Modèles compatibles avec le moteur Python

Tous ces modèles sont de **famille LLaMA** (architecture identique : RoPE, RMSNorm, SwiGLU, GQA).
Les paramètres du modèle (couches, dimensions, RoPE…) sont lus automatiquement depuis le GGUF.
Seul le **chat template** dans l'onglet Chat doit être adapté selon le modèle chargé.

### Petits modèles (< 5 Go)

| Modèle | Taille Q4_K_M | Ajustements requis |
|--------|--------------|-------------------|
| **TinyLlama 1.1B Chat** _(actuel)_ | ~0.7 Go | Aucun — déjà configuré |
| **Llama 3.2 1B Instruct** | ~0.8 Go | Chat template Llama 3 + RoPE base = 500 000 |
| **Llama 3.2 3B Instruct** | ~2 Go | Chat template Llama 3 + RoPE base = 500 000 |
| **Mistral 7B Instruct v0.3** | ~4.1 Go | Chat template `[INST]...[/INST]` |

### Modèles moyens (5–50 Go)

| Modèle | Taille Q4_K_M | Ajustements requis |
|--------|--------------|-------------------|
| **Llama 3.1 8B Instruct** | ~4.7 Go | Chat template Llama 3 + RoPE base = 500 000 + vocab = 128 256 |
| **CodeLlama 13B Instruct** | ~7.4 Go | Chat template code + RoPE base = 1 000 000 |
| **Mistral Nemo 12B** | ~7 Go | Chat template Mistral + Norm eps = 1e-6 |
| **Llama 3.1 70B Instruct** | ~40 Go | Chat template Llama 3 + RoPE base = 500 000 — RAM : ~48 Go |

### Grands modèles (> 50 Go) — cibles P2P

| Modèle | Taille Q4_K_M | Ajustements requis |
|--------|--------------|-------------------|
| **Mistral Large 2 123B** | ~70 Go | Chat template Mistral |
| **Llama 3.1 405B** | ~229 Go | Chat template Llama 3 + RoPE base = 500 000 |
| **Mistral Large 3 675B** _(cible)_ | ~338 Go | Chat template Mistral |

### Chat templates de référence

**TinyLlama** _(actuel)_
```
<|system|>
You are a helpful assistant.</s>
<|user|>
{prompt}</s>
<|assistant|>
```

**Llama 3.x**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**Mistral / Nemo / Large**
```
[INST] {prompt} [/INST]
```

> **Modèles non compatibles avec le moteur Python** (bridge uniquement) :
> Phi-3, Gemma, Qwen, Mixtral 8×7B — ces architectures ne sont pas LLaMA et nécessiteraient une réécriture du moteur.
""")


            # ─────────────────────────────────────────────
            # ONGLET 2 : FRAGMENTATION
            # ─────────────────────────────────────────────
            with gr.Tab("✂️ Fragmentation"):
                gr.Markdown("## Fragmenter un fichier GGUF")

                with gr.Row():
                    with gr.Column():
                        gguf_box = gr.Textbox(
                            label="Fichier GGUF source",
                            placeholder="Ex : C:/models/tinyllama.gguf",
                        )
                        out_dir_box = gr.Textbox(
                            label="Répertoire de sortie",
                            placeholder="Laissez vide → même dossier que le .gguf",
                        )
                        chunk_slider = gr.Slider(
                            1, 100, value=10, step=1,
                            label="Taille des fragments (Mo)",
                            info="10 Mo = recommandé pour P2P",
                        )
                        frag_btn = gr.Button("⚡ Lancer la fragmentation", variant="primary")

                    with gr.Column():
                        gr.Markdown("""
**Formats GGUF supportés**

| Format | Bits/poids | Qualité |
|--------|-----------|---------|
| Q4_K_M | 4.5 | ★★★★☆ |
| Q8_0   | 8   | ★★★★☆ |
| F16    | 16  | ★★★★★ |
| F32    | 32  | ★★★★★ |

Le fichier `manifest.json` généré indexe tous les fragments
et permet la reconstruction ou l'inférence distribuée.
""")

                frag_status = gr.Markdown("_Prêt_")
                frag_log_box = gr.Textbox(
                    label="Logs de fragmentation", lines=12,
                    interactive=False, elem_classes="mono"
                )

                frag_btn.click(
                    run_fragmentation,
                    [gguf_box, out_dir_box, chunk_slider],
                    [frag_status, frag_log_box],
                )

            # ─────────────────────────────────────────────
            # ONGLET 3 : NETTOYAGE
            # ─────────────────────────────────────────────
            with gr.Tab("🗑️ Nettoyage"):
                gr.Markdown("## Supprimer des modèles et leurs fragments")

                with gr.Row():
                    clean_base_box = gr.Textbox(label="Dossier à scanner", value=".", placeholder=".")
                    list_btn = gr.Button("🔍 Lister", variant="secondary")

                clean_table = gr.Markdown("_Cliquez sur Lister_")
                dirs_check = gr.CheckboxGroup(
                    choices=[],
                    label="Répertoires à supprimer",
                    info="Cochez les répertoires de fragments à effacer définitivement",
                )

                with gr.Row():
                    confirm_cb = gr.Checkbox(
                        label="WARN Je confirme la suppression définitive (irréversible)",
                        value=False,
                    )
                    delete_btn = gr.Button("🗑️ Supprimer la sélection", variant="stop")

                delete_result = gr.Markdown()

                list_btn.click(list_sets, [clean_base_box], [dirs_check, clean_table])
                delete_btn.click(delete_selected, [dirs_check, confirm_cb], [delete_result])

            # ─────────────────────────────────────────────
            # ONGLET 4 : CHAT
            # ─────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.Markdown("## Dialogue avec le modèle")

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=460,
                        )
                        with gr.Row():
                            prompt_box = gr.Textbox(
                                placeholder="Votre message…",
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            send_btn = gr.Button("➤ Envoyer", variant="primary", scale=1)
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Effacer", size="sm")
                            stop_btn = gr.Button("⏹️ Arrêter", size="sm", variant="stop")

                    with gr.Column(scale=1):
                        gr.Markdown("**Famille de modèle**")
                        chat_family = gr.Dropdown(
                            choices=CHAT_FAMILY_CHOICES,
                            value=CHAT_FAMILY_CHOICES[0],
                            label="Template de chat",
                            info="Doit correspondre au modèle chargé",
                        )
                        gr.Markdown("**Paramètres de génération**")
                        chat_max_tokens = gr.Slider(1, 500, value=50, step=1, label="Max tokens")
                        chat_temp = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Température")
                        chat_topk = gr.Slider(0, 200, value=40, step=1, label="Top-K")
                        chat_topp = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P")
                        chat_verbose = gr.Checkbox(label="Verbose logs", value=False)
                        gr.Markdown("""
---
**Réglages typiques**

| Usage | Temp | Top-K | Top-P |
|-------|------|-------|-------|
| Factuel | 0.3–0.5 | 20–40 | 0.90 |
| Conversation | 0.7–1.0 | 40 | 0.95 |
| Créatif | 1.0–1.5 | 100+ | 0.98 |
| Déterministe | 0.0 | 1 | — |
""")

                inference_log_box = gr.Textbox(
                    label="Logs d'inférence", lines=5,
                    interactive=False, elem_classes="mono"
                )

                gen_inputs = [prompt_box, chatbot, chat_max_tokens, chat_temp, chat_topk, chat_topp, chat_verbose, chat_family]
                gen_outputs = [chatbot, inference_log_box]

                gen_event = send_btn.click(run_chat, gen_inputs, gen_outputs)
                prompt_box.submit(run_chat, gen_inputs, gen_outputs)
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, inference_log_box])
                stop_btn.click(fn=None, cancels=[gen_event])

            # ─────────────────────────────────────────────
            # ONGLET 5 : TESTS
            # ─────────────────────────────────────────────
            with gr.Tab("🧪 Tests"):
                gr.Markdown("## Vérification du bon fonctionnement")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Tests système")
                        gr.Markdown(
                            "Vérifie les imports, RoPE, tokenizer, chargement de fragments, etc."
                        )
                        sys_test_btn = gr.Button("▶️ Lancer les tests système", variant="primary")
                        sys_test_out = gr.Markdown("_Cliquez pour lancer_")

                    with gr.Column():
                        gr.Markdown("### Tests de qualité")
                        gr.Markdown(
                            "Génère quelques tokens sur des prompts de référence. "
                            "Requiert un modèle chargé."
                        )
                        quality_prompt_box = gr.Textbox(
                            label="Prompt personnalisé (optionnel)",
                            placeholder="Ex : What is machine learning?",
                        )
                        quality_btn = gr.Button("▶️ Tester la qualité", variant="secondary")
                        quality_out = gr.Markdown("_Chargez un modèle puis cliquez_")

                sys_test_btn.click(run_system_tests, outputs=[sys_test_out])
                quality_btn.click(run_quality_test, [quality_prompt_box], [quality_out])

            # ─────────────────────────────────────────────
            # ONGLET 6 : PARAMÈTRES
            # ─────────────────────────────────────────────
            with gr.Tab("PARAMETRES Paramètres"):
                gr.Markdown("## Ajustement des variables d'inférence")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Génération par défaut")
                        p_max_tokens = gr.Slider(1, 2000, value=50, step=1, label="Max tokens")
                        p_temp = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Température (0 = greedy)")
                        p_topk = gr.Slider(0, 500, value=40, step=1, label="Top-K (0 = désactivé)")
                        p_topp = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-P (nucleus)")
                        p_verbose = gr.Checkbox(label="Verbose (afficher les fragments chargés)", value=False)

                    with gr.Column():
                        gr.Markdown("### Configuration du modèle _(avancé)_")
                        gr.Markdown(
                            "⚠️ Modifie la configuration du modèle actuellement chargé. "
                            "Ces valeurs sont normalement lues automatiquement depuis le GGUF. "
                            "Ne les modifier que si le chargement produit des résultats incorrects."
                        )
                        p_rope_base = gr.Number(
                            value=10000.0,
                            label="RoPE freq base",
                            info="Base de fréquence des embeddings positionnels rotatifs (défaut : 10000)",
                        )
                        gr.Markdown("""
**RoPE (Rotary Position Embedding)** encode la position de chaque token dans la séquence en faisant
"tourner" les vecteurs d'attention. La base contrôle la vitesse de rotation :
- `10000` → LLaMA 1 / TinyLlama (contexte ~2 048 tokens)
- `500000` → Llama 3 (contexte long)

Plus la base est élevée, mieux le modèle gère les longues séquences.
""")
                        p_norm_eps = gr.Number(
                            value=1e-5,
                            label="Norm epsilon",
                            info="Epsilon pour RMSNorm (défaut : 1e-5)",
                        )
                        gr.Markdown("""
**RMSNorm** normalise les activations entre chaque couche via `x / sqrt(mean(x²) + ε)`.
L'epsilon évite une division par zéro quand les activations sont proches de 0.
- `1e-5` → LLaMA / TinyLlama
- `1e-6` → Mistral

Modifier cette valeur sans connaître celle du modèle dégrade les sorties.
""")
                        gr.Markdown("""
### Référence des quantifications

La quantification réduit la précision des poids du modèle pour économiser de la mémoire et accélérer l'inférence sur CPU.

| Format | Bits/poids | Qualité | Vitesse CPU |
|--------|-----------|---------|-------------|
| F32    | 32        | ██████  | ░░░░░       |
| F16    | 16        | █████░  | ██░░░       |
| Q8_0   | 8         | █████░  | ███░░       |
| Q6_K   | 6.5       | ████░░  | ████░       |
| Q4_K_M | 4.5      | ████░░  | █████       |
| Q2_K   | 2.6       | ██░░░░  | █████       |

**Pourquoi les formats compressés sont plus rapides ?**
Le goulot d'étranglement sur CPU est la bande passante mémoire, pas le calcul.
Q4 lit 8× moins de données que F32 → 8× moins de lectures RAM.

**Taille pour Mistral Large 3 (675B)**

| Format | Taille disque | RAM requise |
|--------|--------------|-------------|
| F32    | ~2 700 Go    | ~2 700 Go   |
| Q8_0   | ~675 Go      | ~675 Go     |
| Q4_K_M | ~338 Go     | ~338 Go     |
| Q2_K   | ~175 Go      | ~175 Go     |

En P2P avec Q4_K_M, chaque nœud stockant 10 Mo contribue à porter collectivement les 338 Go.
""")

                apply_btn = gr.Button("APPLIQUER Appliquer", variant="primary")
                params_out = gr.Markdown()

                apply_btn.click(
                    apply_params,
                    [p_max_tokens, p_temp, p_topk, p_topp, p_verbose, p_rope_base, p_norm_eps],
                    [params_out],
                )

    return demo


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == "__main__":
    import argparse
    import gradio as gr

    parser = argparse.ArgumentParser(description="P2P Inference — Interface Gradio")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut : 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (défaut : 7860)")
    parser.add_argument("--share", action="store_true", help="Créer un lien public Gradio")
    parser.add_argument(
        "--fragments-dir",
        default=None,
        help="Charger automatiquement ce répertoire de fragments au démarrage",
    )
    args = parser.parse_args()

    # Chargement automatique si spécifié
    if args.fragments_dir:
        print(f"Chargement automatique : {args.fragments_dir}")
        info, log = load_model(args.fragments_dir, verbose=False)
        print(info)
        if log:
            print(log)

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=gr.themes.Soft(),
        css=".mono textarea { font-family: 'Courier New', monospace; font-size: 12px; }",
    )
