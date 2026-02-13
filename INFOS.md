# INFOS.md

Documentation du projet **rouaix.com/inference** — système d'inférence P2P distribué.

---

## Objectif du projet

Système d'inférence distribué pair-à-pair permettant de faire tourner de grands modèles de langage (LLM) sans serveur centralisé. Chaque nœud du réseau stocke seulement **10 Mo** du modèle. Actuellement testé avec **TinyLlama-1.1B-Chat Q8_0**, conçu pour supporter **Mistral Large 3 (675B)**.

---

## Environnement de développement

Toutes les commandes doivent utiliser le virtualenv Python du projet :

```bash
# Installation initiale
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Exécuter un script
.venv\Scripts\python.exe <script.py>

# Ou activer le virtualenv une fois pour toute la session
.venv\Scripts\activate.bat
```

Les modèles et fragments sont dans `models/`. Jeu de fragments actif : `models/tinyllama_q8_fragments_v2/`.

---

## Commandes courantes

```bash
# Inférence — moteur Python pur (lent, pour débogage)
.venv\Scripts\python.exe p2p_inference.py models/tinyllama_q8_fragments_v2 --prompt "Hello" --max-tokens 20 --temperature 0.7

# Inférence — bridge llama.cpp (préféré en production)
.venv\Scripts\python.exe p2p_bridge.py models/tinyllama_q8_fragments_v2 --prompt "Hello"

# Fragmenter un nouveau modèle GGUF
.venv\Scripts\python.exe fragmenter.py models/model.gguf --output models/model_fragments

# Lancer l'interface Gradio
launch_ui.bat   # ou : .venv\Scripts\python.exe app.py

# Tester le chargeur de fragments local
.venv\Scripts\python.exe distribution\local.py models/tinyllama_q8_fragments_v2 --tensor blk.0.attn_q.weight

# Tests unitaires rapides
.venv\Scripts\python.exe tests_debug/validate_inference.py models/tinyllama_q8_fragments_v2 --units-only

# Validation complète contre la référence llama.cpp
.venv\Scripts\python.exe tests_debug/validate_inference.py models/tinyllama_q8_fragments_v2 --gguf models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

---

## Architecture

### Flux de données

```
Fichier GGUF → fragmenter.py → fragments *.dat + manifest.json
                                          ↓
                             distribution/ (couche de chargement)
                             ├── local.py   ✅ implémenté
                             ├── reseau.py  🚧 stub documenté
                             └── p2p.py     🚧 stub documenté
                                          ↓
                              P2PInferenceEngine (p2p_inference.py)
                              OU P2PBridge (p2p_bridge.py)
                                          ↓
                                    app.py (Interface Gradio)
```

### Composants principaux

**`distribution/`**
Couche d'abstraction pour le chargement des fragments. `BaseFragmentLoader` définit l'interface commune :
- `load_raw(fragment_id) → bytes` — lit les octets bruts d'un fragment
- `load_tensor(tensor_name) → np.ndarray` — reconstitue et dequantise un tenseur

`LocalFragmentLoader` est le seul backend fonctionnel. `ReseauFragmentLoader` et `P2PFragmentLoader` sont des stubs documentés qui lèvent `NotImplementedError`. La logique de dequantisation Q8_0 (avec la correction du layout transposé) est dans `LocalFragmentLoader._dequantize_q8_0`.

**`p2p_inference.py`**
Implémentation NumPy pure du transformer LLaMA. Utile pour l'apprentissage et le débogage. Lent (~14s/token, sans cache KV, sans batching). Contient : `P2PInferenceEngine`, `LlamaLayer`, `ModelConfig`, utilitaires de sampling. Possède encore son propre `load_tensor` (logique identique à `LocalFragmentLoader`, pas encore fusionnée).

**`fragmenter.py`**
Découpe n'importe quel fichier GGUF en morceaux de 10 Mo (`.dat`). Génère un `manifest.json` indexant chaque tenseur vers ses fragments. Gère les types Q8_0 et F32.

**`p2p_bridge.py`**
Chemin de production : reconstruit le GGUF en mémoire depuis les fragments, puis lance l'inférence via `llama-cpp-python`. Résultats numériquement identiques à llama.cpp.

**`app.py`**
Interface Gradio 6.x (6 onglets). Importe `rms_norm`, `LlamaLayer`, `_sample_logits` directement depuis `p2p_inference.py`. L'onglet **Modèle** inclut un sélecteur de mode de distribution (Local / Réseau / P2P) — les modes non-local affichent un message d'attente. `find_default_fragments_dir()` détecte automatiquement un dossier de fragments dans `models/`, `.` ou `..` pour pré-remplir le chemin au démarrage.

**`recombiner.py`**
Inverse de `fragmenter.py` : reconstruit un fichier GGUF complet à partir des fragments et du manifest. Utilise la bibliothèque `gguf` pour réécrire les tenseurs. Sert à la vérification d'intégrité.

**`tests_debug/validate_inference.py`**
Suite de validation. Tests unitaires pour RMSNorm, softmax, SwiGLU, RoPE. Compare les logits contre la référence llama.cpp.

**`simulation/`**
Scripts de preuve de concept Phase 1 (`fragmenter_v2.py`, `simulator_v2.py`). Simulent la fragmentation MoE et le comportement d'un réseau P2P sans vrai modèle. Hors du chemin de production.

---

## Format du manifest de fragments

Chaque tenseur dans `manifest.json` est décrit ainsi :

```json
{
  "tensor_name": "blk.0.attn_q.weight",
  "tensor_type": "Q8_0",
  "shape": [2048, 2048],
  "dtype": "uint8",
  "fragment_id": "tinyllama_L0_attn_q_S0_abc123",
  "shard_index": 0,
  "total_shards": 1
}
```

---

## Points d'implémentation critiques

### Dequantisation Q8_0 — Layout physique transposé

**C'est le détail le plus important et le moins évident du projet.**

GGUF stocke les tenseurs Q8_0 avec un layout physique transposé :
- Shape logique dans les métadonnées : `[in_dim, out_dim]`
- Layout physique des données : `[out_dim, in_dim]` (une ligne par unité de sortie)

Correction appliquée dans `load_tensor()` :

```python
if len(shape) == 2:
    out_dim = shape[-1]  # 2e dim logique = nb de lignes physiques
    in_dim  = shape[0]   # 1re dim logique = éléments par ligne physique
    res = decoded.reshape([out_dim, in_dim]).T.astype(np.float32)  # → [in_dim, out_dim]
```

Après cette correction, toutes les matrices de poids sont en format `[in, out]` et utilisées directement via `x @ w`. **Ne pas retransposer** dans `proj()` ou ailleurs.

Ce bug était la cause du boucle de token ">>" (prédiction systématique du token 5099 quel que soit le contexte). Avant la correction, la corrélation avec llama.cpp était de ~0.009.

### Résolution du tokenizer

`P2PInferenceEngine.__init__()` cherche `tokenizer.model` dans l'ordre suivant :
1. `fragments_dir/tokenizer.model`
2. `fragments_dir.parent/tokenizer.model` ← nécessaire car le tokenizer est dans `models/tokenizer.model`
3. `./tokenizer.model` (dossier courant)

### Template de chat TinyLlama

Pour des sorties cohérentes avec TinyLlama-1.1B-Chat :

```
<|system|>
You are a helpful assistant.</s>
<|user|>
{prompt}</s>
<|assistant|>
```

Sans ce template, le modèle produit des tokens arbitraires.

### Prefill — traitement de la séquence complète

Dans `generate()`, chaque étape autoregressive retraite **toute la séquence** (`prompt + tokens générés`) depuis la position 0. Il n'y a pas de cache KV — c'est O(n²) mais mathématiquement correct. Ne pas optimiser en ne passant que le dernier token.

### Projection des poids dans LlamaLayer

```python
def proj(inp, w, out_dim):
    if w.ndim == 2 and w.shape[0] == out_dim and w.shape[1] != out_dim:
        return inp @ w.T   # fallback : poids en [out, in]
    return inp @ w         # standard : poids en [in, out]
```

Après la correction Q8_0, tous les poids sont en `[in, out]`. Le premier branchement ne devrait jamais s'activer pour les tenseurs Q8_0.

### GQA (Grouped Query Attention)

TinyLlama utilise `n_heads=32`, `n_kv_heads=4`. Les têtes KV sont répétées 8× avant l'attention :

```python
n_rep = cfg.n_heads // cfg.n_kv_heads  # = 8
keys   = np.repeat(xk, n_rep, axis=1)
values = np.repeat(xv, n_rep, axis=1)
```

---

## Configuration du modèle TinyLlama Q8_0

| Paramètre | Valeur |
|-----------|--------|
| dim | 2048 |
| hidden_dim | 5632 |
| n_layers | 22 |
| n_heads | 32 |
| n_kv_heads | 4 |
| vocab_size | 32000 |
| norm_eps | 1e-5 |
| rope_freq_base | 10000.0 |

---

## Feuille de route

| Phase | Statut | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Terminé | Simulation PoC (modèle MoE, réseau P2P, tolérance aux pannes) |
| Phase 2 | 🚧 En cours | Fragmenteur GGUF réel (`fragmenter.py` + `recombiner.py` fonctionnels sur TinyLlama) |
| Phase 3 | ⏳ | Inférence distribuée réelle (matmul multi-processus, vérifié vs llama.cpp) |
| Phase 4 | ⏳ | Réseau P2P réel (libp2p, WebRTC, DHT) |
| Phase 5 | ⏳ | Application utilisateur (Tauri desktop, PWA mobile) |
| Phase 6 | ⏳ | Passage à l'échelle, incentives, support multi-modèles |

**Modèle cible : Mistral Large 3 (675B, MoE, ~46 000 fragments × 10 Mo)**

---

## Problèmes connus

- **Vitesse du moteur Python** : ~14s/token (pas de cache KV, pas de batching, NumPy pur). Utiliser `p2p_bridge.py` en production.
- **Gradio 6.x** : `theme=` et `css=` doivent être passés à `.launch()`, pas à `gr.Blocks()`. Pas de `type="messages"` dans `gr.Chatbot()`. Pour ajouter un nouveau backend de distribution, mettre à jour le dict `DISTRIBUTION_MODES` et le dispatch dans `load_model()` dans `app.py`.
- **Scripts de débogage** (`debug*.py`, `test_fix.py`) à la racine du projet sont temporaires — ne font pas partie du code de production.
