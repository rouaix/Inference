# Proposition : Fragment Executor — moteur d'inférence fragment-aware

> Rédigé le 2026-02-14

## Problème

Deux approches actuelles, toutes deux inadaptées à l'objectif du projet :

| Approche | Problème |
|---|---|
| `p2p_inference.py` (Python/NumPy pur) | ~14s/token, pas de KV cache, overhead Python |
| `p2p_bridge.py` (llama.cpp) | Charge **tout** le modèle en RAM — contredit l'objectif de fragmentation |

L'objectif du projet est de n'avoir qu'une fraction du modèle en mémoire à l'instant T. Il faut donc un moteur capable de traiter **un fragment à la fois**.

---

## Solution 1 — Gain immédiat : KV cache

Le coût actuel est **O(n²)** par token car toute la séquence est recalculée depuis zéro à chaque étape (voir `generate()` dans `p2p_inference.py`). Un KV cache réduit ça à **O(n)**.

- Gain estimé : ×5 à ×20 selon la longueur de séquence
- Sans changer le format de fragments ni l'architecture réseau
- Effort : 2–3 heures

---

## Solution 2 — Architecture cible : Fragment Executor

### Principe

Un moteur léger traite **un fragment à la fois**, libère la mémoire, et passe l'activation au fragment suivant.

```
input_activation
      ↓
[Fragment L0 executor]  ← charge blk.0.*.dat, calcule, libère
      ↓ activation
[Fragment L1 executor]  ← charge blk.1.*.dat, calcule, libère
      ↓ activation
     ...
      ↓
   logits
```

Chaque executor :
1. Lit les tenseurs d'**un seul fragment** (`.dat`)
2. Exécute le calcul de la couche (attention + FFN)
3. Retourne l'activation de sortie
4. **Libère toute la mémoire allouée**

Ce modèle est nativement distribuable : chaque nœud du réseau P2P héberge ses fragments et exécute les requêtes qui lui arrivent.

---

## Options d'implémentation

### A) ggml standalone *(recommandé long terme)*

`ggml` est la librairie C sous-jacente à llama.cpp. Elle peut être utilisée **indépendamment**, avec un chargement de tenseurs custom depuis nos fichiers `.dat`.

```c
// fragment_executor.c (concept)
// 1. Parser le .dat → tenseurs bruts (Q8_0)
// 2. Créer un contexte ggml temporaire (mémoire bornée)
// 3. Calculer la couche : RMSNorm → QKV proj → attention → FFN
// 4. Libérer le contexte ggml
// 5. Retourner l'activation de sortie (float32)
```

**Avantages :**
- Même vitesse que llama.cpp
- BLAS (OpenBLAS/MKL) intégré automatiquement
- Support GPU (CUDA, Metal, Vulkan) disponible
- Format de tenseurs Q8_0 déjà supporté nativement

**Inconvénient :** Nécessite d'écrire du C/C++ et de compiler.

---

### B) Rust + `candle` (HuggingFace) *(recommandé long terme alternatif)*

[`candle`](https://github.com/huggingface/candle) est un framework ML minimaliste en Rust, conçu pour être embarqué dans des applications.

**Avantages :**
- Sécurité mémoire garantie par Rust
- Compile en binaire standalone sans dépendances
- Support CPU/GPU/Metal
- Chargement de tenseurs custom facilement implémentable

---

### C) Python + `numba` JIT *(recommandé moyen terme)*

```python
import numba

@numba.njit(parallel=True)
def dequantize_q8_0(data: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # Dequant Q8_0 compilé en C par numba
    ...

@numba.njit(parallel=True)
def matmul_fragment(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    # GEMM parallélisé
    ...
```

**Avantages :**
- Aucun changement d'architecture
- Reste en Python, facilement intégrable dans `p2p_inference.py`
- Gain estimé : ×10 à ×20 sur les matmuls

**Inconvénient :** Moins rapide que ggml/Rust, mais bien plus rapide que NumPy pur.

---

## Projets existants à étudier

Ces projets résolvent le même problème et peuvent servir d'inspiration ou de base :

| Projet | Approche | Lien |
|---|---|---|
| **Petals** | Chaque nœud héberge quelques couches du modèle ; les activations transitent via le réseau | https://github.com/bigscience-workshop/petals |
| **distributed-llama** | Fork de llama.cpp pour inférence distribuée multi-machines | https://github.com/b4rtaz/distributed-llama |
| **exo** | Inférence collaborative sur appareils du quotidien, pipeline par couches | https://github.com/exo-labs/exo |

Petals est architecturalement le plus proche de ce que ce projet vise (Phase 4 du roadmap).

---

## Feuille de route recommandée

| Phase | Action | Gain | Effort |
|---|---|---|---|
| **Court terme** | Ajouter KV cache dans `p2p_inference.py` | ×5–20 perf | 2–3h |
| **Moyen terme** | Réécrire les kernels critiques avec `numba` | ×10–20 perf | 1–2 jours |
| **Long terme** | Fragment executor C/Rust avec `ggml` ou `candle` | Parité llama.cpp | Plusieurs semaines |

Le KV cache est le premier chantier : il ne change rien à l'architecture des fragments, améliore drastiquement les performances, et prépare le terrain pour la version distribuée (les KV caches pourront éventuellement être transmis entre nœuds).

---

## Impact sur l'architecture distribuée

Le Fragment Executor est nativement adapté au réseau P2P :

```
Nœud A (fragments L0–L5)      Nœud B (fragments L6–L11)    Nœud C (fragments L12–L21)
       ↓                               ↓                              ↓
  exécute couches 0–5      reçoit activation, exécute 6–11   exécute 12–21 → logits
       └──────────────────────────────────────────────────────────────┘
                         activations transmises via réseau P2P
```

Chaque nœud ne stocke que ses fragments (≈ 10 MB × N couches) et ne charge en RAM que les tenseurs nécessaires à l'instant T.
