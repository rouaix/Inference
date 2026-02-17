# tests_debug/ — Scripts de test et diagnostic

Scripts de validation, benchmark et diagnostic pour le moteur d'inférence P2P.
Modèles actifs : **Magistral-Small-2509-Q4_K_M** ✅ | **Mistral-7B-Instruct-v0.3-Q4_K_M** ❌ (architecture en cours)

---

## Utilitaires

| Fichier | Rôle |
|---------|------|
| `setup_path.py` | Ajoute la racine projet au `sys.path` — à importer en premier dans chaque script |
| `inspect_gguf.py` | Inspecte la structure d'un fichier GGUF (tenseurs, métadonnées) |

---

## Benchmarks

```bash
.venv\Scripts\python.exe tests_debug/_bench_deq.py          # Temps de dequantisation par tenseur
.venv\Scripts\python.exe tests_debug/_bench_matmul.py        # Comparaison GEMM / GEMV / einsum
.venv\Scripts\python.exe tests_debug/test_benchmarks.py      # Benchmarks sérialisation
.venv\Scripts\python.exe tests_debug/test_load_performance.py # Tests de scalabilité
```

---

## Tests composants individuels

Vérifient les briques de base du moteur (pas de modèle requis pour la plupart) :

```bash
.venv\Scripts\python.exe tests_debug/test_rmsnorm.py         # RMSNorm → variance ≈ 1.0
.venv\Scripts\python.exe tests_debug/test_rope.py            # RoPE reshape correct
.venv\Scripts\python.exe tests_debug/test_rope_freqs.py      # Fréquences RoPE
.venv\Scripts\python.exe tests_debug/test_attention.py       # Mécanisme d'attention
.venv\Scripts\python.exe tests_debug/test_prefill.py         # Importance du contexte
.venv\Scripts\python.exe tests_debug/test_numerical_precision.py  # Précision numérique
```

---

## Tests moteur principal

```bash
# Validation rapide sans modèle (kernels numba uniquement)
.venv\Scripts\python.exe tests_debug/test_fragment_executor.py --units-only
.venv\Scripts\python.exe tests_debug/validate_inference.py models/Magistral-Small-2509-Q4_K_M_fragments --units-only

# Validation complète avec modèle
.venv\Scripts\python.exe tests_debug/test_fragment_executor.py models/Magistral-Small-2509-Q4_K_M_fragments
.venv\Scripts\python.exe tests_debug/validate_inference.py models/Magistral-Small-2509-Q4_K_M_fragments

# KV cache
.venv\Scripts\python.exe tests_debug/test_kv_cache.py
.venv\Scripts\python.exe tests_debug/test_kv_cache_optimization.py
```

---

## Tests multi-architecture

```bash
.venv\Scripts\python.exe tests_debug/test_architecture_detection.py      # Détection Magistral vs Mistral 7B
.venv\Scripts\python.exe tests_debug/test_arch_simple.py                 # Détection via gguf direct
.venv\Scripts\python.exe tests_debug/test_architecture_aware_tensors.py  # Tenseurs par architecture
.venv\Scripts\python.exe tests_debug/test_mistral_7b_architecture.py     # Spécifique Mistral 7B
.venv\Scripts\python.exe tests_debug/test_multi_architecture_serialization.py
```

---

## Tests sérialisation / inférence locale

```bash
.venv\Scripts\python.exe tests_debug/test_serialization.py    # Sérialisation (référencé dans deploy)
.venv\Scripts\python.exe tests_debug/test_compression.py      # Compression zstd
.venv\Scripts\python.exe tests_debug/test_roundtrip.py        # Round-trip sérialisation
.venv\Scripts\python.exe tests_debug/test_local_inference.py  # Inférence locale complète
.venv\Scripts\python.exe tests_debug/test_real_prompt.py      # Prompt réel avec le modèle actif
```

---

## Tests intégration et production

```bash
.venv\Scripts\python.exe tests_debug/test_new_manifest.py         # Chargement manifest (référencé dans deploy)
.venv\Scripts\python.exe tests_debug/test_production_validation.py # Validation complète production
.venv\Scripts\python.exe tests_debug/test_integration.py           # Intégration fragmentation
.venv\Scripts\python.exe tests_debug/test_robustness_extended.py   # Robustesse réseau/données corrompues
.venv\Scripts\python.exe tests_debug/test_advanced_features.py     # Auto-détection et métriques
.venv\Scripts\python.exe tests_debug/test_async_pipeline.py        # Pipeline asynchrone
.venv\Scripts\python.exe tests_debug/test_model_comprehensive.py   # Robustesse modèle
.venv\Scripts\python.exe tests_debug/test_model_inference.py       # Inférence multi-modèles
```

---

## Tests UI / modèles / tokenizers

```bash
.venv\Scripts\python.exe tests_debug/test_all_models_tokenizers.py  # Tokenizers de tous les modèles
.venv\Scripts\python.exe tests_debug/test_model_scanning.py         # Scan des modèles disponibles
.venv\Scripts\python.exe tests_debug/test_model_dropdown.py         # Dropdown Gradio
.venv\Scripts\python.exe tests_debug/test_gradio_interface.py       # Interface Gradio (sans serveur)
.venv\Scripts\python.exe tests_debug/test_download_script.py        # Script de téléchargement Mistral 7B
```

---

## Diagnostics et comparaisons (Python vs llama.cpp)

```bash
# Diagnostic complet
.venv\Scripts\python.exe tests_debug/diagnostic.py models/Magistral-Small-2509-Q4_K_M_fragments --prompt "Hello" --max-tokens 3

# Comparaisons détaillées
.venv\Scripts\python.exe tests_debug/detailed_comparison.py models/Magistral-Small-2509-Q4_K_M_fragments
.venv\Scripts\python.exe tests_debug/layer_by_layer_comparison.py models/Magistral-Small-2509-Q4_K_M_fragments
.venv\Scripts\python.exe tests_debug/deep_diagnostic.py models/Magistral-Small-2509-Q4_K_M_fragments
.venv\Scripts\python.exe tests_debug/compare_logits.py
.venv\Scripts\python.exe tests_debug/compare_weights.py
.venv\Scripts\python.exe tests_debug/verify_weight_loading.py
```

> Note : les tests de comparaison logits Python vs llama.cpp échouent sur le système complet (divergence accumulée sur 40 couches). Les composants individuels sont tous corrects. Voir `docs/KNOWN_ISSUES.md`.
