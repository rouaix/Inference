# Tests et Diagnostics - Débogage Moteur Python

> **Test en cours :** `Magistral-Small-2509-Q4_K_M.gguf`

Ce dossier contient tous les scripts de test, diagnostic et inspection créés pendant le débogage du moteur d'inférence Python P2P.

---

## 📁 Structure

### 📊 Documentation
- **DEBUG_ANALYSIS.md** - Analyse complète du débogage (bugs, composants vérifiés, hypothèses)
- **TEST_SCRIPTS_GUIDE.md** - Guide de référence pour tous les scripts de test
- **NEXT_STEPS.md** - Prochaines étapes possibles pour résoudre le bug restant

### 🧪 Tests de Composants (8 scripts)
- `test_rmsnorm.py` - Vérifier RMSNorm (variance unitaire)
- `test_rope.py` - Vérifier RoPE reshape
- `test_rope_freqs.py` - Vérifier fréquences RoPE
- `test_rope_bug.py` - Tester `.flatten()` dans RoPE
- `test_attention.py` - Vérifier mécanisme d'attention
- `test_mask.py` - Vérifier masque causal
- `test_proj_bug.py` - Vérifier fonction proj()
- `test_output_weight.py` - Vérifier orientation output.weight

### 🔍 Tests de Formes et Poids (4 scripts)
- `inspect_shapes.py` - Afficher formes de tenseurs
- `debug_shapes.py` - Vérifier formes layer 0
- `debug_tensors.py` - Debug détaillé des tenseurs
- `check_weights.py` - Vérifier statistiques des poids

### 🎯 Tests de Prefill et Contexte (2 scripts)
- `test_prefill.py` - Montrer importance du contexte
- `test_prefill_rope.py` - Vérifier positions RoPE

### 📈 Tests de Logits et Comparaisons (5 scripts)
- `test_forward.py` - Test forward pass BOS
- `test_layer_consistency.py` - Vérifier cohérence layers
- `compare_logits.py` - Comparer avec llama.cpp
- `final_logits_test.py` - Test logits finaux
- `diagnostic.py` - Comparaison complète Python vs llama.cpp
- `deep_diagnostic.py` - Analyse layer-by-layer (créé mais non utilisé)

---

## 🚀 Utilisation Rapide

### Exécuter tous les tests de composants
```bash
cd tests_debug
python test_rmsnorm.py
python test_rope.py
python test_attention.py
python test_mask.py
```

### Test de diagnostic complet
```bash
cd tests_debug
python diagnostic.py ../tinyllama_q8_fragments_v2 --prompt "Hello" --max-tokens 3
```

### Test final des logits
```bash
cd tests_debug
python final_logits_test.py
```

---

## 📊 Résultats

| Catégorie | Tests | ✅ PASS | ❌ FAIL |
|-----------|-------|---------|---------|
| Composants individuels | 8 | 8 | 0 |
| Formes et poids | 4 | 4 | 0 |
| Prefill et contexte | 2 | 2 | 0 |
| Logits et comparaisons | 5 | 1 | 4 |
| **TOTAL** | **19** | **15** | **4** |

---

## 🎯 Conclusion

**Composants individuels**: Tous fonctionnent ✅
**Système complet**: Produit des logits incorrects ❌

Le bug restant est subtil et nécessite un débogage plus approfondi. Voir `NEXT_STEPS.md` pour les options.

---

## 📝 Bugs Corrigés

1. ✅ Embedding transpose (GGUF `[dim, vocab]` → `[vocab, dim]`)
2. ✅ Return statement manquant
3. ✅ SwiGLU critique (`x² * sigmoid(x)` → `x * sigmoid(x)`)
4. ✅ Token de départ (`tokens[0]` → `tokens[-1]`)

---

## 💡 Recommandation

Pour la production, utilisez `p2p_bridge.py` (pont hybride) qui fonctionne parfaitement avec llama.cpp.

Ces tests ont une grande valeur éducative et documentent le processus de débogage complet.
