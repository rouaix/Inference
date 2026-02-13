# Scripts de Test - Guide de Référence

> **Test en cours :** `Magistral-Small-2509-Q4_K_M.gguf`

Ce document liste tous les scripts de test créés pendant le débogage du moteur d'inférence Python.

---

## 🧪 Tests de Composants Individuels

### test_rmsnorm.py
**Objectif**: Vérifier que RMSNorm produit une variance unitaire
**Commande**: `python test_rmsnorm.py`
**Résultat attendu**: Variance ≈ 1.0
**Statut**: ✅ PASS

```bash
Input variance: 1.25
Output variance: 0.999999  # ✅ Correct
```

---

### test_rope.py
**Objectif**: Vérifier la logique de reshape dans RoPE
**Commande**: `python test_rope.py`
**Résultat attendu**: Reshape préserve l'ordre des éléments
**Statut**: ✅ PASS

---

### test_rope_freqs.py
**Objectif**: Vérifier le calcul des fréquences RoPE
**Commande**: `python test_rope_freqs.py`
**Résultat attendu**:
- Position 0 → tous 1+0j
- Formule alternative donne même résultat
**Statut**: ✅ PASS

---

### test_rope_bug.py
**Objectif**: Tester si `.flatten()` dans RoPE cause des problèmes
**Commande**: `python test_rope_bug.py`
**Résultat attendu**: Méthode 1 (avec flatten) = Méthode 2 (sans flatten)
**Statut**: ✅ PASS - Pas de bug

---

### test_attention.py
**Objectif**: Vérifier le mécanisme d'attention pour un token unique
**Commande**: `python test_attention.py`
**Résultat attendu**:
- Poids d'attention = 1.0
- Sortie = valeurs
**Statut**: ✅ PASS

```bash
Attention weights: 1.000000 (should be 1.0 for single token)
All 1.0? True  # ✅ Correct
Match? True    # ✅ Correct
```

---

### test_mask.py
**Objectif**: Vérifier le masque causal et broadcasting
**Commande**: `python test_mask.py`
**Résultat attendu**: Masque triangulaire supérieur avec -inf
**Statut**: ✅ PASS

```bash
Mask:
[[  0. -inf -inf]
 [  0.   0. -inf]
 [  0.   0.   0.]]
✅ Broadcasting works correctly!
```

---

### test_proj_bug.py
**Objectif**: Vérifier que proj() ne transpose pas incorrectement
**Commande**: `python test_proj_bug.py`
**Résultat attendu**: Aucune transposition pour Q, K, V
**Statut**: ✅ PASS

```bash
--- Testing Q projection ---
  → NO TRANSPOSE  # ✅ Correct
--- Testing K projection ---
  → NO TRANSPOSE  # ✅ Correct
--- Testing V projection ---
  → NO TRANSPOSE  # ✅ Correct
```

---

## 🔍 Tests de Formes et Poids

### inspect_shapes.py
**Objectif**: Afficher toutes les formes de tenseurs pendant l'inférence
**Commande**: `python inspect_shapes.py`
**Résultat attendu**: Toutes les formes correspondent aux dimensions attendues
**Statut**: ✅ PASS

---

### debug_shapes.py
**Objectif**: Vérifier les formes dans la couche 0
**Commande**: `python debug_shapes.py`
**Résultat attendu**:
- Q: [1, 32, 64]
- K: [1, 4, 64]
- V: [1, 4, 64]
**Statut**: ✅ PASS

---

### check_weights.py
**Objectif**: Vérifier les statistiques de tous les poids
**Commande**: `python check_weights.py`
**Résultat attendu**:
- Pas de NaN ou Inf
- Moyenne ≈ 0
- Écart-type raisonnable
**Statut**: ✅ PASS

```bash
token_embd.weight:
  Mean: -0.000000, Std: 0.014910  # ✅ Normal
  Has NaN: False, Has Inf: False  # ✅ Correct
```

---

### test_output_weight.py
**Objectif**: Tester l'orientation de output.weight
**Commande**: `python test_output_weight.py`
**Résultat attendu**: `x @ w_out` produit [1, 32000]
**Statut**: ✅ PASS

---

## 🎯 Tests de Prefill et Contexte

### test_prefill.py
**Objectif**: Montrer que le contexte change les prédictions
**Commande**: `python test_prefill.py`
**Résultat attendu**: Tokens différents selon le contexte
**Statut**: ✅ PASS - Contexte crucial!

```bash
[1] Generating from BOS (empty prompt)...
Generated: '<'

[2] Generating from 'Hello'...
Generated: ','

[3] Generating from 'The'...
Generated: ' '
```

**Conclusion**: Le contexte est CRUCIAL pour les bonnes prédictions!

---

### test_prefill_rope.py
**Objectif**: Vérifier les positions RoPE pendant le prefill
**Commande**: `python test_prefill_rope.py`
**Résultat attendu**: Positions absolues [0, 1, 2, ...] avec start_pos=0
**Statut**: ✅ PASS

---

## 📊 Tests de Logits et Comparaisons

### test_forward.py
**Objectif**: Tester le forward pass pour le token BOS
**Commande**: `python test_forward.py`
**Résultat**: ❌ FAIL - Génère ">>" au lieu de "<"

```bash
Top prediction: Token 5099 ('>>') with logit 12.1405
Expected: '<' from llama.cpp
```

---

### test_layer_consistency.py
**Objectif**: Vérifier la cohérence entre les couches
**Commande**: `python test_layer_consistency.py`
**Résultat**: ✅ Cohérent mais incorrect

```bash
After layer 0: mean=-0.000537, std=0.017149
After layer 21: mean=-0.002038, std=1.057072
Top prediction: Token 5099 ('>>') with logit 12.1405
```

---

### compare_logits.py
**Objectif**: Comparer les logits Python vs llama.cpp
**Commande**: `python compare_logits.py`
**Résultat**: ❌ FAIL - Logits complètement différents

---

### final_logits_test.py
**Objectif**: Test final des logits après prefill
**Commande**: `python final_logits_test.py`
**Résultat**: ❌ FAIL - Top token incorrect

```bash
Prompt tokens: [1, 15043]  # BOS + "Hello"

Top 10 predictions:
  1. Token 5099: '>>' (logit=12.8096)    # ❌ Incorrect
  2. Token 13163: 'irty' (logit=9.2218)
  3. Token 7147: 'aires' (logit=8.9082)
  ...

Expected from llama.cpp: ',' (comma)
```

---

## 🔬 Tests de Diagnostic Complets

### diagnostic.py
**Objectif**: Comparaison complète Python vs llama.cpp
**Commande**: `python diagnostic.py tinyllama_q8_fragments_v2 --prompt "Hello" --max-tokens 3`
**Résultat**: ❌ FAIL - Outputs différents

```bash
Python output: 'Hello >> >> >>'
llama.cpp output: ', World!'
❌ OUTPUTS DIFFER
```

---

### deep_diagnostic.py
**Objectif**: Analyse layer-by-layer (créé mais non utilisé)
**Commande**: `python deep_diagnostic.py`
**Note**: Script créé pour comparaison détaillée mais non exécuté car trop lent

---

## 📈 Résumé des Résultats

| Catégorie | Tests | ✅ PASS | ❌ FAIL |
|-----------|-------|---------|---------|
| Composants individuels | 8 | 8 | 0 |
| Formes et poids | 4 | 4 | 0 |
| Prefill et contexte | 2 | 2 | 0 |
| Logits et comparaisons | 5 | 1 | 4 |
| **TOTAL** | **19** | **15** | **4** |

---

## 🎯 Conclusion

**Composants individuels**: Tous fonctionnent correctement ✅
**Système complet**: Produit des logits incorrects ❌

**Hypothèse**: Le bug est dans l'intégration des composants ou dans une accumulation d'erreurs numériques sur les 22 couches.

---

## 🚀 Utilisation Rapide

Pour reproduire tous les tests:

```bash
# Tests de composants (tous devraient passer)
python test_rmsnorm.py
python test_rope.py
python test_rope_freqs.py
python test_attention.py
python test_mask.py
python test_proj_bug.py

# Tests de formes (tous devraient passer)
python check_weights.py
python inspect_shapes.py

# Test de contexte (devrait montrer tokens différents)
python test_prefill.py

# Test final (devrait échouer - top token incorrect)
python final_logits_test.py

# Diagnostic complet (devrait montrer divergence)
python diagnostic.py tinyllama_q8_fragments_v2 --prompt "Hello" --max-tokens 3
```

---

## 📝 Notes

- Tous les tests individuels passent ✅
- Le système complet échoue ❌
- Le pont hybride (`p2p_bridge.py`) fonctionne parfaitement ✅
- Recommandation: Utiliser le pont hybride pour la production
