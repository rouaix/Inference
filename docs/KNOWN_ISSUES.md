# Problèmes Connus et Limitations

## Mistral 7B - Incompatibilité d'Architecture

**Statut** : Non résolu (bloquant pour l'utilisation de Mistral 7B)

### Description

Le modèle Mistral 7B a une architecture différente de Magistral, ce qui cause des incompatibilités dans le code actuel :

- **Magistral** : `dim=5120`, `hidden_dim=32768`, `vocab_size=131072` (vocab = 4 × hidden)
- **Mistral 7B** : `dim=4096`, `hidden_dim=14336`, `vocab_size=32768` (vocab ≠ multiple de hidden)

### Symptômes

- Erreur de dimension lors de la multiplication matricielle
- Sorties corrompues (`� پ▌`)
- Incompatibilité dans `rms_norm` et les couches FFN

### Impact

- Mistral 7B ne peut pas être utilisé actuellement
- Seul Magistral est fonctionnel

### Solution Temporaire

Utiliser **Magistral** qui fonctionne parfaitement en attendant une refonte pour supporter les deux architectures.

### Solution Long Terme

Refactoriser le code pour :
1. Détecter automatiquement l'architecture du modèle
2. Adapter les dimensions en conséquence
3. Supporter plusieurs configurations (dim, hidden_dim, vocab_size)

### Priorité

**Moyenne** - Le projet est fonctionnel avec Magistral, mais l'ajout de Mistral 7B serait un plus.

---

## Autres Problèmes Connus

### 1. Fragmentation des Grands Modèles

**Statut** : Limitation connue

Le processus de fragmentation peut être long et consommer beaucoup de mémoire pour les grands modèles (>10 GB).

**Impact** : 
- Fragmentation lente pour les très grands modèles
- Nécessite une machine puissante (32+ GB RAM recommandé)

**Solution** : 
- Fragmenter sur une machine puissante
- Utiliser `screen` ou `nohup` pour les longs processus
- Envisager une fragmentation par lots pour les très grands modèles

---

## Modèles Testés et Fonctionnels

✅ **Magistral-Small-2509-Q4_K_M** - Pleinement fonctionnel
❌ **Mistral-7B-Instruct-v0.3-Q4_K_M** - Problème d'architecture (voir ci-dessus)
✅ **Devstral-Small-2-24B-Instruct-2512-Q4_K_M** - Fonctionnel

---

## Recommandations

1. **Pour la production** : Utiliser Magistral qui est pleinement fonctionnel
2. **Pour le développement** : Corriger l'architecture avant d'ajouter Mistral 7B
3. **Pour les tests** : Tous les tests passent avec Magistral

---

*Dernière mise à jour : 2024-02-17*
