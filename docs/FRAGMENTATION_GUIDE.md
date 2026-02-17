# Guide de Fragmentation des Modèles GGUF

Ce guide explique comment fragmenter de nouveaux modèles GGUF pour une utilisation avec le système d'inférence distribuée.

## Modèles Actuellement Disponibles

### Modèles Déjà Fragmentés
- **Magistral-Small-2509-Q4_K_M** (40 couches, 5120 dimensions)
  - Dossier: `models/Magistral-Small-2509-Q4_K_M_fragments/`
  - Fragments: 1590 fichiers .dat
  - Manifest: `manifest.json` complet
  - Statut: ✅ Prêt pour la production

### Modèles Disponibles (Non Fragmentés)
- Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf (14.3 GB)
- Magistral-Small-2509-Q4_K_M.gguf (14.3 GB) - Original
- Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf (8.2 GB)
- Mistral-7B-Instruct-v0.3-Q4_K_M.gguf (4.1 GB)

## Processus de Fragmentation

### Prérequis
- Python 3.11+
- Bibliothèques: `gguf`, `numpy`
- Espace disque: 2-3x la taille du modèle original
- Mémoire: Suffisante pour charger le modèle entier

### Étapes de Fragmentation

1. **Préparer le modèle GGUF**
   ```bash
   # Placer le fichier GGUF dans le dossier models/
   cp votre_modele.Q4_K_M.gguf models/
   ```

2. **Exécuter le fragmenter**
   ```bash
   python fragments/fragmenter.py models/votre_modele.Q4_K_M.gguf --output models/votre_modele_fragments
   ```

3. **Vérifier les fragments générés**
   ```bash
   ls -lh models/votre_modele_fragments/ | head -10
   # Devrait montrer des fichiers .dat de ~10 Mo chacun
   ```

4. **Générer le manifest**
   ```bash
   python fragments/generate_manifest_for_fragments.py models/votre_modele_fragments/
   ```

5. **Extraire le tokenizer**
   ```bash
   python fragments/generate_tokenizer_model.py models/votre_modele.Q4_K_M.gguf models/votre_modele_fragments/
   ```

### Paramètres Importants

- **Taille des chunks**: 10 Mo (définie dans `fragmenter.py`)
- **Format des fragments**: `modele_L{layer}_component_S{shard}_checksum.dat`
- **Header GGUF**: Sauvegardé séparément dans `gguf_header.dat`

## Optimisation pour les Grands Modèles

Pour les modèles > 10 GB, considérer:

1. **Fragmentation par lots**: Traiter couche par couche
2. **Utilisation de memmap**: Modifier le fragmenter pour utiliser `numpy.memmap`
3. **Machine puissante**: 32+ GB RAM, SSD rapide
4. **Timeout augmenté**: Les grands modèles peuvent prendre des heures

## Structure des Fragments

Chaque fragment contient:
- Données binaires du tenseur (quantifié)
- Taille: ~10 Mo (dernier fragment peut être plus petit)
- Nom: `modele_L{layer}_{component}_S{shard}_{checksum}.dat`

Exemple: `Magistral-Small-2509-Q4_K_M_L0_attn_q_S0_f94c5c4c7f49e625.dat`
- L0: Couche 0
- attn_q: Tenseur d'attention Q
- S0: Shard 0
- f94c5c4c7f49e625: Checksum (premiers 16 chars de SHA256)

## Manifest JSON

Le fichier `manifest.json` contient:
- Métadonnées du modèle (architecture, configuration)
- Liste complète des fragments avec:
  - ID unique
  - Type de fragment (embedding, attention, etc.)
  - Couche et composant
  - Forme du tenseur
  - Taille et checksum
  - Offset dans le fichier original

## Tests Post-Fragmentation

1. **Test de chargement**
   ```bash
   python tests_debug/test_new_manifest.py
   ```

2. **Test d'inférence simple**
   ```bash
   python tests_debug/test_fragment_executor.py
   ```

3. **Validation des tenseurs**
   ```bash
   python tests_debug/test_tensor_validation.py
   ```

## Dépannage

### Problèmes Courants

1. **Timeout lors de la fragmentation**
   - Solution: Augmenter le timeout ou fragmenter sur une machine plus puissante
   - Alternative: Utiliser `screen` ou `nohup` pour les longs processus

2. **Manque de mémoire**
   - Solution: Utiliser une machine avec plus de RAM
   - Alternative: Modifier le fragmenter pour utiliser memmap

3. **Fichier GGUF corrompu**
   - Solution: Vérifier l'intégrité avec SHA256
   - Re-télécharger si nécessaire

### Vérification de l'Intégrité

```bash
# Calculer le hash SHA256 d'un fichier
sha256sum models/votre_modele.gguf

# Ou avec Python
python -c "import hashlib; print(hashlib.sha256(open('models/votre_modele.gguf','rb').read()).hexdigest())"
```

## Bonnes Pratiques

1. **Sauvegarder les modèles originaux** avant fragmentation
2. **Documenter les paramètres** utilisés pour chaque fragmentation
3. **Tester chaque fragment** avant déploiement
4. **Monitorer l'espace disque** pendant le processus
5. **Utiliser des noms cohérents** pour les dossiers de fragments

## Exemple Complet

```bash
# Télécharger un modèle (exemple)
python scripts/download_mistral7b_v03_gguf.py

# Fragmenter le modèle
python fragments/fragmenter.py models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
    --output models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments

# Générer le manifest et le tokenizer
python fragments/generate_manifest_for_fragments.py \
    models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments

# Tester le modèle fragmenté
python tests_debug/test_new_manifest.py
```

## Performance Attendue

| Taille Modèle | Temps Fragmentation | Espace Requis |
|---------------|--------------------|---------------|
| 4 GB          | 10-30 minutes      | 8-12 GB       |
| 8 GB          | 30-60 minutes      | 16-24 GB      |
| 14 GB         | 1-2 heures         | 28-42 GB      |

## Notes Techniques

- Le fragmenter charge le modèle entier en mémoire pendant le traitement
- Les fragments sont optimisés pour une distribution P2P efficace
- Le format est compatible avec le système d'inférence distribuée existant
- Les checksums permettent de vérifier l'intégrité des fragments

## Support

Pour les problèmes de fragmentation:
1. Vérifier les logs d'erreur
2. Consulter la documentation GGUF
3. Examiner le code source du fragmenter
4. Contacter l'équipe de développement avec les détails de l'erreur

---
*Dernière mise à jour: 2024-02-17*
*Version: 1.0*
