# Modèles

> **Test en cours :** `Magistral-Small-2509-Q4_K_M.gguf`

Ce dossier contient les modèles GGUF et leurs fragments pour l'inférence P2P.

---

## 📁 Contenu

### Modèles GGUF
- **tinyllama-1.1b-chat-v1.0.Q8_0.gguf** (1.09 GB) - Modèle principal quantifié Q8_0
- **tinyllama.gguf** (638 MB) - Modèle TinyLlama original

### Fragments P2P
- **tinyllama_q8_fragments_v2/** - Fragments du modèle Q8_0 (279 fragments)
  - `manifest.json` - Métadonnées et index des fragments
  - `gguf_header.dat` - En-tête GGUF
  - `fragment_*.dat` - Fragments de tenseurs

### Tokenizer
- **tokenizer.model** - Tokenizer SentencePiece pour TinyLlama

---

## 🚀 Utilisation

### Avec le Pont Hybride (Recommandé) ✅
```python
from p2p_bridge import P2PBridge

bridge = P2PBridge("models/tinyllama_q8_fragments_v2")
text = bridge.generate("Hello", max_tokens=50)
print(text)
```

### Avec le Moteur Python Pur
```bash
python p2p_inference.py models/tinyllama_q8_fragments_v2 --prompt "Hello" --max-tokens 10
```

### Fragmenter un Nouveau Modèle
```bash
python fragmenter.py models/nouveau_modele.gguf models/nouveau_modele_fragments
```

---

## 📊 Informations sur TinyLlama

**Modèle**: TinyLlama-1.1B-Chat-v1.0
**Architecture**: Llama 2
**Taille**: 1.1 milliard de paramètres
**Quantification**: Q8_0 (8-bit)
**Contexte**: 2048 tokens
**Vocabulaire**: 32,000 tokens

**Configuration**:
- Dimension: 2048
- Têtes d'attention: 32
- Têtes KV (GQA): 4
- Couches: 22
- FFN dimension: 5632
- RoPE theta: 10000.0

---

## 🔧 Fragmentation

Le modèle Q8_0 a été fragmenté en 279 fragments pour permettre le chargement progressif:

```
Total size: 1.09 GB
Fragment size: ~4 MB each
Fragments: 279
Format: Q8_0 (quantifié 8-bit)
```

**Avantages de la fragmentation**:
- ✅ Chargement progressif des couches
- ✅ Réduction de l'empreinte mémoire
- ✅ Distribution P2P facilitée
- ✅ Reconstruction lossless garantie

---

## 📝 Notes

- Le modèle Q8_0 est utilisé pour tous les tests et la production
- Les fragments v2 incluent l'en-tête GGUF pour reconstruction complète
- Le tokenizer SentencePiece est requis pour l'encodage/décodage

---

## 🎯 Recommandation

Pour la production, utilisez le **pont hybride** (`p2p_bridge.py`) qui combine:
- Fragmentation P2P pour la distribution
- llama.cpp pour l'inférence (performance optimale)
- Génération de texte cohérent garantie
