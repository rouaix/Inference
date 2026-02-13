# Présentation du Projet Inference-IA-P2P

> **Test en cours :** `Magistral-Small-2509-Q4_K_M.gguf`

## Introduction

Bienvenue dans le projet **Inference-IA-P2P** ! Ce projet vise à permettre à n'importe qui, même sans GPU, de contribuer à l'exécution de grands modèles de langage open-source comme Mistral Large 3. L'idée est de répartir le modèle en petits fragments de 10 Mo, que chaque utilisateur peut stocker sur son appareil (PC, mobile, tablette) et utiliser pour contribuer à l'inférence distribuée.

### Glossaire des Termes Techniques

Pour mieux comprendre le projet, voici quelques termes techniques expliqués simplement :

- **Tokenisation** : Processus de conversion d'un texte en une séquence de tokens (unités de base que le modèle peut comprendre, comme des mots ou des parties de mots).
- **Embedding** : Conversion des tokens en vecteurs numériques (listes de nombres) qui représentent leur signification dans un espace mathématique.
- **Inférence** : Processus d'utilisation d'un modèle entraîné pour faire des prédictions ou générer du texte.
- **Fragment** : Une petite partie du modèle, découpée pour être stockée et traitée séparément.
- **Nœud** : Un appareil (PC, mobile, tablette) participant au réseau et stockant un fragment.
- **MoE (Mixture of Experts)** : Une architecture de modèle où plusieurs experts (sous-modèles) sont disponibles, mais seuls quelques-uns sont activés pour chaque tâche.
- **GPU** : Unité de traitement graphique, souvent utilisée pour accélérer les calculs d'IA.
- **CPU** : Unité centrale de traitement, présente dans tous les appareils.

## Objectifs Principaux

- **Accessibilité** : Permettre à tous de participer, même avec des appareils modestes.
- **Décentralisation** : Créer un réseau résilient où chaque nœud contribue à l'inférence.
- **Simplicité** : Installation et utilisation en un clic.
- **Open Source** : Le projet est sous licence Apache 2.0, comme Mistral Large 3.

## Pourquoi Mistral Large 3 ?

Mistral Large 3 est un modèle de langage de grande taille avec 675 milliards de paramètres. Son architecture **Mixture of Experts (MoE)** est idéale pour ce projet car elle permet de n'activer que 2 experts sur 128 par couche pour chaque token. Cela réduit considérablement la charge de calcul et la coordination réseau nécessaire.

### Qu'est-ce qu'un Modèle de Langage ?

Un modèle de langage est un programme informatique capable de comprendre et de générer du texte. Il est entraîné sur de grandes quantités de données pour apprendre les motifs et les structures du langage. Plus un modèle a de paramètres, plus il est capable de comprendre et de générer du texte de manière précise et nuancée.

### Qu'est-ce que l'Architecture MoE ?

L'architecture **Mixture of Experts (MoE)** est une technique qui permet de diviser un modèle en plusieurs sous-modèles (experts). Pour chaque tâche, seuls quelques experts sont activés, ce qui réduit la charge de calcul et améliore l'efficacité. Dans le cas de Mistral Large 3, seuls 2 experts sur 128 sont activés pour chaque token, ce qui signifie que seulement une petite partie du modèle est utilisée à chaque fois.

### Caractéristiques de Mistral Large 3

- **Paramètres totaux** : 675 milliards (la taille totale du modèle)
- **Paramètres actifs par token** : 41 milliards (6%) (seulement une petite partie du modèle est utilisée à chaque fois)
- **Experts par couche** : 128 (nombre total d'experts disponibles)
- **Experts activés par token** : 2 (nombre d'experts utilisés pour chaque token)
- **Couches transformer** : 88 (nombre de couches dans le modèle)
- **Licence** : Apache 2.0 (licence open source permissive)

## Architecture du Projet

### Découpage du Modèle

Le modèle est découpé en fragments de 10 Mo, classés en deux catégories :

1. **Fragments "toujours actifs"** (~1 218 fragments, ~12 Go) :
   - **Embedding et LM Head** : Ces fragments sont responsables de la conversion des tokens en vecteurs (embedding) et des vecteurs en tokens (LM Head).
   - **Projections d'attention Q, K, V, O** : Ces fragments sont utilisés pour calculer l'attention, un mécanisme clé des modèles de langage qui permet de se concentrer sur les parties importantes du texte.
   - **Routeurs MoE** : Ces fragments déterminent quels experts doivent être activés pour chaque token.
   - **Normes RMSNorm** : Ces fragments sont utilisés pour normaliser les vecteurs, ce qui améliore la stabilité et la performance du modèle.

2. **Fragments "experts conditionnels"** (~44 907 fragments, ~438 Go) :
   - **Les FFN des 128 experts × 88 couches** : Ces fragments contiennent les experts individuels du modèle MoE. Chaque expert est un sous-modèle spécialisé qui peut être activé ou désactivé en fonction des besoins.

### Architecture d'un Nœud

Chaque nœud dans le réseau stocke un fragment de 10 Mo et exécute des calculs partiels. Voici l'architecture d'un nœud :

```
┌─────────────────────────────────────┐
│  Application (PWA / Tauri)          │
│  ┌───────────────────────────────┐  │
│  │  Interface Chat               │  │
│  │  (envoyer prompts, voir      │  │
│  │   les réponses)               │  │
│  ├───────────────────────────────┤  │
│  │  Worker de calcul (WASM)      │  │
│  │  - Stocke 1 fragment (10 Mo) │  │
│  │  - Exécute matmul partiel    │  │
│  │  - Envoie le résultat        │  │
│  ├───────────────────────────────┤  │
│  │  Module réseau P2P            │  │
│  │  - libp2p (WebRTC)           │  │
│  │  - DHT (découverte pairs)    │  │
│  │  - Heartbeat                  │  │
│  │  - NAT traversal             │  │
│  └───────────────────────────────┘  │
│                                     │
│  Empreinte : ~15 Mo total           │
│  (10 Mo fragment + 5 Mo app)        │
└─────────────────────────────────────┘
```

### Architecture Réseau

Le réseau est composé de trois types de nœuds :

1. **Bootstrap Nodes** : Serveurs d'amorçage pour aider les nouveaux nœuds à rejoindre le réseau.
2. **Super Nodes** : Nœuds élus dynamiquement pour agrégé les résultats partiels.
3. **Nœuds réguliers** : Utilisateurs ordinaires qui stockent un fragment et contribuent aux calculs.

```
                    ┌──────────────┐
                    │  Bootstrap   │
                    │  Nodes       │
                    │  (serveurs   │
                    │  d'amorçage) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼─────┐ ┌───▼─────┐
        │  Super     │ │  Super  │ │  Super  │
        │  Node      │ │  Node   │ │  Node   │
        │  (agrège)  │ │         │ │         │
        └─────┬──────┘ └────┬────┘ └────┬────┘
          ┌───┼───┐     ┌───┼───┐   ┌───┼───┐
          │   │   │     │   │   │   │   │   │
          n1  n2  n3    n4  n5  n6  n7  n8  n9
          │   │   │     │   │   │   │   │   │
         10Mo chacun   10Mo chacun  10Mo chacun
```

## Flux d'Inférence

Voici comment se déroule l'inférence distribuée :

1. **Tokenisation** : Le prompt est converti en tokens (unités de base que le modèle peut comprendre).
2. **Embedding** : Les tokens sont convertis en vecteurs (listes de nombres qui représentent leur signification).
3. **Couches Transformer** : Pour chaque couche, les calculs sont distribués aux nœuds appropriés. Chaque couche transforme les vecteurs pour extraire des informations plus complexes.
4. **LM Head** : Le vecteur final est converti en logits (scores de probabilité pour chaque token possible) pour prédire le prochain token.
5. **Décodage** : Le token prédit est converti en texte compréhensible par l'humain.

### Exemple de Flux pour une Couche

```
Utilisateur tape : "Explique la photosynthèse"
        │
        ▼
   1. TOKENISATION (locale)
      "Explique la photosynthèse" → [15234, 432, 98712]
      (Le texte est converti en une séquence de tokens)
        │
        ▼
   2. EMBEDDING (fragments toujours actifs)
      tokens → vecteurs de dimension 6144
      (Les tokens sont convertis en vecteurs numériques)
      Nœuds contactés : ~70 (700 fragments / 10 par fragment, répliques)
        │
        ▼
   3. COUCHE 1/88 — ATTENTION
      (Calcul de l'attention pour se concentrer sur les parties importantes du texte)
      │
      ├─ Q_proj : 5 nœuds calculent en parallèle
      ├─ K_proj : 2 nœuds
      ├─ V_proj : 2 nœuds
      ├─ Attention computation (agrégateur)
      └─ O_proj : 5 nœuds
      │
      ├─ ROUTEUR : 1 nœud évalue les scores des 128 experts
      │            → sélectionne Expert 42 et Expert 97
      │            (Le routeur détermine quels experts sont les plus pertinents)
      │
      ├─ EXPERT 42 : 5 nœuds (gate + up + down)
      ├─ EXPERT 97 : 5 nœuds (gate + up + down)
      │            (Les experts sélectionnés effectuent des calculs spécialisés)
      │
      └─ Agrégation pondérée des résultats experts
      (Les résultats des experts sont combinés pour produire un vecteur final)
        │
        ▼
   4. COUCHES 2 à 88 — même processus
      (Le processus est répété pour chaque couche du modèle)
        │
        ▼
   5. LM HEAD (fragments toujours actifs)
      vecteur 6144 → logits sur 131 072 tokens du vocabulaire
      (Le vecteur final est converti en scores de probabilité pour chaque token possible)
      → token le plus probable sélectionné
      (Le token avec la plus haute probabilité est sélectionné)
        │
        ▼
   6. DÉCODAGE
      token_id → "La"
      (Le token sélectionné est converti en texte)
        │
        ▼
   7. BOUCLE → retour à l'étape 2 pour le token suivant
      jusqu'à complétion de la réponse
      (Le processus est répété pour générer le texte complet)
```

## Résilience et Tolérance aux Pannes

### Qu'est-ce que la Résilience ?

La résilience est la capacité du réseau à continuer de fonctionner même en cas de pannes ou de défaillances. Dans un réseau distribué comme celui-ci, il est crucial de pouvoir tolérer la perte de certains nœuds sans interrompre le service.

### Stratégie de Réplication

Chaque fragment est hébergé par plusieurs nœuds simultanément pour assurer la résilience :

| Réplication | Survit sans dégradation | Survit en mode dégradé |
|---|---|---|
| ×3 | Jusqu'à ~10% de pannes | Jusqu'à ~30% de pannes |
| ×5 | Jusqu'à ~40% de pannes | Jusqu'à ~50% de pannes |

- **Réplication ×3** : Chaque fragment est stocké sur 3 nœuds différents. Le réseau peut tolérer la perte de jusqu'à 10% des nœuds sans dégradation de performance.
- **Réplication ×5** : Chaque fragment est stocké sur 5 nœuds différents. Le réseau peut tolérer la perte de jusqu'à 40% des nœuds sans dégradation de performance.

### Mécanismes de Résilience

- **Heartbeat** : Chaque nœud envoie un signal "je suis vivant" toutes les 30 secondes. Si un nœud ne répond pas après 3 signaux manqués, il est déclaré hors ligne.
- **Failover automatique** : Si un nœud tombe en panne pendant un calcul, la requête est automatiquement re-routée vers une réplique du fragment. Cela ajoute une latence de ~200-500ms, mais permet de continuer le calcul.
- **Redistribution dynamique** : Si un fragment n'a plus assez de répliques (en dessous du seuil de sécurité), le réseau demande à d'autres nœuds de télécharger et de servir ce fragment.
- **Remplacement d'experts** : Si un expert est totalement indisponible, le routeur peut augmenter le poids des experts restants. La qualité du texte peut baisser légèrement, mais l'inférence continue.

## Stack Technique

### Moteur de Calcul

- **Multiplication matricielle** : Opération mathématique de base pour les modèles de langage, effectuée par llama.cpp compilé en WASM (WebAssembly), un format qui permet d'exécuter du code performant dans un navigateur ou sur un appareil mobile.
- **Format des poids** : GGUF quantifié Q4_K_M. GGUF est un format de fichier pour stocker les modèles de langage, et Q4_K_M est une méthode de quantification qui réduit la taille des poids du modèle tout en conservant une bonne précision.
- **Runtime WASM** : wasmtime (natif) ou navigateur. WASM permet d'exécuter du code compilé à une vitesse proche du natif, même dans un navigateur web.

### Réseau P2P

- **Transport P2P** : libp2p. Une bibliothèque pour créer des réseaux pair-à-pair (P2P), où chaque nœud peut communiquer directement avec les autres.
- **NAT traversal** : WebRTC ICE + TURN relays. Techniques pour permettre aux nœuds derrière des routeurs (NAT) de communiquer directement, même s'ils sont sur des réseaux différents.
- **Découverte** : DHT Kademlia. Un système de table de hachage distribuée pour trouver et localiser les nœuds dans le réseau.
- **Sérialisation** : Protocol Buffers / MessagePack. Formats pour encoder et décoder les données échangées entre les nœuds, optimisés pour la taille et la vitesse.

### Interface Utilisateur

- **Desktop** : Tauri (Rust + WebView). Un framework pour créer des applications desktop légères et performantes en utilisant des technologies web.
- **Mobile / Tablette** : PWA installable. Une Progressive Web App (PWA) est une application web qui peut être installée sur un appareil mobile et fonctionner comme une application native.
- **Web pur** : Page web + Service Worker. Une application web classique, améliorée avec un Service Worker pour permettre un fonctionnement hors ligne et des notifications.

### Langages

- **Moteur d'inférence** : C++ (llama.cpp) → compilé WASM. Le moteur d'inférence est écrit en C++ pour des performances maximales, puis compilé en WASM pour une exécution universelle.
- **Réseau P2P** : Rust (libp2p) ou TypeScript (js-libp2p). Rust est utilisé pour sa performance et sa sécurité, tandis que TypeScript est utilisé pour sa compatibilité avec les navigateurs web.
- **App desktop** : Rust (Tauri). Tauri est utilisé pour créer des applications desktop légères et performantes.
- **Interface** : TypeScript + Svelte ou React. TypeScript est utilisé pour sa robustesse, et Svelte ou React pour créer des interfaces utilisateur réactives et modernes.
- **Orchestrateur** : Rust ou Go. L'orchestrateur est responsable de la coordination des calculs distribués et peut être écrit en Rust pour des performances maximales ou en Go pour une simplicité de développement.

## Feuille de Route

### Phase 1 — Proof of Concept (FAIT ✅)

- Modéliser l'architecture MoE de Mistral Large 3
- Fragmenteur : découpe un modèle simulé en chunks de 10 Mo
- Simulateur de réseau P2P avec nœuds, réplication, DHT
- Tests de tolérance aux pannes (0% à 70% de nœuds perdus)

### Phase 2 — Fragmenteur GGUF réel

- Parser le format GGUF (header, métadonnées, tenseurs)
- Découper les tenseurs en fragments de 10 Mo avec métadonnées correctes
- Reconstruire un modèle à partir des fragments et vérifier l'intégrité

### Phase 3 — Calcul distribué réel

- Implémenter la multiplication matricielle partielle sur un fragment
- Tester : N processus locaux, chacun avec un fragment, qui collaborent via IPC
- Vérifier que le résultat agrégé est identique à llama.cpp standard

### Phase 4 — Réseau P2P réel

- Intégrer libp2p (transport WebRTC + TCP)
- Implémenter la DHT Kademlia pour la découverte de nœuds
- Bootstrap nodes : serveurs d'amorçage pour les nouveaux arrivants
- NAT traversal : hole punching, TURN relay pour les cas difficiles

### Phase 5 — Application utilisateur

- App desktop Tauri (Windows/Mac/Linux)
- PWA pour mobile/tablette
- Onboarding : téléchargement automatique d'un fragment (10 Mo)
- Dashboard réseau (nombre de nœuds, santé)

### Phase 6 — Scaling et optimisation

- Optimisation de la bande passante (compression des activations)
- Système d'incitation (crédits de calcul : tu contribues → tu utilises)
- Architecture en arbre pour l'agrégation
- Cache intelligent (les requêtes fréquentes sont pré-calculées)

## Estimations de Performance

### Temps par Token

| Scénario | Latence réseau | Calcul CPU | Total/token |
|---|---|---|---|
| Réseau local (LAN) | ~1 ms/hop | ~5 ms | ~5-10 secondes |
| Internet fibre | ~20 ms/hop | ~5 ms | ~15-30 secondes |
| Internet mixte (mobile) | ~50-100 ms/hop | ~10 ms | ~30-60 secondes |

### Temps de Génération pour une Réponse Complète

| Longueur réponse | Tokens | Temps estimé (fibre) | Temps estimé (mixte) |
|---|---|---|---|
| Phrase courte | ~30 tokens | ~8 minutes | ~15 minutes |
| Paragraphe | ~100 tokens | ~25 minutes | ~50 minutes |
| Réponse détaillée | ~500 tokens | ~2 heures | ~4 heures |

## Système d'Incitation

Pour encourager les utilisateurs à contribuer, un système de crédits de calcul est mis en place :

| Action | Crédits |
|---|---|
| Héberger un fragment (par heure d'uptime) | +1 crédit |
| Répondre à une requête de calcul | +0.1 crédit |
| Poser une question au modèle | -10 à -50 crédits (selon longueur) |
| Nouveau utilisateur (bonus bienvenue) | +100 crédits |

## Sécurité et Confidentialité

### Menaces Identifiées

| Menace | Impact | Mitigation |
|---|---|---|
| Nœud malveillant qui renvoie de faux résultats | Réponse corrompue | Calcul redondant : 2+ nœuds font le même calcul, comparaison des résultats |
| Interception des activations (prompt sniffing) | Fuite de données | Chiffrement des activations en transit (TLS/DTLS) |
| Nœud qui espionne les prompts | Vie privée | Aucun nœud ne voit le prompt complet, seulement des vecteurs numériques intermédiaires |
| Attaque Sybil (faux nœuds) | Prise de contrôle | Système de réputation basé sur l'historique d'uptime et la cohérence des calculs |
| Déni de service | Réseau inutilisable | Rate limiting, priorité aux nœuds avec bonne réputation |

### Confidentialité Inhérente au Design

Un point fort du design : **aucun nœud individuel ne voit le texte en clair**. Chaque nœud ne reçoit que des vecteurs d'activation (des tableaux de nombres flottants). Seul le nœud de l'utilisateur qui a posé la question fait la tokenisation et le décodage.

## Comparaison avec l'Existant

| | Ollama / llama.cpp | API cloud (OpenAI, etc.) |
|---|---|---|---|---|
| GPU requis | Oui (serveurs) | Non | Recommandé | Non (cloud) |
| Fragment par nœud | Couches entières (~1-5 Go) | 10 Mo | Modèle entier | N/A |
| Installation | pip + Python + CLI | Un clic / PWA | Non | Clé API |
| Mobile | Non | Oui | Non | Via app |
| Nœuds nécessaires | ~10-50 | ~50 000-200 000 | 1 | 0 (centralisé) |
| Vitesse | ~4-6 tokens/s | ~0.02-0.1 token/s | ~10-30 tokens/s | ~50-100 tokens/s |
| Coût | Gratuit | Gratuit | Gratuit | Payant |
| Vie privée | Moyenne (nœuds voient les couches) | Bonne (fragments opaques) | Excellente (local) | Faible (cloud) |
| Censure | Résistant | Très résistant | Local | Censuré |
| Dernier modèle supporté | Llama 3.1 405B | Mistral Large 3 675B | Variable | Derniers modèles |

## Conclusion

Le projet **Inference-IA-P2P** est une initiative ambitieuse pour démocratiser l'accès aux grands modèles de langage. En répartissant le modèle en petits fragments et en utilisant un réseau P2P, il permet à n'importe qui de contribuer à l'inférence, même avec des appareils modestes. Le projet est en cours de développement et suit une feuille de route claire pour atteindre ses objectifs.

Pour plus d'informations, consultez le fichier [README.md](README.md) et explorez le code source.
