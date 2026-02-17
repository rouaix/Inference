#!/usr/bin/env python3
"""
Script d'analyse d'architecture du projet d'inférence distribuée.
Génère des diagrammes et une documentation complète de l'architecture.
"""

import os
import sys
from pathlib import Path
import inspect

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_module(module_path, module_name):
    """Analyse un module Python et retourne sa structure."""
    try:
        module = __import__(module_path, fromlist=[module_name])
        
        # Récupérer les classes et fonctions
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module_path:
                classes.append({
                    'name': name,
                    'methods': [m for m in dir(obj) if not m.startswith('_')],
                    'doc': inspect.getdoc(obj) or "No documentation"
                })
            elif inspect.isfunction(obj) and obj.__module__ == module_path:
                functions.append({
                    'name': name,
                    'args': str(inspect.signature(obj)) if hasattr(inspect, 'signature') else '()',
                    'doc': inspect.getdoc(obj) or "No documentation"
                })
        
        return {
            'module': module_name,
            'classes': classes,
            'functions': functions,
            'file': inspect.getfile(module)
        }
    except ImportError as e:
        return {'module': module_name, 'error': str(e)}

def generate_mermaid_diagram(architecture):
    """Génère un diagramme Mermaid à partir de l'analyse."""
    
    mermaid = ["```mermaid", "graph TD"]
    
    # Ajouter les modules principaux
    for module_info in architecture:
        if 'error' not in module_info:
            module_name = module_info['module'].replace('.', '_')
            mermaid.append(f"    {module_name}[{module_info['module']}]")
            
            # Ajouter les classes
            for cls in module_info['classes']:
                class_name = f"{module_name}_{cls['name']}"
                mermaid.append(f"    {class_name}[{cls['name']}]")
                mermaid.append(f"    {module_name} --> {class_name}")
                
                # Ajouter les méthodes principales
                for method in cls['methods'][:3]:  # Limiter à 3 méthodes pour la lisibilité
                    mermaid.append(f"    {class_name} --> |{method}| {class_name}_{method}")
            
            # Ajouter les fonctions principales
            for func in module_info['functions'][:3]:  # Limiter à 3 fonctions
                mermaid.append(f"    {module_name} --> |{func['name']}| {module_name}_{func['name']}")
    
    mermaid.append("```")
    return "\n".join(mermaid)

def analyze_project_structure():
    """Analyse la structure globale du projet."""
    
    print("=" * 80)
    print("ANALYSE D'ARCHITECTURE DU PROJET")
    print("=" * 80)
    
    # Modules principaux à analyser
    modules_to_analyze = [
        ('inference.p2p_inference', 'p2p_inference'),
        ('distribution.server', 'server'),
        ('distribution.reseau', 'reseau'),
        ('distribution.local', 'local'),
        ('fragments.fragmenter', 'fragmenter'),
        ('inference.fragment_executor', 'fragment_executor'),
    ]
    
    architecture = []
    
    print("\nAnalyse des modules principaux...")
    for module_path, module_name in modules_to_analyze:
        print(f"  - Analyse de {module_name}...")
        result = analyze_module(module_path, module_name)
        architecture.append(result)
        
        if 'error' in result:
            print(f"    [WARNING] Erreur: {result['error']}")
        else:
            print(f"    [OK] {len(result['classes'])} classes, {len(result['functions'])} fonctions")
    
    return architecture

def generate_architecture_docs(architecture):
    """Génère une documentation Markdown de l'architecture."""
    
    docs = ["# Architecture du Projet d'Inférence Distribuée", "", "## Vue d'Ensemble", ""]
    
    docs.append("Ce document décrit l'architecture globale du système d'inférence distribuée pour grands modèles de langage.")
    docs.append("")
    
    # Ajouter le diagramme Mermaid
    docs.append("## Diagramme d'Architecture")
    docs.append("")
    docs.append(generate_mermaid_diagram(architecture))
    docs.append("")
    
    # Ajouter les détails par module
    docs.append("## Modules Principaux")
    docs.append("")
    
    for module_info in architecture:
        if 'error' not in module_info:
            docs.append(f"### {module_info['module']}")
            docs.append(f"")
            docs.append(f"**Fichier** : `{module_info['file']}`")
            docs.append(f"")
            
            # Classes
            if module_info['classes']:
                docs.append("#### Classes")
                for cls in module_info['classes']:
                    docs.append(f"- **{cls['name']}** : {cls['doc'][:100]}...")
                    if cls['methods']:
                        docs.append(f"  - Méthodes principales: {', '.join(cls['methods'][:5])}")
                docs.append("")
            
            # Fonctions
            if module_info['functions']:
                docs.append("#### Fonctions")
                for func in module_info['functions'][:5]:  # Limiter à 5 fonctions
                    docs.append(f"- **{func['name']}{func['args']}** : {func['doc'][:100]}...")
                docs.append("")
    
    # Ajouter l'analyse des flux principaux
    docs.append("## Flux Principaux")
    docs.append("")
    docs.append("### 1. Flux d'Inférence Locale")
    docs.append("```")
    docs.append("1. Chargement du manifest et des fragments")
    docs.append("2. Tokenization du prompt")
    docs.append("3. Exécution couche par couche avec FragmentExecutor")
    docs.append("4. Génération des tokens")
    docs.append("5. Déchargement des fragments")
    docs.append("```")
    docs.append("")
    
    docs.append("### 2. Flux d'Inférence Distribuée (Client/Serveur)")
    docs.append("```")
    docs.append("1. Client: Tokenization du prompt")
    docs.append("2. Client: Envoi du hidden_state au serveur")
    docs.append("3. Serveur: Chargement du fragment approprié")
    docs.append("4. Serveur: Exécution de la couche")
    docs.append("5. Serveur: Retour du résultat")
    docs.append("6. Client: Agrégation des résultats")
    docs.append("7. Client: Génération des tokens")
    docs.append("```")
    docs.append("")
    
    # Ajouter les points d'amélioration
    docs.append("## Points d'Amélioration Identifiés")
    docs.append("")
    docs.append("- **Performance réseau** : Implémenter la compression des tensors")
    docs.append("- **Gestion mémoire** : Optimiser le garbage collection")
    docs.append("- **Résilience** : Ajouter plus de mécanismes de fallback")
    docs.append("- **Monitoring** : Implémenter des métriques de performance")
    docs.append("")
    
    return "\n".join(docs)

def save_analysis_to_file(docs, filename="ARCHITECTURE_ANALYSIS.md"):
    """Sauvegarde l'analyse dans un fichier."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(docs)
    print(f"\n[SUCCESS] Analyse sauvegardée dans {filename}")

if __name__ == "__main__":
    # Analyser le projet
    architecture = analyze_project_structure()
    
    # Générer la documentation
    docs = generate_architecture_docs(architecture)
    
    # Sauvegarder
    save_analysis_to_file(docs)
    
    print("\n" + "=" * 80)
    print("ANALYSE TERMINÉE")
    print("=" * 80)
    print("\nProchaines étapes suggérées:")
    print("1. Examiner le fichier ARCHITECTURE_ANALYSIS.md")
    print("2. Identifier les goulots d'étranglement spécifiques")
    print("3. Prioriser les améliorations")
    print("4. Implémenter les optimisations")