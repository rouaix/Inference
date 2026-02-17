#!/usr/bin/env python3
"""
Script pour fragmenter tous les nouveaux modèles GGUF.
Ce script utilise le système de fragmentation existant pour créer des fragments
pour chaque modèle GGUF disponible.
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def fragment_model(gguf_path, output_dir, model_name):
    """Fragmente un modèle GGUF en utilisant le fragmenter existant."""
    print(f"\n{'='*60}")
    print(f"Fragmentation de {model_name}")
    print(f"{'='*60}")
    
    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Vérifier que le fichier GGUF existe
    if not gguf_path.exists():
        print(f"❌ Fichier non trouvé: {gguf_path}")
        return False
    
    print(f"Fichier source: {gguf_path}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Taille du fichier: {gguf_path.stat().st_size / (1024**3):.2f} GB")
    
    # Commande pour fragmenter le modèle
    # Nous devons utiliser le fragmenter.py avec les bons paramètres
    try:
        # Appeler le script de fragmentation
        cmd = [
            sys.executable, "fragments/fragmenter.py",
            str(gguf_path),
            "--output", str(output_dir)
        ]
        
        print(f"Exécution: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Fragmentation terminée avec succès")
            
            # Compter les fragments générés
            fragments = list(output_dir.glob("*.dat"))
            print(f"Nombre de fragments générés: {len(fragments)}")
            
            # Générer le manifest
            if generate_manifest(output_dir, model_name):
                print(f"✅ Manifest généré avec succès")
                return True
            else:
                print(f"❌ Échec de la génération du manifest")
                return False
        else:
            print(f"❌ Échec de la fragmentation")
            print(f"Erreur: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception lors de la fragmentation: {e}")
        return False

def generate_manifest(fragments_dir, model_name):
    """Génère un manifest.json pour les fragments."""
    try:
        # Utiliser le générateur de manifest existant
        cmd = [
            sys.executable, "fragments/generate_manifest_for_fragments.py",
            str(fragments_dir),
            "--model-name", model_name
        ]
        
        print(f"Génération du manifest...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            manifest_path = fragments_dir / "manifest.json"
            if manifest_path.exists():
                # Vérifier le manifest
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                print(f"Manifest généré: {len(manifest.get('fragments', []))} fragments")
                return True
            else:
                print(f"❌ Fichier manifest non trouvé")
                return False
        else:
            print(f"❌ Échec de la génération du manifest")
            print(f"Erreur: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception lors de la génération du manifest: {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("Script de fragmentation de tous les modèles GGUF")
    print("=" * 60)
    
    # Liste des modèles à fragmenter
    models_dir = Path("models")
    gguf_files = list(models_dir.glob("*.gguf"))
    
    # Filtrer les fichiers mmproj (multimodaux)
    gguf_files = [f for f in gguf_files if "mmproj" not in f.name]
    
    print(f"Modèles GGUF trouvés: {len(gguf_files)}")
    for f in gguf_files:
        print(f"  - {f.name}")
    
    # Fragmenter chaque modèle
    results = {}
    for gguf_file in gguf_files:
        # Extraire le nom du modèle pour le dossier de sortie
        model_name = gguf_file.stem.replace("-", "_")
        output_dir = models_dir / f"{model_name}_fragments"
        
        success = fragment_model(gguf_file, output_dir, model_name)
        results[gguf_file.name] = success
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DE LA FRAGMENTATION")
    print(f"{'='*60}")
    
    total_models = len(results)
    successful = sum(1 for success in results.values() if success)
    
    print(f"Modèles traités: {total_models}")
    print(f"Succès: {successful}")
    print(f"Échecs: {total_models - successful}")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")
    
    if successful == total_models:
        print(f"\n🎉 Tous les modèles ont été fragmentés avec succès!")
    else:
        print(f"\n⚠️ Certains modèles ont échoué. Voir les détails ci-dessus.")
    
    return successful == total_models

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)