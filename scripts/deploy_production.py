#!/usr/bin/env python3
"""
Script de déploiement en production.
Prépare le système d'inférence distribuée avec les modèles fragmentés.
"""

import sys
import subprocess
from pathlib import Path
import json
import shutil

def run_command(cmd, description):
    """Exécute une commande et affiche le résultat."""
    print(f"\n🔧 {description}")
    print(f"   Commande: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"   ✅ Succès")
            return True
        else:
            print(f"   ❌ Échec")
            if result.stderr:
                print(f"   Erreur: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout")
        return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def verify_model_fragments(model_dir):
    """Vérifie qu'un modèle est correctement fragmenté."""
    print(f"\n🔍 Vérification du modèle: {model_dir.name}")
    
    # Vérifier le manifest
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"   ❌ Manifest manquant")
        return False
    
    # Charger le manifest
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        fragments_count = len(manifest.get('fragments', []))
        total_fragments = manifest.get('total_fragments', 0)
        
        print(f"   ✅ Manifest valide")
        print(f"   Fragments déclarés: {total_fragments}")
        print(f"   Fragments dans le manifest: {fragments_count}")
        
        # Vérifier les fichiers de fragments
        fragment_files = list(model_dir.glob("*.dat"))
        print(f"   Fichiers .dat trouvés: {len(fragment_files)}")
        
        # Vérifier le tokenizer
        tokenizer_path = model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            print(f"   ✅ Tokenizer présent")
        else:
            print(f"   ⚠️ Tokenizer manquant")
        
        return fragments_count > 0 and len(fragment_files) > 0
        
    except Exception as e:
        print(f"   ❌ Erreur de lecture du manifest: {e}")
        return False

def prepare_deployment():
    """Prépare le déploiement en production."""
    print("Preparation du déploiement en production")
    print("=" * 60)
    
    # 1. Vérifier l'environnement
    print("\n📋 Vérification de l'environnement...")
    
    # Vérifier Python
    python_cmd = [sys.executable, "--version"]
    result = subprocess.run(python_cmd, capture_output=True, text=True)
    print(f"   Python: {result.stdout.strip()}")
    
    # Vérifier les dépendances
    dependencies = ["numpy", "gguf", "requests", "zstandard"]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep} installé")
        except ImportError:
            print(f"   ❌ {dep} manquant")
    
    # 2. Vérifier les modèles fragmentés
    print(f"\n📦 Vérification des modèles fragmentés...")
    models_dir = Path("models")
    fragment_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "fragments" in d.name]
    
    if not fragment_dirs:
        print("   ❌ Aucun modèle fragmenté trouvé")
        return False
    
    print(f"   Modèles fragmentés trouvés: {len(fragment_dirs)}")
    
    valid_models = []
    for model_dir in fragment_dirs:
        if verify_model_fragments(model_dir):
            valid_models.append(model_dir.name)
    
    print(f"   Modèles valides: {len(valid_models)}")
    for model in valid_models:
        print(f"     ✅ {model}")
    
    # 3. Exécuter les tests
    print(f"\n🧪 Exécution des tests...")
    
    tests = [
        ([sys.executable, "tests_debug/test_serialization.py"], "Tests de sérialisation"),
        ([sys.executable, "tests_debug/test_new_manifest.py"], "Test de chargement de modèle"),
    ]
    
    test_results = []
    for cmd, description in tests:
        success = run_command(cmd, description)
        test_results.append((description, success))
    
    # 4. Préparer les fichiers de configuration
    print(f"\n📝 Préparation des configurations...")
    
    # Copier les fichiers de configuration exemple
    config_files = [
        "distribution/config_example.json",
        "inference/config_example.yaml"
    ]
    
    for config_file in config_files:
        src = Path(config_file)
        if src.exists():
            dst = src.parent / f"{src.stem}.json"
            shutil.copy(src, dst)
            print(f"   ✅ Configuration copiée: {dst}")
    
    # 5. Générer un rapport de déploiement
    print(f"\n📊 Génération du rapport de déploiement...")
    
    report = {
        "status": "ready" if valid_models else "not_ready",
        "timestamp": "2024-02-17",
        "python_version": result.stdout.strip(),
        "valid_models": valid_models,
        "total_models": len(fragment_dirs),
        "test_results": {desc: "PASS" if success else "FAIL" for desc, success in test_results},
        "recommendations": []
    }
    
    if not valid_models:
        report["recommendations"].append("Fragmenter au moins un modèle avant déploiement")
    
    if any(not success for _, success in test_results):
        report["recommendations"].append("Corriger les tests échoués avant déploiement")
    
    # Sauvegarder le rapport
    report_path = Path("distribution/DEPLOYMENT_REPORT.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Rapport généré: {report_path}")
    
    # 6. Afficher le résumé
    print(f"\n📌 Résumé du déploiement")
    print(f"   " + "=" * 56)
    print(f"   Statut: {'PRÊT' if valid_models and all(success for _, success in test_results) else 'NON PRÊT'}")
    print(f"   Modèles disponibles: {len(valid_models)}")
    print(f"   Tests passés: {sum(1 for _, success in test_results if success)}/{len(test_results)}")
    
    if valid_models:
        print(f"\n   🎯 Modèles prêts pour la production:")
        for model in valid_models:
            print(f"      • {model}")
    
    if report["recommendations"]:
        print(f"\n   ⚠️  Recommandations:")
        for rec in report["recommendations"]:
            print(f"      • {rec}")
    
    # 7. Instructions de déploiement
    print(f"\n🚀 Instructions de déploiement")
    print(f"   " + "=" * 56)
    print(f"   1. Vérifier que tous les tests passent")
    print(f"   2. Configurer les paramètres réseau dans distribution/config.json")
    print(f"   3. Lancer le serveur: python distribution/server.py")
    print(f"   4. Lancer les clients: python distribution/client.py")
    print(f"   5. Monitorer avec: python tests_debug/monitor.py")
    
    return len(valid_models) > 0 and all(success for _, success in test_results)

def main():
    """Point d'entrée principal."""
    try:
        success = prepare_deployment()
        
        if success:
            print(f"\n🎉 Déploiement prêt!")
            print(f"   Vous pouvez maintenant lancer le système en production.")
        else:
            print(f"\n⚠️  Déploiement non prêt.")
            print(f"   Veuillez corriger les problèmes identifiés ci-dessus.")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la préparation du déploiement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)