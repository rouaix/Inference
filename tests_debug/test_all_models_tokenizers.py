#!/usr/bin/env python3
"""
Test complet de tous les modèles pour vérifier les tokenizers.
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du projet (remonter de tests_debug/ vers la racine)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_model_tokenizer(model_dir):
    """Test le tokenizer d'un modèle."""
    print(f"\n{'='*60}")
    print(f"Test de {model_dir.name}")
    print(f"{'='*60}")
    
    try:
        from inference.p2p_inference import P2PInferenceEngine
        
        # Vérifier que le tokenizer existe
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            print(f"[ERREUR] Tokenizer manquant: {tokenizer_path}")
            return False
        
        print(f"[OK] Tokenizer présent: {tokenizer_path}")
        
        # Charger le modèle
        engine = P2PInferenceEngine(str(model_dir), verbose=False)
        
        # Tester le tokenizer
        test_prompt = "Bonjour, comment ça va?"
        tokens = engine.tokenizer.encode(test_prompt)
        decoded = engine.tokenizer.decode(tokens)
        
        print(f"Prompt: '{test_prompt}'")
        print(f"Tokens: {len(tokens)} tokens")
        print(f"Décodage: '{decoded}'")
        
        # Vérifier que le décodage correspond
        if decoded.strip() == test_prompt.strip():
            print(f"[OK] Tokenizer fonctionne correctement")
            return True
        else:
            print(f"[ERREUR] Décodage incorrect")
            print(f"   Original: '{test_prompt}'")
            print(f"   Décodé:   '{decoded}'")
            return False
            
    except Exception as e:
        print(f"[ERREUR] Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Teste tous les modèles."""
    print("Test complet des tokenizers pour tous les modèles")
    print("=" * 60)
    
    # Trouver tous les dossiers de fragments
    models_dir = Path("models")
    fragment_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "fragments" in d.name]
    
    print(f"Modèles fragmentés trouvés: {len(fragment_dirs)}")
    
    # Tester chaque modèle
    results = {}
    for model_dir in fragment_dirs:
        success = test_model_tokenizer(model_dir)
        results[model_dir.name] = success
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES TESTS")
    print(f"{'='*60}")
    
    total_models = len(results)
    successful = sum(1 for success in results.values() if success)
    
    print(f"Modèles testés: {total_models}")
    print(f"Succès: {successful}")
    print(f"Échecs: {total_models - successful}")
    
    for model_name, success in results.items():
        status = "[OK]" if success else "[ERREUR]"
        print(f"{status} {model_name}")
    
    if successful == total_models:
        print(f"\n[SUCCES] Tous les modèles ont des tokenizers fonctionnels!")
        return True
    else:
        print(f"\n[AVERTISSEMENT] Certains modèles ont des problèmes de tokenizer")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)