#!/usr/bin/env python3
"""
Test d'inférence simple pour vérifier que les modèles fragmentés fonctionnent.
"""

import sys
import os
from pathlib import Path

# Ajouter le chemin du projet (remonter de tests_debug/ vers la racine)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_model_inference(model_dir_name):
    """Test l'inférence avec un modèle fragmenté."""
    print(f"\n{'='*60}")
    print(f"Test d'inférence: {model_dir_name}")
    print(f"{'='*60}")
    
    try:
        from inference.p2p_inference import P2PInferenceEngine
        
        # Charger le modèle
        model_dir = Path(f"models/{model_dir_name}")
        if not model_dir.exists():
            print(f"[ERREUR] Dossier non trouve: {model_dir}")
            return False
        
        # Vérifier que le manifest existe
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"[ERREUR] Manifest manquant: {manifest_path}")
            print(f"[INFO] Ce modele n'est pas completement fragmente ou le manifest n'a pas ete genere")
            return False
        
        print(f"Chargement du modèle depuis {model_dir}...")
        engine = P2PInferenceEngine(str(model_dir), verbose=True)
        
        # Afficher la configuration
        cfg = engine.config
        print(f"\nConfiguration du modèle:")
        print(f"  Couches: {cfg.n_layers}")
        print(f"  Dimensions: {cfg.dim}")
        print(f"  Têtes: {cfg.n_heads} (KV: {cfg.n_kv_heads})")
        print(f"  Vocabulaire: {cfg.vocab_size}")
        print(f"  FFN: {cfg.hidden_dim}")
        
        # Tester le tokenizer
        print(f"\nTest du tokenizer:")
        test_prompt = "Bonjour, comment ça va?"
        tokens = engine.tokenizer.encode(test_prompt)
        print(f"  Prompt: '{test_prompt}'")
        print(f"  Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Nombre de tokens: {len(tokens)}")
        
        # Décoder les tokens
        decoded = engine.tokenizer.decode(tokens)
        print(f"  Décodage: '{decoded}'")
        
        # Tester le chargement de quelques tenseurs clés
        print(f"\nTest de chargement des tenseurs:")
        key_tensors = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ]
        
        loaded_successfully = 0
        for tensor_name in key_tensors:
            try:
                tensor = engine.load_tensor(tensor_name)
                print(f"  [OK] {tensor_name:<30} shape={tensor.shape}")
                loaded_successfully += 1
            except Exception as e:
                print(f"  [ERREUR] {tensor_name:<30} ERREUR: {e}")
        
        # Tester un forward pass simple (un seul token)
        print(f"\nTest de forward pass (token unique):")
        try:
            # Utiliser le token BOS (début de séquence)
            bos_token = 1  # Typiquement 1 pour les modèles Llama/Mistral
            
            # Charger l'embedding
            embeddings = engine.load_tensor("token_embd.weight")
            if embeddings.ndim == 2 and embeddings.shape[1] == cfg.vocab_size:
                # Transposer si nécessaire
                embeddings = embeddings.T
            
            # Obtenir l'embedding pour le token BOS
            x = embeddings[bos_token].reshape(1, -1)
            print(f"  Embedding BOS shape: {x.shape}")
            print(f"  Embedding stats: mean={x.mean():.4f}, std={x.std():.4f}")
            
            # Passer à travers la première couche
            from inference.p2p_inference import LlamaLayer
            layer0 = LlamaLayer(engine, 0)
            
            # Forward pass (sans cache pour simplifier)
            output, _, _ = layer0.forward(x, engine.freqs_cis, None, None, 0)
            print(f"  Sortie couche 0 shape: {output.shape}")
            print(f"  Sortie stats: mean={output.mean():.4f}, std={output.std():.4f}")
            
            # Vérifier les NaN/Inf
            has_nan = bool(np.any(np.isnan(output)))
            has_inf = bool(np.any(np.isinf(output)))
            
            if has_nan:
                print(f"  ⚠️  Avertissement: NaN détectés dans la sortie")
            if has_inf:
                print(f"  ⚠️  Avertissement: Inf détectés dans la sortie")
            
            if not has_nan and not has_inf:
                print(f"  [OK] Forward pass reussi sans NaN/Inf")
                forward_success = True
            else:
                print(f"  [ERREUR] Forward pass a produit des valeurs numeriques invalides")
                forward_success = False
                
        except Exception as e:
            print(f"  [ERREUR] Echec du forward pass: {e}")
            import traceback
            traceback.print_exc()
            forward_success = False
        
        # Résumé
        print(f"\n{'='*60}")
        print(f"Résumé pour {model_dir_name}:")
        print(f"  Tenseurs chargés: {loaded_successfully}/{len(key_tensors)}")
        print(f"  Forward pass: {'PASS' if forward_success else 'FAIL'}")
        print(f"  Statut global: {'✅ PRÊT' if loaded_successfully == len(key_tensors) and forward_success else '❌ PROBLÈMES'}")
        
        return loaded_successfully == len(key_tensors) and forward_success
        
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Teste tous les modèles fragmentés disponibles."""
    print("Test d'inférence pour les modèles fragmentés")
    print("=" * 60)
    
    # Trouver tous les dossiers de fragments
    models_dir = Path("models")
    fragment_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "fragments" in d.name]
    
    print(f"Modèles fragmentés trouvés: {len(fragment_dirs)}")
    for d in fragment_dirs:
        print(f"  - {d.name}")
    
    # Tester chaque modèle
    results = {}
    for model_dir in fragment_dirs:
        success = test_model_inference(model_dir.name)
        results[model_dir.name] = success
    
    # Résumé final
    print(f"\n{'='*60}")
    print("RÉSUMÉ DES TESTS D'INFÉRENCE")
    print(f"{'='*60}")
    
    total_models = len(results)
    successful = sum(1 for success in results.values() if success)
    
    print(f"Modèles testés: {total_models}")
    print(f"Succès: {successful}")
    print(f"Échecs: {total_models - successful}")
    
    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")
    
    if successful == total_models:
        print(f"\n🎉 Tous les modèles passent les tests d'inférence!")
        print(f"   Les modèles sont prêts pour l'inférence en production.")
    else:
        print(f"\n⚠️  Certains modèles ont échoué les tests d'inférence.")
        print(f"   Veuillez vérifier les erreurs ci-dessus.")
    
    return successful == total_models

if __name__ == "__main__":
    # Importer numpy pour le test
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)