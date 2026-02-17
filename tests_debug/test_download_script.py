#!/usr/bin/env python3
"""Test du script de téléchargement."""

import sys
from pathlib import Path

# Ajouter le chemin pour importer le script
sys.path.insert(0, str(Path(__file__).parent))

from scripts.download_mistral7b_v03_gguf import verify_checksum

def test_hash_verification():
    """Test la vérification du hash sur le fichier existant."""
    file_path = Path("models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
    expected_hash = "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6"
    
    print(f"Test de verification du hash pour {file_path}")
    
    if file_path.exists():
        if verify_checksum(file_path, expected_hash):
            print("Test passe : Hash verifie avec succes")
            return True
        else:
            print("Test echoue : Hash incorrect")
            return False
    else:
        print(f"Fichier non trouve : {file_path}")
        return False

if __name__ == "__main__":
    success = test_hash_verification()
    sys.exit(0 if success else 1)