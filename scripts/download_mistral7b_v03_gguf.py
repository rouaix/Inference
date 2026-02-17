#!/usr/bin/env python3
"""
Script pour télécharger Mistral-7B Instruct v0.3 au format GGUF.

Ce script télécharge le modèle Mistral-7B Instruct v0.3 quantifié au format GGUF,
qui est optimisé pour une utilisation efficace avec des outils comme llama.cpp.
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

# Configuration pour Mistral-7B Instruct v0.3
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.3-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.3.Q4_K_M.gguf"
DOWNLOAD_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
SAVE_DIR = Path("models/Mistral-7B-v0.3-GGUF")
CHUNK_SIZE = 8192

# Informations sur le fichier attendu
EXPECTED_SIZE = 4_372_812_000  # ~4.07 GB pour v0.3
EXPECTED_SHA256 = "1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6"  # Hash vérifié

def download_file(url, save_path, expected_size=None):
    """Télécharge un fichier avec une barre de progression."""
    try:
        # Créer le répertoire si nécessaire
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Téléchargement avec barre de progression
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=save_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                progress_bar.update(len(chunk))
        
        # Vérifier la taille du fichier
        actual_size = save_path.stat().st_size
        if expected_size and actual_size != expected_size:
            print(f"Avertissement : Taille du fichier inattendue "
                  f"({actual_size} octets vs {expected_size} octets attendus)")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"Echec du telechargement : {e}")
        return False

def verify_checksum(file_path, expected_hash):
    """Vérifie l'intégrité du fichier téléchargé."""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256_hash.update(chunk)
        
        actual_hash = sha256_hash.hexdigest()
        if actual_hash.lower() == expected_hash.lower():
            print(f"Verification du hash reussie : {actual_hash}")
            return True
        else:
            print(f"Echec de la verification du hash")
            print(f"   Attendu : {expected_hash}")
            print(f"   Obtenu  : {actual_hash}")
            return False
    except Exception as e:
        print(f"Erreur lors de la verification du hash : {e}")
        return False

def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("Telechargement de Mistral-7B Instruct v0.3 (GGUF)")
    print("=" * 60)
    
    # Vérifier l'espace disque (Windows compatible)
    try:
        import shutil
        free_space = shutil.disk_usage("/").free
        if free_space < EXPECTED_SIZE * 1.2:  # 20% de marge
            print(f"Erreur: Espace disque insuffisant")
            print(f"   Disponible : {free_space / (1024**3):.1f} GB")
            print(f"   Requis : {EXPECTED_SIZE / (1024**3):.1f} GB")
            return
    except Exception as e:
        print(f"Avertissement: Impossible de verifier l'espace disque : {e}")
    
    # Télécharger le fichier
    save_path = SAVE_DIR / MODEL_FILE
    print(f"Téléchargement de {DOWNLOAD_URL}")
    print(f"Enregistrement dans {save_path}")
    
    if download_file(DOWNLOAD_URL, save_path, EXPECTED_SIZE):
        print(f"\nTelechargement termine : {save_path}")
        
        # Vérifier le fichier
        if EXPECTED_SHA256:
            if verify_checksum(save_path, EXPECTED_SHA256):
                print("\nTelechargement et verification reussis !")
                print(f"\nLe modèle est prêt dans : {save_path}")
                print("\nProchaines étapes :")
                print("1. Intégrer le modèle à votre application")
                print("2. Configurer la quantification si nécessaire")
                print("3. Lancer les tests de validation")
            else:
                print("\nEchec de la verification du fichier")
        else:
            print("\nTelechargement termine (verification du hash desactivee)")
    else:
        print("\nEchec du telechargement")

if __name__ == "__main__":
    main()