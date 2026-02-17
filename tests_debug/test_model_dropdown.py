#!/usr/bin/env python3
"""
Test the model dropdown functionality.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from app import scan_fragment_dirs, format_dirs_table

def test_model_scanning():
    """Test that model scanning works correctly."""
    print("Testing model scanning functionality...")
    
    # Scan for models in the current directory
    models = scan_fragment_dirs(".")
    
    print(f"Found {len(models)} models:")
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Path: {model['path']}")
        print(f"   Fragments: {model['fragments']}")
        print(f"   Size: {model['size_mb']:.1f} MB")
        print(f"   Chunk size: {model['chunk_mb']:.1f} MB")
    
    # Test format_dirs_table function
    if models:
        table = format_dirs_table(models)
        print(f"\nFormatted table:\n{table}")
    
    # Test dropdown format
    choices = []
    for model in models:
        choices.append(f"{model['name']} ({model['path']})")
    
    print(f"\nDropdown choices:")
    for choice in choices:
        print(f"  - {choice}")
    
    return models

if __name__ == "__main__":
    try:
        models = test_model_scanning()
        if models:
            print(f"\n✅ Model scanning successful! Found {len(models)} models.")
        else:
            print("⚠️  No models found. This is expected if no fragment directories exist.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()