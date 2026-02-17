#!/usr/bin/env python3
"""
Test model scanning functionality without numpy dependency.
"""

import sys
import os
import json
from pathlib import Path

def scan_fragment_dirs_simple(base_dir: str) -> list:
    """Simple version of scan_fragment_dirs without numpy."""
    results = []
    base = Path(base_dir) if base_dir else Path(".")
    if not base.exists():
        return results
    
    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        manifest_path = item / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with open(manifest_path) as f:
                m = json.load(f)
            model_name = m.get("model_name", item.name)
            n_frags = m.get("total_fragments", len(m.get("fragments", [])))
            chunk_mb = m.get("chunk_size", 0) / (1024 ** 2)
            total_bytes = sum(
                fp.stat().st_size for fp in item.glob("*.dat") if fp.is_file()
            )
            results.append({
                "path": str(item.resolve()),
                "name": model_name,
                "fragments": n_frags,
                "size_mb": total_bytes / (1024 ** 2),
                "chunk_mb": chunk_mb,
            })
        except Exception as e:
            print(f"Error scanning {item}: {e}")
            results.append({
                "path": str(item.resolve()),
                "name": item.name,
                "fragments": 0,
                "size_mb": 0,
                "chunk_mb": 0,
            })
    return results

def test_model_scanning():
    """Test model scanning in different directories."""
    print("=" * 60)
    print("MODEL SCANNING TEST")
    print("=" * 60)
    
    # Test current directory
    print("\n🔍 Scanning current directory...")
    current_dirs = scan_fragment_dirs_simple(".")
    print(f"Found {len(current_dirs)} models in current directory")
    
    # Test models directory
    print("\n🔍 Scanning models/ directory...")
    models_dir = Path("models")
    if models_dir.exists():
        models_dirs = scan_fragment_dirs_simple(str(models_dir))
        print(f"Found {len(models_dirs)} models in models/ directory")
        
        for i, model in enumerate(models_dirs, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Path: {model['path']}")
            print(f"   Fragments: {model['fragments']}")
            print(f"   Size: {model['size_mb']:.1f} MB")
    else:
        print("models/ directory does not exist")
    
    # Test parent directory
    print("\n🔍 Scanning parent directory...")
    parent_dirs = scan_fragment_dirs_simple("..")
    print(f"Found {len(parent_dirs)} models in parent directory")
    
    # Summary
    total_models = len(current_dirs) + len(models_dirs) + len(parent_dirs)
    print(f"\n📊 SUMMARY:")
    print(f"   Current directory: {len(current_dirs)} models")
    print(f"   Models directory: {len(models_dirs)} models")
    print(f"   Parent directory: {len(parent_dirs)} models")
    print(f"   Total: {total_models} models found")
    
    if total_models == 0:
        print("\n⚠️  NO MODELS FOUND")
        print("   Make sure you have fragment directories with manifest.json files")
        print("   Expected structure:")
        print("   models/")
        print("     your_model_fragments/")
        print("       manifest.json")
        print("       fragment1.dat")
        print("       fragment2.dat")
        print("       ...")
    else:
        print(f"\n✅ SUCCESS: {total_models} models found!")
    
    return total_models > 0

if __name__ == "__main__":
    try:
        success = test_model_scanning()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)