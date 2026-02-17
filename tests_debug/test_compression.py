#!/usr/bin/env python3
"""Test script for binary serialization with zstd compression."""

import numpy as np
import sys
import os
import base64

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def test_binary_serialization():
    """Test binary serialization without compression."""
    print("Testing binary serialization (no compression)...")
    
    # Create a test array
    test_array = np.random.rand(100, 200).astype(np.float32)
    original_size = test_array.nbytes
    print(f"Original array size: {original_size} bytes")
    print(f"[TEST] test_array is None: {test_array is None}")
    print(f"[TEST] test_array shape: {test_array.shape}")
    
    # Create a mock executor
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=False,
        verbose=True
    )
    
    # Test serialization
    print(f"[TEST] Calling _serialize_array...")
    try:
        serialized = executor._serialize_array(test_array)
        print(f"[TEST] Got serialized: {type(serialized)}")
    except Exception as e:
        print(f"[TEST] Exception in _serialize_array: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    if isinstance(serialized, dict) and serialized.get("__binary__"):
        # Test deserialization
        deserialized = executor._deserialize_array(serialized, test_array.shape)
        
        # Verify the data is the same
        if np.allclose(test_array, deserialized):
            print("[PASS] Binary serialization test PASSED")
            return True
        else:
            print("[FAIL] Binary serialization test FAILED")
            print(f"Max difference: {np.max(np.abs(test_array - deserialized))}")
            return False
    else:
        print("[FAIL] Serialization did not produce binary format")
        return False

def test_compression_serialization():
    """Test binary serialization with zstd compression."""
    print("\nTesting binary serialization with zstd compression...")
    
    try:
        import zstandard as zstd
        print("zstandard module is available")
    except ImportError:
        print("[SKIP] zstandard not available, skipping compression test")
        return True
    
    # Create a larger test array for better compression ratio
    test_array = np.random.rand(500, 1000).astype(np.float32)
    original_size = test_array.nbytes
    print(f"Original array size: {original_size} bytes")
    
    # Create a mock executor with compression
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        verbose=True
    )
    
    # Test serialization
    serialized = executor._serialize_array(test_array)
    
    if isinstance(serialized, dict) and serialized.get("__binary_zstd__"):
        # Calculate compression ratio
        compressed_data = base64.b64decode(serialized["data"])
        compressed_size = len(compressed_data)
        ratio = compressed_size / original_size * 100
        print(f"Compressed size: {compressed_size} bytes ({ratio:.1f}% of original)")
        
        # Test deserialization
        deserialized = executor._deserialize_array(serialized, test_array.shape)
        
        # Verify the data is the same
        if np.allclose(test_array, deserialized):
            print(f"[PASS] Compression test PASSED (ratio: {ratio:.1f}%)")
            return True
        else:
            print("[FAIL] Compression test FAILED - data mismatch")
            print(f"Max difference: {np.max(np.abs(test_array - deserialized))}")
            return False
    else:
        print("[FAIL] Serialization did not produce compressed format")
        return False

def test_fallback_to_binary():
    """Test that compression falls back to binary when zstd is unavailable."""
    print("\nTesting fallback to binary when compression fails...")
    
    # Temporarily disable zstd
    import distribution.reseau as reseau_module
    original_zstd = reseau_module.zstd
    reseau_module.zstd = None
    reseau_module._ZSTD_AVAILABLE = False
    
    try:
        # Create a test array
        test_array = np.random.rand(50, 100).astype(np.float32)
        
        # Create a mock executor with compression (should fall back to binary)
        executor = RemoteLayerExecutor(
            "http://localhost:8000",
            layers=[0, 1, 2],
            use_binary=True,
            use_compression=True,  # Should fall back to binary
            verbose=True
        )
        
        # Test serialization
        serialized = executor._serialize_array(test_array)
        
        # Should fall back to binary format
        if isinstance(serialized, dict) and serialized.get("__binary__"):
            # Test deserialization
            deserialized = executor._deserialize_array(serialized, test_array.shape)
            
            # Verify the data is the same
            if np.allclose(test_array, deserialized):
                print("[PASS] Fallback test PASSED")
                return True
            else:
                print("[FAIL] Fallback test FAILED - data mismatch")
                return False
        else:
            print("[FAIL] Fallback did not produce binary format")
            return False
    finally:
        # Restore original zstd
        reseau_module.zstd = original_zstd
        reseau_module._ZSTD_AVAILABLE = original_zstd is not None

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Binary Serialization with Compression")
    print("=" * 60)
    
    results = []
    results.append(test_binary_serialization())
    results.append(test_compression_serialization())
    results.append(test_fallback_to_binary())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] All compression tests PASSED")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some compression tests FAILED")
        sys.exit(1)