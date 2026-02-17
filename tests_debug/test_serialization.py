#!/usr/bin/env python3
"""Test script for binary serialization functionality."""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distribution.reseau import RemoteLayerExecutor

def test_binary_serialization():
    """Test binary serialization and deserialization."""
    print("Testing binary serialization...")
    
    # Create a test array
    test_array = np.random.rand(10, 20).astype(np.float32)
    print(f"Original array shape: {test_array.shape}, dtype: {test_array.dtype}")
    
    # Create a mock executor
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        verbose=True
    )
    
    # Test serialization
    serialized = executor._serialize_array(test_array)
    print(f"Serialized type: {type(serialized)}")
    print(f"Serialized keys: {serialized.keys() if isinstance(serialized, dict) else 'N/A'}")
    
    # Test deserialization
    deserialized = executor._deserialize_array(serialized, test_array.shape)
    print(f"Deserialized array shape: {deserialized.shape}, dtype: {deserialized.dtype}")
    
    # Verify the data is the same
    if np.allclose(test_array, deserialized):
        print("[PASS] Binary serialization test PASSED")
        return True
    else:
        print("[FAIL] Binary serialization test FAILED")
        print(f"Max difference: {np.max(np.abs(test_array - deserialized))}")
        return False

def test_json_serialization():
    """Test JSON serialization (fallback)."""
    print("\nTesting JSON serialization...")
    
    # Create a test array
    test_array = np.random.rand(5, 10).astype(np.float32)
    print(f"Original array shape: {test_array.shape}, dtype: {test_array.dtype}")
    
    # Create a mock executor with JSON mode
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=False,
        verbose=True
    )
    
    # Test serialization
    serialized = executor._serialize_array(test_array)
    print(f"Serialized type: {type(serialized)}")
    
    # Test deserialization
    deserialized = executor._deserialize_array(serialized, test_array.shape)
    print(f"Deserialized array shape: {deserialized.shape}, dtype: {deserialized.dtype}")
    
    # Verify the data is the same
    if np.allclose(test_array, deserialized):
        print("[PASS] JSON serialization test PASSED")
        return True
    else:
        print("[FAIL] JSON serialization test FAILED")
        return False

def test_none_serialization():
    """Test serialization of None values."""
    print("\nTesting None serialization...")
    
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True
    )
    
    # Test None serialization
    serialized = executor._serialize_array(None)
    if serialized is None:
        print("[PASS] None serialization test PASSED")
        return True
    else:
        print("[FAIL] None serialization test FAILED")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Serialization Functionality")
    print("=" * 60)
    
    results = []
    results.append(test_binary_serialization())
    results.append(test_json_serialization())
    results.append(test_none_serialization())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] All serialization tests PASSED")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some serialization tests FAILED")
        sys.exit(1)