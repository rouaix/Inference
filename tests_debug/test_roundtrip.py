#!/usr/bin/env python3
"""Test script for complete round-trip serialization/deserialization."""

import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor
from distribution.server import ExecuteLayerResponse

def test_roundtrip():
    """Test complete round-trip: client serialize → server deserialize → server serialize → client deserialize."""
    print("Testing complete round-trip serialization...")
    
    # Create test data
    original_data = np.random.rand(50, 100).astype(np.float32)
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data size: {original_data.nbytes} bytes")
    
    # Step 1: Client serializes (binary mode)
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=False,
        verbose=True
    )
    
    serialized = client._serialize_array(original_data)
    print(f"Client serialized to: {type(serialized)}")
    
    # Step 2: Server deserializes
    def server_deserialize(data):
        """Simulate server deserialization."""
        if isinstance(data, dict):
            if data.get("__binary_zstd__"):
                import base64
                import zstandard as zstd
                compressed_data = base64.b64decode(data["data"])
                decompressed = zstd.ZstdDecompressor().decompress(compressed_data)
                return np.frombuffer(decompressed, dtype=data["dtype"]).reshape(data["shape"])
            elif data.get("__binary__"):
                import binascii
                bytes_data = binascii.unhexlify(data["data"])
                return np.frombuffer(bytes_data, dtype=data["dtype"]).reshape(data["shape"])
        elif isinstance(data, list):
            return np.array(data, dtype=np.float32)
        return None
    
    server_data = server_deserialize(serialized)
    print(f"Server deserialized shape: {server_data.shape}")
    
    # Step 3: Server serializes (binary response)
    response = ExecuteLayerResponse(server_data, server_data, server_data)
    server_serialized = {
        "output": response._serialize_array(server_data),
        "new_k": response._serialize_array(server_data),
        "new_v": response._serialize_array(server_data)
    }
    
    # Step 4: Client deserializes
    client_output = client._deserialize_array(server_serialized["output"], original_data.shape)
    client_new_k = client._deserialize_array(server_serialized["new_k"], original_data.shape)
    client_new_v = client._deserialize_array(server_serialized["new_v"], original_data.shape)
    
    # Verify all data matches
    if (np.allclose(original_data, client_output) and
        np.allclose(original_data, client_new_k) and
        np.allclose(original_data, client_new_v)):
        print("[PASS] Round-trip test PASSED - all data matches")
        return True
    else:
        print("[FAIL] Round-trip test FAILED - data mismatch")
        print(f"Max difference in output: {np.max(np.abs(original_data - client_output))}")
        print(f"Max difference in new_k: {np.max(np.abs(original_data - client_new_k))}")
        print(f"Max difference in new_v: {np.max(np.abs(original_data - client_new_v))}")
        return False

def test_compression_roundtrip():
    """Test complete round-trip with compression."""
    print("\nTesting complete round-trip with compression...")
    
    try:
        import zstandard as zstd
        print("zstandard module is available")
    except ImportError:
        print("[SKIP] zstandard not available, skipping compression round-trip test")
        return True
    
    # Create larger test data for better compression
    original_data = np.random.rand(200, 300).astype(np.float32)
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data size: {original_data.nbytes} bytes")
    
    # Step 1: Client serializes (compressed mode)
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        verbose=True
    )
    
    serialized = client._serialize_array(original_data)
    print(f"Client serialized to: {type(serialized)}")
    
    # Step 2: Server deserializes
    def server_deserialize(data):
        """Simulate server deserialization."""
        if isinstance(data, dict):
            if data.get("__binary_zstd__"):
                import base64
                import zstandard as zstd
                compressed_data = base64.b64decode(data["data"])
                decompressed = zstd.ZstdDecompressor().decompress(compressed_data)
                return np.frombuffer(decompressed, dtype=data["dtype"]).reshape(data["shape"])
            elif data.get("__binary__"):
                import binascii
                bytes_data = binascii.unhexlify(data["data"])
                return np.frombuffer(bytes_data, dtype=data["dtype"]).reshape(data["shape"])
        elif isinstance(data, list):
            return np.array(data, dtype=np.float32)
        return None
    
    server_data = server_deserialize(serialized)
    print(f"Server deserialized shape: {server_data.shape}")
    
    # Step 3: Server serializes (binary response)
    response = ExecuteLayerResponse(server_data, server_data, server_data)
    server_serialized = {
        "output": response._serialize_array(server_data),
        "new_k": response._serialize_array(server_data),
        "new_v": response._serialize_array(server_data)
    }
    
    # Step 4: Client deserializes
    client_output = client._deserialize_array(server_serialized["output"], original_data.shape)
    client_new_k = client._deserialize_array(server_serialized["new_k"], original_data.shape)
    client_new_v = client._deserialize_array(server_serialized["new_v"], original_data.shape)
    
    # Verify all data matches
    if (np.allclose(original_data, client_output) and
        np.allclose(original_data, client_new_k) and
        np.allclose(original_data, client_new_v)):
        print("[PASS] Compression round-trip test PASSED - all data matches")
        return True
    else:
        print("[FAIL] Compression round-trip test FAILED - data mismatch")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Complete Round-Trip Serialization")
    print("=" * 60)
    
    results = []
    results.append(test_roundtrip())
    results.append(test_compression_roundtrip())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] All round-trip tests PASSED")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some round-trip tests FAILED")
        sys.exit(1)