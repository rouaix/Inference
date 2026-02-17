#!/usr/bin/env python3
"""Final verification test for binary serialization implementation."""

import numpy as np
import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor
from distribution.server import ExecuteLayerResponse

def test_all_modes():
    """Test all serialization modes."""
    print("Testing serialization implementation...")
    
    # Test data
    test_array = np.random.rand(10, 20).astype(np.float32)
    
    # Test 1: JSON mode
    print("Testing JSON mode...")
    client_json = RemoteLayerExecutor("http://localhost:8000", use_binary=False)
    serialized_json = client_json._serialize_array(test_array)
    deserialized_json = client_json._deserialize_array(serialized_json, test_array.shape)
    assert np.allclose(test_array, deserialized_json), "JSON mode failed"
    print("PASS: JSON mode working")
    
    # Test 2: Binary mode
    print("Testing binary mode...")
    client_binary = RemoteLayerExecutor("http://localhost:8000", use_binary=True, use_compression=False)
    serialized_binary = client_binary._serialize_array(test_array)
    deserialized_binary = client_binary._deserialize_array(serialized_binary, test_array.shape)
    assert np.allclose(test_array, deserialized_binary), "Binary mode failed"
    print("PASS: Binary mode working")
    
    # Test 3: Compression mode
    print("Testing compression mode...")
    client_compressed = RemoteLayerExecutor("http://localhost:8000", use_binary=True, use_compression=True)
    serialized_compressed = client_compressed._serialize_array(test_array)
    deserialized_compressed = client_compressed._deserialize_array(serialized_compressed, test_array.shape)
    assert np.allclose(test_array, deserialized_compressed), "Compression mode failed"
    print("PASS: Compression mode working")
    
    # Test 4: Server serialization
    print("Testing server serialization...")
    response = ExecuteLayerResponse(test_array, test_array, test_array)
    server_serialized = response._serialize_array(test_array)
    client_deserialized = client_binary._deserialize_array(server_serialized, test_array.shape)
    assert np.allclose(test_array, client_deserialized), "Server serialization failed"
    print("PASS: Server serialization working")
    
    # Test 5: Headers
    print("Testing headers...")
    headers = client_binary._build_headers()
    assert "Accept" in headers, "Headers missing Accept"
    assert "application/octet-stream" in headers["Accept"], "Accept header incorrect"
    print("PASS: Headers working")
    
    print("\nSUCCESS: All serialization tests passed!")
    print("Implementation is working correctly.")
    return True

if __name__ == "__main__":
    test_all_modes()