#!/usr/bin/env python3
"""Test local inference with binary serialization to demonstrate real-world usage."""

import numpy as np
import sys
import os
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor
from distribution.server import ExecuteLayerResponse

def simulate_inference_with_serialization():
    """Simulate a realistic inference scenario with serialization."""
    print("=" * 60)
    print("Simulating Local Inference with Binary Serialization")
    print("=" * 60)
    
    # Simulate typical transformer layer activations
    # These would normally come from the transformer forward pass
    seq_len = 1
    dim = 5120  # Typical dimension for small models
    n_kv_heads = 40
    head_dim = 128
    
    print(f"Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Dimension: {dim}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    print()
    
    # Create realistic activation tensors
    hidden_state = np.random.randn(seq_len, dim).astype(np.float32)
    cache_k = np.random.randn(10, n_kv_heads, head_dim).astype(np.float32)  # past_len=10
    cache_v = np.random.randn(10, n_kv_heads, head_dim).astype(np.float32)
    
    print(f"Tensor sizes:")
    print(f"  Hidden state: {hidden_state.nbytes / 1024:.1f} KB")
    print(f"  Cache K: {cache_k.nbytes / 1024:.1f} KB")
    print(f"  Cache V: {cache_v.nbytes / 1024:.1f} KB")
    print(f"  Total: {(hidden_state.nbytes + cache_k.nbytes + cache_v.nbytes) / 1024:.1f} KB")
    print()
    
    # Test different serialization modes
    modes = [
        ("JSON (baseline)", False, False),
        ("Binary", True, False),
        ("Binary + Compression", True, True)
    ]
    
    results = []
    
    for mode_name, use_binary, use_compression in modes:
        print(f"Testing {mode_name}...")
        
        # Create client with specified mode
        client = RemoteLayerExecutor(
            "http://localhost:8000",
            layers=[0, 1, 2],
            use_binary=use_binary,
            use_compression=use_compression,
            verbose=False
        )
        
        # Measure serialization performance
        start_time = time.time()
        
        # Serialize all tensors (simulating network transmission)
        serialized_hs = client._serialize_array(hidden_state)
        serialized_k = client._serialize_array(cache_k)
        serialized_v = client._serialize_array(cache_v)
        
        serialization_time = time.time() - start_time
        
        # Calculate serialized size
        def get_serialized_size(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    return len(base64.b64decode(data["data"]))
                elif data.get("__binary__"):
                    return len(data["data"]) // 2  # hex encoding
                else:
                    return 0
            elif isinstance(data, list):
                # JSON list - estimate size
                import json
                return len(json.dumps(data))
            return 0
        
        size_hs = get_serialized_size(serialized_hs)
        size_k = get_serialized_size(serialized_k)
        size_v = get_serialized_size(serialized_v)
        total_size = size_hs + size_k + size_v
        
        # Deserialize to verify data integrity
        start_time = time.time()
        deserialized_hs = client._deserialize_array(serialized_hs, hidden_state.shape)
        deserialized_k = client._deserialize_array(serialized_k, cache_k.shape)
        deserialized_v = client._deserialize_array(serialized_v, cache_v.shape)
        deserialization_time = time.time() - start_time
        
        # Verify data integrity
        integrity_ok = (
            np.allclose(hidden_state, deserialized_hs) and
            np.allclose(cache_k, deserialized_k) and
            np.allclose(cache_v, deserialized_v)
        )
        
        results.append({
            "mode": mode_name,
            "size_kb": total_size / 1024,
            "serialization_ms": serialization_time * 1000,
            "deserialization_ms": deserialization_time * 1000,
            "total_ms": (serialization_time + deserialization_time) * 1000,
            "integrity": integrity_ok
        })
        
        print(f"  Serialized size: {total_size / 1024:.1f} KB")
        print(f"  Serialization time: {serialization_time * 1000:.2f} ms")
        print(f"  Deserialization time: {deserialization_time * 1000:.2f} ms")
        print(f"  Total time: {(serialization_time + deserialization_time) * 1000:.2f} ms")
        print(f"  Data integrity: {'PASS' if integrity_ok else 'FAIL'}")
        print()
    
    # Print comparison
    print("Performance Comparison:")
    print("-" * 60)
    print(f"{'Mode':<25} {'Size (KB)':>10} {'Total Time (ms)':>15} {'Integrity':>10}")
    print("-" * 60)
    
    baseline_size = results[0]["size_kb"]
    baseline_time = results[0]["total_ms"]
    
    for result in results:
        size_savings = ((baseline_size - result["size_kb"]) / baseline_size * 100) if baseline_size > 0 else 0
        time_change = ((result["total_ms"] - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
        
        print(f"{result['mode']:<25} {result['size_kb']:>10.1f} {result['total_ms']:>15.2f} {'PASS' if result['integrity'] else 'FAIL':>10}")
        if result['mode'] != "JSON (baseline)":
            print(f"  {'->':<25} {size_savings:>9.1f}% {'smaller' if size_savings > 0 else 'larger'} | {time_change:>14.1f}% {'faster' if time_change < 0 else 'slower'}")
    
    print("-" * 60)
    
    # Test server response simulation
    print(f"\nSimulating Server Response...")
    
    # Simulate what the server would return
    output_tensor = np.random.randn(seq_len, dim).astype(np.float32)
    new_k = np.random.randn(11, n_kv_heads, head_dim).astype(np.float32)  # past_len + 1
    new_v = np.random.randn(11, n_kv_heads, head_dim).astype(np.float32)
    
    # Server creates response
    response = ExecuteLayerResponse(output_tensor, new_k, new_v)
    
    # Client with binary support requests data
    binary_client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        verbose=False
    )
    
    # Server serializes response (in binary format)
    server_output = response._serialize_array(output_tensor)
    server_new_k = response._serialize_array(new_k)
    server_new_v = response._serialize_array(new_v)
    
    # Client deserializes response
    client_output = binary_client._deserialize_array(server_output, output_tensor.shape)
    client_new_k = binary_client._deserialize_array(server_new_k, new_k.shape)
    client_new_v = binary_client._deserialize_array(server_new_v, new_v.shape)
    
    # Verify round-trip integrity
    roundtrip_ok = (
        np.allclose(output_tensor, client_output) and
        np.allclose(new_k, client_new_k) and
        np.allclose(new_v, client_new_v)
    )
    
    print(f"Server -> Client round-trip: {'PASS' if roundtrip_ok else 'FAIL'}")
    
    # Summary
    all_passed = all(result["integrity"] for result in results) and roundtrip_ok
    
    print(f"\n{'=' * 60}")
    print(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print(f"{'=' * 60}")
    
    return all_passed

if __name__ == "__main__":
    success = simulate_inference_with_serialization()
    sys.exit(0 if success else 1)