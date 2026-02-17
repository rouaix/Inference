#!/usr/bin/env python3
"""Integration tests with existing fragmentation system."""

import numpy as np
import sys
import os
import json
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor
from inference.p2p_inference import P2PInferenceEngine

def test_fragment_system_integration():
    """Test integration with the fragment system."""
    print("=" * 60)
    print("Testing Fragment System Integration")
    print("=" * 60)
    
    # Load the inference engine with fragments
    fragments_dir = os.path.join(project_root, "models", "Magistral-Small-2509-Q4_K_M_fragments")
    
    try:
        engine = P2PInferenceEngine(fragments_dir, verbose=True)
        print("PASS: Inference engine loaded successfully")
    except Exception as e:
        print(f"FAIL: Failed to load inference engine: {e}")
        return False
    
    # Create a remote executor that simulates network communication
    remote_executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=list(range(engine.config.n_layers)),
        use_binary=True,
        use_compression=True,
        collect_metrics=True,
        verbose=True
    )
    
    print(f"\nSimulating distributed inference with {engine.config.n_layers} layers...")
    
    # Simulate typical transformer tensors that would be sent over network
    seq_len = 5
    dim = engine.config.dim
    n_kv_heads = engine.config.n_kv_heads
    head_dim = dim // engine.config.n_heads
    
    # Create realistic activation tensors
    hidden_state = np.random.randn(seq_len, dim).astype(np.float32)
    cache_k = np.random.randn(10, n_kv_heads, head_dim).astype(np.float32)  # past_len=10
    cache_v = np.random.randn(10, n_kv_heads, head_dim).astype(np.float32)
    
    print(f"Tensor shapes:")
    print(f"  Hidden state: {hidden_state.shape} ({hidden_state.nbytes / 1024:.1f} KB)")
    print(f"  Cache K: {cache_k.shape} ({cache_k.nbytes / 1024:.1f} KB)")
    print(f"  Cache V: {cache_v.shape} ({cache_v.nbytes / 1024:.1f} KB)")
    
    # Simulate sending data to remote nodes for each layer
    layers_processed = 0
    total_network_data = 0
    
    for layer_idx in range(min(3, engine.config.n_layers)):  # Test first 3 layers
        print(f"\nProcessing layer {layer_idx}...")
        
        # Serialize tensors as they would be sent to remote node
        start_time = time.time()
        
        serialized_hs = remote_executor._serialize_array(hidden_state)
        serialized_k = remote_executor._serialize_array(cache_k)
        serialized_v = remote_executor._serialize_array(cache_v)
        
        serialization_time = time.time() - start_time
        
        # Calculate network data size
        def get_size(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    return len(base64.b64decode(data["data"]))
                elif data.get("__binary__"):
                    return len(data["data"]) // 2
            return 0
        
        layer_data_size = (
            get_size(serialized_hs) + 
            get_size(serialized_k) + 
            get_size(serialized_v)
        )
        total_network_data += layer_data_size
        
        # Simulate remote processing (would normally happen on remote node)
        # In this test, we just deserialize to verify the round-trip
        deserialized_hs = remote_executor._deserialize_array(serialized_hs, hidden_state.shape)
        deserialized_k = remote_executor._deserialize_array(serialized_k, cache_k.shape)
        deserialized_v = remote_executor._deserialize_array(serialized_v, cache_v.shape)
        
        # Verify integrity
        integrity_ok = (
            np.allclose(hidden_state, deserialized_hs) and
            np.allclose(cache_k, deserialized_k) and
            np.allclose(cache_v, deserialized_v)
        )
        
        if integrity_ok:
            print(f"  Serialization: {serialization_time*1000:.3f} ms")
            print(f"  Network data: {layer_data_size/1024:.1f} KB")
            print(f"  Integrity: PASS")
            layers_processed += 1
        else:
            print(f"  Integrity: FAIL")
            break
        
        # Simulate layer output (would come from remote node)
        # In a real scenario, this would be the result of the transformer layer
        output = np.random.randn(seq_len, dim).astype(np.float32)
        new_k = np.random.randn(11, n_kv_heads, head_dim).astype(np.float32)  # past_len + 1
        new_v = np.random.randn(11, n_kv_heads, head_dim).astype(np.float32)
        
        # Update for next iteration
        hidden_state = output
        cache_k = new_k
        cache_v = new_v
    
    # Get metrics
    metrics = remote_executor.get_metrics()
    
    print(f"\n{'=' * 40}")
    print("Integration Test Results:")
    print(f"  Layers processed: {layers_processed}")
    print(f"  Total network data: {total_network_data/1024:.1f} KB")
    print(f"  Average per layer: {total_network_data/layers_processed/1024:.1f} KB")
    print(f"  Compression ratio: {metrics.get('compression_ratio', 0)*100:.1f}%")
    print(f"  Bandwidth savings: {metrics.get('bandwidth_savings_percentage', 0):.1f}%")
    
    success = layers_processed == 3 and metrics.get('serialization_count', 0) == 9
    
    print(f"{'=' * 40}")
    print(f"Fragment System Integration: {'PASS' if success else 'FAIL'}")
    
    return success

def test_kv_cache_integration():
    """Test integration with KV cache system."""
    print("\n" + "=" * 60)
    print("Testing KV Cache Integration")
    print("=" * 60)
    
    # Create remote executor
    remote_executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Simulate KV cache growth during inference
    seq_len = 1
    dim = 5120
    n_kv_heads = 40
    head_dim = 128
    
    print("Simulating KV cache growth over multiple tokens...")
    
    # Start with empty cache
    cache_k = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
    cache_v = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
    
    results = []
    
    for token_idx in range(10):  # Simulate 10 tokens
        # Create current token activation
        hidden_state = np.random.randn(seq_len, dim).astype(np.float32)
        
        # Serialize current state
        serialized_hs = remote_executor._serialize_array(hidden_state)
        serialized_k = remote_executor._serialize_array(cache_k)
        serialized_v = remote_executor._serialize_array(cache_v)
        
        # Calculate sizes
        def get_size(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    return len(base64.b64decode(data["data"]))
                elif data.get("__binary__"):
                    return len(data["data"]) // 2
            return 0
        
        total_size = get_size(serialized_hs) + get_size(serialized_k) + get_size(serialized_v)
        
        # Deserialize to verify
        deserialized_hs = remote_executor._deserialize_array(serialized_hs, hidden_state.shape)
        deserialized_k = remote_executor._deserialize_array(serialized_k, cache_k.shape)
        deserialized_v = remote_executor._deserialize_array(serialized_v, cache_v.shape)
        
        integrity_ok = (
            np.allclose(hidden_state, deserialized_hs) and
            np.allclose(cache_k, deserialized_k) and
            np.allclose(cache_v, deserialized_v)
        )
        
        results.append({
            'token': token_idx,
            'cache_size': cache_k.nbytes + cache_v.nbytes,
            'serialized_size': total_size,
            'integrity': integrity_ok
        })
        
        print(f"  Token {token_idx}: Cache size = {cache_k.shape[0]} tokens, "
              f"Data size = {total_size/1024:.1f} KB, "
              f"Integrity = {'PASS' if integrity_ok else 'FAIL'}")
        
        # Simulate cache update (add current token to cache)
        new_k_token = np.random.randn(seq_len, n_kv_heads, head_dim).astype(np.float32)
        new_v_token = np.random.randn(seq_len, n_kv_heads, head_dim).astype(np.float32)
        
        cache_k = np.concatenate([cache_k, new_k_token], axis=0)
        cache_v = np.concatenate([cache_v, new_v_token], axis=0)
        
        if not integrity_ok:
            break
    
    # Print summary
    print(f"\n{'=' * 40}")
    print("KV Cache Growth Summary:")
    print(f"{'Token':<6} {'Cache Size':<12} {'Serialized':<12} {'Integrity':<10}")
    print("-" * 40)
    
    for result in results:
        print(f"{result['token']:<6} {result['cache_size']/1024:<12.1f} "
              f"{result['serialized_size']/1024:<12.1f} "
              f"{'PASS' if result['integrity'] else 'FAIL':<10}")
    
    # Check that all tokens passed
    all_passed = all(r['integrity'] for r in results)
    
    print(f"{'=' * 40}")
    print(f"KV Cache Integration: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_distributed_workflow():
    """Test complete distributed inference workflow."""
    print("\n" + "=" * 60)
    print("Testing Complete Distributed Workflow")
    print("=" * 60)
    
    # Create client and server executors
    client_executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Simulate a complete inference workflow
    print("Simulating complete distributed inference workflow...")
    
    # Phase 1: Prefill (process all prompt tokens)
    print("\n1. Prefill phase (processing prompt)...")
    
    # Simulate prompt tokens
    prompt_tokens = 5
    seq_len = prompt_tokens
    dim = 5120
    
    # Initial empty cache
    cache_k = np.zeros((0, 40, 128), dtype=np.float32)
    cache_v = np.zeros((0, 40, 128), dtype=np.float32)
    
    # Process each prompt token
    for token_idx in range(prompt_tokens):
        hidden_state = np.random.randn(1, dim).astype(np.float32)
        
        # Serialize and send to remote node
        serialized_hs = client_executor._serialize_array(hidden_state)
        serialized_k = client_executor._serialize_array(cache_k)
        serialized_v = client_executor._serialize_array(cache_v)
        
        # Deserialize on remote node (simulated)
        deserialized_hs = client_executor._deserialize_array(serialized_hs, hidden_state.shape)
        deserialized_k = client_executor._deserialize_array(serialized_k, cache_k.shape)
        deserialized_v = client_executor._deserialize_array(serialized_v, cache_v.shape)
        
        # Verify integrity
        if not (np.allclose(hidden_state, deserialized_hs) and
                np.allclose(cache_k, deserialized_k) and
                np.allclose(cache_v, deserialized_v)):
            print(f"  Token {token_idx}: FAIL")
            return False
        
        # Simulate layer processing and cache update
        output = np.random.randn(1, dim).astype(np.float32)
        new_k = np.random.randn(token_idx + 1, 40, 128).astype(np.float32)
        new_v = np.random.randn(token_idx + 1, 40, 128).astype(np.float32)
        
        # Update for next iteration
        hidden_state = output
        cache_k = new_k
        cache_v = new_v
    
    print(f"  Processed {prompt_tokens} prompt tokens: PASS")
    
    # Phase 2: Decode (generate tokens one by one)
    print("\n2. Decode phase (generating tokens)...")
    
    generated_tokens = 3
    
    for token_idx in range(generated_tokens):
        hidden_state = np.random.randn(1, dim).astype(np.float32)
        
        # Serialize and send to remote node
        serialized_hs = client_executor._serialize_array(hidden_state)
        serialized_k = client_executor._serialize_array(cache_k)
        serialized_v = client_executor._serialize_array(cache_v)
        
        # Deserialize on remote node (simulated)
        deserialized_hs = client_executor._deserialize_array(serialized_hs, hidden_state.shape)
        deserialized_k = client_executor._deserialize_array(serialized_k, cache_k.shape)
        deserialized_v = client_executor._deserialize_array(serialized_v, cache_v.shape)
        
        # Verify integrity
        if not (np.allclose(hidden_state, deserialized_hs) and
                np.allclose(cache_k, deserialized_k) and
                np.allclose(cache_v, deserialized_v)):
            print(f"  Token {token_idx}: FAIL")
            return False
        
        # Simulate layer processing and cache update
        output = np.random.randn(1, dim).astype(np.float32)
        new_k = np.random.randn(cache_k.shape[0] + 1, 40, 128).astype(np.float32)
        new_v = np.random.randn(cache_v.shape[0] + 1, 40, 128).astype(np.float32)
        
        # Update for next iteration
        hidden_state = output
        cache_k = new_k
        cache_v = new_v
    
    print(f"  Generated {generated_tokens} tokens: PASS")
    
    print(f"\n{'=' * 40}")
    print("Complete Distributed Workflow: PASS")
    
    return True

if __name__ == "__main__":
    print("Testing System Integration")
    print("=" * 60)
    
    results = []
    results.append(test_fragment_system_integration())
    results.append(test_kv_cache_integration())
    results.append(test_distributed_workflow())
    
    print(f"\n{'=' * 60}")
    print("Integration Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All integration tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some integration tests failed!")
        sys.exit(1)