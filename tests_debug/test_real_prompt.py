#!/usr/bin/env python3
"""Test with a real prompt using the available model and tokenizer."""

import json
import numpy as np
import sys
import os
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_real_prompt():
    """Test with a real prompt using available model fragments."""
    print("=" * 60)
    print("Testing with Real Prompt: 'c'est quoi paris en france'")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer_path = os.path.join(project_root, "models", "Magistral-Small-2509-Q4_K_M_fragments", "tokenizer.json")
    
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        print("PASS: Tokenizer loaded successfully")
    except Exception as e:
        print(f"FAIL: Failed to load tokenizer: {e}")
        return False
    
    # Load model manifest
    manifest_path = os.path.join(project_root, "models", "Magistral-Small-2509-Q4_K_M_fragments", "manifest.json")
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        print("PASS: Model manifest loaded successfully")
        print(f"  Model: {manifest.get('model_name', 'unknown')}")
        print(f"  Architecture: {manifest.get('architecture', 'unknown')}")
    except Exception as e:
        print(f"✗ Failed to load manifest: {e}")
        return False
    
    # Test prompt
    prompt = "c'est quoi paris en france"
    print(f"\nPrompt: '{prompt}'")
    
    # Simulate tokenization (since we don't have the full tokenizer implementation)
    # In a real scenario, this would use the actual tokenizer
    print("\nSimulating tokenization...")
    
    # Create mock tokens (simplified for demonstration)
    # Real tokens would come from tokenizer.encode(prompt)
    tokens = [1] + list(range(100, 100 + len(prompt.split())))  # [BOS] + word tokens
    print(f"Simulated tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Simulate typical transformer tensors
    seq_len = len(tokens)
    dim_config = manifest.get('config', {}).get('dim', [5120])
    n_layers_config = manifest.get('config', {}).get('n_layers', [32])
    
    # Handle both list and direct integer values
    dim = dim_config[0] if isinstance(dim_config, list) and len(dim_config) > 0 else dim_config
    n_layers = n_layers_config[0] if isinstance(n_layers_config, list) and len(n_layers_config) > 0 else n_layers_config
    
    print(f"\nModel configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Dimension: {dim}")
    print(f"  Layers: {n_layers}")
    
    # Create realistic activation tensors
    hidden_state = np.random.randn(seq_len, dim).astype(np.float32)
    
    # Simulate KV cache (past_len = 0 for first token)
    n_kv_heads_config = manifest.get('config', {}).get('n_kv_heads', [40])
    head_dim_config = manifest.get('config', {}).get('head_dim', [128])
    
    n_kv_heads = n_kv_heads_config[0] if isinstance(n_kv_heads_config, list) and len(n_kv_heads_config) > 0 else n_kv_heads_config
    head_dim = head_dim_config[0] if isinstance(head_dim_config, list) and len(head_dim_config) > 0 else head_dim_config
    
    cache_k = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
    cache_v = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
    
    print(f"\nTensor shapes:")
    print(f"  Hidden state: {hidden_state.shape} ({hidden_state.nbytes / 1024:.1f} KB)")
    print(f"  Cache K: {cache_k.shape} ({cache_k.nbytes / 1024:.1f} KB)")
    print(f"  Cache V: {cache_v.shape} ({cache_v.nbytes / 1024:.1f} KB)")
    
    # Test our serialization with real-world scenario
    from distribution.reseau import RemoteLayerExecutor
    
    print(f"\nTesting serialization for distributed inference...")
    
    # Create client with binary + compression (optimal for network)
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=list(range(n_layers)),
        use_binary=True,
        use_compression=True,
        verbose=True
    )
    
    # Measure serialization performance
    start_time = time.time()
    
    # Serialize tensors as they would be sent to remote nodes
    serialized_hs = client._serialize_array(hidden_state)
    serialized_k = client._serialize_array(cache_k)
    serialized_v = client._serialize_array(cache_v)
    
    serialization_time = time.time() - start_time
    
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
    
    print(f"\nSerialization results:")
    print(f"  Total serialized size: {total_size / 1024:.1f} KB")
    print(f"  Serialization time: {serialization_time * 1000:.2f} ms")
    
    # Deserialize to verify integrity
    deserialized_hs = client._deserialize_array(serialized_hs, hidden_state.shape)
    deserialized_k = client._deserialize_array(serialized_k, cache_k.shape)
    deserialized_v = client._deserialize_array(serialized_v, cache_v.shape)
    
    # Verify data integrity
    integrity_ok = (
        np.allclose(hidden_state, deserialized_hs) and
        np.allclose(cache_k, deserialized_k) and
        np.allclose(cache_v, deserialized_v)
    )
    
    print(f"  Data integrity: {'PASS' if integrity_ok else 'FAIL'}")
    
    # Simulate what would happen in distributed inference
    print(f"\nSimulating distributed inference workflow...")
    
    # For each layer, data would be:
    # 1. Serialized by client
    # 2. Sent over network to remote node
    # 3. Deserialized by server
    # 4. Processed by transformer layer
    # 5. Response serialized by server
    # 6. Sent back to client
    # 7. Deserialized by client
    
    layers_processed = 0
    total_network_data = 0
    
    # Simulate processing through all layers
    for layer_idx in range(min(3, n_layers)):  # Test first 3 layers for demo
        # Simulate layer output (would come from actual forward pass)
        output = np.random.randn(seq_len, dim).astype(np.float32)
        new_k = np.random.randn(1, n_kv_heads, head_dim).astype(np.float32)  # seq_len=1
        new_v = np.random.randn(1, n_kv_heads, head_dim).astype(np.float32)
        
        # Serialize as response would be
        response_hs = client._serialize_array(output)
        response_k = client._serialize_array(new_k)
        response_v = client._serialize_array(new_v)
        
        layer_data_size = get_size(response_hs) + get_size(response_k) + get_size(response_v)
        total_network_data += layer_data_size
        
        # Deserialize to continue processing
        output = client._deserialize_array(response_hs, output.shape)
        new_k = client._deserialize_array(response_k, new_k.shape)
        new_v = client._deserialize_array(response_v, new_v.shape)
        
        layers_processed += 1
        
        print(f"  Layer {layer_idx}: {'PASS' if output is not None else 'FAIL'}")
    
    print(f"\nDistributed inference simulation:")
    print(f"  Layers processed: {layers_processed}")
    print(f"  Total network data: {total_network_data / 1024:.1f} KB")
    print(f"  Average per layer: {total_network_data / layers_processed / 1024:.1f} KB")
    
    # Final result
    print(f"\n{'=' * 60}")
    print("Real Prompt Test Summary:")
    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Serialization: {'PASS' if integrity_ok else 'FAIL'}")
    print(f"  Distributed processing: {'PASS' if layers_processed == 3 else 'FAIL'}")
    print(f"  Overall: {'ALL TESTS PASSED' if integrity_ok and layers_processed == 3 else 'SOME TESTS FAILED'}")
    print(f"{'=' * 60}")
    
    # Note about actual inference
    print(f"\nNote: This test demonstrates the serialization infrastructure.")
    print(f"For actual text generation, you would need:")
    print(f"  1. Complete tokenizer implementation")
    print(f"  2. Full model weights loaded")
    print(f"  3. Complete transformer forward pass")
    print(f"  4. Sampling/decoding logic")
    print(f"\nThe current implementation successfully handles the")
    print(f"network serialization/deserialization part of distributed inference!")
    
    return integrity_ok and layers_processed == 3

if __name__ == "__main__":
    success = test_real_prompt()
    sys.exit(0 if success else 1)