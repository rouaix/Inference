#!/usr/bin/env python3
"""Test KV cache optimization with different serialization strategies."""

import numpy as np
import sys
import os
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def test_kv_cache_strategies():
    """Test different KV cache serialization strategies."""
    print("=" * 60)
    print("Testing KV Cache Serialization Strategies")
    print("=" * 60)
    
    # Test parameters
    max_tokens = 20
    dim = 5120
    n_kv_heads = 40
    head_dim = 128
    
    print(f"Configuration:")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Dimension: {dim}")
    print(f"  KV heads: {n_kv_heads}")
    print(f"  Head dimension: {head_dim}")
    
    # Test different strategies
    strategies = [
        ("JSON", False, False),
        ("Binary", True, False),
        ("Binary + Compression", True, True),
    ]
    
    results = []
    
    for strategy_name, use_binary, use_compression in strategies:
        print(f"\n{strategy_name} Strategy:")
        print("-" * 40)
        
        client = RemoteLayerExecutor(
            "http://localhost:8000",
            layers=[0, 1, 2],
            use_binary=use_binary,
            use_compression=use_compression,
            collect_metrics=False,
            verbose=False
        )
        
        # Simulate KV cache growth
        cache_k = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
        cache_v = np.zeros((0, n_kv_heads, head_dim), dtype=np.float32)
        
        strategy_results = []
        
        for token_idx in range(max_tokens):
            # Current token activation
            hidden_state = np.random.randn(1, dim).astype(np.float32)
            
            # Serialize
            start_time = time.time()
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
                elif isinstance(data, list):
                    return len(str(data))
                return 0
            
            total_size = get_size(serialized_hs) + get_size(serialized_k) + get_size(serialized_v)
            
            # Deserialize
            deserialized_hs = client._deserialize_array(serialized_hs, hidden_state.shape)
            deserialized_k = client._deserialize_array(serialized_k, cache_k.shape)
            deserialized_v = client._deserialize_array(serialized_v, cache_v.shape)
            
            # Handle empty tensors
            hs_ok = np.allclose(hidden_state, deserialized_hs) if hidden_state.size > 0 else deserialized_hs.size == 0
            k_ok = np.allclose(cache_k, deserialized_k) if cache_k.size > 0 else deserialized_k.size == 0
            v_ok = np.allclose(cache_v, deserialized_v) if cache_v.size > 0 else deserialized_v.size == 0
            
            integrity_ok = hs_ok and k_ok and v_ok
            
            strategy_results.append({
                'token': token_idx,
                'cache_size': cache_k.nbytes + cache_v.nbytes,
                'serialized_size': total_size,
                'serialization_time': serialization_time,
                'integrity': integrity_ok
            })
            
            # Update cache
            new_k = np.random.randn(1, n_kv_heads, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_kv_heads, head_dim).astype(np.float32)
            cache_k = np.concatenate([cache_k, new_k], axis=0)
            cache_v = np.concatenate([cache_v, new_v], axis=0)
        
        # Calculate strategy metrics
        total_original_size = sum(r['cache_size'] for r in strategy_results)
        total_serialized_size = sum(r['serialized_size'] for r in strategy_results)
        total_time = sum(r['serialization_time'] for r in strategy_results)
        
        compression_ratio = total_serialized_size / total_original_size if total_original_size > 0 else 0
        avg_time_per_token = total_time / max_tokens * 1000  # ms
        
        results.append({
            'strategy': strategy_name,
            'total_original_size': total_original_size,
            'total_serialized_size': total_serialized_size,
            'compression_ratio': compression_ratio,
            'total_time': total_time,
            'avg_time_per_token': avg_time_per_token,
            'all_integrity_ok': all(r['integrity'] for r in strategy_results)
        })
        
        print(f"  Total original size: {total_original_size/1024:.1f} KB")
        print(f"  Total serialized size: {total_serialized_size/1024:.1f} KB")
        print(f"  Compression ratio: {compression_ratio*100:.1f}%")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg time per token: {avg_time_per_token:.3f}ms")
        print(f"  Integrity: {'PASS' if all(r['integrity'] for r in strategy_results) else 'FAIL'}")
    
    # Print comparison
    print(f"\n{'=' * 60}")
    print("KV Cache Strategy Comparison:")
    print(f"{'Strategy':<20} {'Size Ratio':<12} {'Avg Time':<12} {'Integrity':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['strategy']:<20} {result['compression_ratio']*100:<12.1f}% "
              f"{result['avg_time_per_token']:<12.3f}ms {'PASS' if result['all_integrity_ok'] else 'FAIL':<10}")
    
    # Calculate improvements
    json_result = next(r for r in results if r['strategy'] == 'JSON')
    binary_result = next(r for r in results if r['strategy'] == 'Binary')
    compressed_result = next(r for r in results if r['strategy'] == 'Binary + Compression')
    
    print(f"\n{'=' * 60}")
    print("Improvements vs JSON:")
    print(f"Binary:")
    print(f"  Size reduction: {100 - binary_result['compression_ratio']*100:.1f}%")
    print(f"  Speed improvement: {json_result['avg_time_per_token']/binary_result['avg_time_per_token']:.1f}x faster")
    print(f"Binary + Compression:")
    print(f"  Size reduction: {100 - compressed_result['compression_ratio']*100:.1f}%")
    print(f"  Speed improvement: {json_result['avg_time_per_token']/compressed_result['avg_time_per_token']:.1f}x faster")
    
    all_passed = all(r['all_integrity_ok'] for r in results)
    
    print(f"\n{'=' * 60}")
    print(f"KV Cache Optimization: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_cache_efficiency_scenarios():
    """Test cache efficiency in different scenarios."""
    print("\n" + "=" * 60)
    print("Testing Cache Efficiency Scenarios")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Different scenarios
    scenarios = [
        ("Short conversation", 5),
        ("Medium conversation", 20),
        ("Long conversation", 50),
        ("Very long conversation", 100),
    ]
    
    results = []
    
    for scenario_name, num_tokens in scenarios:
        print(f"\n{scenario_name} ({num_tokens} tokens):")
        
        # Initialize cache
        cache_k = np.zeros((0, 40, 128), dtype=np.float32)
        cache_v = np.zeros((0, 40, 128), dtype=np.float32)
        
        total_original_size = 0
        total_serialized_size = 0
        total_time = 0
        
        for i in range(num_tokens):
            hidden_state = np.random.randn(1, 5120).astype(np.float32)
            
            # Serialize
            start_time = time.time()
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
            
            original_size = hidden_state.nbytes + cache_k.nbytes + cache_v.nbytes
            serialized_size = get_size(serialized_hs) + get_size(serialized_k) + get_size(serialized_v)
            
            total_original_size += original_size
            total_serialized_size += serialized_size
            total_time += serialization_time
            
            # Update cache
            new_k = np.random.randn(1, 40, 128).astype(np.float32)
            new_v = np.random.randn(1, 40, 128).astype(np.float32)
            cache_k = np.concatenate([cache_k, new_k], axis=0)
            cache_v = np.concatenate([cache_v, new_v], axis=0)
        
        compression_ratio = total_serialized_size / total_original_size if total_original_size > 0 else 0
        avg_time_per_token = total_time / num_tokens * 1000  # ms
        
        results.append({
            'scenario': scenario_name,
            'tokens': num_tokens,
            'compression_ratio': compression_ratio,
            'avg_time_per_token': avg_time_per_token,
            'total_original': total_original_size,
            'total_serialized': total_serialized_size
        })
        
        print(f"  Original size: {total_original_size/1024:.1f} KB")
        print(f"  Serialized size: {total_serialized_size/1024:.1f} KB")
        print(f"  Compression ratio: {compression_ratio*100:.1f}%")
        print(f"  Avg time per token: {avg_time_per_token:.3f}ms")
    
    # Print comparison
    print(f"\n{'=' * 60}")
    print("Cache Efficiency by Scenario:")
    print(f"{'Scenario':<20} {'Tokens':<8} {'Ratio':<8} {'Time/Token':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['scenario']:<20} {result['tokens']:<8} "
              f"{result['compression_ratio']*100:<8.1f}% "
              f"{result['avg_time_per_token']:<12.3f}ms")
    
    # Check that compression ratio is reasonable for all scenarios
    # Allow reasonable processing times based on conversation length
    all_efficient = all(
        r['compression_ratio'] < 0.95 and 
        (r['avg_time_per_token'] < 15.0)  # Allow up to 15ms per token for all scenarios
        for r in results
    )
    
    print(f"\n{'=' * 60}")
    print(f"Cache Efficiency: {'PASS' if all_efficient else 'FAIL'}")
    
    return all_efficient

def test_memory_bandwidth_tradeoff():
    """Test memory vs bandwidth tradeoff."""
    print("\n" + "=" * 60)
    print("Testing Memory vs Bandwidth Tradeoff")
    print("=" * 60)
    
    # Test with different cache sizes to see the tradeoff
    cache_sizes = [10, 50, 100, 200, 500]
    
    print("Testing different cache sizes...")
    
    results = []
    
    for cache_size in cache_sizes:
        print(f"\nCache size: {cache_size} tokens")
        
        # Create cache
        cache_k = np.random.randn(cache_size, 40, 128).astype(np.float32)
        cache_v = np.random.randn(cache_size, 40, 128).astype(np.float32)
        hidden_state = np.random.randn(1, 5120).astype(np.float32)
        
        # Test with different configurations
        configs = [
            ("JSON", False, False),
            ("Binary", True, False),
            ("Compressed", True, True),
        ]
        
        config_results = []
        
        for config_name, use_binary, use_compression in configs:
            client = RemoteLayerExecutor(
                "http://localhost:8000",
                layers=[0, 1, 2],
                use_binary=use_binary,
                use_compression=use_compression,
                collect_metrics=False,
                verbose=False
            )
            
            # Serialize
            start_time = time.time()
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
                elif isinstance(data, list):
                    return len(str(data))
                return 0
            
            original_size = hidden_state.nbytes + cache_k.nbytes + cache_v.nbytes
            serialized_size = get_size(serialized_hs) + get_size(serialized_k) + get_size(serialized_v)
            
            config_results.append({
                'config': config_name,
                'original_size': original_size,
                'serialized_size': serialized_size,
                'time': serialization_time
            })
        
        # Find best configuration for this cache size
        best = min(config_results, key=lambda x: x['serialized_size'])
        
        results.append({
            'cache_size': cache_size,
            'best_config': best['config'],
            'original_size': best['original_size'],
            'serialized_size': best['serialized_size'],
            'compression_ratio': best['serialized_size'] / best['original_size'],
            'time': best['time']
        })
        
        print(f"  Best config: {best['config']}")
        print(f"  Original: {best['original_size']/1024:.1f} KB")
        print(f"  Serialized: {best['serialized_size']/1024:.1f} KB")
        print(f"  Ratio: {best['serialized_size']/best['original_size']*100:.1f}%")
    
    # Print tradeoff analysis
    print(f"\n{'=' * 60}")
    print("Memory vs Bandwidth Tradeoff:")
    print(f"{'Cache Size':<12} {'Best Config':<12} {'Ratio':<8} {'Size Saved':<12}")
    print("-" * 60)
    
    for result in results:
        size_saved = result['original_size'] - result['serialized_size']
        print(f"{result['cache_size']:<12} {result['best_config']:<12} "
              f"{result['compression_ratio']*100:<8.1f}% "
              f"{size_saved/1024:<12.1f} KB")
    
    # Check that we always get reasonable compression
    reasonable_tradeoff = all(
        r['compression_ratio'] < 0.95 and r['serialized_size'] < r['original_size']
        for r in results
    )
    
    print(f"\n{'=' * 60}")
    print(f"Memory-Bandwidth Tradeoff: {'PASS' if reasonable_tradeoff else 'FAIL'}")
    
    return reasonable_tradeoff

if __name__ == "__main__":
    print("Testing KV Cache Optimization")
    print("=" * 60)
    
    results = []
    results.append(test_kv_cache_strategies())
    results.append(test_cache_efficiency_scenarios())
    results.append(test_memory_bandwidth_tradeoff())
    
    print(f"\n{'=' * 60}")
    print("KV Cache Optimization Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All KV cache optimization tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some KV cache optimization tests failed!")
        sys.exit(1)