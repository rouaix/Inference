#!/usr/bin/env python3
"""Comparative benchmarks for different serialization approaches."""

import numpy as np
import sys
import os
import time
import json

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def benchmark_serialization_methods():
    """Benchmark different serialization methods."""
    print("=" * 60)
    print("Serialization Methods Benchmark")
    print("=" * 60)
    
    # Test data sizes
    sizes = [
        ("Small", (50, 50)),      # ~10 KB
        ("Medium", (200, 200)),    # ~160 KB
        ("Large", (500, 500)),    # ~1 MB
        ("Very Large", (1000, 1000)),  # ~4 MB
    ]
    
    # Different configurations
    configs = [
        ("JSON", False, False),
        ("Binary", True, False),
        ("Binary + Compression", True, True),
    ]
    
    results = []
    
    for size_name, shape in sizes:
        print(f"\n{'=' * 40}")
        print(f"Testing {size_name} tensors (shape: {shape})")
        print(f"{'=' * 40}")
        
        # Create test tensor
        tensor = np.random.randn(*shape).astype(np.float32)
        original_size = tensor.nbytes
        
        for config_name, use_binary, use_compression in configs:
            # Create client with specific configuration
            client = RemoteLayerExecutor(
                "http://localhost:8000",
                layers=[0, 1, 2],
                use_binary=use_binary,
                use_compression=use_compression,
                collect_metrics=False,
                verbose=False
            )
            
            # Benchmark serialization
            start_time = time.time()
            serialized = client._serialize_array(tensor)
            serialization_time = time.time() - start_time
            
            # Calculate serialized size
            def get_size(data):
                if isinstance(data, dict):
                    if data.get("__binary_zstd__"):
                        import base64
                        return len(base64.b64decode(data["data"]))
                    elif data.get("__binary__"):
                        return len(data["data"]) // 2
                elif isinstance(data, list):
                    return len(json.dumps(data))
                return 0
            
            serialized_size = get_size(serialized)
            
            # Benchmark deserialization
            start_time = time.time()
            deserialized = client._deserialize_array(serialized, tensor.shape)
            deserialization_time = time.time() - start_time
            
            # Verify integrity
            integrity_ok = np.allclose(tensor, deserialized)
            
            results.append({
                'size_name': size_name,
                'config': config_name,
                'original_size': original_size,
                'serialized_size': serialized_size,
                'compression_ratio': serialized_size / original_size if original_size > 0 else 0,
                'serialization_time': serialization_time * 1000,  # ms
                'deserialization_time': deserialization_time * 1000,  # ms
                'total_time': (serialization_time + deserialization_time) * 1000,  # ms
                'integrity': integrity_ok
            })
            
            print(f"  {config_name:<20}:")
            print(f"    Size: {original_size/1024:>6.1f} KB -> {serialized_size/1024:>6.1f} KB ({serialized_size/original_size*100:>5.1f}%)")
            print(f"    Serialize: {serialization_time*1000:>6.3f} ms")
            print(f"    Deserialize: {deserialization_time*1000:>6.3f} ms")
            print(f"    Total: { (serialization_time + deserialization_time)*1000:>6.3f} ms")
            print(f"    Integrity: {'PASS' if integrity_ok else 'FAIL'}")
    
    # Print comprehensive comparison
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    
    for size_name in ["Small", "Medium", "Large", "Very Large"]:
        print(f"\n{size_name} Tensors:")
        size_results = [r for r in results if r['size_name'] == size_name]
        
        # Find best configuration for this size
        best = min(size_results, key=lambda x: x['total_time'])
        
        print(f"  Original Size: {size_results[0]['original_size'] / 1024:.1f} KB")
        print(f"  Best Configuration: {best['config']}")
        print(f"  Best Total Time: {best['total_time']:.3f} ms")
        print(f"  Best Compression: {best['compression_ratio']*100:.1f}%")
        if best['config'] == "JSON":
            perf_vs_json = "N/A"
        else:
            perf_vs_json = f"{size_results[0]['total_time']/best['total_time']:.1f}x faster"
        print(f"  Performance vs JSON: {perf_vs_json}")
    
    # Calculate overall improvements
    json_results = [r for r in results if r['config'] == "JSON"]
    binary_results = [r for r in results if r['config'] == "Binary"]
    compressed_results = [r for r in results if r['config'] == "Binary + Compression"]
    
    avg_json_time = sum(r['total_time'] for r in json_results) / len(json_results)
    avg_binary_time = sum(r['total_time'] for r in binary_results) / len(binary_results)
    avg_compressed_time = sum(r['total_time'] for r in compressed_results) / len(compressed_results)
    
    avg_json_size = sum(r['serialized_size'] for r in json_results) / len(json_results)
    avg_binary_size = sum(r['serialized_size'] for r in binary_results) / len(binary_results)
    avg_compressed_size = sum(r['serialized_size'] for r in compressed_results) / len(compressed_results)
    
    print(f"\n{'=' * 60}")
    print("OVERALL IMPROVEMENTS")
    print(f"{'=' * 60}")
    print(f"Binary vs JSON:")
    print(f"  Speed: {avg_json_time/avg_binary_time:.1f}x faster")
    print(f"  Size: {avg_binary_size/avg_json_size*100:.1f}% of JSON")
    print(f"Binary + Compression vs JSON:")
    print(f"  Speed: {avg_json_time/avg_compressed_time:.1f}x faster")
    print(f"  Size: {avg_compressed_size/avg_json_size*100:.1f}% of JSON")
    
    # Check all tests passed
    all_passed = all(r['integrity'] for r in results)
    
    print(f"\n{'=' * 60}")
    print(f"Benchmark Integrity: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("\n" + "=" * 60)
    print("Memory Usage Benchmark")
    print("=" * 60)
    
    # Create client
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Test with increasing batch sizes
    batch_sizes = [1, 10, 50, 100, 200]
    tensor_size = (100, 100)  # Fixed size per tensor
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batch
        tensors = [np.random.randn(*tensor_size).astype(np.float32) for _ in range(batch_size)]
        
        # Measure memory usage (rough estimate)
        start_time = time.time()
        
        serialized_data = []
        for tensor in tensors:
            serialized = client._serialize_array(tensor)
            serialized_data.append(serialized)
        
        # Calculate sizes
        original_size = sum(t.nbytes for t in tensors)
        
        def get_serialized_size(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    return len(base64.b64decode(data["data"]))
                elif data.get("__binary__"):
                    return len(data["data"]) // 2
            elif isinstance(data, list):
                return len(str(data))
            return 0
        
        serialized_size = sum(get_serialized_size(data) for data in serialized_data)
        processing_time = time.time() - start_time
        
        # Calculate metrics
        memory_efficiency = serialized_size / original_size if original_size > 0 else 0
        throughput = batch_size / processing_time if processing_time > 0 else 0
        
        results.append({
            'batch_size': batch_size,
            'original_size': original_size,
            'serialized_size': serialized_size,
            'memory_efficiency': memory_efficiency,
            'processing_time': processing_time,
            'throughput': throughput
        })
        
        print(f"  Original: {original_size/1024:.1f} KB")
        print(f"  Serialized: {serialized_size/1024:.1f} KB")
        print(f"  Ratio: {memory_efficiency*100:.1f}%")
        print(f"  Time: {processing_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} tensors/sec")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Memory Usage Summary:")
    print(f"{'Batch':<8} {'Original':<10} {'Serialized':<12} {'Ratio':<8} {'Throughput':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batch_size']:<8} {result['original_size']/1024:<10.1f} "
              f"{result['serialized_size']/1024:<12.1f} "
              f"{result['memory_efficiency']*100:<8.1f} "
              f"{result['throughput']:<12.1f}")
    
    # Check scaling
    baseline_throughput = results[0]['throughput']
    scaling_efficient = all(
        r['throughput'] >= baseline_throughput * 0.8 * (r['batch_size'] / results[0]['batch_size'])
        for r in results[1:]
    )
    
    print(f"{'=' * 60}")
    print(f"Memory Scaling: {'PASS' if scaling_efficient else 'FAIL'}")
    
    return scaling_efficient

if __name__ == "__main__":
    print("Running Comparative Benchmarks")
    print("=" * 60)
    
    results = []
    results.append(benchmark_serialization_methods())
    results.append(benchmark_memory_usage())
    
    print(f"\n{'=' * 60}")
    print("Benchmark Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All benchmark tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some benchmark tests failed!")
        sys.exit(1)