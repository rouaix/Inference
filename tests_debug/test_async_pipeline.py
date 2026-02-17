#!/usr/bin/env python3
"""Test async pipeline concepts and future extensions."""

import numpy as np
import sys
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def test_async_serialization():
    """Test async serialization concepts."""
    print("=" * 60)
    print("Testing Async Serialization Concepts")
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
    
    # Simulate multiple concurrent requests
    num_requests = 10
    
    print(f"Simulating {num_requests} concurrent requests...")
    
    def process_request(request_id):
        """Simulate processing a single request."""
        # Create test data
        tensor = np.random.randn(50, 50).astype(np.float32)
        
        # Serialize
        start_time = time.time()
        serialized = client._serialize_array(tensor)
        serialization_time = time.time() - start_time
        
        # Simulate network delay
        time.sleep(0.001)
        
        # Deserialize
        start_time = time.time()
        deserialized = client._deserialize_array(serialized, tensor.shape)
        deserialization_time = time.time() - start_time
        
        # Verify integrity
        integrity_ok = np.allclose(tensor, deserialized)
        
        return {
            'request_id': request_id,
            'serialization_time': serialization_time,
            'deserialization_time': deserialization_time,
            'total_time': serialization_time + deserialization_time,
            'integrity': integrity_ok
        }
    
    # Process requests concurrently
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_request, i) for i in range(num_requests)]
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Analyze results
    success_count = sum(1 for r in results if r['integrity'])
    avg_serialization = sum(r['serialization_time'] for r in results) / len(results) * 1000
    avg_deserialization = sum(r['deserialization_time'] for r in results) / len(results) * 1000
    avg_total = sum(r['total_time'] for r in results) / len(results) * 1000
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Requests completed: {len(results)}/{num_requests}")
    print(f"  Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
    print(f"  Avg serialization time: {avg_serialization:.3f}ms")
    print(f"  Avg deserialization time: {avg_deserialization:.3f}ms")
    print(f"  Avg total time: {avg_total:.3f}ms")
    print(f"  Throughput: {num_requests/total_time:.1f} requests/sec")
    
    success = success_count == num_requests
    
    print(f"\n{'=' * 60}")
    print(f"Async Serialization: {'PASS' if success else 'FAIL'}")
    
    return success

def test_pipeline_parallelism():
    """Test pipeline parallelism concepts."""
    print("\n" + "=" * 60)
    print("Testing Pipeline Parallelism")
    print("=" * 60)
    
    # Create multiple clients (simulating multiple nodes)
    clients = [
        RemoteLayerExecutor(
            f"http://node{i}.local:8000",
            layers=[i*10, i*10+1, i*10+2],
            use_binary=True,
            use_compression=True,
            collect_metrics=False,
            verbose=False
        ) for i in range(3)
    ]
    
    print("Simulating pipeline parallelism across 3 nodes...")
    
    # Simulate processing layers in parallel
    num_layers = 6
    results = []
    
    def process_layer(client, layer_idx, tensor):
        """Process a single layer."""
        start_time = time.time()
        
        # Serialize
        serialized = client._serialize_array(tensor)
        
        # Simulate processing
        time.sleep(0.001)
        
        # Deserialize
        deserialized = client._deserialize_array(serialized, tensor.shape)
        
        processing_time = time.time() - start_time
        integrity_ok = np.allclose(tensor, deserialized)
        
        return {
            'layer': layer_idx,
            'time': processing_time,
            'integrity': integrity_ok
        }
    
    # Process layers in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        for layer_idx in range(num_layers):
            tensor = np.random.randn(10, 10).astype(np.float32)
            client = clients[layer_idx % len(clients)]
            futures.append(executor.submit(process_layer, client, layer_idx, tensor))
        
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Analyze results
    success_count = sum(1 for r in results if r['integrity'])
    
    print(f"\nResults:")
    print(f"  Layers processed: {len(results)}/{num_layers}")
    print(f"  Success rate: {success_count}/{num_layers} ({success_count/num_layers*100:.1f}%)")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Throughput: {num_layers/total_time:.1f} layers/sec")
    
    success = success_count == num_layers
    
    print(f"\n{'=' * 60}")
    print(f"Pipeline Parallelism: {'PASS' if success else 'FAIL'}")
    
    return success

def test_future_extensions():
    """Test concepts for future extensions."""
    print("\n" + "=" * 60)
    print("Testing Future Extension Concepts")
    print("=" * 60)
    
    # Test streaming compression concept
    print("\n1. Testing streaming compression concept...")
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Create large tensor
    large_tensor = np.random.randn(1000, 1000).astype(np.float32)
    
    # Test serialization
    start_time = time.time()
    serialized = client._serialize_array(large_tensor)
    serialization_time = time.time() - start_time
    
    # Calculate size
    def get_size(data):
        if isinstance(data, dict) and data.get("__binary_zstd__"):
            import base64
            return len(base64.b64decode(data["data"]))
        return 0
    
    serialized_size = get_size(serialized)
    original_size = large_tensor.nbytes
    compression_ratio = serialized_size / original_size
    
    print(f"  Large tensor: {original_size/1024:.1f} KB -> {serialized_size/1024:.1f} KB")
    print(f"  Compression ratio: {compression_ratio*100:.1f}%")
    print(f"  Serialization time: {serialization_time*1000:.3f}ms")
    
    # Test deserialization
    start_time = time.time()
    deserialized = client._deserialize_array(serialized, large_tensor.shape)
    deserialization_time = time.time() - start_time
    
    integrity_ok = np.allclose(large_tensor, deserialized)
    
    print(f"  Deserialization time: {deserialization_time*1000:.3f}ms")
    print(f"  Integrity: {'PASS' if integrity_ok else 'FAIL'}")
    
    streaming_success = integrity_ok and compression_ratio < 0.95
    
    # Test batch processing concept
    print("\n2. Testing batch processing concept...")
    
    batch_size = 10
    tensors = [np.random.randn(50, 50).astype(np.float32) for _ in range(batch_size)]
    
    batch_start = time.time()
    
    for tensor in tensors:
        serialized = client._serialize_array(tensor)
        deserialized = client._deserialize_array(serialized, tensor.shape)
        if not np.allclose(tensor, deserialized):
            batch_success = False
            break
    else:
        batch_success = True
    
    batch_time = time.time() - batch_start
    
    print(f"  Processed {batch_size} tensors in {batch_time:.3f}s")
    print(f"  Throughput: {batch_size/batch_time:.1f} tensors/sec")
    print(f"  Integrity: {'PASS' if batch_success else 'FAIL'}")
    
    # Test error handling concept
    print("\n3. Testing error handling concept...")
    
    error_cases = [
        ("Empty tensor", np.array([], dtype=np.float32)),
        ("Single value", np.array([1.0], dtype=np.float32)),
        ("Very large", np.random.randn(2000, 2000).astype(np.float32)),
    ]
    
    error_handling_success = True
    
    for name, tensor in error_cases:
        try:
            serialized = client._serialize_array(tensor)
            deserialized = client._deserialize_array(serialized, tensor.shape)
            
            if tensor.size == 0:
                integrity_ok = deserialized.size == 0
            else:
                integrity_ok = np.allclose(tensor, deserialized)
            
            if not integrity_ok:
                error_handling_success = False
                print(f"  {name}: FAIL")
            else:
                print(f"  {name}: PASS")
        except Exception as e:
            error_handling_success = False
            print(f"  {name}: FAIL - {e}")
    
    overall_success = streaming_success and batch_success and error_handling_success
    
    print(f"\n{'=' * 60}")
    print(f"Future Extensions: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success

if __name__ == "__main__":
    print("Testing Async Pipeline and Future Extensions")
    print("=" * 60)
    
    results = []
    results.append(test_async_serialization())
    results.append(test_pipeline_parallelism())
    results.append(test_future_extensions())
    
    print(f"\n{'=' * 60}")
    print("Async Pipeline Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All async pipeline tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some async pipeline tests failed!")
        sys.exit(1)