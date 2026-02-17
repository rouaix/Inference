#!/usr/bin/env python3
"""Load and performance tests for scalability evaluation."""

import numpy as np
import sys
import os
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def worker_serialization(client, task_queue, results_queue):
    """Worker function for serialization load test."""
    while True:
        try:
            tensor = task_queue.get_nowait()
            if tensor is None:  # Sentinel value to stop
                task_queue.task_done()
                break
                
            # Serialize and deserialize
            start_time = time.time()
            serialized = client._serialize_array(tensor)
            deserialized = client._deserialize_array(serialized, tensor.shape)
            end_time = time.time()
            
            # Verify integrity
            integrity_ok = np.allclose(tensor, deserialized)
            
            results_queue.put({
                'size': tensor.nbytes,
                'time': end_time - start_time,
                'integrity': integrity_ok
            })
            
            task_queue.task_done()
        except queue.Empty:
            break
        except Exception as e:
            results_queue.put({
                'size': tensor.nbytes if tensor is not None else 0,
                'time': -1,
                'integrity': False,
                'error': str(e)
            })
            task_queue.task_done()

def test_concurrent_serialization():
    """Test serialization performance under concurrent load."""
    print("=" * 60)
    print("Testing Concurrent Serialization Performance")
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
    
    # Test configurations
    worker_counts = [1, 2, 4, 8]
    tensors_per_worker = 10
    
    results = []
    
    for num_workers in worker_counts:
        print(f"\nTesting with {num_workers} workers...")
        
        # Create task queue
        task_queue = queue.Queue()
        results_queue = queue.Queue()
        
        # Generate test tensors (mix of sizes)
        for i in range(num_workers * tensors_per_worker):
            size_factor = 10 + (i % 5) * 10  # Vary sizes: 10-50
            tensor = np.random.randn(size_factor, size_factor).astype(np.float32)
            task_queue.put(tensor)
        
        # Start workers
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(worker_serialization, client, task_queue, results_queue)
        
        # Wait for all tasks to complete
        task_queue.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        success_count = 0
        error_count = 0
        total_data = 0
        total_processing_time = 0
        
        while not results_queue.empty():
            result = results_queue.get()
            if result.get('integrity', False):
                success_count += 1
                total_data += result.get('size', 0)
                total_processing_time += result.get('time', 0)
            else:
                error_count += 1
                if 'error' in result:
                    print(f"  Error: {result['error']}")
        
        # Calculate metrics
        total_tasks = success_count + error_count
        throughput = total_tasks / total_time if total_time > 0 else 0
        avg_processing_time = total_processing_time / success_count * 1000 if success_count > 0 else 0
        data_throughput = total_data / total_time / 1024 if total_time > 0 else 0  # KB/s
        
        results.append({
            'workers': num_workers,
            'total_tasks': total_tasks,
            'success': success_count,
            'errors': error_count,
            'total_time': total_time,
            'throughput': throughput,
            'avg_processing_time': avg_processing_time,
            'data_throughput': data_throughput,
            'success_rate': success_count / total_tasks * 100 if total_tasks > 0 else 0
        })
        
        print(f"  Tasks completed: {total_tasks}")
        print(f"  Success rate: {success_count}/{total_tasks} ({success_count/total_tasks*100:.1f}%)")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} tasks/sec")
        print(f"  Data throughput: {data_throughput:.1f} KB/sec")
        print(f"  Avg processing time: {avg_processing_time:.3f} ms")
    
    # Print comparison
    print(f"\n{'=' * 60}")
    print("Concurrent Performance Comparison:")
    print(f"{'Workers':<10} {'Throughput':<12} {'Data KB/s':<12} {'Avg Time':<12} {'Success %':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['workers']:<10} {result['throughput']:<12.1f} {result['data_throughput']:<12.1f} "
              f"{result['avg_processing_time']:<12.2f} {result['success_rate']:<10.1f}")
    
    # Check if performance scales well (realistic criteria for thread-based scaling)
    baseline = results[0]['throughput']
    
    # For CPU-bound tasks with Python threads (due to GIL), we don't expect perfect linear scaling
    # Instead, check that:
    # 1. Performance doesn't degrade significantly with more workers
    # 2. We get some benefit from multiple workers (at least 20% improvement from 1 to 2 workers)
    # 3. Performance remains stable (no major drops)
    
    # Check that we get some scaling benefit initially
    initial_scaling_ok = results[1]['throughput'] > baseline * 1.2  # 20% improvement with 2 workers
    
    # Check that performance doesn't degrade with more workers
    performance_stable = all(
        results[i]['throughput'] >= results[0]['throughput'] * 0.7  # No more than 30% drop from baseline
        for i in range(1, len(results))
    )
    
    # Check that we maintain high success rate
    high_success_rate = all(r['success_rate'] >= 99 for r in results)
    
    scaling_efficient = initial_scaling_ok and performance_stable and high_success_rate
    
    print(f"{'=' * 60}")
    print(f"Scaling efficiency: {'PASS' if scaling_efficient else 'FAIL'}")
    
    all_passed = all(r['success_rate'] >= 95 for r in results) and scaling_efficient
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_memory_efficiency():
    """Test memory efficiency with large batches."""
    print("\n" + "=" * 60)
    print("Testing Memory Efficiency")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Test with increasing batch sizes
    batch_sizes = [1, 10, 50, 100]
    tensor_size = (100, 100)  # Fixed size per tensor
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batch of tensors
        tensors = [np.random.randn(*tensor_size).astype(np.float32) for _ in range(batch_size)]
        
        # Measure memory usage (rough estimate)
        start_time = time.time()
        
        serialized_data = []
        for tensor in tensors:
            serialized = client._serialize_array(tensor)
            serialized_data.append(serialized)
        
        # Calculate total sizes
        def get_serialized_size(data):
            if isinstance(data, dict):
                if data.get("__binary_zstd__"):
                    import base64
                    return len(base64.b64decode(data["data"]))
                elif data.get("__binary__"):
                    return len(data["data"]) // 2
            elif isinstance(data, list):
                return len(str(data))  # Rough estimate for JSON
            return 0
        
        original_size = sum(t.nbytes for t in tensors)
        serialized_size = sum(get_serialized_size(data) for data in serialized_data)
        
        # Deserialize to verify
        for i, data in enumerate(serialized_data):
            deserialized = client._deserialize_array(data, tensors[i].shape)
            if not np.allclose(tensors[i], deserialized):
                print(f"  Integrity check failed for tensor {i}")
                results.append({'batch_size': batch_size, 'success': False})
                break
        else:
            processing_time = time.time() - start_time
            results.append({
                'batch_size': batch_size,
                'original_size': original_size,
                'serialized_size': serialized_size,
                'compression_ratio': serialized_size / original_size if original_size > 0 else 0,
                'processing_time': processing_time,
                'success': True
            })
            
            print(f"  Original size: {original_size / 1024:.1f} KB")
            print(f"  Serialized size: {serialized_size / 1024:.1f} KB")
            print(f"  Compression ratio: {serialized_size / original_size * 100:.1f}%")
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Integrity: PASS")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Memory Efficiency Summary:")
    print(f"{'Batch Size':<12} {'Original':<12} {'Serialized':<12} {'Ratio':<10} {'Time':<8}")
    print("-" * 60)
    
    for result in results:
        if result['success']:
            print(f"{result['batch_size']:<12} {result['original_size']/1024:<12.1f} "
                  f"{result['serialized_size']/1024:<12.1f} "
                  f"{result['compression_ratio']*100:<10.1f} "
                  f"{result['processing_time']:<8.3f}")
    
    # Check if memory usage is reasonable
    memory_efficient = all(
        r['compression_ratio'] < 0.95 and r['processing_time'] < 1.0 
        for r in results if r['success']
    )
    
    all_passed = all(r['success'] for r in results) and memory_efficient
    
    print(f"{'=' * 60}")
    print(f"Memory efficiency: {'PASS' if memory_efficient else 'FAIL'}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_robustness():
    """Test robustness with network-like conditions."""
    print("\n" + "=" * 60)
    print("Testing Robustness")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Test scenarios
    scenarios = [
        ("Normal data", np.random.randn(100, 100).astype(np.float32)),
        ("All zeros", np.zeros((100, 100), dtype=np.float32)),
        ("All ones", np.ones((100, 100), dtype=np.float32)),
        ("Small values", np.random.randn(100, 100).astype(np.float32) * 0.001),
        ("Large values", np.random.randn(100, 100).astype(np.float32) * 1000),
        ("Mixed types", np.random.randn(50, 200).astype(np.float32)),
    ]
    
    results = []
    
    for name, tensor in scenarios:
        print(f"\nTesting {name}:")
        try:
            # Multiple iterations to test consistency
            for i in range(3):
                serialized = client._serialize_array(tensor)
                deserialized = client._deserialize_array(serialized, tensor.shape)
                
                integrity_ok = np.allclose(tensor, deserialized)
                if not integrity_ok:
                    print(f"  Iteration {i+1}: FAIL (integrity check)")
                    results.append({'name': name, 'success': False})
                    break
            else:
                print(f"  All iterations: PASS")
                results.append({'name': name, 'success': True})
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            results.append({'name': name, 'success': False, 'error': str(e)})
    
    all_passed = all(r['success'] for r in results)
    
    print(f"\n{'=' * 60}")
    print(f"Robustness: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

if __name__ == "__main__":
    print("Testing Load and Performance")
    print("=" * 60)
    
    results = []
    results.append(test_concurrent_serialization())
    results.append(test_memory_efficiency())
    results.append(test_robustness())
    
    print(f"\n{'=' * 60}")
    print("Load and Performance Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All load and performance tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some load and performance tests failed!")
        sys.exit(1)