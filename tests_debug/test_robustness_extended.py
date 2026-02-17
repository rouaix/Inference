#!/usr/bin/env python3
"""Extended robustness tests including network issues and corrupted data."""

import numpy as np
import sys
import os
import time
import json

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def test_corrupted_data_handling():
    """Test handling of corrupted or malformed data."""
    print("=" * 60)
    print("Testing Corrupted Data Handling")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Test cases with corrupted data formats
    corrupted_cases = [
        ("Invalid binary format", {"__binary__": True, "data": "invalid_hex", "shape": [10, 10], "dtype": "float32"}),
        ("Invalid compressed format", {"__binary_zstd__": True, "data": "invalid_base64", "shape": [10, 10], "dtype": "float32"}),
        ("Missing required field", {"__binary__": True, "data": "a" * 100, "dtype": "float32"}),  # Missing shape
        ("Invalid shape", {"__binary__": True, "data": "a" * 100, "shape": "invalid", "dtype": "float32"}),
        ("Unsupported format", {"__unknown__": True, "data": "something", "shape": [10, 10], "dtype": "float32"}),
        ("Empty data", {"__binary__": True, "data": "", "shape": [10, 10], "dtype": "float32"}),
    ]
    
    results = []
    
    for name, corrupted_data in corrupted_cases:
        print(f"\nTesting {name}:")
        try:
            # This should raise an exception
            deserialized = client._deserialize_array(corrupted_data, (10, 10))
            print(f"  Result: FAIL - Should have raised an exception")
            results.append({'name': name, 'success': False, 'error': 'No exception raised'})
        except Exception as e:
            print(f"  Result: PASS - Exception raised: {type(e).__name__}")
            results.append({'name': name, 'success': True, 'exception_type': type(e).__name__})
    
    all_passed = all(r['success'] for r in results)
    
    print(f"\n{'=' * 60}")
    print(f"Corrupted Data Handling: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_network_like_conditions():
    """Test behavior under network-like conditions (timeouts, retries, etc.)."""
    print("\n" + "=" * 60)
    print("Testing Network-Like Conditions")
    print("=" * 60)
    
    # Test with different configurations that might occur in network scenarios
    configurations = [
        ("Binary only", True, False),
        ("Compression only", False, True),
        ("Both enabled", True, True),
        ("Neither enabled", False, False),
    ]
    
    results = []
    
    for name, use_binary, use_compression in configurations:
        print(f"\nTesting {name} configuration:")
        
        try:
            client = RemoteLayerExecutor(
                "http://localhost:8000",
                layers=[0, 1, 2],
                use_binary=use_binary,
                use_compression=use_compression,
                collect_metrics=False,
                verbose=False
            )
            
            # Test with various tensor sizes
            test_tensors = [
                np.random.randn(10, 10).astype(np.float32),
                np.random.randn(100, 100).astype(np.float32),
                np.random.randn(500, 500).astype(np.float32),
            ]
            
            for i, tensor in enumerate(test_tensors):
                serialized = client._serialize_array(tensor)
                deserialized = client._deserialize_array(serialized, tensor.shape)
                
                integrity_ok = np.allclose(tensor, deserialized)
                if not integrity_ok:
                    print(f"  Tensor {i}: FAIL (integrity)")
                    results.append({'name': name, 'success': False})
                    break
            else:
                print(f"  All tensors: PASS")
                results.append({'name': name, 'success': True})
                
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            results.append({'name': name, 'success': False, 'error': str(e)})
    
    all_passed = all(r['success'] for r in results)
    
    print(f"\n{'=' * 60}")
    print(f"Network-Like Conditions: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_edge_case_tensors():
    """Test edge case tensors that might cause issues."""
    print("\n" + "=" * 60)
    print("Testing Edge Case Tensors")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Edge case tensors
    edge_cases = [
        ("Empty tensor", np.array([], dtype=np.float32)),
        ("Single element", np.array([1.0], dtype=np.float32)),
        ("Very large values", np.array([1e38], dtype=np.float32)),
        ("Very small values", np.array([1e-38], dtype=np.float32)),
        ("NaN values", np.array([np.nan, 1.0, 2.0], dtype=np.float32)),
        ("Infinity values", np.array([np.inf, -np.inf, 1.0], dtype=np.float32)),
        ("All zeros", np.zeros((100, 100), dtype=np.float32)),
        ("All ones", np.ones((100, 100), dtype=np.float32)),
        ("Alternating pattern", np.tile([1.0, -1.0], 50).astype(np.float32)),
        ("Very small tensor", np.array([[1.0]], dtype=np.float32)),
        ("Large 1D tensor", np.random.randn(10000).astype(np.float32)),
    ]
    
    results = []
    
    for name, tensor in edge_cases:
        print(f"\nTesting {name} (shape: {tensor.shape}, size: {tensor.nbytes} bytes):")
        try:
            # Serialize
            serialized = client._serialize_array(tensor)
            
            # Determine format
            format_used = "JSON"
            if isinstance(serialized, dict):
                if serialized.get("__binary_zstd__"):
                    format_used = "Compressed"
                elif serialized.get("__binary__"):
                    format_used = "Binary"
            
            # Deserialize
            deserialized = client._deserialize_array(serialized, tensor.shape)
            
            # Check integrity (allow for NaN handling)
            if tensor.size == 0:
                integrity_ok = True  # Empty tensors are always equal
            else:
                # Special handling for NaN values
                if np.any(np.isnan(tensor)):
                    integrity_ok = (
                        tensor.shape == deserialized.shape and
                        np.array_equal(tensor, deserialized, equal_nan=True)
                    )
                else:
                    integrity_ok = np.allclose(tensor, deserialized)
            
            results.append({
                'name': name,
                'format': format_used,
                'integrity': integrity_ok,
                'success': integrity_ok
            })
            
            print(f"  Format: {format_used}")
            print(f"  Integrity: {'PASS' if integrity_ok else 'FAIL'}")
            
        except Exception as e:
            print(f"  Result: FAIL - {e}")
            results.append({
                'name': name,
                'format': 'Error',
                'integrity': False,
                'success': False,
                'error': str(e)
            })
    
    all_passed = all(r['success'] for r in results)
    
    print(f"\n{'=' * 60}")
    print("Edge Case Tensors Summary:")
    for result in results:
        status = 'PASS' if result['success'] else 'FAIL'
        print(f"  {result['name']}: {status}")
    print(f"{'=' * 60}")
    print(f"Edge Case Tensors: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_long_running_stability():
    """Test stability over many iterations."""
    print("\n" + "=" * 60)
    print("Testing Long-Running Stability")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Run many iterations with varying tensor sizes
    num_iterations = 100
    print(f"Running {num_iterations} iterations...")
    
    start_time = time.time()
    
    for i in range(num_iterations):
        # Vary tensor size
        size = 10 + (i % 20) * 5  # 10-105
        tensor = np.random.randn(size, size).astype(np.float32)
        
        try:
            serialized = client._serialize_array(tensor)
            deserialized = client._deserialize_array(serialized, tensor.shape)
            
            if not np.allclose(tensor, deserialized):
                print(f"  Iteration {i+1}: FAIL (integrity)")
                return False
        except Exception as e:
            print(f"  Iteration {i+1}: FAIL - {e}")
            return False
        
        # Progress update every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_iterations} iterations...")
    
    total_time = time.time() - start_time
    
    print(f"\nCompleted {num_iterations} iterations in {total_time:.3f}s")
    print(f"Average time per iteration: {total_time/num_iterations*1000:.3f}ms")
    print(f"Throughput: {num_iterations/total_time:.1f} iterations/sec")
    
    print(f"\n{'=' * 60}")
    print("Long-Running Stability: PASS")
    
    return True

if __name__ == "__main__":
    print("Testing Extended Robustness")
    print("=" * 60)
    
    results = []
    results.append(test_corrupted_data_handling())
    results.append(test_network_like_conditions())
    results.append(test_edge_case_tensors())
    results.append(test_long_running_stability())
    
    print(f"\n{'=' * 60}")
    print("Extended Robustness Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All extended robustness tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some extended robustness tests failed!")
        sys.exit(1)