#!/usr/bin/env python3
"""Test advanced features: auto-detection and metrics collection."""

import numpy as np
import sys
import os
import time

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from distribution.reseau import RemoteLayerExecutor

def test_auto_detection():
    """Test automatic format detection."""
    print("=" * 60)
    print("Testing Automatic Format Detection")
    print("=" * 60)
    
    # Create client with auto-detection enabled
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=True,
        verbose=True
    )
    
    # Test different tensor sizes
    test_cases = [
        ("Very small", np.random.randn(10, 10).astype(np.float32)),      # 400 bytes
        ("Small", np.random.randn(50, 50).astype(np.float32)),          # 10 KB
        ("Medium", np.random.randn(100, 100).astype(np.float32)),       # 40 KB
        ("Large", np.random.randn(500, 500).astype(np.float32)),       # 1 MB
        ("Very large", np.random.randn(1000, 1000).astype(np.float32)), # 4 MB
    ]
    
    results = []
    
    for name, tensor in test_cases:
        print(f"\nTesting {name} tensor ({tensor.nbytes} bytes):")
        
        # Serialize
        serialized = client._serialize_array(tensor)
        
        # Determine which format was used
        format_used = "JSON"
        if isinstance(serialized, dict):
            if serialized.get("__binary_zstd__"):
                format_used = "Compressed"
            elif serialized.get("__binary__"):
                format_used = "Binary"
        
        # Deserialize to verify
        deserialized = client._deserialize_array(serialized, tensor.shape)
        integrity_ok = np.allclose(tensor, deserialized)
        
        results.append({
            "name": name,
            "size": tensor.nbytes,
            "format": format_used,
            "integrity": integrity_ok
        })
        
        print(f"  Format used: {format_used}")
        print(f"  Integrity: {'PASS' if integrity_ok else 'FAIL'}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Auto-Detection Summary:")
    print(f"{'Name':<15} {'Size':<12} {'Format':<12} {'Integrity':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<15} {result['size']:<12} {result['format']:<12} {'PASS' if result['integrity'] else 'FAIL':<10}")
    
    all_passed = all(result["integrity"] for result in results)
    print(f"{'=' * 60}")
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

def test_metrics_collection():
    """Test metrics collection functionality."""
    print("\n" + "=" * 60)
    print("Testing Metrics Collection")
    print("=" * 60)
    
    # Create client with metrics enabled
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=True,
        verbose=False
    )
    
    # Generate test data with different sizes to trigger different formats
    test_tensors = [
        np.random.randn(10, 10).astype(np.float32),      # Small - binary
        np.random.randn(100, 100).astype(np.float32),    # Medium - binary
        np.random.randn(500, 500).astype(np.float32),    # Large - compressed
        np.random.randn(1000, 1000).astype(np.float32),  # Very large - compressed
    ]
    
    print("Processing test tensors...")
    
    # Process all tensors
    for i, tensor in enumerate(test_tensors):
        # Serialize and deserialize
        serialized = client._serialize_array(tensor)
        deserialized = client._deserialize_array(serialized, tensor.shape)
        
        # Small delay to simulate real processing
        time.sleep(0.001)
    
    # Get metrics
    metrics = client.get_metrics()
    
    print(f"\nCollected Metrics:")
    print(f"  Total serializations: {metrics.get('serialization_count', 0)}")
    print(f"  Binary format used: {metrics.get('binary_count', 0)} ({metrics.get('binary_percentage', 0):.1f}%)")
    print(f"  Compressed format used: {metrics.get('compressed_count', 0)} ({metrics.get('compressed_percentage', 0):.1f}%)")
    print(f"  JSON format used: {metrics.get('json_count', 0)} ({metrics.get('json_percentage', 0):.1f}%)")
    print(f"  Original data size: {metrics.get('total_original_bytes', 0) / 1024:.1f} KB")
    print(f"  Serialized data size: {metrics.get('total_serialized_bytes', 0) / 1024:.1f} KB")
    print(f"  Compression ratio: {metrics.get('compression_ratio', 0) * 100:.1f}%")
    print(f"  Bandwidth savings: {metrics.get('bandwidth_savings_percentage', 0):.1f}%")
    print(f"  Avg serialization time: {metrics.get('avg_serialization_time_ms', 0):.3f} ms")
    print(f"  Avg deserialization time: {metrics.get('avg_deserialization_time_ms', 0):.3f} ms")
    
    # Verify metrics are reasonable
    success = (
        metrics.get('serialization_count', 0) == len(test_tensors) and
        metrics.get('compressed_count', 0) > 0 and  # Should have some compressed
        metrics.get('binary_count', 0) > 0 and      # Should have some binary
        metrics.get('compression_ratio', 0) < 0.95  # Should have good compression
    )
    
    print(f"\nMetrics validation: {'PASS' if success else 'FAIL'}")
    
    # Test reset
    client.reset_metrics()
    reset_metrics = client.get_metrics()
    reset_success = reset_metrics.get('serialization_count', 1) == 0
    
    print(f"Metrics reset: {'PASS' if reset_success else 'FAIL'}")
    
    return success and reset_success

def test_format_selection_logic():
    """Test the format selection logic with edge cases."""
    print("\n" + "=" * 60)
    print("Testing Format Selection Logic")
    print("=" * 60)
    
    client = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_binary=True,
        use_compression=True,
        collect_metrics=False,
        verbose=False
    )
    
    # Test edge cases
    edge_cases = [
        ("Empty tensor", np.array([], dtype=np.float32)),
        ("Single value", np.array([1.0], dtype=np.float32)),
        ("Very compressible", np.zeros((100, 100), dtype=np.float32)),  # All zeros compress well
        ("Random data", np.random.randn(100, 100).astype(np.float32)),   # Random data
    ]
    
    results = []
    
    for name, tensor in edge_cases:
        print(f"\nTesting {name}:")
        try:
            serialized = client._serialize_array(tensor)
            deserialized = client._deserialize_array(serialized, tensor.shape)
            
            format_used = "JSON"
            if isinstance(serialized, dict):
                if serialized.get("__binary_zstd__"):
                    format_used = "Compressed"
                elif serialized.get("__binary__"):
                    format_used = "Binary"
            
            integrity_ok = np.allclose(tensor, deserialized) if tensor.size > 0 else True
            
            results.append({
                "name": name,
                "format": format_used,
                "integrity": integrity_ok,
                "success": True
            })
            
            print(f"  Format: {format_used}")
            print(f"  Integrity: {'PASS' if integrity_ok else 'FAIL'}")
            print(f"  Result: PASS")
            
        except Exception as e:
            results.append({
                "name": name,
                "format": "Error",
                "integrity": False,
                "success": False,
                "error": str(e)
            })
            print(f"  Result: FAIL - {e}")
    
    all_passed = all(result["success"] for result in results)
    
    print(f"\n{'=' * 60}")
    print(f"Edge Cases: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    print("Testing Advanced Features")
    print("=" * 60)
    
    results = []
    results.append(test_auto_detection())
    results.append(test_metrics_collection())
    results.append(test_format_selection_logic())
    
    print(f"\n{'=' * 60}")
    print("Advanced Features Test Results:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print(f"{'=' * 60}")
    
    if all(results):
        print("\nSUCCESS: All advanced feature tests passed!")
        sys.exit(0)
    else:
        print("\nFAILURE: Some advanced feature tests failed!")
        sys.exit(1)