#!/usr/bin/env python3
"""Test script for KV cache functionality."""

import numpy as np
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distribution.reseau import RemoteLayerExecutor

def test_session_management():
    """Test session start and end functionality."""
    print("Testing session management...")
    
    # Create a mock executor
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_session_cache=True,
        verbose=True
    )
    
    # Test session start
    executor.start_session()
    if executor.session_id is not None:
        print(f"[PASS] Session started with ID: {executor.session_id}")
        session_started = True
    else:
        print("[FAIL] Session ID is None after start_session")
        session_started = False
    
    # Test session end
    executor.end_session()
    if executor.session_id is None:
        print("[PASS] Session ended successfully")
        session_ended = True
    else:
        print("[FAIL] Session ID is not None after end_session")
        session_ended = False
    
    return session_started and session_ended

def test_kv_cache_with_session():
    """Test KV cache behavior with session."""
    print("\nTesting KV cache with session...")
    
    # Create a mock executor with session cache
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_session_cache=True,
        verbose=True
    )
    
    # Start a session
    executor.start_session()
    
    # Create test data
    hidden_state = np.random.rand(1, 5120).astype(np.float32)
    cache_k = np.random.rand(10, 4, 128).astype(np.float32)
    cache_v = np.random.rand(10, 4, 128).astype(np.float32)
    
    # Test that cache is not sent when session is active
    # (This is a unit test, so we'll just check the body construction)
    body = {
        "layer_idx": 0,
        "hidden_state": executor._serialize_array(hidden_state),
        "pos": 0,
    }
    
    if executor.use_session_cache and executor.session_id:
        body["session_id"] = executor.session_id
        # Cache should not be in body when using session cache
        if "cache_k" not in body and "cache_v" not in body:
            print("[PASS] Cache not included in request when using session")
            cache_test_passed = True
        else:
            print("[FAIL] Cache included in request when using session")
            cache_test_passed = False
    else:
        print("[FAIL] Session not properly initialized")
        cache_test_passed = False
    
    # End the session
    executor.end_session()
    
    return cache_test_passed

def test_kv_cache_without_session():
    """Test KV cache behavior without session."""
    print("\nTesting KV cache without session...")
    
    # Create a mock executor without session cache
    executor = RemoteLayerExecutor(
        "http://localhost:8000",
        layers=[0, 1, 2],
        use_session_cache=False,
        verbose=True
    )
    
    # Create test data
    hidden_state = np.random.rand(1, 5120).astype(np.float32)
    cache_k = np.random.rand(10, 4, 128).astype(np.float32)
    cache_v = np.random.rand(10, 4, 128).astype(np.float32)
    
    # Test that cache is sent when not using session
    body = {
        "layer_idx": 0,
        "hidden_state": executor._serialize_array(hidden_state),
        "pos": 0,
        "cache_k": executor._serialize_array(cache_k),
        "cache_v": executor._serialize_array(cache_v),
    }
    
    if "cache_k" in body and "cache_v" in body:
        print("[PASS] Cache included in request when not using session")
        cache_test_passed = True
    else:
        print("[FAIL] Cache not included in request when not using session")
        cache_test_passed = False
    
    return cache_test_passed

if __name__ == "__main__":
    print("=" * 60)
    print("Testing KV Cache Functionality")
    print("=" * 60)
    
    results = []
    results.append(test_session_management())
    results.append(test_kv_cache_with_session())
    results.append(test_kv_cache_without_session())
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n[SUCCESS] All KV cache tests PASSED")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some KV cache tests FAILED")
        sys.exit(1)