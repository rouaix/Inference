#!/usr/bin/env python3
"""
Test suite for Mistral 7B architecture detection and tensor operations.
This verifies that the multi-architecture support works correctly with real Mistral 7B fragments.
"""

import sys
import os
import json
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from inference.fragment_executor import FragmentExecutor, ModelArchitecture
from inference.kernels_numba import (
    create_hidden_state, 
    create_attention_cache, 
    create_ffn_intermediate,
    get_architecture_specific_attention_dims,
    get_architecture_specific_ffn_dims
)

def test_mistral_7b_architecture_detection():
    """Test that Mistral 7B architecture is correctly detected from manifest."""
    print("Testing Mistral 7B architecture detection...")
    
    # Load Mistral 7B manifest
    manifest_path = "models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    config = manifest['config']
    
    # Test architecture detection
    detected_arch = ModelArchitecture.detect_from_config(config)
    
    print(f"Detected architecture: {detected_arch}")
    print(f"Config dim: {config['dim'][0]}")
    print(f"Config hidden_dim: {config['hidden_dim'][0]}")
    
    # Verify it's correctly identified as MISTRAL_7B
    assert detected_arch == ModelArchitecture.MISTRAL_7B, f"Expected MISTRAL_7B, got {detected_arch}"
    
    # Verify dimensions
    assert config['dim'][0] == 4096, f"Expected dim 4096, got {config['dim'][0]}"
    assert config['hidden_dim'][0] == 14336, f"Expected hidden_dim 14336, got {config['hidden_dim'][0]}"
    
    print("✓ Mistral 7B architecture detection successful")

def test_mistral_7b_tensor_creation():
    """Test that tensors are created with correct dimensions for Mistral 7B."""
    print("\nTesting Mistral 7B tensor creation...")
    
    # Mistral 7B architecture config
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8
    }
    
    # Test hidden state creation
    hidden_state = create_hidden_state(mistral_config, batch_size=2)
    expected_shape = (2, 4096)
    assert hidden_state.shape == expected_shape, f"Expected {expected_shape}, got {hidden_state.shape}"
    assert hidden_state.dtype == np.float32, f"Expected float32, got {hidden_state.dtype}"
    print(f"✓ Hidden state shape: {hidden_state.shape}, dtype: {hidden_state.dtype}")
    
    # Test attention cache creation
    attention_cache = create_attention_cache(mistral_config, batch_size=2, seq_len=128)
    expected_shape = (2, 32, 128, 128)  # (batch, n_heads, seq_len, head_dim)
    assert attention_cache.shape == expected_shape, f"Expected {expected_shape}, got {attention_cache.shape}"
    assert attention_cache.dtype == np.float32, f"Expected float32, got {attention_cache.dtype}"
    print(f"✓ Attention cache shape: {attention_cache.shape}, dtype: {attention_cache.dtype}")
    
    # Test FFN intermediate creation
    ffn_intermediate = create_ffn_intermediate(mistral_config, batch_size=2, seq_len=128)
    expected_shape = (2, 128, 14336)  # (batch, seq_len, hidden_dim)
    assert ffn_intermediate.shape == expected_shape, f"Expected {expected_shape}, got {ffn_intermediate.shape}"
    assert ffn_intermediate.dtype == np.float32, f"Expected float32, got {ffn_intermediate.dtype}"
    print(f"✓ FFN intermediate shape: {ffn_intermediate.shape}, dtype: {ffn_intermediate.dtype}")
    
    print("✓ Mistral 7B tensor creation successful")

def test_mistral_7b_attention_dims():
    """Test that attention-specific dimensions are correct for Mistral 7B."""
    print("\nTesting Mistral 7B attention dimensions...")
    
    # Mistral 7B architecture config
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8
    }
    
    # Get attention dimensions
    q_dim, k_dim, v_dim, output_dim = get_architecture_specific_attention_dims(mistral_config)
    
    print(f"Q dimension: {q_dim}")
    print(f"K dimension: {k_dim}")
    print(f"V dimension: {v_dim}")
    print(f"Output dimension: {output_dim}")
    
    # For Mistral 7B with 32 heads and dim=4096, head_dim should be 128 (4096/32)
    expected_head_dim = 128
    expected_q_dim = 4096  # n_heads * head_dim = 32 * 128
    expected_k_dim = 1024  # n_kv_heads * head_dim = 8 * 128  
    expected_v_dim = 1024  # n_kv_heads * head_dim = 8 * 128
    expected_output_dim = 4096  # dim
    
    assert q_dim == expected_q_dim, f"Expected q_dim {expected_q_dim}, got {q_dim}"
    assert k_dim == expected_k_dim, f"Expected k_dim {expected_k_dim}, got {k_dim}"
    assert v_dim == expected_v_dim, f"Expected v_dim {expected_v_dim}, got {v_dim}"
    assert output_dim == expected_output_dim, f"Expected output_dim {expected_output_dim}, got {output_dim}"
    
    print("✓ Mistral 7B attention dimensions correct")

def test_mistral_7b_ffn_dims():
    """Test that FFN-specific dimensions are correct for Mistral 7B."""
    print("\nTesting Mistral 7B FFN dimensions...")
    
    # Mistral 7B architecture config
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8
    }
    
    # Get FFN dimensions
    gate_dim, up_dim, down_dim = get_architecture_specific_ffn_dims(mistral_config)
    
    print(f"Gate dimension: {gate_dim}")
    print(f"Up dimension: {up_dim}")
    print(f"Down dimension: {down_dim}")
    
    # For Mistral 7B
    expected_gate_dim = 14336  # hidden_dim
    expected_up_dim = 14336     # hidden_dim
    expected_down_dim = 4096    # dim
    
    assert gate_dim == expected_gate_dim, f"Expected gate_dim {expected_gate_dim}, got {gate_dim}"
    assert up_dim == expected_up_dim, f"Expected up_dim {expected_up_dim}, got {up_dim}"
    assert down_dim == expected_down_dim, f"Expected down_dim {expected_down_dim}, got {down_dim}"
    
    print("✓ Mistral 7B FFN dimensions correct")

def test_fragment_executor_with_mistral_7b():
    """Test FragmentExecutor initialization with Mistral 7B manifest."""
    print("\nTesting FragmentExecutor with Mistral 7B...")
    
    # Load Mistral 7B manifest
    manifest_path = "models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments/manifest.json"
    
    try:
        # Create FragmentExecutor with Mistral 7B manifest
        executor = FragmentExecutor(manifest_path)
        
        # Verify architecture detection
        assert executor.architecture == ModelArchitecture.MISTRAL_7B
        assert executor.dim == 4096
        assert executor.hidden_dim == 14336
        
        print(f"✓ FragmentExecutor architecture: {executor.architecture}")
        print(f"✓ FragmentExecutor dim: {executor.dim}")
        print(f"✓ FragmentExecutor hidden_dim: {executor.hidden_dim}")
        
        # Test architecture config dict
        arch_config = executor.get_architecture_config_dict()
        assert arch_config["dim"] == 4096
        assert arch_config["hidden_dim"] == 14336
        assert arch_config["n_heads"] == 32
        assert arch_config["n_kv_heads"] == 8
        
        print("✓ FragmentExecutor architecture config dict correct")
        
    except Exception as e:
        print(f"✗ FragmentExecutor initialization failed: {e}")
        raise
    
    print("✓ FragmentExecutor with Mistral 7B successful")

def test_architecture_comparison():
    """Compare Mistral 7B vs Magistral architectures to ensure no regression."""
    print("\nTesting architecture comparison...")
    
    # Load both manifests
    mistral_manifest_path = "models/Mistral_7B_Instruct_v0_3_Q4_K_M_fragments/manifest.json"
    magistral_manifest_path = "models/Magistral-Small-2509-Q4_K_M_fragments/manifest.json"
    
    with open(mistral_manifest_path, 'r') as f:
        mistral_manifest = json.load(f)
    
    with open(magistral_manifest_path, 'r') as f:
        magistral_manifest = json.load(f)
    
    # Detect architectures
    mistral_arch = ModelArchitecture.detect_from_config(mistral_manifest['config'])
    magistral_arch = ModelArchitecture.detect_from_config(magistral_manifest['config'])
    
    print(f"Mistral 7B architecture: {mistral_arch}")
    print(f"Magistral architecture: {magistral_arch}")
    
    # Verify they're different
    assert mistral_arch != magistral_arch, "Architectures should be different"
    assert mistral_arch == ModelArchitecture.MISTRAL_7B
    assert magistral_arch == ModelArchitecture.MAGISTRAL
    
    # Verify different dimensions
    mistral_dim = mistral_manifest['config']['dim'][0]
    magistral_dim = magistral_manifest['config']['dim'][0]
    
    assert mistral_dim == 4096, f"Expected Mistral dim 4096, got {mistral_dim}"
    assert magistral_dim == 5120, f"Expected Magistral dim 5120, got {magistral_dim}"
    
    print(f"Mistral 7B dim: {mistral_dim}")
    print(f"Magistral dim: {magistral_dim}")
    
    print("✓ Architecture comparison successful - both architectures properly distinguished")

def run_all_tests():
    """Run all Mistral 7B architecture tests."""
    print("=" * 60)
    print("MISTRAL 7B ARCHITECTURE TEST SUITE")
    print("=" * 60)
    
    try:
        test_mistral_7b_architecture_detection()
        test_mistral_7b_tensor_creation()
        test_mistral_7b_attention_dims()
        test_mistral_7b_ffn_dims()
        test_fragment_executor_with_mistral_7b()
        test_architecture_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 ALL MISTRAL 7B ARCHITECTURE TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)