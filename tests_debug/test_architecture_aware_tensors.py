#!/usr/bin/env python3
"""Test architecture-aware tensor creation functions."""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.kernels_numba import (
    create_hidden_state,
    create_attention_cache,
    create_ffn_intermediate,
    get_architecture_specific_attention_dims,
    get_architecture_specific_ffn_dims
)

def test_create_hidden_state():
    """Test hidden state creation for both architectures."""
    print("Testing create_hidden_state...")
    
    # Test Magistral
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072
    }
    
    hidden_state = create_hidden_state(magistral_config, batch_size=2)
    assert hidden_state.shape == (2, 5120), f"Expected (2, 5120), got {hidden_state.shape}"
    assert hidden_state.dtype == np.float32, f"Expected float32, got {hidden_state.dtype}"
    
    # Test Mistral 7B
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768
    }
    
    hidden_state = create_hidden_state(mistral_config, batch_size=1)
    assert hidden_state.shape == (1, 4096), f"Expected (1, 4096), got {hidden_state.shape}"
    assert hidden_state.dtype == np.float32, f"Expected float32, got {hidden_state.dtype}"
    
    print("✅ create_hidden_state PASSED")
    return True

def test_create_attention_cache():
    """Test attention cache creation for both architectures."""
    print("Testing create_attention_cache...")
    
    # Test Magistral
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072
    }
    
    cache_k, cache_v = create_attention_cache(magistral_config, max_length=100)
    
    # Magistral: dim=5120, n_heads=32, n_kv_heads=8 → head_dim=160
    assert cache_k.shape == (100, 8, 160), f"Expected (100, 8, 160), got {cache_k.shape}"
    assert cache_v.shape == (100, 8, 160), f"Expected (100, 8, 160), got {cache_v.shape}"
    assert cache_k.dtype == np.float32, f"Expected float32, got {cache_k.dtype}"
    assert cache_v.dtype == np.float32, f"Expected float32, got {cache_v.dtype}"
    
    # Test Mistral 7B
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768
    }
    
    cache_k, cache_v = create_attention_cache(mistral_config, max_length=50)
    
    # Mistral 7B: dim=4096, n_heads=32, n_kv_heads=8 → head_dim=128
    assert cache_k.shape == (50, 8, 128), f"Expected (50, 8, 128), got {cache_k.shape}"
    assert cache_v.shape == (50, 8, 128), f"Expected (50, 8, 128), got {cache_v.shape}"
    assert cache_k.dtype == np.float32, f"Expected float32, got {cache_k.dtype}"
    assert cache_v.dtype == np.float32, f"Expected float32, got {cache_v.dtype}"
    
    print("✅ create_attention_cache PASSED")
    return True

def test_create_ffn_intermediate():
    """Test FFN intermediate tensor creation for both architectures."""
    print("Testing create_ffn_intermediate...")
    
    # Test Magistral
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072
    }
    
    ffn_intermediate = create_ffn_intermediate(magistral_config, batch_size=3)
    assert ffn_intermediate.shape == (3, 32768), f"Expected (3, 32768), got {ffn_intermediate.shape}"
    assert ffn_intermediate.dtype == np.float32, f"Expected float32, got {ffn_intermediate.dtype}"
    
    # Test Mistral 7B
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768
    }
    
    ffn_intermediate = create_ffn_intermediate(mistral_config, batch_size=1)
    assert ffn_intermediate.shape == (1, 14336), f"Expected (1, 14336), got {ffn_intermediate.shape}"
    assert ffn_intermediate.dtype == np.float32, f"Expected float32, got {ffn_intermediate.dtype}"
    
    print("✅ create_ffn_intermediate PASSED")
    return True

def test_get_architecture_specific_attention_dims():
    """Test attention dimension extraction for both architectures."""
    print("Testing get_architecture_specific_attention_dims...")
    
    # Test Magistral with tensor_specifics
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072,
        "tensor_specifics": {
            "attention": {
                "q_dim": 2880,
                "k_dim": 2880,
                "v_dim": 4200,
                "output_dim": 2304
            }
        }
    }
    
    attn_dims = get_architecture_specific_attention_dims(magistral_config)
    assert attn_dims["q_dim"] == 2880, f"Expected q_dim=2880, got {attn_dims['q_dim']}"
    assert attn_dims["k_dim"] == 2880, f"Expected k_dim=2880, got {attn_dims['k_dim']}"
    assert attn_dims["v_dim"] == 4200, f"Expected v_dim=4200, got {attn_dims['v_dim']}"
    assert attn_dims["output_dim"] == 2304, f"Expected output_dim=2304, got {attn_dims['output_dim']}"
    
    # Test Mistral 7B with tensor_specifics
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768,
        "tensor_specifics": {
            "attention": {
                "q_dim": 2304,
                "k_dim": 2304,
                "v_dim": 3360,
                "output_dim": 2304
            }
        }
    }
    
    attn_dims = get_architecture_specific_attention_dims(mistral_config)
    assert attn_dims["q_dim"] == 2304, f"Expected q_dim=2304, got {attn_dims['q_dim']}"
    assert attn_dims["k_dim"] == 2304, f"Expected k_dim=2304, got {attn_dims['k_dim']}"
    assert attn_dims["v_dim"] == 3360, f"Expected v_dim=3360, got {attn_dims['v_dim']}"
    assert attn_dims["output_dim"] == 2304, f"Expected output_dim=2304, got {attn_dims['output_dim']}"
    
    # Test fallback (no tensor_specifics)
    fallback_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768
    }
    
    attn_dims = get_architecture_specific_attention_dims(fallback_config)
    assert attn_dims["q_dim"] == 4096, f"Expected fallback q_dim=4096, got {attn_dims['q_dim']}"
    assert attn_dims["k_dim"] == 4096, f"Expected fallback k_dim=4096, got {attn_dims['k_dim']}"
    assert attn_dims["v_dim"] == 4096, f"Expected fallback v_dim=4096, got {attn_dims['v_dim']}"
    assert attn_dims["output_dim"] == 4096, f"Expected fallback output_dim=4096, got {attn_dims['output_dim']}"
    
    print("✅ get_architecture_specific_attention_dims PASSED")
    return True

def test_get_architecture_specific_ffn_dims():
    """Test FFN dimension extraction for both architectures."""
    print("Testing get_architecture_specific_ffn_dims...")
    
    # Test Magistral with tensor_specifics
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072,
        "tensor_specifics": {
            "ffn": {
                "gate_dim": 2880,
                "up_dim": 2880,
                "down_dim": 26880
            }
        }
    }
    
    ffn_dims = get_architecture_specific_ffn_dims(magistral_config)
    assert ffn_dims["gate_dim"] == 2880, f"Expected gate_dim=2880, got {ffn_dims['gate_dim']}"
    assert ffn_dims["up_dim"] == 2880, f"Expected up_dim=2880, got {ffn_dims['up_dim']}"
    assert ffn_dims["down_dim"] == 26880, f"Expected down_dim=26880, got {ffn_dims['down_dim']}"
    
    # Test Mistral 7B with tensor_specifics
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768,
        "tensor_specifics": {
            "ffn": {
                "gate_dim": 2304,
                "up_dim": 2304,
                "down_dim": 11760
            }
        }
    }
    
    ffn_dims = get_architecture_specific_ffn_dims(mistral_config)
    assert ffn_dims["gate_dim"] == 2304, f"Expected gate_dim=2304, got {ffn_dims['gate_dim']}"
    assert ffn_dims["up_dim"] == 2304, f"Expected up_dim=2304, got {ffn_dims['up_dim']}"
    assert ffn_dims["down_dim"] == 11760, f"Expected down_dim=11760, got {ffn_dims['down_dim']}"
    
    # Test fallback (no tensor_specifics)
    fallback_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768
    }
    
    ffn_dims = get_architecture_specific_ffn_dims(fallback_config)
    assert ffn_dims["gate_dim"] == 4096, f"Expected fallback gate_dim=4096, got {ffn_dims['gate_dim']}"
    assert ffn_dims["up_dim"] == 4096, f"Expected fallback up_dim=4096, got {ffn_dims['up_dim']}"
    assert ffn_dims["down_dim"] == 14336, f"Expected fallback down_dim=14336, got {ffn_dims['down_dim']}"
    
    print("✅ get_architecture_specific_ffn_dims PASSED")
    return True

def test_integration_with_fragment_executor():
    """Test integration with FragmentExecutor architecture parameters."""
    print("Testing integration with FragmentExecutor...")
    
    # Create mock config like FragmentExecutor would have
    magistral_config = {
        "dim": 5120,
        "hidden_dim": 32768,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072,
        "tensor_specifics": {
            "attention": {
                "q_dim": 2880,
                "k_dim": 2880,
                "v_dim": 4200,
                "output_dim": 2304
            },
            "ffn": {
                "gate_dim": 2880,
                "up_dim": 2880,
                "down_dim": 26880
            }
        }
    }
    
    # Test creating all tensors for Magistral
    hidden_state = create_hidden_state(magistral_config, batch_size=1)
    cache_k, cache_v = create_attention_cache(magistral_config, max_length=10)
    ffn_intermediate = create_ffn_intermediate(magistral_config, batch_size=1)
    
    attn_dims = get_architecture_specific_attention_dims(magistral_config)
    ffn_dims = get_architecture_specific_ffn_dims(magistral_config)
    
    # Verify all dimensions are consistent
    assert hidden_state.shape == (1, 5120)
    assert cache_k.shape == (10, 8, 160)  # 5120/32 = 160
    assert cache_v.shape == (10, 8, 160)
    assert ffn_intermediate.shape == (1, 32768)
    assert attn_dims["q_dim"] == 2880
    assert ffn_dims["gate_dim"] == 2880
    
    # Test Mistral 7B
    mistral_config = {
        "dim": 4096,
        "hidden_dim": 14336,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 32768,
        "tensor_specifics": {
            "attention": {
                "q_dim": 2304,
                "k_dim": 2304,
                "v_dim": 3360,
                "output_dim": 2304
            },
            "ffn": {
                "gate_dim": 2304,
                "up_dim": 2304,
                "down_dim": 11760
            }
        }
    }
    
    hidden_state = create_hidden_state(mistral_config, batch_size=1)
    cache_k, cache_v = create_attention_cache(mistral_config, max_length=10)
    ffn_intermediate = create_ffn_intermediate(mistral_config, batch_size=1)
    
    attn_dims = get_architecture_specific_attention_dims(mistral_config)
    ffn_dims = get_architecture_specific_ffn_dims(mistral_config)
    
    # Verify all dimensions are consistent
    assert hidden_state.shape == (1, 4096)
    assert cache_k.shape == (10, 8, 128)  # 4096/32 = 128
    assert cache_v.shape == (10, 8, 128)
    assert ffn_intermediate.shape == (1, 14336)
    assert attn_dims["q_dim"] == 2304
    assert ffn_dims["gate_dim"] == 2304
    
    print("✅ Integration with FragmentExecutor PASSED")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Architecture-Aware Tensor Operations")
    print("=" * 60)
    
    tests = [
        test_create_hidden_state,
        test_create_attention_cache,
        test_create_ffn_intermediate,
        test_get_architecture_specific_attention_dims,
        test_get_architecture_specific_ffn_dims,
        test_integration_with_fragment_executor
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All architecture-aware tensor tests PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Some architecture-aware tensor tests FAILED!")
        sys.exit(1)