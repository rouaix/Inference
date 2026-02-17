#!/usr/bin/env python3
"""Test architecture detection for multi-model support."""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.fragment_executor import ModelArchitecture
from inference.p2p_inference import ModelConfig

def test_magistral_architecture_detection():
    """Test Magistral architecture detection."""
    print("Testing Magistral architecture detection...")
    
    # Create Magistral config
    magistral_config = ModelConfig(
        dim=5120,
        hidden_dim=32768,
        n_layers=40,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=131072,
        norm_eps=1e-5,
        rope_freq_base=1000000000.0
    )
    
    # Detect architecture
    architecture = ModelArchitecture.detect_from_config(magistral_config)
    
    # Verify
    assert architecture == ModelArchitecture.MAGISTRAL, f"Expected MAGISTRAL, got {architecture}"
    print("✅ Magistral architecture detection PASSED")
    return True

def test_mistral_7b_architecture_detection():
    """Test Mistral 7B architecture detection."""
    print("Testing Mistral 7B architecture detection...")
    
    # Create Mistral 7B config
    mistral_config = ModelConfig(
        dim=4096,
        hidden_dim=14336,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=32768,
        norm_eps=1e-5,
        rope_freq_base=1000000.0
    )
    
    # Detect architecture
    architecture = ModelArchitecture.detect_from_config(mistral_config)
    
    # Verify
    assert architecture == ModelArchitecture.MISTRAL_7B, f"Expected MISTRAL_7B, got {architecture}"
    print("✅ Mistral 7B architecture detection PASSED")
    return True

def test_unsupported_architecture():
    """Test that unsupported architectures raise appropriate errors."""
    print("Testing unsupported architecture detection...")
    
    # Create unsupported config
    unsupported_config = ModelConfig(
        dim=2048,
        hidden_dim=8192,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        vocab_size=65536,
        norm_eps=1e-5,
        rope_freq_base=10000.0
    )
    
    # This should raise ValueError
    try:
        architecture = ModelArchitecture.detect_from_config(unsupported_config)
        assert False, "Should have raised ValueError for unsupported architecture"
    except ValueError as e:
        assert "Unsupported architecture" in str(e)
        print(f"✅ Unsupported architecture error correctly raised: {e}")
        return True

def test_architecture_parameters():
    """Test that architecture parameters are correctly extracted."""
    print("Testing architecture parameter extraction...")
    
    # Test Magistral
    magistral_config = ModelConfig(
        dim=5120,
        hidden_dim=32768,
        n_layers=40,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=131072
    )
    
    # Simulate what FragmentExecutor does
    architecture = ModelArchitecture.detect_from_config(magistral_config)
    dim = magistral_config.dim
    hidden_dim = magistral_config.hidden_dim
    n_heads = magistral_config.n_heads
    n_kv_heads = magistral_config.n_kv_heads
    vocab_size = magistral_config.vocab_size
    head_dim = dim // n_heads
    
    # Verify parameters
    assert dim == 5120
    assert hidden_dim == 32768
    assert n_heads == 32
    assert n_kv_heads == 8
    assert vocab_size == 131072
    assert head_dim == 160  # 5120 / 32
    assert architecture == ModelArchitecture.MAGISTRAL
    
    # Test Mistral 7B
    mistral_config = ModelConfig(
        dim=4096,
        hidden_dim=14336,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=32768
    )
    
    architecture = ModelArchitecture.detect_from_config(mistral_config)
    dim = mistral_config.dim
    hidden_dim = mistral_config.hidden_dim
    n_heads = mistral_config.n_heads
    n_kv_heads = mistral_config.n_kv_heads
    vocab_size = mistral_config.vocab_size
    head_dim = dim // n_heads
    
    # Verify parameters
    assert dim == 4096
    assert hidden_dim == 14336
    assert n_heads == 32
    assert n_kv_heads == 8
    assert vocab_size == 32768
    assert head_dim == 128  # 4096 / 32
    assert architecture == ModelArchitecture.MISTRAL_7B
    
    print("✅ Architecture parameter extraction PASSED")
    return True

def test_input_validation():
    """Test input dimension validation."""
    print("Testing input dimension validation...")
    
    # Create a mock FragmentExecutor for testing
    from inference.fragment_executor import FragmentExecutor
    from distribution.local import BaseFragmentLoader
    
    # Mock loader
    class MockLoader(BaseFragmentLoader):
        def load_tensor(self, name: str):
            return np.zeros((10, 10), dtype=np.float32)
        
        def load_raw_tensor(self, name: str):
            return None, "F32", (10, 10)
    
    # Test Mistral 7B executor
    mistral_config = ModelConfig(
        dim=4096,
        hidden_dim=14336,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=32768
    )
    
    loader = MockLoader("test")
    executor = FragmentExecutor(loader, 0, mistral_config)
    
    # Test valid input
    valid_x = np.zeros((1, 4096), dtype=np.float32)
    valid_k = np.zeros((10, 8, 128), dtype=np.float32)  # n_kv_heads=8, head_dim=128
    valid_v = np.zeros((10, 8, 128), dtype=np.float32)
    
    # This should not raise an error
    executor._validate_input_dimensions(valid_x, valid_k, valid_v)
    print("✅ Valid input dimensions accepted")
    
    # Test invalid hidden state dimension
    invalid_x = np.zeros((1, 5120), dtype=np.float32)  # Wrong dimension
    try:
        executor._validate_input_dimensions(invalid_x, valid_k, valid_v)
        assert False, "Should have raised ValueError for wrong hidden dimension"
    except ValueError as e:
        assert "Hidden state dimension mismatch" in str(e)
        assert "4096" in str(e)  # Expected dimension
        assert "5120" in str(e)  # Got dimension
        print("✅ Invalid hidden state dimension correctly rejected")
    
    # Test invalid cache dimensions
    invalid_k = np.zeros((10, 4, 128), dtype=np.float32)  # Wrong n_kv_heads
    try:
        executor._validate_input_dimensions(valid_x, invalid_k, valid_v)
        assert False, "Should have raised ValueError for wrong cache heads"
    except ValueError as e:
        assert "Cache K heads mismatch" in str(e)
        assert "8" in str(e)  # Expected n_kv_heads
        assert "4" in str(e)  # Got n_kv_heads
        print("✅ Invalid cache dimensions correctly rejected")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Architecture Detection")
    print("=" * 60)
    
    tests = [
        test_magistral_architecture_detection,
        test_mistral_7b_architecture_detection,
        test_unsupported_architecture,
        test_architecture_parameters,
        test_input_validation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All architecture detection tests PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Some architecture detection tests FAILED!")
        sys.exit(1)