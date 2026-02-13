#!/usr/bin/env python3
"""
Test: Does llama.cpp generate the same token from BOS as from "Hello"?
If yes, then the issue is that we're not doing prefill correctly.
"""
from pathlib import Path
from llama_cpp import Llama
import sys

sys.path.insert(0, str(Path(__file__).parent))
from p2p_bridge import reconstruct_gguf

def test_llamacpp_bos_vs_prompt():
    """Test llama.cpp generation from BOS vs from prompt."""

    # Reconstruct GGUF
    temp_gguf = Path("temp_test_prefill.gguf")
    reconstruct_gguf(Path("tinyllama_q8_fragments_v2"), temp_gguf)

    llm = Llama(model_path=str(temp_gguf), n_ctx=512, n_threads=4, verbose=False)

    print("="*60)
    print("Testing llama.cpp: BOS vs Prompt")
    print("="*60)

    # Test 1: Generate from empty prompt (BOS only)
    print("\n[1] Generating from BOS (empty prompt)...")
    result1 = llm("", max_tokens=1, temperature=0.0, echo=False)
    token1 = result1["choices"][0]["text"]
    print(f"Generated: '{token1}'")

    # Test 2: Generate from "Hello"
    print("\n[2] Generating from 'Hello'...")
    result2 = llm("Hello", max_tokens=1, temperature=0.0, echo=False)
    token2 = result2["choices"][0]["text"]
    print(f"Generated: '{token2}'")

    # Test 3: Generate from "The"
    print("\n[3] Generating from 'The'...")
    result3 = llm("The", max_tokens=1, temperature=0.0, echo=False)
    token3 = result3["choices"][0]["text"]
    print(f"Generated: '{token3}'")

    print("\n" + "="*60)
    print("If tokens are different, it means context matters!")
    print("Our Python engine uses tokens[-1] without prefill,")
    print("so it's generating as if the prompt was just that one token.")
    print("="*60)

    # Cleanup
    temp_gguf.unlink()

if __name__ == "__main__":
    test_llamacpp_bos_vs_prompt()
