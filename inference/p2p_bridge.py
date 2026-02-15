#!/usr/bin/env python3
"""
P2P Bridge - Reconstructs GGUF from fragments and runs llama.cpp inference
"""
import json
import sys
from pathlib import Path
from llama_cpp import Llama

def reconstruct_gguf(fragments_dir: Path, output_path: Path):
    """Reconstruct a GGUF file from P2P fragments."""
    manifest_path = fragments_dir / "manifest.json"

    print(f"Loading manifest from {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    header_size = manifest["header_size"]
    fragments = manifest["fragments"]

    # 1. Load header
    header_path = fragments_dir / "gguf_header.dat"
    print(f"Loading header ({header_size} bytes)")
    with open(header_path, "rb") as f:
        header_data = f.read()

    # 2. Group fragments by tensor
    tensors = {}
    for frag in fragments:
        tensor_name = frag["tensor_name"]
        if tensor_name not in tensors:
            tensors[tensor_name] = {
                "data_offset": frag["data_offset"],
                "shards": []
            }
        tensors[tensor_name]["shards"].append(frag)

    # 3. Sort tensors by data_offset
    sorted_tensors = sorted(tensors.items(), key=lambda x: x[1]["data_offset"])

    # 4. Write reconstructed file
    print(f"Reconstructing GGUF to {output_path}")
    with open(output_path, "wb") as f_out:
        # Write header
        f_out.write(header_data)

        # Write tensors in order
        for tensor_name, tensor_info in sorted_tensors:
            shards = sorted(tensor_info["shards"], key=lambda x: x["shard_index"])

            for shard in shards:
                shard_path = fragments_dir / f"{shard['fragment_id']}.dat"
                with open(shard_path, "rb") as f_in:
                    f_out.write(f_in.read())

    print(f"Reconstruction complete: {output_path.stat().st_size / (1024**3):.2f} GB")

def run_inference(model_path: Path, prompt: str, max_tokens: int = 10):
    """Run inference using llama.cpp."""
    print(f"\nLoading model with llama.cpp...")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_threads=4,
        verbose=False
    )

    print(f"Prompt: {prompt}")
    print(f"Generating...")

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        echo=False
    )

    generated_text = output["choices"][0]["text"]

    print(f"\n{'='*60}")
    print(f"GENERATED TEXT:")
    print(f"{prompt}{generated_text}")
    print(f"{'='*60}\n")

    return generated_text

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P2P Bridge - Reconstruct and run inference")
    parser.add_argument("fragments_dir", help="Directory containing fragments")
    parser.add_argument("--prompt", default="The capital of France is", help="Inference prompt")
    parser.add_argument("--max-tokens", type=int, default=10, help="Max tokens to generate")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary GGUF file")

    args = parser.parse_args()

    fragments_dir = Path(args.fragments_dir)
    temp_gguf = Path("temp_reconstructed.gguf")

    # Reconstruct
    reconstruct_gguf(fragments_dir, temp_gguf)

    # Run inference
    run_inference(temp_gguf, args.prompt, args.max_tokens)

    # Cleanup
    if not args.keep_temp:
        print(f"Cleaning up temporary file...")
        temp_gguf.unlink()
    else:
        print(f"Temporary file kept: {temp_gguf}")
