import setup_path  # noqa - adds project root to sys.path
import time, numpy as np
from inference.p2p_inference import P2PInferenceEngine

engine = P2PInferenceEngine('P:/Projets/Inference/models/Magistral-Small-2509-Q4_K_M_fragments')

tensors_to_bench = [
    'blk.0.ffn_gate.weight',
    'blk.0.ffn_up.weight',
    'blk.0.ffn_down.weight',
]

print("=== 1er appel (JIT compilation + dequantize) ===")
for t in tensors_to_bench:
    t0 = time.time()
    w = engine.load_tensor(t)
    print(f"  {t.split('.')[-2]}: {time.time()-t0:.3f}s  shape={w.shape}")

print()
print("=== 2e appel (kernel cache, dequantize seul) ===")
# Vider le weight cache pour forcer un vrai benchmark
engine._weight_cache = {}
for t in tensors_to_bench:
    t0 = time.time()
    w = engine.load_tensor(t)
    print(f"  {t.split('.')[-2]}: {time.time()-t0:.3f}s")

print()
print("=== 3e appel (weight cache RAM) ===")
for t in tensors_to_bench:
    t0 = time.time()
    w = engine.load_tensor(t)
    print(f"  {t.split('.')[-2]}: {time.time()-t0:.5f}s")
