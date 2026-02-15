import setup_path  # noqa - adds project root to sys.path
import time
from inference.p2p_inference import P2PInferenceEngine

engine = P2PInferenceEngine('P:/Projets/Inference/models/Magistral-Small-2509-Q4_K_M_fragments')

tensors = ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight']

print('=== 1er acces (disque + dequantize) ===')
for t in tensors:
    t0 = time.time()
    w = engine.load_tensor(t)
    print(f'  {t}: {time.time()-t0:.2f}s  shape={w.shape}')

print()
print('=== 2e acces (cache RAM) ===')
for t in tensors:
    t0 = time.time()
    w = engine.load_tensor(t)
    print(f'  {t}: {time.time()-t0:.5f}s')

print()
print(f'Tenseurs en cache : {len(engine._weight_cache)}')
