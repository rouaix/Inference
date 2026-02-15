import numpy as np
import time

print("=== Benchmark matmul numpy ===")
w = np.random.randn(5120, 32768).astype('float32')
x7 = np.random.randn(7, 5120).astype('float32')
x1 = np.random.randn(1, 5120).astype('float32')
v1 = x1.flatten()  # 1D pour GEMV

# Prefill : seq=7
t0 = time.time()
for _ in range(3): r = x7 @ w
print(f"[7,5120]@[5120,32768] (prefill) : {(time.time()-t0)/3:.3f}s")

# Decode : seq=1 via GEMM
t0 = time.time()
for _ in range(3): r = x1 @ w
print(f"[1,5120]@[5120,32768] (GEMM)    : {(time.time()-t0)/3:.3f}s")

# Decode : seq=1 via GEMV (vecteur 1D → plus efficace)
t0 = time.time()
for _ in range(3): r = w.T @ v1
print(f"w.T @ v (GEMV)                   : {(time.time()-t0)/3:.3f}s")

# Decode : einsum
t0 = time.time()
for _ in range(3): r = np.einsum('i,ij->j', v1, w)
print(f"einsum 'i,ij->j'                 : {(time.time()-t0)/3:.3f}s")

# Config numpy BLAS
np.show_config()
