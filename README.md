# CUDA GEMM + Softmax Kernels (FP32) — Naive → Shared Memory Optimized

This project implements core ML inference primitives — **GEMM** (matrix multiply) and **row-wise softmax** — in CUDA, progressing from a naive baseline to **optimized shared-memory tiled kernels**.

The goal: show **practical GPU kernel engineering** — memory access patterns, tiling, synchronization, and performance measurement — not just correctness.

> Result: **~1347× faster GEMM vs CPU** and **~4.3× faster tiled softmax vs naive GPU** on NVIDIA T4 (Colab).

---

## Why This Project?

Modern ML inference (Transformers, Attention, MatMul layers, classifier heads) is dominated by:
- Matrix multiply (`Q*Kᵀ`)
- Softmax on rows of scores
- Elementwise transformations

This repo builds these *from scratch*, the same way they exist inside TensorRT, DirectML, cuBLAS, and cuDNN — just scaled down for clarity.

---

### Benchmarks — NVIDIA T4 (FP32, M = K = N = 512)

| Category | Variant | Time (s) | Speedup |
|----------|---------|----------:|--------:|
| **GEMM** | CPU baseline | 0.750677 | 1× |
|          | Naive CUDA | 0.00127517 | **588.7× faster** |
|          | Tiled CUDA (16×16 shared memory) | 0.000995072 | **754.4× faster** |
| **Softmax** | CPU baseline | 0.00485313 | 1× |
|            | Naive CUDA (serial per row) | 0.001128 | **4.30× faster** |
|            | Tiled CUDA (shared memory + parallel reduction) | 0.000237248 | **20.46× faster** |


Correctness check:  
✔ `max_abs_error = 4.57e-05` vs CPU reference

Measurement tool:  
✔ CUDA events (`cudaEventRecord`) — not wall-clock timing

---

## Implementation Breakdown

### CPU Baseline
Naive triple-nested GEMM + softmax used solely for **ground-truth correctness**.

### Naive CUDA GEMM
- One thread computes one output element `C[i,j]`
- **Global memory accessed repeatedly**
- Good enough for correctness, not performance

### Tiled CUDA GEMM (Shared Memory)
- Tiles of A and B loaded into **`__shared__`** buffers
- Each tile reused across 16×16 threads → **dramatically fewer global loads**
- Synchronization via `__syncthreads()`

### GPU Softmax
- Naive: one thread loops over row → massively under-utilized GPU
- Optimized: **one block per row**, shared memory + block-wide parallel reduction
- Atomic-based reduction (simple & safe).  
  Future: warp-shuffle reductions to remove atomics.

---

### How to Build & Run

#### Requirements
- NVIDIA GPU (T4 used for measurements — Colab works)
- CUDA toolkit (nvcc)
- C++ compiler

#### Build
```bash
nvcc -arch=sm_75 -O3 -rdc=true \
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu \
  src/softmax_naive.cu src/softmax_tiled.cu \
  -o run
```

