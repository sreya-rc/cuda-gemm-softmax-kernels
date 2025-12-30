# CUDA GEMM + Softmax Kernels (FP32) — Naive → Shared Memory → Warp-Level Optimized

This project implements core ML inference primitives — **GEMM** (matrix multiply) and **row-wise softmax** — directly in CUDA.  
The implementation progresses from simple baselines to optimized GPU kernels using shared memory tiling and warp shuffle intrinsics.

The goal is to demonstrate practical GPU kernel engineering:
- memory access patterns
- parallel execution mapping
- shared memory tiling
- warp-level reductions
- correctness and timing measurement using CUDA events

---

## Benchmarks — NVIDIA T4 (FP32, M = K = N = 512)

### GEMM Performance

| Variant | Time (s) | Speedup vs CPU |
|--------|----------:|----------------:|
| CPU baseline | 0.193987 | 1× |
| Naive CUDA | 0.00130179 | 149.0× |
| Tiled CUDA (16×16 shared memory) | 0.000983168 | 197.3× |

### Softmax Performance

| Variant | Time (s) | Speedup vs CPU | Speedup vs naive |
|--------|----------:|----------------:|------------------:|
| CPU baseline | 0.00299208 | 1× | — |
| Naive CUDA (1 thread/row) | 0.00112278 | 2.66× | — |
| Tiled CUDA (shared memory, block-wide reduction) | 0.000224576 | 13.32× | 4.99× |
| Warp-shuffle optimized softmax (`__shfl_down_sync`) | 0.000071904 | 41.61× | 15.62× |

**Correctness:**  
All kernels produce a maximum absolute error ≤ **4.57e-05** relative to CPU reference.

**Timing method:**  
CUDA events (`cudaEventRecord`) — not wall-clock time.

---

## Implementation Overview

### CPU Reference
Used only for correctness and baseline comparison.

- triple-nested GEMM
- row-wise softmax with max-subtraction for stability

### Naive CUDA GEMM
Each thread computes one output element `C[i,j]`.  
Global memory reused repeatedly → memory-bound performance.

### Tiled CUDA GEMM
Each thread block:
- loads tiles of A and B into `__shared__` memory
- synchronizes (`__syncthreads()`)
- reuses data across 16×16 threads

Reduces global memory traffic significantly.

### Softmax Kernels
- Naive softmax: 1 thread per row
- Tiled softmax: 1 block per row + shared memory scratch buffer
- Warp softmax: No shared memory or atomics; uses warp shuffles for reductions

---

## Build and Run

### Requirements
- NVIDIA GPU with CUDA support
- CUDA toolkit (`nvcc`)
- C++ compiler

### Build from source
```bash
nvcc -arch=sm_75 -O3 -rdc=true \
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu \
  src/softmax_naive.cu src/softmax_tiled.cu src/softmax_warp.cu \
  -o run
