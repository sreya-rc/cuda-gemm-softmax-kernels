#pragma once

__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int K, int N);

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int K, int N);
