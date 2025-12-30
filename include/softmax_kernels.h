#pragma once

__global__ void softmax_naive(float* C, int M, int N);

__global__ void softmax_tiled(float* C, int M, int N);

__global__ void softmax_warp(float* C, int M, int N);
