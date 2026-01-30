#include "cuda_reduce.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void reduce_scores_kernel(const float* __restrict__ L,
                                     int K,
                                     float* __restrict__ out,
                                     int M)
{
    const int m = (int)blockIdx.x;
    if (m >= M) return;

    const int tid = (int)threadIdx.x;

    float minv = INFINITY;
    float sum  = 0.0f;
    int   cnt  = 0;

    const int base = m * K;

    for (int k = tid; k < K; k += (int)blockDim.x) {
        const float v = L[base + k];
        minv = fminf(minv, v);
        if (v > 0.0f) {
            sum += v;
            cnt += 1;
        }
    }

    // Fixed block size = 128
    __shared__ float s_min[128];
    __shared__ float s_sum[128];
    __shared__ int   s_cnt[128];

    s_min[tid] = minv;
    s_sum[tid] = sum;
    s_cnt[tid] = cnt;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_sum[tid] += s_sum[tid + stride];
            s_cnt[tid] += s_cnt[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float frac = (K > 0) ? (float)s_cnt[0] / (float)K : 0.0f;
        const float mean = (s_cnt[0] > 0) ? (s_sum[0] / (float)s_cnt[0]) : 0.0f;
        const float mn   = (s_min[0] == INFINITY) ? 0.0f : s_min[0];

        out[m * 3 + 0] = frac;
        out[m * 3 + 1] = mean;
        out[m * 3 + 2] = mn;
    }
}

void launch_reduce_scores(const float* d_L_inside,
                          int numPairs,
                          int diskSamples,
                          float* d_scores,
                          cudaStream_t stream)
{
    if (numPairs <= 0) return;
    constexpr int BLOCK = 128;
    reduce_scores_kernel<<<dim3(numPairs), dim3(BLOCK), 0, stream>>>(
        d_L_inside, diskSamples, d_scores, numPairs);
}
