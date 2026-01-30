#pragma once
#include <cuda_runtime.h>

// d_L_inside: length numPairs*diskSamples
// d_scores:   length numPairs*3  (blocked_fraction, mean_L_inside, min_L_inside)
void launch_reduce_scores(const float* d_L_inside,
                          int numPairs,
                          int diskSamples,
                          float* d_scores,
                          cudaStream_t stream);
