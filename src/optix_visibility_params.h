#pragma once
#include <optix_types.h>
#include <cuda_runtime.h>

// Launch parameters copied by optixLaunch() into __constant__ params on device.
struct Params
{
    OptixTraversableHandle handle;

    const float3* vertices;   // numVertices
    const uint2*  pairs;      // numPairs, each pair is (a,b) vertex indices

    int   numPairs;

    const float2* disk;       // diskSamples, Fibonacci disk offsets in segment-orthogonal plane
    int   diskSamples;

    float endpoint_eps;
    float t_eps;

    int   occupancy_nsamples; // majority vote samples (like Open3D nsamples=3)
    float occupancy_jitter;   // jitter radius for occupancy samples

    int   max_intersections;  // cap for iterative closest-hit loops (segment + occupancy)

    float* L_inside;          // output: numPairs * diskSamples floats
};
