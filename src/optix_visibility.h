#pragma once
#include <vector>
#include <cstdint>

class OptixVisibility
{
public:
    struct Impl;
    OptixVisibility(const float* vertices0, int numVertices,
                    const std::vector<uint32_t>& faces_flat, // length = numFaces*3
                    const std::vector<uint32_t>& idx_list,
                    int diskSamples = 100,
                    float diskRadius = 0.1f,
                    int deviceId = 0,
                    float endpoint_eps = 1e-4f,
                    float t_eps = 1e-6f,
                    int occupancy_nsamples = 3,
                    float occupancy_jitter = 1e-4f,
                    int max_intersections = 256,
                    bool validation = false);

    ~OptixVisibility();

    int num_pairs() const;
    int disk_samples() const;

    // vertices_frames: num_frames * num_vertices * 3 floats (float32), contiguous
    // out_scores_host: num_frames * num_pairs * 3 floats (float32), contiguous
    void process_frames(const float* vertices_frames,
                        int num_frames,
                        int num_vertices,
                        float* out_scores_host);

private:
    Impl* impl_;
};
