#include <optix_device.h>
#include <cuda_runtime.h>
#include "optix_visibility_params.h"

extern "C" {
__constant__ Params params;
}

static __forceinline__ __device__ float3 f3_add(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static __forceinline__ __device__ float3 f3_sub(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static __forceinline__ __device__ float3 f3_mul(const float3& a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __device__ float f3_dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __device__ float3 f3_cross(const float3& a, const float3& b)
{
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
static __forceinline__ __device__ float f3_len(const float3& v)
{
    return sqrtf(f3_dot(v, v));
}
static __forceinline__ __device__ float3 f3_norm(const float3& v)
{
    const float l = f3_len(v);
    if (l > 1e-20f) return f3_mul(v, 1.0f / l);
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Simple RNG
static __forceinline__ __device__ unsigned int lcg(unsigned int& state)
{
    state = 1664525u * state + 1013904223u;
    return state;
}
static __forceinline__ __device__ float rnd01(unsigned int& state)
{
    return (lcg(state) & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}
static __forceinline__ __device__ float3 random_unit_vector(unsigned int& state)
{
    float3 v;
    do {
        v = make_float3(
            rnd01(state) * 2.0f - 1.0f,
            rnd01(state) * 2.0f - 1.0f,
            rnd01(state) * 2.0f - 1.0f);
    } while (f3_dot(v, v) < 1e-10f);
    return f3_norm(v);
}

// Trace for closest hit; returns (hit?, t_hit)
static __forceinline__ __device__ bool trace_closest(const float3& origin,
                                                     const float3& dir,
                                                     float tmin,
                                                     float tmax,
                                                     float& t_hit)
{
    unsigned int p0 = 0u; // hit flag
    unsigned int p1 = 0u; // t_hit (float as uint)

    optixTrace(
        params.handle,
        origin,
        dir,
        tmin,
        tmax,
        0.0f,                // rayTime
        0xFFu,               // visibility mask
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                   // sbtOffset
        1,                   // sbtStride
        0,                   // missSbtIndex
        p0, p1
    );

    if (p0) {
        t_hit = __uint_as_float(p1);
        return true;
    }
    return false;
}

// Odd/even parity test (single sample)
static __forceinline__ __device__ int point_inside_once(const float3& p, unsigned int seed)
{
    unsigned int s = seed;
    const float3 dir = random_unit_vector(s);

    int count = 0;
    float tmin = 1e-4f;
    const float tmax = 1e16f;

    for (int i = 0; i < params.max_intersections; ++i) {
        float t;
        if (!trace_closest(p, dir, tmin, tmax, t))
            break;
        count++;
        tmin = t + params.t_eps;
    }
    return (count & 1);
}

// Majority vote over nsamples with jittered points + randomized directions
static __forceinline__ __device__ int point_inside_majority(const float3& p, unsigned int seed)
{
    const int ns = params.occupancy_nsamples > 0 ? params.occupancy_nsamples : 1;
    int inside_votes = 0;

    unsigned int s = seed;

    for (int i = 0; i < ns; ++i) {
        // jitter point
        float3 jitter = make_float3(
            (rnd01(s) * 2.0f - 1.0f) * params.occupancy_jitter,
            (rnd01(s) * 2.0f - 1.0f) * params.occupancy_jitter,
            (rnd01(s) * 2.0f - 1.0f) * params.occupancy_jitter);

        inside_votes += point_inside_once(f3_add(p, jitter), s + 0x9e3779b9u * (unsigned)i);
    }

    // majority
    return (inside_votes * 2 >= ns) ? 1 : 0;
}

// Thickness accumulation using repeated closest-hit along the segment.
// Mirrors your Open3D logic: toggle inside/outside each hit and sum inside lengths.
static __forceinline__ __device__ float segment_thickness(const float3& p0,
                                                         const float3& p1,
                                                         unsigned int seed)
{
    const float3 d = f3_sub(p1, p0);
    const float L = f3_len(d);
    if (L <= 1e-12f) return 0.0f;

    const float3 u = f3_mul(d, 1.0f / L);

    const float shrink = fminf(params.endpoint_eps, 0.49f * L);

    const float3 o = f3_add(p0, f3_mul(u, shrink));
    const float3 e = f3_sub(p1, f3_mul(u, shrink));

    const float3 seg = f3_sub(e, o);
    const float L2 = f3_len(seg);
    if (L2 <= 1e-12f) return 0.0f;

    const float3 dir = f3_mul(seg, 1.0f / L2);

    // emulate filtering near the endpoint
    const float tmax = L2 - params.t_eps;
    if (tmax <= params.t_eps) return 0.0f;

    const int inside0 = point_inside_majority(o, seed);

    float Lin = 0.0f;
    float prev = 0.0f;
    bool inside = (inside0 != 0);

    for (int i = 0; i < params.max_intersections; ++i) {
        float t;
        const float tmin = prev + params.t_eps;
        if (!trace_closest(o, dir, tmin, tmax, t))
            break;

        if (inside) {
            Lin += (t - prev);
        }
        inside = !inside;
        prev = t;
    }

    if (inside) {
        Lin += (tmax - prev);
    }

    return (Lin > 0.0f) ? Lin : 0.0f;
}

extern "C" __global__ void __raygen__rg()
{
    const int idx = (int)optixGetLaunchIndex().x;

    const int K = params.diskSamples;
    const int M = params.numPairs;

    const int m = idx / K;
    const int k = idx - m * K;

    if (m >= M) return;

    const uint2 pr = params.pairs[m];
    const float3 p0 = params.vertices[pr.x];
    const float3 p1 = params.vertices[pr.y];

    // Segment direction (for basis)
    const float3 d = f3_sub(p1, p0);
    const float L = f3_len(d);
    if (L <= 1e-12f) {
        params.L_inside[m * K + k] = 0.0f;
        return;
    }
    const float3 u = f3_mul(d, 1.0f / L);

    // Per-segment perpendicular basis v,w (same logic as your Python)
    float3 a = make_float3(1.0f, 0.0f, 0.0f);
    if (fabsf(u.x) > 0.9f) a = make_float3(0.0f, 1.0f, 0.0f);

    float3 v = f3_cross(u, a);
    v = f3_norm(v);
    float3 w = f3_cross(u, v);

    const float2 xy = params.disk[k];
    const float3 off = f3_add(f3_mul(v, xy.x), f3_mul(w, xy.y));

    const float3 p0k = f3_add(p0, off);
    const float3 p1k = f3_add(p1, off);

    unsigned int seed = (unsigned int)(idx + 1) * 9781u + 0x68bc21ebu;
    const float Lin = segment_thickness(p0k, p1k, seed);

    params.L_inside[m * K + k] = Lin;
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0u);
    optixSetPayload_1(__float_as_uint(0.0f));
}

extern "C" __global__ void __closesthit__ch()
{
    optixSetPayload_0(1u);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}
