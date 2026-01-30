#include "optix_visibility.h"
#include "optix_visibility_params.h"
#include "cuda_reduce.h"

// OptiX
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda.h>

// Embedded PTX
#include "optix_programs_ptx.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace {

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*)
{
    std::cerr << "[" << level << "][" << tag << "] " << message << "\n";
}

inline void throw_runtime(const std::string& s) { throw std::runtime_error(s); }

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::ostringstream _oss; \
        _oss << "CUDA error: " << cudaGetErrorString(_e) << " (" << (int)_e << ") at " << __FILE__ << ":" << __LINE__; \
        throw_runtime(_oss.str()); \
    } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char* _name = nullptr; \
        const char* _str  = nullptr; \
        cuGetErrorName(_r, &_name); \
        cuGetErrorString(_r, &_str); \
        std::ostringstream _oss; \
        _oss << "CUDA Driver error: " << (_name ? _name : "<?>") << " / " << (_str ? _str : "<?>") \
             << " at " << __FILE__ << ":" << __LINE__; \
        throw_runtime(_oss.str()); \
    } \
} while(0)

#define OPTIX_CHECK(call) do { \
    OptixResult _r = (call); \
    if (_r != OPTIX_SUCCESS) { \
        std::ostringstream _oss; \
        _oss << "OptiX error: " << (int)_r << " at " << __FILE__ << ":" << __LINE__; \
        throw_runtime(_oss.str()); \
    } \
} while(0)

#define OPTIX_CHECK_LOG(call) do { \
    char log[4096]; \
    size_t logSize = sizeof(log); \
    OptixResult _r = (call); \
    if (_r != OPTIX_SUCCESS) { \
        std::ostringstream _oss; \
        _oss << "OptiX error: " << (int)_r << " at " << __FILE__ << ":" << __LINE__ << "\nLog: " << log; \
        throw_runtime(_oss.str()); \
    } \
    if (logSize > 1) { \
        /* uncomment if you want compile logs every time */ \
        /* std::cerr << log << std::endl; */ \
    } \
} while(0)

static CUdeviceptr cuda_alloc(std::size_t bytes)
{
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return reinterpret_cast<CUdeviceptr>(ptr);
}
static void cuda_free(CUdeviceptr p)
{
    if (p) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(p)));
}

template <typename T>
static void cuda_upload(CUdeviceptr dst, const T* src, std::size_t count, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dst), src, sizeof(T) * count,
                               cudaMemcpyHostToDevice, stream));
}

template <typename T>
static void cuda_download(T* dst, CUdeviceptr src, std::size_t count, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<void*>(src), sizeof(T) * count,
                               cudaMemcpyDeviceToHost, stream));
}

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RaygenData {};
struct MissData {};
struct HitgroupData {};

} // namespace

struct OptixVisibility::Impl
{
    int deviceId = 0;

    int numVertices = 0;
    int numFaces = 0;
    int numPairs = 0;
    int diskSamples = 0;

    float endpoint_eps = 1e-4f;
    float t_eps = 1e-6f;
    int occupancy_nsamples = 3;
    float occupancy_jitter = 1e-4f;
    int max_intersections = 256;
    bool validation = false;

    cudaStream_t stream = nullptr;

    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitPG = nullptr;

    OptixShaderBindingTable sbt = {};

    // Device buffers
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices  = 0;
    CUdeviceptr d_pairs    = 0;
    CUdeviceptr d_disk     = 0;
    CUdeviceptr d_params   = 0;

    CUdeviceptr d_L_inside = 0; // numPairs*diskSamples
    CUdeviceptr d_scores   = 0; // numPairs*3

    // GAS
    OptixTraversableHandle gas_handle = 0;
    CUdeviceptr d_gas_output = 0;
    CUdeviceptr d_gas_temp   = 0;
    std::size_t gas_output_size = 0;
    std::size_t gas_temp_size   = 0;

    // Build input storage (must live across refits)
    OptixBuildInput build_input = {};
    OptixAccelBuildOptions accel_options = {};
    OptixAccelBufferSizes gas_sizes = {};
    CUdeviceptr vertex_buffers[1] = {0};

    std::vector<uint32_t> geom_flags; // size=1, required non-null

    // SBT records
    CUdeviceptr d_raygen_record = 0;
    CUdeviceptr d_miss_record   = 0;
    CUdeviceptr d_hit_record    = 0;

    ~Impl()
    {
        // Sync before teardown
        if (stream) cudaStreamSynchronize(stream);

        // OptiX destroy
        if (pipeline) OPTIX_CHECK(optixPipelineDestroy(pipeline));
        if (raygenPG) OPTIX_CHECK(optixProgramGroupDestroy(raygenPG));
        if (missPG)   OPTIX_CHECK(optixProgramGroupDestroy(missPG));
        if (hitPG)    OPTIX_CHECK(optixProgramGroupDestroy(hitPG));
        if (module)   OPTIX_CHECK(optixModuleDestroy(module));
        if (context)  OPTIX_CHECK(optixDeviceContextDestroy(context));

        // CUDA free
        cuda_free(d_raygen_record);
        cuda_free(d_miss_record);
        cuda_free(d_hit_record);

        cuda_free(d_gas_output);
        cuda_free(d_gas_temp);

        cuda_free(d_vertices);
        cuda_free(d_indices);
        cuda_free(d_pairs);
        cuda_free(d_disk);
        cuda_free(d_params);

        cuda_free(d_L_inside);
        cuda_free(d_scores);

        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    }
};

static std::vector<uint2> make_pairs(const std::vector<uint32_t>& idx_list)
{
    const int P = (int)idx_list.size();
    const int M = P * (P - 1) / 2;
    std::vector<uint2> pairs;
    pairs.reserve(M);

    for (int i = 0; i < P; ++i) {
        for (int j = i + 1; j < P; ++j) {
            uint2 pr;
            pr.x = idx_list[i];
            pr.y = idx_list[j];
            pairs.push_back(pr);
        }
    }
    return pairs;
}

static std::vector<float2> disk_samples_fibonacci(int K, float radius)
{
    constexpr float pi = 3.14159265358979323846f;
    const float ga = pi * (3.0f - std::sqrt(5.0f));

    std::vector<float2> out;
    out.resize(K);

    for (int i = 0; i < K; ++i) {
        const float r = radius * std::sqrt((i + 0.5f) / (float)K);
        const float theta = i * ga;
        float2 p;
        p.x = r * std::cos(theta);
        p.y = r * std::sin(theta);
        out[i] = p;
    }
    return out;
}

static void create_optix_context(OptixVisibility::Impl* impl)
{
    CUDA_CHECK(cudaSetDevice(impl->deviceId));
    CUDA_CHECK(cudaFree(0)); // ensure CUDA context

    CUcontext cuCtx = 0;
    CU_CHECK(cuCtxGetCurrent(&cuCtx));
    if (!cuCtx) {
        throw_runtime("No current CUcontext after cudaFree(0).");
    }

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = context_log_cb;
    options.logCallbackLevel = 1;
    options.logCallbackData = nullptr;

    // OptiX 8.x: debug exceptions moved into validation mode, keep OFF for speed.
    options.validationMode = impl->validation
        ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
        : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &impl->context));

    CUDA_CHECK(cudaStreamCreate(&impl->stream));
}

static void create_pipeline(OptixVisibility::Impl* impl)
{
    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 2;    // hit flag + t
    pipeline_options.numAttributeValues = 2;  // triangle barycentrics
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";

    OPTIX_CHECK_LOG(
        optixModuleCreate(
            impl->context,
            &module_options,
            &pipeline_options,
            optix_programs_ptx,
            optix_programs_ptx_len,
            nullptr, nullptr, // log, logSize are provided by macro
            &impl->module
        )
    );

    // Program groups
    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc rg_desc = {};
    rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_desc.raygen.module = impl->module;
    rg_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(impl->context, &rg_desc, 1, &pg_options,
                                           nullptr, nullptr, &impl->raygenPG));

    OptixProgramGroupDesc ms_desc = {};
    ms_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_desc.miss.module = impl->module;
    ms_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(impl->context, &ms_desc, 1, &pg_options,
                                           nullptr, nullptr, &impl->missPG));

    OptixProgramGroupDesc hg_desc = {};
    hg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hg_desc.hitgroup.moduleCH = impl->module;
    hg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hg_desc.hitgroup.moduleAH = nullptr;
    hg_desc.hitgroup.entryFunctionNameAH = nullptr;
    hg_desc.hitgroup.moduleIS = nullptr;
    hg_desc.hitgroup.entryFunctionNameIS = nullptr;
    OPTIX_CHECK_LOG(optixProgramGroupCreate(impl->context, &hg_desc, 1, &pg_options,
                                           nullptr, nullptr, &impl->hitPG));

    // Pipeline
    OptixProgramGroup pgs[] = { impl->raygenPG, impl->missPG, impl->hitPG };

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;

    OPTIX_CHECK_LOG(optixPipelineCreate(impl->context, &pipeline_options, &link_options,
                                        pgs, 3, nullptr, nullptr, &impl->pipeline));

    // Stack sizes
    OptixStackSizes ss = {};
    //OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->raygenPG, &ss));
    //OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->missPG, &ss));
    //OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->hitPG, &ss));

    OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->raygenPG, &ss, impl->pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->missPG,   &ss, impl->pipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(impl->hitPG,    &ss, impl->pipeline));

    unsigned int dc_trav = 0, dc_state = 0, cont = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&ss, /*maxTraceDepth=*/1, /*maxCCDepth=*/0, /*maxDCDepth=*/0,
                                           &dc_trav, &dc_state, &cont));

    OPTIX_CHECK(optixPipelineSetStackSize(impl->pipeline, dc_trav, dc_state, cont, /*maxTraversableDepth=*/1));

    // SBT
    SbtRecord<RaygenData> rg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl->raygenPG, &rg));

    SbtRecord<MissData> ms = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl->missPG, &ms));

    SbtRecord<HitgroupData> hg = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(impl->hitPG, &hg));

    impl->d_raygen_record = cuda_alloc(sizeof(rg));
    impl->d_miss_record   = cuda_alloc(sizeof(ms));
    impl->d_hit_record    = cuda_alloc(sizeof(hg));

    cuda_upload(impl->d_raygen_record, &rg, 1, impl->stream);
    cuda_upload(impl->d_miss_record,   &ms, 1, impl->stream);
    cuda_upload(impl->d_hit_record,    &hg, 1, impl->stream);

    impl->sbt = {};
    impl->sbt.raygenRecord = impl->d_raygen_record;
    impl->sbt.missRecordBase = impl->d_miss_record;
    impl->sbt.missRecordStrideInBytes = sizeof(SbtRecord<MissData>);
    impl->sbt.missRecordCount = 1;
    impl->sbt.hitgroupRecordBase = impl->d_hit_record;
    impl->sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitgroupData>);
    impl->sbt.hitgroupRecordCount = 1;

    CUDA_CHECK(cudaStreamSynchronize(impl->stream));
}

static void build_gas(OptixVisibility::Impl* impl)
{
    impl->geom_flags = { OPTIX_GEOMETRY_FLAG_NONE };

    impl->vertex_buffers[0] = impl->d_vertices;

    impl->build_input = {};
    impl->build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    OptixBuildInputTriangleArray& tri = impl->build_input.triangleArray;
    tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.vertexStrideInBytes = sizeof(float3);
    tri.numVertices = impl->numVertices;
    tri.vertexBuffers = impl->vertex_buffers;

    tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri.indexStrideInBytes = sizeof(uint3);
    tri.numIndexTriplets = impl->numFaces;
    tri.indexBuffer = impl->d_indices;

    tri.flags = impl->geom_flags.data();    // MUST be non-null. :contentReference[oaicite:4]{index=4}
    tri.numSbtRecords = 1;

    impl->accel_options = {};
    impl->accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    impl->accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        impl->context,
        &impl->accel_options,
        &impl->build_input,
        1,
        &impl->gas_sizes));

    impl->gas_output_size = impl->gas_sizes.outputSizeInBytes;
    impl->gas_temp_size = std::max(impl->gas_sizes.tempSizeInBytes, impl->gas_sizes.tempUpdateSizeInBytes);

    impl->d_gas_output = cuda_alloc(impl->gas_output_size);
    impl->d_gas_temp   = cuda_alloc(impl->gas_temp_size);

    OPTIX_CHECK(optixAccelBuild(
        impl->context,
        impl->stream,
        &impl->accel_options,
        &impl->build_input,
        1,
        impl->d_gas_temp,
        impl->gas_temp_size,
        impl->d_gas_output,
        impl->gas_output_size,
        &impl->gas_handle,
        nullptr,
        0));

    CUDA_CHECK(cudaStreamSynchronize(impl->stream));
}

static void refit_gas(OptixVisibility::Impl* impl)
{
    impl->accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

    OPTIX_CHECK(optixAccelBuild(
        impl->context,
        impl->stream,
        &impl->accel_options,
        &impl->build_input,
        1,
        impl->d_gas_temp,
        impl->gas_temp_size,
        impl->d_gas_output,
        impl->gas_output_size,
        &impl->gas_handle,
        nullptr,
        0));
}

OptixVisibility::OptixVisibility(const float* vertices0, int numVertices,
                                 const std::vector<uint32_t>& faces_flat,
                                 const std::vector<uint32_t>& idx_list,
                                 int diskSamples,
                                 float diskRadius,
                                 int deviceId,
                                 float endpoint_eps,
                                 float t_eps,
                                 int occupancy_nsamples,
                                 float occupancy_jitter,
                                 int max_intersections,
                                 bool validation)
{
    impl_ = new Impl();
    impl_->deviceId = deviceId;

    impl_->numVertices = numVertices;
    impl_->diskSamples = diskSamples;

    impl_->endpoint_eps = endpoint_eps;
    impl_->t_eps = t_eps;
    impl_->occupancy_nsamples = occupancy_nsamples;
    impl_->occupancy_jitter = occupancy_jitter;
    impl_->max_intersections = max_intersections;
    impl_->validation = validation;

    if (faces_flat.size() % 3 != 0) {
        throw_runtime("faces_flat length must be multiple of 3.");
    }
    impl_->numFaces = (int)(faces_flat.size() / 3);

    // Build pairs
    std::vector<uint2> pairs = make_pairs(idx_list);
    impl_->numPairs = (int)pairs.size();

    // Disk samples
    std::vector<float2> disk = disk_samples_fibonacci(diskSamples, diskRadius);

    // Init OptiX & pipeline
    create_optix_context(impl_);
    create_pipeline(impl_);

    // Allocate + upload static buffers (indices, pairs, disk)
    impl_->d_indices = cuda_alloc(sizeof(uint3) * impl_->numFaces);
    // Faces_flat is uint32 triplets; copy as uint3
    cuda_upload(impl_->d_indices, reinterpret_cast<const uint3*>(faces_flat.data()), impl_->numFaces, impl_->stream);

    impl_->d_pairs = cuda_alloc(sizeof(uint2) * impl_->numPairs);
    cuda_upload(impl_->d_pairs, pairs.data(), impl_->numPairs, impl_->stream);

    impl_->d_disk = cuda_alloc(sizeof(float2) * impl_->diskSamples);
    cuda_upload(impl_->d_disk, disk.data(), impl_->diskSamples, impl_->stream);

    // Dynamic buffers
    impl_->d_vertices = cuda_alloc(sizeof(float3) * impl_->numVertices);
    cuda_upload(impl_->d_vertices, reinterpret_cast<const float3*>(vertices0), impl_->numVertices, impl_->stream);

    impl_->d_L_inside = cuda_alloc(sizeof(float) * (std::size_t)impl_->numPairs * (std::size_t)impl_->diskSamples);
    impl_->d_scores   = cuda_alloc(sizeof(float) * (std::size_t)impl_->numPairs * 3ull);

    impl_->d_params = cuda_alloc(sizeof(Params));

    // Build GAS once
    build_gas(impl_);

    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

OptixVisibility::~OptixVisibility()
{
    delete impl_;
    impl_ = nullptr;
}

int OptixVisibility::num_pairs() const { return impl_->numPairs; }
int OptixVisibility::disk_samples() const { return impl_->diskSamples; }

void OptixVisibility::process_frames(const float* vertices_frames,
                                     int num_frames,
                                     int num_vertices,
                                     float* out_scores_host)
{
    if (num_vertices != impl_->numVertices) {
        throw_runtime("num_vertices mismatch vs scene initialization.");
    }
    if (num_frames <= 0) return;

    const std::size_t verts_bytes = sizeof(float3) * (std::size_t)impl_->numVertices;
    const int launch_width = impl_->numPairs * impl_->diskSamples;

    Params hparams = {};
    hparams.handle = impl_->gas_handle;
    hparams.vertices = reinterpret_cast<const float3*>(impl_->d_vertices);
    hparams.pairs = reinterpret_cast<const uint2*>(impl_->d_pairs);
    hparams.numPairs = impl_->numPairs;
    hparams.disk = reinterpret_cast<const float2*>(impl_->d_disk);
    hparams.diskSamples = impl_->diskSamples;
    hparams.endpoint_eps = impl_->endpoint_eps;
    hparams.t_eps = impl_->t_eps;
    hparams.occupancy_nsamples = impl_->occupancy_nsamples;
    hparams.occupancy_jitter = impl_->occupancy_jitter;
    hparams.max_intersections = impl_->max_intersections;
    hparams.L_inside = reinterpret_cast<float*>(impl_->d_L_inside);

    for (int f = 0; f < num_frames; ++f) {
        const float* vf = vertices_frames + (std::size_t)f * (std::size_t)impl_->numVertices * 3ull;

        // Upload vertices
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(impl_->d_vertices), vf, verts_bytes,
                                   cudaMemcpyHostToDevice, impl_->stream));

        // Refit GAS (faces constant, vertex buffer updated)
        refit_gas(impl_);

        // Update params handle (should be stable, but keep correct)
        hparams.handle = impl_->gas_handle;

        // Upload params
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(impl_->d_params), &hparams, sizeof(Params),
                                   cudaMemcpyHostToDevice, impl_->stream));

        // Launch OptiX to compute L_inside per ray
        OPTIX_CHECK(optixLaunch(
            impl_->pipeline,
            impl_->stream,
            impl_->d_params,
            sizeof(Params),
            &impl_->sbt,
            launch_width, 1, 1));

        // Reduce per pair into scores
        launch_reduce_scores(reinterpret_cast<const float*>(impl_->d_L_inside),
                             impl_->numPairs,
                             impl_->diskSamples,
                             reinterpret_cast<float*>(impl_->d_scores),
                             impl_->stream);

        // Download scores for this frame
        cuda_download(out_scores_host + (std::size_t)f * (std::size_t)impl_->numPairs * 3ull,
                      impl_->d_scores,
                      (std::size_t)impl_->numPairs * 3ull,
                      impl_->stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}
