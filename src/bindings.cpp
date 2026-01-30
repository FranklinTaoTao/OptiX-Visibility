#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "optix_visibility.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

static void require(bool cond, const char* msg)
{
    if (!cond) throw py::value_error(msg);
}

static std::vector<uint32_t> flatten_faces(py::array_t<uint32_t, py::array::c_style | py::array::forcecast> faces)
{
    auto info = faces.request();
    require(info.ndim == 2 && info.shape[1] == 3, "faces must be (F,3) uint32");
    const size_t F = (size_t)info.shape[0];
    std::vector<uint32_t> flat(F * 3);
    std::memcpy(flat.data(), info.ptr, F * 3 * sizeof(uint32_t));
    return flat;
}

static std::pair<int,int> parse_vertices_shape(const py::buffer_info& info)
{
    // returns (num_frames, num_vertices)
    if (info.ndim == 2) {
        require(info.shape[1] == 3, "vertices must be (N,3) float32 or (T,N,3) float32");
        return {1, (int)info.shape[0]};
    } else if (info.ndim == 3) {
        require(info.shape[2] == 3, "vertices must be (T,N,3) float32");
        return {(int)info.shape[0], (int)info.shape[1]};
    }
    throw py::value_error("vertices must be 2D or 3D float32 array");
}

static py::array_t<float> disk_samples_fibonacci_py(int K, float radius)
{
    // same as your Python
    constexpr float pi = 3.14159265358979323846f;
    const float ga = pi * (3.0f - std::sqrt(5.0f));

    py::array_t<float> out({K, 2});
    auto r = out.mutable_unchecked<2>();

    for (int i = 0; i < K; ++i) {
        const float rr = radius * std::sqrt((i + 0.5f) / (float)K);
        const float th = i * ga;
        r(i,0) = rr * std::cos(th);
        r(i,1) = rr * std::sin(th);
    }
    return out;
}

static py::tuple pair_indices_py(const std::vector<uint32_t>& idx_list)
{
    const int P = (int)idx_list.size();
    const int M = P * (P - 1) / 2;

    py::array_t<uint32_t> a(M);
    py::array_t<uint32_t> b(M);

    auto aa = a.mutable_unchecked<1>();
    auto bb = b.mutable_unchecked<1>();

    int m = 0;
    for (int i = 0; i < P; ++i) {
        for (int j = i + 1; j < P; ++j) {
            aa(m) = idx_list[i];
            bb(m) = idx_list[j];
            m++;
        }
    }

    return py::make_tuple(a, b);
}

PYBIND11_MODULE(optix_visibility, m)
{
    m.doc() = "OptiX 8.x visibility/thickness batch processing (faces constant, vertices refit per frame)";

    m.def("disk_samples_fibonacci", &disk_samples_fibonacci_py,
          py::arg("K"), py::arg("radius"),
          "Generate (K,2) Fibonacci disk samples, float32.");

    m.def("pair_indices", &pair_indices_py,
          py::arg("idx_list"),
          "Return (pair_a, pair_b) like your numpy triu_indices pairing.");

    py::class_<OptixVisibility>(m, "OptixVisibility")
        .def(py::init([](
            py::array_t<float, py::array::c_style | py::array::forcecast> vertices0,
            py::array_t<uint32_t, py::array::c_style | py::array::forcecast> faces,
            std::vector<uint32_t> idx_list,
            int disk_samples,
            float disk_radius,
            int device_id,
            float endpoint_eps,
            float t_eps,
            int occupancy_nsamples,
            float occupancy_jitter,
            int max_intersections,
            bool validation
        ){
            auto vinfo = vertices0.request();
            require(vinfo.ndim == 2 && vinfo.shape[1] == 3, "vertices0 must be (N,3) float32");
            const int Nv = (int)vinfo.shape[0];

            auto faces_flat = flatten_faces(faces);

            return new OptixVisibility(
                (const float*)vinfo.ptr, Nv,
                faces_flat,
                idx_list,
                disk_samples,
                disk_radius,
                device_id,
                endpoint_eps,
                t_eps,
                occupancy_nsamples,
                occupancy_jitter,
                max_intersections,
                validation
            );
        }),
        py::arg("vertices0"),
        py::arg("faces"),
        py::arg("idx_list"),
        py::arg("disk_samples") = 100,
        py::arg("disk_radius") = 0.1f,
        py::arg("device_id") = 0,
        py::arg("endpoint_eps") = 1e-4f,
        py::arg("t_eps") = 1e-6f,
        py::arg("occupancy_nsamples") = 3,
        py::arg("occupancy_jitter") = 1e-4f,
        py::arg("max_intersections") = 256,
        py::arg("validation") = false
        )

        .def_property_readonly("num_pairs", &OptixVisibility::num_pairs)
        .def_property_readonly("disk_samples", &OptixVisibility::disk_samples)

        .def("process", [](OptixVisibility& self,
                           py::array_t<float, py::array::c_style | py::array::forcecast> vertices_frames)
        {
            auto info = vertices_frames.request();
            auto [T, Nv] = parse_vertices_shape(info);

            const float* ptr = (const float*)info.ptr;

            py::array_t<float> out({T, self.num_pairs(), 3});
            auto oinfo = out.request();

            self.process_frames(ptr, T, Nv, (float*)oinfo.ptr);
            return out;
        }, py::arg("vertices_frames"),
        "Process vertices frames: (T,N,3) or (N,3) float32 -> (T,num_pairs,3) float32.");

    // Convenience: build + process in one call
    m.def("process_frames",
          [](
              py::array_t<float, py::array::c_style | py::array::forcecast> vertices_frames,
              py::array_t<uint32_t, py::array::c_style | py::array::forcecast> faces,
              std::vector<uint32_t> idx_list,
              int disk_samples,
              float disk_radius,
              int device_id,
              float endpoint_eps,
              float t_eps,
              int occupancy_nsamples,
              float occupancy_jitter,
              int max_intersections,
              bool validation
          ){
              auto vinfo = vertices_frames.request();
              auto [T, Nv] = parse_vertices_shape(vinfo);

              // Extract frame0 pointer
              const float* vf = (const float*)vinfo.ptr;
              const float* v0 = vf; // first frame is at base

              auto faces_flat = flatten_faces(faces);

              OptixVisibility vis(
                  v0, Nv,
                  faces_flat,
                  idx_list,
                  disk_samples,
                  disk_radius,
                  device_id,
                  endpoint_eps,
                  t_eps,
                  occupancy_nsamples,
                  occupancy_jitter,
                  max_intersections,
                  validation
              );

              py::array_t<float> out({T, vis.num_pairs(), 3});
              auto oinfo = out.request();
              vis.process_frames(vf, T, Nv, (float*)oinfo.ptr);
              return out;
          },
          py::arg("vertices_frames"),
          py::arg("faces"),
          py::arg("idx_list"),
          py::arg("disk_samples") = 100,
          py::arg("disk_radius") = 0.1f,
          py::arg("device_id") = 0,
          py::arg("endpoint_eps") = 1e-4f,
          py::arg("t_eps") = 1e-6f,
          py::arg("occupancy_nsamples") = 3,
          py::arg("occupancy_jitter") = 1e-4f,
          py::arg("max_intersections") = 256,
          py::arg("validation") = false,
          "One-shot helper: build scene from faces + first frame, refit per frame, return (T,num_pairs,3).");
}
