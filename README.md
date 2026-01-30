# OptiX Visibility

High-performance GPU visibility computation using **NVIDIA OptiX**, **CUDA**, and **pybind11**.
This package computes line-of-sight visibility and occlusion statistics between indexed vertex pairs over animated mesh frames using hardware-accelerated ray tracing.

---

## Features

* GPU-accelerated ray tracing via **OptiX 8**
* Fast dynamic mesh updates using GAS refitting
* Python bindings via **pybind11**
* Batched multi-frame processing
* Disk-sampled endpoint visibility integration
* CUDA reduction kernel for per-pair score aggregation
* Embedded PTX (no runtime shader files)

---

## What This Does

Given:

* A triangle mesh (vertices + faces)
* A subset of vertex indices
* Disk sampling parameters
* Multiple animated vertex frames

The system:

1. Builds an OptiX acceleration structure (GAS)
2. Generates all pairwise visibility rays between selected vertices
3. Casts disk-jittered rays per pair
4. Computes per-pair occlusion statistics
5. Reduces results on GPU
6. Returns per-frame visibility scores to Python

The mesh topology is static, but vertex positions may change every frame (fast refit path).

---

## Requirements

### Hardware

* NVIDIA GPU with RTX support recommended
* Compute Capability ≥ 7.0 (Volta or newer)

---

### Software

| Component     | Required                  |
| ------------- | ------------------------- |
| Linux         | Ubuntu 20.04+ recommended |
| Python        | 3.10+                     |
| CMake         | ≥ 3.24                    |
| CUDA Toolkit  | ≥ 11.8                    |
| NVIDIA Driver | R535+ recommended         |
| OptiX SDK     | 8.x                       |
| Compiler      | GCC 9+                    |

---

## Installing OptiX SDK

Download from NVIDIA:

[https://developer.nvidia.com/designworks/optix/download](https://developer.nvidia.com/designworks/optix/download)

Extract somewhere, for example:

```bash
~/NVIDIA-OptiX-SDK-8.0.0
```

Then export:

```bash
export OPTIX_ROOT=~/NVIDIA-OptiX-SDK-8.0.0
```

This is required at **build time only**.

---

## Installation (pip build)

This project builds using **scikit-build-core** (CMake backend).

### Upgrade build tools

```bash
pip install -U pip scikit-build-core pybind11 build
```

---

### Build & install locally

From project root:

```bash
pip install .
```

Editable dev mode:

```bash
pip install -e .
```

---

### Build wheel

```bash
python -m build -w
pip install dist/optix_visibility-*.whl
```

---

## CUDA Compiler Selection

If not specified, CMake automatically detects:

* `nvcc` from PATH
* `/usr/local/cuda/bin/nvcc`
* `CUDA_HOME`
* `CUDACXX`

Override manually:

```bash
pip install . -C cmake.args="-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc"
```

---

## OptiX PTX Architecture Setting

OptiX device programs are compiled to **PTX** and JIT compiled at runtime by the NVIDIA driver.

Default:

```
compute_70
```

You may override:

```bash
pip install . -C cmake.args="-DOPTIX_PTX_ARCH=compute_89"
```

### Recommended Values

| GPU      | Arch       |
| -------- | ---------- |
| RTX 20xx | compute_75 |
| RTX 30xx | compute_86 |
| RTX 40xx | compute_89 |

Using lower PTX targets improves portability.

---

## Python Usage

### Import

```python
import optix_visibility
```

---

### Constructor

```python
OptixVisibility(
    vertices0,
    num_vertices,
    faces_flat,
    index_list,
    disk_samples,
    disk_radius,
    device_id=0,
    endpoint_eps=1e-4,
    t_eps=1e-6,
    occupancy_nsamples=3,
    occupancy_jitter=1e-4,
    max_intersections=256,
    validation=False
)
```

---

### Parameters

#### Geometry

| Parameter    | Description                                              |
| ------------ | -------------------------------------------------------- |
| `vertices0`  | Initial vertex array (float32, shape Nx3)                |
| `faces_flat` | Triangle indices flattened (uint32 length multiple of 3) |
| `index_list` | Vertex indices to test pairwise visibility between       |

---

#### Sampling

| Parameter      | Description                            |
| -------------- | -------------------------------------- |
| `disk_samples` | Number of disk jitter samples per pair |
| `disk_radius`  | Radius of disk sampling at endpoints   |

---

#### Performance / Accuracy

| Parameter            | Description                          |
| -------------------- | ------------------------------------ |
| `endpoint_eps`       | Ray origin offset epsilon            |
| `t_eps`              | Intersection epsilon                 |
| `occupancy_nsamples` | Secondary stochastic samples         |
| `occupancy_jitter`   | Jitter radius                        |
| `max_intersections`  | Safety cap for traversal loops       |
| `validation`         | Enable OptiX validation layer (slow) |

---

### Processing Frames

```python
process_frames(
    vertices_frames,
    num_frames,
    num_vertices,
    out_scores
)
```

---

### Input Layout

`vertices_frames`:

```
shape = (num_frames, num_vertices, 3)
dtype = float32
```

`out_scores`:

```
shape = (num_frames, num_pairs, 3)
dtype = float32
```

---

### Returned Scores

Per pair:

```
[ visibility_ratio , hit_ratio , average_distance ]
```

(Exact meaning depends on your CUDA reduction kernel logic.)

---

## Example

```python
import numpy as np
from optix_visibility import OptixVisibility

vertices0 = mesh_vertices.astype(np.float32)
faces_flat = mesh_faces.flatten().astype(np.uint32)

idx_list = np.array([10, 50, 100, 300], dtype=np.uint32)

frames = animated_vertices.astype(np.float32)

out = np.zeros((frames.shape[0],
                len(idx_list)*(len(idx_list)-1)//2,
                3), dtype=np.float32)

vis = OptixVisibility(
    vertices0,
    vertices0.shape[0],
    faces_flat,
    idx_list,
    disk_samples=32,
    disk_radius=0.002
)

vis.process_frames(
    frames,
    frames.shape[0],
    vertices0.shape[0],
    out
)

print(out)
```

---

## Performance Notes

* GAS is built once, then **refitted per frame**
* Only vertex buffers are updated per frame
* OptiX pipeline and SBT are persistent
* CUDA streams are reused
* Reduction runs entirely on GPU

This makes it suitable for **real-time or batch animation visibility analysis**.

---

## Troubleshooting

### OptiX header not found

```
Could not find optix.h
```

Fix:

```bash
export OPTIX_ROOT=/path/to/NVIDIA-OptiX-SDK
```

---

### CUDA not detected

Check:

```bash
nvcc --version
```

If needed:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

---

### Illegal memory access / crash

Try:

```python
validation=True
```

This enables OptiX debug checks.

---

## Platform Support

| Platform | Status                        |
| -------- | ----------------------------- |
| Linux    | ✅ Fully supported             |
| Windows  | ⚠ Possible with minor changes |
| MacOS    | ❌ Not supported (no OptiX)    |

---

## License

This project is released under the MIT Licence.