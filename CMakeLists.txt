cmake_minimum_required(VERSION 3.24)

# If the user didn't specify a CUDA compiler, prefer the common default.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  set(_default_nvcc "/usr/local/cuda/bin/nvcc")
  if(EXISTS "${_default_nvcc}")
    set(CMAKE_CUDA_COMPILER "${_default_nvcc}" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
endif()

set(OPTIX_PTX_ARCH "compute_70" CACHE STRING "NVCC -arch for OptiX PTX compilation")

project(optix_visibility LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default GPU arch for the *CUDA kernels* in the extension (not the OptiX PTX).
# You can override at configure time: -DCMAKE_CUDA_ARCHITECTURES=89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---- OptiX headers ----
# You can point OPTIX_ROOT to the OptiX SDK root; we will use OPTIX_ROOT/include
find_path(OPTIX_INCLUDE_DIR optix.h
  HINTS ENV OPTIX_ROOT ENV OptiX_INSTALL_DIR
  PATH_SUFFIXES include
)
if(NOT OPTIX_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find optix.h. Set OPTIX_ROOT or OPTIX_INCLUDE_DIR.")
endif()

message(STATUS "OptiX include dir: ${OPTIX_INCLUDE_DIR}")

# ---- Compile OptiX device programs (src/optix_programs.cu) into PTX ----
set(PTX_DIR ${CMAKE_CURRENT_BINARY_DIR}/ptx)
file(MAKE_DIRECTORY ${PTX_DIR})

set(PTX_OUTPUT ${PTX_DIR}/optix_programs.ptx)

set(_ptx_host_flags "")
if(UNIX AND NOT APPLE)
  set(_ptx_host_flags "-Xcompiler=-fPIC")
endif()

add_custom_command(
  OUTPUT ${PTX_OUTPUT}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
          -std=c++17
          -lineinfo
          -I${CMAKE_CURRENT_SOURCE_DIR}/src
          -I${OPTIX_INCLUDE_DIR}
          -I${CUDAToolkit_INCLUDE_DIRS}
          -arch=${OPTIX_PTX_ARCH}
          -ptx ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_programs.cu
          -o ${PTX_OUTPUT}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_programs.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_visibility_params.h
  COMMENT "Compiling OptiX device programs to PTX"
)

# ---- Convert PTX to a header we can embed into the Python extension ----
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})
set(PTX_HEADER ${GEN_DIR}/optix_programs_ptx.h)

add_custom_command(
  OUTPUT ${PTX_HEADER}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ptx_to_header.py
          ${PTX_OUTPUT} ${PTX_HEADER} optix_programs_ptx
  DEPENDS ${PTX_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ptx_to_header.py
  COMMENT "Embedding PTX into a header"
)

add_custom_target(optix_ptx ALL DEPENDS ${PTX_HEADER})

# ---- Build Python extension ----
pybind11_add_module(optix_visibility MODULE
  src/bindings.cpp
  src/optix_visibility.cpp
  src/optix_function_table.cpp
  src/cuda_reduce.cu
  ${PTX_HEADER}
)

if(WIN32)
  target_compile_definitions(optix_visibility PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_dependencies(optix_visibility optix_ptx)

target_include_directories(optix_visibility PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OPTIX_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${GEN_DIR}
)

target_link_libraries(optix_visibility PRIVATE
  CUDA::cudart
  CUDA::cuda_driver
)

if(UNIX AND NOT APPLE)
  target_link_libraries(optix_visibility PRIVATE dl)
endif()


# Nice errors
target_compile_definitions(optix_visibility PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

include(GNUInstallDirs)

# Put the built extension inside the python package directory in the wheel:
install(TARGETS optix_visibility
  LIBRARY DESTINATION optix_visibility
  RUNTIME DESTINATION optix_visibility
  ARCHIVE DESTINATION optix_visibility
)