cmake_minimum_required(VERSION 3.20)

# Phase 18.4: Check for CUDA before project() to enable CUDA language
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    project(autonvis_raytracer VERSION 1.0.0 LANGUAGES CXX CUDA)
    set(HAVE_CUDA TRUE)
    message(STATUS "CUDA found - GPU ray tracing enabled")
else()
    project(autonvis_raytracer VERSION 1.0.0 LANGUAGES CXX)
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA not found - using CPU fallback for ray tracing")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard (if available)
if(HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Build type (default to Release for optimized builds)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)

# Use pybind11 from venv (required for NumPy 2.x compatibility)
# The venv pybind11 3.0.2+ has the correct ABI for NumPy 2.x
# System pybind11 (2.11.1) causes segfaults with NumPy 2.x
set(PYBIND11_VENV_PATH "/home/n4hy/AutoNVIS/venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

if(EXISTS ${PYBIND11_VENV_PATH})
    message(STATUS "Using venv pybind11 for NumPy 2.x compatibility")
    set(CMAKE_PREFIX_PATH ${PYBIND11_VENV_PATH} ${CMAKE_PREFIX_PATH})
endif()

find_package(pybind11 3.0.0 CONFIG)

# If pybind11 not found or too old, fetch version 3.0.2
if(NOT pybind11_FOUND)
    message(STATUS "Fetching pybind11 v3.0.2 for NumPy 2.x compatibility")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.2
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# OpenMP support for parallel CPU fallback
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel CPU ray tracing")
endif()

# Source files
set(RAYTRACER_SOURCES
    src/ray_tracer_3d.cpp
    src/ray_tracer_cuda_wrapper.cpp
)

# CUDA sources (if available)
if(HAVE_CUDA)
    list(APPEND RAYTRACER_SOURCES src/ray_tracer_cuda.cu)
endif()

# Python module
pybind11_add_module(raytracer
    ${RAYTRACER_SOURCES}
    bindings/python_bindings.cpp
)

# Add CUDA compile definitions
if(HAVE_CUDA)
    target_compile_definitions(raytracer PRIVATE HAVE_CUDA)
endif()

if(OpenMP_CXX_FOUND)
    target_compile_definitions(raytracer PRIVATE HAVE_OPENMP)
    target_link_libraries(raytracer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link Eigen3
target_link_libraries(raytracer PRIVATE Eigen3::Eigen)

# Compiler optimizations based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    target_compile_options(raytracer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O0 -g -Wall -Wextra>
    )
    if(HAVE_CUDA)
        target_compile_options(raytracer PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
        )
    endif()
else()
    message(STATUS "Building in Release mode with optimizations")
    target_compile_options(raytracer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math -DNDEBUG -funroll-loops -fno-strict-aliasing>
    )
    if(HAVE_CUDA)
        # CUDA optimizations for Release build
        target_compile_options(raytracer PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -DNDEBUG>
        )
        # Set CUDA architecture (default to common GPUs)
        set_property(TARGET raytracer PROPERTY CUDA_ARCHITECTURES "60;70;75;80;86")
    endif()
    # Enable link-time optimization for additional performance
    set_property(TARGET raytracer PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Install to Python package directory
install(TARGETS raytracer
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Print configuration
message(STATUS "Ray Tracer Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  Install directory: ${CMAKE_CURRENT_SOURCE_DIR}/python")
