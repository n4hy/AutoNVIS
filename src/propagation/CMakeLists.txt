cmake_minimum_required(VERSION 3.20)
project(autonvis_raytracer VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Source files
set(RAYTRACER_SOURCES
    src/ray_tracer_3d.cpp
)

# Python module
pybind11_add_module(raytracer
    ${RAYTRACER_SOURCES}
    bindings/python_bindings.cpp
)

# Link Eigen3
target_link_libraries(raytracer PRIVATE Eigen3::Eigen)

# Compiler optimizations based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    target_compile_options(raytracer PRIVATE
        -O0
        -g
        -Wall
        -Wextra
    )
else()
    message(STATUS "Building in Release mode with optimizations")
    target_compile_options(raytracer PRIVATE
        -O3
        -march=native
        -ffast-math
        -DNDEBUG
        -funroll-loops
        -fno-strict-aliasing
    )
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
