cmake_minimum_required(VERSION 3.20)
project(autonvis_raytracer VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)
find_package(pybind11 CONFIG)

# If pybind11 not found via CMake, fetch it
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
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

# Compiler optimizations
target_compile_options(raytracer PRIVATE
    -O3
    -march=native
    -ffast-math
    -DNDEBUG
)

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
