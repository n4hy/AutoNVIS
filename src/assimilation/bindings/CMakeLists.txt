cmake_minimum_required(VERSION 3.20)
project(autonvis_python_bindings)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

# If pybind11 is not found, try to fetch it
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include parent directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Source files from parent autonvis_assimilation library
set(SRUKF_SOURCES
    ../src/sr_ukf.cpp
    ../src/state_vector.cpp
    ../src/sigma_points.cpp
    ../src/cholesky_update.cpp
    ../src/observation_model.cpp
    ../models/gauss_markov.cpp
)

# Create Python module
pybind11_add_module(autonvis_srukf
    python_bindings.cpp
    ${SRUKF_SOURCES}
)

# Link against Eigen
target_link_libraries(autonvis_srukf PRIVATE Eigen3::Eigen)

# Set C++ standard
target_compile_features(autonvis_srukf PRIVATE cxx_std_17)

# Compiler flags
target_compile_options(autonvis_srukf PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3 -march=native>
)

# Install to site-packages
install(TARGETS autonvis_srukf
    LIBRARY DESTINATION ${Python3_SITEARCH}
)

# Also install to local python directory for development
set_target_properties(autonvis_srukf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python
)
