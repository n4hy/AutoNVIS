cmake_minimum_required(VERSION 3.20)
project(AutoNVIS_Assimilation VERSION 0.1.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(HDF5 COMPONENTS CXX)  # Optional for now
find_package(Boost COMPONENTS system thread)  # Optional for now

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    add_definitions(-DHAVE_HDF5)
endif()

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

# Source files (only include files that exist)
set(SOURCES
    src/sr_ukf.cpp
    src/state_vector.cpp
    src/sigma_points.cpp
    src/cholesky_update.cpp
    src/observation_model.cpp
    models/gauss_markov.cpp
)

# Header files
set(HEADERS
    include/sr_ukf.hpp
    include/state_vector.hpp
    include/sigma_points.hpp
    include/observation_model.hpp
    include/physics_model.hpp
    include/cholesky_update.hpp
)

# Create library
add_library(autonvis_assimilation STATIC ${SOURCES} ${HEADERS})

target_link_libraries(autonvis_assimilation
    Eigen3::Eigen
)

if(HDF5_FOUND)
    target_link_libraries(autonvis_assimilation ${HDF5_LIBRARIES})
endif()

if(Boost_FOUND)
    target_link_libraries(autonvis_assimilation ${Boost_LIBRARIES})
endif()

# Main executable (service)
add_executable(assimilation_service main.cpp)
target_link_libraries(assimilation_service autonvis_assimilation)

# Tests
enable_testing()

add_executable(test_sigma_points tests/test_sigma_points.cpp)
target_link_libraries(test_sigma_points autonvis_assimilation)
add_test(NAME SigmaPointsTest COMMAND test_sigma_points)

add_executable(test_cholesky tests/test_cholesky.cpp)
target_link_libraries(test_cholesky autonvis_assimilation)
add_test(NAME CholeskyTest COMMAND test_cholesky)

# SR-UKF integration test
add_executable(test_sr_ukf tests/test_sr_ukf.cpp)
target_link_libraries(test_sr_ukf autonvis_assimilation)
add_test(NAME SRUKFTest COMMAND test_sr_ukf)

# Localization demo (not a test, just demonstration)
add_executable(demo_localization tests/demo_localization.cpp)
target_link_libraries(demo_localization autonvis_assimilation)

# Phase 1 validation suite
add_executable(validation_phase1 tests/validation_phase1.cpp)
target_link_libraries(validation_phase1 autonvis_assimilation)

# Installation
install(TARGETS assimilation_service DESTINATION bin)
install(TARGETS autonvis_assimilation DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/autonvis)
