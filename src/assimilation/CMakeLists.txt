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
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Boost REQUIRED COMPONENTS system thread)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/sr_ukf.cpp
    src/state_vector.cpp
    src/sigma_points.cpp
    src/observation_model.cpp
    src/physics_model.cpp
    src/cholesky_update.cpp
    models/iri2020_wrapper.cpp
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
    ${HDF5_LIBRARIES}
    ${Boost_LIBRARIES}
)

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

add_executable(test_sr_ukf tests/test_sr_ukf.cpp)
target_link_libraries(test_sr_ukf autonvis_assimilation)
add_test(NAME SRUKFTest COMMAND test_sr_ukf)

# Installation
install(TARGETS assimilation_service DESTINATION bin)
install(TARGETS autonvis_assimilation DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/autonvis)
