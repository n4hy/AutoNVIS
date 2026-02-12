/**
 * @file test_state_vector.cpp
 * @brief Unit tests for state vector
 */

#include "state_vector.hpp"
#include <iostream>
#include <cassert>

using namespace autonvis;

void test_construction() {
    std::cout << "Test: State vector construction... ";

    const size_t n_lat = 5;
    const size_t n_lon = 6;
    const size_t n_alt = 7;

    StateVector state(n_lat, n_lon, n_alt);

    assert(state.n_lat() == n_lat);
    assert(state.n_lon() == n_lon);
    assert(state.n_alt() == n_alt);
    assert(state.n_grid() == n_lat * n_lon * n_alt);
    assert(state.dimension() == n_lat * n_lon * n_alt + 1);  // Grid + R_eff

    std::cout << "PASSED\n";
}

void test_get_set() {
    std::cout << "Test: Get/Set electron density... ";

    StateVector state(3, 4, 5);

    const double test_value = 1.5e11;  // el/m³
    state.set_ne(1, 2, 3, test_value);

    const double retrieved = state.get_ne(1, 2, 3);
    assert(std::abs(retrieved - test_value) < 1e-6);

    std::cout << "PASSED\n";
}

void test_vector_conversion() {
    std::cout << "Test: Vector conversion... ";

    StateVector state(2, 3, 4);

    // Set some values
    state.set_ne(0, 0, 0, 1e10);
    state.set_ne(1, 2, 3, 2e11);
    state.set_reff(150.0);

    // Convert to vector
    Eigen::VectorXd vec = state.to_vector();

    // Convert back
    StateVector state2(2, 3, 4);
    state2.from_vector(vec);

    // Check values match
    assert(std::abs(state2.get_ne(0, 0, 0) - 1e10) < 1e-6);
    assert(std::abs(state2.get_ne(1, 2, 3) - 2e11) < 1e-6);
    assert(std::abs(state2.get_reff() - 150.0) < 1e-6);

    std::cout << "PASSED\n";
}

void test_constraints() {
    std::cout << "Test: Physical constraints... ";

    StateVector state(2, 2, 2);

    // Set value below minimum
    state.set_ne(0, 0, 0, 1e5);  // Below MIN_NE
    state.apply_constraints();

    // Should be clamped to minimum
    assert(state.get_ne(0, 0, 0) >= 1e8);

    // Set value above maximum
    state.set_ne(1, 1, 1, 1e15);  // Above MAX_NE
    state.apply_constraints();

    // Should be clamped to maximum
    assert(state.get_ne(1, 1, 1) <= 1e13);

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== State Vector Tests ===\n";

    try {
        test_construction();
        test_get_set();
        test_vector_conversion();
        test_constraints();

        std::cout << "\nAll tests PASSED\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
