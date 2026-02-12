/**
 * @file test_sr_ukf.cpp
 * @brief Integration tests for SR-UKF
 */

#include "sr_ukf.hpp"
#include "observation_model.hpp"
#include "physics_model.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace autonvis;

void test_filter_initialization() {
    std::cout << "Test: Filter initialization... ";

    const size_t n_lat = 5;
    const size_t n_lon = 6;
    const size_t n_alt = 7;

    SquareRootUKF filter(n_lat, n_lon, n_alt);

    // Initialize with background state
    StateVector initial_state(n_lat, n_lon, n_alt);

    // Set some typical ionospheric values
    for (size_t i = 0; i < n_lat; ++i) {
        for (size_t j = 0; j < n_lon; ++j) {
            for (size_t k = 0; k < n_alt; ++k) {
                // Typical F2 peak density ~1e12 el/m³
                initial_state.set_ne(i, j, k, 5e11);
            }
        }
    }
    initial_state.set_reff(100.0);  // Typical sunspot number

    // Initialize sqrt covariance
    const size_t dim = initial_state.dimension();
    Eigen::MatrixXd initial_sqrt_cov = Eigen::MatrixXd::Identity(dim, dim) * 1e5;

    filter.initialize(initial_state, initial_sqrt_cov);

    // Verify state was set
    const StateVector& state = filter.get_state();
    assert(std::abs(state.get_reff() - 100.0) < 1e-6);

    std::cout << "PASSED\n";
}

void test_predict_step() {
    std::cout << "Test: Predict step... ";

    const size_t n_lat = 3;
    const size_t n_lon = 3;
    const size_t n_alt = 3;

    SquareRootUKF filter(n_lat, n_lon, n_alt);

    // Set physics model
    auto physics_model = std::make_shared<GaussMarkovModel>();
    filter.set_physics_model(physics_model);

    // Initialize
    StateVector initial_state(n_lat, n_lon, n_alt);
    for (size_t i = 0; i < n_lat; ++i) {
        for (size_t j = 0; j < n_lon; ++j) {
            for (size_t k = 0; k < n_alt; ++k) {
                initial_state.set_ne(i, j, k, 1e11);
            }
        }
    }

    const size_t dim = initial_state.dimension();
    Eigen::MatrixXd initial_sqrt_cov = Eigen::MatrixXd::Identity(dim, dim) * 1e4;

    filter.initialize(initial_state, initial_sqrt_cov);

    // Perform predict step
    const double dt = 900.0;  // 15 minutes
    filter.predict(dt);

    // Verify state still in reasonable range
    const StateVector& predicted_state = filter.get_state();
    const double ne_check = predicted_state.get_ne(1, 1, 1);

    assert(ne_check > 1e8);   // Above minimum
    assert(ne_check < 1e13);  // Below maximum

    std::cout << "PASSED (predicted Ne: " << ne_check << " el/m³)\n";
}

void test_update_with_synthetic_obs() {
    std::cout << "Test: Update with synthetic observations... ";

    const size_t n_lat = 5;
    const size_t n_lon = 5;
    const size_t n_alt = 5;

    SquareRootUKF filter(n_lat, n_lon, n_alt);

    // Set physics model
    auto physics_model = std::make_shared<GaussMarkovModel>();
    filter.set_physics_model(physics_model);

    // Initialize with known state
    StateVector true_state(n_lat, n_lon, n_alt);
    for (size_t i = 0; i < n_lat; ++i) {
        for (size_t j = 0; j < n_lon; ++j) {
            for (size_t k = 0; k < n_alt; ++k) {
                // Simple gradient for testing
                true_state.set_ne(i, j, k, 5e11 + k * 1e10);
            }
        }
    }
    true_state.set_reff(100.0);

    const size_t dim = true_state.dimension();
    Eigen::MatrixXd initial_sqrt_cov = Eigen::MatrixXd::Identity(dim, dim) * 1e5;

    filter.initialize(true_state, initial_sqrt_cov);

    // Create synthetic TEC observation
    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (size_t i = 0; i < n_lat; ++i) lat_grid.push_back(-90.0 + i * 36.0);
    for (size_t i = 0; i < n_lon; ++i) lon_grid.push_back(-180.0 + i * 72.0);
    for (size_t i = 0; i < n_alt; ++i) alt_grid.push_back(60.0 + i * 100.0);

    std::vector<TECObservationModel::TECMeasurement> tec_measurements;
    TECObservationModel::TECMeasurement meas;
    meas.latitude = 0.0;
    meas.longitude = 0.0;
    meas.altitude = 0.0;
    meas.elevation = 45.0;
    meas.azimuth = 0.0;
    meas.tec_value = 20.0;  // TECU
    meas.tec_error = 2.0;   // TECU
    tec_measurements.push_back(meas);

    TECObservationModel obs_model(tec_measurements, lat_grid, lon_grid, alt_grid);

    // Compute predicted observation
    Eigen::VectorXd predicted_obs = obs_model.forward(true_state);

    // Create observation covariance
    Eigen::MatrixXd obs_sqrt_cov = Eigen::MatrixXd::Identity(1, 1) * 2.0;  // 2 TECU error

    // Perform update
    filter.update(obs_model, predicted_obs, obs_sqrt_cov);

    // Verify state is reasonable after update
    const StateVector& updated_state = filter.get_state();
    const double ne_after = updated_state.get_ne(2, 2, 2);

    assert(ne_after > 1e8);
    assert(ne_after < 1e13);

    std::cout << "PASSED (updated Ne: " << ne_after << " el/m³)\n";
}

int main() {
    std::cout << "=== SR-UKF Integration Tests ===\n";

    try {
        test_filter_initialization();
        test_predict_step();
        test_update_with_synthetic_obs();

        std::cout << "\nAll tests PASSED\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
