/**
 * @file test_nvis_model.cpp
 * @brief Unit tests for NVIS observation model
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "nvis_observation_model.hpp"

using namespace autonvis;

const double EPSILON = 1e-6;

/**
 * @brief Create simple test state with Chapman profile
 */
StateVector create_test_state(size_t n_lat, size_t n_lon, size_t n_alt) {
    StateVector state(n_lat, n_lon, n_alt);

    // Create Chapman-like F2 layer profile
    const double peak_ne = 1e12;  // el/m³
    const double peak_height = 300.0;  // km
    const double scale_height = 50.0;  // km

    std::vector<double> alt_grid;
    for (size_t k = 0; k < n_alt; ++k) {
        alt_grid.push_back(60.0 + k * 10.0);  // 60-600 km
    }

    for (size_t i = 0; i < n_lat; ++i) {
        for (size_t j = 0; j < n_lon; ++j) {
            for (size_t k = 0; k < n_alt; ++k) {
                const double alt = alt_grid[k];

                // Chapman profile
                const double z = (alt - peak_height) / scale_height;
                const double ne = peak_ne * std::exp(1.0 - z - std::exp(-z));

                state.set_ne(i, j, k, std::max(0.0, ne));
            }
        }
    }

    state.set_reff(100.0);  // Moderate activity

    return state;
}

/**
 * @brief Test 1: Model construction
 */
void test_construction() {
    std::cout << "Test 1: Model construction... ";

    std::vector<NVISSounderObservationModel::NVISMeasurement> measurements;
    NVISSounderObservationModel::NVISMeasurement meas;

    meas.tx_latitude = 40.0;
    meas.tx_longitude = -105.0;
    meas.tx_altitude = 1500.0;
    meas.rx_latitude = 40.5;
    meas.rx_longitude = -104.5;
    meas.rx_altitude = 1600.0;
    meas.frequency = 7.5;
    meas.elevation_angle = 85.0;
    meas.azimuth = 45.0;
    meas.hop_distance = 75.0;
    meas.signal_strength = -80.0;
    meas.group_delay = 2.5;
    meas.snr = 20.0;
    meas.signal_strength_error = 2.0;
    meas.group_delay_error = 0.1;
    meas.is_o_mode = true;
    meas.tx_power = 100.0;
    meas.tx_antenna_gain = 0.0;
    meas.rx_antenna_gain = 0.0;

    measurements.push_back(meas);

    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (int i = -90; i <= 90; i += 30) lat_grid.push_back(i);
    for (int i = -180; i <= 180; i += 60) lon_grid.push_back(i);
    for (int i = 60; i <= 600; i += 10) alt_grid.push_back(i);

    NVISSounderObservationModel model(measurements, lat_grid, lon_grid, alt_grid);

    assert(model.obs_dimension() == 2);  // signal + delay

    std::cout << "PASSED\n";
}

/**
 * @brief Test 2: Forward model execution
 */
void test_forward_model() {
    std::cout << "Test 2: Forward model execution... ";

    const size_t n_lat = 7;
    const size_t n_lon = 7;
    const size_t n_alt = 55;

    StateVector state = create_test_state(n_lat, n_lon, n_alt);

    std::vector<NVISSounderObservationModel::NVISMeasurement> measurements;
    NVISSounderObservationModel::NVISMeasurement meas;

    meas.tx_latitude = 40.0;
    meas.tx_longitude = -105.0;
    meas.tx_altitude = 1500.0;
    meas.rx_latitude = 40.5;
    meas.rx_longitude = -104.5;
    meas.rx_altitude = 1600.0;
    meas.frequency = 7.5;
    meas.elevation_angle = 85.0;
    meas.azimuth = 45.0;
    meas.hop_distance = 75.0;
    meas.signal_strength = -80.0;
    meas.group_delay = 2.5;
    meas.snr = 20.0;
    meas.signal_strength_error = 2.0;
    meas.group_delay_error = 0.1;
    meas.is_o_mode = true;
    meas.tx_power = 100.0;
    meas.tx_antenna_gain = 0.0;
    meas.rx_antenna_gain = 0.0;

    measurements.push_back(meas);

    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (int i = -90; i <= 90; i += 30) lat_grid.push_back(i);
    for (int i = -180; i <= 180; i += 60) lon_grid.push_back(i);
    for (int i = 60; i <= 600; i += 10) alt_grid.push_back(i);

    NVISSounderObservationModel model(measurements, lat_grid, lon_grid, alt_grid);

    Eigen::VectorXd obs = model.forward(state);

    assert(obs.size() == 2);
    assert(obs(0) >= -140.0 && obs(0) <= 0.0);  // Signal strength in reasonable range
    assert(obs(1) >= 0.0 && obs(1) <= 10.0);    // Group delay < 10 ms

    std::cout << "PASSED (signal=" << obs(0) << " dBm, delay=" << obs(1) << " ms)\n";
}

/**
 * @brief Test 3: Signal strength prediction
 */
void test_signal_strength() {
    std::cout << "Test 3: Signal strength prediction... ";

    const size_t n_lat = 7;
    const size_t n_lon = 7;
    const size_t n_alt = 55;

    StateVector state = create_test_state(n_lat, n_lon, n_alt);

    NVISSounderObservationModel::NVISMeasurement meas;
    meas.tx_latitude = 40.0;
    meas.tx_longitude = -105.0;
    meas.tx_altitude = 1500.0;
    meas.rx_latitude = 40.5;
    meas.rx_longitude = -104.5;
    meas.rx_altitude = 1600.0;
    meas.frequency = 7.5;
    meas.elevation_angle = 85.0;
    meas.azimuth = 45.0;
    meas.hop_distance = 75.0;
    meas.signal_strength = -80.0;
    meas.group_delay = 2.5;
    meas.snr = 20.0;
    meas.signal_strength_error = 2.0;
    meas.group_delay_error = 0.1;
    meas.is_o_mode = true;
    meas.tx_power = 100.0;
    meas.tx_antenna_gain = 0.0;
    meas.rx_antenna_gain = 0.0;

    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (int i = -90; i <= 90; i += 30) lat_grid.push_back(i);
    for (int i = -180; i <= 180; i += 60) lon_grid.push_back(i);
    for (int i = 60; i <= 600; i += 10) alt_grid.push_back(i);

    std::vector<NVISSounderObservationModel::NVISMeasurement> measurements = {meas};
    NVISSounderObservationModel model(measurements, lat_grid, lon_grid, alt_grid);

    double signal = model.predict_signal_strength_simplified(meas, state);

    // Signal should be in reasonable range
    assert(signal >= -140.0 && signal <= 0.0);

    // Higher frequency should give higher path loss (weaker signal)
    meas.frequency = 15.0;
    double signal_high_freq = model.predict_signal_strength_simplified(meas, state);
    assert(signal_high_freq < signal);

    std::cout << "PASSED (7.5 MHz: " << signal << " dBm, 15 MHz: " << signal_high_freq << " dBm)\n";
}

/**
 * @brief Test 4: Group delay prediction
 */
void test_group_delay() {
    std::cout << "Test 4: Group delay prediction... ";

    const size_t n_lat = 7;
    const size_t n_lon = 7;
    const size_t n_alt = 55;

    StateVector state = create_test_state(n_lat, n_lon, n_alt);

    NVISSounderObservationModel::NVISMeasurement meas;
    meas.tx_latitude = 40.0;
    meas.tx_longitude = -105.0;
    meas.tx_altitude = 1500.0;
    meas.rx_latitude = 40.5;
    meas.rx_longitude = -104.5;
    meas.rx_altitude = 1600.0;
    meas.frequency = 7.5;
    meas.elevation_angle = 85.0;
    meas.azimuth = 45.0;
    meas.hop_distance = 75.0;
    meas.signal_strength = -80.0;
    meas.group_delay = 2.5;
    meas.snr = 20.0;
    meas.signal_strength_error = 2.0;
    meas.group_delay_error = 0.1;
    meas.is_o_mode = true;
    meas.tx_power = 100.0;
    meas.tx_antenna_gain = 0.0;
    meas.rx_antenna_gain = 0.0;

    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (int i = -90; i <= 90; i += 30) lat_grid.push_back(i);
    for (int i = -180; i <= 180; i += 60) lon_grid.push_back(i);
    for (int i = 60; i <= 600; i += 10) alt_grid.push_back(i);

    std::vector<NVISSounderObservationModel::NVISMeasurement> measurements = {meas};
    NVISSounderObservationModel model(measurements, lat_grid, lon_grid, alt_grid);

    double delay = model.predict_group_delay_simplified(meas, state);

    // Delay should be positive and < 10 ms for NVIS
    assert(delay >= 0.0 && delay <= 10.0);

    // Lower elevation should give longer delay (obliquity)
    meas.elevation_angle = 75.0;
    double delay_low_elev = model.predict_group_delay_simplified(meas, state);
    assert(delay_low_elev > delay);

    std::cout << "PASSED (85°: " << delay << " ms, 75°: " << delay_low_elev << " ms)\n";
}

/**
 * @brief Test 5: Reflection height sensitivity
 */
void test_reflection_height() {
    std::cout << "Test 5: Reflection height sensitivity... ";

    const size_t n_lat = 7;
    const size_t n_lon = 7;
    const size_t n_alt = 55;

    StateVector state = create_test_state(n_lat, n_lon, n_alt);

    NVISSounderObservationModel::NVISMeasurement meas;
    meas.tx_latitude = 40.0;
    meas.tx_longitude = -105.0;
    meas.tx_altitude = 1500.0;
    meas.rx_latitude = 40.5;
    meas.rx_longitude = -104.5;
    meas.rx_altitude = 1600.0;
    meas.frequency = 5.0;  // Low frequency
    meas.elevation_angle = 85.0;
    meas.azimuth = 45.0;
    meas.hop_distance = 75.0;
    meas.signal_strength = -80.0;
    meas.group_delay = 2.5;
    meas.snr = 20.0;
    meas.signal_strength_error = 2.0;
    meas.group_delay_error = 0.1;
    meas.is_o_mode = true;
    meas.tx_power = 100.0;
    meas.tx_antenna_gain = 0.0;
    meas.rx_antenna_gain = 0.0;

    std::vector<double> lat_grid, lon_grid, alt_grid;
    for (int i = -90; i <= 90; i += 30) lat_grid.push_back(i);
    for (int i = -180; i <= 180; i += 60) lon_grid.push_back(i);
    for (int i = 60; i <= 600; i += 10) alt_grid.push_back(i);

    std::vector<NVISSounderObservationModel::NVISMeasurement> measurements = {meas};
    NVISSounderObservationModel model(measurements, lat_grid, lon_grid, alt_grid);

    double delay_low = model.predict_group_delay_simplified(meas, state);

    // Higher frequency should reflect at higher altitude → longer delay
    meas.frequency = 10.0;
    double delay_high = model.predict_group_delay_simplified(meas, state);

    assert(delay_high > delay_low);

    std::cout << "PASSED (5 MHz: " << delay_low << " ms, 10 MHz: " << delay_high << " ms)\n";
}

int main() {
    std::cout << "=== NVIS Observation Model Tests ===\n\n";

    try {
        test_construction();
        test_forward_model();
        test_signal_strength();
        test_group_delay();
        test_reflection_height();

        std::cout << "\n=== All tests PASSED ===\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n=== TEST FAILED ===\n";
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
