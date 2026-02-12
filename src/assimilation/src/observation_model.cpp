/**
 * @file observation_model.cpp
 * @brief Implementation of observation models for SR-UKF
 */

#include "observation_model.hpp"
#include <cmath>
#include <algorithm>

namespace autonvis {

// =======================
// TECObservationModel
// =======================

TECObservationModel::TECObservationModel(
    const std::vector<TECMeasurement>& measurements,
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid
)
    : measurements_(measurements)
    , lat_grid_(lat_grid)
    , lon_grid_(lon_grid)
    , alt_grid_(alt_grid)
{
}

Eigen::VectorXd TECObservationModel::forward(const StateVector& state) const {
    // For each TEC measurement, integrate electron density along slant path
    const size_t n_meas = measurements_.size();
    Eigen::VectorXd predicted_tec(n_meas);

    for (size_t i = 0; i < n_meas; ++i) {
        predicted_tec(i) = compute_slant_tec(measurements_[i], state);
    }

    return predicted_tec;
}

double TECObservationModel::compute_slant_tec(
    const TECMeasurement& meas,
    const StateVector& state
) const {
    // Simplified vertical TEC computation
    // TODO: Implement proper slant path integration with ray tracing
    //
    // For now, use vertical integration at receiver location
    // Proper implementation would:
    // 1. Compute ionospheric pierce point using elevation/azimuth
    // 2. Integrate along slant path from receiver to satellite
    // 3. Account for Earth's curvature and geomagnetic field

    // Find nearest grid indices using simple nearest neighbor
    const size_t n_lat = lat_grid_.size();
    const size_t n_lon = lon_grid_.size();
    const size_t n_alt = alt_grid_.size();

    size_t i_lat = 0;
    double min_lat_dist = std::abs(lat_grid_[0] - meas.latitude);
    for (size_t i = 1; i < n_lat; ++i) {
        double dist = std::abs(lat_grid_[i] - meas.latitude);
        if (dist < min_lat_dist) {
            min_lat_dist = dist;
            i_lat = i;
        }
    }

    size_t i_lon = 0;
    double min_lon_dist = std::abs(lon_grid_[0] - meas.longitude);
    for (size_t i = 1; i < n_lon; ++i) {
        double dist = std::abs(lon_grid_[i] - meas.longitude);
        if (dist < min_lon_dist) {
            min_lon_dist = dist;
            i_lon = i;
        }
    }

    // Integrate vertically through altitude layers
    double tec = 0.0;  // electrons/m²

    for (size_t i_alt = 0; i_alt < n_alt - 1; ++i_alt) {
        const double ne = state.get_ne(i_lat, i_lon, i_alt);  // electrons/m³
        const double dh = (alt_grid_[i_alt + 1] - alt_grid_[i_alt]) * 1000.0;  // km to m
        tec += ne * dh;  // electrons/m²
    }

    // Convert to TECU (1 TECU = 10^16 electrons/m²)
    return tec / TECU_TO_ELECTRONS_M2;
}

// =======================
// IonosondeObservationModel
// =======================

IonosondeObservationModel::IonosondeObservationModel(
    const std::vector<IonosondeMeasurement>& measurements,
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid
)
    : measurements_(measurements)
    , lat_grid_(lat_grid)
    , lon_grid_(lon_grid)
    , alt_grid_(alt_grid)
{
}

Eigen::VectorXd IonosondeObservationModel::forward(const StateVector& state) const {
    // Each ionosonde measurement provides foF2 and hmF2
    // Return vector: [foF2_1, hmF2_1, foF2_2, hmF2_2, ...]
    const size_t n_meas = measurements_.size();
    Eigen::VectorXd predicted(n_meas * 2);  // 2 parameters per measurement

    for (size_t i = 0; i < n_meas; ++i) {
        const auto [fof2, hmf2] = extract_f2_parameters(measurements_[i], state);
        predicted(2 * i) = fof2;
        predicted(2 * i + 1) = hmf2;
    }

    return predicted;
}

std::pair<double, double> IonosondeObservationModel::extract_f2_parameters(
    const IonosondeMeasurement& meas,
    const StateVector& state
) const {
    // Find nearest grid indices
    const size_t n_lat = lat_grid_.size();
    const size_t n_lon = lon_grid_.size();
    const size_t n_alt = alt_grid_.size();

    size_t i_lat = 0;
    double min_lat_dist = std::abs(lat_grid_[0] - meas.latitude);
    for (size_t i = 1; i < n_lat; ++i) {
        double dist = std::abs(lat_grid_[i] - meas.latitude);
        if (dist < min_lat_dist) {
            min_lat_dist = dist;
            i_lat = i;
        }
    }

    size_t i_lon = 0;
    double min_lon_dist = std::abs(lon_grid_[0] - meas.longitude);
    for (size_t i = 1; i < n_lon; ++i) {
        double dist = std::abs(lon_grid_[i] - meas.longitude);
        if (dist < min_lon_dist) {
            min_lon_dist = dist;
            i_lon = i;
        }
    }

    // Find F2 layer peak: maximum electron density in vertical profile
    double ne_max = 0.0;
    size_t i_alt_peak = 0;

    for (size_t i_alt = 0; i_alt < n_alt; ++i_alt) {
        const double ne = state.get_ne(i_lat, i_lon, i_alt);
        if (ne > ne_max) {
            ne_max = ne;
            i_alt_peak = i_alt;
        }
    }

    // Compute foF2 from peak electron density
    // Plasma frequency: f_p = sqrt(Ne * e^2 / (m_e * epsilon_0)) / (2 * pi)
    // Simplified formula: foF2 (MHz) ≈ 9 * sqrt(NmF2 / 10^12) where NmF2 in electrons/m³
    const double fof2 = 9.0 * std::sqrt(ne_max / 1e12);  // MHz

    // Compute hmF2 from altitude grid
    const double hmf2 = alt_grid_[i_alt_peak];  // km

    return {fof2, hmf2};
}

} // namespace autonvis
