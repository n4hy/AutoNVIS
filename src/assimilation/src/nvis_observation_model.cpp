/**
 * @file nvis_observation_model.cpp
 * @brief Implementation of NVIS sounder observation model
 */

#include "nvis_observation_model.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace autonvis {

NVISSounderObservationModel::NVISSounderObservationModel(
    const std::vector<NVISMeasurement>& measurements,
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid
) : measurements_(measurements),
    lat_grid_(lat_grid),
    lon_grid_(lon_grid),
    alt_grid_(alt_grid)
{
    if (measurements_.empty()) {
        throw std::invalid_argument("No measurements provided");
    }
    if (lat_grid_.empty() || lon_grid_.empty() || alt_grid_.empty()) {
        throw std::invalid_argument("Empty grid provided");
    }
}

Eigen::VectorXd NVISSounderObservationModel::forward(const StateVector& state) const {
    const size_t n_meas = measurements_.size();
    Eigen::VectorXd obs(2 * n_meas);

    // First N elements: signal strengths
    for (size_t i = 0; i < n_meas; ++i) {
        obs(i) = predict_signal_strength_simplified(measurements_[i], state);
    }

    // Second N elements: group delays
    for (size_t i = 0; i < n_meas; ++i) {
        obs(n_meas + i) = predict_group_delay_simplified(measurements_[i], state);
    }

    return obs;
}

double NVISSounderObservationModel::predict_signal_strength_simplified(
    const NVISMeasurement& meas,
    const StateVector& state
) const {
    // Compute midpoint location
    const double mid_lat = (meas.tx_latitude + meas.rx_latitude) / 2.0;
    const double mid_lon = (meas.tx_longitude + meas.rx_longitude) / 2.0;

    // Extract vertical Ne profile at midpoint
    const Eigen::VectorXd ne_profile = get_vertical_profile(mid_lat, mid_lon, state);

    // Find reflection height
    const double reflect_height = find_reflection_height(ne_profile, meas.frequency);

    if (reflect_height < 0) {
        // No reflection (frequency too high or Ne too low)
        return -140.0;  // Very weak signal
    }

    // Compute path length (2 × slant range to reflection height)
    const double slant_factor = 1.0 / std::sin(meas.elevation_angle * PI / 180.0);
    const double slant_distance = reflect_height * slant_factor;
    const double total_path = 2.0 * slant_distance;  // Up and down

    // Free space path loss
    const double free_space_loss = compute_free_space_loss(meas.frequency, total_path);

    // D-region absorption
    const double d_region_path = 2.0 * D_REGION_THICKNESS / std::sin(meas.elevation_angle * PI / 180.0);
    const double absorption_loss = compute_d_region_absorption(
        meas.frequency,
        d_region_path,
        meas.elevation_angle
    );

    // Compute received power
    // Start with transmitted power (default 100W = 50 dBm if not specified)
    double tx_power_dbm = 50.0;
    if (meas.tx_power > 0) {
        tx_power_dbm = 10.0 * std::log10(meas.tx_power * 1000.0);  // W → dBm
    }

    // Add antenna gains (default 0 dBi)
    const double tx_gain = (meas.tx_antenna_gain > -100.0) ? meas.tx_antenna_gain : 0.0;
    const double rx_gain = (meas.rx_antenna_gain > -100.0) ? meas.rx_antenna_gain : 0.0;

    // Predicted signal strength
    double predicted_signal = tx_power_dbm + tx_gain + rx_gain - free_space_loss - absorption_loss;

    // Clamp to reasonable range
    predicted_signal = std::max(-140.0, std::min(0.0, predicted_signal));

    return predicted_signal;
}

double NVISSounderObservationModel::predict_group_delay_simplified(
    const NVISMeasurement& meas,
    const StateVector& state
) const {
    // Compute midpoint location
    const double mid_lat = (meas.tx_latitude + meas.rx_latitude) / 2.0;
    const double mid_lon = (meas.tx_longitude + meas.rx_longitude) / 2.0;

    // Extract vertical Ne profile at midpoint
    const Eigen::VectorXd ne_profile = get_vertical_profile(mid_lat, mid_lon, state);

    // Find reflection height
    const double reflect_height = find_reflection_height(ne_profile, meas.frequency);

    if (reflect_height < 0) {
        // No reflection
        return 0.0;
    }

    // Compute obliquity factor
    const double obliquity = compute_obliquity_factor(meas.elevation_angle);

    // Group delay: 2 × (height / c) × obliquity
    // Convert height from km to m
    const double height_m = reflect_height * 1000.0;
    const double delay_sec = 2.0 * height_m / SPEED_OF_LIGHT * obliquity;
    const double delay_ms = delay_sec * 1000.0;

    return delay_ms;
}

Eigen::VectorXd NVISSounderObservationModel::get_vertical_profile(
    double lat,
    double lon,
    const StateVector& state
) const {
    const size_t n_alt = alt_grid_.size();
    Eigen::VectorXd profile(n_alt);

    for (size_t i = 0; i < n_alt; ++i) {
        profile(i) = interpolate_ne(lat, lon, alt_grid_[i], state);
    }

    return profile;
}

double NVISSounderObservationModel::find_reflection_height(
    const Eigen::VectorXd& ne_profile,
    double frequency
) const {
    const double f_hz = frequency * 1e6;  // MHz → Hz
    const size_t n_alt = alt_grid_.size();

    // Find first altitude where plasma frequency ≥ wave frequency
    for (size_t i = 0; i < n_alt; ++i) {
        const double f_plasma = ne_to_plasma_freq(ne_profile(i));
        if (f_plasma >= f_hz) {
            // Linear interpolation for more accurate height
            if (i > 0) {
                const double f_plasma_prev = ne_to_plasma_freq(ne_profile(i - 1));
                const double alpha = (f_hz - f_plasma_prev) / (f_plasma - f_plasma_prev);
                return alt_grid_[i - 1] + alpha * (alt_grid_[i] - alt_grid_[i - 1]);
            }
            return alt_grid_[i];
        }
    }

    // No reflection found (frequency too high or Ne too low)
    return -1.0;
}

double NVISSounderObservationModel::compute_free_space_loss(
    double frequency,
    double distance
) const {
    // Free space path loss (dB) = 20 log10(d) + 20 log10(f) + 32.45
    // where d is in km, f is in MHz
    const double loss_db = 20.0 * std::log10(distance) +
                          20.0 * std::log10(frequency) +
                          32.45;
    return loss_db;
}

double NVISSounderObservationModel::compute_d_region_absorption(
    double frequency,
    double distance,
    double elevation_angle
) const {
    // Simplified D-region absorption model
    // Absorption ∝ 1/f² × path_length
    // More absorption at lower frequencies and lower elevation angles

    const double freq_factor = 1.0 / (frequency * frequency);
    const double absorption_db = D_REGION_ABSORPTION_COEFF * distance * freq_factor;

    // Additional factor for low elevation (more absorption through D-region)
    const double elev_factor = std::max(0.5, std::sin(elevation_angle * PI / 180.0));
    const double total_absorption = absorption_db / elev_factor;

    return total_absorption;
}

double NVISSounderObservationModel::compute_obliquity_factor(
    double elevation_angle
) const {
    // Obliquity factor accounts for path lengthening
    // For near-vertical: obliquity ≈ 1 / sin(elevation)
    const double sin_elev = std::sin(elevation_angle * PI / 180.0);
    if (sin_elev < 0.1) {
        return 10.0;  // Cap at steep angles
    }
    return 1.0 / sin_elev;
}

double NVISSounderObservationModel::ne_to_plasma_freq(double ne) const {
    // Plasma frequency: f_p = sqrt(e² N_e / (ε_0 m_e)) / (2π)
    if (ne <= 0) {
        return 0.0;
    }

    const double factor = ELECTRON_CHARGE * ELECTRON_CHARGE / (EPSILON_0 * ELECTRON_MASS);
    const double f_plasma = std::sqrt(factor * ne) / (2.0 * PI);

    return f_plasma;  // Hz
}

double NVISSounderObservationModel::interpolate_ne(
    double lat,
    double lon,
    double alt,
    const StateVector& state
) const {
    // Trilinear interpolation
    const size_t n_lat = lat_grid_.size();
    const size_t n_lon = lon_grid_.size();
    const size_t n_alt = alt_grid_.size();

    // Find grid indices
    auto find_index = [](const std::vector<double>& grid, double value) -> size_t {
        auto it = std::lower_bound(grid.begin(), grid.end(), value);
        if (it == grid.end()) return grid.size() - 2;
        if (it == grid.begin()) return 0;
        return std::distance(grid.begin(), it) - 1;
    };

    const size_t i_lat = find_index(lat_grid_, lat);
    const size_t i_lon = find_index(lon_grid_, lon);
    const size_t i_alt = find_index(alt_grid_, alt);

    // Clamp to valid indices
    const size_t i_lat_clamp = std::min(i_lat, n_lat - 2);
    const size_t i_lon_clamp = std::min(i_lon, n_lon - 2);
    const size_t i_alt_clamp = std::min(i_alt, n_alt - 2);

    // Interpolation weights
    const double lat_frac = (lat - lat_grid_[i_lat_clamp]) /
                           (lat_grid_[i_lat_clamp + 1] - lat_grid_[i_lat_clamp]);
    const double lon_frac = (lon - lon_grid_[i_lon_clamp]) /
                           (lon_grid_[i_lon_clamp + 1] - lon_grid_[i_lon_clamp]);
    const double alt_frac = (alt - alt_grid_[i_alt_clamp]) /
                           (alt_grid_[i_alt_clamp + 1] - alt_grid_[i_alt_clamp]);

    // Clamp fractions to [0, 1]
    const double w_lat = std::max(0.0, std::min(1.0, lat_frac));
    const double w_lon = std::max(0.0, std::min(1.0, lon_frac));
    const double w_alt = std::max(0.0, std::min(1.0, alt_frac));

    // Trilinear interpolation
    double ne_interp = 0.0;
    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                const size_t idx = (i_lat_clamp + di) * n_lon * n_alt +
                                  (i_lon_clamp + dj) * n_alt +
                                  (i_alt_clamp + dk);

                const double weight = ((di == 0) ? (1.0 - w_lat) : w_lat) *
                                     ((dj == 0) ? (1.0 - w_lon) : w_lon) *
                                     ((dk == 0) ? (1.0 - w_alt) : w_alt);

                ne_interp += weight * state.ne_grid[idx];
            }
        }
    }

    return std::max(0.0, ne_interp);  // Ensure non-negative
}

double NVISSounderObservationModel::haversine_distance(
    double lat1,
    double lon1,
    double lat2,
    double lon2
) const {
    const double lat1_rad = lat1 * PI / 180.0;
    const double lat2_rad = lat2 * PI / 180.0;
    const double dlon = (lon2 - lon1) * PI / 180.0;
    const double dlat = (lat2 - lat1) * PI / 180.0;

    const double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
                    std::cos(lat1_rad) * std::cos(lat2_rad) *
                    std::sin(dlon / 2.0) * std::sin(dlon / 2.0);

    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

    return EARTH_RADIUS * c;  // km
}

} // namespace autonvis
