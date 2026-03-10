/**
 * @file observation_model.cpp
 * @brief Implementation of observation models for SR-UKF
 */

#include "observation_model.hpp"
#include "../../propagation/include/ray_tracer_3d.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace autonvis {

// Earth parameters for IPP calculation
static constexpr double EARTH_RADIUS_KM = 6371.0;

// =======================
// TECObservationModel
// =======================

TECObservationModel::TECObservationModel(
    const std::vector<TECMeasurement>& measurements,
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid,
    TECIntegrationMethod method
)
    : measurements_(measurements)
    , lat_grid_(lat_grid)
    , lon_grid_(lon_grid)
    , alt_grid_(alt_grid)
    , method_(method)
    , ray_tracer_(nullptr)
{
}

void TECObservationModel::set_ray_tracer(
    std::shared_ptr<propagation::RayTracer3D> ray_tracer
) {
    ray_tracer_ = ray_tracer;
}

void TECObservationModel::set_integration_method(TECIntegrationMethod method) {
    method_ = method;
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
    // Dispatch to appropriate integration method
    switch (method_) {
        case TECIntegrationMethod::RAY_TRACED:
            return compute_slant_tec_raytraced(meas, state);
        case TECIntegrationMethod::LINEAR:
        default:
            return compute_slant_tec_linear(meas, state);
    }
}

double TECObservationModel::compute_slant_tec_linear(
    const TECMeasurement& meas,
    const StateVector& state
) const {
    // Linear (straight-line) TEC integration
    // Uses ionospheric pierce point (IPP) model with thin shell approximation

    const size_t n_alt = alt_grid_.size();

    // Calculate IPP at typical F-region height (350 km)
    const double shell_alt = 350.0;
    double ipp_lat, ipp_lon;
    calculate_pierce_point(meas, shell_alt, ipp_lat, ipp_lon);

    // Find nearest grid indices at IPP location
    size_t i_lat = find_nearest_index(lat_grid_, ipp_lat);
    size_t i_lon = find_nearest_index(lon_grid_, ipp_lon);

    // Integrate vertically through altitude layers
    double vtec = 0.0;  // Vertical TEC (electrons/m²)

    for (size_t i_alt = 0; i_alt < n_alt - 1; ++i_alt) {
        const double ne = state.get_ne(i_lat, i_lon, i_alt);  // electrons/m³
        const double dh = (alt_grid_[i_alt + 1] - alt_grid_[i_alt]) * 1000.0;  // km to m
        vtec += ne * dh;  // electrons/m²
    }

    // Convert vertical TEC to slant TEC using obliquity factor
    double slant_factor = calculate_slant_factor(meas.elevation, shell_alt);
    double stec = vtec * slant_factor;

    // Convert to TECU (1 TECU = 10^16 electrons/m²)
    return stec / TECU_TO_ELECTRONS_M2;
}

double TECObservationModel::compute_slant_tec_raytraced(
    const TECMeasurement& meas,
    const StateVector& state
) const {
    // Ray-traced TEC integration
    // Traces actual ray path through ionosphere and integrates Ne along path

    if (!ray_tracer_) {
        // Fall back to linear integration if no ray tracer available
        return compute_slant_tec_linear(meas, state);
    }

    // Trace ray from receiver toward satellite
    // Using GPS L1 frequency for TEC ray tracing
    propagation::RayPath ray_path = ray_tracer_->trace_ray(
        meas.latitude,
        meas.longitude,
        meas.altitude,
        meas.elevation,
        meas.azimuth,
        GPS_L1_FREQ_MHZ
    );

    // Integrate electron density along the ray path
    double tec = 0.0;  // electrons/m²

    const size_t n_points = ray_path.positions.size();
    if (n_points < 2) {
        // Ray tracing failed, fall back to linear
        return compute_slant_tec_linear(meas, state);
    }

    for (size_t i = 0; i < n_points - 1; ++i) {
        // Get position along ray (lat, lon, alt in degrees/km)
        const auto& pos = ray_path.positions[i];
        const auto& pos_next = ray_path.positions[i + 1];

        double lat = pos(0);
        double lon = pos(1);
        double alt = pos(2);

        // Check if within ionospheric grid bounds
        if (alt < alt_grid_.front() || alt > alt_grid_.back()) {
            continue;  // Skip points outside ionospheric region
        }

        // Interpolate electron density at this position
        double ne = interpolate_ne(state, lat, lon, alt);

        // Calculate path segment length (km to m)
        double ds = (ray_path.path_lengths[i + 1] - ray_path.path_lengths[i]) * 1000.0;

        // Accumulate TEC
        tec += ne * ds;
    }

    // If ray-traced TEC is unreasonably small, use linear as fallback
    // This can happen if ray bends away from satellite direction
    if (tec < 1e10) {  // Less than 0.001 TECU
        return compute_slant_tec_linear(meas, state);
    }

    // Convert to TECU
    return tec / TECU_TO_ELECTRONS_M2;
}

size_t TECObservationModel::find_nearest_index(
    const std::vector<double>& grid,
    double value
) const {
    size_t idx = 0;
    double min_dist = std::abs(grid[0] - value);

    for (size_t i = 1; i < grid.size(); ++i) {
        double dist = std::abs(grid[i] - value);
        if (dist < min_dist) {
            min_dist = dist;
            idx = i;
        }
    }

    return idx;
}

double TECObservationModel::interpolate_ne(
    const StateVector& state,
    double lat, double lon, double alt
) const {
    // Trilinear interpolation of electron density

    const size_t n_lat = lat_grid_.size();
    const size_t n_lon = lon_grid_.size();
    const size_t n_alt = alt_grid_.size();

    // Find bounding indices for each dimension
    size_t i0 = 0, i1 = 0;
    for (size_t i = 0; i < n_lat - 1; ++i) {
        if (lat_grid_[i] <= lat && lat <= lat_grid_[i + 1]) {
            i0 = i;
            i1 = i + 1;
            break;
        }
    }
    // Handle edge cases
    if (lat <= lat_grid_[0]) { i0 = 0; i1 = 0; }
    if (lat >= lat_grid_[n_lat - 1]) { i0 = n_lat - 1; i1 = n_lat - 1; }

    size_t j0 = 0, j1 = 0;
    for (size_t j = 0; j < n_lon - 1; ++j) {
        if (lon_grid_[j] <= lon && lon <= lon_grid_[j + 1]) {
            j0 = j;
            j1 = j + 1;
            break;
        }
    }
    if (lon <= lon_grid_[0]) { j0 = 0; j1 = 0; }
    if (lon >= lon_grid_[n_lon - 1]) { j0 = n_lon - 1; j1 = n_lon - 1; }

    size_t k0 = 0, k1 = 0;
    for (size_t k = 0; k < n_alt - 1; ++k) {
        if (alt_grid_[k] <= alt && alt <= alt_grid_[k + 1]) {
            k0 = k;
            k1 = k + 1;
            break;
        }
    }
    if (alt <= alt_grid_[0]) { k0 = 0; k1 = 0; }
    if (alt >= alt_grid_[n_alt - 1]) { k0 = n_alt - 1; k1 = n_alt - 1; }

    // Calculate interpolation weights
    double t_lat = (i0 == i1) ? 0.0 :
        (lat - lat_grid_[i0]) / (lat_grid_[i1] - lat_grid_[i0]);
    double t_lon = (j0 == j1) ? 0.0 :
        (lon - lon_grid_[j0]) / (lon_grid_[j1] - lon_grid_[j0]);
    double t_alt = (k0 == k1) ? 0.0 :
        (alt - alt_grid_[k0]) / (alt_grid_[k1] - alt_grid_[k0]);

    // Get 8 corner values
    double c000 = state.get_ne(i0, j0, k0);
    double c001 = state.get_ne(i0, j0, k1);
    double c010 = state.get_ne(i0, j1, k0);
    double c011 = state.get_ne(i0, j1, k1);
    double c100 = state.get_ne(i1, j0, k0);
    double c101 = state.get_ne(i1, j0, k1);
    double c110 = state.get_ne(i1, j1, k0);
    double c111 = state.get_ne(i1, j1, k1);

    // Trilinear interpolation
    double c00 = c000 * (1 - t_alt) + c001 * t_alt;
    double c01 = c010 * (1 - t_alt) + c011 * t_alt;
    double c10 = c100 * (1 - t_alt) + c101 * t_alt;
    double c11 = c110 * (1 - t_alt) + c111 * t_alt;

    double c0 = c00 * (1 - t_lon) + c01 * t_lon;
    double c1 = c10 * (1 - t_lon) + c11 * t_lon;

    return c0 * (1 - t_lat) + c1 * t_lat;
}

void TECObservationModel::calculate_pierce_point(
    const TECMeasurement& meas,
    double shell_alt,
    double& ipp_lat,
    double& ipp_lon
) const {
    // Calculate ionospheric pierce point (IPP)
    // Using spherical Earth approximation

    const double deg2rad = M_PI / 180.0;
    const double rad2deg = 180.0 / M_PI;

    // Receiver coordinates in radians
    double lat_r = meas.latitude * deg2rad;
    double lon_r = meas.longitude * deg2rad;
    double el = meas.elevation * deg2rad;
    double az = meas.azimuth * deg2rad;

    // Earth-centered angle from receiver to IPP
    // Using the formula: psi = (pi/2) - el - arcsin((Re/(Re+h)) * cos(el))
    double Re = EARTH_RADIUS_KM;
    double h = shell_alt;
    double cos_el = std::cos(el);
    double sin_arg = (Re / (Re + h)) * cos_el;

    // Clamp to valid range for asin
    sin_arg = std::max(-1.0, std::min(1.0, sin_arg));

    double psi = (M_PI / 2.0) - el - std::asin(sin_arg);

    // IPP latitude
    double sin_lat_ipp = std::sin(lat_r) * std::cos(psi) +
                         std::cos(lat_r) * std::sin(psi) * std::cos(az);
    sin_lat_ipp = std::max(-1.0, std::min(1.0, sin_lat_ipp));
    double lat_ipp = std::asin(sin_lat_ipp);

    // IPP longitude
    double sin_lon_diff = std::sin(psi) * std::sin(az) / std::cos(lat_ipp);
    sin_lon_diff = std::max(-1.0, std::min(1.0, sin_lon_diff));
    double lon_ipp = lon_r + std::asin(sin_lon_diff);

    // Convert back to degrees
    ipp_lat = lat_ipp * rad2deg;
    ipp_lon = lon_ipp * rad2deg;

    // Normalize longitude to [-180, 180]
    while (ipp_lon > 180.0) ipp_lon -= 360.0;
    while (ipp_lon < -180.0) ipp_lon += 360.0;
}

double TECObservationModel::calculate_slant_factor(
    double elevation,
    double shell_alt
) const {
    // Slant factor (mapping function) for single-layer model
    // M(el) = 1 / sqrt(1 - (Re*cos(el)/(Re+h))^2)

    const double deg2rad = M_PI / 180.0;
    double el = elevation * deg2rad;

    double Re = EARTH_RADIUS_KM;
    double h = shell_alt;

    double cos_el = std::cos(el);
    double ratio = (Re * cos_el) / (Re + h);

    // Prevent division by zero for very low elevations
    double sin_zenith_prime_sq = ratio * ratio;
    if (sin_zenith_prime_sq >= 1.0) {
        // Very low elevation, use large but finite value
        return 10.0;
    }

    return 1.0 / std::sqrt(1.0 - sin_zenith_prime_sq);
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
