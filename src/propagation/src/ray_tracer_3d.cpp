/**
 * @file ray_tracer_3d.cpp
 * @brief Implementation of 3D ionospheric ray tracing
 */

#include "ray_tracer_3d.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>

namespace autonvis {
namespace propagation {

// Physical constants
constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;
constexpr double EARTH_RADIUS_KM = 6371.0;

// Electron properties
constexpr double ELECTRON_CHARGE = 1.60217663e-19;  // C
constexpr double ELECTRON_MASS = 9.1093837e-31;     // kg
constexpr double EPSILON_0 = 8.8541878e-12;         // F/m
constexpr double C_LIGHT = 2.99792458e8;            // m/s

// Plasma frequency calculation
inline double plasma_frequency_hz(double ne) {
    // f_p = sqrt(ne * e^2 / (m_e * epsilon_0)) / (2*pi)
    return std::sqrt(ne * ELECTRON_CHARGE * ELECTRON_CHARGE /
                    (ELECTRON_MASS * EPSILON_0)) / (2.0 * PI);
}

inline double gyro_frequency_hz(double B_tesla) {
    // f_g = e * B / (2 * pi * m_e)
    return ELECTRON_CHARGE * B_tesla / (2.0 * PI * ELECTRON_MASS);
}

//=============================================================================
// IonoGrid Implementation
//=============================================================================

IonoGrid::IonoGrid(const Eigen::VectorXd& lat,
                   const Eigen::VectorXd& lon,
                   const Eigen::VectorXd& alt,
                   const std::vector<double>& ne_grid)
    : lat_(lat), lon_(lon), alt_(alt), ne_grid_(ne_grid)
{
    // Verify grid size matches
    if (ne_grid.size() != lat.size() * lon.size() * alt.size()) {
        throw std::invalid_argument("ne_grid size must equal lat.size() * lon.size() * alt.size()");
    }

    dlat_ = lat.size() > 1 ? (lat(1) - lat(0)) : 1.0;
    dlon_ = lon.size() > 1 ? (lon(1) - lon(0)) : 1.0;
    dalt_ = alt.size() > 1 ? (alt(1) - alt(0)) : 1.0;
}

double IonoGrid::electron_density(double lat, double lon, double alt) const {
    // Trilinear interpolation

    // Handle longitude wraparound
    while (lon < lon_(0)) lon += 360.0;
    while (lon > lon_(lon_.size()-1)) lon -= 360.0;

    // Clamp to grid boundaries
    lat = std::clamp(lat, lat_(0), lat_(lat_.size()-1));
    alt = std::clamp(alt, alt_(0), alt_(alt_.size()-1));

    // Find grid indices
    int i = std::lower_bound(lat_.data(), lat_.data() + lat_.size(), lat) - lat_.data() - 1;
    int j = std::lower_bound(lon_.data(), lon_.data() + lon_.size(), lon) - lon_.data() - 1;
    int k = std::lower_bound(alt_.data(), alt_.data() + alt_.size(), alt) - alt_.data() - 1;

    i = std::clamp(i, 0, static_cast<int>(lat_.size()) - 2);
    j = std::clamp(j, 0, static_cast<int>(lon_.size()) - 2);
    k = std::clamp(k, 0, static_cast<int>(alt_.size()) - 2);

    // Interpolation weights
    double wlat = (lat - lat_(i)) / (lat_(i+1) - lat_(i));
    double wlon = (lon - lon_(j)) / (lon_(j+1) - lon_(j));
    double walt = (alt - alt_(k)) / (alt_(k+1) - alt_(k));

    // Trilinear interpolation
    double ne = 0.0;
    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                double w = (di == 0 ? 1.0 - wlat : wlat) *
                          (dj == 0 ? 1.0 - wlon : wlon) *
                          (dk == 0 ? 1.0 - walt : walt);
                ne += w * get_ne(i+di, j+dj, k+dk);
            }
        }
    }

    return std::max(0.0, ne);  // Ensure non-negative
}

Eigen::Vector3d IonoGrid::electron_density_gradient(double lat, double lon, double alt) const {
    // Finite difference approximation
    const double h = 1e-3;  // Small step (degrees or km)

    double ne_center = electron_density(lat, lon, alt);

    double dne_dlat = (electron_density(lat + h, lon, alt) - ne_center) / h;
    double dne_dlon = (electron_density(lat, lon + h, alt) - ne_center) / h;
    double dne_dalt = (electron_density(lat, lon, alt + h) - ne_center) / h;

    return Eigen::Vector3d(dne_dlat, dne_dlon, dne_dalt);
}

double IonoGrid::collision_frequency(double alt, double xray_flux) const {
    // D-region collision frequency model
    // Based on exponential atmosphere and X-ray ionization

    // Neutral atmosphere density (exponential model)
    double h0 = 80.0;  // Scale height reference (km)
    double H = 7.0;    // Scale height (km)
    double n_neutral = 1e20 * std::exp(-(alt - h0) / H);  // molecules/m³

    // Base collision frequency
    double nu_base = 1e-10 * n_neutral;  // Hz

    // X-ray enhancement (if in SHOCK mode)
    if (xray_flux > 1e-6 && alt < 100.0) {
        double enhancement = std::sqrt(xray_flux / 1e-6);
        nu_base *= (1.0 + 10.0 * enhancement);
    }

    return nu_base;
}

//=============================================================================
// GeomagneticField Implementation
//=============================================================================

GeomagneticField::GeomagneticField() {
    // Use simple dipole model if no IGRF file
    // IGRF coefficients can be loaded later
}

GeomagneticField::GeomagneticField(const std::string& igrf_coeffs_file) {
    load_coefficients(igrf_coeffs_file);
}

Eigen::Vector3d GeomagneticField::field(double lat, double lon, double alt, int year) const {
    // If IGRF coefficients not loaded, use dipole approximation
    if (g_coeffs_.empty()) {
        return dipole_field(lat, lon, alt);
    }

    // TODO: Full IGRF-13 implementation using spherical harmonics
    // For now, use dipole approximation
    return dipole_field(lat, lon, alt);
}

double GeomagneticField::field_magnitude(double lat, double lon, double alt, int year) const {
    Eigen::Vector3d B = field(lat, lon, alt, year);
    return B.norm();
}

double GeomagneticField::dip_angle(double lat, double lon, double alt, int year) const {
    Eigen::Vector3d B = field(lat, lon, alt, year);
    double B_horizontal = std::sqrt(B(0)*B(0) + B(1)*B(1));
    return std::atan2(B(2), B_horizontal);  // Radians
}

Eigen::Vector3d GeomagneticField::dipole_field(double lat, double lon, double alt) const {
    // Simple dipole approximation
    // B0 = 31,000 nT at equator

    double B0 = 31000e-9;  // Tesla
    double r = EARTH_RADIUS_KM + alt;
    double lat_rad = lat * DEG_TO_RAD;

    // Dipole field components (local North, East, Down)
    double B_north = -2.0 * B0 * std::pow(EARTH_RADIUS_KM / r, 3) * std::sin(lat_rad);
    double B_east = 0.0;
    double B_down = -B0 * std::pow(EARTH_RADIUS_KM / r, 3) * std::cos(lat_rad);

    return Eigen::Vector3d(B_north, B_east, B_down);
}

void GeomagneticField::load_coefficients(const std::string& filename) {
    // TODO: Parse IGRF-13 coefficient file
    // Format: year, degree, order, g_coeff, h_coeff
    // For now, leave empty to use dipole approximation
}

//=============================================================================
// MagnetoionicTheory Implementation
//=============================================================================

std::complex<double> MagnetoionicTheory::refractive_index(
    double ne,
    double freq,
    double B_mag,
    double theta,
    double nu,
    Mode mode)
{
    // Appleton-Hartree equation for refractive index
    // n² = 1 - X / (1 - iZ - Y²sin²θ/2(1-X-iZ) ± Y⁴sin⁴θ/4(1-X-iZ)² + Y²cos²θ)^(1/2)
    //
    // where:
    // X = (f_p / f)²  (plasma frequency ratio)
    // Y = f_g / f     (gyro frequency ratio)
    // Z = ν / f       (collision frequency ratio)

    double f_p = plasma_frequency_hz(ne);
    double f_g = gyro_frequency_hz(B_mag);

    double X = (f_p / freq) * (f_p / freq);
    double Y = f_g / freq;
    double Z = nu / freq;

    std::complex<double> iZ(0.0, Z);

    double sin_theta = std::sin(theta);
    double cos_theta = std::cos(theta);
    double sin2 = sin_theta * sin_theta;
    double cos2 = cos_theta * cos_theta;

    // Discriminant term
    std::complex<double> term1 = 1.0 - X - iZ;
    std::complex<double> Y2sin2 = Y * Y * sin2;
    std::complex<double> Y2cos2 = Y * Y * cos2;

    std::complex<double> sqrt_term = Y2sin2 * Y2sin2 / (4.0 * term1 * term1) + Y2cos2;
    sqrt_term = std::sqrt(sqrt_term);

    // ± for O-mode (minus) and X-mode (plus)
    std::complex<double> denominator;
    if (mode == O_MODE) {
        denominator = 1.0 - iZ - Y2sin2 / (2.0 * term1) - sqrt_term;
    } else {
        denominator = 1.0 - iZ - Y2sin2 / (2.0 * term1) + sqrt_term;
    }

    std::complex<double> n_squared = 1.0 - X / denominator;

    return std::sqrt(n_squared);
}

double MagnetoionicTheory::group_refractive_index(
    double ne,
    double freq,
    double B_mag,
    double theta,
    double nu,
    Mode mode)
{
    // Group refractive index μ = n + f * dn/df
    // Use numerical derivative

    double df = freq * 1e-6;  // Small frequency perturbation

    std::complex<double> n1 = refractive_index(ne, freq - df/2, B_mag, theta, nu, mode);
    std::complex<double> n2 = refractive_index(ne, freq + df/2, B_mag, theta, nu, mode);

    std::complex<double> n = refractive_index(ne, freq, B_mag, theta, nu, mode);
    std::complex<double> dndf = (n2 - n1) / df;

    return std::real(n + freq * dndf);
}

double MagnetoionicTheory::absorption_coefficient(double ne, double freq, double nu) {
    // Absorption in Nepers/meter
    // α = (2π/c) * f * Im(n)

    // Simplified collisional absorption
    double f_p = plasma_frequency_hz(ne);
    double X = (f_p / freq) * (f_p / freq);
    double Z = nu / freq;

    // For weak collisions: Im(n) ≈ X*Z / (2*(1-X)^(3/2))
    if (X < 0.9) {
        double im_n = X * Z / (2.0 * std::pow(1.0 - X, 1.5));
        return (2.0 * PI / C_LIGHT) * freq * im_n;
    }

    // Full calculation for strong collisions
    std::complex<double> n = refractive_index(ne, freq, 0.0, 0.0, nu, O_MODE);
    return (2.0 * PI / C_LIGHT) * freq * std::imag(n);
}

//=============================================================================
// RayTracer3D Implementation
//=============================================================================

RayTracer3D::RayTracer3D(
    std::shared_ptr<IonoGrid> iono_grid,
    std::shared_ptr<GeomagneticField> geomag,
    const RayTracingConfig& config)
    : iono_grid_(iono_grid),
      geomag_(geomag),
      config_(config)
{
}

RayPath RayTracer3D::trace_ray(
    double lat0,
    double lon0,
    double alt0,
    double elevation,
    double azimuth,
    double freq_mhz)
{
    RayPath path;

    // Initialize ray state
    RayState state;
    state.position = Eigen::Vector3d(lat0, lon0, alt0);

    // Initial wave normal direction from elevation and azimuth
    double elev_rad = elevation * DEG_TO_RAD;
    double azim_rad = azimuth * DEG_TO_RAD;

    state.wave_normal = Eigen::Vector3d(
        std::cos(elev_rad) * std::cos(azim_rad),  // North component
        std::cos(elev_rad) * std::sin(azim_rad),  // East component
        std::sin(elev_rad)                         // Up component
    );
    state.wave_normal.normalize();

    double freq_hz = freq_mhz * 1e6;
    double step_size = config_.initial_step_km;

    // Integration loop
    for (int step = 0; step < config_.max_steps; ++step) {
        // Store current state
        path.positions.push_back(state.position);
        path.wave_normals.push_back(state.wave_normal);
        path.path_lengths.push_back(state.path_length);
        path.absorption_db.push_back(state.absorption_db);

        // Get electron density and magnetic field
        double ne = iono_grid_->electron_density(
            state.position(0), state.position(1), state.position(2));
        Eigen::Vector3d B = geomag_->field(
            state.position(0), state.position(1), state.position(2));

        // Calculate refractive index
        double B_mag = B.norm() * 1e-9;  // nT to Tesla
        double theta = std::acos(state.wave_normal.dot(B.normalized()));

        std::complex<double> n = MagnetoionicTheory::refractive_index(
            ne, freq_hz, B_mag, theta, 0.0, config_.mode);

        path.refractive_indices.push_back(std::real(n));

        // Check for reflection (n → 0) or penetration
        if (std::real(n) < 0.1) {
            path.reflected = true;
            break;
        }

        // Check termination conditions
        if (check_termination(state, path)) {
            break;
        }

        // Integrate one step
        state = integrate_step(state, freq_hz, step_size);

        // Calculate absorption if enabled
        if (config_.calculate_absorption && state.position(2) < 150.0) {
            double nu = iono_grid_->collision_frequency(state.position(2));
            double alpha = MagnetoionicTheory::absorption_coefficient(ne, freq_hz, nu);
            state.absorption_db += alpha * step_size * 1000.0 * 8.686;  // Nepers/m to dB/km
        }
    }

    // Calculate ground range
    if (path.positions.size() >= 2) {
        Eigen::Vector3d start = path.positions.front();
        Eigen::Vector3d end = path.positions.back();

        // Great circle distance
        double lat1 = start(0) * DEG_TO_RAD;
        double lon1 = start(1) * DEG_TO_RAD;
        double lat2 = end(0) * DEG_TO_RAD;
        double lon2 = end(1) * DEG_TO_RAD;

        double dlat = lat2 - lat1;
        double dlon = lon2 - lon1;

        double a = std::sin(dlat/2) * std::sin(dlat/2) +
                  std::cos(lat1) * std::cos(lat2) *
                  std::sin(dlon/2) * std::sin(dlon/2);
        double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1-a));

        path.ground_range = EARTH_RADIUS_KM * c;

        // Find apex
        auto max_alt_it = std::max_element(
            path.positions.begin(), path.positions.end(),
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a(2) < b(2);
            });

        if (max_alt_it != path.positions.end()) {
            path.apex_altitude = (*max_alt_it)(2);
            path.apex_lat = (*max_alt_it)(0);
            path.apex_lon = (*max_alt_it)(1);
        }
    }

    return path;
}

std::vector<RayPath> RayTracer3D::trace_ray_fan(
    double lat0,
    double lon0,
    double alt0,
    const std::vector<double>& elevations,
    const std::vector<double>& azimuths,
    double freq_mhz)
{
    std::vector<RayPath> paths;

    for (double azim : azimuths) {
        for (double elev : elevations) {
            RayPath path = trace_ray(lat0, lon0, alt0, elev, azim, freq_mhz);
            paths.push_back(path);
        }
    }

    return paths;
}

std::vector<RayPath> RayTracer3D::calculate_nvis_coverage(
    double tx_lat,
    double tx_lon,
    double freq_mhz,
    double elevation_min,
    double elevation_max,
    double elevation_step,
    double azimuth_step)
{
    std::vector<double> elevations;
    std::vector<double> azimuths;

    for (double elev = elevation_min; elev <= elevation_max; elev += elevation_step) {
        elevations.push_back(elev);
    }

    for (double azim = 0.0; azim < 360.0; azim += azimuth_step) {
        azimuths.push_back(azim);
    }

    return trace_ray_fan(tx_lat, tx_lon, 0.0, elevations, azimuths, freq_mhz);
}

Eigen::VectorXd RayTracer3D::haselgrove_equations(
    const Eigen::VectorXd& state,
    double freq_hz)
{
    // State: [lat, lon, alt, kx, ky, kz]
    // where k is wave normal direction

    double lat = state(0);
    double lon = state(1);
    double alt = state(2);
    Eigen::Vector3d k(state(3), state(4), state(5));

    // Get electron density and gradient
    double ne = iono_grid_->electron_density(lat, lon, alt);
    Eigen::Vector3d grad_ne = iono_grid_->electron_density_gradient(lat, lon, alt);

    // Safety check: minimum electron density to avoid division by zero
    ne = std::max(ne, 1e6);  // Minimum 1e6 el/m³

    // Refractive index and its gradient
    // For simplified version, assume ∇n ≈ ∇ne / (2*ne)
    double f_p = plasma_frequency_hz(ne);
    double X = (f_p / freq_hz) * (f_p / freq_hz);
    double n = std::sqrt(std::max(0.0, 1.0 - X));  // Ensure non-negative

    // Safety check: avoid division by near-zero refractive index
    n = std::max(n, 0.01);

    Eigen::Vector3d grad_n = grad_ne / (2.0 * ne * n);

    // Haselgrove equations:
    // dr/ds = (c/f) * k / n
    // dk/ds = -(f/c) * ∇n

    Eigen::VectorXd derivatives(6);

    // Position derivatives (convert to km per km of path)
    derivatives(0) = k(0) / n;  // dlat/ds
    derivatives(1) = k(1) / n;  // dlon/ds
    derivatives(2) = k(2) / n;  // dalt/ds

    // Wave normal derivatives
    derivatives(3) = -grad_n(0);  // dkx/ds
    derivatives(4) = -grad_n(1);  // dky/ds
    derivatives(5) = -grad_n(2);  // dkz/ds

    return derivatives;
}

RayState RayTracer3D::integrate_step(
    const RayState& state,
    double freq_hz,
    double& step_size)
{
    // RK45 (Dormand-Prince) adaptive step integrator with iteration instead of recursion

    Eigen::VectorXd y(6);
    y << state.position, state.wave_normal;

    Eigen::VectorXd y_new, y_4th;
    double error = 0.0;
    int retry_count = 0;
    const int max_retries = 20;  // Prevent infinite loop

    // Iterate until acceptable error or max retries
    while (retry_count < max_retries) {
        // RK45 coefficients
        auto k1 = haselgrove_equations(y, freq_hz);
        auto k2 = haselgrove_equations(y + step_size * 0.2 * k1, freq_hz);
        auto k3 = haselgrove_equations(y + step_size * (0.075 * k1 + 0.225 * k2), freq_hz);
        auto k4 = haselgrove_equations(y + step_size * (0.3 * k1 - 0.9 * k2 + 1.2 * k3), freq_hz);
        auto k5 = haselgrove_equations(y + step_size * (-11.0/54.0 * k1 + 2.5 * k2 - 70.0/27.0 * k3 + 35.0/27.0 * k4), freq_hz);
        auto k6 = haselgrove_equations(y + step_size * (1631.0/55296.0 * k1 + 175.0/512.0 * k2 + 575.0/13824.0 * k3 + 44275.0/110592.0 * k4 + 253.0/4096.0 * k5), freq_hz);

        // 5th order solution
        y_new = y + step_size * (37.0/378.0 * k1 + 250.0/621.0 * k3 + 125.0/594.0 * k4 + 512.0/1771.0 * k6);

        // 4th order solution for error estimate
        y_4th = y + step_size * (2825.0/27648.0 * k1 + 18575.0/48384.0 * k3 + 13525.0/55296.0 * k4 + 277.0/14336.0 * k5 + 0.25 * k6);

        // Error estimate
        error = (y_new - y_4th).norm();

        // Check if error is acceptable or NaN
        if (!std::isfinite(error) || error <= config_.tolerance) {
            break;
        }

        // Reduce step size and retry
        step_size *= 0.5;
        step_size = std::max(step_size, config_.min_step_km);
        retry_count++;

        // If we hit minimum step size, accept the result
        if (step_size <= config_.min_step_km) {
            break;
        }
    }

    // Increase step size for next iteration if error was small
    if (std::isfinite(error) && error < config_.tolerance * 0.5) {
        step_size = std::min(step_size * 1.5, config_.max_step_km);
    }

    // Safety check: ensure step_size is finite and positive
    if (!std::isfinite(step_size) || step_size <= 0.0) {
        step_size = config_.initial_step_km;
    }

    // Update state
    RayState new_state;
    new_state.position = y_new.head<3>();
    new_state.wave_normal = y_new.tail<3>();
    new_state.wave_normal.normalize();
    new_state.path_length = state.path_length + step_size;
    new_state.absorption_db = state.absorption_db;

    return new_state;
}

bool RayTracer3D::check_termination(const RayState& state, RayPath& path) {
    // Check ground hit (allow starting exactly at ground level)
    if (state.position(2) < config_.ground_altitude_km) {
        return true;
    }

    // Check escape to space
    if (state.position(2) >= config_.escape_altitude_km) {
        path.escaped = true;
        return true;
    }

    // Check maximum path length
    if (state.path_length >= config_.max_path_length_km) {
        return true;
    }

    // Check absorption
    if (state.absorption_db >= config_.absorption_threshold_db) {
        path.absorbed = true;
        return true;
    }

    return false;
}

double RayTracer3D::calculate_signal_strength(const RayPath& path) {
    // Free space path loss + absorption
    if (path.positions.size() < 2 || path.path_lengths.empty()) return -999.0;

    double distance_m = path.path_lengths.back() * 1000.0;

    // Free space path loss (dB)
    // FSPL = 20*log10(4*pi*d/lambda)
    double lambda_m = C_LIGHT / (path.positions.size() > 0 ? 5e6 : 1.0);  // Approximate
    double fspl_db = 20.0 * std::log10(4.0 * PI * distance_m / lambda_m);

    // Total loss
    double total_loss_db = fspl_db + (path.absorption_db.empty() ? 0.0 : path.absorption_db.back());

    // Assume 100W transmit power (50 dBm)
    double signal_strength_dbm = 50.0 - total_loss_db;

    return signal_strength_dbm;
}

Eigen::Vector3d RayTracer3D::geo_to_cartesian(double lat, double lon, double alt) const {
    double lat_rad = lat * DEG_TO_RAD;
    double lon_rad = lon * DEG_TO_RAD;
    double r = EARTH_RADIUS_KM + alt;

    return Eigen::Vector3d(
        r * std::cos(lat_rad) * std::cos(lon_rad),
        r * std::cos(lat_rad) * std::sin(lon_rad),
        r * std::sin(lat_rad)
    );
}

void RayTracer3D::cartesian_to_geo(const Eigen::Vector3d& cart,
                                   double& lat, double& lon, double& alt) const {
    double r = cart.norm();
    alt = r - EARTH_RADIUS_KM;

    lat = std::asin(cart(2) / r) * RAD_TO_DEG;
    lon = std::atan2(cart(1), cart(0)) * RAD_TO_DEG;
}

} // namespace propagation
} // namespace autonvis
