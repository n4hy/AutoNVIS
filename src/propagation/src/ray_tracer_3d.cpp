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
// DRegionAbsorption Implementation
//=============================================================================

double DRegionAbsorption::solar_zenith_angle(double lat, double lon, int day_of_year, double ut_hour) {
    // Calculate solar zenith angle using standard astronomical formulas

    // Solar declination (radians)
    double gamma = 2.0 * PI * (day_of_year - 1) / 365.0;
    double delta = 0.006918 - 0.399912 * std::cos(gamma) + 0.070257 * std::sin(gamma)
                 - 0.006758 * std::cos(2*gamma) + 0.000907 * std::sin(2*gamma)
                 - 0.002697 * std::cos(3*gamma) + 0.00148 * std::sin(3*gamma);

    // Hour angle (radians)
    double solar_noon_offset = lon / 15.0;  // Hours offset from UTC
    double local_solar_time = ut_hour + solar_noon_offset;
    double hour_angle = (local_solar_time - 12.0) * 15.0 * DEG_TO_RAD;

    // Zenith angle calculation
    double lat_rad = lat * DEG_TO_RAD;
    double cos_zenith = std::sin(lat_rad) * std::sin(delta)
                      + std::cos(lat_rad) * std::cos(delta) * std::cos(hour_angle);

    // Clamp to valid range
    cos_zenith = std::clamp(cos_zenith, -1.0, 1.0);

    return std::acos(cos_zenith);
}

double DRegionAbsorption::collision_frequency(double alt, const DRegionParams& params) {
    // Electron-neutral collision frequency model for D-region
    // Based on Banks & Kockarts (1973) and MSIS atmospheric model

    // Neutral density profile (MSIS-like approximation)
    // Reference values at sea level (molecules/m³)
    double n_N2_0 = 1.95e25;
    double n_O2_0 = 5.18e24;

    // Scale heights (km) - varies with altitude
    double H = 7.0;  // Mean scale height for D-region

    // Simple exponential decay with altitude
    double n_N2 = n_N2_0 * std::exp(-alt / H);
    double n_O2 = n_O2_0 * std::exp(-alt / H);

    // Temperature at altitude (simplified mesospheric profile)
    double T = 220.0;  // ~220 K in D-region (mesopause)

    // Collision frequencies (Banks & Kockarts formulas)
    // ν_N2 = 2.33×10^-17 × n_N2 × (1 - 1.21×10^-4 × Te) × Te  Hz
    // ν_O2 = 1.82×10^-16 × n_O2 × (1 + 3.6×10^-2 × Te^0.5) × Te^0.5  Hz
    // For Te ~ T ~ 220 K (thermal equilibrium in D-region)

    double Te = T;
    double nu_N2 = 2.33e-17 * n_N2 * (1.0 - 1.21e-4 * Te) * Te;
    double nu_O2 = 1.82e-16 * n_O2 * (1.0 + 0.036 * std::sqrt(Te)) * std::sqrt(Te);

    double nu_total = nu_N2 + nu_O2;

    // X-ray enhancement in D-region (increases ionization → changes collision dynamics)
    if (params.xray_flux > 1e-7 && alt < 100.0) {
        double enhancement = xray_enhancement(params.xray_flux, alt);
        // Heating increases collision frequency
        nu_total *= std::pow(enhancement, 0.25);
    }

    return nu_total;
}

double DRegionAbsorption::absorption_db_per_km(double freq, double alt, const DRegionParams& params) {
    // ITU-R P.531-14 non-deviative absorption model
    // Combined with George-Bradley D-region profile

    // Only calculate absorption in D-region (50-100 km)
    if (alt < 50.0 || alt > 100.0) {
        return 0.0;
    }

    // D-region electron density profile (Chapman layer)
    // Peak at ~85 km during day, scale height ~6 km
    double h_peak = 85.0;
    double H_d = 6.0;

    // Daytime peak electron density (el/m³)
    // Ne_max depends on solar zenith angle and solar activity
    double cos_chi = std::cos(params.solar_zenith_angle);
    cos_chi = std::max(cos_chi, 0.0);  // No negative (night)

    // ITU-R P.531: Ne proportional to (cos χ)^0.7 for D-region
    double Ne_max = 1e9 * std::pow(cos_chi + 0.01, 0.7);  // +0.01 for night residual

    // Solar cycle variation (Ne scales ~linearly with R12)
    Ne_max *= (1.0 + 0.004 * params.sunspot_number);  // R12=100 gives factor 1.4

    // Seasonal variation (±15%)
    double seasonal = 1.0 + 0.15 * std::cos(2.0 * PI * (params.month - 6.0) / 12.0);
    Ne_max *= seasonal;

    // Chapman profile
    double z = (alt - h_peak) / H_d;
    double Ne = Ne_max * std::exp(0.5 * (1.0 - z - std::exp(-z)));

    // X-ray enhancement (SID effect)
    if (params.xray_flux > 1e-6) {
        double enhance = xray_enhancement(params.xray_flux, alt);
        Ne *= std::sqrt(enhance);  // Ne scales as sqrt of ionization rate
    }

    // Get collision frequency at this altitude
    double nu = collision_frequency(alt, params);

    // Non-deviative absorption coefficient (ITU-R P.531)
    // κ = (e² / 2 ε₀ m_e c) × (Ne × ν) / (ν² + ω²)  Np/m
    // ω = 2πf, simplified for ω >> ν:
    // κ ≈ (e² Ne ν) / (2 ε₀ m_e c ω²)

    double omega = 2.0 * PI * freq;
    double omega_H = 2.0 * PI * params.gyro_frequency;

    // Include magnetic field effects (ordinary and extraordinary modes)
    double denom_o = nu*nu + (omega - omega_H)*(omega - omega_H);
    double denom_x = nu*nu + (omega + omega_H)*(omega + omega_H);

    // Constants: e²/(2 ε₀ m_e c) = 1.97e-7 m²/s
    double k_const = 1.97e-7;

    double kappa_o = k_const * Ne * nu / denom_o;
    double kappa_x = k_const * Ne * nu / denom_x;

    // Average absorption coefficient (Np/m)
    double kappa = (kappa_o + kappa_x) / 2.0;

    // Convert to dB/km: 1 Np/m = 8686 dB/km
    return kappa * 8686.0;
}

double DRegionAbsorption::total_absorption(double freq, double elevation, const DRegionParams& params) {
    // Integrate absorption through D-region (60-100 km)

    double total_db = 0.0;
    double dh = 1.0;  // 1 km integration step

    // Oblique path factor
    double sec_i = 1.0 / std::sin(std::max(elevation * DEG_TO_RAD, 0.1));
    sec_i = std::min(sec_i, 10.0);  // Limit for very low elevations

    for (double h = 60.0; h <= 100.0; h += dh) {
        double alpha = absorption_db_per_km(freq, h, params);
        total_db += alpha * dh * sec_i;
    }

    return total_db;
}

double DRegionAbsorption::xray_enhancement(double xray_flux, double alt) {
    // X-ray ionization enhancement for SID (Sudden Ionospheric Disturbance)
    // Based on GOES X-ray flux levels

    // Flux thresholds (W/m²)
    // C-class: 1e-6 to 1e-5
    // M-class: 1e-5 to 1e-4
    // X-class: > 1e-4

    if (xray_flux <= 1e-7) {
        return 1.0;  // Quiet conditions
    }

    // Altitude-dependent absorption of X-rays
    double penetration_depth = std::exp(-(90.0 - alt) / 10.0);  // X-rays absorbed above 90 km
    if (alt > 90.0) penetration_depth = 1.0;

    // Enhancement factor (logarithmic with flux)
    double log_flux = std::log10(xray_flux);
    double enhancement = 1.0;

    if (log_flux > -6.0) {
        // C-class and above
        enhancement = 1.0 + 2.0 * (log_flux + 6.0) * penetration_depth;
    }
    if (log_flux > -5.0) {
        // M-class and above
        enhancement += 5.0 * (log_flux + 5.0) * penetration_depth;
    }
    if (log_flux > -4.0) {
        // X-class
        enhancement += 20.0 * (log_flux + 4.0) * penetration_depth;
    }

    return std::max(1.0, enhancement);
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

double IonoGrid::collision_frequency_full(double alt, const DRegionParams& params) const {
    // Full D-region collision frequency using DRegionAbsorption model
    return DRegionAbsorption::collision_frequency(alt, params);
}

//=============================================================================
// GeomagneticField Implementation
//=============================================================================

GeomagneticField::GeomagneticField() {
    // Initialize with built-in IGRF-13 coefficients
    initialize_igrf13();
}

GeomagneticField::GeomagneticField(const std::string& igrf_coeffs_file) {
    load_coefficients(igrf_coeffs_file);
}

Eigen::Vector3d GeomagneticField::field(double lat, double lon, double alt, double year) const {
    // If IGRF coefficients loaded, use full spherical harmonic expansion
    if (igrf_loaded_) {
        return spherical_harmonic_field(lat, lon, alt, year);
    }

    // Otherwise use dipole approximation
    return dipole_field(lat, lon, alt);
}

double GeomagneticField::field_magnitude(double lat, double lon, double alt, double year) const {
    Eigen::Vector3d B = field(lat, lon, alt, year);
    return B.norm();
}

double GeomagneticField::dip_angle(double lat, double lon, double alt, double year) const {
    Eigen::Vector3d B = field(lat, lon, alt, year);
    double B_horizontal = std::sqrt(B(0)*B(0) + B(1)*B(1));
    return std::atan2(B(2), B_horizontal);  // Radians
}

double GeomagneticField::declination(double lat, double lon, double alt, double year) const {
    Eigen::Vector3d B = field(lat, lon, alt, year);
    // Declination: angle from true north to magnetic north (positive eastward)
    return std::atan2(B(1), B(0));  // Radians
}

double GeomagneticField::gyro_frequency(double lat, double lon, double alt, double year) const {
    // Electron gyrofrequency: f_g = e * B / (2 * pi * m_e)
    double B_mag_nT = field_magnitude(lat, lon, alt, year);
    double B_Tesla = B_mag_nT * 1e-9;

    // f_g = e * B / (2 * pi * m_e)
    // e = 1.602e-19 C, m_e = 9.109e-31 kg
    // f_g ≈ 2.8e10 * B (Hz, with B in Tesla)
    // Or f_g ≈ 28 * B (Hz, with B in nT)
    return 2.8e10 * B_Tesla;  // Hz
}

Eigen::Vector3d GeomagneticField::dipole_field(double lat, double lon, double alt) const {
    // Simple dipole approximation
    // B0 = 31,000 nT at equator (returns field in nT)

    double B0 = 31000.0;  // nT
    double r = EARTH_RADIUS_KM + alt;
    double lat_rad = lat * DEG_TO_RAD;
    double r_ratio_cubed = std::pow(EARTH_RADIUS_KM / r, 3);

    // Dipole field components (local North, East, Down) in nT
    // B_r = -2 * B0 * sin(lat) * (R/r)^3  (radial, positive outward)
    // B_theta = -B0 * cos(lat) * (R/r)^3  (southward)
    // Convert to North-East-Down:
    double B_north = B0 * r_ratio_cubed * std::cos(lat_rad);  // Horizontal component toward equator
    double B_east = 0.0;
    double B_down = 2.0 * B0 * r_ratio_cubed * std::sin(lat_rad);  // Vertical component (positive down in NH)

    return Eigen::Vector3d(B_north, B_east, B_down);
}

void GeomagneticField::load_coefficients(const std::string& filename) {
    // Parse IGRF coefficient file (standard IAGA format)
    // For now, use built-in coefficients
    initialize_igrf13();
}

void GeomagneticField::initialize_igrf13() {
    // Initialize IGRF-13 Gauss coefficients for 2020.0 epoch
    // Coefficients from NOAA/NCEI IGRF-13 official release
    // https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html

    // Allocate coefficient arrays (degree 0 to MAX_DEGREE=13)
    g_coeffs_.resize(MAX_DEGREE + 1);
    h_coeffs_.resize(MAX_DEGREE + 1);
    g_sv_.resize(MAX_DEGREE + 1);
    h_sv_.resize(MAX_DEGREE + 1);

    for (int n = 0; n <= MAX_DEGREE; ++n) {
        g_coeffs_[n].resize(n + 1, 0.0);
        h_coeffs_[n].resize(n + 1, 0.0);
        g_sv_[n].resize(n + 1, 0.0);
        h_sv_[n].resize(n + 1, 0.0);
    }

    // IGRF-13 coefficients for 2020.0 epoch (nT)
    // Degree 1 (dipole)
    g_coeffs_[1][0] = -29404.8;
    g_coeffs_[1][1] = -1450.9;  h_coeffs_[1][1] = 4652.5;

    // Degree 2 (quadrupole)
    g_coeffs_[2][0] = -2499.6;
    g_coeffs_[2][1] = 2982.0;   h_coeffs_[2][1] = -2991.6;
    g_coeffs_[2][2] = 1677.0;   h_coeffs_[2][2] = -734.6;

    // Degree 3 (octupole)
    g_coeffs_[3][0] = 1363.2;
    g_coeffs_[3][1] = -2381.2;  h_coeffs_[3][1] = -82.1;
    g_coeffs_[3][2] = 1236.2;   h_coeffs_[3][2] = 241.9;
    g_coeffs_[3][3] = 525.7;    h_coeffs_[3][3] = -543.4;

    // Degree 4
    g_coeffs_[4][0] = 903.0;
    g_coeffs_[4][1] = 809.5;    h_coeffs_[4][1] = 281.9;
    g_coeffs_[4][2] = 86.3;     h_coeffs_[4][2] = -158.4;
    g_coeffs_[4][3] = -309.4;   h_coeffs_[4][3] = 199.7;
    g_coeffs_[4][4] = 48.0;     h_coeffs_[4][4] = -349.7;

    // Degree 5
    g_coeffs_[5][0] = -234.3;
    g_coeffs_[5][1] = 363.2;    h_coeffs_[5][1] = 47.7;
    g_coeffs_[5][2] = 187.8;    h_coeffs_[5][2] = 208.3;
    g_coeffs_[5][3] = -140.7;   h_coeffs_[5][3] = -121.2;
    g_coeffs_[5][4] = -151.2;   h_coeffs_[5][4] = 32.3;
    g_coeffs_[5][5] = 13.5;     h_coeffs_[5][5] = 98.9;

    // Degree 6
    g_coeffs_[6][0] = 66.0;
    g_coeffs_[6][1] = 65.5;     h_coeffs_[6][1] = -19.1;
    g_coeffs_[6][2] = 72.9;     h_coeffs_[6][2] = 25.1;
    g_coeffs_[6][3] = -121.5;   h_coeffs_[6][3] = 52.8;
    g_coeffs_[6][4] = -36.2;    h_coeffs_[6][4] = -64.5;
    g_coeffs_[6][5] = 13.5;     h_coeffs_[6][5] = 8.9;
    g_coeffs_[6][6] = -64.7;    h_coeffs_[6][6] = 68.1;

    // Degree 7
    g_coeffs_[7][0] = 80.6;
    g_coeffs_[7][1] = -76.7;    h_coeffs_[7][1] = -51.5;
    g_coeffs_[7][2] = -8.2;     h_coeffs_[7][2] = -16.9;
    g_coeffs_[7][3] = 56.5;     h_coeffs_[7][3] = 2.2;
    g_coeffs_[7][4] = 15.8;     h_coeffs_[7][4] = 23.5;
    g_coeffs_[7][5] = 6.4;      h_coeffs_[7][5] = -2.2;
    g_coeffs_[7][6] = -7.2;     h_coeffs_[7][6] = -27.2;
    g_coeffs_[7][7] = 9.8;      h_coeffs_[7][7] = -1.8;

    // Degree 8
    g_coeffs_[8][0] = 23.7;
    g_coeffs_[8][1] = 9.7;      h_coeffs_[8][1] = 8.4;
    g_coeffs_[8][2] = -17.6;    h_coeffs_[8][2] = -15.3;
    g_coeffs_[8][3] = -0.5;     h_coeffs_[8][3] = 12.8;
    g_coeffs_[8][4] = -21.1;    h_coeffs_[8][4] = -11.7;
    g_coeffs_[8][5] = 15.3;     h_coeffs_[8][5] = 14.9;
    g_coeffs_[8][6] = 13.7;     h_coeffs_[8][6] = 3.6;
    g_coeffs_[8][7] = -16.5;    h_coeffs_[8][7] = -6.9;
    g_coeffs_[8][8] = -0.3;     h_coeffs_[8][8] = 2.8;

    // Secular variation coefficients (nT/year) for 2020-2025
    // Degree 1
    g_sv_[1][0] = 5.7;
    g_sv_[1][1] = 7.4;          h_sv_[1][1] = -25.9;

    // Degree 2
    g_sv_[2][0] = -11.0;
    g_sv_[2][1] = -7.0;         h_sv_[2][1] = -30.2;
    g_sv_[2][2] = -2.1;         h_sv_[2][2] = -22.4;

    // Degree 3
    g_sv_[3][0] = 2.2;
    g_sv_[3][1] = -5.9;         h_sv_[3][1] = 6.0;
    g_sv_[3][2] = 3.1;          h_sv_[3][2] = -1.1;
    g_sv_[3][3] = -12.0;        h_sv_[3][3] = 0.5;

    // Higher degree SV coefficients are smaller - omit for brevity

    epoch_year_ = 2020.0;
    igrf_loaded_ = true;
}

Eigen::Vector3d GeomagneticField::spherical_harmonic_field(double lat, double lon, double alt, double year) const {
    // Full IGRF-13 spherical harmonic expansion
    // B = -∇V where V = a Σ (a/r)^(n+1) Σ [g_n^m cos(mφ) + h_n^m sin(mφ)] P_n^m(cos θ)

    if (!igrf_loaded_) {
        return dipole_field(lat, lon, alt);
    }

    // Convert to geocentric spherical coordinates
    double lat_rad = lat * DEG_TO_RAD;
    double lon_rad = lon * DEG_TO_RAD;
    double theta = PI / 2.0 - lat_rad;  // Colatitude
    double phi = lon_rad;

    double r = EARTH_RADIUS_KM + alt;
    double a = EARTH_RADIUS_KM;
    double a_over_r = a / r;

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    // Time interpolation for secular variation
    double dt = year - epoch_year_;
    dt = std::clamp(dt, -5.0, 10.0);  // Limit extrapolation

    // Compute field components in spherical coordinates (Br, Btheta, Bphi)
    double Br = 0.0;
    double Bt = 0.0;
    double Bp = 0.0;

    // Precompute powers of (a/r)
    std::vector<double> r_power(MAX_DEGREE + 2);
    r_power[0] = 1.0;
    for (int n = 1; n <= MAX_DEGREE + 1; ++n) {
        r_power[n] = r_power[n-1] * a_over_r;
    }

    // Precompute cos(mφ) and sin(mφ)
    std::vector<double> cos_mphi(MAX_DEGREE + 1);
    std::vector<double> sin_mphi(MAX_DEGREE + 1);
    for (int m = 0; m <= MAX_DEGREE; ++m) {
        cos_mphi[m] = std::cos(m * phi);
        sin_mphi[m] = std::sin(m * phi);
    }

    // Sum over degrees and orders
    for (int n = 1; n <= std::min(8, MAX_DEGREE); ++n) {  // Use degrees 1-8
        double r_n2 = r_power[n + 2];  // (a/r)^(n+2)
        double r_n1 = r_power[n + 1];  // (a/r)^(n+1)

        for (int m = 0; m <= n; ++m) {
            // Get time-interpolated coefficients
            double g = g_coeffs_[n][m] + g_sv_[n][m] * dt;
            double h = (m > 0) ? h_coeffs_[n][m] + h_sv_[n][m] * dt : 0.0;

            // Compute Legendre functions
            double Pnm = legendre_p(n, m, cos_theta);
            double dPnm = legendre_dp(n, m, cos_theta);

            // Accumulate field components
            double Gnm = g * cos_mphi[m] + h * sin_mphi[m];
            double Hnm = (m > 0) ? m * (-g * sin_mphi[m] + h * cos_mphi[m]) : 0.0;

            // Br = -∂V/∂r = Σ (n+1)(a/r)^(n+2) Σ Gnm Pnm
            Br += (n + 1) * r_n2 * Gnm * Pnm;

            // Btheta = -(1/r)∂V/∂θ = -(a/r)^(n+2) Σ Gnm dPnm/dθ
            Bt -= r_n2 * Gnm * dPnm;

            // Bphi = -(1/(r sin θ))∂V/∂φ = (a/r)^(n+2) / sin(θ) Σ Hnm Pnm
            if (std::abs(sin_theta) > 1e-10) {
                Bp += r_n2 * Hnm * Pnm / sin_theta;
            }
        }
    }

    // Convert from spherical (Br, Btheta, Bphi) to North-East-Down (Bn, Be, Bd)
    // Bn = -Btheta (North = -θ direction)
    // Be = Bphi (East = φ direction)
    // Bd = -Br (Down = -r direction)
    double Bn = -Bt;
    double Be = Bp;
    double Bd = -Br;

    return Eigen::Vector3d(Bn, Be, Bd);
}

double GeomagneticField::legendre_p(int n, int m, double x) {
    // Schmidt semi-normalized associated Legendre function P_n^m(cos θ)
    // Uses recurrence relations for numerical stability

    if (m > n) return 0.0;
    if (n < 0) return 0.0;

    // Special cases
    if (n == 0 && m == 0) return 1.0;
    if (n == 1 && m == 0) return x;
    if (n == 1 && m == 1) return std::sqrt(1.0 - x*x);

    double pmm = 1.0;
    double somx2 = std::sqrt(1.0 - x*x);

    // Compute P_m^m using (2m-1)!! factor
    if (m > 0) {
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= somx2 * fact / (fact + 1.0);
            fact += 2.0;
        }
        pmm = std::sqrt(2.0 * pmm);  // Schmidt normalization
    }

    if (n == m) return pmm;

    // Compute P_{m+1}^m
    double pmm1 = x * std::sqrt(2.0 * m + 1.0) * pmm;
    if (n == m + 1) return pmm1;

    // Use recurrence: P_n^m = x*(2n-1)/sqrt((n-m)(n+m)) P_{n-1}^m - sqrt((n-1-m)(n-1+m)/(n-m)(n+m)) P_{n-2}^m
    double pnm = 0.0;
    double pnm2 = pmm;
    double pnm1 = pmm1;

    for (int nn = m + 2; nn <= n; ++nn) {
        double a = std::sqrt((4.0*nn*nn - 1.0) / ((double)(nn*nn - m*m)));
        double b = std::sqrt(((nn-1.0)*(nn-1.0) - m*m) / (4.0*(nn-1.0)*(nn-1.0) - 1.0));
        pnm = a * (x * pnm1 - b * pnm2);
        pnm2 = pnm1;
        pnm1 = pnm;
    }

    return pnm;
}

double GeomagneticField::legendre_dp(int n, int m, double x) {
    // Derivative of Schmidt semi-normalized associated Legendre function
    // dP_n^m/dθ = d(P_n^m(cos θ))/dθ = -sin(θ) * dP_n^m/dx

    double sin_theta = std::sqrt(1.0 - x*x);

    if (std::abs(sin_theta) < 1e-10) {
        // Near poles, use special handling
        return 0.0;
    }

    // Use recurrence: dP_n^m/dθ = n*x*P_n^m/sin²θ - (n+m)*P_{n-1}^m/sinθ
    // Or equivalently: dP_n^m/dθ = (n*cos(θ)*P_n^m - (n+m)*P_{n-1}^m) / sin(θ)

    double Pnm = legendre_p(n, m, x);
    double Pn1m = (n > m) ? legendre_p(n - 1, m, x) : 0.0;

    // Schmidt normalization factor
    double factor = std::sqrt(((double)(n - m) * (n + m + 1.0)));

    return (n * x * Pnm - factor * Pn1m) / sin_theta;
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

        // Calculate absorption if enabled (D-region: 50-100 km)
        if (config_.calculate_absorption && state.position(2) >= 50.0 && state.position(2) <= 100.0) {
            // Set up D-region parameters for current position
            DRegionParams d_params;

            // Calculate solar zenith angle at ray position
            // Use day 172 (June 21) and 12:00 UT as defaults for now
            // TODO: Accept time parameters from config or propagation service
            int day_of_year = 172;  // Summer solstice
            double ut_hour = 12.0;  // Noon UT
            d_params.solar_zenith_angle = DRegionAbsorption::solar_zenith_angle(
                state.position(0), state.position(1), day_of_year, ut_hour);

            // Get gyrofrequency at D-region altitude
            d_params.gyro_frequency = geomag_->gyro_frequency(
                state.position(0), state.position(1), state.position(2));

            d_params.sunspot_number = 100.0;  // Default moderate solar activity
            d_params.xray_flux = 0.0;  // Quiet conditions (can be set from space weather)
            d_params.month = 6;
            d_params.year = 2026;

            // Calculate absorption using ITU-R P.531 / George-Bradley model
            double alpha_db_km = DRegionAbsorption::absorption_db_per_km(
                freq_hz, state.position(2), d_params);

            // Account for oblique path through D-region
            double sec_elevation = 1.0 / std::max(std::sin(elevation * DEG_TO_RAD), 0.1);
            sec_elevation = std::min(sec_elevation, 10.0);  // Cap at 10 for numerical stability

            state.absorption_db += alpha_db_km * step_size * sec_elevation;
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
