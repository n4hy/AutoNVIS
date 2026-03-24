/**
 * @file ray_tracer_3d.hpp
 * @brief 3D Ionospheric Ray Tracing Engine (PHaRLAP-equivalent)
 *
 * Pure C++ implementation of 3D HF ray tracing through ionospheric plasma
 * using Haselgrove equations and magnetoionic theory.
 *
 * Functionally equivalent to PHaRLAP but without MATLAB dependency.
 */

#ifndef RAY_TRACER_3D_HPP
#define RAY_TRACER_3D_HPP

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>

namespace autonvis {
namespace propagation {

/**
 * @brief Ray state: position and wave normal direction
 */
struct RayState {
    Eigen::Vector3d position;      // [lat, lon, alt] in degrees, degrees, km
    Eigen::Vector3d wave_normal;   // Wave normal direction (unit vector)
    double path_length;            // Cumulative path length (km)
    double group_path;             // Group path length (km)
    double phase_path;             // Phase path length (km)
    double absorption_db;          // Cumulative absorption (dB)

    RayState() : position(Eigen::Vector3d::Zero()),
                 wave_normal(Eigen::Vector3d::Zero()),
                 path_length(0.0),
                 group_path(0.0),
                 phase_path(0.0),
                 absorption_db(0.0) {}
};

/**
 * @brief Complete ray path with all state history
 */
struct RayPath {
    std::vector<Eigen::Vector3d> positions;      // Ray positions
    std::vector<Eigen::Vector3d> wave_normals;   // Wave normal directions
    std::vector<double> path_lengths;            // Path length at each point
    std::vector<double> group_paths;             // Group path at each point
    std::vector<double> refractive_indices;      // Refractive index along path
    std::vector<double> absorption_db;           // Cumulative absorption

    double ground_range;      // Total ground range (km)
    double apex_altitude;     // Maximum altitude (km)
    double apex_lat;          // Latitude at apex
    double apex_lon;          // Longitude at apex

    bool reflected;           // Ray reflected from ionosphere
    bool escaped;             // Ray escaped to space
    bool absorbed;            // Ray absorbed in D-region

    RayPath() : ground_range(0.0), apex_altitude(0.0),
                apex_lat(0.0), apex_lon(0.0),
                reflected(false), escaped(false), absorbed(false) {}
};

/**
 * @brief D-region absorption model parameters
 *
 * Based on ITU-R P.531 and George-Bradley model
 */
struct DRegionParams {
    double solar_zenith_angle = 0.0;     // Solar zenith angle (radians)
    double xray_flux = 0.0;              // GOES X-ray flux (W/m²)
    double sunspot_number = 100.0;       // Smoothed sunspot number (R12)
    int month = 6;                       // Month (1-12) for seasonal variation
    int year = 2026;                     // Year for solar cycle
    double gyro_frequency = 1.3e6;       // Gyrofrequency at 90 km (Hz)

    DRegionParams() = default;
};

/**
 * @brief D-region absorption calculator
 *
 * Implements ITU-R P.531 / George-Bradley non-deviative absorption model
 * with X-ray enhancement for Sudden Ionospheric Disturbances (SIDs)
 */
class DRegionAbsorption {
public:
    /**
     * @brief Calculate solar zenith angle
     *
     * @param lat Geographic latitude (degrees)
     * @param lon Geographic longitude (degrees)
     * @param day_of_year Day number (1-365)
     * @param ut_hour Universal time (hours, 0-24)
     * @return Solar zenith angle in radians
     */
    static double solar_zenith_angle(double lat, double lon, int day_of_year, double ut_hour);

    /**
     * @brief Calculate collision frequency using MSIS-like model
     *
     * @param alt Altitude (km)
     * @param params D-region parameters
     * @return Collision frequency (Hz)
     */
    static double collision_frequency(double alt, const DRegionParams& params);

    /**
     * @brief Calculate non-deviative absorption (ITU-R P.531)
     *
     * @param freq Frequency (Hz)
     * @param alt Altitude (km)
     * @param params D-region parameters
     * @return Absorption coefficient (dB/km)
     */
    static double absorption_db_per_km(double freq, double alt, const DRegionParams& params);

    /**
     * @brief Calculate total path absorption through D-region
     *
     * @param freq Frequency (Hz)
     * @param elevation Elevation angle (degrees)
     * @param params D-region parameters
     * @return Total absorption (dB)
     */
    static double total_absorption(double freq, double elevation, const DRegionParams& params);

    /**
     * @brief Calculate X-ray enhancement factor for SID
     *
     * @param xray_flux GOES X-ray flux (W/m²)
     * @param alt Altitude (km)
     * @return Enhancement factor (multiplier, >= 1.0)
     */
    static double xray_enhancement(double xray_flux, double alt);

private:
    // Physical constants
    static constexpr double EARTH_RADIUS_KM = 6371.0;
    static constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;
};

/**
 * @brief Ionospheric grid interpolator
 */
class IonoGrid {
public:
    /**
     * @brief Constructor with grid data
     *
     * @param lat Latitude grid (degrees)
     * @param lon Longitude grid (degrees)
     * @param alt Altitude grid (km)
     * @param ne_grid Electron density grid (flattened, row-major order)
     */
    IonoGrid(const Eigen::VectorXd& lat,
             const Eigen::VectorXd& lon,
             const Eigen::VectorXd& alt,
             const std::vector<double>& ne_grid);

    // Get electron density at arbitrary position (trilinear interpolation)
    double electron_density(double lat, double lon, double alt) const;

    // Get electron density gradient (for ray equation)
    Eigen::Vector3d electron_density_gradient(double lat, double lon, double alt) const;

    // Get collision frequency (legacy interface - use DRegionAbsorption for full model)
    double collision_frequency(double alt, double xray_flux = 0.0) const;

    // Get collision frequency with full D-region model
    double collision_frequency_full(double alt, const DRegionParams& params) const;

    // Get grid dimensions
    size_t n_lat() const { return lat_.size(); }
    size_t n_lon() const { return lon_.size(); }
    size_t n_alt() const { return alt_.size(); }

private:
    Eigen::VectorXd lat_;
    Eigen::VectorXd lon_;
    Eigen::VectorXd alt_;
    std::vector<double> ne_grid_;  // Flattened 3D array (row-major: lat, lon, alt)

    // Grid spacing for finite differences
    double dlat_, dlon_, dalt_;

    // Helper to access 3D grid from flattened array
    inline double get_ne(size_t i, size_t j, size_t k) const {
        size_t idx = i * lon_.size() * alt_.size() + j * alt_.size() + k;
        return ne_grid_[idx];
    }
};

/**
 * @brief IGRF-13 geomagnetic field model
 *
 * Implements International Geomagnetic Reference Field (13th generation)
 * with built-in coefficients for 2020-2025 epoch and secular variation.
 * Falls back to enhanced dipole model if spherical harmonics unavailable.
 */
class GeomagneticField {
public:
    /**
     * @brief Default constructor - uses built-in IGRF-13 coefficients
     */
    GeomagneticField();

    /**
     * @brief Constructor with external coefficient file
     */
    explicit GeomagneticField(const std::string& igrf_coeffs_file);

    /**
     * @brief Get magnetic field vector at position
     *
     * @param lat Geographic latitude (degrees)
     * @param lon Geographic longitude (degrees)
     * @param alt Altitude above sea level (km)
     * @param year Decimal year (e.g., 2026.5)
     * @return Field vector [North, East, Down] in nT
     */
    Eigen::Vector3d field(double lat, double lon, double alt, double year = 2026.0) const;

    // Overload for integer year (legacy compatibility)
    Eigen::Vector3d field(double lat, double lon, double alt, int year) const {
        return field(lat, lon, alt, static_cast<double>(year));
    }

    /**
     * @brief Get magnetic field magnitude (total intensity)
     */
    double field_magnitude(double lat, double lon, double alt, double year = 2026.0) const;
    double field_magnitude(double lat, double lon, double alt, int year) const {
        return field_magnitude(lat, lon, alt, static_cast<double>(year));
    }

    /**
     * @brief Get magnetic dip (inclination) angle
     * @return Dip angle in radians (positive downward in northern hemisphere)
     */
    double dip_angle(double lat, double lon, double alt, double year = 2026.0) const;
    double dip_angle(double lat, double lon, double alt, int year) const {
        return dip_angle(lat, lon, alt, static_cast<double>(year));
    }

    /**
     * @brief Get magnetic declination angle
     * @return Declination in radians (positive eastward)
     */
    double declination(double lat, double lon, double alt, double year = 2026.0) const;

    /**
     * @brief Get gyrofrequency at position
     * @return Electron gyrofrequency in Hz
     */
    double gyro_frequency(double lat, double lon, double alt, double year = 2026.0) const;

    /**
     * @brief Check if full IGRF model is available
     */
    bool has_igrf_coefficients() const { return igrf_loaded_; }

private:
    // IGRF-13 Gauss coefficients (g_n^m, h_n^m)
    // Indexed as g_coeffs_[n][m] for n=1..13, m=0..n
    std::vector<std::vector<double>> g_coeffs_;
    std::vector<std::vector<double>> h_coeffs_;

    // Secular variation coefficients
    std::vector<std::vector<double>> g_sv_;
    std::vector<std::vector<double>> h_sv_;

    // Reference epoch for coefficients
    double epoch_year_ = 2020.0;
    bool igrf_loaded_ = false;

    // Maximum degree of spherical harmonic expansion
    static constexpr int MAX_DEGREE = 13;

    // Initialize with built-in IGRF-13 coefficients
    void initialize_igrf13();

    // Load coefficients from external file
    void load_coefficients(const std::string& filename);

    // Calculate field using spherical harmonics
    Eigen::Vector3d spherical_harmonic_field(double lat, double lon, double alt, double year) const;

    // Enhanced dipole model (fallback)
    Eigen::Vector3d dipole_field(double lat, double lon, double alt) const;

    // Associated Legendre functions
    static double legendre_p(int n, int m, double x);
    static double legendre_dp(int n, int m, double x);
};

/**
 * @brief Magnetoionic dispersion relation (Appleton-Hartree)
 */
class MagnetoionicTheory {
public:
    enum Mode {
        O_MODE = 0,  // Ordinary mode
        X_MODE = 1   // Extraordinary mode
    };

    /**
     * @brief Calculate refractive index from Appleton-Hartree equation
     *
     * @param ne Electron density (el/m³)
     * @param freq Frequency (Hz)
     * @param B_mag Magnetic field magnitude (Tesla)
     * @param theta Angle between wave normal and B field (radians)
     * @param nu Collision frequency (Hz)
     * @param mode O-mode or X-mode
     * @return Complex refractive index
     */
    static std::complex<double> refractive_index(
        double ne,
        double freq,
        double B_mag,
        double theta,
        double nu = 0.0,
        Mode mode = O_MODE
    );

    /**
     * @brief Calculate group refractive index (mu)
     */
    static double group_refractive_index(
        double ne,
        double freq,
        double B_mag,
        double theta,
        double nu = 0.0,
        Mode mode = O_MODE
    );

    /**
     * @brief Calculate absorption coefficient (Nepers/m)
     */
    static double absorption_coefficient(
        double ne,
        double freq,
        double nu
    );

private:
    // Physical constants
    static constexpr double ELECTRON_CHARGE = 1.60217663e-19;  // C
    static constexpr double ELECTRON_MASS = 9.1093837e-31;     // kg
    static constexpr double EPSILON_0 = 8.8541878e-12;         // F/m
    static constexpr double C_LIGHT = 2.99792458e8;            // m/s
};

/**
 * @brief Ray integration parameters
 */
struct RayTracingConfig {
    double tolerance = 1e-7;              // Integration tolerance
    double max_path_length_km = 20000.0;  // Maximum path length
    double initial_step_km = 0.5;         // Initial step size
    double min_step_km = 0.01;            // Minimum step size
    double max_step_km = 10.0;            // Maximum step size
    int max_steps = 100000;               // Maximum integration steps

    double ground_altitude_km = 0.0;      // Ground level
    double escape_altitude_km = 1000.0;   // Escape to space threshold
    double absorption_threshold_db = 100.0; // Ray absorbed if exceeded

    bool calculate_absorption = true;     // Include D-region absorption
    bool calculate_group_path = true;     // Calculate group path

    MagnetoionicTheory::Mode mode = MagnetoionicTheory::O_MODE;

    RayTracingConfig() = default;
};

/**
 * @brief 3D HF Ray Tracer
 *
 * Implements Haselgrove equations for 3D ray tracing through
 * magnetized ionospheric plasma.
 */
class RayTracer3D {
public:
    /**
     * @brief Constructor
     *
     * @param iono_grid Ionospheric electron density grid
     * @param geomag Geomagnetic field model
     * @param config Ray tracing configuration
     */
    RayTracer3D(
        std::shared_ptr<IonoGrid> iono_grid,
        std::shared_ptr<GeomagneticField> geomag,
        const RayTracingConfig& config = RayTracingConfig()
    );

    /**
     * @brief Trace a single ray
     *
     * @param lat0 Initial latitude (degrees)
     * @param lon0 Initial longitude (degrees)
     * @param alt0 Initial altitude (km)
     * @param elevation Elevation angle (degrees)
     * @param azimuth Azimuth angle (degrees, clockwise from North)
     * @param freq_mhz Frequency (MHz)
     * @return Complete ray path
     */
    RayPath trace_ray(
        double lat0,
        double lon0,
        double alt0,
        double elevation,
        double azimuth,
        double freq_mhz
    );

    /**
     * @brief Trace multiple rays (ray fan)
     *
     * @param lat0 Initial latitude
     * @param lon0 Initial longitude
     * @param alt0 Initial altitude
     * @param elevations Vector of elevation angles
     * @param azimuths Vector of azimuth angles
     * @param freq_mhz Frequency
     * @return Vector of ray paths
     */
    std::vector<RayPath> trace_ray_fan(
        double lat0,
        double lon0,
        double alt0,
        const std::vector<double>& elevations,
        const std::vector<double>& azimuths,
        double freq_mhz
    );

    /**
     * @brief Calculate coverage map for NVIS
     *
     * @param tx_lat Transmitter latitude
     * @param tx_lon Transmitter longitude
     * @param freq_mhz Frequency
     * @param elevation_min Minimum elevation (degrees, e.g., 70)
     * @param elevation_max Maximum elevation (degrees, e.g., 90)
     * @param elevation_step Elevation step (degrees)
     * @param azimuth_step Azimuth step (degrees)
     * @return Map of ground ranges and signal strengths
     */
    std::vector<RayPath> calculate_nvis_coverage(
        double tx_lat,
        double tx_lon,
        double freq_mhz,
        double elevation_min = 70.0,
        double elevation_max = 90.0,
        double elevation_step = 2.0,
        double azimuth_step = 15.0
    );

private:
    std::shared_ptr<IonoGrid> iono_grid_;
    std::shared_ptr<GeomagneticField> geomag_;
    RayTracingConfig config_;

    /**
     * @brief Haselgrove ray equations (RHS of ODE system)
     *
     * State vector: [lat, lon, alt, kx, ky, kz]
     * where k is wave normal direction
     */
    Eigen::VectorXd haselgrove_equations(
        const Eigen::VectorXd& state,
        double freq_hz
    );

    /**
     * @brief RK45 adaptive step integrator
     */
    RayState integrate_step(
        const RayState& state,
        double freq_hz,
        double& step_size
    );

    /**
     * @brief Check termination conditions
     */
    bool check_termination(
        const RayState& state,
        RayPath& path
    );

    /**
     * @brief Calculate signal strength at ground
     */
    double calculate_signal_strength(const RayPath& path);

    /**
     * @brief Geographic to Cartesian conversion
     */
    Eigen::Vector3d geo_to_cartesian(double lat, double lon, double alt) const;

    /**
     * @brief Cartesian to Geographic conversion
     */
    void cartesian_to_geo(const Eigen::Vector3d& cart,
                         double& lat, double& lon, double& alt) const;
};

} // namespace propagation
} // namespace autonvis

#endif // RAY_TRACER_3D_HPP
