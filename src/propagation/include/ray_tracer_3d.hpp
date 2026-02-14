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

    // Get collision frequency (from X-ray flux model)
    double collision_frequency(double alt, double xray_flux = 0.0) const;

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
 * @brief IGRF geomagnetic field model
 */
class GeomagneticField {
public:
    GeomagneticField();
    explicit GeomagneticField(const std::string& igrf_coeffs_file);

    // Get magnetic field vector at position (nT)
    Eigen::Vector3d field(double lat, double lon, double alt, int year = 2026) const;

    // Get magnetic field magnitude (nT)
    double field_magnitude(double lat, double lon, double alt, int year = 2026) const;

    // Get magnetic dip angle (radians)
    double dip_angle(double lat, double lon, double alt, int year = 2026) const;

private:
    // IGRF-13 Gauss coefficients
    std::vector<std::vector<double>> g_coeffs_;
    std::vector<std::vector<double>> h_coeffs_;

    // Load IGRF coefficients from file
    void load_coefficients(const std::string& filename);

    // Use simple dipole model if IGRF not available
    Eigen::Vector3d dipole_field(double lat, double lon, double alt) const;
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
