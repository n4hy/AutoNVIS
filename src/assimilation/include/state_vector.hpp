/**
 * @file state_vector.hpp
 * @brief State vector representation for ionospheric electron density
 *
 * Represents the system state as a 4D grid of electron density values
 * plus additional parameters (effective sunspot number, etc.)
 */

#ifndef STATE_VECTOR_HPP
#define STATE_VECTOR_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace autonvis {

/**
 * @brief State vector for SR-UKF
 *
 * The state vector contains:
 * - Electron density at grid points (n_lat × n_lon × n_alt)
 * - Effective sunspot number (R_eff)
 * - Optional: neutral wind components
 */
class StateVector {
public:
    /**
     * @brief Constructor with grid dimensions
     * @param n_lat Number of latitude points
     * @param n_lon Number of longitude points
     * @param n_alt Number of altitude points
     */
    StateVector(size_t n_lat, size_t n_lon, size_t n_alt);

    /**
     * @brief Get total dimension of state vector
     * @return Total number of state variables
     */
    size_t dimension() const;

    /**
     * @brief Get grid dimensions
     */
    size_t n_lat() const { return n_lat_; }
    size_t n_lon() const { return n_lon_; }
    size_t n_alt() const { return n_alt_; }
    size_t n_grid() const { return n_lat_ * n_lon_ * n_alt_; }

    /**
     * @brief Convert to Eigen vector (flatten 3D grid)
     * @return Eigen vector representation
     */
    Eigen::VectorXd to_vector() const;

    /**
     * @brief Initialize from Eigen vector
     * @param vec Eigen vector (must match dimension)
     */
    void from_vector(const Eigen::VectorXd& vec);

    /**
     * @brief Get electron density at grid point
     * @param i_lat Latitude index
     * @param i_lon Longitude index
     * @param i_alt Altitude index
     * @return Electron density (el/m³)
     */
    double get_ne(size_t i_lat, size_t i_lon, size_t i_alt) const;

    /**
     * @brief Set electron density at grid point
     * @param i_lat Latitude index
     * @param i_lon Longitude index
     * @param i_alt Altitude index
     * @param value Electron density (el/m³)
     */
    void set_ne(size_t i_lat, size_t i_lon, size_t i_alt, double value);

    /**
     * @brief Get effective sunspot number
     */
    double get_reff() const { return reff_; }

    /**
     * @brief Set effective sunspot number
     */
    void set_reff(double value) { reff_ = value; }

    /**
     * @brief Initialize with background model values
     * @param background_ne 3D array of background electron density
     * @param reff_initial Initial effective sunspot number
     */
    void initialize_from_background(
        const std::vector<std::vector<std::vector<double>>>& background_ne,
        double reff_initial
    );

    /**
     * @brief Apply physical constraints (enforce min/max Ne)
     */
    void apply_constraints();

    /**
     * @brief Save state to HDF5 file
     * @param filename Output file path
     */
    void save_to_hdf5(const std::string& filename) const;

    /**
     * @brief Load state from HDF5 file
     * @param filename Input file path
     */
    void load_from_hdf5(const std::string& filename);

private:
    size_t n_lat_;  ///< Number of latitude points
    size_t n_lon_;  ///< Number of longitude points
    size_t n_alt_;  ///< Number of altitude points

    /// Electron density grid (flattened: [lat][lon][alt])
    std::vector<double> ne_grid_;

    /// Effective sunspot number
    double reff_;

    /// Convert 3D indices to 1D index
    size_t grid_index(size_t i_lat, size_t i_lon, size_t i_alt) const {
        return i_alt + n_alt_ * (i_lon + n_lon_ * i_lat);
    }

    // Physical constraints
    static constexpr double MIN_NE = 1e8;   // Minimum Ne (el/m³)
    static constexpr double MAX_NE = 1e13;  // Maximum Ne (el/m³)
    static constexpr double MIN_REFF = 0.0;
    static constexpr double MAX_REFF = 300.0;
};

} // namespace autonvis

#endif // STATE_VECTOR_HPP
