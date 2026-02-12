/**
 * @file state_vector.cpp
 * @brief Implementation of state vector representation
 */

#include "state_vector.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace autonvis {

StateVector::StateVector(size_t n_lat, size_t n_lon, size_t n_alt)
    : n_lat_(n_lat)
    , n_lon_(n_lon)
    , n_alt_(n_alt)
    , ne_grid_(n_lat * n_lon * n_alt, 0.0)
    , reff_(100.0)  // Default R_eff
{
    if (n_lat == 0 || n_lon == 0 || n_alt == 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }
}

size_t StateVector::dimension() const {
    return n_grid() + 1;  // Grid points + R_eff
}

Eigen::VectorXd StateVector::to_vector() const {
    Eigen::VectorXd vec(dimension());

    // Copy grid values
    for (size_t i = 0; i < n_grid(); ++i) {
        vec(i) = ne_grid_[i];
    }

    // Add R_eff
    vec(n_grid()) = reff_;

    return vec;
}

void StateVector::from_vector(const Eigen::VectorXd& vec) {
    if (vec.size() != static_cast<Eigen::Index>(dimension())) {
        throw std::invalid_argument("Vector size mismatch");
    }

    // Extract grid values
    for (size_t i = 0; i < n_grid(); ++i) {
        ne_grid_[i] = vec(i);
    }

    // Extract R_eff
    reff_ = vec(n_grid());

    // Apply constraints
    apply_constraints();
}

double StateVector::get_ne(size_t i_lat, size_t i_lon, size_t i_alt) const {
    if (i_lat >= n_lat_ || i_lon >= n_lon_ || i_alt >= n_alt_) {
        throw std::out_of_range("Grid index out of range");
    }

    return ne_grid_[grid_index(i_lat, i_lon, i_alt)];
}

void StateVector::set_ne(size_t i_lat, size_t i_lon, size_t i_alt, double value) {
    if (i_lat >= n_lat_ || i_lon >= n_lon_ || i_alt >= n_alt_) {
        throw std::out_of_range("Grid index out of range");
    }

    ne_grid_[grid_index(i_lat, i_lon, i_alt)] = value;
}

void StateVector::initialize_from_background(
    const std::vector<std::vector<std::vector<double>>>& background_ne,
    double reff_initial
) {
    // Verify dimensions
    if (background_ne.size() != n_lat_ ||
        background_ne[0].size() != n_lon_ ||
        background_ne[0][0].size() != n_alt_) {
        throw std::invalid_argument("Background grid size mismatch");
    }

    // Copy values
    for (size_t i_lat = 0; i_lat < n_lat_; ++i_lat) {
        for (size_t i_lon = 0; i_lon < n_lon_; ++i_lon) {
            for (size_t i_alt = 0; i_alt < n_alt_; ++i_alt) {
                set_ne(i_lat, i_lon, i_alt, background_ne[i_lat][i_lon][i_alt]);
            }
        }
    }

    reff_ = reff_initial;

    apply_constraints();
}

void StateVector::apply_constraints() {
    // Enforce physical bounds on Ne
    for (auto& ne : ne_grid_) {
        ne = std::clamp(ne, MIN_NE, MAX_NE);
    }

    // Enforce bounds on R_eff
    reff_ = std::clamp(reff_, MIN_REFF, MAX_REFF);
}

void StateVector::save_to_hdf5(const std::string& filename) const {
    // TODO: Implement HDF5 save
    // This requires linking with HDF5 library
    (void)filename;  // Suppress unused parameter warning
    throw std::runtime_error("HDF5 save not yet implemented");
}

void StateVector::load_from_hdf5(const std::string& filename) {
    // TODO: Implement HDF5 load
    (void)filename;
    throw std::runtime_error("HDF5 load not yet implemented");
}

} // namespace autonvis
