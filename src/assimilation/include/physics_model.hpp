/**
 * @file physics_model.hpp
 * @brief Physics-based background models for ionosphere
 *
 * Provides interface to IRI-2020, NeQuick-G, and Gauss-Markov
 * perturbation model for state propagation.
 */

#ifndef PHYSICS_MODEL_HPP
#define PHYSICS_MODEL_HPP

#include <Eigen/Dense>
#include "state_vector.hpp"

namespace autonvis {

/**
 * @brief Abstract base class for physics models
 */
class PhysicsModel {
public:
    virtual ~PhysicsModel() = default;

    /**
     * @brief Propagate state forward in time
     * @param state_in Input state at time t
     * @param dt Time step (seconds)
     * @param state_out Output state at time t + dt
     */
    virtual void propagate(
        const StateVector& state_in,
        double dt,
        StateVector& state_out
    ) const = 0;

    /**
     * @brief Get model name
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Gauss-Markov perturbation model
 *
 * Models small perturbations around climatological background:
 * δNe(t+1) = φ * δNe(t) + w
 *
 * where φ = exp(-dt/τ) is correlation coefficient and w ~ N(0, Q)
 */
class GaussMarkovModel : public PhysicsModel {
public:
    /**
     * @brief Constructor
     * @param correlation_time Correlation time constant (seconds)
     * @param process_noise_std Process noise standard deviation
     */
    GaussMarkovModel(
        double correlation_time = 3600.0,  // 1 hour
        double process_noise_std = 1e10    // el/m³
    );

    void propagate(
        const StateVector& state_in,
        double dt,
        StateVector& state_out
    ) const override;

    std::string name() const override { return "Gauss-Markov"; }

private:
    double correlation_time_;
    double process_noise_std_;

    /**
     * @brief Compute correlation coefficient
     * @param dt Time step
     * @return φ = exp(-dt/τ)
     */
    double correlation_coefficient(double dt) const;
};

/**
 * @brief IRI-2020 physics model wrapper
 *
 * Calls IRI-2020 Fortran subroutines to get background ionosphere.
 * Used in SHOCK mode for physics-based state propagation.
 */
class IRI2020Model : public PhysicsModel {
public:
    /**
     * @brief Constructor
     * @param lat_grid Latitude grid (deg)
     * @param lon_grid Longitude grid (deg)
     * @param alt_grid Altitude grid (km)
     */
    IRI2020Model(
        const std::vector<double>& lat_grid,
        const std::vector<double>& lon_grid,
        const std::vector<double>& alt_grid
    );

    void propagate(
        const StateVector& state_in,
        double dt,
        StateVector& state_out
    ) const override;

    std::string name() const override { return "IRI-2020"; }

    /**
     * @brief Compute background Ne from IRI-2020
     * @param lat Latitude (deg)
     * @param lon Longitude (deg)
     * @param alt Altitude (km)
     * @param year Year
     * @param month Month
     * @param day Day
     * @param hour Hour (UT)
     * @param f107 F10.7 solar flux
     * @return Electron density (el/m³)
     */
    double compute_background_ne(
        double lat,
        double lon,
        double alt,
        int year,
        int month,
        int day,
        double hour,
        double f107
    ) const;

private:
    std::vector<double> lat_grid_;
    std::vector<double> lon_grid_;
    std::vector<double> alt_grid_;

    // IRI-2020 Fortran interface
    // extern "C" void iri_sub_(...);  // Declared in implementation
};

} // namespace autonvis

#endif // PHYSICS_MODEL_HPP
