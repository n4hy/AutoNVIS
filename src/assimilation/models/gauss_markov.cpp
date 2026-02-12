/**
 * @file gauss_markov.cpp
 * @brief Implementation of Gauss-Markov perturbation model
 */

#include "physics_model.hpp"
#include <cmath>
#include <random>

namespace autonvis {

GaussMarkovModel::GaussMarkovModel(
    double correlation_time,
    double process_noise_std
)
    : correlation_time_(correlation_time)
    , process_noise_std_(process_noise_std)
{
}

double GaussMarkovModel::correlation_coefficient(double dt) const {
    return std::exp(-dt / correlation_time_);
}

void GaussMarkovModel::propagate(
    const StateVector& state_in,
    double dt,
    StateVector& state_out
) const {
    // Gauss-Markov model: x(t+dt) = φ * x(t) + w
    // where φ = exp(-dt/τ) and w ~ N(0, Q)

    const double phi = correlation_coefficient(dt);

    // Convert to vector
    Eigen::VectorXd x_in = state_in.to_vector();

    // Apply correlation
    Eigen::VectorXd x_out = phi * x_in;

    // Add process noise (simplified - would use proper covariance)
    // For now, just maintain the state with correlation
    // Process noise is added separately in SR-UKF

    // Convert back to state
    state_out.from_vector(x_out);
}

} // namespace autonvis
