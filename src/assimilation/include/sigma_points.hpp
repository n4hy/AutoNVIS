/**
 * @file sigma_points.hpp
 * @brief Sigma point generation for Unscented Kalman Filter
 *
 * Implements the scaled unscented transform for generating sigma points
 * and recovering mean/covariance from propagated points.
 */

#ifndef SIGMA_POINTS_HPP
#define SIGMA_POINTS_HPP

#include <Eigen/Dense>
#include <vector>
#include "state_vector.hpp"

namespace autonvis {

/**
 * @brief Sigma point generator for UKF
 *
 * Generates 2L+1 sigma points from mean and square-root covariance,
 * where L is the state dimension.
 */
class SigmaPointGenerator {
public:
    /**
     * @brief Constructor with UKF parameters
     * @param alpha Spread parameter (1e-4 to 1)
     * @param beta Prior knowledge parameter (2 for Gaussian)
     * @param kappa Secondary scaling (usually 0 or 3-L)
     */
    SigmaPointGenerator(
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    );

    /**
     * @brief Generate sigma points
     * @param mean State mean vector
     * @param sqrt_cov Square-root covariance (Cholesky factor S)
     * @param sigma_points Output vector of sigma points
     */
    void generate(
        const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& sqrt_cov,
        std::vector<Eigen::VectorXd>& sigma_points
    ) const;

    /**
     * @brief Compute mean from sigma points
     * @param sigma_points Vector of sigma points
     * @return Mean vector
     */
    Eigen::VectorXd compute_mean(
        const std::vector<Eigen::VectorXd>& sigma_points
    ) const;

    /**
     * @brief Compute square-root covariance from sigma points
     * @param sigma_points Vector of sigma points
     * @param mean Mean vector (precomputed)
     * @return Square-root covariance matrix (Cholesky factor)
     */
    Eigen::MatrixXd compute_sqrt_cov(
        const std::vector<Eigen::VectorXd>& sigma_points,
        const Eigen::VectorXd& mean
    ) const;

    /**
     * @brief Get weights for mean calculation
     * @param L State dimension
     * @return Vector of weights (length 2L+1)
     */
    Eigen::VectorXd get_mean_weights(size_t L) const;

    /**
     * @brief Get weights for covariance calculation
     * @param L State dimension
     * @return Vector of weights (length 2L+1)
     */
    Eigen::VectorXd get_cov_weights(size_t L) const;

    /**
     * @brief Get number of sigma points for given dimension
     * @param L State dimension
     * @return 2L + 1
     */
    size_t num_sigma_points(size_t L) const { return 2 * L + 1; }

private:
    double alpha_;  ///< Spread parameter
    double beta_;   ///< Prior knowledge parameter
    double kappa_;  ///< Secondary scaling parameter

    /**
     * @brief Compute lambda parameter
     * @param L State dimension
     * @return lambda = alpha^2 * (L + kappa) - L
     */
    double compute_lambda(size_t L) const {
        return alpha_ * alpha_ * (L + kappa_) - L;
    }
};

} // namespace autonvis

#endif // SIGMA_POINTS_HPP
