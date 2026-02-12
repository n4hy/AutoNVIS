/**
 * @file sr_ukf.hpp
 * @brief Square-Root Unscented Kalman Filter
 *
 * Main SR-UKF implementation for ionospheric state estimation.
 * Maintains Cholesky factor of covariance for numerical stability.
 */

#ifndef SR_UKF_HPP
#define SR_UKF_HPP

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <string>

#include "state_vector.hpp"
#include "sigma_points.hpp"
#include "observation_model.hpp"
#include "physics_model.hpp"

namespace autonvis {

/**
 * @brief Square-Root Unscented Kalman Filter
 *
 * Maintains state estimate and square-root covariance (Cholesky factor).
 * Guarantees positive semi-definite covariance through SR formulation.
 */
class SquareRootUKF {
public:
    /**
     * @brief Constructor
     * @param n_lat Number of latitude grid points
     * @param n_lon Number of longitude grid points
     * @param n_alt Number of altitude grid points
     * @param alpha UKF spread parameter
     * @param beta UKF prior knowledge parameter
     * @param kappa UKF secondary scaling
     */
    SquareRootUKF(
        size_t n_lat,
        size_t n_lon,
        size_t n_alt,
        double alpha = 1e-3,
        double beta = 2.0,
        double kappa = 0.0
    );

    /**
     * @brief Initialize filter with background state
     * @param initial_state Initial state vector
     * @param initial_sqrt_cov Initial sqrt covariance (Cholesky factor)
     */
    void initialize(
        const StateVector& initial_state,
        const Eigen::MatrixXd& initial_sqrt_cov
    );

    /**
     * @brief Set physics model for state propagation
     * @param model Pointer to physics model
     */
    void set_physics_model(std::shared_ptr<PhysicsModel> model) {
        physics_model_ = model;
    }

    /**
     * @brief Predict step: propagate state and covariance forward
     * @param dt Time step (seconds)
     */
    void predict(double dt);

    /**
     * @brief Update step: assimilate observations
     * @param obs_model Observation model
     * @param observations Observation vector
     * @param obs_sqrt_cov Observation error sqrt covariance
     */
    void update(
        const ObservationModel& obs_model,
        const Eigen::VectorXd& observations,
        const Eigen::MatrixXd& obs_sqrt_cov
    );

    /**
     * @brief Get current state mean
     */
    const StateVector& get_state() const { return state_mean_; }

    /**
     * @brief Get current sqrt covariance
     */
    const Eigen::MatrixXd& get_sqrt_cov() const { return state_sqrt_cov_; }

    /**
     * @brief Get full covariance (P = S * S^T)
     */
    Eigen::MatrixXd get_covariance() const;

    /**
     * @brief Save checkpoint to file
     * @param filename Output file path
     */
    void save_checkpoint(const std::string& filename) const;

    /**
     * @brief Load checkpoint from file
     * @param filename Input file path
     */
    void load_checkpoint(const std::string& filename);

    /**
     * @brief Get filter statistics
     */
    struct Statistics {
        size_t predict_count;
        size_t update_count;
        double last_predict_time_ms;
        double last_update_time_ms;
        double avg_predict_time_ms;
        double avg_update_time_ms;
        double min_eigenvalue;
        double max_eigenvalue;
    };

    Statistics get_statistics() const { return stats_; }

    /**
     * @brief Set process noise sqrt covariance
     */
    void set_process_noise(const Eigen::MatrixXd& process_sqrt_cov) {
        process_sqrt_cov_ = process_sqrt_cov;
    }

    /**
     * @brief Apply covariance inflation
     * @param factor Inflation factor (e.g., 1.01)
     */
    void apply_inflation(double factor);

private:
    // Grid dimensions
    size_t n_lat_, n_lon_, n_alt_;

    // State estimate
    StateVector state_mean_;
    Eigen::MatrixXd state_sqrt_cov_;  ///< Cholesky factor S (P = S * S^T)

    // Process noise
    Eigen::MatrixXd process_sqrt_cov_;

    // Sigma point generator
    SigmaPointGenerator sigma_gen_;

    // Physics model
    std::shared_ptr<PhysicsModel> physics_model_;

    // Statistics
    Statistics stats_;

    /**
     * @brief Propagate sigma points through physics model
     * @param sigma_points Input sigma points
     * @param dt Time step
     * @param propagated_points Output propagated points
     */
    void propagate_sigma_points(
        const std::vector<Eigen::VectorXd>& sigma_points,
        double dt,
        std::vector<Eigen::VectorXd>& propagated_points
    ) const;

    /**
     * @brief Propagate sigma points through observation model
     * @param sigma_points Input sigma points
     * @param obs_model Observation model
     * @param predicted_obs Output predicted observations
     */
    void predict_observations(
        const std::vector<Eigen::VectorXd>& sigma_points,
        const ObservationModel& obs_model,
        std::vector<Eigen::VectorXd>& predicted_obs
    ) const;

    /**
     * @brief Compute cross-covariance between state and observations
     * @param sigma_points State sigma points
     * @param predicted_obs Predicted observation sigma points
     * @param state_mean State mean
     * @param obs_mean Observation mean
     * @return Cross-covariance matrix P_xy
     */
    Eigen::MatrixXd compute_cross_covariance(
        const std::vector<Eigen::VectorXd>& sigma_points,
        const std::vector<Eigen::VectorXd>& predicted_obs,
        const Eigen::VectorXd& state_mean,
        const Eigen::VectorXd& obs_mean
    ) const;

    /**
     * @brief Verify numerical stability
     * @return True if filter is stable
     */
    bool verify_stability() const;

    /**
     * @brief Update statistics
     */
    void update_eigenvalue_stats();
};

} // namespace autonvis

#endif // SR_UKF_HPP
