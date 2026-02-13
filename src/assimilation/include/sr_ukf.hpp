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
#include <Eigen/Sparse>
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

        // Adaptive inflation statistics
        double last_nis;              ///< Normalized Innovation Squared
        double avg_nis;               ///< Average NIS over updates
        double inflation_factor;      ///< Current inflation factor
        size_t divergence_count;      ///< Number of divergence detections
    };

    Statistics get_statistics() const { return stats_; }

    /**
     * @brief Configuration for adaptive inflation
     */
    struct AdaptiveInflationConfig {
        bool enabled = true;           ///< Enable adaptive inflation
        double initial_inflation = 1.0;///< Initial inflation factor
        double min_inflation = 1.0;    ///< Minimum inflation factor
        double max_inflation = 2.0;    ///< Maximum inflation factor
        double adaptation_rate = 0.95; ///< Exponential smoothing factor
        double divergence_threshold = 3.0; ///< NIS threshold for divergence (in multiples of expected)
    };

    /**
     * @brief Set adaptive inflation configuration
     */
    void set_adaptive_inflation_config(const AdaptiveInflationConfig& config) {
        inflation_config_ = config;
    }

    /**
     * @brief Get current adaptive inflation configuration
     */
    const AdaptiveInflationConfig& get_inflation_config() const {
        return inflation_config_;
    }

    /**
     * @brief Configuration for covariance localization
     */
    struct LocalizationConfig {
        bool enabled = false;          ///< Enable covariance localization
        double radius_km = 500.0;      ///< Localization radius (km)
        bool precompute = true;        ///< Precompute localization matrix (faster but more memory)
    };

    /**
     * @brief Set covariance localization configuration
     *
     * Computes and stores the sparse localization matrix if precompute=true.
     * Must be called after initialization to set grid parameters.
     *
     * @param config Localization configuration
     * @param lat_grid Latitude grid points (degrees)
     * @param lon_grid Longitude grid points (degrees)
     * @param alt_grid Altitude grid points (km)
     */
    void set_localization_config(
        const LocalizationConfig& config,
        const std::vector<double>& lat_grid,
        const std::vector<double>& lon_grid,
        const std::vector<double>& alt_grid
    );

    /**
     * @brief Get current localization configuration
     */
    const LocalizationConfig& get_localization_config() const {
        return localization_config_;
    }

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

    // Adaptive inflation configuration
    AdaptiveInflationConfig inflation_config_;

    // Covariance localization
    LocalizationConfig localization_config_;
    Eigen::SparseMatrix<double> localization_matrix_;  ///< Precomputed localization (optional)

    /**
     * @brief Compute Normalized Innovation Squared (NIS) metric
     * @param innovation Innovation vector (y - h(x))
     * @param S_yy Innovation sqrt covariance
     * @return NIS value (follows χ² distribution with obs_dim DoF)
     */
    double compute_nis(
        const Eigen::VectorXd& innovation,
        const Eigen::MatrixXd& S_yy
    ) const;

    /**
     * @brief Compute and apply adaptive inflation
     * Called after predict step, uses innovation statistics from previous update
     */
    void apply_adaptive_inflation();

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
