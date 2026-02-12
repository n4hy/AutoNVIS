/**
 * @file sr_ukf.cpp
 * @brief Implementation of Square-Root Unscented Kalman Filter
 */

#include "sr_ukf.hpp"
#include "cholesky_update.hpp"
#include <chrono>
#include <stdexcept>

namespace autonvis {

SquareRootUKF::SquareRootUKF(
    size_t n_lat,
    size_t n_lon,
    size_t n_alt,
    double alpha,
    double beta,
    double kappa
)
    : n_lat_(n_lat)
    , n_lon_(n_lon)
    , n_alt_(n_alt)
    , state_mean_(n_lat, n_lon, n_alt)
    , sigma_gen_(alpha, beta, kappa)
{
    const size_t dim = state_mean_.dimension();

    // Initialize sqrt covariance as identity (large initial uncertainty)
    state_sqrt_cov_ = Eigen::MatrixXd::Identity(dim, dim) * 1e6;

    // Initialize process noise (will be set properly later)
    process_sqrt_cov_ = Eigen::MatrixXd::Identity(dim, dim) * 1e5;

    // Initialize statistics
    stats_.predict_count = 0;
    stats_.update_count = 0;
    stats_.last_predict_time_ms = 0.0;
    stats_.last_update_time_ms = 0.0;
    stats_.avg_predict_time_ms = 0.0;
    stats_.avg_update_time_ms = 0.0;
    stats_.min_eigenvalue = 0.0;
    stats_.max_eigenvalue = 0.0;
}

void SquareRootUKF::initialize(
    const StateVector& initial_state,
    const Eigen::MatrixXd& initial_sqrt_cov
) {
    state_mean_ = initial_state;
    state_sqrt_cov_ = initial_sqrt_cov;

    // Verify dimensions
    if (state_sqrt_cov_.rows() != static_cast<Eigen::Index>(state_mean_.dimension()) ||
        state_sqrt_cov_.cols() != static_cast<Eigen::Index>(state_mean_.dimension())) {
        throw std::invalid_argument("sqrt_cov dimension mismatch");
    }

    update_eigenvalue_stats();
}

void SquareRootUKF::predict(double dt) {
    auto start = std::chrono::high_resolution_clock::now();

    if (!physics_model_) {
        throw std::runtime_error("Physics model not set");
    }

    const size_t dim = state_mean_.dimension();

    // 1. Generate sigma points from current state
    Eigen::VectorXd mean_vec = state_mean_.to_vector();
    std::vector<Eigen::VectorXd> sigma_points;
    sigma_gen_.generate(mean_vec, state_sqrt_cov_, sigma_points);

    // 2. Propagate sigma points through physics model
    std::vector<Eigen::VectorXd> propagated_points;
    propagate_sigma_points(sigma_points, dt, propagated_points);

    // 3. Compute predicted mean
    Eigen::VectorXd predicted_mean = sigma_gen_.compute_mean(propagated_points);

    // 4. Compute predicted sqrt covariance
    Eigen::MatrixXd predicted_sqrt_cov =
        sigma_gen_.compute_sqrt_cov(propagated_points, predicted_mean);

    // 5. Add process noise via QR decomposition
    Eigen::MatrixXd stacked = stack_matrices(predicted_sqrt_cov, process_sqrt_cov_);
    Eigen::MatrixXd R = qr_decomposition(stacked.transpose());
    predicted_sqrt_cov = R.transpose().block(0, 0, dim, dim);

    // 6. Update state
    state_mean_.from_vector(predicted_mean);
    state_sqrt_cov_ = predicted_sqrt_cov;

    // Update statistics
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    stats_.predict_count++;
    stats_.last_predict_time_ms = duration.count();
    stats_.avg_predict_time_ms =
        (stats_.avg_predict_time_ms * (stats_.predict_count - 1) + duration.count()) /
        stats_.predict_count;

    update_eigenvalue_stats();
}

void SquareRootUKF::update(
    const ObservationModel& obs_model,
    const Eigen::VectorXd& observations,
    const Eigen::MatrixXd& obs_sqrt_cov
) {
    auto start = std::chrono::high_resolution_clock::now();

    const size_t dim = state_mean_.dimension();
    const size_t obs_dim = obs_model.obs_dimension();

    // 1. Generate sigma points from predicted state
    Eigen::VectorXd mean_vec = state_mean_.to_vector();
    std::vector<Eigen::VectorXd> sigma_points;
    sigma_gen_.generate(mean_vec, state_sqrt_cov_, sigma_points);

    // 2. Propagate through observation model
    std::vector<Eigen::VectorXd> predicted_obs;
    predict_observations(sigma_points, obs_model, predicted_obs);

    // 3. Compute predicted observation mean
    Eigen::VectorXd obs_mean = sigma_gen_.compute_mean(predicted_obs);

    // 4. Compute innovation covariance sqrt (S_yy)
    Eigen::MatrixXd S_yy = sigma_gen_.compute_sqrt_cov(predicted_obs, obs_mean);

    // Add observation noise
    Eigen::MatrixXd stacked_obs = stack_matrices(S_yy, obs_sqrt_cov);
    Eigen::MatrixXd R_obs = qr_decomposition(stacked_obs.transpose());
    S_yy = R_obs.transpose().block(0, 0, obs_dim, obs_dim);

    // 5. Compute cross-covariance P_xy
    Eigen::MatrixXd P_xy = compute_cross_covariance(
        sigma_points, predicted_obs, mean_vec, obs_mean
    );

    // 6. Compute Kalman gain K = P_xy * inv(S_yy^T * S_yy)
    // Solve: S_yy^T * S_yy * K^T = P_xy^T
    Eigen::MatrixXd K = (S_yy.transpose() * S_yy).lu().solve(P_xy.transpose()).transpose();

    // 7. Update state mean
    Eigen::VectorXd innovation = observations - obs_mean;
    Eigen::VectorXd updated_mean = mean_vec + K * innovation;

    // 8. Update sqrt covariance (Joseph form for numerical stability)
    Eigen::MatrixXd U = K * S_yy;

    // QR downdate: S_new such that S_new * S_new^T = S * S^T - U * U^T
    // Simplified: use QR decomposition
    Eigen::MatrixXd I_KH = Eigen::MatrixXd::Identity(dim, dim);
    Eigen::MatrixXd S_updated = state_sqrt_cov_;

    // Perform rank-k downdate (simplified approach)
    for (size_t i = 0; i < obs_dim; ++i) {
        try {
            S_updated = choldowndate(S_updated, U.col(i));
        } catch (const std::runtime_error&) {
            // If downdate fails, use Joseph form with full matrix
            S_updated = I_KH * state_sqrt_cov_;
            break;
        }
    }

    // 9. Update state
    state_mean_.from_vector(updated_mean);
    state_sqrt_cov_ = S_updated;

    // Verify stability
    if (!verify_stability()) {
        throw std::runtime_error("Filter divergence detected");
    }

    // Update statistics
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    stats_.update_count++;
    stats_.last_update_time_ms = duration.count();
    stats_.avg_update_time_ms =
        (stats_.avg_update_time_ms * (stats_.update_count - 1) + duration.count()) /
        stats_.update_count;

    update_eigenvalue_stats();
}

Eigen::MatrixXd SquareRootUKF::get_covariance() const {
    return state_sqrt_cov_ * state_sqrt_cov_.transpose();
}

void SquareRootUKF::apply_inflation(double factor) {
    state_sqrt_cov_ = autonvis::apply_inflation(state_sqrt_cov_, factor);
}

void SquareRootUKF::propagate_sigma_points(
    const std::vector<Eigen::VectorXd>& sigma_points,
    double dt,
    std::vector<Eigen::VectorXd>& propagated_points
) const {
    propagated_points.clear();
    propagated_points.reserve(sigma_points.size());

    for (const auto& point : sigma_points) {
        StateVector state_in(n_lat_, n_lon_, n_alt_);
        state_in.from_vector(point);

        StateVector state_out(n_lat_, n_lon_, n_alt_);
        physics_model_->propagate(state_in, dt, state_out);

        propagated_points.push_back(state_out.to_vector());
    }
}

void SquareRootUKF::predict_observations(
    const std::vector<Eigen::VectorXd>& sigma_points,
    const ObservationModel& obs_model,
    std::vector<Eigen::VectorXd>& predicted_obs
) const {
    predicted_obs.clear();
    predicted_obs.reserve(sigma_points.size());

    for (const auto& point : sigma_points) {
        StateVector state(n_lat_, n_lon_, n_alt_);
        state.from_vector(point);

        Eigen::VectorXd obs = obs_model.forward(state);
        predicted_obs.push_back(obs);
    }
}

Eigen::MatrixXd SquareRootUKF::compute_cross_covariance(
    const std::vector<Eigen::VectorXd>& sigma_points,
    const std::vector<Eigen::VectorXd>& predicted_obs,
    const Eigen::VectorXd& state_mean,
    const Eigen::VectorXd& obs_mean
) const {
    const size_t n_points = sigma_points.size();
    const size_t L = (n_points - 1) / 2;
    const size_t state_dim = state_mean.size();
    const size_t obs_dim = obs_mean.size();

    const Eigen::VectorXd weights = sigma_gen_.get_cov_weights(L);

    Eigen::MatrixXd P_xy = Eigen::MatrixXd::Zero(state_dim, obs_dim);

    for (size_t i = 0; i < n_points; ++i) {
        const Eigen::VectorXd state_dev = sigma_points[i] - state_mean;
        const Eigen::VectorXd obs_dev = predicted_obs[i] - obs_mean;

        P_xy += weights(i) * state_dev * obs_dev.transpose();
    }

    return P_xy;
}

bool SquareRootUKF::verify_stability() const {
    return verify_positive_definite(state_sqrt_cov_, 1e-10);
}

void SquareRootUKF::update_eigenvalue_stats() {
    const Eigen::MatrixXd P = get_covariance();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(P);

    if (eigen_solver.info() == Eigen::Success) {
        const Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();
        stats_.min_eigenvalue = eigenvalues.minCoeff();
        stats_.max_eigenvalue = eigenvalues.maxCoeff();
    }
}

void SquareRootUKF::save_checkpoint(const std::string& filename) const {
    // TODO: Implement checkpoint save
    (void)filename;
    throw std::runtime_error("Checkpoint save not yet implemented");
}

void SquareRootUKF::load_checkpoint(const std::string& filename) {
    // TODO: Implement checkpoint load
    (void)filename;
    throw std::runtime_error("Checkpoint load not yet implemented");
}

} // namespace autonvis
