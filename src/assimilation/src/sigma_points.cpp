/**
 * @file sigma_points.cpp
 * @brief Implementation of sigma point generation
 */

#include "sigma_points.hpp"
#include "cholesky_update.hpp"
#include <cmath>
#include <stdexcept>

namespace autonvis {

SigmaPointGenerator::SigmaPointGenerator(
    double alpha,
    double beta,
    double kappa
)
    : alpha_(alpha)
    , beta_(beta)
    , kappa_(kappa)
{
    if (alpha <= 0 || alpha > 1) {
        throw std::invalid_argument("alpha must be in (0, 1]");
    }
}

void SigmaPointGenerator::generate(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& sqrt_cov,
    std::vector<Eigen::VectorXd>& sigma_points
) const {
    const size_t L = mean.size();
    const size_t n_points = 2 * L + 1;

    // Verify sqrt_cov is square and matches mean dimension
    if (sqrt_cov.rows() != static_cast<Eigen::Index>(L) ||
        sqrt_cov.cols() != static_cast<Eigen::Index>(L)) {
        throw std::invalid_argument("sqrt_cov dimension mismatch");
    }

    sigma_points.clear();
    sigma_points.reserve(n_points);

    // Compute scaling
    const double lambda = compute_lambda(L);
    const double gamma = std::sqrt(L + lambda);

    // Point 0: mean
    sigma_points.push_back(mean);

    // Points 1 to L: mean + gamma * sqrt_cov columns
    for (size_t i = 0; i < L; ++i) {
        sigma_points.push_back(mean + gamma * sqrt_cov.col(i));
    }

    // Points L+1 to 2L: mean - gamma * sqrt_cov columns
    for (size_t i = 0; i < L; ++i) {
        sigma_points.push_back(mean - gamma * sqrt_cov.col(i));
    }
}

Eigen::VectorXd SigmaPointGenerator::compute_mean(
    const std::vector<Eigen::VectorXd>& sigma_points
) const {
    if (sigma_points.empty()) {
        throw std::invalid_argument("sigma_points is empty");
    }

    const size_t n_points = sigma_points.size();
    const size_t L = (n_points - 1) / 2;

    const Eigen::VectorXd weights = get_mean_weights(L);

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(sigma_points[0].size());

    for (size_t i = 0; i < n_points; ++i) {
        mean += weights(i) * sigma_points[i];
    }

    return mean;
}

Eigen::MatrixXd SigmaPointGenerator::compute_sqrt_cov(
    const std::vector<Eigen::VectorXd>& sigma_points,
    const Eigen::VectorXd& mean
) const {
    if (sigma_points.empty()) {
        throw std::invalid_argument("sigma_points is empty");
    }

    const size_t n_points = sigma_points.size();
    const size_t L = (n_points - 1) / 2;
    const size_t dim = sigma_points[0].size();

    const Eigen::VectorXd weights = get_cov_weights(L);

    // Compute weighted deviations matrix
    // Form matrix χ where each column is sqrt(|w_i|) * (sigma_point_i - mean)
    Eigen::MatrixXd chi(dim, n_points - 1);  // Skip first point for now

    for (size_t i = 1; i < n_points; ++i) {
        const double weight = std::sqrt(std::abs(weights(i)));
        const Eigen::VectorXd dev = sigma_points[i] - mean;
        chi.col(i - 1) = (weights(i) >= 0 ? weight : -weight) * dev;
    }

    // QR decomposition to get sqrt covariance
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(chi.transpose());

    // Extract upper triangular R properly (avoid triangularView memory issues)
    Eigen::MatrixXd R = Eigen::MatrixXd(qr.matrixQR().topRows(dim));
    // Zero out below diagonal
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(dim); ++i) {
        for (Eigen::Index j = 0; j < i; ++j) {
            R(i, j) = 0.0;
        }
    }

    // Handle the zeroth sigma point contribution (may have negative weight)
    // Convert R (upper) to S (lower) for cholupdate
    Eigen::MatrixXd S = R.transpose();
    if (weights(0) >= 0) {
        const double weight0 = std::sqrt(weights(0));
        const Eigen::VectorXd dev0 = weight0 * (sigma_points[0] - mean);
        S = cholupdate(S, dev0);
    } else {
        const double weight0 = std::sqrt(-weights(0));
        const Eigen::VectorXd dev0 = weight0 * (sigma_points[0] - mean);
        try {
            S = choldowndate(S, dev0);
        } catch (const std::runtime_error&) {
            // If downdate fails, just use what we have
        }
    }

    return S;
}

Eigen::VectorXd SigmaPointGenerator::get_mean_weights(size_t L) const {
    const double lambda = compute_lambda(L);
    const size_t n_points = 2 * L + 1;

    Eigen::VectorXd weights(n_points);

    // Weight for point 0
    weights(0) = lambda / (L + lambda);

    // Weights for points 1 to 2L
    const double w = 1.0 / (2.0 * (L + lambda));
    for (size_t i = 1; i < n_points; ++i) {
        weights(i) = w;
    }

    return weights;
}

Eigen::VectorXd SigmaPointGenerator::get_cov_weights(size_t L) const {
    const double lambda = compute_lambda(L);
    const size_t n_points = 2 * L + 1;

    Eigen::VectorXd weights(n_points);

    // Weight for point 0
    weights(0) = lambda / (L + lambda) + (1.0 - alpha_ * alpha_ + beta_);

    // Weights for points 1 to 2L
    const double w = 1.0 / (2.0 * (L + lambda));
    for (size_t i = 1; i < n_points; ++i) {
        weights(i) = w;
    }

    return weights;
}

} // namespace autonvis
