/**
 * @file sigma_points.cpp
 * @brief Implementation of sigma point generation
 */

#include "sigma_points.hpp"
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
    Eigen::MatrixXd deviations(dim, n_points);

    for (size_t i = 0; i < n_points; ++i) {
        const double weight = std::sqrt(std::abs(weights(i)));
        const Eigen::VectorXd dev = sigma_points[i] - mean;
        deviations.col(i) = (weights(i) >= 0 ? weight : -weight) * dev;
    }

    // QR decomposition to get sqrt covariance
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(deviations.transpose());
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // Return lower triangular (transpose of R)
    return R.transpose().block(0, 0, dim, dim);
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
