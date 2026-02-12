/**
 * @file test_sigma_points.cpp
 * @brief Unit tests for sigma point generation
 */

#include "sigma_points.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace autonvis;

void test_sigma_point_count() {
    std::cout << "Test: Sigma point count... ";

    const size_t L = 10;  // State dimension
    SigmaPointGenerator gen;

    Eigen::VectorXd mean = Eigen::VectorXd::Random(L);
    Eigen::MatrixXd sqrt_cov = Eigen::MatrixXd::Identity(L, L);

    std::vector<Eigen::VectorXd> sigma_points;
    gen.generate(mean, sqrt_cov, sigma_points);

    // Should have 2L + 1 points
    assert(sigma_points.size() == 2 * L + 1);

    std::cout << "PASSED\n";
}

void test_mean_recovery() {
    std::cout << "Test: Mean recovery... ";

    const size_t L = 5;
    SigmaPointGenerator gen;

    Eigen::VectorXd mean = Eigen::VectorXd::Random(L);
    Eigen::MatrixXd sqrt_cov = Eigen::MatrixXd::Identity(L, L) * 0.1;

    std::vector<Eigen::VectorXd> sigma_points;
    gen.generate(mean, sqrt_cov, sigma_points);

    Eigen::VectorXd recovered_mean = gen.compute_mean(sigma_points);

    // Mean should be recovered accurately
    const double error = (recovered_mean - mean).norm();
    assert(error < 1e-10);

    std::cout << "PASSED (error: " << error << ")\n";
}

void test_weights_sum() {
    std::cout << "Test: Weights sum to one... ";

    const size_t L = 7;
    SigmaPointGenerator gen;

    Eigen::VectorXd mean_weights = gen.get_mean_weights(L);
    Eigen::VectorXd cov_weights = gen.get_cov_weights(L);

    const double mean_sum = mean_weights.sum();
    const double cov_sum = cov_weights.sum();

    // Mean weights should sum to 1
    assert(std::abs(mean_sum - 1.0) < 1e-10);

    std::cout << "PASSED (mean_sum: " << mean_sum
              << ", cov_sum: " << cov_sum << ")\n";
}

int main() {
    std::cout << "=== Sigma Point Tests ===\n";

    try {
        test_sigma_point_count();
        test_mean_recovery();
        test_weights_sum();

        std::cout << "\nAll tests PASSED\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
