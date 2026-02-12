/**
 * @file test_cholesky.cpp
 * @brief Unit tests for Cholesky updates
 */

#include "cholesky_update.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace autonvis;

void test_cholupdate() {
    std::cout << "Test: Cholesky update... ";

    const size_t n = 5;

    // Create initial matrix S
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n) * 2.0;

    // Update vector
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);

    // Perform update
    Eigen::MatrixXd S_new = cholupdate(S, v);

    // Verify: S_new * S_new^T = S * S^T + v * v^T
    Eigen::MatrixXd P_old = S * S.transpose();
    Eigen::MatrixXd P_new = S_new * S_new.transpose();
    Eigen::MatrixXd P_expected = P_old + v * v.transpose();

    const double error = (P_new - P_expected).norm();
    assert(error < 1e-10);

    std::cout << "PASSED (error: " << error << ")\n";
}

void test_positive_definite_check() {
    std::cout << "Test: Positive definite verification... ";

    const size_t n = 4;

    // Create positive definite S
    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);

    assert(verify_positive_definite(S, 1e-10) == true);

    // Create singular S
    Eigen::MatrixXd S_singular = Eigen::MatrixXd::Zero(n, n);

    assert(verify_positive_definite(S_singular, 1e-10) == false);

    std::cout << "PASSED\n";
}

void test_inflation() {
    std::cout << "Test: Covariance inflation... ";

    const size_t n = 3;

    Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd P_old = S * S.transpose();

    const double inflation_factor = 1.1;
    Eigen::MatrixXd S_inflated = apply_inflation(S, inflation_factor);
    Eigen::MatrixXd P_inflated = S_inflated * S_inflated.transpose();

    // P_inflated should be inflation_factor * P_old
    Eigen::MatrixXd P_expected = inflation_factor * P_old;

    const double error = (P_inflated - P_expected).norm();
    assert(error < 1e-10);

    std::cout << "PASSED (error: " << error << ")\n";
}

int main() {
    std::cout << "=== Cholesky Update Tests ===\n";

    try {
        test_cholupdate();
        test_positive_definite_check();
        test_inflation();

        std::cout << "\nAll tests PASSED\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nTest FAILED: " << e.what() << "\n";
        return 1;
    }
}
