/**
 * @file cholesky_update.hpp
 * @brief Numerically stable Cholesky factor updates
 *
 * Provides rank-1 updates and downdates for Cholesky factors,
 * ensuring positive semi-definiteness throughout SR-UKF.
 */

#ifndef CHOLESKY_UPDATE_HPP
#define CHOLESKY_UPDATE_HPP

#include <Eigen/Dense>

namespace autonvis {

/**
 * @brief Rank-1 Cholesky update (cholupdate)
 *
 * Updates Cholesky factor S such that S_new * S_new^T = S * S^T + v * v^T
 *
 * @param S Input Cholesky factor (lower triangular L, such that P = L * L^T)
 * @param v Update vector
 * @return Updated Cholesky factor (lower triangular)
 */
inline Eigen::MatrixXd cholupdate(const Eigen::MatrixXd& S, const Eigen::VectorXd& v) {
    const Eigen::Index n = S.rows();

    // Compute P_new = S * S^T + v*v^T
    Eigen::MatrixXd P = S * S.transpose();
    P += v * v.transpose();

    // Cholesky decomposition of P_new
    Eigen::LLT<Eigen::MatrixXd> llt(P);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("cholupdate: Cholesky decomposition failed");
    }

    // Extract and return lower triangular L
    Eigen::MatrixXd L = Eigen::MatrixXd(llt.matrixL());
    return L;
}

/**
 * @brief Rank-1 Cholesky downdate (choldowndate)
 *
 * Updates Cholesky factor S such that S_new * S_new^T = S * S^T - v * v^T
 *
 * @param S Input Cholesky factor (lower triangular)
 * @param v Downdate vector
 * @return Updated Cholesky factor
 */
Eigen::MatrixXd choldowndate(const Eigen::MatrixXd& S, const Eigen::VectorXd& v);

/**
 * @brief QR decomposition for matrix
 *
 * Computes QR decomposition A = Q * R
 *
 * @param A Input matrix
 * @return R matrix (upper triangular)
 */
Eigen::MatrixXd qr_decomposition(const Eigen::MatrixXd& A);

/**
 * @brief Combine matrices for QR-based covariance update
 *
 * Stacks matrices vertically for QR decomposition in SR-UKF update
 *
 * @param S_x State sqrt covariance
 * @param S_v Process noise sqrt covariance
 * @return Stacked matrix [S_x; S_v]
 */
Eigen::MatrixXd stack_matrices(
    const Eigen::MatrixXd& S_x,
    const Eigen::MatrixXd& S_v
);

/**
 * @brief Verify positive definiteness of covariance
 *
 * Checks that all eigenvalues are positive
 *
 * @param S Cholesky factor
 * @param min_eigenvalue Minimum allowed eigenvalue
 * @return True if positive definite
 */
bool verify_positive_definite(
    const Eigen::MatrixXd& S,
    double min_eigenvalue = 1e-10
);

/**
 * @brief Apply covariance inflation
 *
 * Inflates covariance by multiplicative factor to prevent filter divergence
 *
 * @param S Cholesky factor
 * @param inflation_factor Multiplicative factor (e.g., 1.01)
 * @return Inflated Cholesky factor
 */
Eigen::MatrixXd apply_inflation(
    const Eigen::MatrixXd& S,
    double inflation_factor
);

} // namespace autonvis

#endif // CHOLESKY_UPDATE_HPP
