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
#include <Eigen/Sparse>
#include <vector>

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

// =======================
// Covariance Localization
// =======================

/**
 * @brief Gaspari-Cohn 5th-order piecewise polynomial correlation function
 *
 * Compactly supported correlation function for covariance localization.
 * Smoothly goes to zero at r=2 (2× localization radius).
 *
 * @param r Normalized distance (distance / localization_radius)
 * @return Correlation coefficient [0, 1]
 *
 * Reference: Gaspari & Cohn (1999) QJRMS, Eq. 4.10
 */
double gaspari_cohn_correlation(double r);

/**
 * @brief Compute great circle distance between two points on Earth
 *
 * Uses the Haversine formula for accuracy on spherical Earth.
 *
 * @param lat1 Latitude of point 1 (degrees)
 * @param lon1 Longitude of point 1 (degrees)
 * @param lat2 Latitude of point 2 (degrees)
 * @param lon2 Longitude of point 2 (degrees)
 * @return Distance in kilometers
 */
double great_circle_distance(double lat1, double lon1, double lat2, double lon2);

/**
 * @brief Compute sparse localization matrix for ionospheric grid
 *
 * Creates a sparse matrix where each element (i,j) contains the
 * Gaspari-Cohn correlation between grid points i and j based on
 * their spatial separation.
 *
 * Memory efficiency: Sparse storage exploits compact support (rho=0 beyond 2c).
 * For localization radius 500 km, typical sparsity ~99.5% (0.5% non-zeros).
 *
 * @param lat_grid Latitude grid points (degrees)
 * @param lon_grid Longitude grid points (degrees)
 * @param alt_grid Altitude grid points (km)
 * @param localization_radius_km Localization radius (km), recommended 500-1000
 * @return Sparse localization matrix (symmetric, compact support)
 */
Eigen::SparseMatrix<double> compute_localization_matrix(
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid,
    double localization_radius_km
);

/**
 * @brief Apply Schur (element-wise) product for covariance localization
 *
 * Computes P_localized = P ∘ localization, where ∘ is element-wise multiplication.
 * This tapers long-range spurious correlations while preserving local structure.
 *
 * @param P Full covariance matrix (or sqrt covariance squared)
 * @param localization Sparse localization matrix
 * @return Localized covariance matrix
 */
Eigen::MatrixXd apply_localization(
    const Eigen::MatrixXd& P,
    const Eigen::SparseMatrix<double>& localization
);

} // namespace autonvis

#endif // CHOLESKY_UPDATE_HPP
