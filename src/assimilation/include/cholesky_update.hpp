/**
 * @file cholesky_update.hpp
 * @brief Numerically stable Cholesky factor updates with sparse matrix support
 *
 * Provides rank-1 updates and downdates for Cholesky factors,
 * ensuring positive semi-definiteness throughout SR-UKF.
 *
 * Phase 18.2: Added efficient rank-1 algorithms using Givens rotations
 * to avoid O(n²) covariance materialization.
 */

#ifndef CHOLESKY_UPDATE_HPP
#define CHOLESKY_UPDATE_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>
#include <cmath>

namespace autonvis {

/**
 * @brief Efficient rank-1 Cholesky update using Givens rotations
 *
 * Updates Cholesky factor L such that L_new * L_new^T = L * L^T + v * v^T
 * Uses Givens rotations to avoid O(n²) covariance materialization.
 * Time complexity: O(n²) instead of O(n³) for full recomputation.
 *
 * @param L Input lower triangular Cholesky factor
 * @param v Update vector
 * @return Updated lower triangular Cholesky factor
 */
inline Eigen::MatrixXd cholupdate(const Eigen::MatrixXd& L, const Eigen::VectorXd& v) {
    const Eigen::Index n = L.rows();
    Eigen::MatrixXd L_new = L;
    Eigen::VectorXd w = v;

    // Apply sequence of Givens rotations to update L
    for (Eigen::Index j = 0; j < n; ++j) {
        // Compute Givens rotation to zero w(j)
        double Ljj = L_new(j, j);
        double wj = w(j);
        double r = std::hypot(Ljj, wj);

        if (r < 1e-15) {
            // Skip near-zero entries
            continue;
        }

        double c = Ljj / r;  // cosine
        double s = wj / r;   // sine

        // Update diagonal element
        L_new(j, j) = r;

        // Apply rotation to remaining elements in column j and w
        for (Eigen::Index i = j + 1; i < n; ++i) {
            double Lij = L_new(i, j);
            double wi = w(i);
            L_new(i, j) = c * Lij + s * wi;
            w(i) = c * wi - s * Lij;
        }
    }

    return L_new;
}

/**
 * @brief Efficient rank-1 Cholesky update (in-place version)
 *
 * Updates L in place to avoid memory allocation.
 *
 * @param L Lower triangular Cholesky factor (modified in place)
 * @param v Update vector
 */
inline void cholupdate_inplace(Eigen::MatrixXd& L, Eigen::VectorXd& w) {
    const Eigen::Index n = L.rows();

    for (Eigen::Index j = 0; j < n; ++j) {
        double Ljj = L(j, j);
        double wj = w(j);
        double r = std::hypot(Ljj, wj);

        if (r < 1e-15) continue;

        double c = Ljj / r;
        double s = wj / r;

        L(j, j) = r;

        for (Eigen::Index i = j + 1; i < n; ++i) {
            double Lij = L(i, j);
            double wi = w(i);
            L(i, j) = c * Lij + s * wi;
            w(i) = c * wi - s * Lij;
        }
    }
}

/**
 * @brief Efficient rank-1 Cholesky downdate using hyperbolic rotations
 *
 * Updates Cholesky factor L such that L_new * L_new^T = L * L^T - v * v^T
 * Uses hyperbolic (Givens-like) rotations to avoid O(n²) covariance materialization.
 *
 * @param L Input lower triangular Cholesky factor
 * @param v Downdate vector
 * @return Updated lower triangular Cholesky factor
 * @throws std::runtime_error if result would not be positive definite
 */
Eigen::MatrixXd choldowndate(const Eigen::MatrixXd& L, const Eigen::VectorXd& v);

/**
 * @brief Efficient rank-1 Cholesky downdate (in-place version)
 *
 * @param L Lower triangular Cholesky factor (modified in place)
 * @param w Downdate vector (modified in place)
 * @return true if successful, false if matrix would become non-positive-definite
 */
bool choldowndate_inplace(Eigen::MatrixXd& L, Eigen::VectorXd& w);

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

// ==============================
// Sparse Matrix Support (18.2)
// ==============================

/**
 * @brief Sparse Cholesky decomposition wrapper
 *
 * Uses Eigen's SimplicialLLT for efficient sparse Cholesky factorization.
 * Suitable for localized covariance matrices with high sparsity (>95%).
 *
 * @param P Sparse positive-definite matrix
 * @return Sparse lower triangular Cholesky factor
 * @throws std::runtime_error if decomposition fails
 */
Eigen::SparseMatrix<double> sparse_cholesky(const Eigen::SparseMatrix<double>& P);

/**
 * @brief Convert sparse localization to sparsity pattern
 *
 * Creates a boolean sparsity pattern from a localization matrix,
 * useful for efficient sparse operations.
 *
 * @param localization Sparse localization matrix
 * @return Sparsity pattern (non-zero positions)
 */
std::vector<std::pair<int, int>> get_sparsity_pattern(
    const Eigen::SparseMatrix<double>& localization
);

/**
 * @brief Apply localization directly to sqrt covariance
 *
 * Avoids forming full covariance P = S * S^T by working with S directly.
 * Uses randomized low-rank approximation for efficiency.
 *
 * For localized covariance: P_loc = (S * S^T) ∘ L
 * This function returns S_loc such that S_loc * S_loc^T ≈ P_loc
 *
 * @param S Input sqrt covariance (dense)
 * @param localization Sparse localization matrix
 * @param rank Approximation rank (0 = full rank)
 * @return Localized sqrt covariance
 */
Eigen::MatrixXd apply_sqrt_localization(
    const Eigen::MatrixXd& S,
    const Eigen::SparseMatrix<double>& localization,
    int rank = 0
);

/**
 * @brief Memory-efficient covariance extraction
 *
 * Extracts only the elements of P = S * S^T that correspond to
 * non-zero entries in the localization pattern.
 * Avoids forming the full O(n²) covariance matrix.
 *
 * @param S Sqrt covariance matrix
 * @param localization Sparse localization matrix (defines extraction pattern)
 * @return Sparse matrix containing only localized covariance elements
 */
Eigen::SparseMatrix<double> extract_localized_covariance(
    const Eigen::MatrixXd& S,
    const Eigen::SparseMatrix<double>& localization
);

/**
 * @brief Verify sqrt matrix maintains positive definiteness
 *
 * Efficient check that avoids full eigendecomposition by using
 * diagonal dominance heuristics.
 *
 * @param S Sqrt covariance matrix
 * @return true if S appears positive definite
 */
bool verify_sqrt_positive(const Eigen::MatrixXd& S);

/**
 * @brief Batch rank-1 updates using blocked algorithm
 *
 * Performs multiple rank-1 updates efficiently using blocked operations.
 * More efficient than sequential updates for large numbers of vectors.
 *
 * L_new * L_new^T = L * L^T + sum_i(v_i * v_i^T)
 *
 * @param L Input lower triangular Cholesky factor
 * @param V Matrix where each column is an update vector
 * @return Updated Cholesky factor
 */
Eigen::MatrixXd cholupdate_batch(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& V
);

/**
 * @brief Batch rank-1 downdates using blocked algorithm
 *
 * @param L Input lower triangular Cholesky factor
 * @param V Matrix where each column is a downdate vector
 * @return Updated Cholesky factor
 * @throws std::runtime_error if result would not be positive definite
 */
Eigen::MatrixXd choldowndate_batch(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& V
);

} // namespace autonvis

#endif // CHOLESKY_UPDATE_HPP
