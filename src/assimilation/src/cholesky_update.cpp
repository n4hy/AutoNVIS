/**
 * @file cholesky_update.cpp
 * @brief Implementation of Cholesky factor updates with sparse matrix support
 *
 * Phase 18.2: Efficient rank-1 algorithms using Givens/hyperbolic rotations
 * to avoid O(n²) covariance materialization.
 */

#include "cholesky_update.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

namespace autonvis {

// cholupdate is now inline in header (uses Givens rotations)

Eigen::MatrixXd choldowndate(const Eigen::MatrixXd& L, const Eigen::VectorXd& v) {
    const Eigen::Index n = L.rows();

    if (L.cols() != n) {
        throw std::invalid_argument("L must be square");
    }
    if (v.size() != n) {
        throw std::invalid_argument("v dimension mismatch");
    }

    Eigen::MatrixXd L_new = L;
    Eigen::VectorXd w = v;

    // Use hyperbolic Givens-like rotations for downdate
    // This avoids forming the full covariance matrix P = L * L^T
    for (Eigen::Index j = 0; j < n; ++j) {
        double Ljj = L_new(j, j);
        double wj = w(j);

        // For downdate: L_new(j,j)² = L(j,j)² - w(j)²
        double diff = Ljj * Ljj - wj * wj;

        if (diff <= 0) {
            throw std::runtime_error("choldowndate: matrix would not be positive definite");
        }

        double r = std::sqrt(diff);

        // Hyperbolic rotation parameters
        // cosh(θ) = Ljj / r, sinh(θ) = wj / r
        double c = Ljj / r;  // > 1
        double s = wj / r;

        L_new(j, j) = r;

        // Apply hyperbolic rotation to remaining elements
        for (Eigen::Index i = j + 1; i < n; ++i) {
            double Lij = L_new(i, j);
            double wi = w(i);

            // Hyperbolic rotation: preserves ||L||² - ||w||²
            L_new(i, j) = (c * Lij - s * wi);
            w(i) = (-s * Lij + c * wi);
        }
    }

    return L_new;
}

bool choldowndate_inplace(Eigen::MatrixXd& L, Eigen::VectorXd& w) {
    const Eigen::Index n = L.rows();

    for (Eigen::Index j = 0; j < n; ++j) {
        double Ljj = L(j, j);
        double wj = w(j);
        double diff = Ljj * Ljj - wj * wj;

        if (diff <= 0) {
            return false;  // Would lose positive definiteness
        }

        double r = std::sqrt(diff);
        double c = Ljj / r;
        double s = wj / r;

        L(j, j) = r;

        for (Eigen::Index i = j + 1; i < n; ++i) {
            double Lij = L(i, j);
            double wi = w(i);
            L(i, j) = c * Lij - s * wi;
            w(i) = -s * Lij + c * wi;
        }
    }

    return true;
}

Eigen::MatrixXd qr_decomposition(const Eigen::MatrixXd& A) {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    return qr.matrixQR().triangularView<Eigen::Upper>();
}

Eigen::MatrixXd stack_matrices(
    const Eigen::MatrixXd& S_x,
    const Eigen::MatrixXd& S_v
) {
    if (S_x.cols() != S_v.cols()) {
        throw std::invalid_argument("Column dimension mismatch");
    }

    const size_t rows_total = S_x.rows() + S_v.rows();
    const size_t cols = S_x.cols();

    Eigen::MatrixXd stacked(rows_total, cols);
    stacked << S_x, S_v;

    return stacked;
}

bool verify_positive_definite(
    const Eigen::MatrixXd& S,
    double min_eigenvalue
) {
    // Compute P = S * S^T
    const Eigen::MatrixXd P = S * S.transpose();

    // Compute eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(P);

    if (eigen_solver.info() != Eigen::Success) {
        return false;
    }

    const Eigen::VectorXd eigenvalues = eigen_solver.eigenvalues();

    // Check all eigenvalues are positive
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_eigenvalue) {
            return false;
        }
    }

    return true;
}

Eigen::MatrixXd apply_inflation(
    const Eigen::MatrixXd& S,
    double inflation_factor
) {
    if (inflation_factor < 1.0) {
        throw std::invalid_argument("Inflation factor must be >= 1.0");
    }

    // S_inflated = sqrt(inflation_factor) * S
    return std::sqrt(inflation_factor) * S;
}

// =======================
// Covariance Localization
// =======================

double gaspari_cohn_correlation(double r) {
    const double abs_r = std::abs(r);

    if (abs_r >= 2.0) {
        // Beyond 2x localization radius: zero correlation
        return 0.0;
    } else if (abs_r >= 1.0) {
        // 1 ≤ |r| < 2
        const double r2 = abs_r * abs_r;
        const double r3 = r2 * abs_r;
        const double r4 = r3 * abs_r;
        const double r5 = r4 * abs_r;

        return 4.0 - 5.0*abs_r + (5.0/3.0)*r2 + (5.0/8.0)*r3
               - 0.5*r4 + (1.0/12.0)*r5 - (2.0/(3.0*abs_r));
    } else {
        // 0 ≤ |r| < 1
        const double r2 = abs_r * abs_r;
        const double r3 = r2 * abs_r;
        const double r4 = r3 * abs_r;
        const double r5 = r4 * abs_r;

        return 1.0 - (5.0/3.0)*r2 + (5.0/8.0)*r3 + 0.5*r4 - 0.25*r5;
    }
}

double great_circle_distance(double lat1, double lon1, double lat2, double lon2) {
    const double R_EARTH_KM = 6371.0;  // Mean Earth radius

    // Convert to radians
    const double lat1_rad = lat1 * M_PI / 180.0;
    const double lon1_rad = lon1 * M_PI / 180.0;
    const double lat2_rad = lat2 * M_PI / 180.0;
    const double lon2_rad = lon2 * M_PI / 180.0;

    // Haversine formula
    const double dlat = lat2_rad - lat1_rad;
    const double dlon = lon2_rad - lon1_rad;

    const double a = std::sin(dlat/2.0) * std::sin(dlat/2.0) +
                     std::cos(lat1_rad) * std::cos(lat2_rad) *
                     std::sin(dlon/2.0) * std::sin(dlon/2.0);

    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

    return R_EARTH_KM * c;
}

Eigen::SparseMatrix<double> compute_localization_matrix(
    const std::vector<double>& lat_grid,
    const std::vector<double>& lon_grid,
    const std::vector<double>& alt_grid,
    double localization_radius_km
) {
    const size_t n_lat = lat_grid.size();
    const size_t n_lon = lon_grid.size();
    const size_t n_alt = alt_grid.size();
    const size_t n_grid = n_lat * n_lon * n_alt;
    const size_t dim = n_grid + 1;  // +1 for R_eff

    // Create sparse matrix (triplet format for efficient construction)
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(dim * 100);  // Estimate ~100 non-zeros per row

    // Helper lambda to convert 3D indices to flat index
    auto grid_index = [n_lon, n_alt](size_t i, size_t j, size_t k) -> size_t {
        return i * (n_lon * n_alt) + j * n_alt + k;
    };

    // Compute localization for grid points
    for (size_t i1 = 0; i1 < n_lat; ++i1) {
        for (size_t j1 = 0; j1 < n_lon; ++j1) {
            for (size_t k1 = 0; k1 < n_alt; ++k1) {
                const size_t idx1 = grid_index(i1, j1, k1);
                const double lat1 = lat_grid[i1];
                const double lon1 = lon_grid[j1];

                // Only compute upper triangle (symmetric matrix)
                for (size_t i2 = i1; i2 < n_lat; ++i2) {
                    for (size_t j2 = (i2 == i1 ? j1 : 0); j2 < n_lon; ++j2) {
                        for (size_t k2 = (i2 == i1 && j2 == j1 ? k1 : 0); k2 < n_alt; ++k2) {
                            const size_t idx2 = grid_index(i2, j2, k2);

                            const double lat2 = lat_grid[i2];
                            const double lon2 = lon_grid[j2];

                            // Horizontal distance
                            const double dist_horiz = great_circle_distance(lat1, lon1, lat2, lon2);

                            // Vertical distance
                            const double dist_vert = std::abs(alt_grid[k1] - alt_grid[k2]);

                            // Combined distance (Euclidean in horiz + vert space)
                            const double dist_total = std::sqrt(
                                dist_horiz * dist_horiz + dist_vert * dist_vert
                            );

                            // Normalized distance
                            const double r = dist_total / localization_radius_km;

                            // Gaspari-Cohn correlation
                            const double rho = gaspari_cohn_correlation(r);

                            // Only store non-zero entries (compact support at r=2)
                            if (rho > 1e-8) {
                                triplets.emplace_back(idx1, idx2, rho);
                                if (idx1 != idx2) {
                                    triplets.emplace_back(idx2, idx1, rho);  // Symmetric
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // R_eff is independent of spatial grid (global parameter)
    // Full correlation with itself, zero with grid points
    triplets.emplace_back(n_grid, n_grid, 1.0);

    // Build sparse matrix
    Eigen::SparseMatrix<double> localization(dim, dim);
    localization.setFromTriplets(triplets.begin(), triplets.end());

    return localization;
}

Eigen::MatrixXd apply_localization(
    const Eigen::MatrixXd& P,
    const Eigen::SparseMatrix<double>& localization
) {
    // Schur (element-wise) product: P_localized = P ∘ localization
    // For sparse localization, this significantly reduces memory

    if (P.rows() != localization.rows() || P.cols() != localization.cols()) {
        throw std::invalid_argument("Dimension mismatch in apply_localization");
    }

    // For dense P and sparse localization, iterate over non-zeros
    Eigen::MatrixXd P_localized = Eigen::MatrixXd::Zero(P.rows(), P.cols());

    for (int k = 0; k < localization.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(localization, k); it; ++it) {
            P_localized(it.row(), it.col()) = P(it.row(), it.col()) * it.value();
        }
    }

    return P_localized;
}

// ==============================
// Sparse Matrix Support (18.2)
// ==============================

Eigen::SparseMatrix<double> sparse_cholesky(const Eigen::SparseMatrix<double>& P) {
    // Use SimplicialLLT for sparse Cholesky decomposition
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(P);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("sparse_cholesky: decomposition failed");
    }

    // Extract L from the factorization
    // SimplicialLLT stores L internally
    return solver.matrixL();
}

std::vector<std::pair<int, int>> get_sparsity_pattern(
    const Eigen::SparseMatrix<double>& localization
) {
    std::vector<std::pair<int, int>> pattern;
    pattern.reserve(localization.nonZeros());

    for (int k = 0; k < localization.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(localization, k); it; ++it) {
            pattern.emplace_back(it.row(), it.col());
        }
    }

    return pattern;
}

Eigen::SparseMatrix<double> extract_localized_covariance(
    const Eigen::MatrixXd& S,
    const Eigen::SparseMatrix<double>& localization
) {
    // Extract only elements P(i,j) = sum_k S(i,k)*S(j,k) for (i,j) in sparsity pattern
    // This avoids forming the full O(n²) matrix

    const Eigen::Index n = S.rows();
    const Eigen::Index m = S.cols();

    // Get sparsity pattern
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(localization.nonZeros());

    // For each non-zero in localization, compute P(i,j) = S(i,:) · S(j,:)
    #ifdef HAVE_OPENMP
    // Parallel version with thread-local triplet lists
    std::vector<std::vector<Eigen::Triplet<double>>> thread_triplets;
    #pragma omp parallel
    {
        #pragma omp single
        thread_triplets.resize(omp_get_num_threads());
    }

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < localization.outerSize(); ++k) {
        int tid = omp_get_thread_num();
        for (Eigen::SparseMatrix<double>::InnerIterator it(localization, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            double loc_val = it.value();

            // Compute P(i,j) = sum_l S(i,l) * S(j,l)
            double pij = S.row(i).dot(S.row(j));
            thread_triplets[tid].emplace_back(i, j, pij * loc_val);
        }
    }

    // Merge thread-local triplets
    for (const auto& tlist : thread_triplets) {
        triplets.insert(triplets.end(), tlist.begin(), tlist.end());
    }
    #else
    // Sequential version
    for (int k = 0; k < localization.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(localization, k); it; ++it) {
            int i = it.row();
            int j = it.col();
            double loc_val = it.value();

            // Compute P(i,j) = sum_l S(i,l) * S(j,l)
            double pij = S.row(i).dot(S.row(j));
            triplets.emplace_back(i, j, pij * loc_val);
        }
    }
    #endif

    // Build sparse matrix
    Eigen::SparseMatrix<double> P_loc(n, n);
    P_loc.setFromTriplets(triplets.begin(), triplets.end());

    return P_loc;
}

Eigen::MatrixXd apply_sqrt_localization(
    const Eigen::MatrixXd& S,
    const Eigen::SparseMatrix<double>& localization,
    int rank
) {
    const Eigen::Index n = S.rows();

    // Extract localized covariance (sparse)
    Eigen::SparseMatrix<double> P_loc = extract_localized_covariance(S, localization);

    // Use sparse Cholesky to get sqrt of localized covariance
    try {
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(P_loc);

        if (solver.info() == Eigen::Success) {
            // Return dense version of L
            return Eigen::MatrixXd(solver.matrixL());
        }
    } catch (...) {
        // Fall through to dense fallback
    }

    // Fallback: convert sparse to dense and use standard LLT
    Eigen::MatrixXd P_dense = Eigen::MatrixXd(P_loc);

    // Ensure symmetric (handle numerical issues)
    P_dense = (P_dense + P_dense.transpose()) / 2.0;

    // Add small regularization if needed
    double min_diag = P_dense.diagonal().minCoeff();
    if (min_diag <= 0) {
        P_dense.diagonal().array() += std::abs(min_diag) + 1e-6;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(P_dense);
    if (llt.info() == Eigen::Success) {
        return Eigen::MatrixXd(llt.matrixL());
    }

    // Last resort: return original S
    return S;
}

bool verify_sqrt_positive(const Eigen::MatrixXd& S) {
    // Quick check: diagonal elements must be positive for lower triangular L
    // such that L*L^T is positive definite
    for (Eigen::Index i = 0; i < S.rows(); ++i) {
        if (S(i, i) <= 0) {
            return false;
        }
    }

    // Check for NaN/Inf
    if (!S.allFinite()) {
        return false;
    }

    return true;
}

Eigen::MatrixXd cholupdate_batch(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& V
) {
    const Eigen::Index n = L.rows();
    const Eigen::Index k = V.cols();

    // For small k, sequential updates are fine
    if (k <= 4) {
        Eigen::MatrixXd L_new = L;
        for (Eigen::Index j = 0; j < k; ++j) {
            Eigen::VectorXd w = V.col(j);
            cholupdate_inplace(L_new, w);
        }
        return L_new;
    }

    // For larger k, use blocked QR-based update
    // L_new * L_new^T = L * L^T + V * V^T
    // Form [L; V^T] and compute QR decomposition

    Eigen::MatrixXd stacked(n + k, n);
    stacked.topRows(n) = L;
    stacked.bottomRows(k) = V.transpose();

    // QR decomposition: stacked = Q * R
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(stacked);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // L_new is transpose of upper n×n block of R
    return R.topRows(n).transpose();
}

Eigen::MatrixXd choldowndate_batch(
    const Eigen::MatrixXd& L,
    const Eigen::MatrixXd& V
) {
    const Eigen::Index k = V.cols();

    // Sequential downdates (hyperbolic rotations don't batch well)
    Eigen::MatrixXd L_new = L;
    for (Eigen::Index j = 0; j < k; ++j) {
        Eigen::VectorXd w = V.col(j);
        if (!choldowndate_inplace(L_new, w)) {
            throw std::runtime_error("choldowndate_batch: matrix would not be positive definite");
        }
    }

    return L_new;
}

} // namespace autonvis
