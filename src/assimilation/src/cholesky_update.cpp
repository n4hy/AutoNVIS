/**
 * @file cholesky_update.cpp
 * @brief Implementation of Cholesky factor updates
 */

#include "cholesky_update.hpp"
#include <cmath>
#include <stdexcept>

namespace autonvis {

// cholupdate is now inline in header

Eigen::MatrixXd choldowndate(const Eigen::MatrixXd& S, const Eigen::VectorXd& v) {
    const size_t n = S.rows();

    if (S.cols() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("S must be square");
    }
    if (v.size() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("v dimension mismatch");
    }

    // Compute P_new = S * S^T - v * v^T
    Eigen::MatrixXd P = S * S.transpose();
    Eigen::MatrixXd P_new = P - v * v.transpose();

    // Check if still positive definite
    Eigen::LLT<Eigen::MatrixXd> llt(P_new);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("choldowndate: matrix would not be positive definite");
    }

    // Return upper triangular Cholesky factor (R such that P = R^T * R)
    Eigen::MatrixXd L = llt.matrixL();
    return L.transpose();
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

} // namespace autonvis
