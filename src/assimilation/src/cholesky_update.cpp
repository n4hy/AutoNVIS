/**
 * @file cholesky_update.cpp
 * @brief Implementation of Cholesky factor updates
 */

#include "cholesky_update.hpp"
#include <cmath>
#include <stdexcept>

namespace autonvis {

Eigen::MatrixXd cholupdate(const Eigen::MatrixXd& S, const Eigen::VectorXd& v) {
    const size_t n = S.rows();

    if (S.cols() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("S must be square");
    }
    if (v.size() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("v dimension mismatch");
    }

    // Copy S (we'll modify in place)
    Eigen::MatrixXd S_new = S;
    Eigen::VectorXd v_work = v;

    // Givens rotation-based update
    for (size_t k = 0; k < n; ++k) {
        const double r = std::sqrt(S_new(k, k) * S_new(k, k) + v_work(k) * v_work(k));
        const double c = r / S_new(k, k);
        const double s = v_work(k) / S_new(k, k);

        S_new(k, k) = r;

        if (k + 1 < n) {
            S_new.block(k, k + 1, 1, n - k - 1) =
                (S_new.block(k, k + 1, 1, n - k - 1) +
                 s * v_work.segment(k + 1, n - k - 1).transpose()) / c;

            v_work.segment(k + 1, n - k - 1) =
                c * v_work.segment(k + 1, n - k - 1) -
                s * S_new.block(k, k + 1, 1, n - k - 1).transpose();
        }
    }

    return S_new;
}

Eigen::MatrixXd choldowndate(const Eigen::MatrixXd& S, const Eigen::VectorXd& v) {
    const size_t n = S.rows();

    if (S.cols() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("S must be square");
    }
    if (v.size() != static_cast<Eigen::Index>(n)) {
        throw std::invalid_argument("v dimension mismatch");
    }

    // Copy S
    Eigen::MatrixXd S_new = S;
    Eigen::VectorXd v_work = v;

    // Hyperbolic rotation-based downdate
    for (size_t k = 0; k < n; ++k) {
        const double r_sq = S_new(k, k) * S_new(k, k) - v_work(k) * v_work(k);

        if (r_sq <= 0) {
            throw std::runtime_error("choldowndate: matrix would not be positive definite");
        }

        const double r = std::sqrt(r_sq);
        const double c = r / S_new(k, k);
        const double s = v_work(k) / S_new(k, k);

        S_new(k, k) = r;

        if (k + 1 < n) {
            S_new.block(k, k + 1, 1, n - k - 1) =
                (S_new.block(k, k + 1, 1, n - k - 1) -
                 s * v_work.segment(k + 1, n - k - 1).transpose()) / c;

            v_work.segment(k + 1, n - k - 1) =
                c * v_work.segment(k + 1, n - k - 1) -
                s * S_new.block(k, k + 1, 1, n - k - 1).transpose();
        }
    }

    return S_new;
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

} // namespace autonvis
