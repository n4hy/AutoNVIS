/**
 * @file demo_localization.cpp
 * @brief Demonstration of covariance localization memory savings
 */

#include "cholesky_update.hpp"
#include <iostream>
#include <iomanip>

using namespace autonvis;

int main() {
    std::cout << "=== Covariance Localization Memory Savings Demo ===\n\n";

    // Simulated Auto-NVIS grid (smaller for demo, but scale up to full size)
    std::vector<double> lat_grid, lon_grid, alt_grid;

    // Full Auto-NVIS grid: 73×73×55 = 293,095 + 1 (R_eff) = 293,096
    // Demo grid: 10×10×10 = 1,000 + 1 = 1,001
    const size_t n_lat = 10;
    const size_t n_lon = 10;
    const size_t n_alt = 10;

    for (size_t i = 0; i < n_lat; ++i) {
        lat_grid.push_back(-90.0 + i * (180.0 / (n_lat - 1)));
    }
    for (size_t i = 0; i < n_lon; ++i) {
        lon_grid.push_back(-180.0 + i * (360.0 / (n_lon - 1)));
    }
    for (size_t i = 0; i < n_alt; ++i) {
        alt_grid.push_back(60.0 + i * (500.0 / (n_alt - 1)));
    }

    const size_t state_dim = n_lat * n_lon * n_alt + 1;
    const size_t full_matrix_elements = state_dim * state_dim;

    std::cout << "State dimension: " << state_dim << "\n";
    std::cout << "Full covariance matrix: " << state_dim << " × " << state_dim
              << " = " << full_matrix_elements << " elements\n";

    // Memory for full dense matrix
    const size_t full_memory_bytes = full_matrix_elements * sizeof(double);
    const double full_memory_mb = full_memory_bytes / (1024.0 * 1024.0);

    std::cout << "Full matrix memory: "
              << std::fixed << std::setprecision(2)
              << full_memory_mb << " MB\n\n";

    // Test different localization radii
    std::vector<double> radii = {200.0, 500.0, 1000.0, 2000.0};

    for (double radius : radii) {
        std::cout << "Localization radius: " << radius << " km\n";

        Eigen::SparseMatrix<double> loc_matrix = compute_localization_matrix(
            lat_grid, lon_grid, alt_grid, radius
        );

        const size_t nnz = loc_matrix.nonZeros();
        const double sparsity = 100.0 * (1.0 - static_cast<double>(nnz) / full_matrix_elements);

        const size_t sparse_memory_bytes = nnz * (sizeof(double) + 2 * sizeof(int));  // val + row + col
        const double sparse_memory_mb = sparse_memory_bytes / (1024.0 * 1024.0);

        const double memory_reduction = full_memory_mb / sparse_memory_mb;

        std::cout << "  Non-zeros: " << nnz << " / " << full_matrix_elements
                  << " (" << std::fixed << std::setprecision(1) << sparsity << "% sparse)\n";
        std::cout << "  Sparse memory: " << std::fixed << std::setprecision(2)
                  << sparse_memory_mb << " MB\n";
        std::cout << "  Memory reduction: " << std::fixed << std::setprecision(1)
                  << memory_reduction << "×\n\n";
    }

    // Scale up to full Auto-NVIS grid
    std::cout << "=== Scaling to Full Auto-NVIS Grid (73×73×55 + 1 = 293,096) ===\n\n";

    const size_t full_state_dim = 293096;
    const size_t full_cov_elements = full_state_dim * full_state_dim;
    const size_t full_cov_memory_gb = (full_cov_elements * sizeof(double)) / (1024ULL * 1024ULL * 1024ULL);

    std::cout << "Full covariance matrix: " << full_state_dim << " × " << full_state_dim << "\n";
    std::cout << "Full matrix memory: " << full_cov_memory_gb << " GB\n\n";

    // Estimate sparse memory for 500 km radius
    // Typical sparsity for 500 km radius on Auto-NVIS grid: ~99.5% (0.5% non-zeros)
    const double typical_sparsity = 0.995;
    const size_t sparse_nnz = static_cast<size_t>((1.0 - typical_sparsity) * full_cov_elements);
    const size_t sparse_memory_mb = (sparse_nnz * (sizeof(double) + 2 * sizeof(int))) / (1024 * 1024);

    std::cout << "With 500 km localization (estimated):\n";
    std::cout << "  Sparsity: " << std::fixed << std::setprecision(1)
              << (typical_sparsity * 100.0) << "%\n";
    std::cout << "  Sparse memory: " << sparse_memory_mb << " MB\n";
    std::cout << "  Memory reduction: " << (full_cov_memory_gb * 1024.0) / sparse_memory_mb << "×\n\n";

    std::cout << "Result: Localization makes smoother FEASIBLE!\n";
    std::cout << "  Without localization: " << full_cov_memory_gb << " GB (IMPRACTICAL)\n";
    std::cout << "  With localization: " << sparse_memory_mb << " MB (PRACTICAL)\n";

    return 0;
}
