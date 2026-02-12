/**
 * @file main.cpp
 * @brief Main entry point for assimilation service
 */

#include <iostream>
#include <memory>
#include "sr_ukf.hpp"
#include "physics_model.hpp"

int main() {
    std::cout << "Auto-NVIS SR-UKF Assimilation Service\n";
    std::cout << "Version 0.1.0\n";
    std::cout << "======================================\n\n";

    try {
        // Grid dimensions (simplified for demo)
        const size_t n_lat = 73;   // -90 to +90, 2.5° step
        const size_t n_lon = 73;   // -180 to +180, 5° step
        const size_t n_alt = 55;   // 60 to 600 km, 10 km step

        std::cout << "Grid dimensions: "
                  << n_lat << " × " << n_lon << " × " << n_alt << "\n";
        std::cout << "Total grid points: " << (n_lat * n_lon * n_alt) << "\n";
        std::cout << "State dimension: " << (n_lat * n_lon * n_alt + 1) << "\n\n";

        // Create SR-UKF
        std::cout << "Initializing SR-UKF...\n";
        autonvis::SquareRootUKF filter(n_lat, n_lon, n_alt);

        // Set physics model (Gauss-Markov for now)
        auto physics_model = std::make_shared<autonvis::GaussMarkovModel>();
        filter.set_physics_model(physics_model);

        std::cout << "Physics model: " << physics_model->name() << "\n";

        // Initialize with background state
        autonvis::StateVector initial_state(n_lat, n_lon, n_alt);
        // TODO: Load from IRI-2020 or file

        // Initialize sqrt covariance
        const size_t dim = initial_state.dimension();
        Eigen::MatrixXd initial_sqrt_cov = Eigen::MatrixXd::Identity(dim, dim) * 1e6;

        filter.initialize(initial_state, initial_sqrt_cov);

        std::cout << "\nFilter initialized successfully\n";
        std::cout << "Ready to process observations\n\n";

        // TODO: Set up gRPC server to receive cycle triggers
        // TODO: Implement observation collection
        // TODO: Implement predict/update cycle

        std::cout << "Service would run here (not yet implemented)\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
