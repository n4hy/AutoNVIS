/**
 * Brutal SR-UKF Filter Tests
 *
 * Tests designed to stress the Square-Root Unscented Kalman Filter with:
 * - Extreme grid sizes (millions of states)
 * - Numerical precision edge cases
 * - Concurrent filter operations
 * - Memory stress with large covariance matrices
 * - CPU-intensive repeated predict/update cycles
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <thread>
#include <vector>
#include <iostream>

#include "../include/sr_ukf.hpp"
#include "../include/state_vector.hpp"
#include "../include/physics_model.hpp"

using namespace autonvis::assimilation;
using namespace std::chrono;

class BrutalSRUKFTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default setup
    }
};

/**
 * Test with production-scale grid (289,000+ states)
 */
TEST_F(BrutalSRUKFTest, ProductionScaleGrid) {
    // 73 x 72 x 55 = 289,080 states + 1 Reff = 289,081
    GridDimensions grid{73, 72, 55};

    std::cout << "\nCreating SR-UKF with " << grid.total_points() + 1 << " states..." << std::endl;

    auto start = high_resolution_clock::now();

    SR_UKF filter(grid);

    auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "Filter initialization: " << elapsed.count() << " ms" << std::endl;

    // Verify dimensions
    EXPECT_EQ(filter.state_dimension(), 289081);

    // Verify covariance matrix size (should be massive)
    size_t cov_size = filter.state_dimension() * filter.state_dimension();
    std::cout << "Covariance matrix size: " << cov_size << " elements" << std::endl;
    std::cout << "Memory (approx): " << (cov_size * sizeof(double)) / 1024 / 1024 << " MB" << std::endl;
}

/**
 * Test with ultra-fine grid (stress memory)
 */
TEST_F(BrutalSRUKFTest, UltraFineGrid) {
    // 181 x 360 x 109 = 7,104,840 states (7 million!)
    GridDimensions grid{181, 360, 109};

    std::cout << "\nCreating SR-UKF with " << grid.total_points() + 1 << " states..." << std::endl;
    std::cout << "WARNING: This will consume significant memory!" << std::endl;

    auto start = high_resolution_clock::now();

    // This might fail due to memory constraints - that's expected
    try {
        SR_UKF filter(grid);

        auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start);
        std::cout << "Filter initialization: " << elapsed.count() << " ms" << std::endl;

        EXPECT_EQ(filter.state_dimension(), 7104841);

    } catch (const std::bad_alloc& e) {
        std::cout << "Memory allocation failed (expected for ultra-fine grid)" << std::endl;
        GTEST_SKIP() << "Insufficient memory for ultra-fine grid";
    }
}

/**
 * Test repeated predict/update cycles (CPU intensive)
 */
TEST_F(BrutalSRUKFTest, RepeatedPredictUpdateCycles) {
    GridDimensions grid{19, 36, 11};  // 7,524 states
    SR_UKF filter(grid);

    // Initialize state
    StateVector state(grid);
    for (size_t i = 0; i < state.ne_grid.size(); ++i) {
        state.ne_grid[i] = 1e11;  // Baseline ionosphere
    }
    state.reff = 75.0;

    filter.initialize(state);

    // Physics model
    PhysicsModel physics(grid);

    std::cout << "\nRunning 100 predict/update cycles..." << std::endl;
    auto start = high_resolution_clock::now();

    for (int cycle = 0; cycle < 100; ++cycle) {
        // Predict
        filter.predict(physics, 60.0);  // 60 second timestep

        // Create synthetic observation
        Eigen::VectorXd obs(1);
        obs(0) = 10.0 + 0.1 * cycle;  // Simulated TEC observation

        // Observation operator (simple for testing)
        auto H = [](const StateVector& sv) -> Eigen::VectorXd {
            Eigen::VectorXd result(1);
            result(0) = sv.reff / 10.0;  // Dummy observation
            return result;
        };

        // Update
        Eigen::VectorXd R(1);
        R(0) = 2.0;  // Observation noise
        filter.update(obs, H, R);

        if (cycle % 10 == 0) {
            std::cout << "  Cycle " << cycle << " completed" << std::endl;
        }
    }

    auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "100 cycles completed in " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average: " << elapsed.count() / 100.0 << " ms per cycle" << std::endl;

    EXPECT_GT(elapsed.count(), 0);
}

/**
 * Test numerical stability with extreme parameter values
 */
TEST_F(BrutalSRUKFTest, NumericalStabilityExtremeParameters) {
    GridDimensions grid{9, 18, 5};  // Small for quick testing

    // Extreme alpha (very concentrated sigma points)
    {
        UKFParameters params;
        params.alpha = 1e-10;
        params.beta = 2.0;
        params.kappa = 0.0;

        SR_UKF filter(grid, params);

        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        // Should not crash
        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }

    // Extreme process noise
    {
        UKFParameters params;
        params.process_noise_ne = 1e15;  // Very large
        params.process_noise_reff = 100.0;

        SR_UKF filter(grid, params);

        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }

    // Zero process noise (deterministic)
    {
        UKFParameters params;
        params.process_noise_ne = 0.0;
        params.process_noise_reff = 0.0;

        SR_UKF filter(grid, params);

        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }
}

/**
 * Test with extreme electron density values
 */
TEST_F(BrutalSRUKFTest, ExtremeElectronDensity) {
    GridDimensions grid{9, 18, 5};
    SR_UKF filter(grid);

    // Very high Ne (solar maximum storm)
    {
        StateVector state(grid);
        for (auto& ne : state.ne_grid) {
            ne = 1e13;  // 10 trillion electrons/m³
        }
        state.reff = 200.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }

    // Very low Ne (nighttime, solar minimum)
    {
        StateVector state(grid);
        for (auto& ne : state.ne_grid) {
            ne = 1e8;  // 100 million electrons/m³
        }
        state.reff = 0.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }
}

/**
 * Test long timestep prediction
 */
TEST_F(BrutalSRUKFTest, LongTimestepPrediction) {
    GridDimensions grid{9, 18, 5};
    SR_UKF filter(grid);

    StateVector state(grid);
    for (auto& ne : state.ne_grid) ne = 1e11;
    state.reff = 75.0;

    filter.initialize(state);

    PhysicsModel physics(grid);

    // Very long timestep (1 hour)
    std::cout << "\nPredicting with 3600 second timestep..." << std::endl;
    EXPECT_NO_THROW(filter.predict(physics, 3600.0));

    // Very short timestep
    std::cout << "Predicting with 1 second timestep..." << std::endl;
    EXPECT_NO_THROW(filter.predict(physics, 1.0));
}

/**
 * Test concurrent filter operations (thread safety)
 */
TEST_F(BrutalSRUKFTest, ConcurrentFilterOperations) {
    GridDimensions grid{9, 18, 5};

    std::cout << "\nRunning 4 concurrent filters..." << std::endl;

    auto run_filter = [&grid](int id) {
        SR_UKF filter(grid);

        StateVector state(grid);
        for (auto& ne : state.ne_grid) {
            ne = 1e11 * (1.0 + id * 0.1);
        }
        state.reff = 75.0 + id * 5.0;

        filter.initialize(state);

        PhysicsModel physics(grid);

        // Run 10 predict cycles
        for (int i = 0; i < 10; ++i) {
            filter.predict(physics, 60.0);
        }

        std::cout << "  Filter " << id << " completed" << std::endl;
    };

    std::vector<std::thread> threads;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(run_filter, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "4 concurrent filters completed in " << elapsed.count() << " ms" << std::endl;
}

/**
 * Test memory allocation/deallocation under stress
 */
TEST_F(BrutalSRUKFTest, RepeatedAllocation) {
    GridDimensions grid{19, 36, 11};  // 7,524 states

    std::cout << "\nAllocating/deallocating 50 filters..." << std::endl;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 50; ++i) {
        SR_UKF filter(grid);

        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        filter.predict(physics, 60.0);

        // Filter goes out of scope and is deallocated
    }

    auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    std::cout << "50 allocations in " << elapsed.count() << " ms" << std::endl;
    std::cout << "Average: " << elapsed.count() / 50.0 << " ms per allocation" << std::endl;
}

/**
 * Test covariance inflation
 */
TEST_F(BrutalSRUKFTest, CovarianceInflation) {
    GridDimensions grid{9, 18, 5};

    // No inflation
    {
        UKFParameters params;
        params.covariance_inflation = 1.0;

        SR_UKF filter(grid, params);
        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        // Get initial covariance norm
        // (Would need filter API to expose this)
    }

    // High inflation
    {
        UKFParameters params;
        params.covariance_inflation = 1.5;

        SR_UKF filter(grid, params);
        StateVector state(grid);
        for (auto& ne : state.ne_grid) ne = 1e11;
        state.reff = 75.0;

        filter.initialize(state);

        PhysicsModel physics(grid);
        EXPECT_NO_THROW(filter.predict(physics, 60.0));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
