/**
 * @file python_bindings.cpp
 * @brief pybind11 Python bindings for Auto-NVIS SR-UKF
 *
 * Exposes C++ SR-UKF implementation to Python for integration
 * with the supervisor and mode controller.
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "sr_ukf.hpp"
#include "state_vector.hpp"
#include "observation_model.hpp"
#include "physics_model.hpp"
#include "cholesky_update.hpp"

namespace py = pybind11;
using namespace autonvis;

PYBIND11_MODULE(autonvis_srukf, m) {
    m.doc() = "Auto-NVIS Square-Root UKF Python Bindings";

    // ============================
    // StateVector
    // ============================
    py::class_<StateVector>(m, "StateVector")
        .def(py::init<size_t, size_t, size_t>(),
             py::arg("n_lat"), py::arg("n_lon"), py::arg("n_alt"),
             "Create state vector with specified grid dimensions")

        .def("dimension", &StateVector::dimension,
             "Get total state dimension")

        .def("get_ne", &StateVector::get_ne,
             py::arg("i"), py::arg("j"), py::arg("k"),
             "Get electron density at grid point (i, j, k)")

        .def("set_ne", &StateVector::set_ne,
             py::arg("i"), py::arg("j"), py::arg("k"), py::arg("ne"),
             "Set electron density at grid point (i, j, k)")

        .def("get_reff", &StateVector::get_reff,
             "Get effective sunspot number")

        .def("set_reff", &StateVector::set_reff,
             py::arg("reff"),
             "Set effective sunspot number")

        .def("to_vector", &StateVector::to_vector,
             "Convert to Eigen vector")

        .def("from_vector", &StateVector::from_vector,
             py::arg("vec"),
             "Set from Eigen vector")

        .def("to_numpy", [](const StateVector& self) {
            // Convert to NumPy array (will be handled by pybind11/eigen)
            return self.to_vector();
        }, "Convert to NumPy array")

        .def("from_numpy", [](StateVector& self, const Eigen::VectorXd& vec) {
            self.from_vector(vec);
        }, py::arg("array"), "Set from NumPy array");

    // ============================
    // PhysicsModel (abstract base)
    // ============================
    py::class_<PhysicsModel, std::shared_ptr<PhysicsModel>>(m, "PhysicsModel")
        .def("name", &PhysicsModel::name,
             "Get physics model name");

    // ============================
    // GaussMarkovModel
    // ============================
    py::class_<GaussMarkovModel, PhysicsModel, std::shared_ptr<GaussMarkovModel>>(m, "GaussMarkovModel")
        .def(py::init<double, double>(),
             py::arg("correlation_time") = 3600.0,
             py::arg("process_noise_std") = 1e10,
             "Create Gauss-Markov physics model")

        .def("name", &GaussMarkovModel::name);

    // ============================
    // ObservationModel (abstract base)
    // ============================
    py::class_<ObservationModel, std::shared_ptr<ObservationModel>>(m, "ObservationModel")
        .def("obs_dimension", &ObservationModel::obs_dimension,
             "Get observation dimension");

    // ============================
    // TECObservationModel
    // ============================
    py::class_<TECObservationModel::TECMeasurement>(m, "TECMeasurement")
        .def(py::init<>())
        .def_readwrite("latitude", &TECObservationModel::TECMeasurement::latitude)
        .def_readwrite("longitude", &TECObservationModel::TECMeasurement::longitude)
        .def_readwrite("altitude", &TECObservationModel::TECMeasurement::altitude)
        .def_readwrite("elevation", &TECObservationModel::TECMeasurement::elevation)
        .def_readwrite("azimuth", &TECObservationModel::TECMeasurement::azimuth)
        .def_readwrite("tec_value", &TECObservationModel::TECMeasurement::tec_value)
        .def_readwrite("tec_error", &TECObservationModel::TECMeasurement::tec_error);

    py::class_<TECObservationModel, ObservationModel, std::shared_ptr<TECObservationModel>>(m, "TECObservationModel")
        .def(py::init<const std::vector<TECObservationModel::TECMeasurement>&,
                      const std::vector<double>&,
                      const std::vector<double>&,
                      const std::vector<double>&>(),
             py::arg("measurements"),
             py::arg("lat_grid"),
             py::arg("lon_grid"),
             py::arg("alt_grid"),
             "Create TEC observation model");

    // ============================
    // SquareRootUKF Configuration
    // ============================
    py::class_<SquareRootUKF::AdaptiveInflationConfig>(m, "AdaptiveInflationConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &SquareRootUKF::AdaptiveInflationConfig::enabled)
        .def_readwrite("initial_inflation", &SquareRootUKF::AdaptiveInflationConfig::initial_inflation)
        .def_readwrite("min_inflation", &SquareRootUKF::AdaptiveInflationConfig::min_inflation)
        .def_readwrite("max_inflation", &SquareRootUKF::AdaptiveInflationConfig::max_inflation)
        .def_readwrite("adaptation_rate", &SquareRootUKF::AdaptiveInflationConfig::adaptation_rate)
        .def_readwrite("divergence_threshold", &SquareRootUKF::AdaptiveInflationConfig::divergence_threshold);

    py::class_<SquareRootUKF::LocalizationConfig>(m, "LocalizationConfig")
        .def(py::init<>())
        .def_readwrite("enabled", &SquareRootUKF::LocalizationConfig::enabled)
        .def_readwrite("radius_km", &SquareRootUKF::LocalizationConfig::radius_km)
        .def_readwrite("precompute", &SquareRootUKF::LocalizationConfig::precompute);

    py::class_<SquareRootUKF::Statistics>(m, "FilterStatistics")
        .def_readonly("predict_count", &SquareRootUKF::Statistics::predict_count)
        .def_readonly("update_count", &SquareRootUKF::Statistics::update_count)
        .def_readonly("last_predict_time_ms", &SquareRootUKF::Statistics::last_predict_time_ms)
        .def_readonly("last_update_time_ms", &SquareRootUKF::Statistics::last_update_time_ms)
        .def_readonly("avg_predict_time_ms", &SquareRootUKF::Statistics::avg_predict_time_ms)
        .def_readonly("avg_update_time_ms", &SquareRootUKF::Statistics::avg_update_time_ms)
        .def_readonly("min_eigenvalue", &SquareRootUKF::Statistics::min_eigenvalue)
        .def_readonly("max_eigenvalue", &SquareRootUKF::Statistics::max_eigenvalue)
        .def_readonly("last_nis", &SquareRootUKF::Statistics::last_nis)
        .def_readonly("avg_nis", &SquareRootUKF::Statistics::avg_nis)
        .def_readonly("inflation_factor", &SquareRootUKF::Statistics::inflation_factor)
        .def_readonly("divergence_count", &SquareRootUKF::Statistics::divergence_count);

    // ============================
    // SquareRootUKF (Main class)
    // ============================
    py::class_<SquareRootUKF>(m, "SquareRootUKF")
        .def(py::init<size_t, size_t, size_t, double, double, double>(),
             py::arg("n_lat"),
             py::arg("n_lon"),
             py::arg("n_alt"),
             py::arg("alpha") = 1e-3,
             py::arg("beta") = 2.0,
             py::arg("kappa") = 0.0,
             "Create Square-Root UKF filter")

        .def("initialize", &SquareRootUKF::initialize,
             py::arg("initial_state"),
             py::arg("initial_sqrt_cov"),
             "Initialize filter with background state")

        .def("set_physics_model", &SquareRootUKF::set_physics_model,
             py::arg("model"),
             "Set physics model for state propagation")

        .def("predict", &SquareRootUKF::predict,
             py::arg("dt"),
             "Predict step: propagate state forward")

        .def("update", &SquareRootUKF::update,
             py::arg("obs_model"),
             py::arg("observations"),
             py::arg("obs_sqrt_cov"),
             "Update step: assimilate observations")

        .def("get_state", &SquareRootUKF::get_state,
             py::return_value_policy::reference_internal,
             "Get current state estimate")

        .def("get_sqrt_cov", &SquareRootUKF::get_sqrt_cov,
             py::return_value_policy::reference_internal,
             "Get current sqrt covariance")

        .def("get_covariance", &SquareRootUKF::get_covariance,
             "Get full covariance matrix (P = S * S^T)")

        .def("get_statistics", &SquareRootUKF::get_statistics,
             "Get filter statistics")

        .def("set_adaptive_inflation_config", &SquareRootUKF::set_adaptive_inflation_config,
             py::arg("config"),
             "Set adaptive inflation configuration")

        .def("get_inflation_config", &SquareRootUKF::get_inflation_config,
             py::return_value_policy::reference_internal,
             "Get current inflation configuration")

        .def("set_localization_config", &SquareRootUKF::set_localization_config,
             py::arg("config"),
             py::arg("lat_grid"),
             py::arg("lon_grid"),
             py::arg("alt_grid"),
             "Set covariance localization configuration")

        .def("get_localization_config", &SquareRootUKF::get_localization_config,
             py::return_value_policy::reference_internal,
             "Get current localization configuration")

        .def("set_process_noise", &SquareRootUKF::set_process_noise,
             py::arg("process_sqrt_cov"),
             "Set process noise sqrt covariance")

        .def("apply_inflation", &SquareRootUKF::apply_inflation,
             py::arg("factor"),
             "Manually apply covariance inflation");

    // ============================
    // Utility Functions
    // ============================
    m.def("gaspari_cohn_correlation", &gaspari_cohn_correlation,
          py::arg("r"),
          "Compute Gaspari-Cohn correlation function");

    m.def("great_circle_distance", &great_circle_distance,
          py::arg("lat1"), py::arg("lon1"), py::arg("lat2"), py::arg("lon2"),
          "Compute great circle distance between two points (km)");

    m.def("compute_localization_matrix", &compute_localization_matrix,
          py::arg("lat_grid"), py::arg("lon_grid"), py::arg("alt_grid"),
          py::arg("localization_radius_km"),
          "Compute sparse localization matrix");

    // Version info
    m.attr("__version__") = "0.1.0";
}
