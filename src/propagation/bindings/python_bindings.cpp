/**
 * @file python_bindings.cpp
 * @brief pybind11 bindings for 3D ray tracer
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "ray_tracer_3d.hpp"

namespace py = pybind11;
using namespace autonvis::propagation;

PYBIND11_MODULE(raytracer, m) {
    m.doc() = "Auto-NVIS 3D Ionospheric Ray Tracer (PHaRLAP-equivalent)";

    // RayState structure
    py::class_<RayState>(m, "RayState")
        .def(py::init<>())
        .def_readwrite("position", &RayState::position)
        .def_readwrite("wave_normal", &RayState::wave_normal)
        .def_readwrite("path_length", &RayState::path_length)
        .def_readwrite("group_path", &RayState::group_path)
        .def_readwrite("phase_path", &RayState::phase_path)
        .def_readwrite("absorption_db", &RayState::absorption_db);

    // RayPath structure
    py::class_<RayPath>(m, "RayPath")
        .def(py::init<>())
        .def_readwrite("positions", &RayPath::positions)
        .def_readwrite("wave_normals", &RayPath::wave_normals)
        .def_readwrite("path_lengths", &RayPath::path_lengths)
        .def_readwrite("group_paths", &RayPath::group_paths)
        .def_readwrite("refractive_indices", &RayPath::refractive_indices)
        .def_readwrite("absorption_db", &RayPath::absorption_db)
        .def_readwrite("ground_range", &RayPath::ground_range)
        .def_readwrite("apex_altitude", &RayPath::apex_altitude)
        .def_readwrite("apex_lat", &RayPath::apex_lat)
        .def_readwrite("apex_lon", &RayPath::apex_lon)
        .def_readwrite("reflected", &RayPath::reflected)
        .def_readwrite("escaped", &RayPath::escaped)
        .def_readwrite("absorbed", &RayPath::absorbed)
        .def("__repr__", [](const RayPath& p) {
            return "<RayPath: " + std::to_string(p.positions.size()) +
                   " points, range=" + std::to_string(p.ground_range) + " km>";
        });

    // IonoGrid class
    py::class_<IonoGrid, std::shared_ptr<IonoGrid>>(m, "IonoGrid")
        .def(py::init<const Eigen::VectorXd&,
                      const Eigen::VectorXd&,
                      const Eigen::VectorXd&,
                      const std::vector<double>&>(),
             py::arg("lat"),
             py::arg("lon"),
             py::arg("alt"),
             py::arg("ne_grid"),
             "Create ionospheric grid\n\n"
             "Parameters:\n"
             "  lat: Latitude grid (degrees)\n"
             "  lon: Longitude grid (degrees)\n"
             "  alt: Altitude grid (km)\n"
             "  ne_grid: Electron density flattened array (row-major: lat, lon, alt)")
        .def("electron_density", &IonoGrid::electron_density,
             py::arg("lat"), py::arg("lon"), py::arg("alt"),
             "Get electron density at position (trilinear interpolation)")
        .def("electron_density_gradient", &IonoGrid::electron_density_gradient,
             py::arg("lat"), py::arg("lon"), py::arg("alt"),
             "Get electron density gradient")
        .def("collision_frequency", &IonoGrid::collision_frequency,
             py::arg("alt"), py::arg("xray_flux") = 0.0,
             "Get collision frequency at altitude");

    // GeomagneticField class
    py::class_<GeomagneticField, std::shared_ptr<GeomagneticField>>(m, "GeomagneticField")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("igrf_coeffs_file"))
        .def("field", &GeomagneticField::field,
             py::arg("lat"), py::arg("lon"), py::arg("alt"), py::arg("year") = 2026,
             "Get magnetic field vector (North, East, Down) in nT")
        .def("field_magnitude", &GeomagneticField::field_magnitude,
             py::arg("lat"), py::arg("lon"), py::arg("alt"), py::arg("year") = 2026,
             "Get magnetic field magnitude in nT")
        .def("dip_angle", &GeomagneticField::dip_angle,
             py::arg("lat"), py::arg("lon"), py::arg("alt"), py::arg("year") = 2026,
             "Get magnetic dip angle in radians");

    // MagnetoionicTheory class
    py::enum_<MagnetoionicTheory::Mode>(m, "Mode")
        .value("O_MODE", MagnetoionicTheory::O_MODE)
        .value("X_MODE", MagnetoionicTheory::X_MODE);

    py::class_<MagnetoionicTheory>(m, "MagnetoionicTheory")
        .def_static("refractive_index", &MagnetoionicTheory::refractive_index,
                   py::arg("ne"),
                   py::arg("freq"),
                   py::arg("B_mag"),
                   py::arg("theta"),
                   py::arg("nu") = 0.0,
                   py::arg("mode") = MagnetoionicTheory::O_MODE,
                   "Calculate complex refractive index from Appleton-Hartree equation")
        .def_static("group_refractive_index", &MagnetoionicTheory::group_refractive_index,
                   py::arg("ne"),
                   py::arg("freq"),
                   py::arg("B_mag"),
                   py::arg("theta"),
                   py::arg("nu") = 0.0,
                   py::arg("mode") = MagnetoionicTheory::O_MODE,
                   "Calculate group refractive index")
        .def_static("absorption_coefficient", &MagnetoionicTheory::absorption_coefficient,
                   py::arg("ne"),
                   py::arg("freq"),
                   py::arg("nu"),
                   "Calculate absorption coefficient in Nepers/m");

    // RayTracingConfig structure
    py::class_<RayTracingConfig>(m, "RayTracingConfig")
        .def(py::init<>())
        .def_readwrite("tolerance", &RayTracingConfig::tolerance)
        .def_readwrite("max_path_length_km", &RayTracingConfig::max_path_length_km)
        .def_readwrite("initial_step_km", &RayTracingConfig::initial_step_km)
        .def_readwrite("min_step_km", &RayTracingConfig::min_step_km)
        .def_readwrite("max_step_km", &RayTracingConfig::max_step_km)
        .def_readwrite("max_steps", &RayTracingConfig::max_steps)
        .def_readwrite("ground_altitude_km", &RayTracingConfig::ground_altitude_km)
        .def_readwrite("escape_altitude_km", &RayTracingConfig::escape_altitude_km)
        .def_readwrite("absorption_threshold_db", &RayTracingConfig::absorption_threshold_db)
        .def_readwrite("calculate_absorption", &RayTracingConfig::calculate_absorption)
        .def_readwrite("calculate_group_path", &RayTracingConfig::calculate_group_path)
        .def_readwrite("mode", &RayTracingConfig::mode);

    // RayTracer3D class
    py::class_<RayTracer3D>(m, "RayTracer3D")
        .def(py::init<std::shared_ptr<IonoGrid>,
                      std::shared_ptr<GeomagneticField>,
                      const RayTracingConfig&>(),
             py::arg("iono_grid"),
             py::arg("geomag"),
             py::arg("config") = RayTracingConfig(),
             "Initialize 3D ray tracer")
        .def("trace_ray", &RayTracer3D::trace_ray,
             py::arg("lat0"),
             py::arg("lon0"),
             py::arg("alt0"),
             py::arg("elevation"),
             py::arg("azimuth"),
             py::arg("freq_mhz"),
             "Trace a single ray\n\n"
             "Parameters:\n"
             "  lat0: Initial latitude (degrees)\n"
             "  lon0: Initial longitude (degrees)\n"
             "  alt0: Initial altitude (km)\n"
             "  elevation: Elevation angle (degrees, 0=horizontal, 90=vertical)\n"
             "  azimuth: Azimuth angle (degrees, clockwise from North)\n"
             "  freq_mhz: Frequency (MHz)\n\n"
             "Returns:\n"
             "  RayPath object with complete ray trajectory")
        .def("trace_ray_fan", &RayTracer3D::trace_ray_fan,
             py::arg("lat0"),
             py::arg("lon0"),
             py::arg("alt0"),
             py::arg("elevations"),
             py::arg("azimuths"),
             py::arg("freq_mhz"),
             "Trace multiple rays (ray fan)\n\n"
             "Parameters:\n"
             "  lat0, lon0, alt0: Initial position\n"
             "  elevations: List of elevation angles (degrees)\n"
             "  azimuths: List of azimuth angles (degrees)\n"
             "  freq_mhz: Frequency (MHz)\n\n"
             "Returns:\n"
             "  List of RayPath objects")
        .def("calculate_nvis_coverage", &RayTracer3D::calculate_nvis_coverage,
             py::arg("tx_lat"),
             py::arg("tx_lon"),
             py::arg("freq_mhz"),
             py::arg("elevation_min") = 70.0,
             py::arg("elevation_max") = 90.0,
             py::arg("elevation_step") = 2.0,
             py::arg("azimuth_step") = 15.0,
             "Calculate NVIS coverage map\n\n"
             "Parameters:\n"
             "  tx_lat, tx_lon: Transmitter position (degrees)\n"
             "  freq_mhz: Frequency (MHz)\n"
             "  elevation_min: Minimum elevation for NVIS (degrees, default 70)\n"
             "  elevation_max: Maximum elevation for NVIS (degrees, default 90)\n"
             "  elevation_step: Elevation angle step (degrees, default 2)\n"
             "  azimuth_step: Azimuth angle step (degrees, default 15)\n\n"
             "Returns:\n"
             "  List of RayPath objects covering all directions");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Auto-NVIS Development Team";
}
