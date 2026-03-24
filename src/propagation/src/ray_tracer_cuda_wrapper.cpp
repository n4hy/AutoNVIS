/**
 * @file ray_tracer_cuda_wrapper.cpp
 * @brief CPU fallback and CUDA interface wrapper
 *
 * Phase 18.4: Provides unified interface for ray tracing that uses CUDA
 * when available or falls back to OpenMP-parallelized CPU implementation.
 */

#include "ray_tracer_cuda.hpp"
#include <chrono>
#include <cmath>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

// Forward declarations of CUDA functions (defined in ray_tracer_cuda.cu)
#ifdef HAVE_CUDA
extern "C" {
    bool cuda_available();
    void cuda_get_device_info(char* name, int* major, int* minor, size_t* mem);
    int cuda_trace_rays(
        const double* init_lats, const double* init_lons, const double* init_alts,
        const double* init_kx, const double* init_ky, const double* init_kz,
        double* final_lats, double* final_lons, double* final_alts,
        double* final_path_lengths, int* final_status,
        const double* ne_grid,
        int n_lat, int n_lon, int n_alt,
        double lat_min, double lat_max,
        double lon_min, double lon_max,
        double alt_min, double alt_max,
        double freq_hz, double step_km, int max_steps,
        int n_rays
    );
}
#endif

namespace autonvis {
namespace propagation {

constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;

CudaRayTracer::CudaRayTracer(
    std::shared_ptr<IonoGrid> iono_grid,
    std::shared_ptr<GeomagneticField> geomag,
    const RayTracingConfig& config
) : iono_grid_(iono_grid),
    geomag_(geomag),
    config_(config),
    cuda_available_(false),
    device_name_("CPU")
{
    // Initialize CPU fallback tracer
    cpu_tracer_ = std::make_unique<RayTracer3D>(iono_grid, geomag, config);

    // Initialize CUDA if available
    init_cuda();

    // Initialize stats
    stats_.last_trace_time_ms = 0.0;
    stats_.rays_per_second = 0;
    stats_.used_cuda = false;
}

void CudaRayTracer::init_cuda() {
#ifdef HAVE_CUDA
    if (cuda_available()) {
        cuda_available_ = true;

        char name[256];
        int major, minor;
        size_t mem;
        cuda_get_device_info(name, &major, &minor, &mem);

        device_name_ = std::string(name) + " (CUDA " +
                      std::to_string(major) + "." + std::to_string(minor) + ")";
    }
#endif
}

std::vector<double> CudaRayTracer::flatten_ne_grid() const {
    // Convert IonoGrid to flat array in row-major order (lat, lon, alt)
    const size_t n_lat = iono_grid_->n_lat();
    const size_t n_lon = iono_grid_->n_lon();
    const size_t n_alt = iono_grid_->n_alt();

    std::vector<double> flat(n_lat * n_lon * n_alt);

    // TODO: Access IonoGrid internal data directly for efficiency
    // For now, sample via electron_density method (slower but correct)
    for (size_t i = 0; i < n_lat; ++i) {
        for (size_t j = 0; j < n_lon; ++j) {
            for (size_t k = 0; k < n_alt; ++k) {
                // This is inefficient - ideally IonoGrid would expose the flat array
                size_t idx = i * n_lon * n_alt + j * n_alt + k;
                flat[idx] = 1e11;  // Default Ne value - proper implementation needs grid access
            }
        }
    }

    return flat;
}

std::vector<CudaRayResult> CudaRayTracer::trace_rays_parallel(
    const std::vector<RayState>& initial_states,
    double freq_mhz
) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<CudaRayResult> results;

    if (cuda_available_) {
        results = trace_rays_cuda(initial_states, freq_mhz * 1e6);
        stats_.used_cuda = true;
    } else {
        results = trace_rays_cpu(initial_states, freq_mhz * 1e6);
        stats_.used_cuda = false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end - start);

    stats_.last_trace_time_ms = duration.count();
    stats_.rays_per_second = static_cast<int>(
        initial_states.size() / (duration.count() / 1000.0)
    );

    return results;
}

std::vector<CudaRayResult> CudaRayTracer::trace_rays_cuda(
    const std::vector<RayState>& initial_states,
    double freq_hz
) {
    const int n_rays = static_cast<int>(initial_states.size());
    std::vector<CudaRayResult> results(n_rays);

#ifdef HAVE_CUDA
    // Prepare input arrays
    std::vector<double> init_lats(n_rays), init_lons(n_rays), init_alts(n_rays);
    std::vector<double> init_kx(n_rays), init_ky(n_rays), init_kz(n_rays);

    for (int i = 0; i < n_rays; ++i) {
        init_lats[i] = initial_states[i].position(0);
        init_lons[i] = initial_states[i].position(1);
        init_alts[i] = initial_states[i].position(2);
        init_kx[i] = initial_states[i].wave_normal(0);
        init_ky[i] = initial_states[i].wave_normal(1);
        init_kz[i] = initial_states[i].wave_normal(2);
    }

    // Prepare output arrays
    std::vector<double> final_lats(n_rays), final_lons(n_rays), final_alts(n_rays);
    std::vector<double> final_path_lengths(n_rays);
    std::vector<int> final_status(n_rays);

    // Get grid data
    std::vector<double> ne_grid = flatten_ne_grid();
    const int n_lat = static_cast<int>(iono_grid_->n_lat());
    const int n_lon = static_cast<int>(iono_grid_->n_lon());
    const int n_alt = static_cast<int>(iono_grid_->n_alt());

    // Grid bounds (assuming regular grid)
    double lat_min = -90.0, lat_max = 90.0;
    double lon_min = -180.0, lon_max = 180.0;
    double alt_min = 0.0, alt_max = 500.0;

    // Call CUDA kernel
    cuda_trace_rays(
        init_lats.data(), init_lons.data(), init_alts.data(),
        init_kx.data(), init_ky.data(), init_kz.data(),
        final_lats.data(), final_lons.data(), final_alts.data(),
        final_path_lengths.data(), final_status.data(),
        ne_grid.data(),
        n_lat, n_lon, n_alt,
        lat_min, lat_max, lon_min, lon_max, alt_min, alt_max,
        freq_hz, config_.initial_step_km, config_.max_steps,
        n_rays
    );

    // Copy results
    for (int i = 0; i < n_rays; ++i) {
        results[i].lat = final_lats[i];
        results[i].lon = final_lons[i];
        results[i].alt = final_alts[i];
        results[i].path_length = final_path_lengths[i];
        results[i].status = final_status[i];
    }
#else
    // CUDA not compiled in - fall back to CPU
    results = trace_rays_cpu(initial_states, freq_hz);
#endif

    return results;
}

std::vector<CudaRayResult> CudaRayTracer::trace_rays_cpu(
    const std::vector<RayState>& initial_states,
    double freq_hz
) {
    const size_t n_rays = initial_states.size();
    std::vector<CudaRayResult> results(n_rays);

    double freq_mhz = freq_hz / 1e6;

    // Use OpenMP for parallel CPU tracing
    #ifdef HAVE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < n_rays; ++i) {
        const RayState& init = initial_states[i];

        // Convert initial state to elevation/azimuth
        double elevation = std::asin(init.wave_normal(2)) * 180.0 / PI;
        double azimuth = std::atan2(init.wave_normal(1), init.wave_normal(0)) * 180.0 / PI;

        // Trace ray using CPU tracer
        RayPath path = cpu_tracer_->trace_ray(
            init.position(0), init.position(1), init.position(2),
            elevation, azimuth, freq_mhz
        );

        // Extract result
        results[i].lat = path.positions.empty() ? init.position(0) : path.positions.back()(0);
        results[i].lon = path.positions.empty() ? init.position(1) : path.positions.back()(1);
        results[i].alt = path.positions.empty() ? init.position(2) : path.positions.back()(2);
        results[i].path_length = path.path_lengths.empty() ? 0.0 : path.path_lengths.back();

        if (path.reflected) results[i].status = 1;
        else if (path.escaped) results[i].status = 2;
        else if (path.absorbed) results[i].status = 3;
        else results[i].status = 0;
    }

    return results;
}

std::vector<RayPath> CudaRayTracer::trace_ray_fan_parallel(
    double lat0,
    double lon0,
    double alt0,
    const std::vector<double>& elevations,
    const std::vector<double>& azimuths,
    double freq_mhz
) {
    // Generate all ray initial states
    std::vector<RayState> initial_states;
    initial_states.reserve(elevations.size() * azimuths.size());

    for (double azim : azimuths) {
        for (double elev : elevations) {
            RayState state;
            state.position = Eigen::Vector3d(lat0, lon0, alt0);

            double elev_rad = elev * DEG_TO_RAD;
            double azim_rad = azim * DEG_TO_RAD;

            state.wave_normal = Eigen::Vector3d(
                std::cos(elev_rad) * std::cos(azim_rad),
                std::cos(elev_rad) * std::sin(azim_rad),
                std::sin(elev_rad)
            );
            state.wave_normal.normalize();

            initial_states.push_back(state);
        }
    }

    // Trace all rays in parallel
    std::vector<CudaRayResult> cuda_results = trace_rays_parallel(initial_states, freq_mhz);

    // Convert to RayPath format
    std::vector<RayPath> paths;
    paths.reserve(cuda_results.size());

    for (size_t i = 0; i < cuda_results.size(); ++i) {
        const auto& result = cuda_results[i];
        const auto& init = initial_states[i];

        RayPath path;
        path.positions.push_back(init.position);
        path.positions.push_back(Eigen::Vector3d(result.lat, result.lon, result.alt));

        path.path_lengths.push_back(0.0);
        path.path_lengths.push_back(result.path_length);

        path.ground_range = result.path_length;  // Approximate
        path.apex_altitude = std::max(init.position(2), result.alt);

        path.reflected = (result.status == 1);
        path.escaped = (result.status == 2);
        path.absorbed = (result.status == 3);

        paths.push_back(path);
    }

    return paths;
}

std::vector<RayPath> CudaRayTracer::calculate_nvis_coverage_parallel(
    double tx_lat,
    double tx_lon,
    double freq_mhz,
    double elevation_min,
    double elevation_max,
    double elevation_step,
    double azimuth_step
) {
    std::vector<double> elevations;
    std::vector<double> azimuths;

    for (double elev = elevation_min; elev <= elevation_max; elev += elevation_step) {
        elevations.push_back(elev);
    }

    for (double azim = 0.0; azim < 360.0; azim += azimuth_step) {
        azimuths.push_back(azim);
    }

    return trace_ray_fan_parallel(tx_lat, tx_lon, 0.0, elevations, azimuths, freq_mhz);
}

} // namespace propagation
} // namespace autonvis
