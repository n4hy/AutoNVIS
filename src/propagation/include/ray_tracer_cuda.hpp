/**
 * @file ray_tracer_cuda.hpp
 * @brief CUDA-accelerated ray tracing interface
 *
 * Phase 18.4: GPU-accelerated ray tracing with automatic CPU fallback.
 * Provides unified interface that works with or without CUDA.
 */

#ifndef RAY_TRACER_CUDA_HPP
#define RAY_TRACER_CUDA_HPP

#include "ray_tracer_3d.hpp"
#include <vector>
#include <memory>
#include <string>

namespace autonvis {
namespace propagation {

/**
 * @brief CUDA ray tracing results
 */
struct CudaRayResult {
    double lat, lon, alt;      // Final position
    double path_length;        // Total path length (km)
    int status;                // 0=propagating, 1=reflected, 2=escaped, 3=absorbed
};

/**
 * @brief CUDA-accelerated ray tracer
 *
 * Provides GPU-accelerated ray tracing with automatic fallback to CPU
 * when CUDA is not available.
 */
class CudaRayTracer {
public:
    /**
     * @brief Constructor
     *
     * @param iono_grid Ionospheric electron density grid
     * @param geomag Geomagnetic field model
     * @param config Ray tracing configuration
     */
    CudaRayTracer(
        std::shared_ptr<IonoGrid> iono_grid,
        std::shared_ptr<GeomagneticField> geomag,
        const RayTracingConfig& config = RayTracingConfig()
    );

    /**
     * @brief Check if CUDA is available and initialized
     */
    bool cuda_available() const { return cuda_available_; }

    /**
     * @brief Get CUDA device name
     */
    std::string device_name() const { return device_name_; }

    /**
     * @brief Trace multiple rays in parallel
     *
     * Uses CUDA if available, otherwise falls back to OpenMP-parallelized CPU.
     *
     * @param initial_states Vector of initial ray states
     * @return Vector of ray results
     */
    std::vector<CudaRayResult> trace_rays_parallel(
        const std::vector<RayState>& initial_states,
        double freq_mhz
    );

    /**
     * @brief Trace ray fan in parallel
     *
     * @param lat0 Initial latitude (degrees)
     * @param lon0 Initial longitude (degrees)
     * @param alt0 Initial altitude (km)
     * @param elevations Vector of elevation angles
     * @param azimuths Vector of azimuth angles
     * @param freq_mhz Frequency (MHz)
     * @return Vector of ray paths
     */
    std::vector<RayPath> trace_ray_fan_parallel(
        double lat0,
        double lon0,
        double alt0,
        const std::vector<double>& elevations,
        const std::vector<double>& azimuths,
        double freq_mhz
    );

    /**
     * @brief Calculate NVIS coverage in parallel
     *
     * @param tx_lat Transmitter latitude
     * @param tx_lon Transmitter longitude
     * @param freq_mhz Frequency
     * @param elevation_min Minimum elevation (degrees)
     * @param elevation_max Maximum elevation (degrees)
     * @param elevation_step Elevation step (degrees)
     * @param azimuth_step Azimuth step (degrees)
     * @return Vector of ray paths
     */
    std::vector<RayPath> calculate_nvis_coverage_parallel(
        double tx_lat,
        double tx_lon,
        double freq_mhz,
        double elevation_min = 70.0,
        double elevation_max = 90.0,
        double elevation_step = 2.0,
        double azimuth_step = 15.0
    );

    /**
     * @brief Get performance statistics
     */
    struct PerfStats {
        double last_trace_time_ms;
        int rays_per_second;
        bool used_cuda;
    };

    PerfStats get_stats() const { return stats_; }

private:
    std::shared_ptr<IonoGrid> iono_grid_;
    std::shared_ptr<GeomagneticField> geomag_;
    RayTracingConfig config_;

    bool cuda_available_;
    std::string device_name_;
    PerfStats stats_;

    // CPU fallback ray tracer
    std::unique_ptr<RayTracer3D> cpu_tracer_;

    // Initialize CUDA
    void init_cuda();

    // Trace rays using CUDA
    std::vector<CudaRayResult> trace_rays_cuda(
        const std::vector<RayState>& initial_states,
        double freq_hz
    );

    // Trace rays using CPU (OpenMP parallel)
    std::vector<CudaRayResult> trace_rays_cpu(
        const std::vector<RayState>& initial_states,
        double freq_hz
    );

    // Convert Ne grid to flat array for CUDA
    std::vector<double> flatten_ne_grid() const;
};

} // namespace propagation
} // namespace autonvis

#endif // RAY_TRACER_CUDA_HPP
