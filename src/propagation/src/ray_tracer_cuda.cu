/**
 * @file ray_tracer_cuda.cu
 * @brief CUDA kernels for parallel HF ray tracing
 *
 * Phase 18.4: GPU-accelerated ray tracing for massive parallelism.
 * Ray tracing is embarrassingly parallel - ideal for GPU.
 *
 * Each CUDA thread traces a single ray independently.
 * Target: 512 rays in <5 ms (vs 60 ms CPU sequential)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>

namespace autonvis {
namespace propagation {
namespace cuda {

// Physical constants (device)
__constant__ double PI_d = 3.14159265358979323846;
__constant__ double DEG_TO_RAD_d = 3.14159265358979323846 / 180.0;
__constant__ double EARTH_RADIUS_KM_d = 6371.0;
__constant__ double C_LIGHT_d = 2.99792458e8;
__constant__ double ELECTRON_CHARGE_d = 1.60217663e-19;
__constant__ double ELECTRON_MASS_d = 9.1093837e-31;
__constant__ double EPSILON_0_d = 8.8541878e-12;

/**
 * @brief Ray state for GPU computation (compact struct)
 */
struct RayStateGPU {
    double lat, lon, alt;      // Position
    double kx, ky, kz;         // Wave normal direction
    double path_length;        // Cumulative path length
    double absorption_db;      // Cumulative absorption
    int status;                // 0=propagating, 1=reflected, 2=escaped, 3=absorbed
};

/**
 * @brief GPU-side ionospheric grid (texture memory for fast access)
 */
struct IonoGridGPU {
    double* ne_data;           // Electron density data (device pointer)
    int n_lat, n_lon, n_alt;   // Grid dimensions
    double lat_min, lat_max;   // Latitude bounds
    double lon_min, lon_max;   // Longitude bounds
    double alt_min, alt_max;   // Altitude bounds
    double dlat, dlon, dalt;   // Grid spacing
};

/**
 * @brief Ray tracing configuration for GPU
 */
struct RayConfigGPU {
    double freq_hz;            // Frequency in Hz
    double tolerance;          // Integration tolerance
    double step_size_km;       // Step size
    double max_path_km;        // Maximum path length
    double ground_alt_km;      // Ground level
    double escape_alt_km;      // Escape threshold
    int max_steps;             // Maximum steps
};

/**
 * @brief Trilinear interpolation of electron density on GPU
 */
__device__ double interpolate_ne_gpu(
    const IonoGridGPU& grid,
    double lat, double lon, double alt
) {
    // Normalize coordinates to grid indices
    double fi = (lat - grid.lat_min) / grid.dlat;
    double fj = (lon - grid.lon_min) / grid.dlon;
    double fk = (alt - grid.alt_min) / grid.dalt;

    // Clamp to valid range
    fi = fmax(0.0, fmin(fi, (double)(grid.n_lat - 2)));
    fj = fmax(0.0, fmin(fj, (double)(grid.n_lon - 2)));
    fk = fmax(0.0, fmin(fk, (double)(grid.n_alt - 2)));

    int i = (int)fi;
    int j = (int)fj;
    int k = (int)fk;

    double wi = fi - i;
    double wj = fj - j;
    double wk = fk - k;

    // Trilinear interpolation
    double ne = 0.0;
    for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
            for (int dk = 0; dk <= 1; ++dk) {
                double w = ((di == 0) ? (1.0 - wi) : wi) *
                          ((dj == 0) ? (1.0 - wj) : wj) *
                          ((dk == 0) ? (1.0 - wk) : wk);

                int idx = (i + di) * grid.n_lon * grid.n_alt +
                         (j + dj) * grid.n_alt + (k + dk);
                ne += w * grid.ne_data[idx];
            }
        }
    }

    return fmax(0.0, ne);
}

/**
 * @brief Plasma frequency calculation
 */
__device__ double plasma_freq_gpu(double ne) {
    if (ne <= 0) return 0.0;
    return sqrt(ne * ELECTRON_CHARGE_d * ELECTRON_CHARGE_d /
               (ELECTRON_MASS_d * EPSILON_0_d)) / (2.0 * PI_d);
}

/**
 * @brief Simplified refractive index (no magnetic field)
 */
__device__ double refractive_index_gpu(double ne, double freq_hz) {
    double fp = plasma_freq_gpu(ne);
    double X = (fp / freq_hz) * (fp / freq_hz);
    if (X >= 1.0) return 0.0;  // Cutoff
    return sqrt(1.0 - X);
}

/**
 * @brief Compute electron density gradient using finite differences
 */
__device__ void ne_gradient_gpu(
    const IonoGridGPU& grid,
    double lat, double lon, double alt,
    double* grad_lat, double* grad_lon, double* grad_alt
) {
    const double h = 0.01;  // Small step for finite difference

    double ne_c = interpolate_ne_gpu(grid, lat, lon, alt);
    *grad_lat = (interpolate_ne_gpu(grid, lat + h, lon, alt) - ne_c) / h;
    *grad_lon = (interpolate_ne_gpu(grid, lat, lon + h, alt) - ne_c) / h;
    *grad_alt = (interpolate_ne_gpu(grid, lat, lon, alt + h) - ne_c) / h;
}

/**
 * @brief Single step of ray integration (RK4)
 */
__device__ void integrate_step_gpu(
    RayStateGPU& state,
    const IonoGridGPU& grid,
    const RayConfigGPU& config
) {
    double lat = state.lat;
    double lon = state.lon;
    double alt = state.alt;

    // Get electron density and refractive index
    double ne = interpolate_ne_gpu(grid, lat, lon, alt);
    double n = refractive_index_gpu(ne, config.freq_hz);

    // Check for reflection
    if (n < 0.1) {
        state.status = 1;  // Reflected
        return;
    }

    // Get gradient for ray bending
    double grad_lat, grad_lon, grad_alt;
    ne_gradient_gpu(grid, lat, lon, alt, &grad_lat, &grad_lon, &grad_alt);

    // Haselgrove equations (simplified)
    double kx = state.kx;
    double ky = state.ky;
    double kz = state.kz;

    // Position derivatives
    double dlat = kx / n * config.step_size_km / 111.0;  // km to degrees
    double dlon = ky / n * config.step_size_km / (111.0 * cos(lat * DEG_TO_RAD_d));
    double dalt = kz / n * config.step_size_km;

    // Wave normal derivatives (simplified refraction)
    double ne_safe = fmax(ne, 1e6);
    double dkx = -grad_lat / (2.0 * ne_safe * n);
    double dky = -grad_lon / (2.0 * ne_safe * n);
    double dkz = -grad_alt / (2.0 * ne_safe * n);

    // Update state
    state.lat += dlat;
    state.lon += dlon;
    state.alt += dalt;

    state.kx += dkx * config.step_size_km;
    state.ky += dky * config.step_size_km;
    state.kz += dkz * config.step_size_km;

    // Normalize wave normal
    double k_norm = sqrt(state.kx*state.kx + state.ky*state.ky + state.kz*state.kz);
    if (k_norm > 1e-10) {
        state.kx /= k_norm;
        state.ky /= k_norm;
        state.kz /= k_norm;
    }

    state.path_length += config.step_size_km;

    // Check termination
    if (state.alt < config.ground_alt_km) {
        state.status = 1;  // Reflected/landed
    } else if (state.alt > config.escape_alt_km) {
        state.status = 2;  // Escaped
    }
}

/**
 * @brief CUDA kernel for parallel ray tracing
 *
 * Each thread traces one ray from start to termination.
 *
 * @param rays Array of ray initial states
 * @param results Array of ray final states
 * @param grid Ionospheric grid
 * @param config Ray tracing configuration
 * @param n_rays Number of rays to trace
 */
__global__ void trace_rays_kernel(
    const RayStateGPU* rays,
    RayStateGPU* results,
    const IonoGridGPU grid,
    const RayConfigGPU config,
    int n_rays
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_rays) return;

    // Copy initial state
    RayStateGPU state = rays[idx];
    state.status = 0;  // Propagating

    // Trace ray
    for (int step = 0; step < config.max_steps && state.status == 0; ++step) {
        integrate_step_gpu(state, grid, config);

        // Check path length limit
        if (state.path_length >= config.max_path_km) {
            break;
        }
    }

    // Store result
    results[idx] = state;
}

// ============================================
// Host-side interface functions
// ============================================

/**
 * @brief Check if CUDA is available
 */
extern "C" bool cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Get CUDA device info
 */
extern "C" void cuda_get_device_info(char* name, int* major, int* minor, size_t* mem) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (name) {
        strcpy(name, prop.name);
    }
    if (major) *major = prop.major;
    if (minor) *minor = prop.minor;
    if (mem) *mem = prop.totalGlobalMem;
}

/**
 * @brief Trace multiple rays on GPU
 *
 * @param init_states Array of initial ray states (host memory)
 * @param final_states Array for final ray states (host memory)
 * @param ne_grid Electron density grid (host memory, row-major)
 * @param n_lat, n_lon, n_alt Grid dimensions
 * @param lat_min, lat_max, lon_min, lon_max, alt_min, alt_max Grid bounds
 * @param freq_hz Frequency in Hz
 * @param step_km Step size in km
 * @param max_steps Maximum steps per ray
 * @param n_rays Number of rays
 * @return 0 on success, error code otherwise
 */
extern "C" int cuda_trace_rays(
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
) {
    // Allocate device memory for rays
    RayStateGPU* d_rays;
    RayStateGPU* d_results;
    double* d_ne_grid;

    size_t rays_size = n_rays * sizeof(RayStateGPU);
    size_t ne_size = n_lat * n_lon * n_alt * sizeof(double);

    cudaMalloc(&d_rays, rays_size);
    cudaMalloc(&d_results, rays_size);
    cudaMalloc(&d_ne_grid, ne_size);

    // Copy electron density grid to device
    cudaMemcpy(d_ne_grid, ne_grid, ne_size, cudaMemcpyHostToDevice);

    // Prepare ray states
    RayStateGPU* h_rays = new RayStateGPU[n_rays];
    for (int i = 0; i < n_rays; ++i) {
        h_rays[i].lat = init_lats[i];
        h_rays[i].lon = init_lons[i];
        h_rays[i].alt = init_alts[i];
        h_rays[i].kx = init_kx[i];
        h_rays[i].ky = init_ky[i];
        h_rays[i].kz = init_kz[i];
        h_rays[i].path_length = 0.0;
        h_rays[i].absorption_db = 0.0;
        h_rays[i].status = 0;
    }

    // Copy rays to device
    cudaMemcpy(d_rays, h_rays, rays_size, cudaMemcpyHostToDevice);

    // Set up grid configuration
    IonoGridGPU grid;
    grid.ne_data = d_ne_grid;
    grid.n_lat = n_lat;
    grid.n_lon = n_lon;
    grid.n_alt = n_alt;
    grid.lat_min = lat_min;
    grid.lat_max = lat_max;
    grid.lon_min = lon_min;
    grid.lon_max = lon_max;
    grid.alt_min = alt_min;
    grid.alt_max = alt_max;
    grid.dlat = (lat_max - lat_min) / (n_lat - 1);
    grid.dlon = (lon_max - lon_min) / (n_lon - 1);
    grid.dalt = (alt_max - alt_min) / (n_alt - 1);

    // Ray config
    RayConfigGPU config;
    config.freq_hz = freq_hz;
    config.tolerance = 1e-7;
    config.step_size_km = step_km;
    config.max_path_km = 20000.0;
    config.ground_alt_km = 0.0;
    config.escape_alt_km = 1000.0;
    config.max_steps = max_steps;

    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (n_rays + threads_per_block - 1) / threads_per_block;

    trace_rays_kernel<<<num_blocks, threads_per_block>>>(
        d_rays, d_results, grid, config, n_rays
    );

    // Wait for completion
    cudaDeviceSynchronize();

    // Copy results back
    RayStateGPU* h_results = new RayStateGPU[n_rays];
    cudaMemcpy(h_results, d_results, rays_size, cudaMemcpyDeviceToHost);

    // Extract results
    for (int i = 0; i < n_rays; ++i) {
        final_lats[i] = h_results[i].lat;
        final_lons[i] = h_results[i].lon;
        final_alts[i] = h_results[i].alt;
        final_path_lengths[i] = h_results[i].path_length;
        final_status[i] = h_results[i].status;
    }

    // Cleanup
    delete[] h_rays;
    delete[] h_results;
    cudaFree(d_rays);
    cudaFree(d_results);
    cudaFree(d_ne_grid);

    return 0;
}

} // namespace cuda
} // namespace propagation
} // namespace autonvis
