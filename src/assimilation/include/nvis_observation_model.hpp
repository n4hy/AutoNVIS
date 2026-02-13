/**
 * @file nvis_observation_model.hpp
 * @brief NVIS sounder observation model for SR-UKF
 *
 * Models NVIS (Near Vertical Incidence Skywave) propagation observations:
 * - Signal strength (dBm)
 * - Group delay (ms)
 *
 * Uses simplified forward model with vertical Ne profile extraction
 * and obliquity factor for near-vertical propagation.
 */

#ifndef NVIS_OBSERVATION_MODEL_HPP
#define NVIS_OBSERVATION_MODEL_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "state_vector.hpp"
#include "observation_model.hpp"

namespace autonvis {

/**
 * @brief NVIS sounder observation model
 *
 * Forward model for NVIS propagation:
 * 1. Extract vertical Ne profile at midpoint
 * 2. Find reflection height (f_plasma = frequency)
 * 3. Compute path loss (free space + D-region absorption)
 * 4. Compute group delay with obliquity factor
 */
class NVISSounderObservationModel : public ObservationModel {
public:
    /**
     * @brief NVIS measurement structure
     */
    struct NVISMeasurement {
        // Transmitter location
        double tx_latitude;   ///< Tx latitude (deg)
        double tx_longitude;  ///< Tx longitude (deg)
        double tx_altitude;   ///< Tx altitude (m)

        // Receiver location
        double rx_latitude;   ///< Rx latitude (deg)
        double rx_longitude;  ///< Rx longitude (deg)
        double rx_altitude;   ///< Rx altitude (m)

        // Propagation parameters
        double frequency;         ///< Frequency (MHz)
        double elevation_angle;   ///< Elevation angle (deg) [70-90°]
        double azimuth;          ///< Azimuth (deg)
        double hop_distance;     ///< Hop distance (km)

        // Observables
        double signal_strength;   ///< Signal strength (dBm)
        double group_delay;       ///< Group delay (ms)
        double snr;              ///< Signal-to-noise ratio (dB)

        // Quality/error (from quality assessor)
        double signal_strength_error;  ///< Signal error (dB)
        double group_delay_error;      ///< Delay error (ms)

        // Mode
        bool is_o_mode;          ///< true=Ordinary, false=Extraordinary

        // Optional metadata
        double tx_power;         ///< Tx power (W) [optional]
        double tx_antenna_gain;  ///< Tx antenna gain (dBi) [optional]
        double rx_antenna_gain;  ///< Rx antenna gain (dBi) [optional]
    };

    /**
     * @brief Constructor
     * @param measurements Vector of NVIS measurements
     * @param lat_grid Latitude grid (deg)
     * @param lon_grid Longitude grid (deg)
     * @param alt_grid Altitude grid (km)
     */
    NVISSounderObservationModel(
        const std::vector<NVISMeasurement>& measurements,
        const std::vector<double>& lat_grid,
        const std::vector<double>& lon_grid,
        const std::vector<double>& alt_grid
    );

    /**
     * @brief Forward model: state → predicted observations
     * @param state Current state vector
     * @return Predicted observation vector [signal_1, ..., signal_N, delay_1, ..., delay_N]
     */
    Eigen::VectorXd forward(const StateVector& state) const override;

    /**
     * @brief Get observation dimension
     * @return Number of observations (2 × measurements for signal + delay)
     */
    size_t obs_dimension() const override { return measurements_.size() * 2; }

    /**
     * @brief Predict signal strength for one measurement (simplified model)
     * @param meas NVIS measurement
     * @param state Current state
     * @return Predicted signal strength (dBm)
     */
    double predict_signal_strength_simplified(
        const NVISMeasurement& meas,
        const StateVector& state
    ) const;

    /**
     * @brief Predict group delay for one measurement (simplified model)
     * @param meas NVIS measurement
     * @param state Current state
     * @return Predicted group delay (ms)
     */
    double predict_group_delay_simplified(
        const NVISMeasurement& meas,
        const StateVector& state
    ) const;

private:
    std::vector<NVISMeasurement> measurements_;
    std::vector<double> lat_grid_;
    std::vector<double> lon_grid_;
    std::vector<double> alt_grid_;

    /**
     * @brief Get vertical Ne profile at location
     * @param lat Latitude (deg)
     * @param lon Longitude (deg)
     * @param state Current state
     * @return Ne profile vector (el/m³) at each altitude
     */
    Eigen::VectorXd get_vertical_profile(
        double lat,
        double lon,
        const StateVector& state
    ) const;

    /**
     * @brief Find reflection height where f_plasma = frequency
     * @param ne_profile Vertical Ne profile (el/m³)
     * @param frequency Frequency (MHz)
     * @return Reflection height (km), or -1 if no reflection
     */
    double find_reflection_height(
        const Eigen::VectorXd& ne_profile,
        double frequency
    ) const;

    /**
     * @brief Compute free space path loss
     * @param frequency Frequency (MHz)
     * @param distance Distance (km)
     * @return Path loss (dB)
     */
    double compute_free_space_loss(
        double frequency,
        double distance
    ) const;

    /**
     * @brief Compute D-region absorption
     * @param frequency Frequency (MHz)
     * @param distance Path through D-region (km)
     * @param elevation_angle Elevation angle (deg)
     * @return Absorption loss (dB)
     */
    double compute_d_region_absorption(
        double frequency,
        double distance,
        double elevation_angle
    ) const;

    /**
     * @brief Compute obliquity factor for near-vertical propagation
     * @param elevation_angle Elevation angle (deg)
     * @return Obliquity factor (dimensionless)
     */
    double compute_obliquity_factor(
        double elevation_angle
    ) const;

    /**
     * @brief Convert Ne to plasma frequency
     * @param ne Electron density (el/m³)
     * @return Plasma frequency (Hz)
     */
    double ne_to_plasma_freq(double ne) const;

    /**
     * @brief Interpolate Ne at arbitrary lat/lon/alt
     * @param lat Latitude (deg)
     * @param lon Longitude (deg)
     * @param alt Altitude (km)
     * @param state Current state
     * @return Interpolated Ne (el/m³)
     */
    double interpolate_ne(
        double lat,
        double lon,
        double alt,
        const StateVector& state
    ) const;

    /**
     * @brief Compute great circle distance
     * @param lat1 Latitude 1 (deg)
     * @param lon1 Longitude 1 (deg)
     * @param lat2 Latitude 2 (deg)
     * @param lon2 Longitude 2 (deg)
     * @return Distance (km)
     */
    double haversine_distance(
        double lat1,
        double lon1,
        double lat2,
        double lon2
    ) const;

    // Physical constants
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;  ///< C
    static constexpr double ELECTRON_MASS = 9.10938356e-31;     ///< kg
    static constexpr double EPSILON_0 = 8.8541878128e-12;       ///< F/m
    static constexpr double SPEED_OF_LIGHT = 299792458.0;       ///< m/s
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double EARTH_RADIUS = 6371.0;              ///< km

    // Model parameters
    static constexpr double D_REGION_HEIGHT = 70.0;             ///< km
    static constexpr double D_REGION_THICKNESS = 20.0;          ///< km
    static constexpr double D_REGION_ABSORPTION_COEFF = 0.1;    ///< dB/km/MHz
};

} // namespace autonvis

#endif // NVIS_OBSERVATION_MODEL_HPP
