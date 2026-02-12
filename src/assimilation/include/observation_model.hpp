/**
 * @file observation_model.hpp
 * @brief Observation models for TEC and ionosonde measurements
 *
 * Provides forward models to convert state (electron density) to
 * observable quantities (TEC, foF2, hmF2).
 */

#ifndef OBSERVATION_MODEL_HPP
#define OBSERVATION_MODEL_HPP

#include <Eigen/Dense>
#include <vector>
#include "state_vector.hpp"

namespace autonvis {

/**
 * @brief Abstract base class for observation models
 */
class ObservationModel {
public:
    virtual ~ObservationModel() = default;

    /**
     * @brief Forward model: state → observation
     * @param state Current state vector
     * @return Predicted observation
     */
    virtual Eigen::VectorXd forward(const StateVector& state) const = 0;

    /**
     * @brief Get observation dimension
     * @return Number of observation variables
     */
    virtual size_t obs_dimension() const = 0;
};

/**
 * @brief TEC observation model
 *
 * Models Total Electron Content as slant path integral of Ne:
 * TEC = ∫ Ne(s) ds  [electrons/m²]
 */
class TECObservationModel : public ObservationModel {
public:
    /**
     * @brief TEC measurement structure
     */
    struct TECMeasurement {
        double latitude;      ///< Receiver latitude (deg)
        double longitude;     ///< Receiver longitude (deg)
        double altitude;      ///< Receiver altitude (km)
        double sat_latitude;  ///< Satellite latitude (deg)
        double sat_longitude; ///< Satellite longitude (deg)
        double sat_altitude;  ///< Satellite altitude (km)
        double azimuth;       ///< Azimuth angle (deg)
        double elevation;     ///< Elevation angle (deg)
        double tec_value;     ///< Measured TEC (TECU)
        double tec_error;     ///< TEC measurement error (TECU)
    };

    /**
     * @brief Constructor
     * @param measurements Vector of TEC measurements
     * @param lat_grid Latitude grid (deg)
     * @param lon_grid Longitude grid (deg)
     * @param alt_grid Altitude grid (km)
     */
    TECObservationModel(
        const std::vector<TECMeasurement>& measurements,
        const std::vector<double>& lat_grid,
        const std::vector<double>& lon_grid,
        const std::vector<double>& alt_grid
    );

    Eigen::VectorXd forward(const StateVector& state) const override;
    size_t obs_dimension() const override { return measurements_.size(); }

    /**
     * @brief Compute slant path TEC for one measurement
     * @param meas TEC measurement
     * @param state Current state
     * @return Predicted TEC (TECU)
     */
    double compute_slant_tec(
        const TECMeasurement& meas,
        const StateVector& state
    ) const;

private:
    std::vector<TECMeasurement> measurements_;
    std::vector<double> lat_grid_;
    std::vector<double> lon_grid_;
    std::vector<double> alt_grid_;

    static constexpr int N_INTEGRATION_STEPS = 100;
    static constexpr double TECU_TO_ELECTRONS_M2 = 1e16;
};

/**
 * @brief Ionosonde observation model
 *
 * Models critical frequency and peak height:
 * - foF2: Peak F2 layer frequency (MHz)
 * - hmF2: Peak F2 layer height (km)
 */
class IonosondeObservationModel : public ObservationModel {
public:
    /**
     * @brief Ionosonde measurement structure
     */
    struct IonosondeMeasurement {
        double latitude;   ///< Station latitude (deg)
        double longitude;  ///< Station longitude (deg)
        double fof2;       ///< Critical frequency (MHz)
        double hmf2;       ///< Peak height (km)
        double fof2_error; ///< foF2 error (MHz)
        double hmf2_error; ///< hmF2 error (km)
    };

    /**
     * @brief Constructor
     * @param measurements Vector of ionosonde measurements
     * @param lat_grid Latitude grid (deg)
     * @param lon_grid Longitude grid (deg)
     * @param alt_grid Altitude grid (km)
     */
    IonosondeObservationModel(
        const std::vector<IonosondeMeasurement>& measurements,
        const std::vector<double>& lat_grid,
        const std::vector<double>& lon_grid,
        const std::vector<double>& alt_grid
    );

    Eigen::VectorXd forward(const StateVector& state) const override;
    size_t obs_dimension() const override { return measurements_.size() * 2; }  // foF2 + hmF2

    /**
     * @brief Extract foF2 and hmF2 from vertical Ne profile
     * @param meas Ionosonde measurement
     * @param state Current state
     * @return (foF2, hmF2) pair
     */
    std::pair<double, double> extract_f2_parameters(
        const IonosondeMeasurement& meas,
        const StateVector& state
    ) const;

private:
    std::vector<IonosondeMeasurement> measurements_;
    std::vector<double> lat_grid_;
    std::vector<double> lon_grid_;
    std::vector<double> alt_grid_;

    /**
     * @brief Convert Ne to plasma frequency
     * @param ne Electron density (el/m³)
     * @return Plasma frequency (Hz)
     */
    double ne_to_plasma_freq(double ne) const;

    // Physical constants
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;  // C
    static constexpr double ELECTRON_MASS = 9.10938356e-31;     // kg
    static constexpr double EPSILON_0 = 8.8541878128e-12;       // F/m
};

} // namespace autonvis

#endif // OBSERVATION_MODEL_HPP
