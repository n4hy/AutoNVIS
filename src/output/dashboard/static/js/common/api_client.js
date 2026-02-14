/**
 * API Client for AutoNVIS Dashboard
 *
 * Provides methods to interact with all REST API endpoints.
 */

class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    /**
     * Generic fetch wrapper with error handling
     */
    async _fetch(url, options = {}) {
        try {
            const response = await fetch(this.baseURL + url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...(options.headers || {})
                }
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error (${url}):`, error);
            throw error;
        }
    }

    // === Ionosphere API ===

    async getGridMetadata() {
        return this._fetch('/api/ionosphere/grid/metadata');
    }

    async getFullGrid(maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/grid/full?max_age_seconds=${maxAgeSeconds}`);
    }

    async getHorizontalSlice(altitudeKm, maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/slice/horizontal?altitude_km=${altitudeKm}&max_age_seconds=${maxAgeSeconds}`);
    }

    async getVerticalProfile(latitude, longitude, maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/profile/vertical?latitude=${latitude}&longitude=${longitude}&max_age_seconds=${maxAgeSeconds}`);
    }

    async getFoF2Map(maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/parameters/fof2?max_age_seconds=${maxAgeSeconds}`);
    }

    async getHmF2Map(maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/parameters/hmf2?max_age_seconds=${maxAgeSeconds}`);
    }

    async getTECMap(maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/parameters/tec?max_age_seconds=${maxAgeSeconds}`);
    }

    async getGridStatistics(maxAgeSeconds = 1200) {
        return this._fetch(`/api/ionosphere/statistics?max_age_seconds=${maxAgeSeconds}`);
    }

    async getGridHistory(limit = 100) {
        return this._fetch(`/api/ionosphere/history/grids?limit=${limit}`);
    }

    // === Propagation API ===

    async getLatestFrequencyPlan() {
        return this._fetch('/api/propagation/frequency_plan/latest');
    }

    async getFrequencyPlanHistory(hours = 24) {
        return this._fetch(`/api/propagation/frequency_plan/history?hours=${hours}`);
    }

    async getLatestCoverageMap() {
        return this._fetch('/api/propagation/coverage_map/latest');
    }

    async getLUFData(hours = 24) {
        return this._fetch(`/api/propagation/luf?hours=${hours}`);
    }

    async getMUFData(hours = 24) {
        return this._fetch(`/api/propagation/muf?hours=${hours}`);
    }

    async getFOTData(hours = 24) {
        return this._fetch(`/api/propagation/fot?hours=${hours}`);
    }

    async getPropagationStatistics() {
        return this._fetch('/api/propagation/statistics');
    }

    // === Space Weather API ===

    async getXRayHistory(hours = 24) {
        return this._fetch(`/api/spaceweather/xray/history?hours=${hours}`);
    }

    async getXRayLatest() {
        return this._fetch('/api/spaceweather/xray/latest');
    }

    async getSolarWindLatest() {
        return this._fetch('/api/spaceweather/solar_wind/latest');
    }

    async getSolarWindHistory(hours = 24) {
        return this._fetch(`/api/spaceweather/solar_wind/history?hours=${hours}`);
    }

    async getCurrentMode() {
        return this._fetch('/api/spaceweather/mode/current');
    }

    async getModeHistory(hours = 24) {
        return this._fetch(`/api/spaceweather/mode/history?hours=${hours}`);
    }

    async getRecentAlerts(hours = 24, severity = null) {
        const params = new URLSearchParams({ hours: hours.toString() });
        if (severity) params.append('severity', severity);
        return this._fetch(`/api/spaceweather/alerts/recent?${params}`);
    }

    async getSpaceWeatherStatistics() {
        return this._fetch('/api/spaceweather/statistics');
    }

    // === Control API ===

    async getServicesStatus() {
        return this._fetch('/api/control/services/status');
    }

    async controlService(serviceName, action) {
        return this._fetch('/api/control/services/control', {
            method: 'POST',
            body: JSON.stringify({ service_name: serviceName, action: action })
        });
    }

    async triggerFilterCycle(force = false, reason = 'Manual trigger from dashboard') {
        return this._fetch('/api/control/filter/trigger_cycle', {
            method: 'POST',
            body: JSON.stringify({ force, reason })
        });
    }

    async updateFilterParameters(params) {
        return this._fetch('/api/control/filter/update_parameters', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    async getFilterMetrics() {
        return this._fetch('/api/control/filter/metrics');
    }

    async switchMode(mode, reason = 'Manual override from dashboard') {
        return this._fetch('/api/control/mode/switch', {
            method: 'POST',
            body: JSON.stringify({ mode, reason })
        });
    }

    async toggleDataSource(sourceName, enabled) {
        return this._fetch('/api/control/datasource/toggle', {
            method: 'POST',
            body: JSON.stringify({ source_name: sourceName, enabled })
        });
    }

    async getObservationCounts() {
        return this._fetch('/api/control/observations/counts');
    }

    async getRecentObservations(hours = 1, obsType = null) {
        const params = new URLSearchParams({ hours: hours.toString() });
        if (obsType) params.append('obs_type', obsType);
        return this._fetch(`/api/control/observations/recent?${params}`);
    }

    async getSystemStatistics() {
        return this._fetch('/api/control/system/statistics');
    }

    async healthCheck() {
        return this._fetch('/api/control/health/check');
    }

    // === NVIS Network API (Legacy) ===

    async getNVISSounders() {
        return this._fetch('/api/nvis/sounders');
    }

    async getNVISSounder(sounderId) {
        return this._fetch(`/api/nvis/sounder/${sounderId}`);
    }

    async getNetworkAnalysis() {
        return this._fetch('/api/nvis/network/analysis');
    }

    async getPlacementRecommendations(nSounders = 3, tier = 'gold') {
        return this._fetch(`/api/nvis/placement/recommend?n_sounders=${nSounders}&tier=${tier}`);
    }

    async simulatePlacement(latitude, longitude, tier = 'gold') {
        return this._fetch('/api/nvis/placement/simulate', {
            method: 'POST',
            body: JSON.stringify({ latitude, longitude, tier })
        });
    }

    async getPlacementHeatmap(resolution = 50) {
        return this._fetch(`/api/nvis/placement/heatmap?resolution=${resolution}`);
    }
}

// Create global API client instance
const api = new APIClient();
