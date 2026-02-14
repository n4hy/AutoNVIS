/**
 * Ionosphere Visualization JavaScript
 *
 * Handles all ionospheric data visualization including:
 * - 2D horizontal slices
 * - Vertical profiles
 * - Parameter maps (foF2, hmF2, TEC)
 */

// Current state
let currentAltitude = 300; // km
let currentLat = 40.0;
let currentLon = -105.0;
let gridMetadata = null;

/**
 * Initialize ionosphere view
 */
async function initIonosphereView() {
    // Setup altitude slider
    const altitudeSlider = document.getElementById('altitude-slider');
    altitudeSlider.addEventListener('input', (e) => {
        currentAltitude = parseFloat(e.target.value);
        document.getElementById('altitude-value').textContent = currentAltitude;
        document.getElementById('slice-altitude').textContent = currentAltitude;
        updateHorizontalSlice();
    });

    // Load all data
    await loadAllData();

    // Setup auto-refresh (every 60 seconds)
    setInterval(loadGridMetadata, 60000);
}

/**
 * Load all ionosphere data
 */
async function loadAllData() {
    try {
        await loadGridMetadata();
        await Promise.all([
            updateHorizontalSlice(),
            updateVerticalProfile(),
            updateFoF2Map(),
            updateHmF2Map(),
            updateTECMap(),
            updateStatistics()
        ]);
    } catch (error) {
        console.error('Error loading ionosphere data:', error);
        Utils.showNotification('Failed to load ionosphere data', 'error');
    }
}

/**
 * Load grid metadata
 */
async function loadGridMetadata() {
    try {
        gridMetadata = await api.getGridMetadata();

        // Update status cards
        document.getElementById('grid-ne-max').textContent =
            (gridMetadata.ne_max / 1e11).toFixed(2);
        document.getElementById('grid-quality').textContent = gridMetadata.quality;
        document.getElementById('grid-age').textContent =
            Utils.formatRelativeTime(new Date(Date.now() - gridMetadata.age_seconds * 1000));

        // Note: observations_used may not be in metadata, will add if available
        const obsCount = gridMetadata.observations_used || '--';
        document.getElementById('grid-obs-count').textContent = obsCount;

    } catch (error) {
        console.error('Error loading grid metadata:', error);
    }
}

/**
 * Update horizontal slice visualization
 */
async function updateHorizontalSlice() {
    try {
        const sliceData = await api.getHorizontalSlice(currentAltitude);

        const trace = {
            type: 'heatmap',
            x: sliceData.longitude,
            y: sliceData.latitude,
            z: sliceData.ne_values,
            colorscale: 'Viridis',
            colorbar: {
                title: 'Ne (el/m³)',
                exponentformat: 'e'
            },
            hovertemplate: 'Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>Ne: %{z:.2e} el/m³<extra></extra>'
        };

        const layout = {
            ...PlotlyDefaultLayout,
            title: `Electron Density at ${sliceData.actual_altitude_km.toFixed(0)} km`,
            xaxis: { title: 'Longitude (°)' },
            yaxis: { title: 'Latitude (°)' },
            margin: { l: 60, r: 80, t: 60, b: 60 }
        };

        Plotly.newPlot('ne-slice-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating horizontal slice:', error);
        Utils.showError('ne-slice-plot', 'Failed to load horizontal slice');
    }
}

/**
 * Update vertical profile visualization
 */
async function updateVerticalProfile() {
    try {
        const lat = parseFloat(document.getElementById('location-lat').value);
        const lon = parseFloat(document.getElementById('location-lon').value);

        currentLat = lat;
        currentLon = lon;

        const profileData = await api.getVerticalProfile(lat, lon);

        // Update location display
        const latStr = Math.abs(lat).toFixed(1) + (lat >= 0 ? '°N' : '°S');
        const lonStr = Math.abs(lon).toFixed(1) + (lon >= 0 ? '°E' : '°W');
        document.getElementById('profile-location').textContent = `${latStr}, ${lonStr}`;

        const trace = {
            type: 'scatter',
            x: profileData.ne_values,
            y: profileData.altitude_km,
            mode: 'lines+markers',
            line: { color: '#2196F3', width: 2 },
            marker: { size: 4 },
            hovertemplate: 'Alt: %{y:.0f} km<br>Ne: %{x:.2e} el/m³<extra></extra>'
        };

        // Add layer markers if detected
        const annotations = [];
        if (profileData.layers) {
            if (profileData.layers.E_layer) {
                const e = profileData.layers.E_layer;
                annotations.push({
                    x: e.ne_el_m3,
                    y: e.altitude_km,
                    text: `E (${e.fof_mhz.toFixed(2)} MHz)`,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 30,
                    ay: -30
                });
            }
            if (profileData.layers.F1_layer) {
                const f1 = profileData.layers.F1_layer;
                annotations.push({
                    x: f1.ne_el_m3,
                    y: f1.altitude_km,
                    text: `F1 (${f1.fof_mhz.toFixed(2)} MHz)`,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 30,
                    ay: 0
                });
            }
            if (profileData.layers.F2_layer) {
                const f2 = profileData.layers.F2_layer;
                annotations.push({
                    x: f2.ne_el_m3,
                    y: f2.altitude_km,
                    text: `F2 (${f2.fof_mhz.toFixed(2)} MHz)`,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 30,
                    ay: 30
                });
            }
        }

        const layout = {
            ...PlotlyDefaultLayout,
            title: `Vertical Ne Profile at ${latStr}, ${lonStr}`,
            xaxis: {
                title: 'Electron Density (el/m³)',
                type: 'log',
                exponentformat: 'e'
            },
            yaxis: {
                title: 'Altitude (km)',
                range: [60, 600]
            },
            annotations: annotations,
            margin: { l: 60, r: 40, t: 60, b: 60 }
        };

        Plotly.newPlot('ne-profile-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating vertical profile:', error);
        Utils.showError('ne-profile-plot', 'Failed to load vertical profile');
    }
}

/**
 * Update foF2 map
 */
async function updateFoF2Map() {
    try {
        const fof2Data = await api.getFoF2Map();

        const trace = {
            type: 'heatmap',
            x: fof2Data.longitude,
            y: fof2Data.latitude,
            z: fof2Data.fof2_mhz,
            colorscale: 'Plasma',
            colorbar: {
                title: 'foF2 (MHz)',
                len: 0.7
            },
            hovertemplate: 'Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>foF2: %{z:.2f} MHz<extra></extra>'
        };

        const layout = {
            ...PlotlyDefaultLayout,
            title: `foF2: ${fof2Data.fof2_min.toFixed(1)}-${fof2Data.fof2_max.toFixed(1)} MHz`,
            xaxis: { title: 'Longitude (°)', showticklabels: false },
            yaxis: { title: 'Latitude (°)' },
            margin: { l: 50, r: 60, t: 50, b: 30 }
        };

        Plotly.newPlot('fof2-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating foF2 map:', error);
        Utils.showError('fof2-plot', 'Failed to load foF2 map');
    }
}

/**
 * Update hmF2 map
 */
async function updateHmF2Map() {
    try {
        const hmf2Data = await api.getHmF2Map();

        const trace = {
            type: 'heatmap',
            x: hmf2Data.longitude,
            y: hmf2Data.latitude,
            z: hmf2Data.hmf2_km,
            colorscale: 'RdBu',
            reversescale: true,
            colorbar: {
                title: 'hmF2 (km)',
                len: 0.7
            },
            hovertemplate: 'Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>hmF2: %{z:.0f} km<extra></extra>'
        };

        const layout = {
            ...PlotlyDefaultLayout,
            title: `hmF2: ${hmf2Data.hmf2_min.toFixed(0)}-${hmf2Data.hmf2_max.toFixed(0)} km`,
            xaxis: { title: 'Longitude (°)', showticklabels: false },
            yaxis: { title: 'Latitude (°)', showticklabels: false },
            margin: { l: 50, r: 60, t: 50, b: 30 }
        };

        Plotly.newPlot('hmf2-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating hmF2 map:', error);
        Utils.showError('hmf2-plot', 'Failed to load hmF2 map');
    }
}

/**
 * Update TEC map
 */
async function updateTECMap() {
    try {
        const tecData = await api.getTECMap();

        const trace = {
            type: 'heatmap',
            x: tecData.longitude,
            y: tecData.latitude,
            z: tecData.tec_tecu,
            colorscale: 'Jet',
            colorbar: {
                title: 'TEC (TECU)',
                len: 0.7
            },
            hovertemplate: 'Lat: %{y:.1f}°<br>Lon: %{x:.1f}°<br>TEC: %{z:.1f} TECU<extra></extra>'
        };

        const layout = {
            ...PlotlyDefaultLayout,
            title: `TEC: ${tecData.tec_min.toFixed(1)}-${tecData.tec_max.toFixed(1)} TECU`,
            xaxis: { title: 'Longitude (°)', showticklabels: false },
            yaxis: { title: 'Latitude (°)', showticklabels: false },
            margin: { l: 50, r: 60, t: 50, b: 30 }
        };

        Plotly.newPlot('tec-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating TEC map:', error);
        Utils.showError('tec-plot', 'Failed to load TEC map');
    }
}

/**
 * Update grid statistics
 */
async function updateStatistics() {
    try {
        const stats = await api.getGridStatistics();

        const statsHTML = `
            <div class="grid-4col">
                <div>
                    <strong>Ne Range:</strong><br>
                    ${Utils.formatScientific(stats.ne_min)} - ${Utils.formatScientific(stats.ne_max)} el/m³
                </div>
                <div>
                    <strong>Ne Mean:</strong><br>
                    ${Utils.formatScientific(stats.ne_mean)} el/m³
                </div>
                <div>
                    <strong>Ne Std Dev:</strong><br>
                    ${Utils.formatScientific(stats.ne_std)} el/m³
                </div>
                <div>
                    <strong>Grid Shape:</strong><br>
                    ${stats.grid_shape.join(' × ')} points
                </div>
            </div>
            <div style="margin-top: 15px; color: #757575; font-size: 14px;">
                <strong>Timestamp:</strong> ${Utils.formatTimestamp(stats.timestamp)}<br>
                <strong>Cycle ID:</strong> ${stats.cycle_id}
            </div>
        `;

        document.getElementById('grid-statistics').innerHTML = statsHTML;

    } catch (error) {
        console.error('Error updating statistics:', error);
        document.getElementById('grid-statistics').innerHTML =
            '<p style="color: #F44336;">Failed to load statistics</p>';
    }
}

/**
 * Refresh all data
 */
function refreshAllData() {
    Utils.showNotification('Refreshing ionosphere data...', 'info', 2000);
    loadAllData();
}

// WebSocket handlers for real-time updates
ws.on('grid_update', async (data) => {
    console.log('Grid update received, refreshing visualizations...');
    await loadAllData();
    Utils.showNotification('Grid updated', 'success', 3000);
});

// Initialize on page load
window.addEventListener('load', initIonosphereView);
