/**
 * Propagation Visualization JavaScript
 *
 * Handles propagation prediction visualization:
 * - LUF/MUF/FOT time series
 * - Coverage maps
 * - Frequency plan details
 */

let coverageMapLeaflet = null;
let currentTimeRange = 24;

/**
 * Initialize propagation view
 */
async function initPropagationView() {
    // Initialize Leaflet map
    initCoverageMap();

    // Load all data
    await loadAllPropagationData();

    // Setup auto-refresh (every 60 seconds)
    setInterval(loadCurrentFrequencyPlan, 60000);
}

/**
 * Load all propagation data
 */
async function loadAllPropagationData() {
    try {
        await Promise.all([
            loadCurrentFrequencyPlan(),
            updateTimeSeries(),
            loadPropagationStatistics(),
            loadFrequencyPlanDetails()
        ]);
    } catch (error) {
        console.error('Error loading propagation data:', error);
        Utils.showNotification('Failed to load propagation data', 'error');
    }
}

/**
 * Load current frequency plan (LUF/MUF/FOT)
 */
async function loadCurrentFrequencyPlan() {
    try {
        const plan = await api.getLatestFrequencyPlan();

        document.getElementById('current-luf').textContent =
            plan.luf_mhz ? plan.luf_mhz.toFixed(2) : '--';
        document.getElementById('current-muf').textContent =
            plan.muf_mhz ? plan.muf_mhz.toFixed(2) : '--';
        document.getElementById('current-fot').textContent =
            plan.fot_mhz ? plan.fot_mhz.toFixed(2) : '--';

    } catch (error) {
        console.error('Error loading current frequency plan:', error);
    }
}

/**
 * Update time series chart
 */
async function updateTimeSeries() {
    try {
        const hours = parseInt(document.getElementById('history-hours').value);
        currentTimeRange = hours;

        const history = await api.getFrequencyPlanHistory(hours);

        // Prepare traces
        const lufTrace = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'LUF',
            x: history.luf.map(d => new Date(d.timestamp)),
            y: history.luf.map(d => d.value),
            line: { color: '#F44336', width: 2 },
            marker: { size: 4 }
        };

        const mufTrace = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'MUF',
            x: history.muf.map(d => new Date(d.timestamp)),
            y: history.muf.map(d => d.value),
            line: { color: '#4CAF50', width: 2 },
            marker: { size: 4 }
        };

        const fotTrace = {
            type: 'scatter',
            mode: 'lines+markers',
            name: 'FOT',
            x: history.fot.map(d => new Date(d.timestamp)),
            y: history.fot.map(d => d.value),
            line: { color: '#2196F3', width: 2 },
            marker: { size: 4 }
        };

        const layout = {
            ...PlotlyDefaultLayout,
            title: `Frequency Plan History (${hours} hours)`,
            xaxis: {
                title: 'Time (UTC)',
                type: 'date'
            },
            yaxis: {
                title: 'Frequency (MHz)',
                range: [0, 30]
            },
            legend: {
                x: 0.02,
                y: 0.98,
                bgcolor: 'rgba(255,255,255,0.8)'
            },
            hovermode: 'x unified',
            margin: { l: 60, r: 40, t: 60, b: 80 }
        };

        Plotly.newPlot(
            'frequency-history-plot',
            [lufTrace, fotTrace, mufTrace],
            layout,
            PlotlyDefaultConfig
        );

    } catch (error) {
        console.error('Error updating time series:', error);
        Utils.showError('frequency-history-plot', 'Failed to load time series');
    }
}

/**
 * Initialize coverage map with Leaflet
 */
function initCoverageMap() {
    if (coverageMapLeaflet) {
        return; // Already initialized
    }

    coverageMapLeaflet = L.map('coverage-map').setView([40.0, -105.0], 4);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 10
    }).addTo(coverageMapLeaflet);

    // Add a marker for transmitter location (will be updated from API)
    L.marker([40.0, -105.0])
        .addTo(coverageMapLeaflet)
        .bindPopup('<b>TX Location</b><br>Boulder, CO');

    // Load coverage data if available
    loadCoverageMap();
}

/**
 * Load coverage map data
 */
async function loadCoverageMap() {
    try {
        const coverage = await api.getLatestCoverageMap();

        // Note: Coverage map visualization would require additional implementation
        // For now, we show a placeholder message
        console.log('Coverage map data:', coverage);

        // Add info overlay
        const info = L.control({ position: 'topright' });
        info.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'coverage-info');
            div.style.background = 'white';
            div.style.padding = '10px';
            div.style.borderRadius = '4px';
            div.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            div.innerHTML = '<strong>Coverage Data</strong><br>Available at API';
            return div;
        };
        info.addTo(coverageMapLeaflet);

    } catch (error) {
        console.error('Error loading coverage map:', error);
    }
}

/**
 * Load propagation statistics
 */
async function loadPropagationStatistics() {
    try {
        const stats = await api.getPropagationStatistics();

        let statsHTML = '<div class="grid-2col">';

        // Latest values
        if (stats.latest) {
            statsHTML += `
                <div>
                    <h3 style="margin-bottom: 10px; color: var(--text-color);">Latest Values</h3>
                    <strong>LUF:</strong> ${stats.latest.luf_mhz ? stats.latest.luf_mhz.toFixed(2) : 'N/A'} MHz<br>
                    <strong>MUF:</strong> ${stats.latest.muf_mhz ? stats.latest.muf_mhz.toFixed(2) : 'N/A'} MHz<br>
                    <strong>FOT:</strong> ${stats.latest.fot_mhz ? stats.latest.fot_mhz.toFixed(2) : 'N/A'} MHz<br>
                    <strong>Time:</strong> ${Utils.formatTimestamp(stats.latest.timestamp)}
                </div>
            `;
        }

        // 24-hour statistics
        let stats24h = '';
        if (stats.luf_24h) {
            stats24h += `
                <div>
                    <h3 style="margin-bottom: 10px; color: var(--text-color);">24h Statistics</h3>
                    <strong>LUF Range:</strong> ${stats.luf_24h.min.toFixed(2)} - ${stats.luf_24h.max.toFixed(2)} MHz<br>
                    <strong>MUF Range:</strong> ${stats.muf_24h.min.toFixed(2)} - ${stats.muf_24h.max.toFixed(2)} MHz<br>
                    <strong>FOT Range:</strong> ${stats.fot_24h.min.toFixed(2)} - ${stats.fot_24h.max.toFixed(2)} MHz<br>
                    <strong>Updates:</strong> ${stats.luf_24h.count}
                </div>
            `;
        }
        statsHTML += stats24h;
        statsHTML += '</div>';

        document.getElementById('prop-statistics').innerHTML = statsHTML;

    } catch (error) {
        console.error('Error loading propagation statistics:', error);
        document.getElementById('prop-statistics').innerHTML =
            '<p style="color: #F44336;">Failed to load statistics</p>';
    }
}

/**
 * Load frequency plan details
 */
async function loadFrequencyPlanDetails() {
    try {
        const plan = await api.getLatestFrequencyPlan();

        let detailsHTML = '<div class="grid-2col">';

        // Basic info
        detailsHTML += `
            <div>
                <h3 style="margin-bottom: 10px; color: var(--text-color);">Frequency Plan</h3>
                <strong>LUF:</strong> ${plan.luf_mhz ? plan.luf_mhz.toFixed(2) : 'N/A'} MHz<br>
                <strong>MUF:</strong> ${plan.muf_mhz ? plan.muf_mhz.toFixed(2) : 'N/A'} MHz<br>
                <strong>FOT (85% MUF):</strong> ${plan.fot_mhz ? plan.fot_mhz.toFixed(2) : 'N/A'} MHz<br>
                <strong>Usable Band:</strong> ${plan.luf_mhz && plan.muf_mhz ? (plan.muf_mhz - plan.luf_mhz).toFixed(2) : 'N/A'} MHz
            </div>
        `;

        // Additional details if available
        detailsHTML += `
            <div>
                <h3 style="margin-bottom: 10px; color: var(--text-color);">Details</h3>
                <strong>Timestamp:</strong> ${Utils.formatTimestamp(plan.timestamp)}<br>
                <strong>Age:</strong> ${Utils.formatRelativeTime(new Date(plan.timestamp))}<br>
        `;

        if (plan.tx_location) {
            detailsHTML += `<strong>TX Location:</strong> ${plan.tx_location.lat.toFixed(2)}°, ${plan.tx_location.lon.toFixed(2)}°<br>`;
        }

        detailsHTML += '</div>';
        detailsHTML += '</div>';

        // Recommendations
        if (plan.fot_mhz) {
            detailsHTML += `
                <div style="margin-top: 20px; padding: 15px; background: #E3F2FD; border-radius: 4px;">
                    <strong>Recommended Operating Frequency:</strong>
                    <span style="font-size: 24px; color: #2196F3; font-weight: bold;">${plan.fot_mhz.toFixed(2)} MHz</span><br>
                    <small style="color: #757575;">FOT provides the best balance between reliability and signal strength</small>
                </div>
            `;
        }

        document.getElementById('frequency-plan-details').innerHTML = detailsHTML;

    } catch (error) {
        console.error('Error loading frequency plan details:', error);
        document.getElementById('frequency-plan-details').innerHTML =
            '<p style="color: #F44336;">Failed to load frequency plan</p>';
    }
}

// WebSocket handlers for real-time updates
ws.on('frequency_plan_update', async (data) => {
    console.log('Frequency plan update received');
    await loadCurrentFrequencyPlan();
    await updateTimeSeries();
    await loadPropagationStatistics();
    await loadFrequencyPlanDetails();
    Utils.showNotification('Frequency plan updated', 'success', 2000);
});

ws.on('coverage_map_update', async (data) => {
    console.log('Coverage map update received');
    await loadCoverageMap();
});

// Initialize on page load
window.addEventListener('load', initPropagationView);
