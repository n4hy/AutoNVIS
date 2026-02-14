/**
 * Space Weather Visualization JavaScript
 *
 * Handles space weather monitoring:
 * - X-ray flux with flare classification
 * - Solar wind parameters
 * - Autonomous mode history
 * - Alert feed
 */

let currentXRayHours = 24;

/**
 * Initialize space weather view
 */
async function initSpaceWeatherView() {
    await loadAllSpaceWeatherData();

    // Setup auto-refresh (every 60 seconds)
    setInterval(loadCurrentStatus, 60000);
}

/**
 * Load all space weather data
 */
async function loadAllSpaceWeatherData() {
    try {
        await Promise.all([
            loadCurrentStatus(),
            updateXRayChart(),
            loadSolarWindData(),
            loadModeHistory(),
            loadSpaceWeatherStatistics(),
            loadAlerts()
        ]);
    } catch (error) {
        console.error('Error loading space weather data:', error);
        Utils.showNotification('Failed to load space weather data', 'error');
    }
}

/**
 * Load current status
 */
async function loadCurrentStatus() {
    try {
        // X-ray status
        const xray = await api.getXRayLatest();
        const flareClass = xray.flare_class + xray.flare_magnitude.toFixed(1);
        document.getElementById('current-xray-class').textContent = flareClass;
        document.getElementById('current-xray-class').style.color =
            Utils.getFlareColor(xray.flare_class);

        // Mode status
        const mode = await api.getCurrentMode();
        document.getElementById('current-mode').textContent = mode.mode;
        document.getElementById('current-mode').style.color =
            Utils.getModeColor(mode.mode);

        // Solar wind
        const solarWind = await api.getSolarWindLatest();
        if (solarWind) {
            document.getElementById('sw-speed').textContent =
                solarWind.speed ? solarWind.speed.toFixed(0) : '--';
            document.getElementById('sw-bz').textContent =
                solarWind.bz !== undefined ? solarWind.bz.toFixed(1) : '--';

            // Color Bz based on value (negative = southward = bad for geomagnetic conditions)
            const bzElement = document.getElementById('sw-bz');
            if (solarWind.bz < -5) {
                bzElement.style.color = '#F44336'; // Red for strong southward
            } else if (solarWind.bz < 0) {
                bzElement.style.color = '#FF9800'; // Orange for southward
            } else {
                bzElement.style.color = '#4CAF50'; // Green for northward
            }
        }

    } catch (error) {
        console.error('Error loading current status:', error);
    }
}

/**
 * Update X-ray flux chart
 */
async function updateXRayChart() {
    try {
        const hours = parseInt(document.getElementById('xray-hours').value);
        currentXRayHours = hours;

        const xrayData = await api.getXRayHistory(hours);

        // Create trace
        const trace = {
            type: 'scatter',
            mode: 'lines',
            name: 'X-Ray Flux',
            x: xrayData.data.map(d => new Date(d.timestamp)),
            y: xrayData.data.map(d => d.flux_wm2),
            line: { color: '#FF6347', width: 2 },
            hovertemplate: 'Time: %{x}<br>Flux: %{y:.2e} W/m²<br>Class: %{text}<extra></extra>',
            text: xrayData.data.map(d => d.flare_class + d.flare_magnitude.toFixed(1))
        };

        // Add flare class threshold lines
        const shapes = [
            { y: 1e-7, label: 'B', color: '#90EE90' },
            { y: 1e-6, label: 'C', color: '#FFD700' },
            { y: 1e-5, label: 'M', color: '#FFA500' },
            { y: 1e-4, label: 'X', color: '#DC143C' }
        ];

        const annotations = shapes.map(s => ({
            x: 0.02,
            y: s.y,
            xref: 'paper',
            yref: 'y',
            text: s.label,
            showarrow: false,
            font: { color: s.color, size: 14, weight: 'bold' },
            xanchor: 'left'
        }));

        const layout = {
            ...PlotlyDefaultLayout,
            title: `X-Ray Flux History (${hours} hours)`,
            xaxis: {
                title: 'Time (UTC)',
                type: 'date'
            },
            yaxis: {
                title: 'Flux (W/m²)',
                type: 'log',
                exponentformat: 'e',
                range: [-8, -3]
            },
            shapes: shapes.map(s => ({
                type: 'line',
                x0: 0,
                x1: 1,
                xref: 'paper',
                y0: s.y,
                y1: s.y,
                line: { color: s.color, width: 1, dash: 'dash' }
            })),
            annotations: annotations,
            margin: { l: 70, r: 40, t: 60, b: 80 }
        };

        Plotly.newPlot('xray-flux-plot', [trace], layout, PlotlyDefaultConfig);

    } catch (error) {
        console.error('Error updating X-ray chart:', error);
        Utils.showError('xray-flux-plot', 'Failed to load X-ray data');
    }
}

/**
 * Load solar wind data
 */
async function loadSolarWindData() {
    try {
        const solarWind = await api.getSolarWindLatest();

        if (!solarWind) {
            document.getElementById('solar-wind-gauges').innerHTML =
                '<p style="color: #757575;">No solar wind data available</p>';
            return;
        }

        const gaugesHTML = `
            <div class="grid-2col">
                <div class="gauge-card">
                    <div class="gauge-value">${solarWind.speed ? solarWind.speed.toFixed(0) : 'N/A'}</div>
                    <div class="gauge-label">Speed (km/s)</div>
                    <div class="gauge-range">Typical: 300-500 km/s</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-value">${solarWind.density ? solarWind.density.toFixed(1) : 'N/A'}</div>
                    <div class="gauge-label">Density (p/cm³)</div>
                    <div class="gauge-range">Typical: 5-10 p/cm³</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-value" style="color: ${solarWind.bz < 0 ? '#F44336' : '#4CAF50'}">
                        ${solarWind.bz !== undefined ? solarWind.bz.toFixed(1) : 'N/A'}
                    </div>
                    <div class="gauge-label">Bz (nT)</div>
                    <div class="gauge-range">Southward (<0) = disturbed</div>
                </div>
                <div class="gauge-card">
                    <div class="gauge-value">${solarWind.bt ? solarWind.bt.toFixed(1) : 'N/A'}</div>
                    <div class="gauge-label">Bt (nT)</div>
                    <div class="gauge-range">Total IMF magnitude</div>
                </div>
            </div>
            <div style="margin-top: 15px; font-size: 12px; color: #757575;">
                <strong>Last Update:</strong> ${Utils.formatTimestamp(solarWind.timestamp)}
            </div>
        `;

        document.getElementById('solar-wind-gauges').innerHTML = gaugesHTML;

    } catch (error) {
        console.error('Error loading solar wind data:', error);
        document.getElementById('solar-wind-gauges').innerHTML =
            '<p style="color: #F44336;">Failed to load solar wind data</p>';
    }
}

/**
 * Load mode history timeline
 */
async function loadModeHistory() {
    try {
        const modeData = await api.getModeHistory(24);

        if (modeData.count === 0) {
            document.getElementById('mode-timeline').innerHTML =
                '<p style="color: #757575;">No mode changes in the last 24 hours</p>';
            return;
        }

        let timelineHTML = '<div class="timeline">';

        modeData.data.forEach((entry, index) => {
            const modeColor = Utils.getModeColor(entry.mode);
            const isFirst = index === 0;

            timelineHTML += `
                <div class="timeline-item">
                    <div class="timeline-marker" style="background: ${modeColor};"></div>
                    <div class="timeline-content">
                        <div class="timeline-time">${Utils.formatTimestamp(entry.timestamp)}</div>
                        <div class="timeline-mode">
                            <span class="badge" style="background: ${modeColor}; color: white;">${entry.mode}</span>
                        </div>
                        ${entry.reason ? `<div class="timeline-reason">${entry.reason}</div>` : ''}
                    </div>
                </div>
            `;
        });

        timelineHTML += '</div>';

        document.getElementById('mode-timeline').innerHTML = timelineHTML;

    } catch (error) {
        console.error('Error loading mode history:', error);
        document.getElementById('mode-timeline').innerHTML =
            '<p style="color: #F44336;">Failed to load mode history</p>';
    }
}

/**
 * Load space weather statistics
 */
async function loadSpaceWeatherStatistics() {
    try {
        const stats = await api.getSpaceWeatherStatistics();

        let statsHTML = '<div class="grid-2col">';

        // X-ray statistics
        if (stats.xray) {
            statsHTML += `
                <div>
                    <h3 style="margin-bottom: 10px; color: var(--text-color);">X-Ray Flux</h3>
                    <strong>Range:</strong> ${Utils.formatScientific(stats.xray.flux_min_wm2)} -
                    ${Utils.formatScientific(stats.xray.flux_max_wm2)} W/m²<br>
                    <strong>Mean:</strong> ${Utils.formatScientific(stats.xray.flux_mean_wm2)} W/m²<br>
                    <strong>Measurements:</strong> ${stats.xray.measurements_24h}
                </div>
            `;
        }

        // Flare counts
        if (stats.flares_24h) {
            const flares = stats.flares_24h;
            const totalFlares = flares.C + flares.M + flares.X;
            statsHTML += `
                <div>
                    <h3 style="margin-bottom: 10px; color: var(--text-color);">Flare Activity (24h)</h3>
                    <strong>C-class:</strong> ${flares.C}<br>
                    <strong>M-class:</strong> ${flares.M}<br>
                    <strong>X-class:</strong> ${flares.X}<br>
                    <strong>Total:</strong> ${totalFlares}
                </div>
            `;
        }

        statsHTML += '</div>';

        // Max flare
        if (stats.max_flare_24h && stats.max_flare_24h.timestamp) {
            const maxFlare = stats.max_flare_24h;
            statsHTML += `
                <div style="margin-top: 15px; padding: 15px; background: #FFF3E0; border-radius: 4px; border-left: 4px solid ${Utils.getFlareColor(maxFlare.class)};">
                    <strong>Maximum Flare (24h):</strong>
                    <span style="font-size: 20px; color: ${Utils.getFlareColor(maxFlare.class)}; font-weight: bold;">
                        ${maxFlare.class}${maxFlare.magnitude.toFixed(1)}
                    </span><br>
                    <small style="color: #757575;">${Utils.formatTimestamp(maxFlare.timestamp)}</small>
                </div>
            `;
        }

        // Mode statistics
        if (stats.mode) {
            statsHTML += `
                <div style="margin-top: 15px;">
                    <strong>Current Mode:</strong>
                    <span class="badge" style="background: ${Utils.getModeColor(stats.mode.current)}; color: white;">
                        ${stats.mode.current}
                    </span><br>
                    <strong>Mode Changes (24h):</strong> ${stats.mode.changes_24h}<br>
                    ${stats.mode.duration_hours.QUIET ? `<strong>Time in QUIET:</strong> ${stats.mode.duration_hours.QUIET.toFixed(1)} hours<br>` : ''}
                    ${stats.mode.duration_hours.SHOCK ? `<strong>Time in SHOCK:</strong> ${stats.mode.duration_hours.SHOCK.toFixed(1)} hours` : ''}
                </div>
            `;
        }

        document.getElementById('sw-statistics').innerHTML = statsHTML;

    } catch (error) {
        console.error('Error loading space weather statistics:', error);
        document.getElementById('sw-statistics').innerHTML =
            '<p style="color: #F44336;">Failed to load statistics</p>';
    }
}

/**
 * Load alerts feed
 */
async function loadAlerts() {
    try {
        const filterSelect = document.getElementById('alert-filter');
        const severity = filterSelect.value || null;

        const alertsData = await api.getRecentAlerts(24, severity);

        if (alertsData.count === 0) {
            document.getElementById('alerts-feed').innerHTML =
                '<p style="color: #757575; text-align: center;">No alerts</p>';
            return;
        }

        let alertsHTML = '<div class="alerts-list">';

        alertsData.data.slice(0, 10).forEach(alert => {
            const severityColor = Utils.getSeverityColor(alert.severity || 'info');
            alertsHTML += `
                <div class="alert-item" style="border-left-color: ${severityColor};">
                    <div class="alert-severity">
                        <span class="badge" style="background: ${severityColor}; color: white;">
                            ${alert.severity || 'info'}
                        </span>
                    </div>
                    <div class="alert-message">${alert.message || 'Alert'}</div>
                    <div class="alert-time">${Utils.formatTimestamp(alert.timestamp)}</div>
                </div>
            `;
        });

        alertsHTML += '</div>';

        document.getElementById('alerts-feed').innerHTML = alertsHTML;

    } catch (error) {
        console.error('Error loading alerts:', error);
        document.getElementById('alerts-feed').innerHTML =
            '<p style="color: #F44336;">Failed to load alerts</p>';
    }
}

// WebSocket handlers for real-time updates
ws.on('xray_update', async (data) => {
    console.log('X-ray update received');
    await loadCurrentStatus();
    await updateXRayChart();
});

ws.on('solar_wind_update', async (data) => {
    console.log('Solar wind update received');
    await loadCurrentStatus();
    await loadSolarWindData();
});

ws.on('mode_change', async (data) => {
    console.log('Mode change received:', data.mode);
    await loadCurrentStatus();
    await loadModeHistory();
    Utils.showNotification(`Mode changed to ${data.mode}`, 'warning', 5000);
});

ws.on('alert', async (data) => {
    console.log('Alert received:', data);
    await loadAlerts();
    const severity = data.severity || 'info';
    Utils.showNotification(data.message || 'New alert', severity, 5000);
});

// Initialize on page load
window.addEventListener('load', initSpaceWeatherView);
