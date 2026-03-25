/**
 * Raytracer Visualization JavaScript
 *
 * Handles PHaRLAP ray tracing visualization:
 * - Ray path cross-sections
 * - Coverage maps with winner landing points
 * - Synthetic ionograms
 * - MUF/LUF/FOT time series
 * - Winner triplets table
 * - Link budget analysis
 */

// Global state
let coverageMap = null;
let winnersData = [];
let currentFreqHistoryHours = 24;
let legendCollapsed = false;

/**
 * Toggle legend panel visibility
 */
function toggleLegend() {
    legendCollapsed = !legendCollapsed;
    const content = document.getElementById('legend-content');
    const toggle = document.getElementById('legend-toggle');

    if (legendCollapsed) {
        content.classList.add('collapsed');
        toggle.classList.add('collapsed');
    } else {
        content.classList.remove('collapsed');
        toggle.classList.remove('collapsed');
    }

    // Save preference
    localStorage.setItem('raytracerLegendCollapsed', legendCollapsed);
}

/**
 * Restore legend state from localStorage
 */
function restoreLegendState() {
    const saved = localStorage.getItem('raytracerLegendCollapsed');
    if (saved === 'true') {
        legendCollapsed = true;
        const content = document.getElementById('legend-content');
        const toggle = document.getElementById('legend-toggle');
        if (content) content.classList.add('collapsed');
        if (toggle) toggle.classList.add('collapsed');
    }
}

// Color scheme constants
const COLORS = {
    O_MODE: '#00BFFF',        // DeepSkyBlue
    X_MODE: '#FF6B6B',        // Coral
    IONOSPHERE: 'rgba(100, 149, 237, 0.2)',  // Cornflower blue transparent
    GROUND: '#228B22',        // Forest green
    LUF: '#FFD700',           // Gold/Yellow
    MUF: '#F44336',           // Red
    FOT: '#4CAF50',           // Green
    BACKGROUND: '#1e1e1e',
    PAPER: '#1e1e1e',
    GRID: '#333333',
    TEXT: '#e0e0e0'
};

// Dark theme Plotly layout
const DarkPlotlyLayout = {
    font: { family: 'Arial, sans-serif', size: 12, color: COLORS.TEXT },
    margin: { l: 60, r: 40, t: 40, b: 60 },
    paper_bgcolor: COLORS.PAPER,
    plot_bgcolor: COLORS.BACKGROUND,
    hovermode: 'closest',
    xaxis: {
        gridcolor: COLORS.GRID,
        zerolinecolor: COLORS.GRID
    },
    yaxis: {
        gridcolor: COLORS.GRID,
        zerolinecolor: COLORS.GRID
    },
    legend: {
        bgcolor: 'rgba(30, 30, 30, 0.8)',
        font: { color: COLORS.TEXT }
    }
};

/**
 * Initialize raytracer view
 */
async function initRaytracerView() {
    // Restore legend state
    restoreLegendState();

    // Initialize coverage map
    initCoverageMap();

    // Load all data
    await loadAllRaytracerData();

    // Setup auto-refresh (every 30 seconds)
    setInterval(loadAllRaytracerData, 30000);
}

/**
 * Load all raytracer data
 */
async function loadAllRaytracerData() {
    try {
        await Promise.all([
            loadCurrentFrequencies(),
            loadRayPaths(),
            loadCoverageMapData(),
            loadIonogram(),
            updateFrequencyHistory(),
            loadWinnersData(),
            loadLinkBudget()
        ]);
    } catch (error) {
        console.error('Error loading raytracer data:', error);
        Utils.showNotification('Failed to load raytracer data', 'error');
    }
}

/**
 * Load current MUF/LUF/FOT frequencies
 */
async function loadCurrentFrequencies() {
    try {
        const data = await api.getRaytracerCurrentFrequencies();

        document.getElementById('current-luf').textContent =
            data.luf != null ? data.luf.toFixed(2) : '--';
        document.getElementById('current-muf').textContent =
            data.muf != null ? data.muf.toFixed(2) : '--';
        document.getElementById('current-fot').textContent =
            data.fot != null ? data.fot.toFixed(2) : '--';

        // Update mini stats
        document.getElementById('num-winners').textContent =
            data.num_winners != null ? data.num_winners : '--';
        document.getElementById('path-range').textContent =
            data.great_circle_range_km != null ? data.great_circle_range_km.toFixed(0) : '--';
        document.getElementById('compute-time').textContent =
            data.computation_time_s != null ? data.computation_time_s.toFixed(1) : '--';

        // Update result age
        if (data.timestamp) {
            document.getElementById('result-age').textContent =
                Utils.formatRelativeTime(new Date(data.timestamp));
        }

    } catch (error) {
        console.error('Error loading current frequencies:', error);
    }
}

/**
 * Load and plot ray paths cross-section
 */
async function loadRayPaths() {
    try {
        const data = await api.getRaytracerRayPaths();

        if (!data.paths || data.paths.length === 0) {
            plotEmptyRayPaths();
            return;
        }

        plotRayPaths(data);

    } catch (error) {
        console.error('Error loading ray paths:', error);
        plotEmptyRayPaths();
    }
}

/**
 * Plot ray paths cross-section
 */
function plotRayPaths(data) {
    const traces = [];

    // Add ionosphere shading if profile available
    if (data.ionosphere_profile) {
        const profile = data.ionosphere_profile;
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: 'Ionosphere',
            x: profile.ground_ranges,
            y: profile.altitudes,
            fill: 'tozeroy',
            fillcolor: COLORS.IONOSPHERE,
            line: { color: 'rgba(100, 149, 237, 0.5)', width: 1 },
            hoverinfo: 'skip'
        });
    }

    // Add ground line
    const maxRange = data.paths.reduce((max, p) => {
        const pMax = Math.max(...(p.ground_ranges_km || [0]));
        return pMax > max ? pMax : max;
    }, 0);

    traces.push({
        type: 'scatter',
        mode: 'lines',
        name: 'Ground',
        x: [0, maxRange * 1.1],
        y: [0, 0],
        line: { color: COLORS.GROUND, width: 3 },
        hoverinfo: 'skip'
    });

    // Group paths by mode
    const oModePaths = data.paths.filter(p => p.mode === 'O' || p.mode === 'O_MODE');
    const xModePaths = data.paths.filter(p => p.mode === 'X' || p.mode === 'X_MODE');

    // Add O-mode paths
    oModePaths.forEach((path, i) => {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: i === 0 ? 'O-Mode' : undefined,
            legendgroup: 'O-Mode',
            showlegend: i === 0,
            x: path.ground_ranges_km,
            y: path.altitudes_km,
            line: { color: COLORS.O_MODE, width: 1.5 },
            hovertemplate: `${path.frequency_mhz?.toFixed(1)} MHz<br>` +
                           `Range: %{x:.0f} km<br>Alt: %{y:.0f} km<extra></extra>`
        });
    });

    // Add X-mode paths
    xModePaths.forEach((path, i) => {
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: i === 0 ? 'X-Mode' : undefined,
            legendgroup: 'X-Mode',
            showlegend: i === 0,
            x: path.ground_ranges_km,
            y: path.altitudes_km,
            line: { color: COLORS.X_MODE, width: 1.5 },
            hovertemplate: `${path.frequency_mhz?.toFixed(1)} MHz<br>` +
                           `Range: %{x:.0f} km<br>Alt: %{y:.0f} km<extra></extra>`
        });
    });

    const layout = {
        ...DarkPlotlyLayout,
        title: '',
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'Ground Range (km)',
            range: [0, maxRange * 1.1]
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: 'Altitude (km)',
            range: [0, 500]
        },
        showlegend: true,
        legend: {
            ...DarkPlotlyLayout.legend,
            x: 0.02,
            y: 0.98
        }
    };

    Plotly.newPlot('ray-paths-plot', traces, layout, PlotlyDefaultConfig);
}

/**
 * Plot empty ray paths placeholder
 */
function plotEmptyRayPaths() {
    const layout = {
        ...DarkPlotlyLayout,
        title: '',
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'Ground Range (km)',
            range: [0, 1000]
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: 'Altitude (km)',
            range: [0, 500]
        },
        annotations: [{
            text: 'No ray path data available',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.5,
            showarrow: false,
            font: { size: 16, color: '#888' }
        }]
    };

    Plotly.newPlot('ray-paths-plot', [], layout, PlotlyDefaultConfig);
}

/**
 * Initialize Leaflet coverage map
 */
function initCoverageMap() {
    if (coverageMap) return;

    coverageMap = L.map('coverage-map', {
        center: [40.0, -105.0],
        zoom: 5
    });

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(coverageMap);

    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function() {
        const div = L.DomUtil.create('div', 'map-legend');
        div.innerHTML = `
            <h4>Map Legend</h4>
            <div class="legend-item">
                <span class="legend-marker" style="background:#FF5722;"></span> TX
            </div>
            <div class="legend-item">
                <span class="legend-marker" style="background:#4CAF50;"></span> RX
            </div>
            <div class="legend-item">
                <span class="legend-color o-mode"></span> O-Mode
            </div>
            <div class="legend-item">
                <span class="legend-color x-mode"></span> X-Mode
            </div>
            <div class="legend-snr-hint">Circle size = SNR</div>
        `;
        return div;
    };
    legend.addTo(coverageMap);
}

/**
 * Load and update coverage map
 */
async function loadCoverageMapData() {
    try {
        const data = await api.getRaytracerCoverageMap();
        updateCoverageMap(data);
    } catch (error) {
        console.error('Error loading coverage map:', error);
    }
}

/**
 * Update coverage map with landing points
 */
function updateCoverageMap(data) {
    if (!coverageMap) return;

    // Clear existing markers
    coverageMap.eachLayer(layer => {
        if (layer instanceof L.CircleMarker || layer instanceof L.Marker) {
            coverageMap.removeLayer(layer);
        }
    });

    // Add TX marker
    if (data.tx_position) {
        const [txLat, txLon] = data.tx_position;
        L.marker([txLat, txLon], {
            icon: L.divIcon({
                className: 'tx-marker',
                html: '<div style="background:#FF5722;width:12px;height:12px;border-radius:50%;border:2px solid white;"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            })
        }).addTo(coverageMap).bindPopup('<b>Transmitter</b>');
    }

    // Add RX marker
    if (data.rx_position) {
        const [rxLat, rxLon] = data.rx_position;
        L.marker([rxLat, rxLon], {
            icon: L.divIcon({
                className: 'rx-marker',
                html: '<div style="background:#4CAF50;width:12px;height:12px;border-radius:50%;border:2px solid white;"></div>',
                iconSize: [16, 16],
                iconAnchor: [8, 8]
            })
        }).addTo(coverageMap).bindPopup('<b>Receiver</b>');
    }

    // Add winner landing points
    if (data.points && data.points.length > 0) {
        data.points.forEach(point => {
            if (point.lat == null || point.lon == null) return;

            const isOMode = point.mode === 'O' || point.mode === 'O_MODE';
            const color = isOMode ? COLORS.O_MODE : COLORS.X_MODE;

            // Size based on SNR
            const snr = point.snr_db || 0;
            const radius = Math.max(4, Math.min(12, 4 + (snr + 20) / 10));

            const circle = L.circleMarker([point.lat, point.lon], {
                radius: radius,
                fillColor: color,
                color: 'white',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(coverageMap);

            circle.bindPopup(`
                <b>${point.frequency_mhz?.toFixed(1)} MHz</b><br>
                Mode: ${isOMode ? 'O-Mode' : 'X-Mode'}<br>
                SNR: ${point.snr_db?.toFixed(0) || 'N/A'} dB<br>
                Hops: ${point.hop_count || 1}<br>
                Elev: ${point.elevation_deg?.toFixed(1)}°<br>
                Error: ${point.landing_error_km?.toFixed(1)} km
            `);
        });

        // Fit map to show all points
        const bounds = L.latLngBounds(data.points.map(p => [p.lat, p.lon]));
        if (data.tx_position) bounds.extend(data.tx_position);
        if (data.rx_position) bounds.extend(data.rx_position);
        coverageMap.fitBounds(bounds, { padding: [20, 20] });
    }
}

/**
 * Load and plot synthetic ionogram
 */
async function loadIonogram() {
    try {
        const data = await api.getRaytracerIonogramLatest();
        plotIonogram(data);
    } catch (error) {
        console.error('Error loading ionogram:', error);
        plotEmptyIonogram();
    }
}

/**
 * Plot synthetic ionogram
 */
function plotIonogram(data) {
    const traces = [];

    // O-mode trace
    if (data.o_mode && data.o_mode.frequencies && data.o_mode.frequencies.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'O-Mode',
            x: data.o_mode.frequencies,
            y: data.o_mode.delays,
            marker: {
                color: COLORS.O_MODE,
                size: 6,
                symbol: 'circle'
            },
            hovertemplate: 'Freq: %{x:.2f} MHz<br>Delay: %{y:.2f} ms<extra>O-Mode</extra>'
        });
    }

    // X-mode trace
    if (data.x_mode && data.x_mode.frequencies && data.x_mode.frequencies.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'X-Mode',
            x: data.x_mode.frequencies,
            y: data.x_mode.delays,
            marker: {
                color: COLORS.X_MODE,
                size: 6,
                symbol: 'diamond'
            },
            hovertemplate: 'Freq: %{x:.2f} MHz<br>Delay: %{y:.2f} ms<extra>X-Mode</extra>'
        });
    }

    if (traces.length === 0) {
        plotEmptyIonogram();
        return;
    }

    const layout = {
        ...DarkPlotlyLayout,
        title: '',
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'Frequency (MHz)',
            range: [0, 30]
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: 'Group Delay (ms)',
            autorange: 'reversed'  // Ionograms typically show delay increasing downward
        },
        showlegend: true,
        legend: {
            ...DarkPlotlyLayout.legend,
            x: 0.98,
            y: 0.02,
            xanchor: 'right',
            yanchor: 'bottom'
        }
    };

    Plotly.newPlot('ionogram-plot', traces, layout, PlotlyDefaultConfig);
}

/**
 * Plot empty ionogram placeholder
 */
function plotEmptyIonogram() {
    const layout = {
        ...DarkPlotlyLayout,
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'Frequency (MHz)',
            range: [0, 30]
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: 'Group Delay (ms)',
            range: [0, 10]
        },
        annotations: [{
            text: 'No ionogram data available',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.5,
            showarrow: false,
            font: { size: 16, color: '#888' }
        }]
    };

    Plotly.newPlot('ionogram-plot', [], layout, PlotlyDefaultConfig);
}

/**
 * Update frequency history chart
 */
async function updateFrequencyHistory() {
    try {
        const hours = parseInt(document.getElementById('freq-history-hours').value) || 24;
        currentFreqHistoryHours = hours;

        const history = await api.getRaytracerFrequencyHistory(hours);
        plotFrequencyHistory(history);

    } catch (error) {
        console.error('Error updating frequency history:', error);
    }
}

/**
 * Plot MUF/LUF/FOT frequency history
 */
function plotFrequencyHistory(history) {
    const traces = [];

    // LUF trace
    if (history.luf && history.luf.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: 'LUF',
            x: history.luf.map(d => new Date(d.timestamp)),
            y: history.luf.map(d => d.value),
            line: { color: COLORS.LUF, width: 2 },
            marker: { size: 4 }
        });
    }

    // MUF trace
    if (history.muf && history.muf.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: 'MUF',
            x: history.muf.map(d => new Date(d.timestamp)),
            y: history.muf.map(d => d.value),
            line: { color: COLORS.MUF, width: 2 },
            marker: { size: 4 }
        });
    }

    // FOT trace
    if (history.fot && history.fot.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            name: 'FOT',
            x: history.fot.map(d => new Date(d.timestamp)),
            y: history.fot.map(d => d.value),
            line: { color: COLORS.FOT, width: 2 },
            marker: { size: 4 }
        });
    }

    if (traces.length === 0) {
        const layout = {
            ...DarkPlotlyLayout,
            xaxis: { ...DarkPlotlyLayout.xaxis, title: 'Time (UTC)' },
            yaxis: { ...DarkPlotlyLayout.yaxis, title: 'Frequency (MHz)', range: [0, 30] },
            annotations: [{
                text: 'No frequency history data',
                xref: 'paper',
                yref: 'paper',
                x: 0.5,
                y: 0.5,
                showarrow: false,
                font: { size: 16, color: '#888' }
            }]
        };
        Plotly.newPlot('frequency-history-plot', [], layout, PlotlyDefaultConfig);
        return;
    }

    const layout = {
        ...DarkPlotlyLayout,
        title: '',
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'Time (UTC)',
            type: 'date'
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: 'Frequency (MHz)',
            range: [0, 30]
        },
        showlegend: true,
        legend: {
            ...DarkPlotlyLayout.legend,
            x: 0.02,
            y: 0.98
        },
        hovermode: 'x unified'
    };

    Plotly.newPlot('frequency-history-plot', traces, layout, PlotlyDefaultConfig);
}

/**
 * Load winners data for table
 */
async function loadWinnersData() {
    try {
        const data = await api.getRaytracerWinnersLatest(200);
        winnersData = data.winners || [];
        renderWinnersTable();
    } catch (error) {
        console.error('Error loading winners data:', error);
        winnersData = [];
        renderWinnersTable();
    }
}

/**
 * Render winner triplets table
 */
function renderWinnersTable() {
    const tbody = document.getElementById('winners-table-body');
    const modeFilter = document.getElementById('winners-mode-filter').value;
    const sortBy = document.getElementById('winners-sort').value;

    // Filter by mode
    let filtered = winnersData;
    if (modeFilter !== 'all') {
        filtered = winnersData.filter(w => {
            const mode = w.mode === 'O' || w.mode === 'O_MODE' ? 'O' : 'X';
            return mode === modeFilter;
        });
    }

    // Sort
    filtered.sort((a, b) => {
        switch (sortBy) {
            case 'frequency':
                return (a.frequency_mhz || 0) - (b.frequency_mhz || 0);
            case 'snr':
                return (b.snr_db || -100) - (a.snr_db || -100);  // Descending
            case 'elevation':
                return (a.elevation_deg || 0) - (b.elevation_deg || 0);
            case 'delay':
                return (a.group_delay_ms || 0) - (b.group_delay_ms || 0);
            default:
                return 0;
        }
    });

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No winner triplets available</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(w => {
        const isOMode = w.mode === 'O' || w.mode === 'O_MODE';
        const modeClass = isOMode ? 'o-mode' : 'x-mode';
        const modeLabel = isOMode ? 'O' : 'X';

        // SNR color class
        let snrClass = '';
        if (w.snr_db != null) {
            if (w.snr_db >= 20) snrClass = 'snr-excellent';
            else if (w.snr_db >= 10) snrClass = 'snr-good';
            else if (w.snr_db >= 0) snrClass = 'snr-marginal';
            else snrClass = 'snr-poor';
        }

        return `
            <tr>
                <td>${w.frequency_mhz?.toFixed(2) || '--'}</td>
                <td>${w.elevation_deg?.toFixed(1) || '--'}</td>
                <td>${w.azimuth_deg?.toFixed(1) || '--'}</td>
                <td>${w.group_delay_ms?.toFixed(2) || '--'}</td>
                <td>${w.ground_range_km?.toFixed(0) || '--'}</td>
                <td><span class="mode-badge ${modeClass}">${modeLabel}</span></td>
                <td>${w.hop_count || 1}</td>
                <td>${w.reflection_height_km?.toFixed(0) || '--'}</td>
                <td class="${snrClass}">${w.snr_db?.toFixed(0) || '--'}</td>
                <td>${w.path_loss_db?.toFixed(0) || '--'}</td>
            </tr>
        `;
    }).join('');
}

/**
 * Load and plot link budget analysis
 */
async function loadLinkBudget() {
    try {
        const data = await api.getRaytracerLinkBudget();
        plotLinkBudget(data);
    } catch (error) {
        console.error('Error loading link budget:', error);
        plotEmptyLinkBudget();
    }
}

/**
 * Plot link budget horizontal bar chart
 */
function plotLinkBudget(data) {
    if (!data.link_budgets || data.link_budgets.length === 0) {
        plotEmptyLinkBudget();
        return;
    }

    // Group by frequency and take best SNR
    const byFreq = {};
    data.link_budgets.forEach(lb => {
        const freq = lb.frequency_mhz?.toFixed(1);
        if (!byFreq[freq] || (lb.snr_db || -100) > (byFreq[freq].snr_db || -100)) {
            byFreq[freq] = lb;
        }
    });

    const entries = Object.values(byFreq).sort((a, b) =>
        (a.frequency_mhz || 0) - (b.frequency_mhz || 0)
    );

    const labels = entries.map(e => `${e.frequency_mhz?.toFixed(1)} MHz`);
    const snrValues = entries.map(e => e.snr_db || 0);
    const pathLossValues = entries.map(e => -(e.path_loss_db || 0));  // Negative for display

    const traces = [
        {
            type: 'bar',
            name: 'SNR (dB)',
            y: labels,
            x: snrValues,
            orientation: 'h',
            marker: {
                color: snrValues.map(snr => {
                    if (snr >= 20) return '#4CAF50';
                    if (snr >= 10) return '#8BC34A';
                    if (snr >= 0) return '#FF9800';
                    return '#F44336';
                })
            },
            text: snrValues.map(v => v.toFixed(0) + ' dB'),
            textposition: 'outside',
            hovertemplate: 'SNR: %{x:.1f} dB<extra></extra>'
        }
    ];

    const layout = {
        ...DarkPlotlyLayout,
        title: '',
        xaxis: {
            ...DarkPlotlyLayout.xaxis,
            title: 'SNR (dB)',
            range: [-20, Math.max(40, ...snrValues) + 10]
        },
        yaxis: {
            ...DarkPlotlyLayout.yaxis,
            title: '',
            automargin: true
        },
        showlegend: false,
        barmode: 'group',
        margin: { l: 100, r: 60, t: 40, b: 60 }
    };

    Plotly.newPlot('link-budget-plot', traces, layout, PlotlyDefaultConfig);
}

/**
 * Plot empty link budget placeholder
 */
function plotEmptyLinkBudget() {
    const layout = {
        ...DarkPlotlyLayout,
        xaxis: { ...DarkPlotlyLayout.xaxis, title: 'SNR (dB)', range: [-20, 40] },
        yaxis: { ...DarkPlotlyLayout.yaxis, title: '' },
        annotations: [{
            text: 'No link budget data available',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.5,
            showarrow: false,
            font: { size: 16, color: '#888' }
        }]
    };

    Plotly.newPlot('link-budget-plot', [], layout, PlotlyDefaultConfig);
}

// WebSocket handlers for real-time updates
ws.on('raytracer_update', async (data) => {
    console.log('Raytracer update received');
    await loadAllRaytracerData();
    Utils.showNotification('Ray tracer data updated', 'success', 2000);
});

ws.on('homing_result', async (data) => {
    console.log('Homing result received');
    await loadAllRaytracerData();
});

// Initialize on page load
window.addEventListener('load', initRaytracerView);
