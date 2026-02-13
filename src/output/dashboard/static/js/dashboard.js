// NVIS Analytics Dashboard JavaScript

class NVISDashboard {
    constructor() {
        this.ws = null;
        this.map = null;
        this.markers = {};
        this.charts = {};
        this.updateInterval = null;

        this.init();
    }

    async init() {
        // Initialize map
        this.initMap();

        // Initialize charts
        this.initCharts();

        // Connect WebSocket
        this.connectWebSocket();

        // Initial data load
        await this.loadAllData();

        // Set up periodic updates
        this.updateInterval = setInterval(() => this.loadAllData(), 30000); // 30 seconds
    }

    initMap() {
        // Initialize Leaflet map
        this.map = L.map('map').setView([40.0, -100.0], 4);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(this.map);
    }

    initCharts() {
        // Information Gain Chart
        const infoGainCtx = document.getElementById('info-gain-chart').getContext('2d');
        this.charts.infoGain = new Chart(infoGainCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Relative Contribution (%)',
                    data: [],
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgba(52, 152, 219, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Contribution: ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        // Quality Tier Distribution Chart
        const qualityCtx = document.getElementById('quality-chart').getContext('2d');
        this.charts.quality = new Chart(qualityCtx, {
            type: 'doughnut',
            data: {
                labels: ['Platinum', 'Gold', 'Silver', 'Bronze'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(163, 213, 255, 0.8)',
                        'rgba(255, 215, 0, 0.8)',
                        'rgba(192, 192, 192, 0.8)',
                        'rgba(205, 127, 50, 0.8)'
                    ],
                    borderColor: [
                        'rgba(163, 213, 255, 1)',
                        'rgba(255, 215, 0, 1)',
                        'rgba(192, 192, 192, 1)',
                        'rgba(205, 127, 50, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);

            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }

    handleWebSocketMessage(message) {
        if (message.type === 'analysis_update') {
            console.log('Received analysis update');
            this.updateDashboard(message.data);
        }
    }

    updateConnectionStatus(connected) {
        const statusIndicator = document.querySelector('.status-indicator');
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');

        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }

    async loadAllData() {
        try {
            // Load sounders
            const sounders = await this.fetchAPI('/api/nvis/sounders');
            this.updateSounderList(sounders);
            this.updateMapMarkers(sounders);

            // Load network analysis
            const analysis = await this.fetchAPI('/api/nvis/network/analysis');
            this.updateDashboard(analysis);

            // Load placement recommendations
            const placements = await this.fetchAPI('/api/nvis/placement/recommend?n_sounders=3');
            this.updatePlacementRecommendations(placements);
            this.updateMapRecommendations(placements);

            // Update timestamp
            document.getElementById('last-update').textContent = new Date().toLocaleString();

        } catch (error) {
            console.error('Error loading data:', error);
        }
    }

    async fetchAPI(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`API error: ${response.statusText}`);
        }
        return await response.json();
    }

    updateDashboard(analysis) {
        if (!analysis || analysis.error) return;

        // Update summary cards
        const overview = analysis.network_overview || {};
        document.getElementById('total-sounders').textContent = overview.n_sounders || '-';
        document.getElementById('total-observations').textContent = overview.n_observations || '-';

        const infoGain = analysis.information_gain || {};
        document.getElementById('information-gain').textContent =
            infoGain.total_information_gain ? this.formatScientific(infoGain.total_information_gain) : '-';
        document.getElementById('uncertainty-reduction').textContent =
            infoGain.relative_uncertainty_reduction ? (infoGain.relative_uncertainty_reduction * 100).toFixed(1) + '%' : '-';

        // Update information gain chart
        if (infoGain.top_contributors) {
            const labels = infoGain.top_contributors.slice(0, 10).map(c => c.sounder_id);
            const data = infoGain.top_contributors.slice(0, 10).map(c => c.contribution * 100);

            this.charts.infoGain.data.labels = labels;
            this.charts.infoGain.data.datasets[0].data = data;
            this.charts.infoGain.update();
        }

        // Update quality distribution chart
        if (overview.quality_tier_distribution) {
            const dist = overview.quality_tier_distribution;
            this.charts.quality.data.datasets[0].data = [
                dist.platinum || 0,
                dist.gold || 0,
                dist.silver || 0,
                dist.bronze || 0
            ];
            this.charts.quality.update();
        }

        // Update upgrade recommendations
        if (analysis.recommendations && analysis.recommendations.upgrades) {
            this.updateUpgradeRecommendations(analysis.recommendations.upgrades);
        }
    }

    updateSounderList(sounders) {
        const tbody = document.getElementById('sounder-tbody');
        tbody.innerHTML = '';

        if (!sounders || sounders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">No sounders available</td></tr>';
            return;
        }

        sounders.forEach(sounder => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${sounder.sounder_id}</td>
                <td><span class="tier-badge ${sounder.quality_tier}">${sounder.quality_tier}</span></td>
                <td>${sounder.n_observations}</td>
                <td>${(sounder.relative_contribution * 100).toFixed(1)}%</td>
                <td>${this.formatScientific(sounder.marginal_gain)}</td>
            `;
            row.style.cursor = 'pointer';
            row.onclick = () => this.showSounderDetail(sounder.sounder_id);
            tbody.appendChild(row);
        });
    }

    updateMapMarkers(sounders) {
        // Clear existing markers
        Object.values(this.markers).forEach(marker => {
            if (marker.type === 'sounder') {
                this.map.removeLayer(marker);
            }
        });

        // Add new markers
        if (!sounders) return;

        sounders.forEach(sounder => {
            const color = this.getTierColor(sounder.quality_tier);
            const marker = L.circleMarker([sounder.latitude, sounder.longitude], {
                radius: 8,
                fillColor: color,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            marker.bindPopup(`
                <strong>${sounder.sounder_id}</strong><br>
                Tier: ${sounder.quality_tier}<br>
                Observations: ${sounder.n_observations}<br>
                Contribution: ${(sounder.relative_contribution * 100).toFixed(1)}%
            `);

            this.markers[sounder.sounder_id] = marker;
            marker.type = 'sounder';
        });
    }

    updateMapRecommendations(placements) {
        // Clear existing recommendation markers
        Object.values(this.markers).forEach(marker => {
            if (marker.type === 'recommendation') {
                this.map.removeLayer(marker);
            }
        });

        if (!placements) return;

        placements.forEach(place => {
            const marker = L.circleMarker([place.latitude, place.longitude], {
                radius: 10,
                fillColor: '#9b59b6',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            marker.bindPopup(`
                <strong>Recommendation #${place.priority}</strong><br>
                Location: ${place.latitude.toFixed(2)}°, ${place.longitude.toFixed(2)}°<br>
                Expected Gain: ${place.expected_gain.toFixed(3)}<br>
                Coverage Gap: ${place.coverage_gap_score.toFixed(3)}<br>
                Redundancy: ${place.redundancy_score.toFixed(3)}
            `);

            this.markers[`rec_${place.priority}`] = marker;
            marker.type = 'recommendation';
        });
    }

    updatePlacementRecommendations(placements) {
        const container = document.getElementById('placement-recommendations');
        container.innerHTML = '';

        if (!placements || placements.length === 0) {
            container.innerHTML = '<div class="text-muted">No recommendations available</div>';
            return;
        }

        placements.forEach(place => {
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.innerHTML = `
                <div class="recommendation-header">
                    <div class="recommendation-title">Priority ${place.priority}</div>
                    <div class="recommendation-score">${place.combined_score.toFixed(3)}</div>
                </div>
                <div class="recommendation-details">
                    📍 ${place.latitude.toFixed(2)}°, ${place.longitude.toFixed(2)}°<br>
                    Expected Gain: ${place.expected_gain.toFixed(3)} |
                    Coverage Gap: ${place.coverage_gap_score.toFixed(3)} |
                    Redundancy: ${place.redundancy_score.toFixed(3)}
                </div>
            `;
            container.appendChild(div);
        });
    }

    updateUpgradeRecommendations(upgrades) {
        const container = document.getElementById('upgrade-recommendations');
        container.innerHTML = '';

        if (!upgrades || upgrades.length === 0) {
            container.innerHTML = '<div class="text-muted">No upgrade recommendations</div>';
            return;
        }

        upgrades.forEach(upgrade => {
            const div = document.createElement('div');
            div.className = 'recommendation-item upgrade';
            div.innerHTML = `
                <div class="recommendation-header">
                    <div class="recommendation-title">${upgrade.sounder_id}</div>
                    <div class="recommendation-score text-warning">
                        ${(upgrade.relative_improvement * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="recommendation-details">
                    ${upgrade.current_tier} → ${upgrade.recommended_tier}<br>
                    Expected Improvement: ${this.formatScientific(upgrade.expected_improvement)}
                </div>
            `;
            container.appendChild(div);
        });
    }

    getTierColor(tier) {
        const colors = {
            'platinum': '#a3d5ff',
            'gold': '#ffd700',
            'silver': '#c0c0c0',
            'bronze': '#cd7f32'
        };
        return colors[tier] || '#7f8c8d';
    }

    formatScientific(value) {
        if (value === 0 || value === null || value === undefined) return '0';
        if (Math.abs(value) < 1000) return value.toFixed(2);
        return value.toExponential(2);
    }

    async showSounderDetail(sounderId) {
        try {
            const detail = await this.fetchAPI(`/api/nvis/sounder/${sounderId}`);
            alert(`
Sounder: ${detail.sounder_id}
Name: ${detail.name}
Operator: ${detail.operator}
Location: ${detail.latitude.toFixed(2)}°, ${detail.longitude.toFixed(2)}°
Equipment: ${detail.equipment_type}
Observations: ${detail.n_observations}
${detail.information_gain ? `Information Gain: ${this.formatScientific(detail.information_gain.marginal_gain)}` : ''}
            `.trim());
        } catch (error) {
            console.error('Error loading sounder detail:', error);
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new NVISDashboard();
});
