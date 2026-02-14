/**
 * System Control JavaScript
 *
 * Handles system control operations:
 * - Service management
 * - Filter control
 * - Data source toggles
 * - System monitoring
 */

let confirmCallback = null;

/**
 * Initialize control view
 */
async function initControlView() {
    await loadAllControlData();

    // Setup auto-refresh (every 30 seconds)
    setInterval(loadSystemHealth, 30000);
}

/**
 * Load all control data
 */
async function loadAllControlData() {
    try {
        await Promise.all([
            loadSystemHealth(),
            loadServices(),
            loadFilterParameters(),
            loadObservationCounts(),
            loadRecentObservations(),
            loadSystemStatistics()
        ]);
    } catch (error) {
        console.error('Error loading control data:', error);
        Utils.showNotification('Failed to load control data', 'error');
    }
}

/**
 * Load system health overview
 */
async function loadSystemHealth() {
    try {
        const health = await api.healthCheck();
        const stats = await api.getSystemStatistics();

        // Update stat cards
        document.getElementById('total-observations').textContent =
            stats.observations_received || 0;
        document.getElementById('grids-received').textContent =
            stats.grids_received || 0;

        // Health status
        const healthElement = document.getElementById('system-health');
        healthElement.textContent = health.status.toUpperCase();
        healthElement.style.color = health.status === 'healthy' ? '#4CAF50' : '#FF9800';

        // Count running services
        const services = await api.getServicesStatus();
        const runningCount = Object.values(services.services).filter(
            s => s.status === 'running'
        ).length;
        document.getElementById('services-running').textContent =
            `${runningCount}/${services.count}`;

    } catch (error) {
        console.error('Error loading system health:', error);
    }
}

/**
 * Load services table
 */
async function loadServices() {
    try {
        const servicesData = await api.getServicesStatus();

        if (servicesData.count === 0) {
            document.getElementById('services-table').innerHTML =
                '<p style="color: #757575;">No services registered</p>';
            return;
        }

        let tableHTML = `
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Service</th>
                            <th>Status</th>
                            <th>Last Update</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        Object.entries(servicesData.services).forEach(([name, info]) => {
            const statusColor = info.status === 'running' ? '#4CAF50' : '#F44336';
            tableHTML += `
                <tr>
                    <td><strong>${name}</strong></td>
                    <td>
                        <span class="badge" style="background: ${statusColor}; color: white;">
                            ${info.status || 'unknown'}
                        </span>
                    </td>
                    <td>${Utils.formatTimestamp(info.last_update)}</td>
                    <td>
                        <div class="button-group">
                            <button class="button button-success" onclick="controlService('${name}', 'start')">Start</button>
                            <button class="button button-error" onclick="controlService('${name}', 'stop')">Stop</button>
                            <button class="button button-secondary" onclick="controlService('${name}', 'restart')">Restart</button>
                        </div>
                    </td>
                </tr>
            `;
        });

        tableHTML += '</tbody></table></div>';
        document.getElementById('services-table').innerHTML = tableHTML;

    } catch (error) {
        console.error('Error loading services:', error);
        document.getElementById('services-table').innerHTML =
            '<p style="color: #F44336;">Failed to load services</p>';
    }
}

/**
 * Control service (start/stop/restart)
 */
async function controlService(serviceName, action) {
    showConfirmation(
        `${action.charAt(0).toUpperCase() + action.slice(1)} Service`,
        `Are you sure you want to ${action} the ${serviceName} service?`,
        async () => {
            try {
                const result = await api.controlService(serviceName, action);
                Utils.showNotification(result.message, 'success');
                await loadServices();
            } catch (error) {
                Utils.showNotification(`Failed to ${action} service: ${error.message}`, 'error');
            }
        }
    );
}

/**
 * Trigger filter cycle
 */
async function triggerFilterCycle() {
    const force = document.getElementById('force-cycle').checked;

    showConfirmation(
        'Trigger Filter Cycle',
        `Are you sure you want to ${force ? 'force ' : ''}trigger a filter cycle?`,
        async () => {
            try {
                const result = await api.triggerFilterCycle(force, 'Manual trigger from dashboard');
                Utils.showNotification(result.message, 'success');
            } catch (error) {
                Utils.showNotification(`Failed to trigger cycle: ${error.message}`, 'error');
            }
        }
    );
}

/**
 * Switch autonomous mode
 */
async function switchMode(mode) {
    showConfirmation(
        `Switch to ${mode} Mode`,
        `Are you sure you want to switch to ${mode} mode? This will override autonomous mode switching.`,
        async () => {
            try {
                const result = await api.switchMode(mode, 'Manual override from dashboard');
                Utils.showNotification(result.message, 'success');
                await loadSystemHealth();
            } catch (error) {
                Utils.showNotification(`Failed to switch mode: ${error.message}`, 'error');
            }
        }
    );
}

/**
 * Load filter parameters and metrics
 */
async function loadFilterParameters() {
    try {
        const metrics = await api.getFilterMetrics();

        let paramsHTML = '<div class="grid-3col">';

        // Display current metrics
        if (metrics.cycle_time) {
            paramsHTML += `
                <div>
                    <strong>Cycle Time:</strong><br>
                    ${metrics.cycle_time.toFixed(2)} seconds
                </div>
            `;
        }

        if (metrics.uncertainty) {
            paramsHTML += `
                <div>
                    <strong>Uncertainty:</strong><br>
                    ${Utils.formatScientific(metrics.uncertainty)}
                </div>
            `;
        }

        if (metrics.nis) {
            paramsHTML += `
                <div>
                    <strong>NIS:</strong><br>
                    ${metrics.nis.toFixed(2)}
                </div>
            `;
        }

        paramsHTML += '</div>';

        document.getElementById('filter-parameters').innerHTML = paramsHTML;

    } catch (error) {
        console.error('Error loading filter parameters:', error);
        document.getElementById('filter-parameters').innerHTML =
            '<p style="color: #F44336;">Failed to load filter metrics</p>';
    }
}

/**
 * Toggle data source
 */
async function toggleDataSource(sourceName) {
    showConfirmation(
        `Toggle ${sourceName}`,
        `Toggle the ${sourceName} data source?`,
        async () => {
            try {
                // Get current state (simplified - would need actual state tracking)
                const enabled = Math.random() > 0.5; // Placeholder
                const result = await api.toggleDataSource(sourceName, !enabled);
                Utils.showNotification(result.message, 'success');
            } catch (error) {
                Utils.showNotification(`Failed to toggle source: ${error.message}`, 'error');
            }
        }
    );
}

/**
 * Load observation counts
 */
async function loadObservationCounts() {
    try {
        const counts = await api.getObservationCounts();

        const countsHTML = `
            <div class="grid-2col">
                <div>
                    <strong>GNSS TEC:</strong> ${counts.gnss_tec || 0}<br>
                    <strong>Ionosonde:</strong> ${counts.ionosonde || 0}<br>
                    <strong>NVIS Sounder:</strong> ${counts.nvis_sounder || 0}
                </div>
                <div>
                    <strong>Total:</strong> ${counts.total || 0}
                </div>
            </div>
        `;

        document.getElementById('observation-counts').innerHTML = countsHTML;

    } catch (error) {
        console.error('Error loading observation counts:', error);
        document.getElementById('observation-counts').innerHTML =
            '<p style="color: #F44336;">Failed to load counts</p>';
    }
}

/**
 * Load recent observations
 */
async function loadRecentObservations() {
    try {
        const obsType = document.getElementById('obs-type-filter').value || null;
        const observations = await api.getRecentObservations(1, obsType);

        if (observations.count === 0) {
            document.getElementById('recent-observations').innerHTML =
                '<p style="color: #757575;">No recent observations</p>';
            return;
        }

        let obsHTML = '<div class="observations-list">';

        observations.data.slice(0, 5).forEach(obs => {
            obsHTML += `
                <div class="obs-item">
                    <div class="obs-type">
                        <span class="badge badge-info">${obs.obs_type}</span>
                    </div>
                    <div class="obs-details">
                        ${obs.value ? `Value: ${obs.value}` : 'Data received'}
                    </div>
                    <div class="obs-time">${Utils.formatTimestamp(obs.timestamp)}</div>
                </div>
            `;
        });

        obsHTML += '</div>';
        document.getElementById('recent-observations').innerHTML = obsHTML;

    } catch (error) {
        console.error('Error loading recent observations:', error);
        document.getElementById('recent-observations').innerHTML =
            '<p style="color: #F44336;">Failed to load observations</p>';
    }
}

/**
 * Load system statistics
 */
async function loadSystemStatistics() {
    try {
        const stats = await api.getSystemStatistics();

        const statsHTML = `
            <div class="grid-4col">
                <div>
                    <strong>Grids Received:</strong> ${stats.grids_received || 0}<br>
                    <strong>Propagation Updates:</strong> ${stats.propagation_updates || 0}
                </div>
                <div>
                    <strong>Space Weather Updates:</strong> ${stats.spaceweather_updates || 0}<br>
                    <strong>Observations:</strong> ${stats.observations_received || 0}
                </div>
                <div>
                    <strong>Alerts Generated:</strong> ${stats.alerts_generated || 0}<br>
                    <strong>Current Mode:</strong> <span class="badge" style="background: ${Utils.getModeColor(stats.current_mode)}; color: white;">${stats.current_mode}</span>
                </div>
                <div>
                    <strong>Grid Age:</strong> ${stats.grid_age_seconds ? Utils.formatRelativeTime(new Date(Date.now() - stats.grid_age_seconds * 1000)) : 'N/A'}<br>
                    <strong>Active Services:</strong> ${stats.active_services || 0}
                </div>
            </div>
        `;

        document.getElementById('system-statistics').innerHTML = statsHTML;

    } catch (error) {
        console.error('Error loading system statistics:', error);
        document.getElementById('system-statistics').innerHTML =
            '<p style="color: #F44336;">Failed to load statistics</p>';
    }
}

/**
 * Show confirmation modal
 */
function showConfirmation(title, message, callback) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-message').textContent = message;
    document.getElementById('confirmation-modal').style.display = 'flex';

    confirmCallback = callback;

    // Setup confirm button
    const confirmButton = document.getElementById('modal-confirm');
    confirmButton.onclick = () => {
        if (confirmCallback) {
            confirmCallback();
        }
        closeModal();
    };
}

/**
 * Close confirmation modal
 */
function closeModal() {
    document.getElementById('confirmation-modal').style.display = 'none';
    confirmCallback = null;
}

// WebSocket handlers for real-time updates
ws.on('grid_update', async () => {
    await loadSystemHealth();
    await loadSystemStatistics();
});

ws.on('observation_update', async () => {
    await loadObservationCounts();
    await loadRecentObservations();
});

// Initialize on page load
window.addEventListener('load', initControlView);
