/**
 * Utility Functions for AutoNVIS Dashboard
 *
 * Common utilities for formatting, color scales, and data processing.
 */

const Utils = {
    /**
     * Format timestamp for display
     */
    formatTimestamp(timestamp, includeSeconds = true) {
        const date = new Date(timestamp);
        const options = {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            ...(includeSeconds && { second: '2-digit' }),
            hour12: false
        };
        return date.toLocaleString('en-US', options);
    },

    /**
     * Format timestamp as relative time (e.g., "5 minutes ago")
     */
    formatRelativeTime(timestamp) {
        const now = new Date();
        const date = new Date(timestamp);
        const seconds = Math.floor((now - date) / 1000);

        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    },

    /**
     * Format number with appropriate precision
     */
    formatNumber(value, decimals = 2) {
        if (typeof value !== 'number' || isNaN(value)) return 'N/A';
        return value.toFixed(decimals);
    },

    /**
     * Format scientific notation
     */
    formatScientific(value, decimals = 2) {
        if (typeof value !== 'number' || isNaN(value)) return 'N/A';
        return value.toExponential(decimals);
    },

    /**
     * Format large numbers with SI prefixes
     */
    formatSI(value, decimals = 2) {
        if (typeof value !== 'number' || isNaN(value)) return 'N/A';

        const prefixes = ['', 'k', 'M', 'G', 'T', 'P'];
        let magnitude = 0;

        while (Math.abs(value) >= 1000 && magnitude < prefixes.length - 1) {
            value /= 1000;
            magnitude++;
        }

        return value.toFixed(decimals) + ' ' + prefixes[magnitude];
    },

    /**
     * Get color for electron density value
     */
    getNeColor(ne, neMin, neMax) {
        // Plasma colorscale for electron density
        const normalized = (ne - neMin) / (neMax - neMin);
        return this.plasmaColorscale(normalized);
    },

    /**
     * Plasma colorscale (similar to matplotlib plasma)
     */
    plasmaColorscale(t) {
        // Simplified plasma colorscale
        const r = Math.floor(255 * Math.min(1, Math.max(0, (1.5 - Math.abs(t - 0.5) * 2))));
        const g = Math.floor(255 * Math.min(1, Math.max(0, (t * 2 - 0.5))));
        const b = Math.floor(255 * Math.min(1, Math.max(0, (1.2 - t * 1.5))));
        return `rgb(${r}, ${g}, ${b})`;
    },

    /**
     * Get color for flare class
     */
    getFlareColor(flareClass) {
        const colors = {
            'A': '#90EE90',  // Light green
            'B': '#FFD700',  // Gold
            'C': '#FFA500',  // Orange
            'M': '#FF6347',  // Tomato
            'X': '#DC143C'   // Crimson
        };
        return colors[flareClass] || '#CCCCCC';
    },

    /**
     * Get color for quality tier
     */
    getQualityTierColor(tier) {
        const colors = {
            'platinum': '#E5E4E2',  // Platinum
            'gold': '#FFD700',      // Gold
            'silver': '#C0C0C0',    // Silver
            'bronze': '#CD7F32',    // Bronze
            'unknown': '#808080'    // Gray
        };
        return colors[tier.toLowerCase()] || '#808080';
    },

    /**
     * Get color for autonomous mode
     */
    getModeColor(mode) {
        return mode === 'QUIET' ? '#4CAF50' : '#FF9800';
    },

    /**
     * Get color for severity level
     */
    getSeverityColor(severity) {
        const colors = {
            'info': '#2196F3',
            'warning': '#FF9800',
            'error': '#F44336',
            'critical': '#9C27B0'
        };
        return colors[severity.toLowerCase()] || '#757575';
    },

    /**
     * Debounce function calls
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Show loading indicator
     */
    showLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="loading-spinner">Loading...</div>';
        }
    },

    /**
     * Show error message
     */
    showError(elementId, message) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `<div class="error-message">Error: ${message}</div>`;
        }
    },

    /**
     * Show success notification
     */
    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.classList.add('fade-out');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    },

    /**
     * Convert 2D array to Plotly format
     */
    arrayToPlotly(array2D) {
        // Transpose for Plotly's expected format
        return array2D[0].map((_, colIndex) => array2D.map(row => row[colIndex]));
    },

    /**
     * Create Plotly colorscale
     */
    createColorscale(name = 'Viridis') {
        const scales = {
            'Viridis': [[0, '#440154'], [0.25, '#31688e'], [0.5, '#35b779'], [0.75, '#fde724'], [1, '#fde724']],
            'Plasma': [[0, '#0d0887'], [0.25, '#7e03a8'], [0.5, '#cc4778'], [0.75, '#f89540'], [1, '#f0f921']],
            'Jet': [[0, '#00007F'], [0.25, '#0000FF'], [0.5, '#00FF00'], [0.75, '#FF0000'], [1, '#7F0000']],
            'RdBu': [[0, '#053061'], [0.25, '#2166ac'], [0.5, '#f7f7f7'], [0.75, '#b2182b'], [1, '#67001f']]
        };
        return scales[name] || scales['Viridis'];
    },

    /**
     * Download data as JSON file
     */
    downloadJSON(data, filename) {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    },

    /**
     * Download data as CSV file
     */
    downloadCSV(data, filename) {
        // Assume data is array of objects
        if (!data || data.length === 0) return;

        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(h => row[h]).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    },

    /**
     * Calculate statistics from array
     */
    calculateStats(array) {
        if (!array || array.length === 0) {
            return { min: null, max: null, mean: null, std: null };
        }

        const min = Math.min(...array);
        const max = Math.max(...array);
        const mean = array.reduce((a, b) => a + b, 0) / array.length;
        const variance = array.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / array.length;
        const std = Math.sqrt(variance);

        return { min, max, mean, std };
    },

    /**
     * Interpolate between two colors
     */
    interpolateColor(color1, color2, factor) {
        const c1 = color1.match(/\d+/g).map(Number);
        const c2 = color2.match(/\d+/g).map(Number);
        const r = Math.round(c1[0] + factor * (c2[0] - c1[0]));
        const g = Math.round(c1[1] + factor * (c2[1] - c1[1]));
        const b = Math.round(c1[2] + factor * (c2[2] - c1[2]));
        return `rgb(${r}, ${g}, ${b})`;
    }
};

// Plotly default configuration
const PlotlyDefaultConfig = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false
};

// Plotly default layout
const PlotlyDefaultLayout = {
    font: { family: 'Arial, sans-serif', size: 12 },
    margin: { l: 60, r: 40, t: 40, b: 60 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#f8f9fa',
    hovermode: 'closest'
};
