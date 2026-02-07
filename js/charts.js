// Charts Module for Developer Mode
class BugCharts {
    constructor() {
        this.charts = {};
        this.init();
    }

    init() {
        // Initialize all charts
        this.initBugDistributionChart();
        this.initSeverityTrendChart();
        this.initComplexityChart();
    }

    initBugDistributionChart() {
        const ctx = document.getElementById('bugChart');
        if (!ctx) return;

        this.charts.bugDistribution = new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'High', 'Medium', 'Low'],
                datasets: [{
                    data: [25, 25, 25, 25],
                    backgroundColor: [
                        'rgba(255, 68, 68, 0.8)',
                        'rgba(255, 170, 0, 0.8)',
                        'rgba(255, 255, 0, 0.8)',
                        'rgba(0, 255, 136, 0.8)'
                    ],
                    borderColor: [
                        'rgba(255, 68, 68, 1)',
                        'rgba(255, 170, 0, 1)',
                        'rgba(255, 255, 0, 1)',
                        'rgba(0, 255, 136, 1)'
                    ],
                    borderWidth: 2,
                    hoverOffset: 15
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                size: 12,
                                family: "'Segoe UI', sans-serif"
                            },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#00dbde',
                        borderWidth: 1,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateScale: true,
                    animateRotate: true,
                    duration: 1000
                }
            }
        });
    }

    updateBugDistribution(severity) {
        const chart = this.charts.bugDistribution;
        if (!chart) return;

        const severityMap = {
            'critical': [80, 10, 5, 5],
            'high': [10, 70, 10, 10],
            'medium': [5, 15, 60, 20],
            'low': [2, 8, 20, 70]
        };

        const newData = severityMap[severity] || [25, 25, 25, 25];

        // Animate update
        chart.data.datasets[0].data = newData;
        chart.update('active');
    }

    initSeverityTrendChart() {
        const ctx = document.getElementById('severityTrendChart');
        if (!ctx) return;

        // This would be used if we had a trend chart
        const labels = Array.from({ length: 10 }, (_, i) => `Analysis ${i + 1}`);
        const data = Array.from({ length: 10 }, () => Math.random() * 100);

        this.charts.severityTrend = new Chart(ctx.getContext('2d'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Bug Probability Trend',
                    data: data,
                    borderColor: '#00dbde',
                    backgroundColor: 'rgba(0, 219, 222, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#fc00ff',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#b0b0ff'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#b0b0ff',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initComplexityChart() {
        const ctx = document.getElementById('complexityChart');
        if (!ctx) return;

        // This would show complexity metrics
        this.charts.complexity = new Chart(ctx.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Cyclomatic', 'Cognitive', 'Maintainability', 'Duplication', 'Size'],
                datasets: [{
                    label: 'Current Code',
                    data: [65, 59, 90, 81, 56],
                    backgroundColor: 'rgba(0, 219, 222, 0.2)',
                    borderColor: '#00dbde',
                    borderWidth: 2,
                    pointBackgroundColor: '#00dbde'
                }, {
                    label: 'Ideal Range',
                    data: [80, 80, 80, 80, 80],
                    backgroundColor: 'rgba(252, 0, 255, 0.1)',
                    borderColor: '#fc00ff',
                    borderWidth: 1,
                    pointBackgroundColor: '#fc00ff',
                    borderDash: [5, 5]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: '#b0b0ff'
                        },
                        ticks: {
                            color: '#b0b0ff',
                            backdropColor: 'transparent'
                        }
                    }
                }
            }
        });
    }

    updateComplexityChart(complexityScore) {
        const chart = this.charts.complexity;
        if (!chart) return;

        // Update radar chart based on complexity
        const normalizedScore = Math.min(100, complexityScore * 20);
        chart.data.datasets[0].data = [
            normalizedScore,
            normalizedScore * 0.8,
            100 - normalizedScore * 0.6,
            100 - normalizedScore * 0.4,
            normalizedScore * 0.5
        ];

        chart.update();
    }

    createProgressChart(elementId, value, max = 100) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;

        const percentage = (value / max) * 100;

        return new Chart(ctx.getContext('2d'), {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [percentage, 100 - percentage],
                    backgroundColor: [
                        this.getSeverityColor(percentage),
                        'rgba(255, 255, 255, 0.1)'
                    ],
                    borderWidth: 0,
                    circumference: 270,
                    rotation: 225
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '80%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    getSeverityColor(value) {
        if (value >= 80) return '#ff4444'; // Critical
        if (value >= 60) return '#ffaa00'; // High
        if (value >= 40) return '#ffff00'; // Medium
        return '#00ff88'; // Low
    }

    createComparisonChart(elementId, data1, data2, labels) {
        const ctx = document.getElementById(elementId);
        if (!ctx) return null;

        return new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Before Fix',
                    data: data1,
                    backgroundColor: 'rgba(255, 68, 68, 0.7)',
                    borderColor: '#ff4444',
                    borderWidth: 1
                }, {
                    label: 'After Fix',
                    data: data2,
                    backgroundColor: 'rgba(0, 255, 136, 0.7)',
                    borderColor: '#00ff88',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#b0b0ff'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#b0b0ff'
                        }
                    }
                }
            }
        });
    }

    // Utility function to update chart colors based on theme
    updateChartTheme(isDark = true) {
        const textColor = isDark ? '#ffffff' : '#000000';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

        Object.values(this.charts).forEach(chart => {
            if (chart.options?.plugins?.legend?.labels) {
                chart.options.plugins.legend.labels.color = textColor;
            }

            if (chart.options?.scales) {
                Object.values(chart.options.scales).forEach(scale => {
                    if (scale.grid) {
                        scale.grid.color = gridColor;
                    }
                    if (scale.ticks) {
                        scale.ticks.color = textColor;
                    }
                });
            }

            chart.update();
        });
    }

    // Destroy all charts
    destroyAll() {
        Object.values(this.charts).forEach(chart => {
            chart.destroy();
        });
        this.charts = {};
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.BugCharts = BugCharts;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = BugCharts;
}