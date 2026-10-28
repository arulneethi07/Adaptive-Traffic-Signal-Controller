// Analytics & Comparison Charts Manager
class TrafficChartsManager {
  constructor() {
    this.mainChart = null;
    this.sparklineChart = null;
    this.currentMetric = 'waiting_time'; // 'waiting_time', 'queue_length', 'throughput'
    this.historyData = [];
  }

  init(historyData) {
    this.historyData = historyData;
    this.renderMainChart();
    this.renderSparkline();
  }

  setMetric(metric) {
    this.currentMetric = metric;
    if (this.mainChart) {
      this.updateMainChart();
    }
  }

  renderMainChart() {
    const ctx = document.getElementById('mainComparisonChart');
    if (!ctx || typeof Chart === 'undefined') return;

    const labels = this.historyData.map(d => d.hour_label);
    let adaptiveData = [];
    let fixedData = [];
    let yAxisLabel = "Waiting Time (seconds)";

    if (this.currentMetric === 'waiting_time') {
      adaptiveData = this.historyData.map(d => d.adaptive_waiting_time);
      fixedData = this.historyData.map(d => d.fixed_waiting_time);
      yAxisLabel = "Average Delay (seconds)";
    } else if (this.currentMetric === 'queue_length') {
      adaptiveData = this.historyData.map(d => d.adaptive_queue_length);
      fixedData = this.historyData.map(d => d.fixed_queue_length);
      yAxisLabel = "Queue Length (vehicles)";
    } else if (this.currentMetric === 'throughput') {
      adaptiveData = this.historyData.map(d => d.throughput_count);
      fixedData = this.historyData.map(d => Math.round(d.throughput_count * 0.83));
      yAxisLabel = "Throughput (veh/hr)";
    }

    if (this.mainChart) {
      this.mainChart.destroy();
    }

    this.mainChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Adaptive Signal Plan (SmartFlow)',
            data: adaptiveData,
            borderColor: '#0284c7',
            backgroundColor: 'rgba(2, 132, 199, 0.12)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointBackgroundColor: '#0284c7'
          },
          {
            label: 'Fixed-Time Baseline Plan',
            data: fixedData,
            borderColor: '#94a3b8',
            borderWidth: 2,
            borderDash: [5, 5],
            fill: false,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#94a3b8'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              boxWidth: 12,
              color: '#334155',
              font: { family: 'Inter', size: 12, weight: '600' }
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            borderColor: '#334155',
            borderWidth: 1,
            titleColor: '#ffffff',
            bodyColor: '#e2e8f0',
            titleFont: { family: 'Inter', size: 12, weight: '700' },
            bodyFont: { family: 'Inter', size: 12 }
          }
        },
        scales: {
          x: {
            ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
            grid: { display: false }
          },
          y: {
            title: {
              display: true,
              text: yAxisLabel,
              color: '#0f172a',
              font: { family: 'Inter', size: 12, weight: '700' }
            },
            ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } },
            grid: { color: '#e2e8f0' },
            beginAtZero: true
          }
        }
      }
    });
  }

  updateMainChart() {
    this.renderMainChart();
  }

  renderSparkline() {
    const ctx = document.getElementById('sparklineDelay');
    if (!ctx || typeof Chart === 'undefined') return;

    const data = [38, 35, 32, 28, 25, 21, 18.2];
    const labels = ['60m', '50m', '40m', '30m', '20m', '10m', 'Now'];

    if (this.sparklineChart) {
      this.sparklineChart.destroy();
    }

    this.sparklineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          borderColor: '#16a34a',
          borderWidth: 2,
          fill: false,
          tension: 0.4,
          pointRadius: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { x: { display: false }, y: { display: false } }
      }
    });
  }
}
