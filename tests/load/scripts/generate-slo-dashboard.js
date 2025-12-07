#!/usr/bin/env node

/**
 * RingRift SLO Dashboard Generator
 * 
 * Generates a visual HTML dashboard from SLO verification reports.
 * 
 * Usage:
 *   node generate-slo-dashboard.js <slo_report.json> [output.html]
 * 
 * Examples:
 *   node generate-slo-dashboard.js results/baseline_staging_20251207_slo_report.json
 *   node generate-slo-dashboard.js results/baseline_staging_20251207_slo_report.json dashboard.html
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const reportFile = process.argv[2];
const outputFile = process.argv[3];

if (!reportFile || process.argv.includes('--help') || process.argv.includes('-h')) {
  console.log(`
RingRift SLO Dashboard Generator

Usage:
  node generate-slo-dashboard.js <slo_report.json> [output.html]

Arguments:
  slo_report.json   Path to SLO verification report JSON
  output.html       Output HTML file (default: replaces .json with .html)

Examples:
  node generate-slo-dashboard.js results/baseline_staging_20251207_slo_report.json
  node generate-slo-dashboard.js results/baseline_staging_20251207_slo_report.json dashboard.html
`);
  process.exit(reportFile ? 0 : 1);
}

if (!fs.existsSync(reportFile)) {
  console.error(`Error: Report file not found: ${reportFile}`);
  process.exit(1);
}

// Load the SLO report
let report;
try {
  report = JSON.parse(fs.readFileSync(reportFile, 'utf8'));
} catch (error) {
  console.error(`Error parsing report file: ${error.message}`);
  process.exit(1);
}

/**
 * Get color based on SLO status and priority
 */
function getStatusColor(slo) {
  if (slo.passed) return '#22c55e'; // green-500
  if (slo.priority === 'critical') return '#dc2626'; // red-600
  if (slo.priority === 'high') return '#ea580c'; // orange-600
  return '#eab308'; // yellow-500
}

/**
 * Get background color for cards
 */
function getCardBackground(slo) {
  if (slo.passed) return '#f0fdf4'; // green-50
  if (slo.priority === 'critical') return '#fef2f2'; // red-50
  if (slo.priority === 'high') return '#fff7ed'; // orange-50
  return '#fefce8'; // yellow-50
}

/**
 * Get border color for cards
 */
function getCardBorder(slo) {
  if (slo.passed) return '#86efac'; // green-300
  if (slo.priority === 'critical') return '#fca5a5'; // red-300
  if (slo.priority === 'high') return '#fdba74'; // orange-300
  return '#fde047'; // yellow-300
}

/**
 * Format value with unit for display
 */
function formatValue(value, unit) {
  if (unit === 'percent') return `${value}%`;
  if (unit === 'ms') return `${value}ms`;
  if (unit === 'games') return `${value}`;
  if (unit === 'players') return `${value}`;
  if (unit === 'count') return `${value}`;
  return `${value}`;
}

/**
 * Get progress percentage (capped at 100%)
 */
function getProgressPercent(slo) {
  if (slo.target === 0) return slo.actual === 0 ? 100 : 0;
  
  // For latency/error metrics, lower is better
  if (slo.unit === 'ms' || (slo.unit === 'percent' && slo.name.includes('Error'))) {
    if (slo.actual <= slo.target) return 100;
    return Math.max(0, 100 - ((slo.actual - slo.target) / slo.target) * 100);
  }
  
  // For availability/capacity metrics, higher is better
  if (slo.name.includes('Availability') || slo.name.includes('Capacity') || slo.name.includes('Success')) {
    return Math.min(100, (slo.actual / slo.target) * 100);
  }
  
  // For count metrics (should be 0)
  if (slo.unit === 'count') {
    return slo.actual === 0 ? 100 : 0;
  }
  
  // Default: lower is better
  if (slo.actual <= slo.target) return 100;
  return Math.max(0, 100 - ((slo.actual - slo.target) / slo.target) * 100);
}

/**
 * Generate the HTML dashboard
 */
function generateDashboard(report) {
  const slos = report.slos;
  const passedCount = Object.values(slos).filter(s => s.passed).length;
  const totalCount = Object.keys(slos).length;
  const allPassed = report.all_passed;
  
  const criticalBreaches = Object.values(slos).filter(s => !s.passed && s.priority === 'critical');
  const highBreaches = Object.values(slos).filter(s => !s.passed && s.priority === 'high');
  const mediumBreaches = Object.values(slos).filter(s => !s.passed && s.priority === 'medium');
  
  // Group SLOs by category
  const latencySLOs = Object.entries(slos).filter(([key]) => key.startsWith('latency'));
  const capacitySLOs = Object.entries(slos).filter(([key]) => 
    key.includes('concurrent') || key.includes('capacity'));
  const reliabilitySLOs = Object.entries(slos).filter(([key]) => 
    key.includes('availability') || key.includes('error') || 
    key.includes('contract') || key.includes('lifecycle') || key.includes('stall'));

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RingRift SLO Dashboard</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: #f8fafc;
      color: #1e293b;
      line-height: 1.5;
    }
    
    .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    
    .header {
      background: ${allPassed ? '#22c55e' : criticalBreaches.length > 0 ? '#dc2626' : '#ea580c'};
      color: white;
      padding: 32px;
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    
    .header h1 {
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    
    .header-meta {
      opacity: 0.9;
      font-size: 14px;
    }
    
    .header-stats {
      display: flex;
      gap: 32px;
      margin-top: 16px;
    }
    
    .header-stat {
      text-align: center;
    }
    
    .header-stat-value {
      font-size: 36px;
      font-weight: 700;
    }
    
    .header-stat-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      opacity: 0.8;
    }
    
    .section {
      margin-bottom: 32px;
    }
    
    .section-title {
      font-size: 18px;
      font-weight: 600;
      color: #475569;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 2px solid #e2e8f0;
    }
    
    .slo-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }
    
    .slo-card {
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
      border-left: 4px solid;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .slo-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px 0 rgb(0 0 0 / 0.15);
    }
    
    .slo-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 12px;
    }
    
    .slo-name {
      font-weight: 600;
      font-size: 14px;
      color: #334155;
    }
    
    .slo-priority {
      font-size: 10px;
      padding: 2px 8px;
      border-radius: 9999px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    
    .priority-critical {
      background: #fee2e2;
      color: #dc2626;
    }
    
    .priority-high {
      background: #ffedd5;
      color: #ea580c;
    }
    
    .priority-medium {
      background: #e0e7ff;
      color: #4f46e5;
    }
    
    .slo-values {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
    }
    
    .slo-actual {
      font-size: 28px;
      font-weight: 700;
    }
    
    .slo-target {
      font-size: 14px;
      color: #64748b;
    }
    
    .slo-progress {
      height: 6px;
      background: #e2e8f0;
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    
    .slo-progress-bar {
      height: 100%;
      border-radius: 3px;
      transition: width 0.3s ease;
    }
    
    .slo-status {
      font-size: 12px;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    
    .slo-status-icon {
      font-size: 14px;
    }
    
    .slo-note {
      font-size: 11px;
      color: #94a3b8;
      margin-top: 4px;
      font-style: italic;
    }
    
    .breaches-summary {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }
    
    .breach-card {
      padding: 20px;
      border-radius: 12px;
      text-align: center;
    }
    
    .breach-card-critical {
      background: #fef2f2;
      border: 1px solid #fecaca;
    }
    
    .breach-card-high {
      background: #fff7ed;
      border: 1px solid #fed7aa;
    }
    
    .breach-card-medium {
      background: #f0f9ff;
      border: 1px solid #bae6fd;
    }
    
    .breach-count {
      font-size: 32px;
      font-weight: 700;
    }
    
    .breach-label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #64748b;
    }
    
    .footer {
      text-align: center;
      padding: 24px;
      color: #94a3b8;
      font-size: 12px;
    }
    
    @media (max-width: 768px) {
      .container {
        padding: 16px;
      }
      
      .header-stats {
        flex-direction: column;
        gap: 16px;
      }
      
      .breaches-summary {
        grid-template-columns: 1fr;
      }
      
      .slo-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>${allPassed ? '✅ All SLOs Met' : criticalBreaches.length > 0 ? '❌ Critical SLO Breaches Detected' : '⚠️ SLO Breaches Detected'}</h1>
      <div class="header-meta">
        <div>Environment: <strong>${report.environment || 'staging'}</strong></div>
        <div>Generated: ${new Date(report.timestamp).toLocaleString()}</div>
        <div>Source: ${path.basename(report.source_file || reportFile)}</div>
      </div>
      <div class="header-stats">
        <div class="header-stat">
          <div class="header-stat-value">${passedCount}/${totalCount}</div>
          <div class="header-stat-label">SLOs Passed</div>
        </div>
        <div class="header-stat">
          <div class="header-stat-value">${criticalBreaches.length}</div>
          <div class="header-stat-label">Critical Breaches</div>
        </div>
        <div class="header-stat">
          <div class="header-stat-value">${highBreaches.length}</div>
          <div class="header-stat-label">High Priority Breaches</div>
        </div>
      </div>
    </div>
    
    ${(criticalBreaches.length > 0 || highBreaches.length > 0 || mediumBreaches.length > 0) ? `
    <div class="breaches-summary">
      <div class="breach-card breach-card-critical">
        <div class="breach-count" style="color: #dc2626">${criticalBreaches.length}</div>
        <div class="breach-label">Critical</div>
      </div>
      <div class="breach-card breach-card-high">
        <div class="breach-count" style="color: #ea580c">${highBreaches.length}</div>
        <div class="breach-label">High Priority</div>
      </div>
      <div class="breach-card breach-card-medium">
        <div class="breach-count" style="color: #0284c7">${mediumBreaches.length}</div>
        <div class="breach-label">Medium Priority</div>
      </div>
    </div>
    ` : ''}
    
    <div class="section">
      <h2 class="section-title">⚡ Latency SLOs</h2>
      <div class="slo-grid">
        ${latencySLOs.map(([key, slo]) => generateSLOCard(slo)).join('')}
      </div>
    </div>
    
    <div class="section">
      <h2 class="section-title">📊 Capacity SLOs</h2>
      <div class="slo-grid">
        ${capacitySLOs.map(([key, slo]) => generateSLOCard(slo)).join('')}
      </div>
    </div>
    
    <div class="section">
      <h2 class="section-title">🛡️ Reliability SLOs</h2>
      <div class="slo-grid">
        ${reliabilitySLOs.map(([key, slo]) => generateSLOCard(slo)).join('')}
      </div>
    </div>
    
    <div class="footer">
      <p>RingRift SLO Verification Framework v${report.slo_definitions_version || '1.0.0'}</p>
      <p>Report generated from load test results</p>
    </div>
  </div>
</body>
</html>`;

  return html;
}

/**
 * Generate an individual SLO card
 */
function generateSLOCard(slo) {
  const statusColor = getStatusColor(slo);
  const cardBg = getCardBackground(slo);
  const cardBorder = getCardBorder(slo);
  const progress = getProgressPercent(slo);
  const statusIcon = slo.passed ? '✅' : '❌';
  const statusText = slo.passed ? 'Passed' : 'Breached';
  
  return `
    <div class="slo-card" style="background: ${cardBg}; border-left-color: ${cardBorder}">
      <div class="slo-header">
        <div class="slo-name">${slo.name}</div>
        <span class="slo-priority priority-${slo.priority}">${slo.priority}</span>
      </div>
      <div class="slo-values">
        <span class="slo-actual" style="color: ${statusColor}">${formatValue(slo.actual, slo.unit)}</span>
        <span class="slo-target">Target: ${formatValue(slo.target, slo.unit)}</span>
      </div>
      <div class="slo-progress">
        <div class="slo-progress-bar" style="width: ${progress}%; background: ${statusColor}"></div>
      </div>
      <div class="slo-status">
        <span class="slo-status-icon">${statusIcon}</span>
        <span style="color: ${statusColor}">${statusText}</span>
      </div>
      ${slo.note ? `<div class="slo-note">⚠️ ${slo.note}</div>` : ''}
    </div>
  `;
}

// Generate the dashboard
const html = generateDashboard(report);

// Determine output file
const outputPath = outputFile || reportFile.replace('.json', '.html');

// Write the HTML
try {
  fs.writeFileSync(outputPath, html);
  console.log(`✅ Dashboard generated: ${outputPath}`);
} catch (error) {
  console.error(`Error writing dashboard: ${error.message}`);
  process.exit(1);
}