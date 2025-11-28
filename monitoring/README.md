# RingRift Monitoring Configuration

This directory contains the Prometheus and Alertmanager configurations for monitoring the RingRift application.

## Directory Structure

```
monitoring/
├── README.md                      # This file
├── prometheus/
│   ├── prometheus.yml             # Main Prometheus configuration
│   └── alerts.yml                 # Alerting rules (616+ alert rules)
└── alertmanager/
    └── alertmanager.yml           # Alertmanager routing and receivers
```

## Configuration Files

### Prometheus Configuration (`prometheus/prometheus.yml`)

Main Prometheus server configuration including:

- Global settings (scrape interval, evaluation interval)
- Rule files location
- Alertmanager configuration
- Scrape targets (app, AI service, etc.)

### Prometheus Alerts (`prometheus/alerts.yml`)

Comprehensive alerting rules organized by category:

- **Availability**: Service up/down, critical error rates
- **Latency**: P50, P95, P99 response time thresholds
- **Resources**: Memory usage, event loop lag
- **Business**: Game activity, WebSocket connections
- **AI Service**: Fallback rates, request latency
- **Degradation**: Service status changes
- **Rate Limiting**: Excessive rate limit hits
- **Rules Parity**: TS/Python engine consistency

### Alertmanager Configuration (`alertmanager/alertmanager.yml`)

Alert routing and notification configuration:

- Receiver definitions (Slack, email, PagerDuty)
- Routing rules by severity and team
- Inhibition rules to prevent alert storms
- Grouping and notification timing

## Validation

### Running Locally

Validate monitoring configurations before committing:

```bash
# Using npm script (recommended)
npm run validate:monitoring

# Or directly run the script
./scripts/validate-monitoring-configs.sh

# With options
./scripts/validate-monitoring-configs.sh --help      # Show help
./scripts/validate-monitoring-configs.sh --verbose   # Verbose output
./scripts/validate-monitoring-configs.sh --yaml-only # YAML syntax only (no Docker)
./scripts/validate-monitoring-configs.sh --docker-only # Full validation (requires Docker)
```

### What Gets Validated

1. **YAML Syntax**: Basic syntax validation to catch malformed YAML
2. **Prometheus Config**: Using official `promtool check config`
3. **Alert Rules**: Using official `promtool check rules` (validates PromQL)
4. **Alertmanager Config**: Using official `amtool check-config`
5. **Common Issues**: Hardcoded secrets, tabs vs spaces, short intervals

### CI Integration

Validation runs automatically in GitHub Actions when:

- Any file in `monitoring/**` is modified
- The validation script itself is modified

The CI job uses Docker containers with official Prometheus/Alertmanager images for accurate validation using `promtool` and `amtool`.

## Making Changes

### Best Practices

1. **Test locally first**: Run `npm run validate:monitoring` before committing
2. **Use environment variables**: Never hardcode secrets (use `${VAR}` syntax)
3. **Use spaces, not tabs**: YAML prefers 2-space indentation
4. **Set reasonable intervals**:
   - `scrape_interval`: ≥ 15s recommended
   - `group_wait`: ≥ 30s to avoid alert storms
5. **Document changes**: Update this README if adding new alert categories

### Adding New Alert Rules

1. Add rules to `prometheus/alerts.yml` under the appropriate group
2. Follow the existing format with:
   - Clear `alert` name
   - Valid PromQL `expr`
   - Appropriate `for` duration
   - Labels: `severity`, `team`, `component`
   - Annotations: `summary`, `description`, `runbook_url`, `impact`

Example:

```yaml
- alert: NewAlertName
  expr: metric_name > threshold
  for: 5m
  labels:
    severity: warning
    team: backend
  annotations:
    summary: 'Brief one-line summary'
    description: 'Detailed description of what this alert means.'
    runbook_url: 'https://github.com/ringrift/ringrift/blob/main/docs/runbooks/ALERT_NAME.md'
    impact: 'What is affected when this alert fires.'
```

### Adding New Scrape Targets

1. Add to `prometheus/prometheus.yml` under `scrape_configs`
2. Include appropriate labels for service and component
3. Set reasonable `scrape_interval` based on metric importance

### Modifying Alert Routing

1. Update `alertmanager/alertmanager.yml`
2. Add new receivers or routes as needed
3. Consider inhibition rules to prevent cascading alerts

## Troubleshooting

### Validation Fails Locally

```bash
# Check if Docker is running
docker info

# Try YAML-only validation if Docker isn't available
./scripts/validate-monitoring-configs.sh --yaml-only

# Check specific error messages in verbose mode
./scripts/validate-monitoring-configs.sh --verbose
```

### Common Errors

| Error              | Solution                                               |
| ------------------ | ------------------------------------------------------ |
| YAML syntax error  | Check indentation, quotes, and special characters      |
| PromQL parse error | Verify metric names and query syntax                   |
| Unknown receiver   | Ensure receiver is defined in `receivers` section      |
| Invalid duration   | Use valid Go duration format (e.g., `5m`, `1h`, `30s`) |

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [RingRift Alerting Thresholds](../docs/ALERTING_THRESHOLDS.md)
- [RingRift Runbooks](../docs/runbooks/)

---

**Last Updated**: November 2025
**Maintainer**: RingRift Development Team
