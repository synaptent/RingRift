# Configuration Backup Files

This directory contains historical backup files from the RingRift AI service configuration.

## Files Moved (December 26, 2025)

### From ai-service/config/
- `notification_hooks.yaml.bak` - Backup of notification hooks configuration
- `distributed_hosts.yaml.bak_1766332604` - Backup of distributed hosts configuration (timestamp: 1766332604)

### From ai-service/archive/deprecated_config/
- `unified_loop.optimized.yaml.bak` - Backup of unified loop configuration
- `nginx-cluster.conf.bak` - Backup of nginx cluster configuration
- `pipeline.json.bak` - Backup of pipeline configuration

## Purpose

These backups were created during various configuration changes and updates. They are preserved here for historical reference and rollback purposes if needed.

## Usage

To restore a backup:
1. Identify the backup file you need
2. Copy (do NOT move) to the appropriate location in the active config directories
3. Remove the `.bak` or timestamp suffix
4. Verify the configuration is valid before deploying

## Cleanup Policy

Backups older than 6 months may be removed unless they represent significant configuration milestones.
