#!/bin/bash
# Cleanup Deprecated Scripts
# Run this script to remove scripts superseded by P2P orchestrator
#
# Usage:
#   ./scripts/cleanup_deprecated.sh --dry-run   # Preview what will be deleted
#   ./scripts/cleanup_deprecated.sh             # Actually delete files
#
# Created: 2025-12-19
# See: scripts/DEPRECATED.md for full details

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be deleted ==="
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DELETED_COUNT=0

delete_file() {
    local file="$1"
    if [[ -f "$SCRIPT_DIR/$file" ]]; then
        if $DRY_RUN; then
            echo "[DRY-RUN] Would delete: $file"
        else
            echo "Deleting: $file"
            rm -f "$SCRIPT_DIR/$file"
        fi
        ((DELETED_COUNT++))
    fi
}

echo ""
echo "=== Cleaning up deprecated monitoring scripts ==="
delete_file "cluster_monitor.py"
delete_file "cluster_monitor.sh"
delete_file "cluster_monitor_daemon.sh"
delete_file "cluster_monitor_unified.sh"
delete_file "cluster_monitoring.sh"
delete_file "cluster_health_monitor.sh"
delete_file "monitor_10h.sh"
delete_file "monitor_10h_enhanced.sh"
delete_file "monitor_all_jobs.sh"
delete_file "monitor_and_test.sh"

echo ""
echo "=== Cleaning up deprecated cluster management scripts ==="
delete_file "cluster_automation.py"
delete_file "cluster_control.py"
delete_file "cluster_manager.py"
delete_file "cluster_health_check.py"
delete_file "cluster_health_check.sh"
delete_file "cluster_ssh_init.py"
delete_file "cluster_sync_coordinator.py"

echo ""
echo "=== Cleaning up deprecated training orchestration scripts ==="
delete_file "training_orchestrator.py"
delete_file "job_scheduler.py"

echo ""
echo "=== Cleaning up deprecated unified scripts (superseded by P2P) ==="
delete_file "unified_work_orchestrator.py"
delete_file "unified_cluster_monitor.py"
delete_file "cluster_auto_recovery.py"

echo ""
echo "=== Cleaning up deprecated data collection scripts ==="
delete_file "collect_and_merge_selfplay.sh"
delete_file "collect_diverse_selfplay.sh"
delete_file "collect_selfplay_results.sh"
delete_file "cron_aggregate.sh"
delete_file "cron_diverse_selfplay.sh"
delete_file "cron_sync_selfplay.sh"
delete_file "cron_training.sh"

echo ""
echo "=== Summary ==="
if $DRY_RUN; then
    echo "Would delete $DELETED_COUNT files"
    echo "Run without --dry-run to actually delete"
else
    echo "Deleted $DELETED_COUNT files"
fi
echo ""
