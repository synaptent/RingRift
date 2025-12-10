#!/bin/bash
# sync_selfplay_data.sh - Sync selfplay data from all remote machines to local
#
# This script reads host configuration from config/remote_hosts.yaml and syncs
# game databases from all configured remote hosts to a local directory.
#
# Usage:
#   ./scripts/sync_selfplay_data.sh [OPTIONS]
#
# Options:
#   --merge           After syncing, merge all DBs into a single merged.db
#   --to-mac-studio   Also sync merged DB to Mac Studio for training
#   --to-lambda       Also sync merged DB to Lambda GPU for training
#   --config FILE     Use custom config file (default: config/remote_hosts.yaml)
#   --dry-run         Show what would be synced without actually syncing
#   -h, --help        Show this help message
#
# Configuration:
#   Copy config/remote_hosts.example.yaml to config/remote_hosts.yaml and
#   customize with your actual instance details. See the example file for
#   documentation of all available options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$AI_SERVICE_DIR/config"
DATA_DIR="$AI_SERVICE_DIR/data/games"

# Default config file
CONFIG_FILE="$CONFIG_DIR/remote_hosts.yaml"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SYNC_DIR="$DATA_DIR/synced_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Parse arguments
DO_MERGE=false
TO_MAC_STUDIO=false
TO_LAMBDA=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --merge) DO_MERGE=true; shift ;;
        --to-mac-studio) TO_MAC_STUDIO=true; shift ;;
        --to-lambda) TO_LAMBDA=true; shift ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h)
            head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "  RingRift Selfplay Data Sync"
echo "  Timestamp: $TIMESTAMP"
echo "============================================="

# Check for config file
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_warning "Config file not found: $CONFIG_FILE"
    log_info "Using fallback hardcoded hosts (for backward compatibility)"
    log_info "To use config-based sync, copy config/remote_hosts.example.yaml to config/remote_hosts.yaml"
    USE_CONFIG=false
else
    log_info "Using config: $CONFIG_FILE"
    USE_CONFIG=true
fi

mkdir -p "$SYNC_DIR"
mkdir -p "$DATA_DIR"

# Track success/failure
SYNC_SUCCESS=0
SYNC_FAILED=0

# Function to sync from a standard SSH host (port 22)
sync_from_host() {
    local host_alias="$1"
    local remote_path="$2"
    local local_subdir="$3"
    local ssh_key="${4:-}"

    log_info "Syncing from $host_alias..."
    local dest="$SYNC_DIR/$local_subdir"
    mkdir -p "$dest"

    local ssh_opts=""
    if [[ -n "$ssh_key" ]]; then
        ssh_opts="-e \"ssh -i $ssh_key\""
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] rsync $ssh_opts -avz $host_alias:$remote_path/*.db $dest/"
        return 0
    fi

    if rsync -avz --progress "$host_alias:$remote_path/"*.db "$dest/" 2>/dev/null; then
        local count=$(find "$dest" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        log_success "$host_alias: synced $count database(s)"
        ((SYNC_SUCCESS++)) || true
    else
        log_warning "$host_alias: no databases found or sync failed"
        ((SYNC_FAILED++)) || true
    fi
}

# Function to sync from Vast instances (custom SSH port, RAM storage)
sync_from_vast() {
    local host="$1"
    local port="$2"
    local name="$3"
    local remote_path="${4:-/dev/shm/games}"

    log_info "Syncing from $name (Vast, port $port, RAM storage)..."
    local dest="$SYNC_DIR/$name"
    mkdir -p "$dest"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] rsync -e \"ssh -p $port\" -avz root@$host:$remote_path/*.db $dest/"
        return 0
    fi

    # Vast uses /dev/shm for RAM storage
    if rsync -avz --progress -e "ssh -p $port -o ConnectTimeout=15" "root@$host:$remote_path/"*.db "$dest/" 2>/dev/null; then
        local count=$(find "$dest" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        log_success "$name: synced $count database(s) from RAM storage"
        ((SYNC_SUCCESS++)) || true
    else
        log_warning "$name: no databases found or sync failed (instance may be terminated)"
        ((SYNC_FAILED++)) || true
    fi
}

# Parse YAML config if available (simple key: value parsing)
parse_yaml_hosts() {
    local config_file="$1"
    local section="$2"

    # Extract hosts from the given section using basic sed/awk
    # This is a simple parser - for complex configs, use Python
    awk -v section="$section" '
        /^[a-z_]+:$/ { current_section = $1; gsub(/:$/, "", current_section) }
        current_section == section && /^  [a-z_]+:$/ {
            host = $1; gsub(/:$/, "", host); print host
        }
    ' "$config_file"
}

# ============================================
# SYNC LOGIC
# ============================================

if [[ "$USE_CONFIG" == "true" ]]; then
    log_section "Syncing from configured hosts"

    # For now, we'll use a Python helper for YAML parsing
    # Fallback to inline parsing for simple cases

    # Check if we have Python available for proper YAML parsing
    if command -v python3 &>/dev/null; then
        # Use Python to parse YAML and generate sync commands
        python3 - "$CONFIG_FILE" "$SYNC_DIR" "$DRY_RUN" << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path

config_file = sys.argv[1]
sync_dir = sys.argv[2]
dry_run = sys.argv[3] == "true"

try:
    import yaml
except ImportError:
    print("WARNING: PyYAML not installed, using fallback sync", file=sys.stderr)
    sys.exit(1)

with open(config_file) as f:
    config = yaml.safe_load(f)

# Output shell commands for each host
commands = []

# Standard hosts
for name, host_cfg in config.get("standard_hosts", {}).items():
    ssh_host = host_cfg.get("ssh_host", "")
    remote_path = host_cfg.get("remote_path", "~/ringrift/ai-service/data/games")
    ssh_key = host_cfg.get("ssh_key", "")

    if ssh_key:
        cmd = f'sync_from_host_with_key "{ssh_host}" "{remote_path}" "{name}" "{ssh_key}"'
    else:
        cmd = f'sync_from_host "{ssh_host}" "{remote_path}" "{name}"'
    print(cmd)

# Vast hosts
for name, host_cfg in config.get("vast_hosts", {}).items():
    host = host_cfg.get("host", "")
    port = host_cfg.get("port", 22)
    remote_path = host_cfg.get("remote_path", "/dev/shm/games")

    cmd = f'sync_from_vast "{host}" "{port}" "{name}" "{remote_path}"'
    print(cmd)
PYTHON_SCRIPT

        # Execute the generated commands
        while IFS= read -r cmd; do
            eval "$cmd"
        done < <(python3 - "$CONFIG_FILE" "$SYNC_DIR" "$DRY_RUN" << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path

config_file = sys.argv[1]
sync_dir = sys.argv[2]
dry_run = sys.argv[3] == "true"

try:
    import yaml
except ImportError:
    sys.exit(1)

with open(config_file) as f:
    config = yaml.safe_load(f)

# Standard hosts
for name, host_cfg in config.get("standard_hosts", {}).items():
    ssh_host = host_cfg.get("ssh_host", "")
    remote_path = host_cfg.get("remote_path", "~/ringrift/ai-service/data/games")
    ssh_key = host_cfg.get("ssh_key", "")

    if ssh_key:
        print(f'sync_from_host "{ssh_host}" "{remote_path}" "{name}" "{ssh_key}"')
    else:
        print(f'sync_from_host "{ssh_host}" "{remote_path}" "{name}"')

# Vast hosts
for name, host_cfg in config.get("vast_hosts", {}).items():
    host = host_cfg.get("host", "")
    port = host_cfg.get("port", 22)
    remote_path = host_cfg.get("remote_path", "/dev/shm/games")

    print(f'sync_from_vast "{host}" "{port}" "{name}" "{remote_path}"')
PYTHON_SCRIPT
        ) 2>/dev/null || {
            log_warning "YAML parsing failed, falling back to hardcoded hosts"
            USE_CONFIG=false
        }
    else
        log_warning "Python not available, falling back to hardcoded hosts"
        USE_CONFIG=false
    fi
fi

# Fallback: hardcoded hosts (backward compatibility)
if [[ "$USE_CONFIG" == "false" ]]; then
    log_section "Syncing from hardcoded hosts (fallback mode)"
    log_info "Create config/remote_hosts.yaml for config-based sync"

    # Standard SSH hosts
    sync_from_host "ringrift-staging" "~/ringrift/ai-service/data/games" "aws_staging"
    sync_from_host "ringrift-selfplay-extra" "~/ringrift/ai-service/data/games" "aws_extra"
    sync_from_host "m1-pro" "~/Development/RingRift/ai-service/data/games" "m1_pro"

    # Mac Studio (skip if we're on it)
    if [[ "$(hostname)" != *"Mac-Studio"* ]]; then
        sync_from_host "mac-studio" "~/Development/RingRift/ai-service/data/games" "mac_studio"
    fi

    # Lambda
    sync_from_host "ubuntu@150.136.65.197" "~/ringrift/ai-service/data/games" "lambda"

    # Vast instances (ephemeral, data in RAM)
    # NOTE: Update these IPs/ports when provisioning new Vast instances
    sync_from_vast "211.72.13.202" "45875" "vast_4x5090"
    sync_from_vast "178.43.61.252" "18080" "vast_2x5090"
    sync_from_vast "79.116.93.241" "47070" "vast_1x3090"
fi

# ============================================
# SUMMARY
# ============================================

log_section "Sync Summary"
echo "  Successful: $SYNC_SUCCESS"
echo "  Failed/Empty: $SYNC_FAILED"
echo "  Output directory: $SYNC_DIR"

# List all synced databases
TOTAL_DBS=$(find "$SYNC_DIR" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
log_info "Total databases synced: $TOTAL_DBS"

if [[ "$TOTAL_DBS" -eq 0 ]]; then
    log_warning "No databases were synced. Check remote hosts and paths."
    exit 0
fi

# ============================================
# MERGE (optional)
# ============================================

if [[ "$DO_MERGE" == "true" ]]; then
    log_section "Merging Databases"

    MERGED_DB="$DATA_DIR/merged_${TIMESTAMP}.db"

    # Build the merge command with all found DBs
    DB_ARGS=""
    while IFS= read -r db; do
        DB_ARGS="$DB_ARGS --db $db"
    done < <(find "$SYNC_DIR" -name "*.db")

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] python $SCRIPT_DIR/merge_game_dbs.py --output $MERGED_DB $DB_ARGS --on-conflict skip"
    elif python "$SCRIPT_DIR/merge_game_dbs.py" \
        --output "$MERGED_DB" \
        $DB_ARGS \
        --on-conflict skip; then
        log_success "Merged database: $MERGED_DB"

        # Get game count
        if command -v sqlite3 &>/dev/null; then
            GAME_COUNT=$(sqlite3 "$MERGED_DB" "SELECT COUNT(*) FROM games;" 2>/dev/null || echo "?")
            log_info "Total games in merged DB: $GAME_COUNT"
        fi

        # Create a symlink to latest
        LATEST_LINK="$DATA_DIR/merged_latest.db"
        rm -f "$LATEST_LINK"
        ln -s "$MERGED_DB" "$LATEST_LINK"
        log_info "Created symlink: merged_latest.db -> $(basename "$MERGED_DB")"

        # Optionally sync to Mac Studio
        if [[ "$TO_MAC_STUDIO" == "true" ]]; then
            log_section "Syncing to Mac Studio"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY-RUN] rsync -avz $MERGED_DB mac-studio:~/Development/RingRift/ai-service/data/games/"
            elif rsync -avz --progress "$MERGED_DB" \
                mac-studio:~/Development/RingRift/ai-service/data/games/; then
                log_success "Synced to Mac Studio"
            else
                log_error "Failed to sync to Mac Studio"
            fi
        fi

        # Optionally sync to Lambda GPU (ubuntu@209.20.157.81)
        if [[ "$TO_LAMBDA" == "true" ]]; then
            log_section "Syncing to Lambda GPU"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "[DRY-RUN] rsync -avz $MERGED_DB ubuntu@209.20.157.81:~/ringrift/ai-service/data/games/"
            elif rsync -avz --progress "$MERGED_DB" \
                ubuntu@209.20.157.81:~/ringrift/ai-service/data/games/; then
                log_success "Synced to Lambda GPU"
            else
                log_error "Failed to sync to Lambda GPU"
            fi
        fi
    else
        log_error "Failed to merge databases"
        exit 1
    fi
fi

log_section "Done!"
