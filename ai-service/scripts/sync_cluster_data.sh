#!/bin/bash
# sync_cluster_data.sh - Sync selfplay game data from all cluster hosts
#
# This script syncs selfplay game databases (.db) and JSONL files from all
# cluster hosts defined in config/distributed_hosts.yaml to a target directory.
#
# Common use cases:
#   1. Sync to external drive (OWS/WD): --target /Volumes/RingRift-Data/selfplay_repository/raw
#   2. Sync to local data directory: --target ./data/games/synced
#   3. Sync to training machine: --target /path/to/training/data
#
# Usage:
#   ./scripts/sync_cluster_data.sh [OPTIONS]
#
# Options:
#   --target PATH   Target directory for synced data (default: ./data/games/cluster_sync)
#   --config FILE   Hosts config file (default: config/distributed_hosts.yaml)
#   --dry-run       Show what would be synced without actually syncing
#   --no-tailscale  Skip Tailscale fallback attempts
#   --no-http-check Skip HTTP health checks before SSH
#   --include-daemon  Include daemon selfplay data (GPU-generated games from data/selfplay/daemon_*)
#   -h, --help      Show this help message
#
# Resilient Connection Methods:
#   The script attempts multiple connection strategies for each host:
#   1. HTTP health check via worker_url (if configured) to verify host is alive
#   2. Primary SSH connection via ssh_host
#   3. Tailscale SSH fallback via tailscale_ip (if primary fails)
#
# Configuration:
#   Hosts are read from config/distributed_hosts.yaml. Each host entry can have:
#     - ssh_host: Primary IP or hostname
#     - tailscale_ip: (optional) Tailscale IP for fallback
#     - ssh_user: SSH username
#     - ssh_port: (optional) SSH port, default 22
#     - ssh_key: (optional) path to SSH key
#     - ringrift_path: path to ringrift/ai-service on the remote host
#     - worker_url: (optional) HTTP worker URL for health checks
#
# The script automatically handles:
#   - Standard SSH hosts (port 22, persistent storage)
#   - Vast.ai instances (custom ports, RAM storage at /dev/shm)
#   - Tailscale network fallback when primary connection fails
#   - Failed/unreachable hosts (continues with remaining hosts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$AI_SERVICE_DIR/config/distributed_hosts.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default target (local cluster sync directory)
TARGET_DIR="$AI_SERVICE_DIR/data/games/cluster_sync"

# Connection settings
USE_TAILSCALE=true
USE_HTTP_CHECK=true
SSH_CONNECT_TIMEOUT=15
HTTP_CHECK_TIMEOUT=5

# Data source settings
INCLUDE_DAEMON_DATA=false

# Colors
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

# Disk capacity limits (consistent with orchestrator MAX_DISK_USAGE_PERCENT)
MAX_DISK_USAGE_PERCENT=${RINGRIFT_MAX_DISK_PERCENT:-70}

check_disk_capacity() {
    # Check if target directory disk has capacity
    # Returns 0 if OK, 1 if disk is too full
    local target_path="${1:-$TARGET_DIR}"

    # Get disk usage percentage for the target path
    local usage
    if [[ -d "$target_path" ]]; then
        usage=$(df "$target_path" | tail -1 | awk '{print $5}' | sed 's/%//')
    else
        # Target doesn't exist yet, check parent
        local parent_dir=$(dirname "$target_path")
        if [[ -d "$parent_dir" ]]; then
            usage=$(df "$parent_dir" | tail -1 | awk '{print $5}' | sed 's/%//')
        else
            log_warning "Cannot determine disk usage for $target_path"
            return 0  # Allow sync to continue
        fi
    fi

    if [[ -z "$usage" ]]; then
        log_warning "Could not parse disk usage"
        return 0
    fi

    if (( usage >= MAX_DISK_USAGE_PERCENT )); then
        log_error "Disk usage at ${usage}% exceeds limit (${MAX_DISK_USAGE_PERCENT}%)"
        return 1
    fi

    return 0
}

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --target) TARGET_DIR="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --no-tailscale) USE_TAILSCALE=false; shift ;;
        --no-http-check) USE_HTTP_CHECK=false; shift ;;
        --include-daemon) INCLUDE_DAEMON_DATA=true; shift ;;
        --help|-h)
            head -45 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "  RingRift Cluster Data Sync"
echo "  Timestamp: $TIMESTAMP"
echo "  Target: $TARGET_DIR"
echo "  Config: $CONFIG_FILE"
[[ "$DRY_RUN" == "true" ]] && echo "  Mode: DRY RUN"
[[ "$USE_TAILSCALE" == "true" ]] && echo "  Tailscale fallback: enabled"
[[ "$USE_HTTP_CHECK" == "true" ]] && echo "  HTTP health check: enabled"
[[ "$INCLUDE_DAEMON_DATA" == "true" ]] && echo "  Include daemon selfplay: enabled"
echo "============================================="

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    log_info "Copy config/distributed_hosts.template.yaml to config/distributed_hosts.yaml"
    exit 1
fi

# Verify target parent is accessible (create if needed)
PARENT_DIR="$(dirname "$TARGET_DIR")"
if [[ ! -d "$PARENT_DIR" ]]; then
    log_info "Creating parent directory: $PARENT_DIR"
    mkdir -p "$PARENT_DIR" || {
        log_error "Cannot create target parent directory: $PARENT_DIR"
        log_info "If syncing to external drive, ensure it's mounted"
        exit 1
    }
fi

mkdir -p "$TARGET_DIR"

# Track stats
SYNC_SUCCESS=0
SYNC_FAILED=0
SYNC_VIA_TAILSCALE=0
TOTAL_FILES=0

# Check if host is reachable via HTTP health endpoint
check_http_health() {
    local worker_url="$1"
    local timeout="${2:-$HTTP_CHECK_TIMEOUT}"

    if [[ -z "$worker_url" ]]; then
        return 1
    fi

    # Ensure URL has protocol
    if [[ ! "$worker_url" =~ ^https?:// ]]; then
        worker_url="http://$worker_url"
    fi

    local health_url="${worker_url%/}/health"

    if curl -s --connect-timeout "$timeout" --max-time "$((timeout + 2))" "$health_url" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Test SSH connectivity
test_ssh_connection() {
    local ssh_host="$1"
    local ssh_user="$2"
    local ssh_key="${3:-}"
    local ssh_port="${4:-22}"
    local timeout="${5:-$SSH_CONNECT_TIMEOUT}"

    local ssh_opts="-o ConnectTimeout=$timeout -o StrictHostKeyChecking=no -o BatchMode=yes"
    [[ -n "$ssh_key" ]] && ssh_opts="$ssh_opts -i $ssh_key"
    [[ "$ssh_port" != "22" ]] && ssh_opts="$ssh_opts -p $ssh_port"

    if ssh -n $ssh_opts "$ssh_user@$ssh_host" "echo ok" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Sync from host with rsync
do_rsync_sync() {
    local name="$1"
    local ssh_host="$2"
    local ssh_user="$3"
    local remote_path="$4"
    local ssh_key="${5:-}"
    local ssh_port="${6:-22}"
    local dest="$7"

    # Check disk capacity before syncing
    if ! check_disk_capacity "$dest"; then
        log_error "Skipping sync to $name - disk capacity limit reached"
        return 1
    fi

    local ssh_opts="-o ConnectTimeout=$SSH_CONNECT_TIMEOUT -o StrictHostKeyChecking=no -o BatchMode=yes"
    [[ -n "$ssh_key" ]] && ssh_opts="$ssh_opts -i $ssh_key"
    [[ "$ssh_port" != "22" ]] && ssh_opts="$ssh_opts -p $ssh_port"

    local synced_any=false

    # Try standard path first
    if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_path/data/games/"*.db "$dest/" 2>/dev/null; then
        synced_any=true
    fi
    if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_path/data/games/"*.jsonl "$dest/" 2>/dev/null; then
        synced_any=true
    fi

    # For Vast.ai instances, also try RAM storage at /dev/shm
    if [[ "$ssh_user" == "root" ]]; then
        if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:/dev/shm/games/"*.db "$dest/" 2>/dev/null; then
            synced_any=true
        fi
        if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:/dev/shm/games/"*.jsonl "$dest/" 2>/dev/null; then
            synced_any=true
        fi
    fi

    # Sync daemon selfplay data (GPU-generated games)
    if [[ "$INCLUDE_DAEMON_DATA" == "true" ]]; then
        local daemon_dest="$dest/daemon"
        mkdir -p "$daemon_dest"

        # Sync all daemon_* directories
        for daemon_dir in square8_2p square8_3p square8_4p square19_2p square19_3p square19_4p hexagonal_2p hexagonal_3p hexagonal_4p; do
            local remote_daemon="$remote_path/data/selfplay/daemon_${daemon_dir}"
            if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_daemon/*.jsonl" "$daemon_dest/${daemon_dir}/" 2>/dev/null; then
                synced_any=true
            fi
        done

        # Also sync asymmetric variants
        for daemon_dir in square8_2p_asymmetric square8_3p_asymmetric square8_4p_asymmetric square19_2p_asymmetric square19_3p_asymmetric square19_4p_asymmetric hexagonal_2p_asymmetric hexagonal_3p_asymmetric hexagonal_4p_asymmetric; do
            local remote_daemon="$remote_path/data/selfplay/daemon_${daemon_dir}"
            if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_daemon/*.jsonl" "$daemon_dest/${daemon_dir}/" 2>/dev/null; then
                synced_any=true
            fi
        done
    fi

    if [[ "$synced_any" == "true" ]]; then
        return 0
    fi
    return 1
}

# Main sync function with resilient connection handling
sync_from_host() {
    local name="$1"
    local ssh_host="$2"
    local ssh_user="$3"
    local remote_path="$4"
    local ssh_key="${5:-}"
    local ssh_port="${6:-22}"
    local tailscale_ip="${7:-}"
    local worker_url="${8:-}"

    log_info "Syncing from $name..."
    local dest="$TARGET_DIR/$name"
    mkdir -p "$dest"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] Would sync from $ssh_user@$ssh_host:$remote_path/data/games/ to $dest/"
        [[ -n "$tailscale_ip" ]] && echo "  [DRY-RUN] Tailscale fallback available: $tailscale_ip"
        [[ -n "$worker_url" ]] && echo "  [DRY-RUN] HTTP health check: $worker_url"
        return 0
    fi

    local connection_method=""
    local effective_host="$ssh_host"
    local effective_port="$ssh_port"

    # Step 1: Optional HTTP health check to quickly verify host is alive
    if [[ "$USE_HTTP_CHECK" == "true" && -n "$worker_url" ]]; then
        if check_http_health "$worker_url"; then
            log_info "  HTTP health check passed for $name"
        else
            log_warning "  HTTP health check failed for $name (will try SSH anyway)"
        fi
    fi

    # Step 2: Try primary SSH connection
    if test_ssh_connection "$ssh_host" "$ssh_user" "$ssh_key" "$ssh_port"; then
        connection_method="primary"
        effective_host="$ssh_host"
        effective_port="$ssh_port"
    # Step 3: Try Tailscale fallback if available
    elif [[ "$USE_TAILSCALE" == "true" && -n "$tailscale_ip" ]]; then
        log_info "  Primary SSH failed, trying Tailscale ($tailscale_ip)..."
        if test_ssh_connection "$tailscale_ip" "$ssh_user" "$ssh_key" "22"; then
            connection_method="tailscale"
            effective_host="$tailscale_ip"
            effective_port="22"
            ((SYNC_VIA_TAILSCALE++)) || true
        fi
    fi

    if [[ -z "$connection_method" ]]; then
        log_warning "$name: unreachable (tried primary${tailscale_ip:+ + tailscale})"
        ((SYNC_FAILED++)) || true
        return 1
    fi

    # Step 4: Perform the actual sync
    if do_rsync_sync "$name" "$effective_host" "$ssh_user" "$remote_path" "$ssh_key" "$effective_port" "$dest"; then
        local db_count=$(find "$dest" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        local jsonl_count=$(find "$dest" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
        local method_note=""
        [[ "$connection_method" == "tailscale" ]] && method_note=" (via Tailscale)"
        log_success "$name: $db_count DB(s), $jsonl_count JSONL$method_note"
        ((SYNC_SUCCESS++)) || true
        ((TOTAL_FILES += db_count + jsonl_count)) || true
    else
        log_warning "$name: connected but no data found"
        ((SYNC_FAILED++)) || true
    fi
}

# ============================================
# PARSE CONFIG AND SYNC
# ============================================

log_section "Reading host configuration"

# Use Python to parse YAML and extract host details
if ! command -v python3 &>/dev/null; then
    log_error "Python3 required for YAML parsing"
    exit 1
fi

# Generate sync commands from config (now includes tailscale_ip and worker_url)
SYNC_COMMANDS=$(python3 - "$CONFIG_FILE" << 'PYTHON_SCRIPT'
import sys
import os

config_file = sys.argv[1]

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

with open(config_file) as f:
    config = yaml.safe_load(f)

hosts = config.get("hosts", {})

for name, host_cfg in hosts.items():
    # Skip disabled/commented hosts
    status = host_cfg.get("status", "ready")
    if status not in ("ready", "setup"):
        continue

    ssh_host = host_cfg.get("ssh_host", "")
    ssh_user = host_cfg.get("ssh_user", "")
    ssh_port = host_cfg.get("ssh_port", 22)
    ssh_key = host_cfg.get("ssh_key", "")
    ringrift_path = host_cfg.get("ringrift_path", "~/ringrift/ai-service")

    # Resilient connection options
    tailscale_ip = host_cfg.get("tailscale_ip", "")
    worker_url = host_cfg.get("worker_url", "")

    # If no explicit worker_url, construct from ssh_host and default port
    if not worker_url and ssh_host:
        worker_port = host_cfg.get("worker_port", 8765)
        # Only construct URL for non-Vast.ai hosts (they use non-standard ports)
        if ssh_port == 22:
            worker_url = f"http://{ssh_host}:{worker_port}"

    # Expand ~ in paths
    ssh_key = ssh_key.replace("~", os.environ.get("HOME", "~"))

    if ssh_host and ssh_user:
        # Escape any special characters in paths
        ringrift_path = ringrift_path.replace('"', '\\"')
        print(f'sync_from_host "{name}" "{ssh_host}" "{ssh_user}" "{ringrift_path}" "{ssh_key}" "{ssh_port}" "{tailscale_ip}" "{worker_url}"')
PYTHON_SCRIPT
) || {
    log_error "Failed to parse config file"
    exit 1
}

HOST_COUNT=$(echo "$SYNC_COMMANDS" | grep -c 'sync_from_host' || echo "0")
log_info "Found $HOST_COUNT hosts in configuration"

log_section "Syncing from all hosts"

# Execute sync commands
# Note: Use fd 3 to avoid SSH/rsync consuming stdin and breaking the loop
while IFS= read -r cmd <&3; do
    [[ -n "$cmd" ]] && eval "$cmd" || true
done 3<<< "$SYNC_COMMANDS"

# ============================================
# SUMMARY
# ============================================

log_section "Sync Summary"
echo "  Successful: $SYNC_SUCCESS hosts"
echo "  Failed/Empty: $SYNC_FAILED hosts"
[[ "$SYNC_VIA_TAILSCALE" -gt 0 ]] && echo "  Connected via Tailscale: $SYNC_VIA_TAILSCALE hosts"
echo "  Total files synced: $TOTAL_FILES"
echo ""

# Disk usage summary
if [[ -d "$TARGET_DIR" ]]; then
    log_info "Target directory status:"
    du -sh "$TARGET_DIR" 2>/dev/null || true

    # Count total games if sqlite3 is available
    if command -v sqlite3 &>/dev/null; then
        log_info "Counting games in synced databases..."
        TOTAL_GAMES=0
        while IFS= read -r db; do
            count=$(sqlite3 "$db" "SELECT COUNT(*) FROM games;" 2>/dev/null || echo "0")
            ((TOTAL_GAMES += count)) || true
        done < <(find "$TARGET_DIR" -name "*.db" 2>/dev/null)
        log_success "Total games across all synced DBs: $TOTAL_GAMES"
    fi
fi

log_section "Done! $(date)"
