#!/bin/bash
# sync_selfplay_data.sh - Sync selfplay data from all remote machines to local
#
# Usage:
#   ./sync_selfplay_data.sh [--merge] [--to-mac-studio]
#
# Options:
#   --merge         After syncing, merge all DBs into a single merged.db
#   --to-mac-studio Also sync merged DB to Mac Studio for training

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$AI_SERVICE_DIR/data/games"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SYNC_DIR="$DATA_DIR/synced_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
DO_MERGE=false
TO_MAC_STUDIO=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --merge) DO_MERGE=true; shift ;;
        --to-mac-studio) TO_MAC_STUDIO=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--merge] [--to-mac-studio]"
            echo ""
            echo "Sync selfplay databases from all remote machines."
            echo ""
            echo "Options:"
            echo "  --merge          Merge all synced DBs into a single file"
            echo "  --to-mac-studio  Copy merged DB to Mac Studio for training"
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
echo ""

mkdir -p "$SYNC_DIR"
mkdir -p "$DATA_DIR"

# Track success/failure
SYNC_SUCCESS=0
SYNC_FAILED=0

# Function to sync from a remote host
sync_from_host() {
    local host_alias="$1"
    local remote_path="$2"
    local local_subdir="$3"

    log_info "Syncing from $host_alias..."
    local dest="$SYNC_DIR/$local_subdir"
    mkdir -p "$dest"

    if rsync -avz --progress "$host_alias:$remote_path/*.db" "$dest/" 2>/dev/null; then
        local count=$(find "$dest" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        log_success "$host_alias: synced $count database(s)"
        ((SYNC_SUCCESS++))
    else
        log_warning "$host_alias: no databases found or sync failed"
        ((SYNC_FAILED++))
    fi
}

# Sync from AWS Staging
sync_from_host "ringrift-staging" "~/ringrift/ai-service/data/games" "aws_staging"

# Sync from AWS Extra (selfplay-extra instance)
sync_from_host "ringrift-selfplay-extra" "~/ringrift/ai-service/data/games" "aws_extra"

# Sync from M1 Pro
sync_from_host "m1-pro" "~/Development/RingRift/ai-service/data/games" "m1_pro"

# Sync from Mac Studio (if we're not ON the Mac Studio)
if [[ "$(hostname)" != *"Mac-Studio"* ]]; then
    sync_from_host "mac-studio" "~/Development/RingRift/ai-service/data/games" "mac_studio"
fi

echo ""
echo "============================================="
echo "  Sync Summary"
echo "============================================="
echo "  Successful: $SYNC_SUCCESS"
echo "  Failed/Empty: $SYNC_FAILED"
echo "  Output directory: $SYNC_DIR"
echo ""

# List all synced databases
TOTAL_DBS=$(find "$SYNC_DIR" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
log_info "Total databases synced: $TOTAL_DBS"

if [[ "$TOTAL_DBS" -eq 0 ]]; then
    log_warning "No databases were synced. Check remote hosts and paths."
    exit 0
fi

# Optionally merge all databases
if [[ "$DO_MERGE" == "true" ]]; then
    echo ""
    log_info "Merging all databases..."

    MERGED_DB="$DATA_DIR/merged_${TIMESTAMP}.db"

    # Build the merge command with all found DBs
    DB_ARGS=""
    while IFS= read -r db; do
        DB_ARGS="$DB_ARGS --db $db"
    done < <(find "$SYNC_DIR" -name "*.db")

    if python "$SCRIPT_DIR/merge_game_dbs.py" \
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
            echo ""
            log_info "Syncing merged DB to Mac Studio..."
            if rsync -avz --progress "$MERGED_DB" \
                mac-studio:~/Development/RingRift/ai-service/data/games/; then
                log_success "Synced to Mac Studio"
            else
                log_error "Failed to sync to Mac Studio"
            fi
        fi
    else
        log_error "Failed to merge databases"
        exit 1
    fi
fi

echo ""
echo "============================================="
echo "  Done!"
echo "============================================="
