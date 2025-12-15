#!/bin/bash
# Setup RingRift node resilience on macOS via launchd (LaunchAgents).
#
# Usage:
#   ./setup_node_resilience_macos.sh <node-id> <coordinator-url>
#
# Env overrides:
#   RINGRIFT_ROOT   (default: ~/Development/RingRift)
#   P2P_PORT        (default: 8770)
#   PYTHON          (default: ai-service/venv/bin/python if present, else python3)
#
# This installs:
#   ~/Library/LaunchAgents/com.ringrift.p2p.plist
#   ~/Library/LaunchAgents/com.ringrift.resilience.plist
#
set -euo pipefail

NODE_ID="${1:?Usage: $0 <node-id> <coordinator-url>}"
COORDINATOR_URL="${2:?Usage: $0 <node-id> <coordinator-url>}"

RINGRIFT_ROOT="${RINGRIFT_ROOT:-$HOME/Development/RingRift}"
P2P_PORT="${P2P_PORT:-8770}"
P2P_VOTERS="${RINGRIFT_P2P_VOTERS:-}"
P2P_VOTERS="${P2P_VOTERS//[[:space:]]/}"
ENABLE_IMPROVEMENT_DAEMON_RAW="${RINGRIFT_ENABLE_IMPROVEMENT_DAEMON:-}"

is_truthy() {
  local v
  v="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$v" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_int_or_empty() {
  local val="${1:-}"
  if [ -z "$val" ]; then
    echo ""
    return 0
  fi
  if echo "$val" | grep -Eq '^[0-9]+$'; then
    echo "$val"
    return 0
  fi
  echo ""
  return 0
}

PEER_RETIRE_AFTER_SECONDS="$(sanitize_int_or_empty "${RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS:-}")"
RETRY_RETIRED_NODE_INTERVAL="$(sanitize_int_or_empty "${RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL:-}")"
DISK_WARNING_THRESHOLD_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_DISK_WARNING_THRESHOLD:-}")"
DISK_CLEANUP_THRESHOLD_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD:-}")"
DISK_CRITICAL_THRESHOLD_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD:-}")"
MEMORY_WARNING_THRESHOLD_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD:-}")"
MEMORY_CRITICAL_THRESHOLD_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD:-}")"
LOAD_MAX_FOR_NEW_JOBS_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS:-}")"
ADVERTISE_HOST="${RINGRIFT_ADVERTISE_HOST:-}"
ADVERTISE_PORT="${RINGRIFT_ADVERTISE_PORT:-$P2P_PORT}"
LAUNCHD_PATH="${RINGRIFT_LAUNCHD_PATH:-/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
P2P_AUTO_UPDATE_RAW="${RINGRIFT_P2P_AUTO_UPDATE:-1}"
P2P_AUTO_UPDATE="$(sanitize_int_or_empty "${P2P_AUTO_UPDATE_RAW}")"
if [ -z "$P2P_AUTO_UPDATE" ]; then
  P2P_AUTO_UPDATE="1"
fi
P2P_GIT_UPDATE_CHECK_INTERVAL_OVERRIDE="$(sanitize_int_or_empty "${RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL:-}")"

AI_SERVICE_DIR="${RINGRIFT_ROOT}/ai-service"
if [ ! -d "$AI_SERVICE_DIR" ]; then
  echo "Error: ai-service dir not found at: $AI_SERVICE_DIR" >&2
  exit 1
fi

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "${AI_SERVICE_DIR}/venv/bin/python" ]; then
    PYTHON="${AI_SERVICE_DIR}/venv/bin/python"
  else
    PYTHON="$(command -v python3)"
  fi
fi

LOG_DIR="${HOME}/Library/Logs/RingRift"
PLIST_DIR="${HOME}/Library/LaunchAgents"
mkdir -p "$LOG_DIR" "$PLIST_DIR"

# Store the cluster auth token in a local user-only file and reference it via
# RINGRIFT_CLUSTER_AUTH_TOKEN_FILE (avoid embedding secrets in launchd plists).
TOKEN_DIR="${HOME}/Library/Application Support/RingRift"
TOKEN_FILE="${TOKEN_DIR}/cluster_auth_token"
if [ -n "${RINGRIFT_CLUSTER_AUTH_TOKEN:-}" ]; then
  umask 077
  mkdir -p "$TOKEN_DIR"
  printf "%s" "$RINGRIFT_CLUSTER_AUTH_TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi

P2P_PLIST="${PLIST_DIR}/com.ringrift.p2p.plist"
RES_PLIST="${PLIST_DIR}/com.ringrift.resilience.plist"
IMP_PLIST="${PLIST_DIR}/com.ringrift.improvement.plist"

cat > "$P2P_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ringrift.p2p</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${AI_SERVICE_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${LAUNCHD_PATH}</string>
    <key>PYTHONPATH</key>
    <string>${AI_SERVICE_DIR}</string>
    <key>RINGRIFT_ADVERTISE_HOST</key>
    <string>${ADVERTISE_HOST}</string>
    <key>RINGRIFT_ADVERTISE_PORT</key>
    <string>${ADVERTISE_PORT}</string>
    <key>RINGRIFT_P2P_VOTERS</key>
    <string>${P2P_VOTERS}</string>
    <key>RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS</key>
    <string>${PEER_RETIRE_AFTER_SECONDS}</string>
    <key>RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL</key>
    <string>${RETRY_RETIRED_NODE_INTERVAL}</string>
    <key>RINGRIFT_P2P_DISK_WARNING_THRESHOLD</key>
    <string>${DISK_WARNING_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD</key>
    <string>${DISK_CLEANUP_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD</key>
    <string>${DISK_CRITICAL_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD</key>
    <string>${MEMORY_WARNING_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD</key>
    <string>${MEMORY_CRITICAL_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS</key>
    <string>${LOAD_MAX_FOR_NEW_JOBS_OVERRIDE}</string>
    <key>RINGRIFT_P2P_AUTO_UPDATE</key>
    <string>${P2P_AUTO_UPDATE}</string>
    <key>RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL</key>
    <string>${P2P_GIT_UPDATE_CHECK_INTERVAL_OVERRIDE}</string>
    <key>RINGRIFT_CLUSTER_AUTH_TOKEN_FILE</key>
    <string>${TOKEN_FILE}</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>${AI_SERVICE_DIR}/scripts/p2p_orchestrator.py</string>
    <string>--node-id</string>
    <string>${NODE_ID}</string>
    <string>--port</string>
    <string>${P2P_PORT}</string>
    <string>--peers</string>
    <string>${COORDINATOR_URL}</string>
    <string>--ringrift-path</string>
    <string>${RINGRIFT_ROOT}</string>
  </array>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/p2p.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/p2p.log</string>
</dict>
</plist>
EOF

cat > "$RES_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ringrift.resilience</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${AI_SERVICE_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${LAUNCHD_PATH}</string>
    <key>PYTHONPATH</key>
    <string>${AI_SERVICE_DIR}</string>
    <key>RINGRIFT_ADVERTISE_HOST</key>
    <string>${ADVERTISE_HOST}</string>
    <key>RINGRIFT_ADVERTISE_PORT</key>
    <string>${ADVERTISE_PORT}</string>
    <key>RINGRIFT_P2P_VOTERS</key>
    <string>${P2P_VOTERS}</string>
    <key>RINGRIFT_P2P_PEER_RETIRE_AFTER_SECONDS</key>
    <string>${PEER_RETIRE_AFTER_SECONDS}</string>
    <key>RINGRIFT_P2P_RETRY_RETIRED_NODE_INTERVAL</key>
    <string>${RETRY_RETIRED_NODE_INTERVAL}</string>
    <key>RINGRIFT_P2P_DISK_WARNING_THRESHOLD</key>
    <string>${DISK_WARNING_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_DISK_CLEANUP_THRESHOLD</key>
    <string>${DISK_CLEANUP_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_DISK_CRITICAL_THRESHOLD</key>
    <string>${DISK_CRITICAL_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_MEMORY_WARNING_THRESHOLD</key>
    <string>${MEMORY_WARNING_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_MEMORY_CRITICAL_THRESHOLD</key>
    <string>${MEMORY_CRITICAL_THRESHOLD_OVERRIDE}</string>
    <key>RINGRIFT_P2P_LOAD_MAX_FOR_NEW_JOBS</key>
    <string>${LOAD_MAX_FOR_NEW_JOBS_OVERRIDE}</string>
    <key>RINGRIFT_P2P_AUTO_UPDATE</key>
    <string>${P2P_AUTO_UPDATE}</string>
    <key>RINGRIFT_P2P_GIT_UPDATE_CHECK_INTERVAL</key>
    <string>${P2P_GIT_UPDATE_CHECK_INTERVAL_OVERRIDE}</string>
    <key>RINGRIFT_CLUSTER_AUTH_TOKEN_FILE</key>
    <string>${TOKEN_FILE}</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>${AI_SERVICE_DIR}/scripts/node_resilience.py</string>
    <string>--node-id</string>
    <string>${NODE_ID}</string>
    <string>--coordinator</string>
    <string>${COORDINATOR_URL}</string>
    <string>--ai-service-dir</string>
    <string>${AI_SERVICE_DIR}</string>
    <string>--p2p-port</string>
    <string>${P2P_PORT}</string>
  </array>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/resilience.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/resilience.log</string>
</dict>
</plist>
EOF

if is_truthy "$ENABLE_IMPROVEMENT_DAEMON_RAW"; then
cat > "$IMP_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ringrift.improvement</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${AI_SERVICE_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${LAUNCHD_PATH}</string>
    <key>PYTHONPATH</key>
    <string>${AI_SERVICE_DIR}</string>
    <key>USE_P2P_ORCHESTRATOR</key>
    <string>true</string>
    <key>P2P_ORCHESTRATOR_URL</key>
    <string>http://localhost:${P2P_PORT}</string>
    <key>RINGRIFT_IMPROVEMENT_LEADER_ONLY</key>
    <string>1</string>
    <key>RINGRIFT_CLUSTER_AUTH_TOKEN_FILE</key>
    <string>${TOKEN_FILE}</string>
  </dict>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>${AI_SERVICE_DIR}/scripts/unified_ai_loop.py</string>
    <string>--foreground</string>
  </array>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/improvement.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/improvement.log</string>
</dict>
</plist>
EOF
else
  rm -f "$IMP_PLIST" 2>/dev/null || true
fi

echo "Wrote:"
echo "  $P2P_PLIST"
echo "  $RES_PLIST"
if is_truthy "$ENABLE_IMPROVEMENT_DAEMON_RAW"; then
  echo "  $IMP_PLIST"
fi
echo ""
echo "Loading LaunchAgents..."
launchctl unload "$P2P_PLIST" >/dev/null 2>&1 || true
launchctl unload "$RES_PLIST" >/dev/null 2>&1 || true
launchctl unload "$IMP_PLIST" >/dev/null 2>&1 || true
launchctl load "$P2P_PLIST"
launchctl load "$RES_PLIST"
if is_truthy "$ENABLE_IMPROVEMENT_DAEMON_RAW"; then
  launchctl load "$IMP_PLIST"
else
  echo "Continuous improvement daemon disabled (set RINGRIFT_ENABLE_IMPROVEMENT_DAEMON=1 to enable)"
fi
echo ""
echo "Done."
echo ""
echo "Check status:"
echo "  tail -f \"$LOG_DIR/p2p.log\""
echo "  tail -f \"$LOG_DIR/resilience.log\""
echo "  curl \"http://localhost:$P2P_PORT/health\""
