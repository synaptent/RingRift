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

P2P_PLIST="${PLIST_DIR}/com.ringrift.p2p.plist"
RES_PLIST="${PLIST_DIR}/com.ringrift.resilience.plist"

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
    <key>PYTHONPATH</key>
    <string>${AI_SERVICE_DIR}</string>
    <key>RINGRIFT_CLUSTER_AUTH_TOKEN</key>
    <string>${RINGRIFT_CLUSTER_AUTH_TOKEN:-}</string>
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
    <key>PYTHONPATH</key>
    <string>${AI_SERVICE_DIR}</string>
    <key>RINGRIFT_CLUSTER_AUTH_TOKEN</key>
    <string>${RINGRIFT_CLUSTER_AUTH_TOKEN:-}</string>
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

echo "Wrote:"
echo "  $P2P_PLIST"
echo "  $RES_PLIST"
echo ""
echo "Loading LaunchAgents..."
launchctl unload "$P2P_PLIST" >/dev/null 2>&1 || true
launchctl unload "$RES_PLIST" >/dev/null 2>&1 || true
launchctl load "$P2P_PLIST"
launchctl load "$RES_PLIST"
echo ""
echo "Done."
echo ""
echo "Check status:"
echo "  tail -f \"$LOG_DIR/p2p.log\""
echo "  tail -f \"$LOG_DIR/resilience.log\""
echo "  curl \"http://localhost:$P2P_PORT/health\""
