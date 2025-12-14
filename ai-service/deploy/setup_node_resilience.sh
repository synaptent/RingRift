#!/bin/bash
# Setup node resilience on a RingRift cluster node
#
# Usage:
#   ./setup_node_resilience.sh <node-id> <coordinator-url>
#
# Example:
#   ./setup_node_resilience.sh vast-5090-quad http://192.222.53.22:8770

set -e

NODE_ID="${1:?Usage: $0 <node-id> <coordinator-url>}"
COORDINATOR_URL="${2:?Usage: $0 <node-id> <coordinator-url>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_RINGRIFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RINGRIFT_DIR="${RINGRIFT_DIR:-$DEFAULT_RINGRIFT_DIR}"
LOG_DIR="/var/log/ringrift"
P2P_PORT="${P2P_PORT:-}"
# Default P2P listening port. Prefer 8770 everywhere; it avoids common Vast.ai
# collisions with Jupyter binding to 8080. If you need an externally-exposed
# port-mapped listener, explicitly set P2P_PORT (and optionally
# RINGRIFT_ADVERTISE_*).
if [ -z "$P2P_PORT" ]; then
    P2P_PORT="8770"
fi
SSH_PORT="${SSH_PORT:-}"
P2P_VOTERS="${RINGRIFT_P2P_VOTERS:-}"
P2P_VOTERS="${P2P_VOTERS//[[:space:]]/}"

is_port_available() {
    local port="$1"
    python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
finally:
    try:
        s.close()
    except Exception:
        pass
sys.exit(0)
PY
}

# If the chosen port is already in use (e.g. Jupyter on Vast.ai), fall back to
# a known-safe port.
if ! is_port_available "$P2P_PORT"; then
    # If the port is already serving the orchestrator health endpoint, keep it.
    # This avoids bouncing to a new port on redeploy when the daemon is already
    # running on the intended port.
    if curl -s --connect-timeout 2 "http://localhost:${P2P_PORT}/health" | grep -q '"node_id"' 2>/dev/null; then
        echo "P2P_PORT=$P2P_PORT is in use by an existing orchestrator; keeping port"
    else
        echo "Warning: P2P_PORT=$P2P_PORT already in use; selecting fallback port"
        for candidate in 8770 8772 8773 8774 8780 8781; do
            if is_port_available "$candidate"; then
                P2P_PORT="$candidate"
                echo "Using fallback P2P_PORT=$P2P_PORT"
                break
            fi
        done
    fi
fi

echo "Setting up node resilience for $NODE_ID"
echo "Coordinator: $COORDINATOR_URL"
echo "RingRift dir: $RINGRIFT_DIR"

# Determine whether systemd is actually usable (systemctl may exist in containers).
HAS_USABLE_SYSTEMD=0
if command -v systemctl &> /dev/null && [ -d /etc/systemd/system ]; then
    # Some providers report `degraded` due to unrelated units failing; treat it
    # as usable for our purposes if systemd is otherwise operational.
    STATE="$(systemctl is-system-running 2>/dev/null || true)"
    if [ "$STATE" = "running" ] || [ "$STATE" = "degraded" ]; then
        HAS_USABLE_SYSTEMD=1
    fi
fi

# Try to infer SSH port from distributed_hosts.yaml if not explicitly provided.
if [ -z "$SSH_PORT" ]; then
    SSH_PORT="$(python3 - "$NODE_ID" "$RINGRIFT_DIR" <<'PY' 2>/dev/null || true
import sys
from pathlib import Path
try:
    import yaml
except Exception:
    sys.exit(1)

node_id = sys.argv[1]
ringrift_dir = Path(sys.argv[2])
cfg = ringrift_dir / "config" / "distributed_hosts.yaml"
if not cfg.exists():
    print("22")
    sys.exit(0)

data = yaml.safe_load(cfg.read_text()) or {}
hosts = data.get("hosts", {}) or {}
node = hosts.get(node_id, {}) or {}
port = node.get("ssh_port", 22) or 22
print(int(port))
PY
)"
fi
SSH_PORT="${SSH_PORT:-22}"

# Determine RingRift repo root (parent of ai-service).
RINGRIFT_ROOT="$(cd "$(dirname "$RINGRIFT_DIR")" && pwd)"

# Git 2.35+ safe.directory: services often run as root against an ubuntu-owned
# checkout; ensure root can run git commands (auto-update, status, etc.).
if command -v git &> /dev/null; then
    EXISTING_SAFE_DIRS="$(HOME=/root git config --global --get-all safe.directory 2>/dev/null || true)"
    if ! printf '%s\n' "$EXISTING_SAFE_DIRS" | grep -Fxq "$RINGRIFT_ROOT"; then
        HOME=/root git config --global --add safe.directory "$RINGRIFT_ROOT" 2>/dev/null || true
    fi

    # Prefer HTTPS remotes so root-run services can `git fetch/pull` without
    # relying on per-user SSH keys. (The repo is public; HTTPS avoids auth.)
    ORIGIN_URL="$(git -C "$RINGRIFT_ROOT" remote get-url origin 2>/dev/null || true)"
    HTTPS_URL=""
    case "$ORIGIN_URL" in
        git@github.com:*)
            HTTPS_URL="https://github.com/${ORIGIN_URL#git@github.com:}"
            ;;
        ssh://git@github.com/*)
            HTTPS_URL="https://github.com/${ORIGIN_URL#ssh://git@github.com/}"
            ;;
    esac
    if [ -n "$HTTPS_URL" ]; then
        git -C "$RINGRIFT_ROOT" remote set-url origin "$HTTPS_URL" 2>/dev/null || true
    fi
fi

# Create directories
mkdir -p /etc/ringrift
mkdir -p "$LOG_DIR"

# Write node configuration
cat > /etc/ringrift/node.conf << EOF
NODE_ID=$NODE_ID
COORDINATOR_URL=$COORDINATOR_URL
RINGRIFT_DIR=$RINGRIFT_DIR
P2P_PORT=$P2P_PORT
SSH_PORT=$SSH_PORT
EOF

# Quorum voter set for split-brain resistance. This must be consistent across
# the cluster. Prefer passing it via deploy tooling (env), which writes it to
# node.conf so systemd units and watchdog restarts inherit it.
if [ -n "$P2P_VOTERS" ]; then
    echo "RINGRIFT_P2P_VOTERS=$P2P_VOTERS" >> /etc/ringrift/node.conf
fi

# Persist an explicit advertised P2P port for port-mapped environments (Vast.ai)
# so the orchestrator can report a reachable endpoint even in minimal
# environments where VAST_* vars may not be propagated to daemons.
if [ -z "${RINGRIFT_ADVERTISE_PORT:-}" ]; then
    varname="VAST_TCP_PORT_${P2P_PORT}"
    mapped="${!varname:-}"
    if [ -n "$mapped" ]; then
        export RINGRIFT_ADVERTISE_PORT="$mapped"
    fi
fi
if [ -n "${RINGRIFT_ADVERTISE_HOST:-}" ]; then
    echo "RINGRIFT_ADVERTISE_HOST=$RINGRIFT_ADVERTISE_HOST" >> /etc/ringrift/node.conf
fi
if [ -n "${RINGRIFT_ADVERTISE_PORT:-}" ]; then
    echo "RINGRIFT_ADVERTISE_PORT=$RINGRIFT_ADVERTISE_PORT" >> /etc/ringrift/node.conf
fi

# Store cluster auth token in a root-only file (avoid leaking secrets into env files/cron).
TOKEN_FILE="/etc/ringrift/cluster_auth_token"
if [ -n "${RINGRIFT_CLUSTER_AUTH_TOKEN:-}" ]; then
    umask 077
    printf "%s" "$RINGRIFT_CLUSTER_AUTH_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi
if [ -f "$TOKEN_FILE" ]; then
    echo "RINGRIFT_CLUSTER_AUTH_TOKEN_FILE=$TOKEN_FILE" >> /etc/ringrift/node.conf
fi
chmod 600 /etc/ringrift/node.conf

echo "Created /etc/ringrift/node.conf"

# Install systemd services if available
if [ "$HAS_USABLE_SYSTEMD" = "1" ]; then
    echo "Installing systemd services..."

    # Copy service files
    cp "$RINGRIFT_DIR/deploy/systemd/ringrift-p2p.service" /etc/systemd/system/
    cp "$RINGRIFT_DIR/deploy/systemd/ringrift-resilience.service" /etc/systemd/system/

    # Rewrite hard-coded paths for non-root installations.
    if [ "$RINGRIFT_DIR" != "/root/ringrift/ai-service" ]; then
        sed -i "s|/root/ringrift/ai-service|$RINGRIFT_DIR|g" /etc/systemd/system/ringrift-*.service
        sed -i "s|/root/ringrift|$RINGRIFT_ROOT|g" /etc/systemd/system/ringrift-*.service
    fi

    # Reload and enable
    systemctl daemon-reload
    systemctl enable ringrift-p2p.service
    systemctl enable ringrift-resilience.service

    echo "Systemd services installed and enabled"
else
    echo "Systemd not available, using cron fallback"
fi

# Setup cron jobs as fallback/backup
CRON_FILE="/etc/cron.d/ringrift"
cat > "$CRON_FILE" << EOF
# RingRift node resilience cron jobs
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
PYTHONPATH=$RINGRIFT_DIR
P2P_PORT=$P2P_PORT
SSH_PORT=$SSH_PORT
RINGRIFT_CLUSTER_AUTH_TOKEN_FILE=/etc/ringrift/cluster_auth_token
RINGRIFT_P2P_VOTERS=$P2P_VOTERS

# Health check and reconnection every 5 minutes
*/5 * * * * root python3 $RINGRIFT_DIR/scripts/node_resilience.py --node-id $NODE_ID --coordinator $COORDINATOR_URL --ai-service-dir $RINGRIFT_DIR --p2p-port $P2P_PORT --once >> $LOG_DIR/cron.log 2>&1

# Node registration every 10 minutes
*/10 * * * * root python3 $RINGRIFT_DIR/scripts/register_node.py --node-id $NODE_ID --coordinator $COORDINATOR_URL --auto-ip --port $SSH_PORT >> $LOG_DIR/registration.log 2>&1

# Log rotation daily
0 0 * * * root find $LOG_DIR -name "*.log" -size +100M -exec truncate -s 10M {} \;
EOF

chmod 644 "$CRON_FILE"
echo "Cron jobs installed at $CRON_FILE"

# Create watchdog script
cat > /usr/local/bin/ringrift-watchdog << 'EOF'
#!/bin/bash
# Quick watchdog to restart services if they crash
set -a
source /etc/ringrift/node.conf
set +a

# Check P2P health
if ! curl -s --connect-timeout 5 "http://localhost:${P2P_PORT}/health" > /dev/null 2>&1; then
    echo "$(date): P2P orchestrator not responding, attempting restart"

    # Kill any zombie processes
    pkill -f '[p]2p_orchestrator.py' 2>/dev/null || true
    sleep 2

    # Start fresh
    cd "$RINGRIFT_DIR"
    PYTHONPATH="$RINGRIFT_DIR" nohup python3 scripts/p2p_orchestrator.py --node-id "$NODE_ID" --port "$P2P_PORT" --peers "$COORDINATOR_URL" --ringrift-path "$RINGRIFT_DIR/.." >> /var/log/ringrift/p2p.log 2>&1 &
    echo "$(date): P2P orchestrator restarted (PID $!)"
fi
EOF
chmod +x /usr/local/bin/ringrift-watchdog

# Add watchdog to crontab if not using systemd
if [ "$HAS_USABLE_SYSTEMD" != "1" ]; then
    echo "*/2 * * * * root /usr/local/bin/ringrift-watchdog >> $LOG_DIR/watchdog.log 2>&1" >> "$CRON_FILE"
fi

# Vast.ai containers often lack a running cron daemon and systemd. They do,
# however, invoke `/root/onstart.sh` on container start (see `/.launch`). Install
# an idempotent onstart hook so the orchestrator and resilience daemon come up
# automatically and restart after host-side disruption.
if [ "$HAS_USABLE_SYSTEMD" != "1" ]; then
    if [ -f "/.launch" ] || [ -n "${VAST_CONTAINERLABEL:-}" ] || [ -n "${VAST_TCP_PORT_22:-}" ]; then
        ONSTART="/root/onstart.sh"
        if [ ! -f "$ONSTART" ]; then
            echo '#!/bin/bash' > "$ONSTART"
        fi
        chmod +x "$ONSTART" 2>/dev/null || true

        # Remove any previous RingRift block and replace it.
        TMP_ONSTART="$(mktemp)"
        awk '
            BEGIN { skip=0 }
            /^# BEGIN RINGRIFT RESILIENCE/ { skip=1; next }
            /^# END RINGRIFT RESILIENCE/ { skip=0; next }
            skip==0 { print }
        ' "$ONSTART" > "$TMP_ONSTART" 2>/dev/null || cat "$ONSTART" > "$TMP_ONSTART"
        mv "$TMP_ONSTART" "$ONSTART"

        cat >> "$ONSTART" <<'EOF'

# BEGIN RINGRIFT RESILIENCE
(
  set -euo pipefail
  if [ -f /etc/ringrift/node.conf ]; then
    set -a
    source /etc/ringrift/node.conf
    set +a
  fi

  mkdir -p /var/log/ringrift || true
  export PYTHONPATH="${RINGRIFT_DIR:-/root/ringrift/ai-service}"
  export RINGRIFT_CLUSTER_AUTH_TOKEN_FILE="/etc/ringrift/cluster_auth_token"

  # Start P2P orchestrator if not healthy.
  if ! curl -s --connect-timeout 5 "http://localhost:${P2P_PORT:-8770}/health" >/dev/null 2>&1; then
    pkill -f '[p]2p_orchestrator.py' 2>/dev/null || true
    cd "${RINGRIFT_DIR:-/root/ringrift/ai-service}"
    nohup python3 scripts/p2p_orchestrator.py \
      --node-id "${NODE_ID:-unknown}" \
      --port "${P2P_PORT:-8770}" \
      --peers "${COORDINATOR_URL:-}" \
      --ringrift-path "${RINGRIFT_DIR:-/root/ringrift/ai-service}/.." \
      >> /var/log/ringrift/p2p.log 2>&1 &
  fi

  # Ensure node_resilience is running (uses a singleton lock).
  cd "${RINGRIFT_DIR:-/root/ringrift/ai-service}"
  nohup python3 scripts/node_resilience.py \
    --node-id "${NODE_ID:-unknown}" \
    --coordinator "${COORDINATOR_URL:-}" \
    --ai-service-dir "${RINGRIFT_DIR:-/root/ringrift/ai-service}" \
    --p2p-port "${P2P_PORT:-8770}" \
    >> /var/log/ringrift/resilience.log 2>&1 &
) &
# END RINGRIFT RESILIENCE
EOF

        echo "Installed /root/onstart.sh RingRift resilience hook"
    fi
fi

echo ""
echo "Node resilience setup complete!"
echo ""
echo "To start services now:"
if [ "$HAS_USABLE_SYSTEMD" = "1" ]; then
    echo "  systemctl start ringrift-p2p"
    echo "  systemctl start ringrift-resilience"
else
    echo "  python3 $RINGRIFT_DIR/scripts/p2p_orchestrator.py --node-id $NODE_ID --port $P2P_PORT --peers $COORDINATOR_URL &"
    echo "  python3 $RINGRIFT_DIR/scripts/node_resilience.py --node-id $NODE_ID --coordinator $COORDINATOR_URL &"
fi
echo ""
echo "To check status:"
echo "  curl http://localhost:$P2P_PORT/health"
echo "  tail -f $LOG_DIR/resilience.log"
