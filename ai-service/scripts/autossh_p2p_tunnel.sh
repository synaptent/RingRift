#!/bin/bash
#
# Autossh P2P Tunnel Manager
#
# Creates persistent SSH tunnels for P2P network connectivity:
# - Forward tunnel: Local node can reach remote P2P (for NAT-blocked remotes)
# - Reverse tunnel: Remote can reach local P2P (for NAT-blocked locals)
#
# Use cases:
# 1. Vast instances behind carrier NAT need reverse tunnels
# 2. Backup path when Tailscale/Cloudflare tunnel fails
# 3. Cross-provider mesh (Lambda <-> Vast direct tunnels)
#
# Usage:
#   # Reverse tunnel (run on NAT-blocked node like Vast)
#   ./autossh_p2p_tunnel.sh reverse --relay ubuntu@<coordinator-ip> --node-id vast-5080
#
#   # Forward tunnel (run on relay to reach NAT-blocked node)
#   ./autossh_p2p_tunnel.sh forward --target root@<target-ip> --node-id vast-3070-b
#
#   # Status check
#   ./autossh_p2p_tunnel.sh status
#
#   # Stop all tunnels
#   ./autossh_p2p_tunnel.sh stop

set -e

# Configuration
P2P_PORT="${P2P_PORT:-8770}"
TUNNEL_BASE_PORT="${TUNNEL_BASE_PORT:-21000}"  # Base port for tunnel forwarding
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
AUTOSSH_PIDFILE="/tmp/autossh_p2p.pid"
AUTOSSH_LOG="/tmp/autossh_p2p.log"
TUNNEL_INFO_FILE="/tmp/p2p_tunnel_info.json"
MONITOR_PORT=0  # Disable autossh monitoring port (use ServerAlive instead)

# SSH options for reliability
SSH_OPTS="-o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes -o BatchMode=yes -o ConnectTimeout=10"

usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
  reverse   Create reverse tunnel (NAT-blocked node -> relay)
  forward   Create forward tunnel (relay -> NAT-blocked node)
  status    Show tunnel status
  stop      Stop all tunnels
  restart   Restart tunnels

Options for reverse/forward:
  --relay <user@host>     Relay server (required for reverse)
  --target <user@host>    Target server (required for forward)
  --node-id <id>          Node identifier for port assignment
  --ssh-key <path>        SSH private key (default: ~/.ssh/id_ed25519)
  --p2p-port <port>       P2P port to forward (default: 8770)

Examples:
  # Vast instance creates reverse tunnel to Lambda H100
  $0 reverse --relay ubuntu@<coordinator-ip> --node-id vast-5080

  # Lambda creates forward tunnel to reach Vast via its reverse tunnel
  $0 forward --target root@localhost --local-port 21005 --remote-port 8770

  # Check all tunnel status
  $0 status
EOF
    exit 1
}

# Generate deterministic port from node-id
get_tunnel_port() {
    local node_id="$1"
    local hash=$(echo "$node_id" | md5sum | cut -c1-4)
    echo $((TUNNEL_BASE_PORT + (0x$hash % 999)))
}

# Check if autossh is installed
check_autossh() {
    if ! command -v autossh &> /dev/null; then
        echo "Installing autossh..."
        if command -v apt-get &> /dev/null; then
            apt-get update -qq && apt-get install -y -qq autossh
        elif command -v yum &> /dev/null; then
            yum install -y -q autossh
        else
            echo "ERROR: Cannot install autossh. Please install manually."
            exit 1
        fi
    fi
}

# Create reverse tunnel (run on NAT-blocked node)
# This allows the relay to connect back to this node's P2P port
create_reverse_tunnel() {
    local relay="$1"
    local node_id="$2"
    local tunnel_port=$(get_tunnel_port "$node_id")

    echo "=== Creating Reverse P2P Tunnel ==="
    echo "Node ID: $node_id"
    echo "Relay: $relay"
    echo "Tunnel Port: $tunnel_port"
    echo "P2P Port: $P2P_PORT"
    echo ""

    # Kill existing tunnel
    stop_tunnels

    # Create the tunnel
    # -R tunnel_port:localhost:p2p_port means:
    # - On relay, connections to localhost:tunnel_port forward to this node's P2P
    AUTOSSH_PIDFILE="$AUTOSSH_PIDFILE" \
    autossh -M $MONITOR_PORT -f \
        $SSH_OPTS \
        -i "$SSH_KEY" \
        -R ${tunnel_port}:localhost:${P2P_PORT} \
        -N "$relay"

    sleep 2

    if pgrep -f "autossh.*$relay" > /dev/null; then
        echo "Reverse tunnel established!"
        echo ""
        echo "From relay ($relay), P2P is now accessible at:"
        echo "  curl http://localhost:$tunnel_port/health"
        echo ""

        # Save tunnel info
        save_tunnel_info "reverse" "$relay" "$tunnel_port" "$node_id"

        # Register with P2P coordinator
        register_tunnel "$tunnel_port" "$node_id"
    else
        echo "ERROR: Tunnel failed to start"
        cat "$AUTOSSH_LOG" 2>/dev/null | tail -20
        exit 1
    fi
}

# Create forward tunnel (run on relay to reach remote P2P)
create_forward_tunnel() {
    local target="$1"
    local local_port="$2"
    local remote_port="${3:-$P2P_PORT}"

    echo "=== Creating Forward P2P Tunnel ==="
    echo "Target: $target"
    echo "Local Port: $local_port"
    echo "Remote P2P Port: $remote_port"
    echo ""

    # Create the tunnel
    # -L local_port:localhost:remote_port means:
    # - Local connections to localhost:local_port forward to target's P2P
    AUTOSSH_PIDFILE="$AUTOSSH_PIDFILE" \
    autossh -M $MONITOR_PORT -f \
        $SSH_OPTS \
        -i "$SSH_KEY" \
        -L ${local_port}:localhost:${remote_port} \
        -N "$target"

    sleep 2

    if pgrep -f "autossh.*$target" > /dev/null; then
        echo "Forward tunnel established!"
        echo ""
        echo "P2P is now accessible locally at:"
        echo "  curl http://localhost:$local_port/health"
    else
        echo "ERROR: Tunnel failed to start"
        exit 1
    fi
}

# Save tunnel info for P2P registration
save_tunnel_info() {
    local tunnel_type="$1"
    local endpoint="$2"
    local tunnel_port="$3"
    local node_id="$4"

    cat > "$TUNNEL_INFO_FILE" << EOF
{
    "tunnel_type": "$tunnel_type",
    "node_id": "$node_id",
    "tunnel_port": $tunnel_port,
    "endpoint": "$endpoint",
    "p2p_port": $P2P_PORT,
    "created_at": "$(date -Iseconds)",
    "pid": $(pgrep -f "autossh" | head -1 || echo 0)
}
EOF

    echo "Tunnel info saved to $TUNNEL_INFO_FILE"
}

# Register tunnel with P2P coordinator
register_tunnel() {
    local tunnel_port="$1"
    local node_id="$2"

    # Try to register with local P2P first, then coordinator
    for url in "http://localhost:$P2P_PORT" "https://p2p.ringrift.ai" "http://${COORDINATOR_IP:-localhost}:8770"; do
        echo "Registering tunnel with $url..."

        response=$(curl -s -X POST "$url/tunnel/register" \
            -H "Content-Type: application/json" \
            -d "{\"node_id\": \"$node_id\", \"tunnel_port\": $tunnel_port, \"type\": \"autossh\"}" \
            2>/dev/null || echo "")

        if [[ "$response" == *"ok"* ]] || [[ "$response" == *"registered"* ]]; then
            echo "Registered with $url"
            return 0
        fi
    done

    echo "Warning: Could not register tunnel with coordinator (P2P may not have this endpoint yet)"
}

# Show tunnel status
show_status() {
    echo "=== Autossh P2P Tunnel Status ==="
    echo ""

    # Check running tunnels
    echo "Running autossh processes:"
    if pgrep -a autossh 2>/dev/null; then
        pgrep -a autossh
    else
        echo "  (none)"
    fi
    echo ""

    # Check SSH tunnels
    echo "Active SSH tunnels:"
    if pgrep -a "ssh.*-[LR]" 2>/dev/null | grep -v grep; then
        pgrep -a "ssh.*-[LR]" | grep -v grep
    else
        echo "  (none)"
    fi
    echo ""

    # Show saved tunnel info
    if [ -f "$TUNNEL_INFO_FILE" ]; then
        echo "Last tunnel info:"
        cat "$TUNNEL_INFO_FILE"
        echo ""
    fi

    # Check listening ports in tunnel range
    echo "Listening ports in tunnel range ($TUNNEL_BASE_PORT-$((TUNNEL_BASE_PORT+999))):"
    netstat -tlnp 2>/dev/null | grep -E ":2[01][0-9]{3}" || ss -tlnp 2>/dev/null | grep -E ":2[01][0-9]{3}" || echo "  (none or netstat/ss not available)"
}

# Stop all tunnels
stop_tunnels() {
    echo "Stopping autossh tunnels..."
    pkill -f "autossh" 2>/dev/null || true
    pkill -f "ssh.*-[LR].*21[0-9][0-9][0-9]" 2>/dev/null || true
    rm -f "$AUTOSSH_PIDFILE"
    echo "Tunnels stopped"
}

# Restart tunnels using saved info
restart_tunnels() {
    if [ ! -f "$TUNNEL_INFO_FILE" ]; then
        echo "No saved tunnel info. Cannot restart."
        exit 1
    fi

    local tunnel_type=$(cat "$TUNNEL_INFO_FILE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tunnel_type',''))" 2>/dev/null)
    local endpoint=$(cat "$TUNNEL_INFO_FILE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('endpoint',''))" 2>/dev/null)
    local node_id=$(cat "$TUNNEL_INFO_FILE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('node_id',''))" 2>/dev/null)
    local tunnel_port=$(cat "$TUNNEL_INFO_FILE" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tunnel_port',0))" 2>/dev/null)

    if [ "$tunnel_type" == "reverse" ]; then
        create_reverse_tunnel "$endpoint" "$node_id"
    elif [ "$tunnel_type" == "forward" ]; then
        create_forward_tunnel "$endpoint" "$tunnel_port"
    else
        echo "Unknown tunnel type: $tunnel_type"
        exit 1
    fi
}

# Main
main() {
    local cmd="${1:-}"
    shift || true

    case "$cmd" in
        reverse)
            check_autossh
            local relay=""
            local node_id=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --relay) relay="$2"; shift 2 ;;
                    --node-id) node_id="$2"; shift 2 ;;
                    --ssh-key) SSH_KEY="$2"; shift 2 ;;
                    --p2p-port) P2P_PORT="$2"; shift 2 ;;
                    *) echo "Unknown option: $1"; usage ;;
                esac
            done
            [ -z "$relay" ] && echo "ERROR: --relay required" && exit 1
            [ -z "$node_id" ] && node_id="$(hostname | cut -c1-12)"
            create_reverse_tunnel "$relay" "$node_id"
            ;;
        forward)
            check_autossh
            local target=""
            local local_port=""
            local remote_port="$P2P_PORT"
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --target) target="$2"; shift 2 ;;
                    --local-port) local_port="$2"; shift 2 ;;
                    --remote-port) remote_port="$2"; shift 2 ;;
                    --ssh-key) SSH_KEY="$2"; shift 2 ;;
                    *) echo "Unknown option: $1"; usage ;;
                esac
            done
            [ -z "$target" ] && echo "ERROR: --target required" && exit 1
            [ -z "$local_port" ] && echo "ERROR: --local-port required" && exit 1
            create_forward_tunnel "$target" "$local_port" "$remote_port"
            ;;
        status)
            show_status
            ;;
        stop)
            stop_tunnels
            ;;
        restart)
            check_autossh
            restart_tunnels
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"
