#!/bin/bash
# Tunnel Manager - start, stop, restart tunnels

ACTION=${1:-status}

PRIMARY_PORTS="8771 8772 8773 8774 8775 8776 8777 8778 8779 8780 8781 8782"

case $ACTION in
    status)
        echo "=== Primary Hub Tunnel Status ==="
        for port in $PRIMARY_PORTS; do
            if ss -tlnp | grep -q ":$port "; then
                echo "Port $port: LISTENING"
            else
                echo "Port $port: DOWN"
            fi
        done
        ;;
    
    check)
        echo "=== Quick Health Check ==="
        for port in $PRIMARY_PORTS; do
            if nc -z -w 1 127.0.0.1 $port 2>/dev/null; then
                node=$(curl -s --connect-timeout 2 http://127.0.0.1:$port/health 2>/dev/null | grep -o '"node_id":"[^"]*"' | cut -d'"' -f4)
                echo "Port $port: OK ($node)"
            else
                echo "Port $port: UNREACHABLE"
            fi
        done
        ;;
    
    restart-systemd)
        echo "=== Restarting systemd tunnels ==="
        for node in lambda-gh200-h lambda-gh200-a lambda-gh200-c lambda-gh200-g lambda-gh200-i lambda-a10; do
            ip=$(grep $node /home/ubuntu/ringrift/docs/SSH_ACCESS_GUIDE.md 2>/dev/null | grep -oE '100\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            if [ -n "$ip" ]; then
                echo "Restarting tunnel on $node ($ip)..."
                ssh -o ConnectTimeout=10 ubuntu@$ip "sudo systemctl restart autossh-tunnel" 2>/dev/null && echo "  OK" || echo "  FAILED"
            fi
        done
        ;;
    
    *)
        echo "Usage: $0 {status|check|restart-systemd}"
        exit 1
        ;;
esac
