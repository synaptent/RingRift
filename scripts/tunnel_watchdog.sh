#\!/bin/bash
# Tunnel Auto-Recovery Watchdog
# Monitors tunnels and attempts automatic recovery for Lambda nodes

LOG_FILE="/var/log/tunnel_watchdog.log"

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

SSH_KEY="/home/ubuntu/.ssh/id_cluster"

# Lambda node tunnel mappings
LAMBDA_PORTS="8771 8772 8773 8774 8775 8776 8779 8781"
declare -A LAMBDA_IPS
LAMBDA_IPS[8771]="100.97.158.115"   # lambda-gh200-a
LAMBDA_IPS[8772]="100.82.197.120"   # lambda-gh200-b
LAMBDA_IPS[8773]="100.110.145.4"    # lambda-gh200-c
LAMBDA_IPS[8774]="100.86.7.119"     # lambda-gh200-d
LAMBDA_IPS[8775]="100.99.151.5"     # lambda-gh200-e
LAMBDA_IPS[8776]="100.115.35.57"    # lambda-gh200-i
LAMBDA_IPS[8779]="100.116.32.54"    # lambda-gh200-h
LAMBDA_IPS[8781]="100.91.213.103"   # lambda-gh200-g

# Vast nodes (no auto-recovery)
VAST_PORTS="8777 8778 8780 8782"

recover_lambda_tunnel() {
    local port=$1
    local ip=${LAMBDA_IPS[$port]}
    
    log "Attempting recovery for Lambda tunnel on port $port (IP: $ip)"
    
    if ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$ip "sudo systemctl restart autossh-tunnel" 2>/dev/null; then
        log "Successfully restarted autossh-tunnel on $ip"
        sleep 5
        if nc -z -w3 localhost $port 2>/dev/null; then
            log "Tunnel on port $port is now UP"
            return 0
        else
            log "Tunnel on port $port still DOWN after restart"
            return 1
        fi
    else
        log "Failed to restart on $ip - SSH failed"
        return 1
    fi
}

check_and_recover() {
    local recovery_count=0
    local failed_count=0
    
    log "Starting tunnel health check..."
    
    # Check Lambda tunnels
    for port in $LAMBDA_PORTS; do
        if \! nc -z -w2 localhost $port 2>/dev/null; then
            log "Lambda tunnel on port $port is DOWN"
            if recover_lambda_tunnel $port; then
                recovery_count=$((recovery_count + 1))
            else
                failed_count=$((failed_count + 1))
            fi
        fi
    done
    
    # Check Vast tunnels (log only)
    for port in $VAST_PORTS; do
        if \! nc -z -w2 localhost $port 2>/dev/null; then
            log "Vast tunnel on port $port is DOWN - manual intervention required"
            failed_count=$((failed_count + 1))
        fi
    done
    
    log "Check complete: $recovery_count recovered, $failed_count still down"
}

log "========================================"
log "Tunnel Watchdog Started"
check_and_recover
