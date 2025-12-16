#!/usr/bin/env python3
"""Simple cluster alerting script.

Checks cluster health and sends alerts via webhook.
Configure RINGRIFT_WEBHOOK_URL environment variable.

Usage:
    # Run once
    python scripts/cluster_alert.py

    # Run as daemon (check every 5 minutes)  
    python scripts/cluster_alert.py --daemon --interval 300
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

WEBHOOK_URL = os.environ.get("RINGRIFT_WEBHOOK_URL", "")

NODES = [
    "lambda-gh200-a", "lambda-gh200-b", "lambda-gh200-c", "lambda-gh200-d",
    "lambda-gh200-e", "lambda-gh200-g", "lambda-gh200-h", "lambda-gh200-i",
    "lambda-gh200-k", "lambda-gh200-l", "lambda-2xh100"
]

GPU_UTIL_THRESHOLD = 20  # Alert if below this
WORKER_MIN_COUNT = 5     # Alert if fewer workers


def run_ssh(host: str, cmd: str, timeout: int = 10) -> str:
    """Run SSH command and return output."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, cmd],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip()
    except Exception:
        return ""


def send_alert(message: str, level: str = "warning"):
    """Send alert via webhook."""
    if not WEBHOOK_URL:
        print(f"[{level.upper()}] {message}")
        return
    
    payload = {
        "text": f"🔔 **RingRift Cluster Alert** [{level.upper()}]\n{message}",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        subprocess.run(
            ["curl", "-X", "POST", "-H", "Content-Type: application/json",
             "-d", json.dumps(payload), WEBHOOK_URL],
            capture_output=True, timeout=10
        )
    except Exception as e:
        print(f"Failed to send alert: {e}")


def check_cluster():
    """Check cluster health and return alerts."""
    alerts = []
    
    for host in NODES:
        # Check GPU utilization
        gpu_util = run_ssh(host, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")
        if gpu_util:
            try:
                util = int(gpu_util.split()[0])
                if util < GPU_UTIL_THRESHOLD:
                    alerts.append(f"{host}: GPU utilization low ({util}%)")
            except ValueError:
                pass
        else:
            alerts.append(f"{host}: Unreachable or no GPU")
        
        # Check worker count
        workers = run_ssh(host, "ps aux | grep -E '(selfplay|train)' | grep -v grep | wc -l")
        if workers:
            try:
                count = int(workers)
                if count < WORKER_MIN_COUNT:
                    alerts.append(f"{host}: Low worker count ({count})")
            except ValueError:
                pass
    
    return alerts


def main():
    parser = argparse.ArgumentParser(description="Cluster alerting")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")
    args = parser.parse_args()
    
    print(f"Cluster alerting started (webhook: {'configured' if WEBHOOK_URL else 'not configured'})")
    
    while True:
        alerts = check_cluster()
        
        if alerts:
            message = f"Issues detected at {datetime.now().strftime('%H:%M')}:\n" + "\n".join(f"• {a}" for a in alerts)
            send_alert(message)
        else:
            print(f"[{datetime.now().strftime('%H:%M')}] All nodes healthy")
        
        if not args.daemon:
            break
        
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
