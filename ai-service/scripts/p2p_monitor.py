#!/usr/bin/env python3
"""P2P cluster monitor - checks status every 10 minutes for 60 minutes."""

import json
import time
import urllib.request
from datetime import datetime

def get_p2p_status():
    """Fetch P2P status from local endpoint."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8770/health", timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=== P2P STABILITY MONITOR ===")
    print(f"Start time: {datetime.now()}")
    print()

    checks = []
    for i in range(1, 7):  # 6 checks = 60 minutes
        print(f"=== Check {i} at {datetime.now()} ===")

        status = get_p2p_status()

        if "error" in status:
            print(f"ERROR: {status['error']}")
        else:
            leader = status.get("leader_id") or "NONE"
            role = status.get("role")
            active = status.get("active_peers", 0)
            total = status.get("total_peers", 0)
            uptime = int(status.get("uptime_seconds", 0))

            print(f"Leader: {leader}")
            print(f"Role: {role}")
            print(f"Active peers: {active}")
            print(f"Total peers: {total}")
            print(f"Uptime: {uptime}s")

            checks.append({
                "check": i,
                "timestamp": datetime.now().isoformat(),
                "active_peers": active,
                "total_peers": total,
                "leader": leader
            })

            if active < 20:
                print(f"WARNING: Only {active} peers alive (target: 20+)")

        print()

        if i < 6:
            print("Sleeping 10 minutes...")
            time.sleep(600)

    print(f"=== MONITORING COMPLETE at {datetime.now()} ===")
    print()
    print("Summary:")
    if checks:
        avg_active = sum(c["active_peers"] for c in checks) / len(checks)
        min_active = min(c["active_peers"] for c in checks)
        max_active = max(c["active_peers"] for c in checks)
        print(f"  Average active peers: {avg_active:.1f}")
        print(f"  Min active peers: {min_active}")
        print(f"  Max active peers: {max_active}")
        print(f"  Target: 20+ peers")

if __name__ == "__main__":
    main()
