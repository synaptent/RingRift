#!/usr/bin/env python3
"""Monitor hex8_2p game generation progress across cluster."""

import shlex
import subprocess
import time
import sys
from datetime import datetime

NODES = [
    ("nebius-backbone-1", "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_cluster ubuntu@89.169.112.47"),
    ("runpod-a100-1", "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p 33085 root@38.128.233.145"),
    ("runpod-a100-2", "ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i ~/.ssh/id_ed25519 -p 11681 root@104.255.9.187"),
]

def run_ssh_cmd(ssh_cmd: str, remote_cmd: str, timeout: int = 10) -> str:
    """Run a command via SSH without shell=True for security."""
    try:
        ssh_parts = shlex.split(ssh_cmd)
        result = subprocess.run(
            ssh_parts + [remote_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else f"error: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"error: {e}"

def get_game_count(ssh_cmd: str, base_path: str) -> int:
    # Use Python to count since sqlite3 may not be installed
    remote_cmd = f'''cd {base_path} && python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('data/games/hex8_2p.db')
    count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
    print(count)
    conn.close()
except Exception: print(0)
"'''
    result = run_ssh_cmd(ssh_cmd, remote_cmd, timeout=15)
    try:
        return int(result.strip().split('\n')[-1])
    except (ValueError, AttributeError, IndexError):
        return 0

def get_selfplay_progress(ssh_cmd: str, log_pattern: str) -> str:
    remote_cmd = f'grep -E "Progress:|Game [0-9]+/" {log_pattern} 2>/dev/null | tail -1'
    result = run_ssh_cmd(ssh_cmd, remote_cmd)
    return result if result and "error" not in result.lower() else "no progress"

def get_local_count() -> int:
    try:
        import sqlite3
        from pathlib import Path
        total = 0
        for db in Path("/Users/armand/Development/RingRift/ai-service/data/games").glob("*hex8*2p*.db"):
            try:
                conn = sqlite3.connect(str(db))
                count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
                conn.close()
                total += count
            except (sqlite3.Error, TypeError):
                pass
        return total
    except (ImportError, OSError):
        return 0

def monitor_once():
    print(f"\n{'='*60}")
    print(f"hex8_2p Game Generation Monitor - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    total_remote = 0

    for name, ssh_cmd in NODES:
        # Determine paths based on node
        if "nebius" in name:
            base_path = "~/ringrift/ai-service"
            log_path = "/tmp/selfplay_hex8_2p.log"
        else:
            base_path = "/workspace/ringrift/ai-service"
            log_path = "/workspace/ringrift/ai-service/logs/*hex8*2p*.log"

        count = get_game_count(ssh_cmd, base_path)
        progress = get_selfplay_progress(ssh_cmd, log_path)
        total_remote += count

        status = "✓" if count > 0 else "○"
        print(f"  {status} {name}: {count:,} games | {progress[:50]}")

    local_count = get_local_count()
    print(f"\n  📦 Local hex8_2p: {local_count:,} games")
    print(f"  🌐 Remote total: {total_remote:,} games")
    print(f"  📊 Combined: {local_count + total_remote:,} games")

    # Target check
    target = 10000
    current = local_count + total_remote
    if current >= target:
        print(f"\n  🎉 TARGET REACHED! Ready for training.")
        return True
    else:
        pct = (current / target) * 100
        print(f"\n  🎯 Progress: {pct:.1f}% of {target:,} target")

    return False

def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"Monitoring hex8_2p generation every {interval}s (Ctrl+C to stop)")

    while True:
        try:
            done = monitor_once()
            if done:
                print("\nTarget reached! Run sync and training.")
                break
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break

if __name__ == "__main__":
    main()
