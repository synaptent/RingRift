#!/usr/bin/env python3
"""Memory guard — kills master loop if RSS exceeds threshold.

Run as a cron job every 5 minutes:
  */5 * * * * cd ~/Development/RingRift/ai-service && python3 scripts/memory_guard.py >> /tmp/memory_guard.log 2>&1
"""
import os
import signal
import subprocess
import sys
from datetime import datetime

MAX_RSS_MB = 8000  # 8GB threshold — master loop shouldn't need more than 2-3GB


def main():
    try:
        r = subprocess.run(
            ["pgrep", "-f", "master_loop.py"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return  # Not running

        for pid_str in r.stdout.strip().split("\n"):
            pid = int(pid_str.strip())
            # Read RSS from /proc or ps
            ps = subprocess.run(
                ["ps", "-p", str(pid), "-o", "rss="],
                capture_output=True, text=True, timeout=5
            )
            rss_kb = int(ps.stdout.strip())
            rss_mb = rss_kb // 1024

            if rss_mb > MAX_RSS_MB:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] MEMORY GUARD: master_loop PID {pid} RSS={rss_mb}MB > {MAX_RSS_MB}MB — killing")
                os.kill(pid, signal.SIGTERM)
                return
    except Exception as e:
        print(f"Memory guard error: {e}")


if __name__ == "__main__":
    main()
