#!/usr/bin/env python3
"""Remote Watchdog - Deploy to Lambda instances for local monitoring and restart.

This script runs ON the remote instances to monitor and restart jobs locally,
avoiding SSH connection issues.

Deploy: scp scripts/remote_watchdog.py ubuntu@host:~/ringrift/ai-service/scripts/
Run: nohup python3 scripts/remote_watchdog.py > /tmp/watchdog.log 2>&1 &
"""

import subprocess
import time
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        get_disk_usage as unified_get_disk_usage,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    unified_get_disk_usage = None
    RESOURCE_LIMITS = None

PROJECT_DIR = Path(__file__).parent.parent
LOG_FILE = Path("/tmp/watchdog.log")
STATE_FILE = Path("/tmp/watchdog_state.json")

# Configuration
CHECK_INTERVAL = 120  # 2 minutes
MIN_SELFPLAY_JOBS = 3
SELFPLAY_CONFIGS = [
    {"board": "square8", "players": 2, "games": 2000},
    {"board": "square8", "players": 3, "games": 1000},
    {"board": "square8", "players": 4, "games": 500},
    {"board": "hex", "players": 2, "games": 500},
    {"board": "square19", "players": 2, "games": 500},
]

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"selfplay_idx": 0, "restarts": 0, "games_completed": 0}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def count_selfplay_jobs():
    result = subprocess.run(
        ["pgrep", "-f", "run_hybrid_selfplay|run_self_play"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

def count_training_jobs():
    result = subprocess.run(
        ["pgrep", "-f", "train_nnue|train.py"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

def count_cmaes_jobs():
    result = subprocess.run(
        ["pgrep", "-f", "cmaes"],
        capture_output=True, text=True
    )
    return len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0

def start_selfplay(config, seed_offset=0):
    board = config["board"]
    players = config["players"]
    games = config["games"]
    seed = int(time.time()) + seed_offset
    
    output_dir = PROJECT_DIR / f"data/selfplay/{board}_{players}p"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, str(PROJECT_DIR / "scripts/run_hybrid_selfplay.py"),
        "--num-games", str(games),
        "--board-type", board,
        "--num-players", str(players),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
    ]
    
    log_file = f"/tmp/selfplay_{board}_{players}p.log"
    log(f"Starting selfplay: {board} {players}p, {games} games")
    
    with open(log_file, "a") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, cwd=str(PROJECT_DIR),
                        env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)})

def restart_selfplay_if_needed(state):
    current = count_selfplay_jobs()
    log(f"Selfplay jobs: {current} (min: {MIN_SELFPLAY_JOBS})")
    
    if current < MIN_SELFPLAY_JOBS:
        needed = MIN_SELFPLAY_JOBS - current
        for i in range(needed):
            idx = (state["selfplay_idx"] + i) % len(SELFPLAY_CONFIGS)
            start_selfplay(SELFPLAY_CONFIGS[idx], seed_offset=i)
        state["selfplay_idx"] = (state["selfplay_idx"] + needed) % len(SELFPLAY_CONFIGS)
        state["restarts"] += needed
        save_state(state)

def get_disk_usage():
    """Get disk usage percentage.

    Uses unified resource_guard utilities when available for consistent
    80% max utilization enforcement across the codebase.
    """
    # Use unified utilities when available
    if HAS_RESOURCE_GUARD and unified_get_disk_usage is not None:
        try:
            percent, _, _ = unified_get_disk_usage(str(PROJECT_DIR))
            return f"{percent:.1f}%"
        except Exception:
            pass  # Fall through to original implementation

    # Fallback to original implementation
    result = subprocess.run(["df", "-h", str(PROJECT_DIR)], capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 5:
            return parts[4]  # Use percentage
    return "unknown"

def main():
    log("=== Remote Watchdog Started ===")
    log(f"Project dir: {PROJECT_DIR}")
    log(f"Check interval: {CHECK_INTERVAL}s")
    
    state = load_state()
    
    while True:
        try:
            log("--- Health Check ---")
            
            # Count jobs
            selfplay = count_selfplay_jobs()
            training = count_training_jobs()
            cmaes = count_cmaes_jobs()
            disk = get_disk_usage()
            
            log(f"Jobs: selfplay={selfplay}, training={training}, cmaes={cmaes}, disk={disk}")
            
            # Restart selfplay if needed
            restart_selfplay_if_needed(state)
            
            # Check disk space warning - 70% limit enforced 2025-12-16
            if disk != "unknown":
                pct = int(disk.replace("%", ""))
                if pct > 70:
                    log(f"WARNING: Disk usage at {disk}! (limit: 70%)")
            
            log(f"State: restarts={state['restarts']}")
            
        except Exception as e:
            log(f"ERROR: {e}")
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
