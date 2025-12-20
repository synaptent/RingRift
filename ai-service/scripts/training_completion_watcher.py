#!/usr/bin/env python3
"""Training Completion Watcher - Auto-triggers Elo tournaments when training completes.

Monitors training log files and triggers Elo calibration tournaments when
training jobs complete successfully.

Usage:
    # Run as daemon
    python scripts/training_completion_watcher.py --daemon

    # Check once and run pending tournaments
    python scripts/training_completion_watcher.py --check-once

    # Add to cron (every 10 minutes):
    */10 * * * * cd ~/ringrift/ai-service && python scripts/training_completion_watcher.py --check-once >> logs/watcher.log 2>&1
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from scripts.lib.state_manager import load_json_state, save_json_state

LOG_DIR = AI_SERVICE_ROOT / "logs"
STATE_FILE = AI_SERVICE_ROOT / "data" / "training_watcher_state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_STATE = {"completed_trainings": [], "pending_elo": []}


def load_state() -> Dict:
    """Load watcher state."""
    return load_json_state(STATE_FILE, default=DEFAULT_STATE)


def save_state(state: Dict) -> None:
    """Save watcher state."""
    save_json_state(STATE_FILE, state)


def parse_log_for_completion(log_path: Path) -> Optional[Dict]:
    """Parse training log to check if training completed."""
    if not log_path.exists():
        return None

    try:
        content = log_path.read_text()

        # Check for completion indicators
        if "Training completed" in content or "Early stopping" in content:
            # Extract board type and players from filename
            # e.g., hex_2p_training.log, sq19_2p_training.log
            match = re.match(r"(\w+)_(\d)p_training\.log", log_path.name)
            if match:
                board_map = {
                    "hex": "hexagonal",
                    "sq19": "square19",
                    "sq8": "square8",
                    "hex8": "hex8",
                }
                board_raw = match.group(1)
                board_type = board_map.get(board_raw, board_raw)
                num_players = int(match.group(2))

                # Extract final metrics
                epoch_matches = re.findall(
                    r"Epoch \[(\d+)/(\d+)\].*Val Loss: ([\d.]+)",
                    content
                )
                if epoch_matches:
                    last_epoch = epoch_matches[-1]
                    return {
                        "board_type": board_type,
                        "num_players": num_players,
                        "final_epoch": int(last_epoch[0]),
                        "total_epochs": int(last_epoch[1]),
                        "final_val_loss": float(last_epoch[2]),
                        "log_file": str(log_path),
                        "completed_at": datetime.now().isoformat(),
                    }

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")

    return None


def check_training_logs() -> List[Dict]:
    """Check all training logs for completed trainings."""
    completed = []

    # Look for training logs
    log_patterns = [
        "hex_*_training.log",
        "sq*_*_training.log",
    ]

    for pattern in log_patterns:
        for log_path in LOG_DIR.glob(pattern):
            result = parse_log_for_completion(log_path)
            if result:
                completed.append(result)

    return completed


def run_elo_tournament(board_type: str, num_players: int) -> bool:
    """Run Elo tournament for a board/player configuration."""
    print(f"[Elo] Starting tournament for {board_type} {num_players}p...")

    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "run_model_elo_tournament.py"),
        "--board", board_type,
        "--players", str(num_players),
        "--games", "30",
        "--quick",
        "--include-baselines",
    ]

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,
        )

        if result.returncode == 0:
            print(f"[Elo] Tournament complete for {board_type} {num_players}p")
            return True
        else:
            # Check if it's a load issue
            if "load too high" in result.stderr.lower():
                print(f"[Elo] System load too high, will retry later")
                return False
            print(f"[Elo] Tournament failed: {result.stderr[:300]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[Elo] Tournament timeout")
        return False
    except Exception as e:
        print(f"[Elo] Error: {e}")
        return False


def check_system_load() -> float:
    """Get current system load average."""
    try:
        load = os.getloadavg()[0]
        return load
    except Exception:
        return 0.0


def run_check(state: Dict) -> Dict:
    """Run a single check iteration."""
    # Find newly completed trainings
    completed = check_training_logs()
    completed_keys = set(state.get("completed_trainings", []))

    new_completions = []
    for training in completed:
        key = f"{training['board_type']}_{training['num_players']}p_{training['log_file']}"
        if key not in completed_keys:
            new_completions.append(training)
            completed_keys.add(key)
            print(f"[Watcher] New training completed: {training['board_type']} "
                  f"{training['num_players']}p (val_loss={training['final_val_loss']:.4f})")

            # Add to pending Elo queue
            elo_key = f"{training['board_type']}_{training['num_players']}p"
            if elo_key not in state.get("pending_elo", []):
                state.setdefault("pending_elo", []).append(elo_key)

    state["completed_trainings"] = list(completed_keys)

    # Try to run pending Elo tournaments if load is acceptable
    load = check_system_load()
    print(f"[Watcher] System load: {load:.1f}")

    if load < 50:  # Only run if load is reasonable
        pending = state.get("pending_elo", [])
        still_pending = []

        for elo_key in pending:
            parts = elo_key.rsplit("_", 1)
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))

            success = run_elo_tournament(board_type, num_players)
            if not success:
                still_pending.append(elo_key)

        state["pending_elo"] = still_pending
    else:
        print(f"[Watcher] Load too high ({load:.1f}), skipping Elo tournaments")

    return state


def run_daemon(interval: int = 300):
    """Run as continuous daemon."""
    print(f"[Watcher] Starting daemon (interval: {interval}s)")
    state = load_state()

    while True:
        try:
            state = run_check(state)
            save_state(state)
        except Exception as e:
            print(f"[Watcher] Error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Training completion watcher")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--check-once", action="store_true", help="Check once and exit")
    parser.add_argument("--interval", type=int, default=300, help="Daemon check interval")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    if args.status:
        state = load_state()
        print("Completed trainings:", len(state.get("completed_trainings", [])))
        print("Pending Elo tournaments:", state.get("pending_elo", []))
        return 0

    if args.daemon:
        run_daemon(args.interval)
    elif args.check_once:
        state = load_state()
        state = run_check(state)
        save_state(state)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
