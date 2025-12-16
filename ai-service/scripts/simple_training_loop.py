#!/usr/bin/env python3
import os, sys, time, sqlite3, subprocess
from datetime import datetime

DB_PATH = os.path.expanduser("~/ringrift/ai-service/data/games/selfplay.db")
TRAINING_THRESHOLD = 500
MAX_EXPORT_GAMES = 50
SAMPLE_EVERY = 10

last_training_count = 0

def get_game_count():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM games")
    count = cur.fetchone()[0]
    conn.close()
    return count

def run_training_pipeline(current_count):
    global last_training_count
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[{ts}] Training triggered...", flush=True)
    
    npz = f"data/training/selfplay_sq8_2p_{ts}.npz"
    exp_cmd = [sys.executable, "scripts/export_replay_dataset.py",
        "--db", DB_PATH, "--board-type", "square8", "--num-players", "2",
        "--output", npz, "--sample-every", str(SAMPLE_EVERY), "--max-games", str(MAX_EXPORT_GAMES)]
    
    print("  Exporting...", flush=True)
    try:
        r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  Export failed", flush=True)
            last_training_count = current_count
            return False
    except subprocess.TimeoutExpired:
        print("  Export timeout", flush=True)
        last_training_count = current_count
        return False
    
    run_dir = f"data/training/runs/auto_{ts}"
    train_cmd = [sys.executable, "scripts/run_nn_training_baseline.py",
        "--board", "square8", "--num-players", "2", "--run-dir", run_dir,
        "--data-path", npz, "--epochs", "5"]
    
    print("  Training...", flush=True)
    try:
        r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  Train failed", flush=True)
            last_training_count = current_count
            return False
    except subprocess.TimeoutExpired:
        print("  Train timeout", flush=True)
        last_training_count = current_count
        return False
    
    print("  Done!", flush=True)
    last_training_count = current_count
    return True

def main():
    global last_training_count
    print("Training Loop Started", flush=True)
    last_training_count = get_game_count()
    print(f"  Start: {last_training_count} games", flush=True)
    
    while True:
        try:
            current = get_game_count()
            new = current - last_training_count
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {current} (+{new}/{TRAINING_THRESHOLD})", flush=True)
            if new >= TRAINING_THRESHOLD:
                run_training_pipeline(current)
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", flush=True)
            time.sleep(60)

if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/ringrift/ai-service"))
    os.environ["PYTHONPATH"] = os.path.expanduser("~/ringrift/ai-service")
    main()
