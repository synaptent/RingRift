#!/usr/bin/env python3
import os, sys, time, sqlite3, subprocess
from datetime import datetime

DB_PATH = os.path.expanduser("~/ringrift/ai-service/data/games/selfplay.db")
MAX_EXPORT_GAMES = 50
SAMPLE_EVERY = 10

# Config: (board_type, num_players) -> threshold
CONFIGS = {
    ("square8", 2): 500,
    ("square8", 3): 200,
    ("square8", 4): 200,
    ("square19", 2): 100,
    ("square19", 3): 50,
    ("square19", 4): 50,
    ("hexagonal", 2): 100,
    ("hexagonal", 3): 50,
    ("hexagonal", 4): 50,
}

# Track last training count per config - START AT 0 to train on existing data
last_counts = {k: 0 for k in CONFIGS}

def get_config_counts():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cur = conn.cursor()
    cur.execute("SELECT board_type, num_players, COUNT(*) FROM games GROUP BY board_type, num_players")
    result = {(row[0], row[1]): row[2] for row in cur.fetchall()}
    conn.close()
    return result

def run_training(board_type, num_players, current_count):
    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = f"{board_type[:3]}{num_players}p"
    print(f"[{ts}] Training {short}...", flush=True)

    npz = f"data/training/selfplay_{short}_{ts}.npz"
    exp_cmd = [sys.executable, "scripts/export_replay_dataset.py",
        "--db", DB_PATH, "--board-type", board_type, "--num-players", str(num_players),
        "--output", npz, "--sample-every", str(SAMPLE_EVERY), "--max-games", str(MAX_EXPORT_GAMES)]

    print(f"  Exporting {short}...", flush=True)
    try:
        r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  Export failed: {r.stderr[:200] if r.stderr else 'unknown'}", flush=True)
            last_counts[key] = current_count
            return False
    except subprocess.TimeoutExpired:
        print(f"  Export timeout", flush=True)
        last_counts[key] = current_count
        return False

    run_dir = f"data/training/runs/auto_{short}_{ts}"
    train_cmd = [sys.executable, "scripts/run_nn_training_baseline.py",
        "--board", board_type, "--num-players", str(num_players), "--run-dir", run_dir,
        "--data-path", npz, "--epochs", "5"]

    print(f"  Training {short}...", flush=True)
    try:
        r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  Train failed: {r.stderr[:200] if r.stderr else 'unknown'}", flush=True)
            last_counts[key] = current_count
            return False
    except subprocess.TimeoutExpired:
        print(f"  Train timeout", flush=True)
        last_counts[key] = current_count
        return False

    print(f"  {short} done!", flush=True)
    last_counts[key] = current_count
    return True

def main():
    global last_counts
    print("Multi-Config Training Loop Started (fresh counts)", flush=True)
    print(f"Will train configs above threshold immediately", flush=True)

    while True:
        try:
            current_counts = get_config_counts()
            ts = datetime.now().strftime("%H:%M:%S")

            # Status line
            status_parts = []
            for (bt, np), threshold in CONFIGS.items():
                curr = current_counts.get((bt, np), 0)
                last = last_counts.get((bt, np), 0)
                new = curr - last
                if curr > 0:
                    short = f"{bt[:3]}{np}p"
                    status_parts.append(f"{short}:{curr}(+{new}/{threshold})")

            print(f"[{ts}] {' '.join(status_parts)}", flush=True)

            # Check for training triggers - train ONE config per iteration to avoid overload
            for (bt, np), threshold in CONFIGS.items():
                curr = current_counts.get((bt, np), 0)
                last = last_counts.get((bt, np), 0)
                if curr - last >= threshold:
                    run_training(bt, np, curr)
                    break  # Only train one per iteration

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
