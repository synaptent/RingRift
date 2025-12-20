#!/usr/bin/env python3
"""Automated Hyperparameter A/B Testing Framework.

Uses the A/B testing infrastructure to systematically test hyperparameter
variations and find optimal settings.

Features:
1. Define hyperparameter experiments (learning rate, batch size, etc.)
2. Train variant models with different hyperparameters
3. Create A/B tests comparing baseline vs variant
4. Track results and recommend best hyperparameters
5. Support for multi-armed bandit style exploration

Usage:
    # Run a learning rate experiment
    python scripts/hyperparameter_ab_testing.py \
        --experiment learning_rate \
        --board-type square8 --num-players 2

    # List pending experiments
    python scripts/hyperparameter_ab_testing.py --list

    # Check experiment results
    python scripts/hyperparameter_ab_testing.py --results

    # Auto-run next recommended experiment
    python scripts/hyperparameter_ab_testing.py --auto
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add ai-service to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Configuration
EXPERIMENTS_DB = AI_SERVICE_ROOT / "data" / "hyperparameter_experiments.db"
ORCHESTRATOR_URL = os.environ.get("RINGRIFT_ORCHESTRATOR_URL", "http://localhost:8770")

# Hyperparameter experiment definitions
EXPERIMENT_DEFINITIONS = {
    "learning_rate": {
        "description": "Test different learning rates",
        "param": "learning_rate",
        "baseline": 0.001,
        "variants": [0.0001, 0.0005, 0.002, 0.005],
        "training_args": lambda v: ["--learning-rate", str(v)],
    },
    "batch_size": {
        "description": "Test different batch sizes",
        "param": "batch_size",
        "baseline": 256,
        "variants": [64, 128, 512, 1024],
        "training_args": lambda v: ["--batch-size", str(v)],
    },
    "weight_decay": {
        "description": "Test different weight decay values",
        "param": "weight_decay",
        "baseline": 0.0001,
        "variants": [0, 0.00001, 0.001, 0.01],
        "training_args": lambda v: ["--weight-decay", str(v)],
    },
    "hidden_size": {
        "description": "Test different hidden layer sizes",
        "param": "hidden_size",
        "baseline": 256,
        "variants": [128, 512, 1024],
        "training_args": lambda v: ["--hidden-size", str(v)],
    },
    "num_layers": {
        "description": "Test different network depths",
        "param": "num_layers",
        "baseline": 3,
        "variants": [2, 4, 5, 6],
        "training_args": lambda v: ["--num-layers", str(v)],
    },
    "dropout": {
        "description": "Test different dropout rates",
        "param": "dropout",
        "baseline": 0.1,
        "variants": [0, 0.05, 0.2, 0.3],
        "training_args": lambda v: ["--dropout", str(v)],
    },
    "mcts_simulations": {
        "description": "Test different MCTS simulation counts for selfplay",
        "param": "mcts_simulations",
        "baseline": 100,
        "variants": [50, 200, 400, 800],
        "training_args": lambda v: [],  # This affects selfplay, not training
        "selfplay_args": lambda v: ["--mcts-simulations", str(v)],
    },
    "temperature": {
        "description": "Test different exploration temperatures",
        "param": "temperature",
        "baseline": 1.0,
        "variants": [0.5, 0.8, 1.2, 1.5],
        "training_args": lambda v: [],
        "selfplay_args": lambda v: ["--temperature", str(v)],
    },
}


@dataclass
class Experiment:
    """A hyperparameter experiment."""
    id: str
    name: str
    board_type: str
    num_players: int
    param_name: str
    baseline_value: Any
    variant_value: Any
    status: str = "pending"  # pending, training, testing, completed, failed
    baseline_model_path: str | None = None
    variant_model_path: str | None = None
    ab_test_id: str | None = None
    winner: str | None = None  # "baseline", "variant", "tie"
    confidence: float = 0.0
    games_played: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    notes: str = ""


def get_db_connection() -> sqlite3.Connection:
    """Get connection to experiments database."""
    EXPERIMENTS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(EXPERIMENTS_DB)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            baseline_value TEXT NOT NULL,
            variant_value TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            baseline_model_path TEXT,
            variant_model_path TEXT,
            ab_test_id TEXT,
            winner TEXT,
            confidence REAL DEFAULT 0,
            games_played INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            notes TEXT DEFAULT ''
        );

        CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_exp_config ON experiments(board_type, num_players, param_name);

        CREATE TABLE IF NOT EXISTS experiment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL,
            metric TEXT NOT NULL,
            baseline_value REAL,
            variant_value REAL,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        );
    """)
    conn.commit()
    return conn


def create_experiment(
    name: str,
    board_type: str,
    num_players: int,
    baseline_model: str | None = None,
) -> Experiment | None:
    """Create a new hyperparameter experiment."""
    if name not in EXPERIMENT_DEFINITIONS:
        print(f"Unknown experiment: {name}")
        print(f"Available: {', '.join(EXPERIMENT_DEFINITIONS.keys())}")
        return None

    defn = EXPERIMENT_DEFINITIONS[name]
    conn = get_db_connection()

    # Check for existing pending/running experiments with same config
    existing = conn.execute("""
        SELECT id FROM experiments
        WHERE board_type = ? AND num_players = ? AND param_name = ? AND status IN ('pending', 'training', 'testing')
    """, (board_type, num_players, defn["param"])).fetchone()

    if existing:
        print(f"Experiment already in progress: {existing['id']}")
        conn.close()
        return None

    # Get next variant to test
    tested_variants = conn.execute("""
        SELECT variant_value FROM experiments
        WHERE board_type = ? AND num_players = ? AND param_name = ?
    """, (board_type, num_players, defn["param"])).fetchall()
    tested_values = {row["variant_value"] for row in tested_variants}

    # Find first untested variant
    variant_value = None
    for v in defn["variants"]:
        if str(v) not in tested_values:
            variant_value = v
            break

    if variant_value is None:
        print(f"All variants for {name} have been tested for {board_type}_{num_players}p")
        conn.close()
        return None

    exp_id = f"{name}_{board_type}_{num_players}p_{int(time.time())}"

    experiment = Experiment(
        id=exp_id,
        name=name,
        board_type=board_type,
        num_players=num_players,
        param_name=defn["param"],
        baseline_value=defn["baseline"],
        variant_value=variant_value,
        baseline_model_path=baseline_model,
    )

    conn.execute("""
        INSERT INTO experiments
        (id, name, board_type, num_players, param_name, baseline_value, variant_value,
         status, baseline_model_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        experiment.id, experiment.name, experiment.board_type, experiment.num_players,
        experiment.param_name, str(experiment.baseline_value), str(experiment.variant_value),
        experiment.status, experiment.baseline_model_path, experiment.created_at,
    ))
    conn.commit()
    conn.close()

    print(f"Created experiment: {exp_id}")
    print(f"  Testing {defn['param']}: {defn['baseline']} (baseline) vs {variant_value} (variant)")

    return experiment


def train_variant_model(experiment: Experiment) -> str | None:
    """Train a model with the variant hyperparameter."""
    defn = EXPERIMENT_DEFINITIONS[experiment.name]

    # Build training command
    training_script = AI_SERVICE_ROOT / "scripts" / "train_nnue.py"

    cmd = [
        sys.executable, str(training_script),
        "--board-type", experiment.board_type,
        "--num-players", str(experiment.num_players),
        "--output-suffix", f"_hp_{experiment.param_name}_{experiment.variant_value}",
    ]

    # Add variant-specific args
    if "training_args" in defn:
        cmd.extend(defn["training_args"](experiment.variant_value))

    print(f"Training variant model: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"Training failed: {result.stderr}")
            return None

        # Parse output for model path
        for line in result.stdout.split("\n"):
            if "Saved model to" in line or "Model saved:" in line:
                parts = line.split()
                for p in parts:
                    if p.endswith(".pt"):
                        return p

        # Fallback: construct expected path
        model_dir = AI_SERVICE_ROOT / "models"
        expected = model_dir / f"{experiment.board_type}_{experiment.num_players}p_hp_{experiment.param_name}_{experiment.variant_value}.pt"
        if expected.exists():
            return str(expected)

    except subprocess.TimeoutExpired:
        print("Training timed out")
    except Exception as e:
        print(f"Training error: {e}")

    return None


def create_ab_test_for_experiment(experiment: Experiment) -> str | None:
    """Create an A/B test comparing baseline and variant models."""
    if not experiment.baseline_model_path or not experiment.variant_model_path:
        print("Missing model paths for A/B test")
        return None

    try:
        url = f"{ORCHESTRATOR_URL}/abtest/create"
        data = json.dumps({
            "name": f"HP_{experiment.param_name}_{experiment.variant_value}",
            "description": f"Hyperparameter experiment: {experiment.param_name} = {experiment.variant_value} vs baseline {experiment.baseline_value}",
            "board_type": experiment.board_type,
            "num_players": experiment.num_players,
            "model_a": experiment.baseline_model_path,
            "model_b": experiment.variant_model_path,
            "target_games": 100,
            "confidence_threshold": 0.90,
            "metadata": {
                "experiment_id": experiment.id,
                "param_name": experiment.param_name,
                "baseline_value": experiment.baseline_value,
                "variant_value": experiment.variant_value,
            },
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("test_id")

    except Exception as e:
        print(f"Failed to create A/B test: {e}")
        return None


def check_experiment_status(experiment_id: str) -> dict[str, Any] | None:
    """Check the status of an experiment's A/B test."""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    conn.close()

    if not row:
        return None

    if not row["ab_test_id"]:
        return {"status": row["status"], "ab_test": None}

    try:
        url = f"{ORCHESTRATOR_URL}/abtest/status?test_id={row['ab_test_id']}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            ab_status = json.loads(resp.read().decode("utf-8"))

        return {
            "status": row["status"],
            "ab_test": ab_status,
            "experiment": dict(row),
        }

    except Exception as e:
        return {"status": row["status"], "error": str(e)}


def update_experiment_from_ab_test(experiment_id: str):
    """Update experiment based on A/B test results."""
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()

    if not row or not row["ab_test_id"]:
        conn.close()
        return

    try:
        url = f"{ORCHESTRATOR_URL}/abtest/status?test_id={row['ab_test_id']}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            ab_status = json.loads(resp.read().decode("utf-8"))

        if ab_status.get("status") == "completed":
            winner = ab_status.get("winner")
            if winner == "model_a":
                exp_winner = "baseline"
            elif winner == "model_b":
                exp_winner = "variant"
            else:
                exp_winner = "tie"

            stats = ab_status.get("stats", {})

            conn.execute("""
                UPDATE experiments
                SET status = 'completed', winner = ?, confidence = ?, games_played = ?,
                    completed_at = ?
                WHERE id = ?
            """, (
                exp_winner,
                stats.get("confidence", 0),
                stats.get("games_played", 0),
                datetime.utcnow().isoformat(),
                experiment_id,
            ))
            conn.commit()

            print(f"Experiment {experiment_id} completed: {exp_winner} wins!")

    except Exception as e:
        print(f"Error updating experiment: {e}")

    conn.close()


def get_experiment_recommendations(board_type: str, num_players: int) -> list[dict[str, Any]]:
    """Get recommended hyperparameter settings based on experiment results."""
    conn = get_db_connection()

    recommendations = []

    for _name, defn in EXPERIMENT_DEFINITIONS.items():
        param = defn["param"]

        # Get completed experiments for this param
        rows = conn.execute("""
            SELECT variant_value, winner, confidence, games_played
            FROM experiments
            WHERE board_type = ? AND num_players = ? AND param_name = ? AND status = 'completed'
            ORDER BY confidence DESC
        """, (board_type, num_players, param)).fetchall()

        if not rows:
            recommendations.append({
                "param": param,
                "recommendation": defn["baseline"],
                "confidence": 0,
                "source": "default",
                "experiments_run": 0,
            })
            continue

        # Find best variant
        best_variant = None
        best_confidence = 0

        for row in rows:
            if row["winner"] == "variant" and row["confidence"] > best_confidence:
                best_variant = row["variant_value"]
                best_confidence = row["confidence"]

        if best_variant:
            recommendations.append({
                "param": param,
                "recommendation": best_variant,
                "confidence": best_confidence,
                "source": "experiment",
                "experiments_run": len(rows),
            })
        else:
            # Baseline won all tests
            recommendations.append({
                "param": param,
                "recommendation": defn["baseline"],
                "confidence": max(r["confidence"] for r in rows),
                "source": "baseline_validated",
                "experiments_run": len(rows),
            })

    conn.close()
    return recommendations


def run_experiment(
    name: str,
    board_type: str,
    num_players: int,
    baseline_model: str | None = None,
) -> bool:
    """Run a complete hyperparameter experiment."""
    # Create experiment
    exp = create_experiment(name, board_type, num_players, baseline_model)
    if not exp:
        return False

    conn = get_db_connection()

    # Update status to training
    conn.execute("UPDATE experiments SET status = 'training' WHERE id = ?", (exp.id,))
    conn.commit()

    print(f"\n[1/3] Training variant model...")
    variant_path = train_variant_model(exp)

    if not variant_path:
        conn.execute("UPDATE experiments SET status = 'failed', notes = 'Training failed' WHERE id = ?", (exp.id,))
        conn.commit()
        conn.close()
        return False

    exp.variant_model_path = variant_path
    conn.execute("UPDATE experiments SET variant_model_path = ? WHERE id = ?", (variant_path, exp.id))
    conn.commit()

    # Update status to testing
    conn.execute("UPDATE experiments SET status = 'testing' WHERE id = ?", (exp.id,))
    conn.commit()

    print(f"\n[2/3] Creating A/B test...")
    ab_test_id = create_ab_test_for_experiment(exp)

    if not ab_test_id:
        conn.execute("UPDATE experiments SET status = 'failed', notes = 'A/B test creation failed' WHERE id = ?", (exp.id,))
        conn.commit()
        conn.close()
        return False

    exp.ab_test_id = ab_test_id
    conn.execute("UPDATE experiments SET ab_test_id = ? WHERE id = ?", (ab_test_id, exp.id))
    conn.commit()
    conn.close()

    print(f"\n[3/3] A/B test created: {ab_test_id}")
    print(f"Monitor progress at: {ORCHESTRATOR_URL}/abtest/status?test_id={ab_test_id}")

    return True


def list_experiments(status: str | None = None) -> list[dict]:
    """List all experiments."""
    conn = get_db_connection()

    if status:
        rows = conn.execute(
            "SELECT * FROM experiments WHERE status = ? ORDER BY created_at DESC",
            (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50"
        ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter A/B Testing")
    parser.add_argument("--experiment", "-e", help="Experiment name to run")
    parser.add_argument("--board-type", "-b", default="square8", help="Board type")
    parser.add_argument("--num-players", "-p", type=int, default=2, help="Number of players")
    parser.add_argument("--baseline-model", help="Path to baseline model")
    parser.add_argument("--list", action="store_true", help="List experiments")
    parser.add_argument("--status", help="Filter by status (pending, training, testing, completed, failed)")
    parser.add_argument("--results", action="store_true", help="Show experiment results")
    parser.add_argument("--recommendations", action="store_true", help="Show hyperparameter recommendations")
    parser.add_argument("--check", help="Check status of specific experiment")
    parser.add_argument("--auto", action="store_true", help="Auto-run next recommended experiment")

    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.status)
        if not experiments:
            print("No experiments found")
            return

        print(f"\nExperiments ({len(experiments)}):\n")
        for exp in experiments:
            status_icon = {
                "pending": "⏳",
                "training": "🔧",
                "testing": "🧪",
                "completed": "✅",
                "failed": "❌",
            }.get(exp["status"], "?")

            winner_str = f" -> {exp['winner']}" if exp["winner"] else ""
            print(f"  {status_icon} {exp['id']}: {exp['param_name']}={exp['variant_value']}{winner_str}")

    elif args.results:
        experiments = list_experiments("completed")
        if not experiments:
            print("No completed experiments")
            return

        print("\nCompleted Experiments:\n")
        for exp in experiments:
            print(f"  {exp['board_type']}_{exp['num_players']}p - {exp['param_name']}")
            print(f"    Baseline: {exp['baseline_value']}")
            print(f"    Variant:  {exp['variant_value']}")
            print(f"    Winner:   {exp['winner']} ({exp['confidence']:.1%} confidence, {exp['games_played']} games)")
            print()

    elif args.recommendations:
        recs = get_experiment_recommendations(args.board_type, args.num_players)
        print(f"\nRecommended Hyperparameters for {args.board_type}_{args.num_players}p:\n")
        for rec in recs:
            source_icon = {"default": "📋", "experiment": "🔬", "baseline_validated": "✅"}.get(rec["source"], "?")
            print(f"  {source_icon} {rec['param']}: {rec['recommendation']}")
            print(f"     Source: {rec['source']} ({rec['experiments_run']} experiments)")

    elif args.check:
        status = check_experiment_status(args.check)
        if not status:
            print(f"Experiment not found: {args.check}")
            return

        print(f"\nExperiment: {args.check}")
        print(f"Status: {status['status']}")

        if status.get("ab_test"):
            ab = status["ab_test"]
            print(f"\nA/B Test: {ab.get('test_id', 'N/A')}")
            print(f"  Status: {ab.get('status')}")
            if "stats" in ab:
                stats = ab["stats"]
                print(f"  Games: {stats.get('games_played', 0)}")
                print(f"  Model A win rate: {stats.get('model_a_winrate', 0):.1%}")
                print(f"  Model B win rate: {stats.get('model_b_winrate', 0):.1%}")
                print(f"  Confidence: {stats.get('confidence', 0):.1%}")

    elif args.auto:
        # Find next experiment to run
        for name in EXPERIMENT_DEFINITIONS:
            exp = create_experiment(name, args.board_type, args.num_players, args.baseline_model)
            if exp:
                print(f"\nRunning experiment: {name}")
                run_experiment(name, args.board_type, args.num_players, args.baseline_model)
                return

        print("No more experiments to run - all variants have been tested")

    elif args.experiment:
        run_experiment(args.experiment, args.board_type, args.num_players, args.baseline_model)

    else:
        print("Available experiments:")
        for name, defn in EXPERIMENT_DEFINITIONS.items():
            print(f"  {name}: {defn['description']}")
            print(f"    Baseline: {defn['baseline']}")
            print(f"    Variants: {defn['variants']}")
        print("\nUse --experiment <name> to run an experiment")


if __name__ == "__main__":
    main()
