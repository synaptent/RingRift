#!/usr/bin/env python3
"""Model lineage and version tracking for RingRift AI.

Tracks model ancestry, training history, and performance metrics to enable:
- Understanding model evolution over time
- Reproducing training results
- Debugging performance regressions
- Identifying best training configurations

Usage:
    python scripts/model_lineage.py --register model.pth --parent parent.pth
    python scripts/model_lineage.py --tree models/ringrift_best_sq8_2p.pth
    python scripts/model_lineage.py --export-report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
LINEAGE_DB_PATH = AI_SERVICE_ROOT / "data" / "model_lineage.db"


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    model_path: str
    model_hash: str  # SHA256 of model file
    board_type: str
    num_players: int
    parent_id: Optional[str] = None
    architecture: str = "unknown"
    created_at: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    training_data: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]  # Short hash for readability


def init_lineage_db() -> None:
    """Initialize the lineage database."""
    LINEAGE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LINEAGE_DB_PATH))

    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            model_path TEXT NOT NULL,
            model_hash TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            parent_id TEXT,
            architecture TEXT DEFAULT 'unknown',
            created_at TEXT NOT NULL,
            training_config TEXT,
            training_data TEXT,
            performance TEXT,
            tags TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS performance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            recorded_at TEXT NOT NULL,
            context TEXT,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT DEFAULT 'running',
            config TEXT,
            metrics TEXT,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_board
        ON models(board_type, num_players)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_models_parent
        ON models(parent_id)
    """)

    conn.commit()
    conn.close()


def register_model(
    model_path: str,
    board_type: str,
    num_players: int,
    parent_id: Optional[str] = None,
    architecture: str = "unknown",
    training_config: Optional[Dict] = None,
    training_data: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Register a model in the lineage database.

    Returns:
        The model_id of the registered model
    """
    init_lineage_db()

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Generate model ID from path and hash
    model_hash = compute_file_hash(path)
    model_id = f"{path.stem}_{model_hash}"

    # Check if already registered
    conn = sqlite3.connect(str(LINEAGE_DB_PATH))
    cursor = conn.execute(
        "SELECT model_id FROM models WHERE model_hash = ?",
        (model_hash,)
    )
    existing = cursor.fetchone()

    if existing:
        conn.close()
        return existing[0]

    # Register new model
    now = datetime.utcnow().isoformat() + "Z"

    conn.execute("""
        INSERT INTO models
        (model_id, model_path, model_hash, board_type, num_players, parent_id,
         architecture, created_at, training_config, training_data, performance, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_id,
        str(path.absolute()),
        model_hash,
        board_type,
        num_players,
        parent_id,
        architecture,
        now,
        json.dumps(training_config or {}),
        json.dumps(training_data or {}),
        json.dumps({}),
        json.dumps(tags or []),
    ))

    conn.commit()
    conn.close()

    return model_id


def update_performance(
    model_id: str,
    metric_name: str,
    metric_value: float,
    context: Optional[str] = None,
) -> None:
    """Record a performance metric for a model."""
    init_lineage_db()

    now = datetime.utcnow().isoformat() + "Z"
    conn = sqlite3.connect(str(LINEAGE_DB_PATH))

    conn.execute("""
        INSERT INTO performance_history
        (model_id, metric_name, metric_value, recorded_at, context)
        VALUES (?, ?, ?, ?, ?)
    """, (model_id, metric_name, metric_value, now, context))

    # Also update the models table with latest performance
    cursor = conn.execute(
        "SELECT performance FROM models WHERE model_id = ?",
        (model_id,)
    )
    row = cursor.fetchone()
    if row:
        perf = json.loads(row[0] or "{}")
        perf[metric_name] = metric_value
        conn.execute(
            "UPDATE models SET performance = ? WHERE model_id = ?",
            (json.dumps(perf), model_id)
        )

    conn.commit()
    conn.close()


def get_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    """Get metadata for a model."""
    if not LINEAGE_DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(LINEAGE_DB_PATH))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        "SELECT * FROM models WHERE model_id = ?",
        (model_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return ModelMetadata(
        model_id=row["model_id"],
        model_path=row["model_path"],
        model_hash=row["model_hash"],
        board_type=row["board_type"],
        num_players=row["num_players"],
        parent_id=row["parent_id"],
        architecture=row["architecture"],
        created_at=row["created_at"],
        training_config=json.loads(row["training_config"] or "{}"),
        training_data=json.loads(row["training_data"] or "{}"),
        performance=json.loads(row["performance"] or "{}"),
        tags=json.loads(row["tags"] or "[]"),
    )


def get_model_ancestry(model_id: str, max_depth: int = 10) -> List[str]:
    """Get the ancestry chain of a model (parent, grandparent, etc.)."""
    if not LINEAGE_DB_PATH.exists():
        return []

    ancestry = []
    current_id = model_id

    conn = sqlite3.connect(str(LINEAGE_DB_PATH))

    for _ in range(max_depth):
        cursor = conn.execute(
            "SELECT parent_id FROM models WHERE model_id = ?",
            (current_id,)
        )
        row = cursor.fetchone()

        if not row or not row[0]:
            break

        current_id = row[0]
        ancestry.append(current_id)

    conn.close()
    return ancestry


def get_model_descendants(model_id: str) -> List[str]:
    """Get all models that descend from this model."""
    if not LINEAGE_DB_PATH.exists():
        return []

    conn = sqlite3.connect(str(LINEAGE_DB_PATH))

    descendants = []
    queue = [model_id]

    while queue:
        current = queue.pop(0)
        cursor = conn.execute(
            "SELECT model_id FROM models WHERE parent_id = ?",
            (current,)
        )
        for row in cursor:
            descendants.append(row[0])
            queue.append(row[0])

    conn.close()
    return descendants


def print_lineage_tree(model_id: str) -> None:
    """Print a visual tree of model lineage."""
    ancestry = get_model_ancestry(model_id)
    descendants = get_model_descendants(model_id)

    print(f"\nLineage tree for {model_id}:")
    print("=" * 60)

    # Print ancestors
    if ancestry:
        print("\nAncestors (oldest first):")
        for i, ancestor in enumerate(reversed(ancestry)):
            meta = get_model_metadata(ancestor)
            indent = "  " * i
            elo = meta.performance.get("elo", "?") if meta else "?"
            print(f"{indent}└── {ancestor[:40]} (Elo: {elo})")

    # Print current model
    meta = get_model_metadata(model_id)
    elo = meta.performance.get("elo", "?") if meta else "?"
    indent = "  " * len(ancestry)
    print(f"{indent}└── [{model_id[:40]}] (Elo: {elo}) <-- current")

    # Print descendants
    if descendants:
        print("\nDescendants:")
        for desc in descendants:
            meta = get_model_metadata(desc)
            elo = meta.performance.get("elo", "?") if meta else "?"
            print(f"    └── {desc[:40]} (Elo: {elo})")

    print()


def export_lineage_report(output_path: Optional[str] = None) -> Dict:
    """Export a comprehensive lineage report."""
    if not LINEAGE_DB_PATH.exists():
        return {"models": [], "stats": {}}

    conn = sqlite3.connect(str(LINEAGE_DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get all models
    cursor = conn.execute("SELECT * FROM models ORDER BY created_at DESC")
    models = []

    for row in cursor:
        models.append({
            "model_id": row["model_id"],
            "board_type": row["board_type"],
            "num_players": row["num_players"],
            "parent_id": row["parent_id"],
            "architecture": row["architecture"],
            "created_at": row["created_at"],
            "performance": json.loads(row["performance"] or "{}"),
            "tags": json.loads(row["tags"] or "[]"),
        })

    # Compute stats
    stats = {
        "total_models": len(models),
        "by_board_type": {},
        "generation_depth": {},
    }

    for model in models:
        bt = f"{model['board_type']}_{model['num_players']}p"
        stats["by_board_type"][bt] = stats["by_board_type"].get(bt, 0) + 1

        depth = len(get_model_ancestry(model["model_id"]))
        stats["generation_depth"][depth] = stats["generation_depth"].get(depth, 0) + 1

    conn.close()

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "models": models,
        "stats": stats,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report exported to: {output_path}")

    return report


def find_by_hash(model_hash: str) -> Optional[str]:
    """Find a model by its file hash."""
    if not LINEAGE_DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(LINEAGE_DB_PATH))
    cursor = conn.execute(
        "SELECT model_id FROM models WHERE model_hash = ?",
        (model_hash,)
    )
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None


def main():
    parser = argparse.ArgumentParser(description="Model lineage tracking")
    parser.add_argument("--register", help="Register a model file")
    parser.add_argument("--parent", help="Parent model ID for registration")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--tree", help="Show lineage tree for model")
    parser.add_argument("--export-report", action="store_true", help="Export lineage report")
    parser.add_argument("--output", help="Output path for report")
    parser.add_argument("--list", action="store_true", help="List all models")

    args = parser.parse_args()

    init_lineage_db()

    if args.register:
        model_id = register_model(
            args.register,
            args.board,
            args.players,
            parent_id=args.parent,
        )
        print(f"Registered model: {model_id}")

    elif args.tree:
        # Try to find model by path or ID
        if Path(args.tree).exists():
            model_hash = compute_file_hash(Path(args.tree))
            model_id = find_by_hash(model_hash)
            if not model_id:
                print(f"Model not found in lineage database. Register it first with --register")
                return 1
        else:
            model_id = args.tree

        print_lineage_tree(model_id)

    elif args.export_report:
        output = args.output or str(AI_SERVICE_ROOT / "data" / "lineage_report.json")
        report = export_lineage_report(output)
        print(f"Total models tracked: {report['stats']['total_models']}")

    elif args.list:
        if not LINEAGE_DB_PATH.exists():
            print("No models tracked yet")
            return 0

        conn = sqlite3.connect(str(LINEAGE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT model_id, board_type, num_players, created_at, performance "
            "FROM models ORDER BY created_at DESC LIMIT 20"
        )

        print("\nRecent models:")
        print("-" * 80)
        for row in cursor:
            perf = json.loads(row["performance"] or "{}")
            elo = perf.get("elo", "?")
            print(f"{row['model_id'][:40]:<40} {row['board_type']:<10} "
                  f"{row['num_players']}p  Elo:{elo}")

        conn.close()

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
