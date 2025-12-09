from __future__ import annotations

"""
Scan all replay databases for canonical phase-history compliance and
optionally delete non-canonical DBs and models trained on them.

This is a thin orchestrator over scripts.check_canonical_phase_history
and a simple JSON model registry.

Model registry format (models/registry.json example):

{
  "models": [
    {
      "id": "ringrift_from_replays_square8_v1",
      "path": "models/ringrift_from_replays_square8.pth",
      "training_sources": [
        "data/games/canonical_square8.db",
        "data/games/selfplay_square8.db"
      ]
    }
  ]
}

Usage:

  PYTHONPATH=. python -m scripts.scan_canonical_phase_dbs \\
      --root data/games \\
      --pattern \"*.db\" \\
      --model-registry models/registry.json

To aggressively delete non-canonical DBs and models:

  PYTHONPATH=. python -m scripts.scan_canonical_phase_dbs \\
      --root data/games \\
      --pattern \"*.db\" \\
      --model-registry models/registry.json \\
      --delete-bad-dbs \\
      --delete-bad-models
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

from scripts.check_canonical_phase_history import check_db


def load_model_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"models": []}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalise to {"models": [...]} shape
    if isinstance(data, list):
        return {"models": data}
    if "models" not in data:
        data = {"models": data}
    return data


def scan_databases(
    root: Path,
    pattern: str,
    delete_bad_dbs: bool,
) -> Dict[str, Any]:
    db_paths = sorted(root.glob(pattern))
    report: Dict[str, Any] = {
        "root": str(root),
        "pattern": pattern,
        "databases": [],
        "bad_databases": [],
        "ok_databases": [],
    }
    bad_db_set: Set[str] = set()

    for db_path in db_paths:
        db_str = str(db_path)
        code = check_db(db_str)
        status = "ok" if code == 0 else "invalid"
        report["databases"].append({"path": db_str, "status": status})
        if status == "ok":
            report["ok_databases"].append(db_str)
        else:
            report["bad_databases"].append(db_str)
            bad_db_set.add(db_str)
            if delete_bad_dbs:
                try:
                    os.remove(db_str)
                except OSError as e:
                    # Surface the failure but continue scanning others.
                    report.setdefault("delete_errors", []).append({"path": db_str, "error": str(e)})

    report["bad_database_count"] = len(report["bad_databases"])
    report["ok_database_count"] = len(report["ok_databases"])
    return {"report": report, "bad_db_set": bad_db_set}


def scan_models(
    registry_path: Path,
    bad_db_set: Set[str],
    delete_bad_models: bool,
) -> Dict[str, Any]:
    registry = load_model_registry(registry_path)
    models = registry.get("models", [])

    bad_models: List[Dict[str, Any]] = []
    for entry in models:
        # Accept either "path" or "model_path"
        model_path = entry.get("path") or entry.get("model_path")
        if not model_path:
            continue
        sources = entry.get("training_sources", []) or []
        # Normalise DB paths to strings for comparison
        source_set = {str(s) for s in sources}
        if not source_set & bad_db_set:
            continue

        bad_entry: Dict[str, Any] = {
            "id": entry.get("id"),
            "path": model_path,
            "training_sources": list(source_set),
        }
        bad_models.append(bad_entry)

        if delete_bad_models:
            try:
                os.remove(model_path)
                bad_entry["deleted"] = True
            except OSError as e:
                bad_entry["deleted"] = False
                bad_entry["delete_error"] = str(e)

    return {
        "registry_path": str(registry_path),
        "bad_models": bad_models,
        "bad_model_count": len(bad_models),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan replay DBs for canonical phase-history compliance and optionally delete bad DBs/models."
    )
    parser.add_argument(
        "--root",
        default="data/games",
        help="Root directory to search for .db files (default: data/games)",
    )
    parser.add_argument(
        "--pattern",
        default="*.db",
        help='Glob pattern for DB files under root (default: "*.db")',
    )
    parser.add_argument(
        "--model-registry",
        default=None,
        help="Optional path to model registry JSON for cross-checking models",
    )
    parser.add_argument(
        "--delete-bad-dbs",
        action="store_true",
        help="If set, delete databases that fail the canonical phase-history check",
    )
    parser.add_argument(
        "--delete-bad-models",
        action="store_true",
        help="If set, delete models whose training_sources include any bad DB",
    )

    args = parser.parse_args(argv)

    root = Path(args.root)
    db_result = scan_databases(root, args.pattern, delete_bad_dbs=args.delete_bad_dbs)
    bad_db_set: Set[str] = db_result["bad_db_set"]

    model_report: Dict[str, Any] = {}
    if args.model_registry:
        registry_path = Path(args.model_registry)
        model_report = scan_models(
            registry_path,
            bad_db_set,
            delete_bad_models=args.delete_bad_models,
        )

    full_report = {
        "databases": db_result["report"],
        "models": model_report,
    }
    print(json.dumps(full_report, indent=2))

    # Non-zero exit if any bad DBs or bad models were found
    if db_result["report"]["bad_database_count"] > 0:
        return 1
    if model_report.get("bad_model_count", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
