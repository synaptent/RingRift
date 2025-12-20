#!/usr/bin/env python3
"""Database Health Check and Repair Utility.

This script checks SQLite databases for corruption and attempts repairs where possible.

Features:
- Integrity checks on all databases
- Schema validation
- Automatic repair attempts for recoverable issues
- Quarantine of corrupted databases
- Health reporting

Usage:
    # Check all databases
    python scripts/db_health_check.py

    # Check specific database
    python scripts/db_health_check.py --db data/games/selfplay.db

    # Attempt repairs
    python scripts/db_health_check.py --repair

    # Quarantine corrupted databases
    python scripts/db_health_check.py --quarantine
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.paths import AI_SERVICE_ROOT, DATA_DIR, QUARANTINE_DIR
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("db_health_check")

# Required tables for training databases
REQUIRED_TABLES = {
    "games",
    "game_moves",
    "game_players",
    "game_initial_state",
    "game_state_snapshots",
    "game_history_entries",
    "game_choices",
    "schema_metadata",
}


class DBHealthChecker:
    """Database health checker and repair utility."""

    def __init__(self, quarantine_dir: Path | None = None):
        self.quarantine_dir = quarantine_dir or QUARANTINE_DIR
        self.results: dict[str, dict[str, Any]] = {}

    def check_integrity(self, db_path: Path) -> tuple[bool, str]:
        """Run SQLite integrity check."""
        try:
            conn = sqlite3.connect(str(db_path))
            result = conn.execute("PRAGMA integrity_check").fetchone()
            conn.close()

            if result[0] == "ok":
                return True, "OK"
            else:
                return False, f"Integrity check failed: {result[0]}"
        except sqlite3.DatabaseError as e:
            return False, f"Database error: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def check_schema(self, db_path: Path) -> tuple[bool, list[str]]:
        """Check if database has required tables."""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            missing = REQUIRED_TABLES - tables
            if missing:
                return False, list(missing)
            return True, []
        except Exception as e:
            return False, [str(e)]

    def get_game_count(self, db_path: Path) -> int:
        """Get number of games in database."""
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            conn.close()
            return count
        except (sqlite3.Error, OSError):
            return -1

    def get_db_size(self, db_path: Path) -> int:
        """Get database file size in bytes."""
        return db_path.stat().st_size if db_path.exists() else 0

    def attempt_repair(self, db_path: Path) -> tuple[bool, str]:
        """Attempt to repair a corrupted database."""
        try:
            # Create backup first
            backup_path = db_path.with_suffix(".db.bak")
            shutil.copy2(db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")

            # Try vacuum to recover
            conn = sqlite3.connect(str(db_path))
            conn.execute("VACUUM")
            conn.close()

            # Re-check integrity
            healthy, _msg = self.check_integrity(db_path)
            if healthy:
                backup_path.unlink()  # Remove backup if successful
                return True, "Repaired via VACUUM"

            # Try dump and restore
            dump_path = db_path.with_suffix(".sql")
            try:
                import subprocess
                # Dump what we can (use list-form to avoid shell injection)
                with open(dump_path, "w") as dump_file:
                    subprocess.run(
                        ["sqlite3", str(db_path), ".dump"],
                        stdout=dump_file, check=True, timeout=300
                    )

                # Create new database
                new_db_path = db_path.with_suffix(".db.new")
                with open(dump_path) as dump_file:
                    subprocess.run(
                        ["sqlite3", str(new_db_path)],
                        stdin=dump_file, check=True, timeout=300
                    )

                # Replace original
                db_path.unlink()
                new_db_path.rename(db_path)
                dump_path.unlink()

                return True, "Repaired via dump/restore"
            except Exception as e:
                if dump_path.exists():
                    dump_path.unlink()
                return False, f"Repair failed: {e}"

        except Exception as e:
            return False, f"Repair failed: {e}"

    def quarantine_db(self, db_path: Path) -> Path:
        """Move corrupted database to quarantine."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_name = f"{db_path.stem}_{timestamp}{db_path.suffix}"
        quarantine_path = self.quarantine_dir / quarantine_name

        shutil.move(str(db_path), str(quarantine_path))
        logger.info(f"Quarantined: {db_path} -> {quarantine_path}")

        return quarantine_path

    def check_database(self, db_path: Path, repair: bool = False,
                       quarantine: bool = False) -> dict[str, Any]:
        """Perform full health check on a database."""
        result = {
            "path": str(db_path),
            "exists": db_path.exists(),
            "size_bytes": 0,
            "healthy": False,
            "integrity": None,
            "schema_valid": None,
            "missing_tables": [],
            "game_count": -1,
            "repaired": False,
            "quarantined": False,
            "message": "",
        }

        if not db_path.exists():
            result["message"] = "File not found"
            return result

        result["size_bytes"] = self.get_db_size(db_path)

        if result["size_bytes"] == 0:
            result["message"] = "Empty file"
            if quarantine:
                result["quarantined"] = True
                self.quarantine_db(db_path)
            return result

        # Check integrity
        integrity_ok, integrity_msg = self.check_integrity(db_path)
        result["integrity"] = integrity_msg

        if not integrity_ok:
            result["message"] = integrity_msg

            if repair:
                repaired, repair_msg = self.attempt_repair(db_path)
                result["repaired"] = repaired
                if repaired:
                    result["message"] = repair_msg
                    integrity_ok = True
                elif quarantine:
                    result["quarantined"] = True
                    self.quarantine_db(db_path)
                    return result

        if not integrity_ok:
            if quarantine:
                result["quarantined"] = True
                self.quarantine_db(db_path)
            return result

        # Check schema
        schema_ok, missing = self.check_schema(db_path)
        result["schema_valid"] = schema_ok
        result["missing_tables"] = missing

        # Get game count
        result["game_count"] = self.get_game_count(db_path)

        # Determine overall health
        result["healthy"] = integrity_ok and schema_ok and result["game_count"] >= 0

        if result["healthy"]:
            result["message"] = f"Healthy ({result['game_count']} games)"
        elif missing:
            result["message"] = f"Missing tables: {', '.join(missing)}"

        return result

    def check_all_databases(self, data_dir: Path | None = None,
                            repair: bool = False,
                            quarantine: bool = False) -> dict[str, dict[str, Any]]:
        """Check all databases in data directory."""
        if data_dir is None:
            data_dir = DATA_DIR

        results = {}

        # Find all .db files
        for db_path in data_dir.rglob("*.db"):
            # Skip WAL and journal files
            if "-wal" in db_path.name or "-shm" in db_path.name:
                continue

            # Skip quarantine directory
            if "quarantine" in str(db_path):
                continue

            logger.info(f"Checking: {db_path}")
            result = self.check_database(db_path, repair=repair, quarantine=quarantine)
            results[str(db_path)] = result

            status = "✓" if result["healthy"] else "✗"
            logger.info(f"  {status} {result['message']}")

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate a summary report."""
        if not self.results:
            return "No databases checked."

        lines = [
            "=" * 70,
            "DATABASE HEALTH CHECK REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
            "",
        ]

        healthy = sum(1 for r in self.results.values() if r["healthy"])
        total = len(self.results)

        lines.append(f"Summary: {healthy}/{total} databases healthy")
        lines.append("")

        # Group by health status
        unhealthy = [(p, r) for p, r in self.results.items() if not r["healthy"]]
        if unhealthy:
            lines.append("UNHEALTHY DATABASES:")
            for path, result in unhealthy:
                lines.append(f"  ✗ {path}")
                lines.append(f"    {result['message']}")
                if result["repaired"]:
                    lines.append("    (Repair attempted)")
                if result["quarantined"]:
                    lines.append("    (Quarantined)")
            lines.append("")

        # List healthy databases
        lines.append("HEALTHY DATABASES:")
        for path, result in self.results.items():
            if result["healthy"]:
                size_mb = result["size_bytes"] / (1024 * 1024)
                lines.append(f"  ✓ {path} ({result['game_count']} games, {size_mb:.1f} MB)")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Database Health Check Utility")
    parser.add_argument("--db", type=str, help="Check specific database")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair corrupted DBs")
    parser.add_argument("--quarantine", action="store_true", help="Quarantine corrupted DBs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--data-dir", type=str, help="Data directory to scan")
    args = parser.parse_args()

    checker = DBHealthChecker()

    if args.db:
        db_path = Path(args.db)
        if not db_path.is_absolute():
            db_path = AI_SERVICE_ROOT / db_path
        result = checker.check_database(db_path, repair=args.repair, quarantine=args.quarantine)
        checker.results[str(db_path)] = result
    else:
        data_dir = Path(args.data_dir) if args.data_dir else None
        checker.check_all_databases(data_dir, repair=args.repair, quarantine=args.quarantine)

    if args.json:
        print(json.dumps(checker.results, indent=2, default=str))
    else:
        print(checker.generate_report())


if __name__ == "__main__":
    main()
