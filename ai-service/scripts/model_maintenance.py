#!/usr/bin/env python3
"""Model lifecycle maintenance CLI.

Run periodic model cleanup based on retention policies:
- Archive low-performing models (bottom 75% by Elo)
- Delete archived models older than 90 days
- Maintain per-config model counts under threshold

Usage:
    # Run full maintenance
    python scripts/model_maintenance.py

    # Check specific config
    python scripts/model_maintenance.py --config square8_2p

    # Force run (ignore cooldown)
    python scripts/model_maintenance.py --force

    # Show status only
    python scripts/model_maintenance.py --status

    # Dry run (no changes)
    python scripts/model_maintenance.py --dry-run

Cron example (run hourly):
    0 * * * * cd /path/to/ai-service && python scripts/model_maintenance.py >> logs/maintenance.log 2>&1
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("model_maintenance")


def main():
    parser = argparse.ArgumentParser(
        description="Model lifecycle maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Specific config to maintain (e.g., square8_2p). Default: all configs",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force maintenance even within cooldown period",
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show status only, don't run maintenance",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Override model directory (default: models/)",
    )
    parser.add_argument(
        "--elo-db",
        type=str,
        help="Override Elo database path (default: data/unified_elo.db)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=100,
        help="Max models per config before culling (default: 100)",
    )
    parser.add_argument(
        "--keep-top",
        type=int,
        default=25,
        help="Number of top models by Elo to keep (default: 25)",
    )
    parser.add_argument(
        "--archive-days",
        type=int,
        default=30,
        help="Archive models older than N days (default: 30)",
    )
    parser.add_argument(
        "--delete-days",
        type=int,
        default=90,
        help="Delete archived models older than N days (default: 90)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import after path setup
    from app.training.model_lifecycle import (
        ModelLifecycleManager,
        RetentionPolicy,
    )

    # Build policy from args
    policy = RetentionPolicy(
        max_models_per_config=args.max_models,
        keep_top_by_elo=args.keep_top,
        archive_after_days=args.archive_days,
        delete_archived_after_days=args.delete_days,
    )

    # Initialize manager
    manager = ModelLifecycleManager(
        model_dir=Path(args.model_dir) if args.model_dir else None,
        elo_db_path=Path(args.elo_db) if args.elo_db else None,
        policy=policy,
    )

    # Status only
    if args.status:
        status = manager.get_status()
        if args.json:
            print(json.dumps(status, indent=2, default=str))
        else:
            print("\n=== Model Lifecycle Status ===\n")
            for config, info in sorted(status.items()):
                print(f"{config}:")
                print(f"  Active models: {info['active_models']}")
                print(f"  Archived models: {info['archived_models']}")
                print(f"  Needs culling: {'Yes' if info['needs_culling'] else 'No'}")
                if info['last_maintenance']:
                    print(f"  Last maintenance: {info['last_maintenance']}")
                print()
        return 0

    # Dry run
    if args.dry_run:
        print("\n=== Dry Run (no changes will be made) ===\n")
        configs = [args.config] if args.config else None

        from app.utils.canonical_naming import CANONICAL_CONFIG_KEYS
        configs_to_check = configs or CANONICAL_CONFIG_KEYS

        for config_key in configs_to_check:
            models = manager.get_models_for_config(config_key)
            archived = manager.get_archived_models(config_key)

            print(f"{config_key}:")
            print(f"  Active: {len(models)}, Archived: {len(archived)}")

            if len(models) > policy.max_models_per_config:
                to_cull = len(models) - policy.keep_top_by_elo
                print(f"  Would archive: ~{to_cull} models (keeping top {policy.keep_top_by_elo})")

            # Check old archives
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=policy.delete_archived_after_days)
            old_archives = 0
            for p in archived:
                try:
                    mtime = datetime.fromtimestamp(p.stat().st_mtime)
                    if mtime < cutoff:
                        old_archives += 1
                except OSError:
                    pass

            if old_archives:
                print(f"  Would delete: {old_archives} archived models (>{policy.delete_archived_after_days} days old)")
            print()

        return 0

    # Run maintenance
    logger.info(f"Starting model maintenance at {datetime.now().isoformat()}")

    if args.config:
        # Single config
        result = manager.check_config(args.config, force=args.force)
        results = {args.config: result}
        total_archived = result.archived
        total_deleted = result.deleted
        total_errors = len(result.errors)
    else:
        # All configs
        full_result = manager.run_maintenance(force=args.force)
        results = full_result.per_config_results
        total_archived = full_result.total_archived
        total_deleted = full_result.total_deleted
        total_errors = full_result.total_errors

    # Output results
    if args.json:
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_archived": total_archived,
            "total_deleted": total_deleted,
            "total_errors": total_errors,
            "per_config": {
                k: {
                    "models_before": v.models_before,
                    "models_after": v.models_after,
                    "archived": v.archived,
                    "deleted": v.deleted,
                    "errors": v.errors,
                }
                for k, v in results.items()
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print("\n=== Maintenance Results ===\n")
        for config, result in sorted(results.items()):
            if result.archived > 0 or result.deleted > 0 or result.errors:
                print(f"{config}:")
                print(f"  Models: {result.models_before} -> {result.models_after}")
                if result.archived:
                    print(f"  Archived: {result.archived}")
                if result.deleted:
                    print(f"  Deleted: {result.deleted}")
                if result.errors:
                    print(f"  Errors: {result.errors}")
                print()

        print(f"Total: archived={total_archived}, deleted={total_deleted}, errors={total_errors}")

    logger.info(
        f"Maintenance complete: archived={total_archived}, deleted={total_deleted}, errors={total_errors}"
    )

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
