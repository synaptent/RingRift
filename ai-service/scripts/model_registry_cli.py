#!/usr/bin/env python3
"""CLI tool for managing the RingRift model registry.

Provides commands to list, inspect, promote, archive, rollback,
and export models from the registry.

Usage:
    # List all models
    python scripts/model_registry_cli.py list

    # List models by stage
    python scripts/model_registry_cli.py list --stage production

    # List heuristic models only
    python scripts/model_registry_cli.py list --type heuristic

    # Show model details
    python scripts/model_registry_cli.py show square8_2p_v1

    # Promote a model
    python scripts/model_registry_cli.py promote square8_2p --version 3 --to staging

    # Archive a model
    python scripts/model_registry_cli.py archive square8_2p --version 2

    # Rollback to previous production model
    python scripts/model_registry_cli.py rollback square8_2p

    # Export a model
    python scripts/model_registry_cli.py export square8_2p --version 1 --output /tmp/model.pt

    # Show stage history
    python scripts/model_registry_cli.py history square8_2p --version 1

    # Compare two models
    python scripts/model_registry_cli.py compare square8_2p:1 square8_2p:2
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training.model_registry import (
    ModelRegistry,
    ModelStage,
    ModelType,
)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_DIR = AI_SERVICE_ROOT / "data" / "model_registry"


def get_registry(registry_dir: Path | None = None) -> ModelRegistry:
    """Get or create the model registry."""
    registry_dir = registry_dir or DEFAULT_REGISTRY_DIR
    return ModelRegistry(registry_dir)


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16] if len(ts) > 16 else ts


def format_metrics(metrics: dict) -> str:
    """Format metrics dict for display."""
    parts = []
    if metrics.get("elo"):
        parts.append(f"Elo: {metrics['elo']:.0f}")
    if metrics.get("win_rate"):
        parts.append(f"WR: {metrics['win_rate']:.1%}")
    if metrics.get("games_played"):
        parts.append(f"Games: {metrics['games_played']}")
    return ", ".join(parts) if parts else "No metrics"


def cmd_list(args):
    """List models in the registry."""
    registry = get_registry(args.registry_dir)

    # Apply filters
    stage_filter = ModelStage(args.stage) if args.stage else None
    type_filter = ModelType(args.type) if args.type else None

    models = registry.list_models(stage=stage_filter, model_type=type_filter)

    if not models:
        print("No models found.")
        return

    # Group by model_id for cleaner display
    if args.group:
        grouped = {}
        for m in models:
            mid = m["model_id"]
            if mid not in grouped:
                grouped[mid] = []
            grouped[mid].append(m)

        print(f"{'Model ID':<30} {'Versions':<10} {'Latest Stage':<12} {'Type':<15}")
        print("-" * 70)
        for mid, versions in sorted(grouped.items()):
            latest = max(versions, key=lambda x: x["version"])
            print(f"{mid:<30} {len(versions):<10} {latest['stage']:<12} {latest['model_type']:<15}")
    else:
        print(f"{'Model ID':<25} {'Ver':<5} {'Stage':<12} {'Type':<12} {'Metrics':<25} {'Updated':<16}")
        print("-" * 100)
        for m in models:
            metrics_str = format_metrics(m.get("metrics", {}))
            updated = format_timestamp(m.get("updated_at", ""))
            print(f"{m['model_id']:<25} v{m['version']:<4} {m['stage']:<12} {m['model_type']:<12} {metrics_str:<25} {updated:<16}")

    print(f"\nTotal: {len(models)} model version(s)")


def cmd_show(args):
    """Show detailed information about a model."""
    registry = get_registry(args.registry_dir)

    # Parse model_id:version format
    if ":" in args.model_id:
        model_id, version_str = args.model_id.rsplit(":", 1)
        version = int(version_str)
    else:
        model_id = args.model_id
        version = args.version

    model = registry.get_model(model_id, version)
    if not model:
        print(f"Model not found: {model_id}" + (f":v{version}" if version else ""))
        return 1

    print(f"\n{'='*60}")
    print(f"Model: {model.model_id}:v{model.version}")
    print(f"{'='*60}")
    print(f"Name:        {model.name}")
    print(f"Type:        {model.model_type.value}")
    print(f"Stage:       {model.stage.value}")
    print(f"Description: {model.description or 'N/A'}")
    print(f"File:        {model.file_path}")
    print(f"File Hash:   {model.file_hash[:16]}...")
    print(f"File Size:   {model.file_size_bytes / 1024:.1f} KB")
    print(f"Created:     {model.created_at}")
    print(f"Updated:     {model.updated_at}")

    print(f"\nTags: {', '.join(model.tags) if model.tags else 'None'}")

    print("\nMetrics:")
    metrics = model.metrics
    print(f"  Elo:            {metrics.elo or 'N/A'}")
    print(f"  Elo Uncertainty:{metrics.elo_uncertainty or 'N/A'}")
    print(f"  Win Rate:       {metrics.win_rate:.1%}" if metrics.win_rate else "  Win Rate:       N/A")
    print(f"  Draw Rate:      {metrics.draw_rate:.1%}" if metrics.draw_rate else "  Draw Rate:      N/A")
    print(f"  Games Played:   {metrics.games_played}")

    print("\nTraining Config:")
    config = model.training_config
    print(f"  Optimizer:      {config.optimizer}")
    print(f"  Learning Rate:  {config.learning_rate}")
    print(f"  Batch Size:     {config.batch_size}")
    print(f"  Epochs:         {config.epochs}")
    if config.extra_config:
        print("  Extra Config:")
        for k, v in config.extra_config.items():
            print(f"    {k}: {v}")

    return 0


def cmd_promote(args):
    """Promote a model to a new stage."""
    registry = get_registry(args.registry_dir)

    model_id = args.model_id
    version = args.version
    to_stage = ModelStage(args.to)

    # Get current model to check
    model = registry.get_model(model_id, version)
    if not model:
        print(f"Model not found: {model_id}:v{version}")
        return 1

    current_stage = model.stage
    print(f"Promoting {model_id}:v{version} from {current_stage.value} to {to_stage.value}")

    if not args.yes:
        confirm = input("Confirm? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        registry.promote(
            model_id,
            version,
            to_stage,
            reason=args.reason or "Manual promotion via CLI",
            promoted_by=os.environ.get("USER", "cli"),
        )
        print(f"Successfully promoted {model_id}:v{version} to {to_stage.value}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_archive(args):
    """Archive a model."""
    registry = get_registry(args.registry_dir)

    model_id = args.model_id
    version = args.version

    model = registry.get_model(model_id, version)
    if not model:
        print(f"Model not found: {model_id}:v{version}")
        return 1

    if model.stage == ModelStage.ARCHIVED:
        print(f"Model {model_id}:v{version} is already archived.")
        return 0

    print(f"Archiving {model_id}:v{version} (current stage: {model.stage.value})")

    if not args.yes:
        confirm = input("Confirm? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        registry.promote(
            model_id,
            version,
            ModelStage.ARCHIVED,
            reason=args.reason or "Manual archive via CLI",
            promoted_by=os.environ.get("USER", "cli"),
        )
        print(f"Successfully archived {model_id}:v{version}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_rollback(args):
    """Rollback to a previous model version."""
    registry = get_registry(args.registry_dir)

    model_id = args.model_id

    # Get current production model
    current_prod = registry.get_production_model()
    if not current_prod or current_prod.model_id != model_id:
        print(f"No production model found for {model_id}")

    # Find the most recent archived model to rollback to
    archived_models = registry.list_models(stage=ModelStage.ARCHIVED)
    candidates = [m for m in archived_models if m["model_id"] == model_id]

    if not candidates:
        print(f"No archived versions found for {model_id} to rollback to.")
        return 1

    # Sort by version descending
    candidates.sort(key=lambda x: x["version"], reverse=True)
    target = candidates[0]

    if args.version:
        # Specific version requested
        target = next((c for c in candidates if c["version"] == args.version), None)
        if not target:
            print(f"Version {args.version} not found in archived models.")
            return 1

    print("\nRollback Plan:")
    if current_prod:
        print(f"  Current Production: {current_prod.model_id}:v{current_prod.version}")
        print(f"  Will Archive:       {current_prod.model_id}:v{current_prod.version}")
    print(f"  Will Restore:       {target['model_id']}:v{target['version']}")
    print(f"  Target Metrics:     {format_metrics(target.get('metrics', {}))}")

    if not args.yes:
        confirm = input("\nProceed with rollback? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        # First move archived model to staging
        registry.promote(
            target["model_id"],
            target["version"],
            ModelStage.DEVELOPMENT,
            reason="Rollback: restoring from archive",
            promoted_by=os.environ.get("USER", "cli"),
        )
        registry.promote(
            target["model_id"],
            target["version"],
            ModelStage.STAGING,
            reason="Rollback: moving to staging",
            promoted_by=os.environ.get("USER", "cli"),
        )
        # Then promote to production (this will archive current production)
        registry.promote(
            target["model_id"],
            target["version"],
            ModelStage.PRODUCTION,
            reason=args.reason or "Rollback via CLI",
            promoted_by=os.environ.get("USER", "cli"),
        )
        print(f"\nSuccessfully rolled back to {target['model_id']}:v{target['version']}")
    except ValueError as e:
        print(f"Error during rollback: {e}")
        return 1

    return 0


def cmd_export(args):
    """Export a model to a file."""
    registry = get_registry(args.registry_dir)

    model_id = args.model_id
    version = args.version
    output_path = Path(args.output)

    model = registry.get_model(model_id, version)
    if not model:
        print(f"Model not found: {model_id}" + (f":v{version}" if version else ""))
        return 1

    try:
        exported_path = registry.export_model(model_id, model.version, output_path)
        print(f"Exported {model_id}:v{model.version} to {exported_path}")
        print(f"Metadata saved to {output_path.with_suffix('.json')}")
    except Exception as e:
        print(f"Error exporting model: {e}")
        return 1

    return 0


def cmd_history(args):
    """Show stage transition history for a model."""
    registry = get_registry(args.registry_dir)

    model_id = args.model_id
    version = args.version

    model = registry.get_model(model_id, version)
    if not model:
        print(f"Model not found: {model_id}" + (f":v{version}" if version else ""))
        return 1

    history = registry.get_stage_history(model_id, model.version)

    if not history:
        print(f"No stage history found for {model_id}:v{model.version}")
        return 0

    print(f"\nStage History for {model_id}:v{model.version}")
    print("-" * 80)
    print(f"{'From':<12} {'To':<12} {'Reason':<30} {'By':<12} {'When':<16}")
    print("-" * 80)

    for h in history:
        from_stage = h.get("from_stage") or "(new)"
        to_stage = h.get("to_stage", "")
        reason = (h.get("reason") or "")[:30]
        by = (h.get("transitioned_by") or "system")[:12]
        when = format_timestamp(h.get("transitioned_at", ""))
        print(f"{from_stage:<12} {to_stage:<12} {reason:<30} {by:<12} {when:<16}")

    return 0


def cmd_compare(args):
    """Compare two models."""
    registry = get_registry(args.registry_dir)

    # Parse model_a and model_b (format: model_id:version)
    def parse_model_ref(ref: str):
        if ":" in ref:
            mid, ver = ref.rsplit(":", 1)
            return mid, int(ver)
        return ref, None

    model_a_id, model_a_ver = parse_model_ref(args.model_a)
    model_b_id, model_b_ver = parse_model_ref(args.model_b)

    model_a = registry.get_model(model_a_id, model_a_ver)
    model_b = registry.get_model(model_b_id, model_b_ver)

    if not model_a:
        print(f"Model A not found: {args.model_a}")
        return 1
    if not model_b:
        print(f"Model B not found: {args.model_b}")
        return 1

    print(f"\n{'Comparison':<20} {'Model A':<25} {'Model B':<25}")
    print("-" * 70)
    print(f"{'ID':<20} {model_a.model_id}:v{model_a.version:<18} {model_b.model_id}:v{model_b.version:<18}")
    print(f"{'Stage':<20} {model_a.stage.value:<25} {model_b.stage.value:<25}")
    print(f"{'Type':<20} {model_a.model_type.value:<25} {model_b.model_type.value:<25}")

    # Metrics comparison
    ma, mb = model_a.metrics, model_b.metrics
    print(f"\n{'Metrics':<20}")
    print(f"{'  Elo':<20} {ma.elo or 'N/A'!s:<25} {mb.elo or 'N/A'!s:<25}")

    if ma.elo and mb.elo:
        diff = mb.elo - ma.elo
        print(f"{'  Elo Diff':<20} {'':<25} {diff:+.1f}")

    print(f"{'  Win Rate':<20} {f'{ma.win_rate:.1%}' if ma.win_rate else 'N/A':<25} {f'{mb.win_rate:.1%}' if mb.win_rate else 'N/A':<25}")
    print(f"{'  Games Played':<20} {ma.games_played:<25} {mb.games_played:<25}")

    # Training config comparison
    ca, cb = model_a.training_config, model_b.training_config
    print(f"\n{'Training Config':<20}")
    print(f"{'  Optimizer':<20} {ca.optimizer:<25} {cb.optimizer:<25}")
    print(f"{'  Learning Rate':<20} {ca.learning_rate:<25} {cb.learning_rate:<25}")
    print(f"{'  Batch Size':<20} {ca.batch_size:<25} {cb.batch_size:<25}")
    print(f"{'  Epochs':<20} {ca.epochs:<25} {cb.epochs:<25}")

    return 0


def cmd_dashboard(args):
    """Show a rich dashboard of the model registry state."""
    registry = get_registry(args.registry_dir)

    # Header
    print("\n" + "=" * 60)
    print("        RingRift Model Registry Dashboard")
    print("=" * 60)

    # Production models
    prod_models = registry.list_models(stage=ModelStage.PRODUCTION)
    print(f"\n{'PRODUCTION':<15} ({len(prod_models)} models)")
    print("-" * 50)
    if prod_models:
        for m in prod_models:
            metrics = m.get("metrics", {})
            elo = metrics.get("elo", "N/A")
            elo_str = f"{elo:.0f}" if isinstance(elo, (int, float)) else elo
            config = m.get("config", {})
            board = config.get("board_type", "?")
            players = config.get("num_players", "?")
            print(f"  {m['model_id']:<25} Elo: {elo_str:<6} ({board}_{players}p)")
    else:
        print("  No production models")

    # Staging models
    staging_models = registry.list_models(stage=ModelStage.STAGING)
    print(f"\n{'STAGING':<15} ({len(staging_models)} models)")
    print("-" * 50)
    if staging_models:
        for m in staging_models[:5]:  # Show top 5
            metrics = m.get("metrics", {})
            games = metrics.get("games_played", 0)
            print(f"  {m['model_id']:<25} Games: {games}")
        if len(staging_models) > 5:
            print(f"  ... and {len(staging_models) - 5} more")
    else:
        print("  No staging models")

    # Development models (count only)
    dev_models = registry.list_models(stage=ModelStage.DEVELOPMENT)
    archived_models = registry.list_models(stage=ModelStage.ARCHIVED)

    print(f"\n{'DEVELOPMENT':<15} {len(dev_models)} models awaiting evaluation")
    print(f"{'ARCHIVED':<15} {len(archived_models)} models")

    # Total count
    total = len(prod_models) + len(staging_models) + len(dev_models) + len(archived_models)
    print(f"\n{'TOTAL':<15} {total} model versions in registry")

    # Recent activity (if available)
    print("\n" + "=" * 60)

    return 0


def cmd_cleanup(args):
    """Run the archival policy to clean up old models."""
    import subprocess

    daemon_script = Path(__file__).parent / "model_archival_daemon.py"

    if not daemon_script.exists():
        print(f"Error: Archival daemon not found at {daemon_script}")
        return 1

    cmd = ["python", str(daemon_script)]

    if args.execute:
        cmd.append("--execute")
    else:
        cmd.append("--dry-run")

    if args.max_age:
        cmd.extend(["--max-age-days", str(args.max_age)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def cmd_stats(args):
    """Show registry statistics."""
    registry = get_registry(args.registry_dir)

    # Count by stage
    stages = {}
    for stage in ModelStage:
        models = registry.list_models(stage=stage)
        stages[stage.value] = len(models)

    # Count by type
    types = {}
    for model_type in ModelType:
        models = registry.list_models(model_type=model_type)
        types[model_type.value] = len(models)

    total = sum(stages.values())

    print("\nModel Registry Statistics")
    print("=" * 40)
    print("\nBy Stage:")
    for stage, count in stages.items():
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"  {stage:<12} {count:>5} ({pct:>5.1f}%) {bar}")

    print("\nBy Type:")
    for mtype, count in types.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {mtype:<15} {count:>5} ({pct:>5.1f}%)")

    print(f"\nTotal: {total} model versions")

    # Show production models
    print("\nProduction Models:")
    prod_models = registry.list_models(stage=ModelStage.PRODUCTION)
    if prod_models:
        for m in prod_models:
            metrics_str = format_metrics(m.get("metrics", {}))
            print(f"  {m['model_id']}:v{m['version']} - {metrics_str}")
    else:
        print("  None")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="RingRift Model Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Path to registry directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list command
    list_parser = subparsers.add_parser("list", help="List models")
    list_parser.add_argument("--stage", choices=[s.value for s in ModelStage], help="Filter by stage")
    list_parser.add_argument("--type", choices=[t.value for t in ModelType], help="Filter by type")
    list_parser.add_argument("--group", action="store_true", help="Group by model ID")

    # show command
    show_parser = subparsers.add_parser("show", help="Show model details")
    show_parser.add_argument("model_id", help="Model ID (or model_id:version)")
    show_parser.add_argument("--version", "-v", type=int, help="Version number")

    # promote command
    promote_parser = subparsers.add_parser("promote", help="Promote a model")
    promote_parser.add_argument("model_id", help="Model ID")
    promote_parser.add_argument("--version", "-v", type=int, required=True, help="Version number")
    promote_parser.add_argument("--to", required=True, choices=[s.value for s in ModelStage], help="Target stage")
    promote_parser.add_argument("--reason", "-r", help="Promotion reason")
    promote_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a model")
    archive_parser.add_argument("model_id", help="Model ID")
    archive_parser.add_argument("--version", "-v", type=int, required=True, help="Version number")
    archive_parser.add_argument("--reason", "-r", help="Archive reason")
    archive_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback to previous version")
    rollback_parser.add_argument("model_id", help="Model ID")
    rollback_parser.add_argument("--version", "-v", type=int, help="Specific version to rollback to")
    rollback_parser.add_argument("--reason", "-r", help="Rollback reason")
    rollback_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # export command
    export_parser = subparsers.add_parser("export", help="Export a model")
    export_parser.add_argument("model_id", help="Model ID")
    export_parser.add_argument("--version", "-v", type=int, help="Version number")
    export_parser.add_argument("--output", "-o", required=True, help="Output path")

    # history command
    history_parser = subparsers.add_parser("history", help="Show stage history")
    history_parser.add_argument("model_id", help="Model ID")
    history_parser.add_argument("--version", "-v", type=int, help="Version number")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model_a", help="First model (model_id:version)")
    compare_parser.add_argument("model_b", help="Second model (model_id:version)")

    # stats command
    subparsers.add_parser("stats", help="Show registry statistics")

    # dashboard command
    subparsers.add_parser("dashboard", help="Show rich dashboard of registry state")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Run archival policy to clean up old models")
    cleanup_parser.add_argument("--execute", action="store_true", help="Execute cleanup (default is dry-run)")
    cleanup_parser.add_argument("--max-age", type=int, help="Max age in days for archival")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "list": cmd_list,
        "show": cmd_show,
        "promote": cmd_promote,
        "archive": cmd_archive,
        "rollback": cmd_rollback,
        "export": cmd_export,
        "history": cmd_history,
        "compare": cmd_compare,
        "stats": cmd_stats,
        "dashboard": cmd_dashboard,
        "cleanup": cmd_cleanup,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args) or 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
