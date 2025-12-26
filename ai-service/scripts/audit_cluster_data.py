#!/usr/bin/env python3
"""Cluster-Wide Data Audit Script.

Audits all cluster nodes for:
1. Database inventory (path, size, game count)
2. Orphaned data (on disk but not in manifest)
3. Under-replicated games (<2 copies)
4. Data silos (unique data on single node)

Usage:
    # Full audit of all nodes
    python scripts/audit_cluster_data.py

    # Audit specific nodes
    python scripts/audit_cluster_data.py --nodes lambda-gh200-a,lambda-gh200-b

    # Dry-run (report only, no imports)
    python scripts/audit_cluster_data.py --dry-run

    # Export report to JSON
    python scripts/audit_cluster_data.py --output audit_report.json

    # Include model/NPZ audit
    python scripts/audit_cluster_data.py --include-models --include-npz
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.distributed.cluster_manifest import (
    ClusterManifest,
    get_cluster_manifest,
    REPLICATION_TARGET_COUNT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseEntry:
    """Information about a database on a node."""
    path: str
    size_bytes: int = 0
    game_count: int = 0
    board_type: str | None = None
    num_players: int | None = None
    is_canonical: bool = False
    in_manifest: bool = False
    error: str | None = None


@dataclass
class ModelEntry:
    """Information about a model on a node."""
    path: str
    size_bytes: int = 0
    board_type: str | None = None
    num_players: int | None = None
    in_manifest: bool = False


@dataclass
class NPZEntry:
    """Information about an NPZ file on a node."""
    path: str
    size_bytes: int = 0
    sample_count: int = 0
    board_type: str | None = None
    num_players: int | None = None
    in_manifest: bool = False


@dataclass
class NodeAuditResult:
    """Audit result for a single node."""
    node_id: str
    ssh_host: str
    reachable: bool = False
    error: str | None = None
    databases: list[DatabaseEntry] = field(default_factory=list)
    models: list[ModelEntry] = field(default_factory=list)
    npz_files: list[NPZEntry] = field(default_factory=list)
    total_db_size_bytes: int = 0
    total_game_count: int = 0
    total_model_count: int = 0
    total_npz_count: int = 0
    orphaned_databases: list[str] = field(default_factory=list)
    unique_databases: list[str] = field(default_factory=list)
    audit_time_seconds: float = 0.0


@dataclass
class ClusterAuditReport:
    """Full cluster audit report."""
    timestamp: float = 0.0
    nodes_audited: int = 0
    nodes_reachable: int = 0
    nodes_unreachable: list[str] = field(default_factory=list)
    total_databases: int = 0
    total_games: int = 0
    total_models: int = 0
    total_npz_files: int = 0
    orphaned_databases: dict[str, list[str]] = field(default_factory=dict)
    under_replicated_games: int = 0
    data_silos: dict[str, list[str]] = field(default_factory=dict)
    node_results: dict[str, NodeAuditResult] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


def load_hosts_config(config_path: Path | None = None) -> dict[str, dict]:
    """Load hosts configuration from YAML."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "distributed_hosts.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config.get("hosts", {})


def get_active_hosts(hosts_config: dict[str, dict]) -> list[tuple[str, dict]]:
    """Get list of active hosts to audit."""
    active = []
    for name, config in hosts_config.items():
        # Skip coordinator/dev machines
        if config.get("role") == "coordinator":
            continue
        # Skip explicitly disabled
        if config.get("status") not in ("ready", "active"):
            continue
        # Include all GPU nodes
        active.append((name, config))
    return active


def ssh_command(host_config: dict, command: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute SSH command and return stdout, stderr, returncode."""
    ssh_host = host_config.get("tailscale_ip") or host_config.get("ssh_host")
    ssh_user = host_config.get("ssh_user", "ubuntu")
    ssh_key = os.path.expanduser(host_config.get("ssh_key", "~/.ssh/id_cluster"))
    ssh_port = host_config.get("ssh_port", 22)

    cmd = [
        "ssh",
        "-i", ssh_key,
        "-p", str(ssh_port),
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        f"{ssh_user}@{ssh_host}",
        command,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"SSH timeout after {timeout}s", -1
    except Exception as e:
        return "", str(e), -1


def audit_node_databases(
    node_id: str,
    host_config: dict,
    include_models: bool = False,
    include_npz: bool = False,
) -> NodeAuditResult:
    """Audit a single node for databases, models, and NPZ files."""
    start_time = time.time()
    result = NodeAuditResult(
        node_id=node_id,
        ssh_host=host_config.get("tailscale_ip") or host_config.get("ssh_host", ""),
    )

    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")

    # Build audit script to run on remote
    audit_script = f'''
import json
import os
import sqlite3
from pathlib import Path

def get_db_info(db_path):
    """Get database info."""
    try:
        stat = db_path.stat()
        size = stat.st_size

        # Query game count and config
        conn = sqlite3.connect(f"file:{{db_path}}?mode=ro", uri=True, timeout=5.0)
        cursor = conn.cursor()

        # Total games
        cursor.execute("SELECT COUNT(*) FROM games WHERE winner IS NOT NULL")
        game_count = cursor.fetchone()[0]

        # Board type and players (from first game)
        cursor.execute("SELECT board_type, num_players FROM games LIMIT 1")
        row = cursor.fetchone()
        board_type = row[0] if row else None
        num_players = row[1] if row else None

        conn.close()

        return {{
            "path": str(db_path),
            "size_bytes": size,
            "game_count": game_count,
            "board_type": board_type,
            "num_players": num_players,
            "is_canonical": "canonical" in db_path.name.lower(),
            "error": None,
        }}
    except Exception as e:
        return {{
            "path": str(db_path),
            "size_bytes": 0,
            "game_count": 0,
            "board_type": None,
            "num_players": None,
            "is_canonical": False,
            "error": str(e),
        }}

def get_model_info(model_path):
    """Get model info."""
    try:
        stat = model_path.stat()
        name = model_path.stem

        # Parse board type and players from name
        board_type = None
        num_players = None
        for bt in ["hex8", "hexagonal", "square8", "square19"]:
            if bt in name:
                board_type = bt
                break
        for np in ["2p", "3p", "4p"]:
            if np in name:
                num_players = int(np[0])
                break

        return {{
            "path": str(model_path),
            "size_bytes": stat.st_size,
            "board_type": board_type,
            "num_players": num_players,
        }}
    except Exception as e:
        return None

def get_npz_info(npz_path):
    """Get NPZ info."""
    try:
        import numpy as np
        stat = npz_path.stat()
        name = npz_path.stem

        # Parse board type and players
        board_type = None
        num_players = None
        for bt in ["hex8", "hexagonal", "square8", "square19"]:
            if bt in name:
                board_type = bt
                break
        for np_str in ["2p", "3p", "4p"]:
            if np_str in name:
                num_players = int(np_str[0])
                break

        # Get sample count
        sample_count = 0
        try:
            data = np.load(npz_path, allow_pickle=True)
            if "features" in data:
                sample_count = len(data["features"])
            elif "states" in data:
                sample_count = len(data["states"])
        except (OSError, ValueError, KeyError):
            pass  # File not readable or missing expected keys

        return {{
            "path": str(npz_path),
            "size_bytes": stat.st_size,
            "sample_count": sample_count,
            "board_type": board_type,
            "num_players": num_players,
        }}
    except Exception as e:
        return None

# Find all databases
base_path = Path("{ringrift_path}").expanduser()
databases = []

# Search directories
search_dirs = [
    base_path / "data" / "games",
    base_path / "data" / "selfplay",
    base_path / "staging",
]

for search_dir in search_dirs:
    if search_dir.exists():
        for db_path in search_dir.rglob("*.db"):
            if db_path.is_file() and db_path.stat().st_size > 0:
                info = get_db_info(db_path)
                if info:
                    databases.append(info)

# Find models if requested
models = []
if {include_models}:
    model_dir = base_path / "models"
    if model_dir.exists():
        for model_path in model_dir.glob("*.pth"):
            info = get_model_info(model_path)
            if info:
                models.append(info)
        for model_path in model_dir.glob("*.pt"):
            info = get_model_info(model_path)
            if info:
                models.append(info)

# Find NPZ files if requested
npz_files = []
if {include_npz}:
    npz_dir = base_path / "data" / "training"
    if npz_dir.exists():
        for npz_path in npz_dir.rglob("*.npz"):
            info = get_npz_info(npz_path)
            if info:
                npz_files.append(info)

# Output as JSON
print(json.dumps({{
    "databases": databases,
    "models": models,
    "npz_files": npz_files,
}}))
'''

    # Execute remote audit
    stdout, stderr, returncode = ssh_command(
        host_config,
        f"cd {ringrift_path} && python3 -c '{audit_script}' 2>/dev/null",
        timeout=60,
    )

    if returncode != 0:
        result.reachable = False
        result.error = stderr or f"SSH failed with code {returncode}"
        result.audit_time_seconds = time.time() - start_time
        return result

    result.reachable = True

    # Parse JSON output
    try:
        # Find JSON in output (skip any welcome messages)
        for line in stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                data = json.loads(line)
                break
        else:
            raise ValueError("No JSON output found")

        # Process databases
        for db_data in data.get("databases", []):
            entry = DatabaseEntry(
                path=db_data["path"],
                size_bytes=db_data.get("size_bytes", 0),
                game_count=db_data.get("game_count", 0),
                board_type=db_data.get("board_type"),
                num_players=db_data.get("num_players"),
                is_canonical=db_data.get("is_canonical", False),
                error=db_data.get("error"),
            )
            result.databases.append(entry)
            result.total_db_size_bytes += entry.size_bytes
            result.total_game_count += entry.game_count

        # Process models
        for model_data in data.get("models", []):
            entry = ModelEntry(
                path=model_data["path"],
                size_bytes=model_data.get("size_bytes", 0),
                board_type=model_data.get("board_type"),
                num_players=model_data.get("num_players"),
            )
            result.models.append(entry)
            result.total_model_count += 1

        # Process NPZ files
        for npz_data in data.get("npz_files", []):
            entry = NPZEntry(
                path=npz_data["path"],
                size_bytes=npz_data.get("size_bytes", 0),
                sample_count=npz_data.get("sample_count", 0),
                board_type=npz_data.get("board_type"),
                num_players=npz_data.get("num_players"),
            )
            result.npz_files.append(entry)
            result.total_npz_count += 1

    except Exception as e:
        result.error = f"Failed to parse output: {e}"

    result.audit_time_seconds = time.time() - start_time
    return result


def check_manifest_registration(
    manifest: ClusterManifest,
    node_results: dict[str, NodeAuditResult],
) -> None:
    """Check which databases are registered in the manifest."""
    for node_id, result in node_results.items():
        for db in result.databases:
            # Check if any games from this DB are in manifest
            db_name = Path(db.path).name

            # Query manifest for games with this DB path
            try:
                with manifest._connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) FROM game_locations WHERE db_path LIKE ? AND node_id = ?",
                        (f"%{db_name}", node_id),
                    )
                    count = cursor.fetchone()[0]
                    db.in_manifest = count > 0
            except Exception:
                db.in_manifest = False

            if not db.in_manifest and db.game_count > 0:
                result.orphaned_databases.append(db.path)


def find_data_silos(node_results: dict[str, NodeAuditResult]) -> dict[str, list[str]]:
    """Find databases that exist on only one node (data silos)."""
    # Build map of database names to nodes
    db_to_nodes: dict[str, list[str]] = {}

    for node_id, result in node_results.items():
        if not result.reachable:
            continue

        for db in result.databases:
            db_name = Path(db.path).name
            if db_name not in db_to_nodes:
                db_to_nodes[db_name] = []
            db_to_nodes[db_name].append(node_id)

    # Find databases on only one node
    silos: dict[str, list[str]] = {}
    for db_name, nodes in db_to_nodes.items():
        if len(nodes) == 1:
            node = nodes[0]
            if node not in silos:
                silos[node] = []
            silos[node].append(db_name)

    return silos


def generate_recommendations(report: ClusterAuditReport) -> list[str]:
    """Generate actionable recommendations from audit."""
    recommendations = []

    # Unreachable nodes
    if report.nodes_unreachable:
        recommendations.append(
            f"CRITICAL: {len(report.nodes_unreachable)} nodes unreachable. "
            f"Check SSH connectivity: {', '.join(report.nodes_unreachable[:3])}"
        )

    # Data silos
    total_silos = sum(len(dbs) for dbs in report.data_silos.values())
    if total_silos > 0:
        recommendations.append(
            f"WARNING: {total_silos} databases exist on only one node (data silos). "
            "Run replication to ensure data safety."
        )
        # List specific high-value silos
        for node, dbs in report.data_silos.items():
            for db in dbs[:3]:  # First 3
                if "improvement" in db or "canonical" in db:
                    recommendations.append(
                        f"  HIGH PRIORITY: {db} on {node} - replicate immediately"
                    )

    # Orphaned databases
    total_orphaned = sum(len(dbs) for dbs in report.orphaned_databases.values())
    if total_orphaned > 0:
        recommendations.append(
            f"ACTION: {total_orphaned} databases not in manifest. "
            "Run `python scripts/import_orphaned_databases.py` to register."
        )

    # Under-replicated games
    if report.under_replicated_games > 0:
        recommendations.append(
            f"WARNING: {report.under_replicated_games:,} games have < {REPLICATION_TARGET_COUNT} replicas. "
            "Trigger sync to improve data safety."
        )

    if not recommendations:
        recommendations.append("Cluster data is healthy. No immediate actions required.")

    return recommendations


def run_cluster_audit(
    nodes: list[str] | None = None,
    include_models: bool = False,
    include_npz: bool = False,
    parallel: bool = True,
    max_workers: int = 10,
) -> ClusterAuditReport:
    """Run full cluster audit."""
    report = ClusterAuditReport(timestamp=time.time())

    # Load hosts
    hosts_config = load_hosts_config()
    active_hosts = get_active_hosts(hosts_config)

    # Filter to specified nodes
    if nodes:
        node_set = set(nodes)
        active_hosts = [(n, c) for n, c in active_hosts if n in node_set]

    report.nodes_audited = len(active_hosts)
    logger.info(f"Auditing {report.nodes_audited} nodes...")

    # Audit each node
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    audit_node_databases, name, config, include_models, include_npz
                ): name
                for name, config in active_hosts
            }

            for future in concurrent.futures.as_completed(futures):
                node_id = futures[future]
                try:
                    result = future.result()
                    report.node_results[node_id] = result

                    if result.reachable:
                        report.nodes_reachable += 1
                        logger.info(
                            f"  {node_id}: {len(result.databases)} DBs, "
                            f"{result.total_game_count:,} games"
                        )
                    else:
                        report.nodes_unreachable.append(node_id)
                        logger.warning(f"  {node_id}: UNREACHABLE - {result.error}")

                except Exception as e:
                    logger.error(f"  {node_id}: ERROR - {e}")
                    report.nodes_unreachable.append(node_id)
    else:
        for name, config in active_hosts:
            result = audit_node_databases(name, config, include_models, include_npz)
            report.node_results[name] = result

            if result.reachable:
                report.nodes_reachable += 1
            else:
                report.nodes_unreachable.append(name)

    # Aggregate counts
    for result in report.node_results.values():
        if result.reachable:
            report.total_databases += len(result.databases)
            report.total_games += result.total_game_count
            report.total_models += result.total_model_count
            report.total_npz_files += result.total_npz_count

    # Check manifest registration
    manifest = get_cluster_manifest()
    check_manifest_registration(manifest, report.node_results)

    # Build orphaned databases report
    for node_id, result in report.node_results.items():
        if result.orphaned_databases:
            report.orphaned_databases[node_id] = result.orphaned_databases

    # Find data silos
    report.data_silos = find_data_silos(report.node_results)

    # Count under-replicated games
    try:
        under_replicated = manifest.get_under_replicated_games(limit=100000)
        report.under_replicated_games = len(under_replicated)
    except Exception as e:
        logger.warning(f"Failed to count under-replicated games: {e}")

    # Generate recommendations
    report.recommendations = generate_recommendations(report)

    return report


def print_report(report: ClusterAuditReport) -> None:
    """Print formatted audit report."""
    print("\n" + "=" * 70)
    print("CLUSTER DATA AUDIT REPORT")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Nodes audited:    {report.nodes_audited}")
    print(f"Nodes reachable:  {report.nodes_reachable}")
    print(f"Nodes unreachable: {len(report.nodes_unreachable)}")
    print(f"Total databases:  {report.total_databases}")
    print(f"Total games:      {report.total_games:,}")
    print(f"Total models:     {report.total_models}")
    print(f"Total NPZ files:  {report.total_npz_files}")

    if report.nodes_unreachable:
        print("\n" + "-" * 70)
        print("UNREACHABLE NODES")
        print("-" * 70)
        for node in report.nodes_unreachable:
            result = report.node_results.get(node)
            error = result.error if result else "Unknown"
            print(f"  {node}: {error}")

    if report.orphaned_databases:
        print("\n" + "-" * 70)
        print("ORPHANED DATABASES (not in manifest)")
        print("-" * 70)
        for node, dbs in report.orphaned_databases.items():
            print(f"\n  {node}:")
            for db in dbs[:10]:  # First 10
                print(f"    - {Path(db).name}")
            if len(dbs) > 10:
                print(f"    ... and {len(dbs) - 10} more")

    if report.data_silos:
        print("\n" + "-" * 70)
        print("DATA SILOS (single-node databases)")
        print("-" * 70)
        for node, dbs in report.data_silos.items():
            print(f"\n  {node}:")
            for db in dbs[:10]:
                # Get game count if available
                result = report.node_results.get(node)
                if result:
                    for db_entry in result.databases:
                        if Path(db_entry.path).name == db:
                            print(f"    - {db} ({db_entry.game_count:,} games)")
                            break
                    else:
                        print(f"    - {db}")
                else:
                    print(f"    - {db}")
            if len(dbs) > 10:
                print(f"    ... and {len(dbs) - 10} more")

    print("\n" + "-" * 70)
    print("REPLICATION STATUS")
    print("-" * 70)
    print(f"Under-replicated games: {report.under_replicated_games:,}")
    print(f"Target replicas:        {REPLICATION_TARGET_COUNT}")

    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")

    print("\n" + "=" * 70)


def export_report(report: ClusterAuditReport, output_path: Path) -> None:
    """Export report to JSON."""
    # Convert dataclasses to dicts
    data = {
        "timestamp": report.timestamp,
        "nodes_audited": report.nodes_audited,
        "nodes_reachable": report.nodes_reachable,
        "nodes_unreachable": report.nodes_unreachable,
        "total_databases": report.total_databases,
        "total_games": report.total_games,
        "total_models": report.total_models,
        "total_npz_files": report.total_npz_files,
        "orphaned_databases": report.orphaned_databases,
        "under_replicated_games": report.under_replicated_games,
        "data_silos": report.data_silos,
        "recommendations": report.recommendations,
        "node_results": {},
    }

    for node_id, result in report.node_results.items():
        data["node_results"][node_id] = {
            "node_id": result.node_id,
            "ssh_host": result.ssh_host,
            "reachable": result.reachable,
            "error": result.error,
            "total_db_size_bytes": result.total_db_size_bytes,
            "total_game_count": result.total_game_count,
            "total_model_count": result.total_model_count,
            "total_npz_count": result.total_npz_count,
            "databases": [asdict(db) for db in result.databases],
            "models": [asdict(m) for m in result.models],
            "npz_files": [asdict(n) for n in result.npz_files],
            "orphaned_databases": result.orphaned_databases,
            "unique_databases": result.unique_databases,
            "audit_time_seconds": result.audit_time_seconds,
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Report exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster-wide data audit")
    parser.add_argument(
        "--nodes",
        type=str,
        help="Comma-separated list of nodes to audit (default: all)",
    )
    parser.add_argument(
        "--include-models",
        action="store_true",
        help="Include model files in audit",
    )
    parser.add_argument(
        "--include-npz",
        action="store_true",
        help="Include NPZ training files in audit",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Export report to JSON file",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel auditing",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum parallel SSH connections",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Parse node list
    nodes = None
    if args.nodes:
        nodes = [n.strip() for n in args.nodes.split(",")]

    # Run audit
    report = run_cluster_audit(
        nodes=nodes,
        include_models=args.include_models,
        include_npz=args.include_npz,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
    )

    # Print report
    print_report(report)

    # Export if requested
    if args.output:
        export_report(report, Path(args.output))

    # Exit with error code if issues found
    if report.nodes_unreachable or report.under_replicated_games > 1000:
        sys.exit(1)


if __name__ == "__main__":
    main()
