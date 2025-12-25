#!/usr/bin/env python3
"""Cluster-wide game statistics analysis for RingRift.

Scans all cluster nodes via SSH and aggregates comprehensive game statistics:
- Victory type distribution (normalized: territory, elimination, LPS, stalemate)
- Forced elimination and recovery rates with win correlation
- Per-configuration breakdown
- Game length analysis
- Move type distribution
- First-player advantage by config
- Balance metrics

Usage:
    # Full cluster analysis
    python scripts/analyze_cluster_games.py

    # Filter by board type
    python scripts/analyze_cluster_games.py --board-type hex8 --num-players 2

    # Filter by AI type (for games with engine_mode metadata)
    python scripts/analyze_cluster_games.py --ai-type gumbel

    # Specific nodes only
    python scripts/analyze_cluster_games.py --nodes lambda-gh200-b lambda-h100

    # JSON output
    python scripts/analyze_cluster_games.py --format json --output report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Victory Type Normalization
# =============================================================================

VICTORY_TYPE_ALIASES = {
    # Elimination variants -> elimination
    "elimination": "elimination",
    "ring_elimination": "elimination",
    "forced_elimination": "elimination",
    "eliminate": "elimination",
    # Territory variants -> territory
    "territory": "territory",
    "territory_control": "territory",
    # LPS variants -> lps
    "lps": "lps",
    "last_player_standing": "lps",
    # Stalemate
    "stalemate": "stalemate",
    # Draw/timeout
    "draw": "draw",
    "timeout": "timeout",
    "max_moves": "max_moves",
    # Structural (valid victory type)
    "structural": "structural",
    # Line-based
    "line": "line",
}


def normalize_victory_type(raw: str | None) -> str:
    """Normalize victory type to canonical form."""
    if not raw:
        return "unknown"

    raw = str(raw).lower().strip()

    # Handle status:completed:X format
    if raw.startswith("status:"):
        parts = raw.split(":")
        if len(parts) >= 3:
            raw = parts[-1]

    # Check aliases
    if raw in VICTORY_TYPE_ALIASES:
        return VICTORY_TYPE_ALIASES[raw]

    # Handle partial matches
    if "elim" in raw:
        return "elimination"
    if "territory" in raw:
        return "territory"
    if "stalemate" in raw:
        return "stalemate"
    if "lps" in raw or "last_player" in raw or "standing" in raw:
        return "lps"

    # Special cases
    if raw in ("completed", "in_progress", "active"):
        return "incomplete"

    return raw if raw else "unknown"


@dataclass
class ConfigStats:
    """Statistics for a specific board_type + num_players configuration."""

    config_key: str
    board_type: str
    num_players: int

    total_games: int = 0

    # Victory types (normalized)
    victory_types: dict[str, int] = field(default_factory=dict)

    # Win rates by player position (0-indexed)
    wins_by_player: dict[int, int] = field(default_factory=dict)

    # Game length
    total_moves: int = 0
    game_lengths: list[int] = field(default_factory=list)

    # Recovery and forced elimination
    games_with_recovery: int = 0
    games_with_fe: int = 0
    recovery_winner_used_recovery: int = 0  # Winner used recovery slide
    fe_victim_won: int = 0  # Player who was FE'd still won

    # Move types (from detailed game_moves table)
    move_types: dict[str, int] = field(default_factory=dict)

    # AI types
    games_by_ai_type: dict[str, int] = field(default_factory=dict)

    # Stalemate tiebreakers
    stalemate_tiebreakers: dict[str, int] = field(default_factory=dict)

    # High move count (potential draws/timeouts)
    high_move_count_games: int = 0  # games with 400+ moves

    @property
    def avg_game_length(self) -> float:
        if not self.game_lengths:
            return self.total_moves / max(1, self.total_games)
        return sum(self.game_lengths) / len(self.game_lengths) if self.game_lengths else 0

    @property
    def first_player_advantage(self) -> float:
        """Calculate first-player advantage as deviation from expected win rate."""
        total_wins = sum(self.wins_by_player.values())
        if total_wins == 0 or self.num_players == 0:
            return 0.0
        p0_wins = self.wins_by_player.get(0, 0) + self.wins_by_player.get(1, 0)  # Handle 0 or 1-indexed
        expected = total_wins / self.num_players
        return (p0_wins - expected) / total_wins if total_wins > 0 else 0.0

    @property
    def recovery_rate(self) -> float:
        return self.games_with_recovery / max(1, self.total_games)

    @property
    def fe_rate(self) -> float:
        return self.games_with_fe / max(1, self.total_games)


@dataclass
class NodeGameStats:
    """Game statistics from a single cluster node."""

    host: str
    reachable: bool = False
    error: str | None = None

    total_games: int = 0
    databases_found: int = 0
    databases_with_moves: int = 0  # DBs with game_moves table

    # Per-config stats
    config_stats: dict[str, ConfigStats] = field(default_factory=dict)

    # Aggregate stats
    victory_types: dict[str, int] = field(default_factory=dict)
    games_by_ai_type: dict[str, int] = field(default_factory=dict)

    scan_duration_seconds: float = 0.0


@dataclass
class ClusterGameStats:
    """Aggregated game statistics across the cluster."""

    timestamp: str = ""

    # Node summary
    nodes_scanned: int = 0
    nodes_reachable: int = 0
    nodes_with_data: int = 0

    # Aggregate stats
    total_games: int = 0
    total_databases: int = 0
    databases_with_moves: int = 0

    # Per-config breakdown
    config_stats: dict[str, ConfigStats] = field(default_factory=dict)

    # Aggregate victory types (normalized)
    victory_types: dict[str, int] = field(default_factory=dict)

    # AI types
    games_by_ai_type: dict[str, int] = field(default_factory=dict)
    games_by_node: dict[str, int] = field(default_factory=dict)

    # Move analysis (from detailed DBs)
    move_types: dict[str, int] = field(default_factory=dict)

    # Recovery/FE aggregate
    games_with_recovery: int = 0
    games_with_fe: int = 0

    # Per-node details
    node_stats: dict[str, NodeGameStats] = field(default_factory=dict)

    errors: list[str] = field(default_factory=list)


def load_cluster_config() -> dict[str, Any]:
    """Load cluster configuration from distributed_hosts.yaml."""
    config_paths = [
        PROJECT_ROOT / "config" / "distributed_hosts.yaml",
        PROJECT_ROOT / "ai-service" / "config" / "distributed_hosts.yaml",
        Path.home() / ".ringrift" / "distributed_hosts.yaml",
    ]

    for path in config_paths:
        if path.exists() and HAS_YAML:
            with open(path) as f:
                return yaml.safe_load(f)

    # Fallback
    return {"hosts": {}}


def get_active_hosts(config: dict[str, Any], filter_hosts: list[str] | None = None) -> list[dict[str, Any]]:
    """Get list of active hosts to scan."""
    hosts = []

    host_config = config.get("hosts", {})
    for name, info in host_config.items():
        if filter_hosts and name not in filter_hosts:
            continue

        if isinstance(info, dict):
            ip = (
                info.get("ssh_host") or
                info.get("tailscale_ip") or
                info.get("ip") or
                info.get("address") or
                name
            )
            hosts.append({
                "name": name,
                "ip": ip,
                "user": info.get("ssh_user", info.get("user", "ubuntu")),
                "role": info.get("role", "worker"),
                "gpu": info.get("gpu", ""),
                "ringrift_path": info.get("ringrift_path", "~/ringrift/ai-service"),
                "ssh_key": info.get("ssh_key", "~/.ssh/id_cluster"),
            })
        else:
            hosts.append({
                "name": name,
                "ip": str(info),
                "user": "ubuntu",
                "role": "worker",
                "gpu": "",
                "ringrift_path": "~/ringrift/ai-service",
                "ssh_key": "~/.ssh/id_cluster",
            })

    return hosts


def run_ssh_command(
    host: str,
    command: str,
    user: str = "ubuntu",
    timeout: int = 60,
    key_path: str | None = None,
) -> tuple[bool, str, str]:
    """Run command on remote host via SSH."""
    ssh_opts = [
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
    ]

    if key_path:
        ssh_opts.extend(["-i", key_path])

    ssh_cmd = ["ssh"] + ssh_opts + [f"{user}@{host}", command]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "SSH command timed out"
    except Exception as e:
        return False, "", str(e)


# The remote analysis script - will be base64 encoded
REMOTE_ANALYSIS_SCRIPT = '''
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

VICTORY_ALIASES = {
    "elimination": "elimination",
    "ring_elimination": "elimination",
    "forced_elimination": "elimination",
    "territory": "territory",
    "territory_control": "territory",
    "lps": "lps",
    "last_player_standing": "lps",
    "stalemate": "stalemate",
    "structural": "structural",
    "line": "line",
    "draw": "draw",
    "timeout": "timeout",
}

def normalize_vt(raw):
    if not raw:
        return "unknown"
    raw = str(raw).lower().strip()
    if raw.startswith("status:"):
        parts = raw.split(":")
        raw = parts[-1] if len(parts) >= 3 else raw
    if raw in VICTORY_ALIASES:
        return VICTORY_ALIASES[raw]
    if "elim" in raw:
        return "elimination"
    if "territory" in raw:
        return "territory"
    if "stalemate" in raw:
        return "stalemate"
    if "lps" in raw or "standing" in raw:
        return "lps"
    if raw in ("completed", "in_progress", "active"):
        return "incomplete"
    return raw or "unknown"

def scan_databases():
    results = {
        "total_games": 0,
        "databases_found": 0,
        "databases_with_moves": 0,
        "config_stats": {},
        "move_types": {},
        "games_with_recovery": 0,
        "games_with_fe": 0,
    }

    search_paths = [
        Path.home() / "ringrift" / "ai-service" / "data" / "games",
        Path.home() / "ringrift" / "ai-service" / "data" / "selfplay",
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue

        for db_path in base_path.rglob("*.db"):
            try:
                conn = sqlite3.connect(str(db_path), timeout=5)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='games'")
                if not cursor.fetchone():
                    conn.close()
                    continue

                results["databases_found"] += 1

                cursor.execute("PRAGMA table_info(games)")
                columns = {row[1] for row in cursor.fetchall()}

                # Check for game_moves table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
                has_moves_table = cursor.fetchone() is not None
                if has_moves_table:
                    results["databases_with_moves"] += 1

                # Determine status column
                if 'game_status' in columns:
                    status_filter = "game_status = 'completed'"
                elif 'status' in columns:
                    status_filter = "status = 'completed'"
                else:
                    status_filter = "1=1"

                cursor.execute(f"SELECT * FROM games WHERE {status_filter}")

                for row in cursor.fetchall():
                    game = dict(row)
                    results["total_games"] += 1

                    bt = game.get("board_type", "unknown")
                    np = game.get("num_players", 2)
                    if np is None:
                        np = 2
                    config_key = f"{bt}_{np}p"

                    if config_key not in results["config_stats"]:
                        results["config_stats"][config_key] = {
                            "board_type": bt,
                            "num_players": np,
                            "total_games": 0,
                            "victory_types": {},
                            "wins_by_player": {},
                            "total_moves": 0,
                            "high_move_games": 0,
                            "games_with_recovery": 0,
                            "games_with_fe": 0,
                            "games_by_ai_type": {},
                            "stalemate_tiebreakers": {},
                        }

                    cs = results["config_stats"][config_key]
                    cs["total_games"] += 1

                    # Victory type
                    vt_raw = game.get("victory_type") or game.get("termination_reason") or "unknown"
                    vt = normalize_vt(vt_raw)
                    cs["victory_types"][vt] = cs["victory_types"].get(vt, 0) + 1

                    # Winner
                    winner = game.get("winner")
                    if winner is not None:
                        wk = str(winner)
                        cs["wins_by_player"][wk] = cs["wins_by_player"].get(wk, 0) + 1

                    # Move count
                    moves = game.get("total_moves") or game.get("move_count") or 0
                    cs["total_moves"] += moves
                    if moves >= 400:
                        cs["high_move_games"] += 1

                    # AI type
                    ai = game.get("engine_mode") or game.get("engine") or game.get("ai_type") or "unknown"
                    ai = str(ai).lower()
                    if "gumbel" in ai:
                        ai = "gumbel"
                    elif "mcts" in ai and "gumbel" not in ai:
                        ai = "mcts"
                    elif "heuristic" in ai:
                        ai = "heuristic"
                    cs["games_by_ai_type"][ai] = cs["games_by_ai_type"].get(ai, 0) + 1

                    # Stalemate tiebreaker
                    if vt == "stalemate" and "stalemate_tiebreaker" in columns:
                        tb = game.get("stalemate_tiebreaker") or "unknown"
                        cs["stalemate_tiebreakers"][tb] = cs["stalemate_tiebreakers"].get(tb, 0) + 1

                # Analyze game_moves if available
                if has_moves_table:
                    try:
                        cursor.execute("SELECT move_type, COUNT(*) FROM game_moves GROUP BY move_type")
                        for row in cursor.fetchall():
                            mt = row[0] or "unknown"
                            results["move_types"][mt] = results["move_types"].get(mt, 0) + row[1]

                        # Count games with recovery
                        cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_moves WHERE move_type = 'recovery_slide'")
                        rec_count = cursor.fetchone()[0]
                        results["games_with_recovery"] += rec_count

                        # Count games with forced elimination
                        cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_moves WHERE move_type = 'forced_elimination'")
                        fe_count = cursor.fetchone()[0]
                        results["games_with_fe"] += fe_count
                    except:
                        pass

                conn.close()
            except Exception as e:
                continue

    return results

if __name__ == "__main__":
    print(json.dumps(scan_databases()))
'''


def scan_node_games(host: dict[str, Any]) -> NodeGameStats:
    """Scan a single node for game statistics."""
    import time
    start_time = time.time()

    stats = NodeGameStats(host=host["name"])

    # Base64 encode the script
    script_b64 = base64.b64encode(REMOTE_ANALYSIS_SCRIPT.encode()).decode()

    # Expand ssh key path
    ssh_key = os.path.expanduser(host.get("ssh_key", "~/.ssh/id_cluster"))

    # Run the script remotely
    success, stdout, stderr = run_ssh_command(
        host["ip"],
        f'python3 -c "import base64; exec(base64.b64decode(\\"{script_b64}\\").decode())"',
        user=host.get("user", "ubuntu"),
        timeout=180,  # Increased timeout for larger scans
        key_path=ssh_key if os.path.exists(ssh_key) else None,
    )

    stats.scan_duration_seconds = time.time() - start_time

    if not success:
        stats.reachable = False
        stats.error = stderr[:100] if stderr else "SSH connection failed"
        return stats

    stats.reachable = True

    try:
        data = json.loads(stdout.strip())

        stats.total_games = data.get("total_games", 0)
        stats.databases_found = data.get("databases_found", 0)
        stats.databases_with_moves = data.get("databases_with_moves", 0)

        # Parse config stats
        for config_key, cs_data in data.get("config_stats", {}).items():
            cs = ConfigStats(
                config_key=config_key,
                board_type=cs_data.get("board_type", "unknown"),
                num_players=cs_data.get("num_players", 2),
                total_games=cs_data.get("total_games", 0),
                victory_types=cs_data.get("victory_types", {}),
                total_moves=cs_data.get("total_moves", 0),
                high_move_count_games=cs_data.get("high_move_games", 0),
                games_with_recovery=cs_data.get("games_with_recovery", 0),
                games_with_fe=cs_data.get("games_with_fe", 0),
                games_by_ai_type=cs_data.get("games_by_ai_type", {}),
                stalemate_tiebreakers=cs_data.get("stalemate_tiebreakers", {}),
            )
            # Parse wins by player
            for k, v in cs_data.get("wins_by_player", {}).items():
                try:
                    cs.wins_by_player[int(k)] = v
                except ValueError:
                    pass
            stats.config_stats[config_key] = cs

            # Aggregate victory types
            for vt, count in cs.victory_types.items():
                stats.victory_types[vt] = stats.victory_types.get(vt, 0) + count

            # Aggregate AI types
            for ai, count in cs.games_by_ai_type.items():
                stats.games_by_ai_type[ai] = stats.games_by_ai_type.get(ai, 0) + count

    except json.JSONDecodeError as e:
        stats.error = f"JSON parse error: {str(e)[:50]}"
    except Exception as e:
        stats.error = str(e)[:100]

    return stats


def analyze_cluster(
    hosts: list[dict[str, Any]],
    parallel: bool = True,
) -> ClusterGameStats:
    """Analyze game statistics across all cluster nodes."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cluster_stats = ClusterGameStats(
        timestamp=datetime.now().isoformat(),
    )

    def scan_host(host):
        logger.info(f"Scanning {host['name']} ({host['ip']})...")
        return scan_node_games(host)

    if parallel and len(hosts) > 1:
        with ThreadPoolExecutor(max_workers=min(10, len(hosts))) as executor:
            futures = {executor.submit(scan_host, h): h for h in hosts}
            for future in as_completed(futures):
                host = futures[future]
                try:
                    stats = future.result()
                    cluster_stats.node_stats[host["name"]] = stats
                except Exception as e:
                    cluster_stats.errors.append(f"{host['name']}: {e}")
    else:
        for host in hosts:
            try:
                stats = scan_host(host)
                cluster_stats.node_stats[host["name"]] = stats
            except Exception as e:
                cluster_stats.errors.append(f"{host['name']}: {e}")

    # Aggregate statistics
    cluster_stats.nodes_scanned = len(hosts)

    for node_name, node_stats in cluster_stats.node_stats.items():
        if node_stats.reachable:
            cluster_stats.nodes_reachable += 1

            if node_stats.total_games > 0:
                cluster_stats.nodes_with_data += 1
                cluster_stats.total_games += node_stats.total_games
                cluster_stats.total_databases += node_stats.databases_found
                cluster_stats.databases_with_moves += node_stats.databases_with_moves
                cluster_stats.games_by_node[node_name] = node_stats.total_games

                # Merge config stats
                for config_key, cs in node_stats.config_stats.items():
                    if config_key not in cluster_stats.config_stats:
                        cluster_stats.config_stats[config_key] = ConfigStats(
                            config_key=config_key,
                            board_type=cs.board_type,
                            num_players=cs.num_players,
                        )

                    agg = cluster_stats.config_stats[config_key]
                    agg.total_games += cs.total_games
                    agg.total_moves += cs.total_moves
                    agg.high_move_count_games += cs.high_move_count_games
                    agg.games_with_recovery += cs.games_with_recovery
                    agg.games_with_fe += cs.games_with_fe

                    for vt, count in cs.victory_types.items():
                        agg.victory_types[vt] = agg.victory_types.get(vt, 0) + count

                    for p, wins in cs.wins_by_player.items():
                        agg.wins_by_player[p] = agg.wins_by_player.get(p, 0) + wins

                    for ai, count in cs.games_by_ai_type.items():
                        agg.games_by_ai_type[ai] = agg.games_by_ai_type.get(ai, 0) + count

                    for tb, count in cs.stalemate_tiebreakers.items():
                        agg.stalemate_tiebreakers[tb] = agg.stalemate_tiebreakers.get(tb, 0) + count

                # Aggregate victory types
                for vt, count in node_stats.victory_types.items():
                    cluster_stats.victory_types[vt] = cluster_stats.victory_types.get(vt, 0) + count

                # Aggregate AI types
                for ai, count in node_stats.games_by_ai_type.items():
                    cluster_stats.games_by_ai_type[ai] = cluster_stats.games_by_ai_type.get(ai, 0) + count

    # Aggregate recovery/FE from config stats
    for cs in cluster_stats.config_stats.values():
        cluster_stats.games_with_recovery += cs.games_with_recovery
        cluster_stats.games_with_fe += cs.games_with_fe

    return cluster_stats


def format_report_markdown(stats: ClusterGameStats) -> str:
    """Format cluster statistics as comprehensive markdown report."""
    lines = [
        "# Cluster Game Statistics Report",
        "",
        f"**Generated:** {stats.timestamp}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Games:** {stats.total_games:,}",
        f"- **Databases Scanned:** {stats.total_databases:,}",
        f"- **Databases with Move Data:** {stats.databases_with_moves:,}",
        f"- **Nodes Scanned:** {stats.nodes_scanned} ({stats.nodes_reachable} reachable, {stats.nodes_with_data} with data)",
        "",
    ]

    # Victory Type Distribution (aggregated)
    lines.extend([
        "## Victory Type Distribution (Normalized)",
        "",
        "| Type | Count | Rate |",
        "|------|-------|------|",
    ])

    total = stats.total_games
    for vtype, count in sorted(stats.victory_types.items(), key=lambda x: -x[1]):
        rate = count / total if total > 0 else 0
        lines.append(f"| {vtype} | {count:,} | {rate:.1%} |")
    lines.append("")

    # Per-Configuration Breakdown
    lines.extend([
        "## Per-Configuration Analysis",
        "",
    ])

    for config_key in sorted(stats.config_stats.keys()):
        cs = stats.config_stats[config_key]
        if cs.total_games == 0:
            continue

        lines.extend([
            f"### {config_key.upper()}",
            "",
            f"**Games:** {cs.total_games:,} | **Avg Length:** {cs.avg_game_length:.1f} moves",
            "",
        ])

        # Victory types for this config
        lines.append("**Victory Types:**")
        lines.append("")
        lines.append("| Type | Count | Rate |")
        lines.append("|------|-------|------|")
        for vt, count in sorted(cs.victory_types.items(), key=lambda x: -x[1]):
            rate = count / cs.total_games
            lines.append(f"| {vt} | {count:,} | {rate:.1%} |")
        lines.append("")

        # Win rates by player
        if cs.wins_by_player:
            total_wins = sum(cs.wins_by_player.values())
            lines.append("**Win Rates by Position:**")
            lines.append("")
            lines.append("| Player | Wins | Rate |")
            lines.append("|--------|------|------|")
            for p in sorted(cs.wins_by_player.keys()):
                wins = cs.wins_by_player[p]
                rate = wins / total_wins if total_wins > 0 else 0
                lines.append(f"| P{p} | {wins:,} | {rate:.1%} |")
            lines.append("")

            # First player advantage
            fpa = cs.first_player_advantage
            lines.append(f"**First-Player Advantage:** {fpa:+.1%}")
            lines.append("")

        # Recovery/FE stats
        if cs.games_with_recovery > 0 or cs.games_with_fe > 0:
            lines.append("**Game Mechanics:**")
            lines.append("")
            lines.append(f"- Recovery Used: {cs.games_with_recovery:,} ({cs.recovery_rate:.1%})")
            lines.append(f"- Forced Elimination: {cs.games_with_fe:,} ({cs.fe_rate:.1%})")
            lines.append(f"- High Move Count (400+): {cs.high_move_count_games:,}")
            lines.append("")

        # Stalemate tiebreakers
        if cs.stalemate_tiebreakers:
            lines.append("**Stalemate Tiebreakers:**")
            lines.append("")
            for tb, count in sorted(cs.stalemate_tiebreakers.items(), key=lambda x: -x[1]):
                lines.append(f"- {tb}: {count:,}")
            lines.append("")

    # AI Type Breakdown
    if stats.games_by_ai_type:
        lines.extend([
            "## AI Type Breakdown",
            "",
            "| AI Type | Games | Percentage |",
            "|---------|-------|------------|",
        ])
        for ai_type, count in sorted(stats.games_by_ai_type.items(), key=lambda x: -x[1]):
            pct = count / total if total > 0 else 0
            lines.append(f"| {ai_type} | {count:,} | {pct:.1%} |")
        lines.append("")

    # Recovery and FE Summary
    lines.extend([
        "## Recovery & Forced Elimination Summary",
        "",
        f"- **Games with Recovery Slide:** {stats.games_with_recovery:,}",
        f"- **Games with Forced Elimination:** {stats.games_with_fe:,}",
        "",
    ])

    # Node breakdown
    lines.extend([
        "## Per-Node Statistics",
        "",
        "| Node | Games | DBs | Move DBs | Status |",
        "|------|-------|-----|----------|--------|",
    ])
    for node_name in sorted(stats.node_stats.keys()):
        ns = stats.node_stats[node_name]
        status = "✓" if ns.reachable else "✗"
        if ns.error:
            status = f"⚠ {ns.error[:25]}..."
        lines.append(
            f"| {node_name} | {ns.total_games:,} | "
            f"{ns.databases_found} | {ns.databases_with_moves} | {status} |"
        )
    lines.append("")

    # Balance Assessment
    lines.extend([
        "## Balance Assessment",
        "",
    ])

    for config_key, cs in sorted(stats.config_stats.items()):
        if cs.total_games < 100:
            continue

        issues = []

        # Check first-player advantage
        fpa = cs.first_player_advantage
        if abs(fpa) > 0.15:
            issues.append(f"⚠️ Strong first-player {'advantage' if fpa > 0 else 'disadvantage'}: {fpa:+.1%}")
        elif abs(fpa) > 0.10:
            issues.append(f"📊 Moderate first-player deviation: {fpa:+.1%}")

        # Check victory type distribution
        vt_total = sum(cs.victory_types.values())
        for vt, count in cs.victory_types.items():
            rate = count / vt_total if vt_total > 0 else 0
            if vt == "elimination" and rate > 0.50:
                issues.append(f"⚠️ High elimination rate: {rate:.1%}")
            if vt == "incomplete" and rate > 0.10:
                issues.append(f"⚠️ High incomplete game rate: {rate:.1%}")

        # Check high move count games (potential draws/stalemates)
        if cs.high_move_count_games > 0:
            hm_rate = cs.high_move_count_games / cs.total_games
            if hm_rate > 0.05:
                issues.append(f"📊 {hm_rate:.1%} games reach 400+ moves")

        if issues:
            lines.append(f"**{config_key}:**")
            for issue in issues:
                lines.append(f"  - {issue}")
            lines.append("")

    if not any(cs.total_games >= 100 for cs in stats.config_stats.values()):
        lines.append("*Insufficient data for balance assessment (need 100+ games per config)*")
        lines.append("")

    return "\n".join(lines)


def format_report_json(stats: ClusterGameStats) -> str:
    """Format cluster statistics as JSON."""
    data = {
        "timestamp": stats.timestamp,
        "summary": {
            "total_games": stats.total_games,
            "total_databases": stats.total_databases,
            "databases_with_moves": stats.databases_with_moves,
            "nodes_scanned": stats.nodes_scanned,
            "nodes_reachable": stats.nodes_reachable,
            "nodes_with_data": stats.nodes_with_data,
        },
        "victory_types": stats.victory_types,
        "games_by_ai_type": stats.games_by_ai_type,
        "games_by_node": stats.games_by_node,
        "recovery_games": stats.games_with_recovery,
        "forced_elimination_games": stats.games_with_fe,
        "config_stats": {},
        "errors": stats.errors,
    }

    for config_key, cs in stats.config_stats.items():
        data["config_stats"][config_key] = {
            "board_type": cs.board_type,
            "num_players": cs.num_players,
            "total_games": cs.total_games,
            "avg_game_length": cs.avg_game_length,
            "first_player_advantage": cs.first_player_advantage,
            "victory_types": cs.victory_types,
            "wins_by_player": {str(k): v for k, v in cs.wins_by_player.items()},
            "recovery_rate": cs.recovery_rate,
            "fe_rate": cs.fe_rate,
            "high_move_games": cs.high_move_count_games,
            "games_by_ai_type": cs.games_by_ai_type,
            "stalemate_tiebreakers": cs.stalemate_tiebreakers,
        }

    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive cluster-wide game statistics analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--board-type", "-b",
        choices=["hex8", "square8", "square19", "hexagonal"],
        help="Filter by board type",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        choices=[2, 3, 4],
        help="Filter by number of players",
    )
    parser.add_argument(
        "--ai-type", "-a",
        help="Filter by AI type (gumbel, mcts, heuristic)",
    )
    parser.add_argument(
        "--nodes", "-n",
        nargs="+",
        help="Specific nodes to scan",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Scan nodes sequentially",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_cluster_config()
    hosts = get_active_hosts(config, args.nodes)

    if not hosts:
        logger.error("No hosts configured")
        sys.exit(1)

    logger.info(f"Scanning {len(hosts)} cluster nodes...")

    # Analyze cluster
    stats = analyze_cluster(hosts, parallel=not args.sequential)

    # Filter by config if requested
    if args.board_type or args.num_players:
        filtered_configs = {}
        for config_key, cs in stats.config_stats.items():
            if args.board_type and cs.board_type != args.board_type:
                continue
            if args.num_players and cs.num_players != args.num_players:
                continue
            filtered_configs[config_key] = cs
        stats.config_stats = filtered_configs

    # Format output
    if args.format == "json":
        output = format_report_json(stats)
    else:
        output = format_report_markdown(stats)

    # Write output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Report saved to {args.output}")
    else:
        print(output)

    logger.info(
        f"Scan complete: {stats.total_games:,} games across "
        f"{stats.nodes_with_data}/{stats.nodes_reachable} nodes"
    )


if __name__ == "__main__":
    main()
