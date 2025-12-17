#!/usr/bin/env python
"""Run a difficulty-tier tournament across multiple SSH hosts.

This is an orchestration wrapper around `scripts/run_distributed_tournament.py`
that distributes *matchups* (tier_a vs tier_b) across configured hosts in
`config/distributed_hosts.yaml` and aggregates the shard checkpoints into a
single report JSON.

Example (from ai-service/):

    python scripts/run_ssh_distributed_tournament.py \
        --tiers D1-D10 \
        --board square8 \
        --games-per-matchup 100 \
        --hosts mac-studio,aws-staging \
        --output-dir results/tournaments
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.ladder_config import get_ladder_tier_config
from app.distributed.hosts import (
    BOARD_MEMORY_REQUIREMENTS,
    HostConfig,
    SSHExecutor,
    load_remote_hosts,
)
from app.models import BoardType

from scripts.run_distributed_tournament import DistributedTournament, TournamentState

# Unified logging setup
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging("run_ssh_distributed_tournament", log_dir="logs")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


Matchup = Tuple[str, str]


@dataclass(frozen=True)
class WorkerSlot:
    host_name: str
    slot_index: int

    @property
    def id(self) -> str:
        return f"{self.host_name}#{self.slot_index}"


def _normalise_tier_name(tier: str) -> str:
    cleaned = tier.strip().upper()
    if not cleaned:
        raise ValueError("tier must be non-empty")
    if cleaned.startswith("D"):
        cleaned = cleaned[1:]
    if not cleaned.isdigit():
        raise ValueError(f"Invalid tier label: {tier!r}")
    return f"D{int(cleaned)}"


def parse_tiers_spec(spec: str) -> List[str]:
    """Parse a comma list (D1,D2) or range (D1-D10)."""
    raw = spec.strip()
    if not raw:
        raise ValueError("tiers must be non-empty")

    if "," not in raw and "-" in raw:
        start_s, end_s = (part.strip() for part in raw.split("-", 1))
        start = int(_normalise_tier_name(start_s)[1:])
        end = int(_normalise_tier_name(end_s)[1:])
        if end < start:
            raise ValueError(f"Invalid tier range: {spec!r}")
        return [f"D{i}" for i in range(start, end + 1)]

    tiers = [_normalise_tier_name(part) for part in raw.split(",") if part.strip()]
    if not tiers:
        raise ValueError("tiers must be non-empty")
    # De-dup while preserving order.
    seen = set()
    result: List[str] = []
    for t in tiers:
        if t not in seen:
            result.append(t)
            seen.add(t)
    return result


def enumerate_matchups(tiers: Sequence[str]) -> List[Matchup]:
    normalized = [_normalise_tier_name(t) for t in tiers]
    sorted_tiers = sorted(normalized, key=lambda t: int(t[1:]))
    matchups: List[Matchup] = []
    for i, tier_a in enumerate(sorted_tiers):
        for tier_b in sorted_tiers[i + 1 :]:
            matchups.append((tier_a, tier_b))
    return matchups


def _tier_to_difficulty(tier: str) -> int:
    return int(_normalise_tier_name(tier)[1:])


def _scaled_think_time_ms(think_time_ms: int, scale: float) -> int:
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        factor = 1.0
    if factor <= 0.0:
        factor = 0.0
    return max(0, int(round(think_time_ms * factor)))


def estimate_tier_think_time_ms(
    tier: str,
    *,
    board_type: BoardType,
    think_time_scale: float,
) -> int:
    difficulty = _tier_to_difficulty(tier)
    if difficulty == 1:
        return _scaled_think_time_ms(150, think_time_scale)
    ladder = get_ladder_tier_config(difficulty, board_type, num_players=2)
    return _scaled_think_time_ms(ladder.think_time_ms, think_time_scale)


def estimate_matchup_cost(
    matchup: Matchup,
    *,
    board_type: BoardType,
    think_time_scale: float,
    games_per_matchup: int,
) -> float:
    tier_a, tier_b = matchup
    a_ms = estimate_tier_think_time_ms(
        tier_a, board_type=board_type, think_time_scale=think_time_scale
    )
    b_ms = estimate_tier_think_time_ms(
        tier_b, board_type=board_type, think_time_scale=think_time_scale
    )
    return float(games_per_matchup) * float(max(a_ms, b_ms))


def assign_matchups_to_worker_slots(
    matchups: Sequence[Matchup],
    worker_slots: Sequence[WorkerSlot],
    cost_fn: Callable[[Matchup], float],
) -> Dict[WorkerSlot, List[Matchup]]:
    if not worker_slots:
        raise ValueError("worker_slots must be non-empty")

    loads: Dict[WorkerSlot, float] = {slot: 0.0 for slot in worker_slots}
    assignments: Dict[WorkerSlot, List[Matchup]] = {slot: [] for slot in worker_slots}

    for matchup in sorted(matchups, key=cost_fn, reverse=True):
        chosen = min(worker_slots, key=lambda s: loads[s])
        assignments[chosen].append(matchup)
        loads[chosen] += cost_fn(matchup)

    return assignments


def resolve_remote_path(host: HostConfig, path: str) -> str:
    """Resolve a remote file path for scp (needs absolute or ~-relative)."""
    cleaned = path.strip()
    if cleaned.startswith("/") or cleaned.startswith("~"):
        return cleaned
    return f"{host.work_directory.rstrip('/')}/{cleaned}"


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def build_remote_tournament_command(
    *,
    tier_a: str,
    tier_b: str,
    board: str,
    games_per_matchup: int,
    output_dir: str,
    output_checkpoint: str,
    output_report: str,
    seed: int,
    think_time_scale: float,
    max_moves: int,
    wilson_confidence: float,
    worker_label: str,
    nn_model_id: Optional[str],
    require_neural_net: bool,
) -> str:
    cmd: List[str] = [
        "python",
        "scripts/run_distributed_tournament.py",
        "--tiers",
        f"{tier_a},{tier_b}",
        "--board",
        board,
        "--games-per-matchup",
        str(int(games_per_matchup)),
        "--workers",
        "1",
        "--output-dir",
        output_dir,
        "--output-checkpoint",
        output_checkpoint,
        "--output-report",
        output_report,
        "--seed",
        str(int(seed)),
        "--think-time-scale",
        str(float(think_time_scale)),
        "--max-moves",
        str(int(max_moves)),
        "--wilson-confidence",
        str(float(wilson_confidence)),
        "--worker-label",
        worker_label,
    ]
    if nn_model_id:
        cmd.extend(["--nn-model-id", nn_model_id])
    if require_neural_net:
        cmd.append("--require-neural-net")
    return _shell_join(cmd)


def _board_to_board_type(board: str) -> BoardType:
    key = board.strip().lower()
    if key == "square8":
        return BoardType.SQUARE8
    if key == "square19":
        return BoardType.SQUARE19
    if key in {"hex", "hexagonal"}:
        return BoardType.HEXAGONAL
    raise ValueError(f"Unknown board: {board!r}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a difficulty-tier tournament across SSH hosts"
    )
    parser.add_argument(
        "--tiers",
        type=str,
        default="D1-D6",
        help="Tier list or range (e.g. D1-D10 or D1,D2,D4) (default: D1-D6).",
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex"],
        help="Board type (default: square8).",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=50,
        help="Games per matchup (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base seed forwarded to shard tournaments (default: 1).",
    )
    parser.add_argument(
        "--think-time-scale",
        type=float,
        default=1.0,
        help="Multiply ladder think_time_ms by this factor (default: 1.0).",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=10000,
        help="Max moves per game before timeout tie-break (default: 10000).",
    )
    parser.add_argument(
        "--wilson-confidence",
        type=float,
        default=0.95,
        help="Wilson CI confidence for decisive matchups (default: 0.95).",
    )
    parser.add_argument(
        "--nn-model-id",
        type=str,
        default=None,
        help="Optional override for the CNN model id used by MCTS/Descent tiers.",
    )
    parser.add_argument(
        "--require-neural-net",
        action="store_true",
        help=(
            "Pass --require-neural-net through to each shard tournament, causing shards "
            "to fail fast if CNN checkpoints cannot be loaded for neural tiers."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to distributed_hosts.yaml (defaults to config/distributed_hosts.yaml).",
    )
    parser.add_argument(
        "--hosts",
        type=str,
        default=None,
        help="Comma-separated host names to use (default: all configured ready hosts).",
    )
    parser.add_argument(
        "--include-nonready",
        action="store_true",
        help="Include hosts whose config status is not 'ready'.",
    )
    parser.add_argument(
        "--max-parallel-per-host",
        type=int,
        default=None,
        help="Optional cap for host max_parallel_jobs from config.",
    )
    parser.add_argument(
        "--remote-output-dir",
        type=str,
        default="results/tournaments/ssh_shards",
        help="Remote output directory (relative to ai-service unless absolute).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tournaments",
        help="Local output directory for aggregated results (default: results/tournaments).",
    )
    parser.add_argument(
        "--job-timeout-sec",
        type=int,
        default=6 * 60 * 60,
        help="Per-shard SSH timeout in seconds (default: 21600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retries per shard on failure (default: 1).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional explicit run id (used for remote shard dir + report naming). "
            "Defaults to a random short uuid."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Optional explicit local output root directory for this run "
            "(writes manifest.json, shards/, tournament_<run-id>.json, report_<run-id>.json). "
            "When unset, uses output-dir/ssh_tournament_<board>_<ts>_<run-id>."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; do not SSH.",
    )
    return parser.parse_args(argv)


def _select_hosts(
    all_hosts: Dict[str, HostConfig],
    *,
    selected: Optional[Iterable[str]],
    include_nonready: bool,
) -> Dict[str, HostConfig]:
    if selected:
        selected_set = {name.strip() for name in selected if name.strip()}
        hosts = {k: v for k, v in all_hosts.items() if k in selected_set}
    else:
        hosts = dict(all_hosts)

    filtered: Dict[str, HostConfig] = {}
    for name, cfg in hosts.items():
        status = str(cfg.properties.get("status") or "").strip().lower()
        if include_nonready or not status or status == "ready":
            filtered[name] = cfg
        else:
            logger.info(f"Skipping host {name} (status={status!r})")

    return filtered


def _build_worker_slots(
    hosts: Dict[str, HostConfig],
    *,
    max_parallel_per_host: Optional[int],
) -> List[WorkerSlot]:
    slots: List[WorkerSlot] = []
    for name, cfg in hosts.items():
        count = int(cfg.max_parallel_jobs or 1)
        if max_parallel_per_host is not None:
            count = max(1, min(count, int(max_parallel_per_host)))
        for idx in range(count):
            slots.append(WorkerSlot(host_name=name, slot_index=idx))
    return slots


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    tiers = parse_tiers_spec(args.tiers)
    matchups = enumerate_matchups(tiers)
    board_type = _board_to_board_type(args.board)

    remote_hosts = load_remote_hosts(args.config)
    selected_names = (
        [part.strip() for part in args.hosts.split(",") if part.strip()]
        if args.hosts
        else None
    )
    hosts = _select_hosts(
        remote_hosts,
        selected=selected_names,
        include_nonready=bool(args.include_nonready),
    )
    if not hosts:
        raise ValueError("No eligible hosts found (check --hosts / config/distributed_hosts.yaml)")

    required_gb = BOARD_MEMORY_REQUIREMENTS.get(
        ("hexagonal" if args.board == "hex" else args.board), 8
    )
    eligible_hosts: Dict[str, HostConfig] = {}
    for name, cfg in hosts.items():
        if cfg.memory_gb is not None and int(cfg.memory_gb) < int(required_gb):
            logger.info(
                f"Skipping host {name} (memory_gb={cfg.memory_gb} < required={required_gb})"
            )
            continue
        eligible_hosts[name] = cfg
    if not eligible_hosts:
        raise ValueError("No hosts meet the board memory requirement")

    worker_slots = _build_worker_slots(
        eligible_hosts, max_parallel_per_host=args.max_parallel_per_host
    )
    if not worker_slots:
        raise ValueError("No worker slots available (max_parallel_jobs=0?)")

    cost_fn = lambda m: estimate_matchup_cost(
        m,
        board_type=board_type,
        think_time_scale=args.think_time_scale,
        games_per_matchup=args.games_per_matchup,
    )
    assignments = assign_matchups_to_worker_slots(matchups, worker_slots, cost_fn)

    run_id = (args.run_id or "").strip() or str(uuid.uuid4())[:8]
    started_at = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime(started_at))

    if args.output_root:
        local_root = Path(args.output_root)
    else:
        local_root = Path(args.output_dir) / f"ssh_tournament_{args.board}_{ts}_{run_id}"
    local_shards_dir = local_root / "shards"
    local_shards_dir.mkdir(parents=True, exist_ok=True)

    remote_run_dir = f"{args.remote_output_dir.rstrip('/')}/{run_id}"

    manifest = {
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "board": args.board,
        "tiers": tiers,
        "games_per_matchup": int(args.games_per_matchup),
        "seed": int(args.seed),
        "think_time_scale": float(args.think_time_scale),
        "max_moves": int(args.max_moves),
        "wilson_confidence": float(args.wilson_confidence),
        "nn_model_id": args.nn_model_id,
        "remote_run_dir": remote_run_dir,
        "hosts": {
            name: {
                "ssh_target": cfg.ssh_target,
                "ssh_port": cfg.ssh_port,
                "max_parallel_jobs": cfg.max_parallel_jobs,
                "memory_gb": cfg.memory_gb,
                "work_directory": cfg.work_directory,
            }
            for name, cfg in eligible_hosts.items()
        },
        "assignments": {
            slot.id: [{"tier_a": a, "tier_b": b} for (a, b) in tasks]
            for slot, tasks in assignments.items()
        },
    }
    with open(local_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info(f"Run id: {run_id}")
    logger.info(f"Matchups: {len(matchups)}")
    logger.info(f"Worker slots: {len(worker_slots)}")
    logger.info(f"Local output: {local_root}")
    logger.info(f"Remote output: {remote_run_dir}")

    if args.dry_run:
        for slot, tasks in assignments.items():
            logger.info(f"{slot.id}: {len(tasks)} matchup(s)")
        return 0

    # Ensure remote output dir exists per host before launching shards.
    for host_name, cfg in eligible_hosts.items():
        executor = SSHExecutor(cfg)
        mkdir_cmd = _shell_join(["mkdir", "-p", remote_run_dir])
        result = executor.run(mkdir_cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create {remote_run_dir!r} on {host_name}: {result.stderr}"
            )

    def run_slot(slot: WorkerSlot) -> List[Path]:
        cfg = eligible_hosts[slot.host_name]
        executor = SSHExecutor(cfg)
        fetched: List[Path] = []
        tasks = assignments.get(slot, [])
        for tier_a, tier_b in tasks:
            matchup_id = f"{tier_a}_vs_{tier_b}"
            checkpoint_rel = f"{remote_run_dir}/{matchup_id}.checkpoint.json"
            report_rel = f"{remote_run_dir}/{matchup_id}.report.json"

            remote_cmd = build_remote_tournament_command(
                tier_a=tier_a,
                tier_b=tier_b,
                board=args.board,
                games_per_matchup=args.games_per_matchup,
                output_dir=remote_run_dir,
                output_checkpoint=checkpoint_rel,
                output_report=report_rel,
                seed=args.seed,
                think_time_scale=args.think_time_scale,
                max_moves=args.max_moves,
                wilson_confidence=args.wilson_confidence,
                worker_label=slot.host_name,
                nn_model_id=args.nn_model_id,
                require_neural_net=args.require_neural_net,
            )

            last_error: Optional[str] = None
            for attempt in range(max(1, int(args.retries))):
                logger.info(f"[{slot.id}] {matchup_id} (attempt {attempt + 1})")
                result = executor.run(remote_cmd, timeout=int(args.job_timeout_sec))
                if result.returncode == 0:
                    last_error = None
                    break
                last_error = (
                    f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
                logger.warning(f"[{slot.id}] {matchup_id} failed: {result.returncode}")

            if last_error is not None:
                raise RuntimeError(f"[{slot.id}] {matchup_id} failed:\n{last_error}")

            remote_checkpoint_abs = resolve_remote_path(cfg, checkpoint_rel)
            local_checkpoint = local_shards_dir / f"{matchup_id}.checkpoint.json"
            scp_result = executor.scp_from(
                remote_checkpoint_abs, str(local_checkpoint), timeout=300
            )
            if scp_result.returncode != 0:
                raise RuntimeError(
                    f"[{slot.id}] scp failed for {matchup_id}: {scp_result.stderr}"
                )
            fetched.append(local_checkpoint)
        return fetched

    # Execute all worker slots concurrently.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    shard_paths: List[Path] = []
    with ThreadPoolExecutor(max_workers=len(worker_slots)) as pool:
        futures = {pool.submit(run_slot, slot): slot for slot in worker_slots}
        for future in as_completed(futures):
            slot = futures[future]
            shard_paths.extend(future.result())
            logger.info(f"[{slot.id}] done")

    # Aggregate checkpoints into one report.
    combined_matches = []
    completed_matchups: List[Matchup] = []
    for path in sorted(shard_paths):
        with open(path, "r", encoding="utf-8") as f:
            state = TournamentState.from_dict(json.load(f))
        combined_matches.extend(state.matches)
        if len(state.tiers) == 2:
            completed_matchups.append((state.tiers[0], state.tiers[1]))

    report_path = local_root / f"report_{run_id}.json"
    checkpoint_path = local_root / f"tournament_{run_id}.json"

    tournament = DistributedTournament(
        tiers=tiers,
        games_per_matchup=args.games_per_matchup,
        board_type=board_type,
        max_workers=1,
        output_dir=str(local_root),
        resume_file=None,
        checkpoint_path=str(checkpoint_path),
        nn_model_id=args.nn_model_id,
        base_seed=args.seed,
        think_time_scale=args.think_time_scale,
        max_moves=args.max_moves,
        confidence=args.wilson_confidence,
        report_path=str(report_path),
        worker_label="aggregate",
    )
    tournament.state.tournament_id = run_id
    tournament.state.started_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)
    )
    tournament.state.matches = combined_matches
    tournament.state.completed_matchups = completed_matchups
    tournament._save_checkpoint()

    duration = time.time() - started_at
    report = tournament.generate_report(duration)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info(f"Report saved: {report_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
