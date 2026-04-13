#!/usr/bin/env python3
"""Validate and convert policy-bearing selfplay JSONL into supplemental NPZ shards.

This is the safe replacement for dropping ad hoc ``iter_p2p_*.npz`` files into a
trainer work directory. It preserves trainer-owned iteration files while making
worker-produced Gumbel data available through an explicit supplemental path.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from scripts.jsonl_to_npz import convert_jsonl_to_npz
from scripts.lib.data_quality_sentinel import compute_fingerprint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_policy_selfplay")


@dataclass
class IngestSummary:
    board_type: str
    num_players: int
    files_seen: int
    files_processed: int
    games_seen: int
    games_kept: int
    duplicates_skipped: int
    policy_target_moves: int
    completion_rate: float
    policy_entropy_mean: float
    output_npz: str
    fingerprint_entropy_median: float
    fingerprint_value_std: float
    provenance: dict[str, Any]


def _policy_entropy_bits(mcts_policy: dict[str, float]) -> float:
    probs = [float(prob) for prob in mcts_policy.values() if float(prob) > 0]
    if len(probs) <= 1:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for prob in probs:
        normalized = prob / total
        entropy -= normalized * np.log2(normalized)
    return max(float(entropy), 0.0)


def _load_index(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _append_index(path: Path, values: Iterable[str]) -> None:
    ordered = [value for value in values if value]
    if not ordered:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for value in ordered:
            handle.write(value + "\n")


def _iter_input_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            files.append(input_path)
            continue
        if input_path.is_dir():
            files.extend(sorted(input_path.rglob("*.jsonl")))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def _normalize_policy_target(move: dict[str, Any]) -> bool:
    if "policy_target" in move:
        return bool(move.get("policy_target"))
    return bool(move.get("mcts_policy"))


def _augment_npz_with_manifest(npz_path: Path, manifest: dict[str, Any]) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["supplemental_manifest_json"] = np.asarray(json.dumps(manifest, sort_keys=True))
    payload["supplemental_source"] = np.asarray("policy-gumbel")
    payload["supplemental_config_key"] = np.asarray(f"{manifest['board_type']}_{manifest['num_players']}p")
    payload["supplemental_model_sha"] = np.asarray(str(manifest["provenance"].get("model_sha", "")))
    payload["supplemental_node_ids"] = np.asarray(
        ",".join(sorted(str(v) for v in manifest["provenance"].get("node_ids", [])))
    )
    payload["supplemental_opponent_types"] = np.asarray(
        ",".join(sorted(str(v) for v in manifest["provenance"].get("opponent_types", [])))
    )
    np.savez_compressed(npz_path, **payload)


def _sync_to_remote(
    *,
    files: list[Path],
    remote_host: str,
    remote_dir: str,
    remote_user: str,
    remote_key: str,
    remote_port: int,
) -> None:
    ssh_base = [
        "ssh",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        remote_key,
        "-p",
        str(remote_port),
    ]
    subprocess.run(
        [*ssh_base, f"{remote_user}@{remote_host}", f"mkdir -p {remote_dir}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for file_path in files:
        subprocess.run(
            [
                "scp",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-i",
                remote_key,
                "-P",
                str(remote_port),
                str(file_path),
                f"{remote_user}@{remote_host}:{remote_dir}/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def ingest_policy_selfplay_files(
    *,
    input_paths: list[Path],
    output_dir: Path,
    state_dir: Path,
    board_type: str,
    num_players: int,
    policy_entropy_threshold: float = 0.5,
    completion_rate_threshold: float = 0.95,
    min_value_std: float = 1e-6,
    remote_host: str = "",
    remote_dir: str = "",
    remote_user: str = "ubuntu",
    remote_key: str = "",
    remote_port: int = 22,
) -> IngestSummary:
    files = _iter_input_files(input_paths)
    seen_game_ids = _load_index(state_dir / "seen_game_ids.txt")
    processed_sources = _load_index(state_dir / "processed_sources.txt")

    accepted_records: list[dict[str, Any]] = []
    new_game_ids: list[str] = []
    processed_sources_now: list[str] = []
    games_seen = 0
    duplicates_skipped = 0
    policy_target_moves = 0
    completed_games = 0
    policy_entropies: list[float] = []
    provenance: dict[str, set[str]] = {
        "node_ids": set(),
        "model_shas": set(),
        "opponent_types": set(),
        "engine_modes": set(),
    }

    for file_path in files:
        resolved_source = str(file_path.resolve())
        if resolved_source in processed_sources:
            continue

        records_added = 0
        with open(file_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                games_seen += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("board_type") != board_type or int(record.get("num_players", 0) or 0) != num_players:
                    continue
                moves = record.get("moves", [])
                if not isinstance(moves, list) or not moves:
                    continue
                game_id = str(record.get("game_id", "")).strip()
                if not game_id:
                    continue
                if game_id in seen_game_ids:
                    duplicates_skipped += 1
                    continue

                record_policy_moves = 0
                for move in moves:
                    if not isinstance(move, dict):
                        continue
                    policy_target = _normalize_policy_target(move)
                    mcts_policy = move.get("mcts_policy")
                    if not (policy_target and isinstance(mcts_policy, dict) and len(mcts_policy) > 1):
                        move["policy_target"] = False
                        continue
                    move["policy_target"] = True
                    record_policy_moves += 1
                    policy_target_moves += 1
                    policy_entropies.append(_policy_entropy_bits(mcts_policy))

                if record_policy_moves == 0:
                    continue

                record_provenance = record.get("provenance", {}) if isinstance(record.get("provenance"), dict) else {}
                for key, target in (
                    ("node_id", "node_ids"),
                    ("model_sha", "model_shas"),
                    ("opponent_type", "opponent_types"),
                    ("engine_mode", "engine_modes"),
                ):
                    value = str(record_provenance.get(key, "")).strip()
                    if value:
                        provenance[target].add(value)

                if str(record.get("status", "")).lower() == "completed":
                    completed_games += 1

                accepted_records.append(record)
                new_game_ids.append(game_id)
                seen_game_ids.add(game_id)
                records_added += 1

        if records_added > 0:
            processed_sources_now.append(resolved_source)

    if not accepted_records:
        raise ValueError("No policy-bearing selfplay records matched the requested config")

    completion_rate = completed_games / len(accepted_records)
    policy_entropy_mean = sum(policy_entropies) / len(policy_entropies) if policy_entropies else 0.0

    if completion_rate < completion_rate_threshold:
        raise ValueError(
            f"Completion rate too low: {completion_rate:.3f} < {completion_rate_threshold:.3f}"
        )
    if policy_entropy_mean < policy_entropy_threshold:
        raise ValueError(
            f"Policy entropy too low: {policy_entropy_mean:.3f} < {policy_entropy_threshold:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    filtered_jsonl = state_dir / f"filtered_{board_type}_{num_players}p_{timestamp}.jsonl"
    filtered_jsonl.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in accepted_records),
        encoding="utf-8",
    )

    output_npz = output_dir / f"supplemental_{board_type}_{num_players}p_{timestamp}.npz"
    stats = convert_jsonl_to_npz(
        [filtered_jsonl],
        output_npz,
        board_type,
        players_filter=num_players,
        gpu_selfplay_mode=True,
    )
    if stats.positions_extracted <= 0 or not output_npz.exists():
        raise ValueError("JSONL conversion produced no usable training samples")

    fingerprint = compute_fingerprint(str(output_npz))
    if fingerprint.value_std <= min_value_std:
        raise ValueError(f"Degenerate value targets: std={fingerprint.value_std:.6g}")

    manifest = {
        "board_type": board_type,
        "num_players": num_players,
        "files_seen": len(files),
        "files_processed": len(processed_sources_now),
        "games_seen": games_seen,
        "games_kept": len(accepted_records),
        "duplicates_skipped": duplicates_skipped,
        "policy_target_moves": policy_target_moves,
        "completion_rate": completion_rate,
        "policy_entropy_mean": policy_entropy_mean,
        "fingerprint_entropy_median": fingerprint.policy_entropy_median,
        "fingerprint_value_std": fingerprint.value_std,
        "provenance": {
            "node_ids": sorted(provenance["node_ids"]),
            "model_sha": sorted(provenance["model_shas"])[0] if len(provenance["model_shas"]) == 1 else "",
            "model_shas": sorted(provenance["model_shas"]),
            "opponent_types": sorted(provenance["opponent_types"]),
            "engine_modes": sorted(provenance["engine_modes"]),
        },
    }
    _augment_npz_with_manifest(output_npz, manifest)

    summary_path = output_npz.with_suffix(".meta.json")
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if remote_host and remote_dir and remote_key:
        _sync_to_remote(
            files=[output_npz, summary_path],
            remote_host=remote_host,
            remote_dir=remote_dir,
            remote_user=remote_user,
            remote_key=remote_key,
            remote_port=remote_port,
        )

    _append_index(state_dir / "seen_game_ids.txt", new_game_ids)
    _append_index(state_dir / "processed_sources.txt", processed_sources_now)

    return IngestSummary(
        board_type=board_type,
        num_players=num_players,
        files_seen=len(files),
        files_processed=len(processed_sources_now),
        games_seen=games_seen,
        games_kept=len(accepted_records),
        duplicates_skipped=duplicates_skipped,
        policy_target_moves=policy_target_moves,
        completion_rate=completion_rate,
        policy_entropy_mean=policy_entropy_mean,
        output_npz=str(output_npz),
        fingerprint_entropy_median=fingerprint.policy_entropy_median,
        fingerprint_value_std=fingerprint.value_std,
        provenance=manifest["provenance"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest policy-bearing selfplay JSONL into supplemental NPZ shards")
    parser.add_argument("--input", action="append", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output-dir", required=True, help="Directory for supplemental NPZ outputs")
    parser.add_argument("--state-dir", required=True, help="Directory for dedupe/processing state")
    parser.add_argument("--board-type", required=True, choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--num-players", required=True, type=int, choices=[2, 3, 4])
    parser.add_argument("--policy-entropy-threshold", type=float, default=0.5)
    parser.add_argument("--completion-rate-threshold", type=float, default=0.95)
    parser.add_argument("--min-value-std", type=float, default=1e-6)
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--remote-dir", default="")
    parser.add_argument("--remote-user", default="ubuntu")
    parser.add_argument("--remote-key", default="")
    parser.add_argument("--remote-port", type=int, default=22)
    args = parser.parse_args()

    summary = ingest_policy_selfplay_files(
        input_paths=[Path(value) for value in args.input],
        output_dir=Path(args.output_dir),
        state_dir=Path(args.state_dir),
        board_type=args.board_type,
        num_players=args.num_players,
        policy_entropy_threshold=args.policy_entropy_threshold,
        completion_rate_threshold=args.completion_rate_threshold,
        min_value_std=args.min_value_std,
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        remote_user=args.remote_user,
        remote_key=args.remote_key,
        remote_port=args.remote_port,
    )
    logger.info("Ingested supplemental shard: %s", json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
