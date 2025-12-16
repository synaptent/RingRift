#!/usr/bin/env python3
"""Distributed Export Infrastructure for parallel game dataset generation.

This module provides worker-based distributed export that:
1. Splits games into chunks for parallel processing across nodes
2. Uses HTTP serving for NAT-friendly result collection
3. Supports aria2 for resilient multi-source downloads
4. Merges NPZ chunks into final datasets

Architecture:
    Coordinator (this machine)
        |
        +-- SSH to Node 1: export --chunk 0/8 -> chunk_0.npz (served via HTTP)
        +-- SSH to Node 2: export --chunk 1/8 -> chunk_1.npz (served via HTTP)
        +-- ...
        +-- SSH to Node 8: export --chunk 7/8 -> chunk_7.npz (served via HTTP)
        |
        +-- aria2c download all chunks from all nodes
        |
        +-- Merge chunks into final dataset

Usage:
    # On worker nodes (started by coordinator):
    python distributed_export.py worker \
        --db /path/to/db.db \
        --board-type hexagonal --num-players 3 \
        --chunk-index 0 --total-chunks 8 \
        --output /tmp/export_chunk_0.npz \
        --serve --port 8780

    # On coordinator:
    python distributed_export.py coordinate \
        --config export_config.yaml \
        --output data/training/hex_3p.npz

    # Merge chunks locally:
    python distributed_export.py merge \
        --chunks chunk_0.npz chunk_1.npz chunk_2.npz \
        --output merged.npz
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import yaml

# Add ai-service to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

DEFAULT_HTTP_PORT = 8780  # Different from model sync port (8765)
ARIA2_CONNECTIONS = 16
ARIA2_TIMEOUT = 300  # 5 min timeout for large chunks
CHUNK_DIR = "/tmp/ringrift_export_chunks"


# ============================================
# Data Classes
# ============================================

@dataclass
class ExportConfig:
    """Configuration for distributed export."""
    db_paths: List[str]
    board_type: str
    num_players: int
    output_path: str
    total_chunks: int = 8
    history_length: int = 3
    min_moves: Optional[int] = 10
    max_moves: Optional[int] = None
    require_completed: bool = True
    encoder_version: str = "default"
    hosts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WorkerStatus:
    """Status of a worker node."""
    host_name: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    chunk_index: int
    status: str  # pending, running, completed, failed
    http_port: int
    output_file: Optional[str] = None
    samples: int = 0
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


# ============================================
# Game ID Extraction for Chunking
# ============================================

def get_game_ids_from_db(
    db_path: str,
    board_type: str,
    num_players: int,
    require_completed: bool = True,
    min_moves: Optional[int] = None,
) -> List[str]:
    """Extract game IDs from database matching criteria."""
    import sqlite3

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Build query
    query = """
        SELECT DISTINCT g.game_id
        FROM games g
        WHERE g.board_type = ?
        AND g.num_players = ?
    """
    params: List[Any] = [board_type, num_players]

    if require_completed:
        query += " AND g.game_status = 'completed'"

    if min_moves is not None:
        query += " AND g.total_moves >= ?"
        params.append(min_moves)

    # Only include games with move data
    query += """
        AND EXISTS (
            SELECT 1 FROM game_moves m WHERE m.game_id = g.game_id
        )
    """

    cursor.execute(query, params)
    game_ids = [row["game_id"] for row in cursor.fetchall()]
    conn.close()

    return game_ids


def split_game_ids_into_chunks(
    game_ids: List[str],
    total_chunks: int,
) -> List[List[str]]:
    """Split game IDs into roughly equal chunks."""
    # Shuffle for even distribution of game lengths
    shuffled = game_ids.copy()
    random.seed(42)  # Reproducible shuffle
    random.shuffle(shuffled)

    chunks = [[] for _ in range(total_chunks)]
    for i, game_id in enumerate(shuffled):
        chunks[i % total_chunks].append(game_id)

    return chunks


# ============================================
# Worker: Export + HTTP Serve
# ============================================

def run_worker_export(
    db_paths: List[str],
    board_type: str,
    num_players: int,
    game_ids: List[str],
    output_path: str,
    history_length: int = 3,
    encoder_version: str = "default",
) -> int:
    """Run export for specific game IDs (worker mode)."""
    from app.db import GameReplayDB
    from app.models import BoardType

    # Import the export function components
    from scripts.export_replay_dataset import (
        build_encoder,
        encode_state_with_history,
        value_from_final_ranking,
        compute_multi_player_values,
        BOARD_TYPE_MAP,
    )
    from app.ai.neural_net import INVALID_MOVE_INDEX
    from app.game_engine import GameEngine

    board_type_enum = BOARD_TYPE_MAP[board_type]
    encoder = build_encoder(board_type_enum, encoder_version)

    # Convert game_ids to set for fast lookup
    game_id_set = set(game_ids)

    features_list: List[np.ndarray] = []
    globals_list: List[np.ndarray] = []
    values_list: List[float] = []
    values_mp_list: List[np.ndarray] = []
    num_players_list: List[int] = []
    policy_indices_list: List[np.ndarray] = []
    policy_values_list: List[np.ndarray] = []
    move_numbers_list: List[int] = []
    total_game_moves_list: List[int] = []
    phases_list: List[str] = []

    games_processed = 0
    games_scanned = 0
    games_matched = 0

    for db_path in db_paths:
        if not os.path.exists(db_path):
            logger.warning(f"DB path does not exist: {db_path}")
            continue

        logger.info(f"Opening database: {db_path}")
        db = GameReplayDB(db_path)

        for meta, initial_state, moves in db.iterate_games(
            board_type=board_type_enum,
            num_players=num_players,
            require_moves=True,
        ):
            games_scanned += 1
            game_id = meta.get("game_id")

            # Only process games in our assigned chunk
            if game_id not in game_id_set:
                if games_scanned <= 5:
                    logger.debug(f"Skipping game {game_id} - not in chunk")
                continue

            games_matched += 1
            if games_matched <= 3:
                logger.info(f"Matched game {game_id}, moves={len(moves) if moves else 0}")

            total_moves = meta.get("total_moves")
            if total_moves is None:
                total_moves = len(moves) if moves else 0
            total_moves = int(total_moves)

            if total_moves <= 0 or not moves:
                continue

            num_players_in_game = len(initial_state.players)

            # Incremental replay
            game_samples = []
            history_frames: List[np.ndarray] = []
            current_state = initial_state
            replay_succeeded = True

            for move_index, move in enumerate(moves):
                state_before = current_state

                try:
                    current_state = GameEngine.apply_move(current_state, move, trace_mode=True)
                except Exception:
                    replay_succeeded = False
                    break

                stacked, globals_vec = encode_state_with_history(
                    encoder, state_before, history_frames, history_length=history_length
                )

                hex_encoder = getattr(encoder, "_hex_encoder", None)
                if hex_encoder is not None:
                    base_features, _ = hex_encoder.encode_state(state_before)
                else:
                    base_features, _ = encoder._extract_features(state_before)
                history_frames.append(base_features)
                if len(history_frames) > history_length + 1:
                    history_frames.pop(0)

                idx = encoder.encode_move(move, state_before.board)
                if idx == INVALID_MOVE_INDEX:
                    continue

                phase_str = (
                    str(state_before.current_phase.value)
                    if hasattr(state_before.current_phase, "value")
                    else str(state_before.current_phase)
                )
                game_samples.append((
                    stacked, globals_vec, idx, state_before.current_player,
                    move_index, phase_str
                ))

            if not replay_succeeded or not game_samples:
                continue

            final_state = current_state
            values_vec = np.asarray(
                compute_multi_player_values(final_state, num_players=num_players_in_game),
                dtype=np.float32,
            )

            for stacked, globals_vec, idx, perspective, move_index, phase_str in game_samples:
                value = value_from_final_ranking(
                    final_state, perspective=perspective, num_players=num_players
                )

                features_list.append(stacked)
                globals_list.append(globals_vec)
                values_list.append(float(value))
                policy_indices_list.append(np.array([idx], dtype=np.int32))
                policy_values_list.append(np.array([1.0], dtype=np.float32))
                values_mp_list.append(values_vec)
                num_players_list.append(num_players_in_game)
                move_numbers_list.append(move_index)
                total_game_moves_list.append(total_moves)
                phases_list.append(phase_str)

            games_processed += 1
            if games_processed % 10 == 0:
                logger.info(f"Processed {games_processed}/{len(game_ids)} games, {len(features_list)} samples")

    logger.info(f"Scan complete: scanned={games_scanned}, matched={games_matched}, processed={games_processed}")

    if not features_list:
        logger.warning(f"No samples generated (matched {games_matched} games but processed {games_processed})")
        return 0

    # Save chunk
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    np.savez_compressed(
        output_path,
        features=np.stack(features_list, axis=0).astype(np.float32),
        globals=np.stack(globals_list, axis=0).astype(np.float32),
        values=np.array(values_list, dtype=np.float32),
        policy_indices=np.array(policy_indices_list, dtype=object),
        policy_values=np.array(policy_values_list, dtype=object),
        values_mp=np.stack(values_mp_list, axis=0).astype(np.float32),
        num_players=np.array(num_players_list, dtype=np.int32),
        board_type=np.asarray(board_type),
        move_numbers=np.array(move_numbers_list, dtype=np.int32),
        total_game_moves=np.array(total_game_moves_list, dtype=np.int32),
        phases=np.array(phases_list, dtype=object),
    )

    logger.info(f"Saved {len(features_list)} samples from {games_processed} games to {output_path}")
    return len(features_list)


def _mp_worker_task(args: Tuple) -> Tuple[int, str]:
    """Multiprocessing worker task - processes a sub-chunk of games."""
    (
        worker_id,
        db_paths,
        board_type,
        num_players,
        game_ids,
        output_path,
        history_length,
        encoder_version,
    ) = args

    # Configure logging for this process
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - Worker {worker_id} - %(levelname)s - %(message)s",
    )
    proc_logger = logging.getLogger(__name__)

    proc_logger.info(f"Starting with {len(game_ids)} games, board={board_type}, players={num_players}")
    proc_logger.info(f"DB paths: {db_paths}")
    proc_logger.info(f"First 3 game IDs: {game_ids[:3]}")

    try:
        samples = run_worker_export(
            db_paths=db_paths,
            board_type=board_type,
            num_players=num_players,
            game_ids=game_ids,
            output_path=output_path,
            history_length=history_length,
            encoder_version=encoder_version,
        )
        proc_logger.info(f"Completed with {samples} samples")
        return samples, output_path
    except Exception as e:
        import traceback
        proc_logger.error(f"Failed - {e}")
        proc_logger.error(traceback.format_exc())
        return 0, output_path


def run_worker_export_parallel(
    db_paths: List[str],
    board_type: str,
    num_players: int,
    game_ids: List[str],
    output_dir: str,
    num_workers: int = 4,
    history_length: int = 3,
    encoder_version: str = "default",
) -> Tuple[int, List[str]]:
    """
    Run export in parallel using multiple local processes.

    Splits game_ids across num_workers processes, each producing a chunk file.
    Returns total samples and list of chunk file paths.
    """
    if num_workers < 1:
        num_workers = 1

    # Split games across workers
    chunks = split_game_ids_into_chunks(game_ids, num_workers)

    # Prepare worker arguments
    os.makedirs(output_dir, exist_ok=True)
    worker_args = []
    for i, chunk_game_ids in enumerate(chunks):
        if not chunk_game_ids:
            continue
        output_path = os.path.join(output_dir, f"subchunk_{i}.npz")
        worker_args.append((
            i,
            db_paths,
            board_type,
            num_players,
            chunk_game_ids,
            output_path,
            history_length,
            encoder_version,
        ))

    logger.info(f"Starting {len(worker_args)} parallel workers for {len(game_ids)} games")

    # Run workers in parallel using multiprocessing
    total_samples = 0
    chunk_files = []

    # Use spawn to avoid fork issues with numpy/torch
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(_mp_worker_task, worker_args)

    for samples, output_path in results:
        total_samples += samples
        if samples > 0 and os.path.exists(output_path):
            chunk_files.append(output_path)

    logger.info(f"Parallel export complete: {total_samples} samples in {len(chunk_files)} chunks")
    return total_samples, chunk_files


class ChunkHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for serving export chunks."""

    def __init__(self, *args, chunk_dir: Path, **kwargs):
        self.chunk_dir = chunk_dir
        super().__init__(*args, directory=str(chunk_dir), **kwargs)

    def do_GET(self):
        if self.path == "/status":
            self.send_status()
        elif self.path == "/inventory.json":
            self.send_inventory()
        else:
            super().do_GET()

    def send_status(self):
        """Send worker status."""
        status = {"status": "ready", "files": list(self.chunk_dir.glob("*.npz"))}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(status, default=str).encode())

    def send_inventory(self):
        """Send inventory of available chunks."""
        inventory = {}
        for f in self.chunk_dir.glob("*.npz"):
            inventory[f.name] = {
                "size": f.stat().st_size,
                "mtime": f.stat().st_mtime,
            }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(inventory, indent=2).encode())

    def log_message(self, format, *args):
        logger.debug(f"HTTP: {args[0]}")


def serve_chunks(chunk_dir: Path, port: int):
    """Serve export chunks over HTTP."""
    handler = lambda *args, **kwargs: ChunkHTTPHandler(*args, chunk_dir=chunk_dir, **kwargs)

    with socketserver.TCPServer(("0.0.0.0", port), handler) as httpd:
        logger.info(f"Serving chunks from {chunk_dir} on port {port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


# ============================================
# Coordinator: Distribute Work + Collect Results
# ============================================

def load_hosts_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load hosts configuration."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Default to distributed_hosts.yaml
    default_path = ROOT / "config" / "distributed_hosts.yaml"
    if default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f)

    return {"hosts": {}}


def select_worker_hosts(
    hosts_config: Dict[str, Any],
    num_workers: int,
    prefer_high_cpu: bool = True,
) -> List[Dict[str, Any]]:
    """Select best hosts for export workers."""
    hosts = hosts_config.get("hosts", {})

    candidates = []
    for name, config in hosts.items():
        if config.get("status") not in ("ready", "active"):
            continue

        # Prefer hosts with tailscale or direct IP
        ssh_host = config.get("tailscale_ip") or config.get("ssh_host")
        if not ssh_host:
            continue

        candidates.append({
            "name": name,
            "ssh_host": ssh_host,
            "ssh_port": config.get("ssh_port", 22),
            "ssh_user": config.get("ssh_user", "root"),
            "ssh_key": config.get("ssh_key"),
            "ringrift_path": config.get("ringrift_path", "~/ringrift/ai-service"),
            "cpus": config.get("cpus", 1),
            "memory_gb": config.get("memory_gb", 0),
        })

    # Sort by CPU count if preferring high CPU
    if prefer_high_cpu:
        candidates.sort(key=lambda h: h["cpus"], reverse=True)

    return candidates[:num_workers]


def run_ssh_command(
    host: Dict[str, Any],
    command: str,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """Run SSH command on remote host."""
    ssh_cmd = ["ssh"]

    if host.get("ssh_key"):
        key_path = os.path.expanduser(host["ssh_key"])
        ssh_cmd.extend(["-i", key_path])

    ssh_cmd.extend([
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(host["ssh_port"]),
        f"{host['ssh_user']}@{host['ssh_host']}",
        command,
    ])

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "SSH timeout"
    except Exception as e:
        return False, str(e)


def sync_db_to_host(
    host: Dict[str, Any],
    local_db_path: str,
    remote_db_dir: str = "/tmp/ringrift_export",
) -> Tuple[bool, str]:
    """Sync database file to remote host using rsync."""
    remote_path = f"{remote_db_dir}/{os.path.basename(local_db_path)}"

    # Create remote directory
    run_ssh_command(host, f"mkdir -p {remote_db_dir}", timeout=30)

    # Rsync the database
    rsync_cmd = ["rsync", "-avz", "--progress"]

    if host.get("ssh_key"):
        key_path = os.path.expanduser(host["ssh_key"])
        rsync_cmd.extend(["-e", f"ssh -i {key_path} -p {host['ssh_port']} -o StrictHostKeyChecking=no"])
    else:
        rsync_cmd.extend(["-e", f"ssh -p {host['ssh_port']} -o StrictHostKeyChecking=no"])

    rsync_cmd.extend([
        local_db_path,
        f"{host['ssh_user']}@{host['ssh_host']}:{remote_path}",
    ])

    try:
        result = subprocess.run(
            rsync_cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min for large DBs
        )
        if result.returncode == 0:
            return True, remote_path
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def start_worker_on_host(
    host: Dict[str, Any],
    db_paths: List[str],
    board_type: str,
    num_players: int,
    game_ids: List[str],
    chunk_index: int,
    http_port: int,
    encoder_version: str = "default",
) -> Tuple[bool, str]:
    """Start export worker on remote host."""
    ringrift_path = host["ringrift_path"]

    # Write game IDs to temp file on remote
    game_ids_json = json.dumps(game_ids)
    game_ids_path = f"/tmp/ringrift_export/game_ids_{chunk_index}.json"

    # Create game IDs file
    write_cmd = f"mkdir -p /tmp/ringrift_export && echo '{game_ids_json}' > {game_ids_path}"
    success, output = run_ssh_command(host, write_cmd, timeout=60)
    if not success:
        return False, f"Failed to write game IDs: {output}"

    # Build worker command
    db_args = " ".join(f"--db {p}" for p in db_paths)
    worker_cmd = f"""
cd {ringrift_path} && \\
nohup bash -c '
export PYTHONPATH={ringrift_path}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python scripts/distributed_export.py worker \\
    {db_args} \\
    --board-type {board_type} \\
    --num-players {num_players} \\
    --game-ids-file {game_ids_path} \\
    --output /tmp/ringrift_export/chunk_{chunk_index}.npz \\
    --serve --port {http_port} \\
    --encoder-version {encoder_version}
' > /tmp/ringrift_export/worker_{chunk_index}.log 2>&1 &
echo $!
"""

    success, output = run_ssh_command(host, worker_cmd, timeout=60)
    if success:
        return True, output.strip()
    return False, output


def collect_chunks_with_aria2(
    sources: List[Tuple[str, str]],  # [(url, filename), ...]
    output_dir: Path,
) -> Tuple[int, int]:
    """Collect chunk files using aria2."""
    if not shutil.which("aria2c"):
        logger.error("aria2c not found. Install with: apt install aria2")
        return 0, len(sources)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create input file for aria2
    input_file = output_dir / "aria2_input.txt"
    with open(input_file, "w") as f:
        for url, filename in sources:
            f.write(f"{url}\n")
            f.write(f"  out={filename}\n")

    cmd = [
        "aria2c",
        "--input-file", str(input_file),
        "--dir", str(output_dir),
        "--max-connection-per-server", str(ARIA2_CONNECTIONS),
        "--split", "4",
        "--continue", "true",
        "--timeout", str(ARIA2_TIMEOUT),
        "--max-tries", "5",
        "--retry-wait", "5",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Count successes
        downloaded = sum(1 for _, f in sources if (output_dir / f).exists())
        failed = len(sources) - downloaded

        return downloaded, failed
    except Exception as e:
        logger.error(f"aria2 failed: {e}")
        return 0, len(sources)


# ============================================
# Merge Chunks
# ============================================

def merge_npz_chunks(
    chunk_paths: List[str],
    output_path: str,
) -> int:
    """Merge multiple NPZ chunks into single dataset."""
    all_features = []
    all_globals = []
    all_values = []
    all_policy_indices = []
    all_policy_values = []
    all_values_mp = []
    all_num_players = []
    all_move_numbers = []
    all_total_game_moves = []
    all_phases = []

    board_type = None

    for chunk_path in chunk_paths:
        if not os.path.exists(chunk_path):
            logger.warning(f"Chunk not found: {chunk_path}")
            continue

        with np.load(chunk_path, allow_pickle=True) as data:
            if "features" not in data:
                logger.warning(f"Invalid chunk (no features): {chunk_path}")
                continue

            all_features.append(data["features"])
            all_globals.append(data["globals"])
            all_values.append(data["values"])
            all_policy_indices.append(data["policy_indices"])
            all_policy_values.append(data["policy_values"])

            if "values_mp" in data:
                all_values_mp.append(data["values_mp"])
            if "num_players" in data:
                all_num_players.append(data["num_players"])
            if "move_numbers" in data:
                all_move_numbers.append(data["move_numbers"])
            if "total_game_moves" in data:
                all_total_game_moves.append(data["total_game_moves"])
            if "phases" in data:
                all_phases.append(data["phases"])

            if board_type is None and "board_type" in data:
                board_type = data["board_type"]

            logger.info(f"Loaded {len(data['features'])} samples from {chunk_path}")

    if not all_features:
        logger.error("No valid chunks to merge")
        return 0

    # Concatenate all arrays
    save_kwargs = {
        "features": np.concatenate(all_features, axis=0),
        "globals": np.concatenate(all_globals, axis=0),
        "values": np.concatenate(all_values, axis=0),
        "policy_indices": np.concatenate(all_policy_indices, axis=0),
        "policy_values": np.concatenate(all_policy_values, axis=0),
    }

    if all_values_mp:
        save_kwargs["values_mp"] = np.concatenate(all_values_mp, axis=0)
    if all_num_players:
        save_kwargs["num_players"] = np.concatenate(all_num_players, axis=0)
    if all_move_numbers:
        save_kwargs["move_numbers"] = np.concatenate(all_move_numbers, axis=0)
    if all_total_game_moves:
        save_kwargs["total_game_moves"] = np.concatenate(all_total_game_moves, axis=0)
    if all_phases:
        save_kwargs["phases"] = np.concatenate(all_phases, axis=0)
    if board_type is not None:
        save_kwargs["board_type"] = board_type

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **save_kwargs)

    total_samples = len(save_kwargs["features"])
    logger.info(f"Merged {total_samples} samples from {len(chunk_paths)} chunks into {output_path}")
    return total_samples


# ============================================
# CLI Commands
# ============================================

def cmd_worker(args):
    """Worker command: export assigned games and optionally serve."""
    # Load game IDs
    if args.game_ids_file:
        with open(args.game_ids_file) as f:
            game_ids = json.load(f)
    elif args.game_ids:
        game_ids = args.game_ids.split(",")
    else:
        # If no specific game IDs, export all (chunk mode not supported here)
        logger.error("Must specify --game-ids or --game-ids-file for worker mode")
        return 1

    logger.info(f"Worker starting: {len(game_ids)} games to process")

    num_workers = getattr(args, 'workers', 1) or 1
    output_path = args.output
    chunk_dir = Path(output_path).parent

    if num_workers > 1:
        # Use parallel processing within this node
        logger.info(f"Using {num_workers} parallel workers per node")
        samples, chunk_files = run_worker_export_parallel(
            db_paths=args.db_paths,
            board_type=args.board_type,
            num_players=args.num_players,
            game_ids=game_ids,
            output_dir=str(chunk_dir / "subchunks"),
            num_workers=num_workers,
            history_length=args.history_length,
            encoder_version=args.encoder_version,
        )

        # Merge subchunks into final output
        if chunk_files:
            logger.info(f"Merging {len(chunk_files)} subchunks...")
            samples = merge_npz_chunks(chunk_files, output_path)
    else:
        # Single-threaded export
        samples = run_worker_export(
            db_paths=args.db_paths,
            board_type=args.board_type,
            num_players=args.num_players,
            game_ids=game_ids,
            output_path=output_path,
            history_length=args.history_length,
            encoder_version=args.encoder_version,
        )

    logger.info(f"Export complete: {samples} samples")

    # Optionally serve the output
    if args.serve:
        serve_chunks(chunk_dir, args.port)

    return 0


def cmd_coordinate(args):
    """Coordinator command: distribute work and collect results."""
    # Load configuration
    hosts_config = load_hosts_config(args.hosts_config)

    # Parse database paths
    db_paths = args.db_paths

    # Get all game IDs
    logger.info("Scanning databases for game IDs...")
    all_game_ids = []
    for db_path in db_paths:
        if os.path.exists(db_path):
            ids = get_game_ids_from_db(
                db_path,
                args.board_type,
                args.num_players,
                require_completed=True,
                min_moves=args.min_moves,
            )
            logger.info(f"  {db_path}: {len(ids)} games")
            all_game_ids.extend(ids)

    # Deduplicate
    all_game_ids = list(set(all_game_ids))
    logger.info(f"Total unique games: {len(all_game_ids)}")

    if not all_game_ids:
        logger.error("No games found matching criteria")
        return 1

    # Select worker hosts
    num_chunks = min(args.chunks, len(all_game_ids))
    workers = select_worker_hosts(hosts_config, num_chunks)

    if not workers:
        logger.error("No available worker hosts")
        return 1

    logger.info(f"Selected {len(workers)} worker hosts")

    # Split games into chunks
    chunks = split_game_ids_into_chunks(all_game_ids, len(workers))

    # Sync databases to workers
    logger.info("Syncing databases to workers...")
    remote_db_paths = {}
    for worker in workers:
        remote_paths = []
        for db_path in db_paths:
            success, remote_path = sync_db_to_host(worker, db_path)
            if success:
                remote_paths.append(remote_path)
            else:
                logger.warning(f"Failed to sync {db_path} to {worker['name']}: {remote_path}")
        remote_db_paths[worker["name"]] = remote_paths

    # Start workers
    logger.info("Starting workers...")
    worker_statuses = []
    base_port = 8780

    for i, (worker, game_ids) in enumerate(zip(workers, chunks)):
        http_port = base_port + i

        logger.info(f"  {worker['name']}: {len(game_ids)} games, port {http_port}")

        remote_dbs = remote_db_paths.get(worker["name"], [])
        if not remote_dbs:
            logger.warning(f"  No databases available on {worker['name']}, skipping")
            continue

        success, output = start_worker_on_host(
            worker,
            remote_dbs,
            args.board_type,
            args.num_players,
            game_ids,
            i,
            http_port,
            args.encoder_version,
        )

        if success:
            worker_statuses.append(WorkerStatus(
                host_name=worker["name"],
                ssh_host=worker["ssh_host"],
                ssh_port=worker["ssh_port"],
                ssh_user=worker["ssh_user"],
                chunk_index=i,
                status="running",
                http_port=http_port,
            ))
        else:
            logger.error(f"  Failed to start worker on {worker['name']}: {output}")

    if not worker_statuses:
        logger.error("No workers started successfully")
        return 1

    # Wait for workers and collect results
    logger.info("Waiting for workers to complete...")
    logger.info("(Monitor progress by checking worker logs or /status endpoints)")

    # Poll for completion
    completed = set()
    max_wait = 3600 * 4  # 4 hours
    start_time = time.time()

    while len(completed) < len(worker_statuses) and (time.time() - start_time) < max_wait:
        time.sleep(60)  # Check every minute

        for ws in worker_statuses:
            if ws.chunk_index in completed:
                continue

            # Check if chunk file exists via HTTP
            try:
                url = f"http://{ws.ssh_host}:{ws.http_port}/inventory.json"
                with urlopen(url, timeout=10) as response:
                    inv = json.loads(response.read().decode())
                    if f"chunk_{ws.chunk_index}.npz" in inv:
                        logger.info(f"  {ws.host_name}: chunk ready")
                        completed.add(ws.chunk_index)
            except Exception:
                pass

        logger.info(f"Progress: {len(completed)}/{len(worker_statuses)} chunks complete")

    # Collect chunks via aria2
    logger.info("Collecting chunks via aria2...")
    output_dir = Path(args.output).parent / "chunks"

    sources = []
    for ws in worker_statuses:
        if ws.chunk_index in completed:
            url = f"http://{ws.ssh_host}:{ws.http_port}/chunk_{ws.chunk_index}.npz"
            sources.append((url, f"chunk_{ws.chunk_index}.npz"))

    downloaded, failed = collect_chunks_with_aria2(sources, output_dir)
    logger.info(f"Downloaded {downloaded} chunks, {failed} failed")

    # Merge chunks
    if downloaded > 0:
        chunk_files = sorted(output_dir.glob("chunk_*.npz"))
        total = merge_npz_chunks([str(f) for f in chunk_files], args.output)
        logger.info(f"Final dataset: {total} samples in {args.output}")

    return 0


def cmd_merge(args):
    """Merge command: combine NPZ chunks."""
    total = merge_npz_chunks(args.chunks, args.output)
    return 0 if total > 0 else 1


def cmd_split(args):
    """Split command: extract game IDs and create chunk assignment."""
    all_game_ids = []
    for db_path in args.db_paths:
        if os.path.exists(db_path):
            ids = get_game_ids_from_db(
                db_path,
                args.board_type,
                args.num_players,
                require_completed=True,
                min_moves=args.min_moves,
            )
            logger.info(f"  {db_path}: {len(ids)} games")
            all_game_ids.extend(ids)

    all_game_ids = list(set(all_game_ids))
    logger.info(f"Total unique games: {len(all_game_ids)}")

    chunks = split_game_ids_into_chunks(all_game_ids, args.chunks)

    # Save chunk assignments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, game_ids in enumerate(chunks):
        chunk_file = output_dir / f"chunk_{i}_game_ids.json"
        with open(chunk_file, "w") as f:
            json.dump(game_ids, f)
        logger.info(f"  Chunk {i}: {len(game_ids)} games -> {chunk_file}")

    # Save summary
    summary = {
        "total_games": len(all_game_ids),
        "num_chunks": len(chunks),
        "board_type": args.board_type,
        "num_players": args.num_players,
        "chunks": [{"index": i, "games": len(c)} for i, c in enumerate(chunks)],
    }
    with open(output_dir / "chunk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Export Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Worker command
    worker_parser = subparsers.add_parser("worker", help="Run export worker")
    worker_parser.add_argument("--db", action="append", dest="db_paths", required=True)
    worker_parser.add_argument("--board-type", required=True)
    worker_parser.add_argument("--num-players", type=int, required=True)
    worker_parser.add_argument("--game-ids", help="Comma-separated game IDs")
    worker_parser.add_argument("--game-ids-file", help="JSON file with game IDs")
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--history-length", type=int, default=3)
    worker_parser.add_argument("--encoder-version", default="default")
    worker_parser.add_argument("--workers", type=int, default=1,
                               help="Number of parallel workers within this node (default: 1)")
    worker_parser.add_argument("--serve", action="store_true", help="Serve output via HTTP")
    worker_parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)

    # Coordinate command
    coord_parser = subparsers.add_parser("coordinate", help="Coordinate distributed export")
    coord_parser.add_argument("--db", action="append", dest="db_paths", required=True)
    coord_parser.add_argument("--board-type", required=True)
    coord_parser.add_argument("--num-players", type=int, required=True)
    coord_parser.add_argument("--output", required=True)
    coord_parser.add_argument("--chunks", type=int, default=8)
    coord_parser.add_argument("--min-moves", type=int, default=10)
    coord_parser.add_argument("--encoder-version", default="default")
    coord_parser.add_argument("--hosts-config", help="Path to hosts YAML config")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge NPZ chunks")
    merge_parser.add_argument("--chunks", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)

    # Split command (for manual distributed runs)
    split_parser = subparsers.add_parser("split", help="Split games into chunks for manual distribution")
    split_parser.add_argument("--db", action="append", dest="db_paths", required=True)
    split_parser.add_argument("--board-type", required=True)
    split_parser.add_argument("--num-players", type=int, required=True)
    split_parser.add_argument("--chunks", type=int, default=8)
    split_parser.add_argument("--min-moves", type=int, default=10)
    split_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.command == "worker":
        return cmd_worker(args)
    elif args.command == "coordinate":
        return cmd_coordinate(args)
    elif args.command == "merge":
        return cmd_merge(args)
    elif args.command == "split":
        return cmd_split(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
