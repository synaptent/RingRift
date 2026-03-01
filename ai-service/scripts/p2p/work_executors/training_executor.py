"""Training work executor - handles GPU model training subprocess.

Extracted from P2POrchestrator._execute_claimed_work (Feb 2026).

Critical fixes preserved:
- Feb 2026: Awaits subprocess completion (not fire-and-forget)
- Feb 2026: Saves to candidate_ (not canonical_) to prevent untested overwrites
- Feb 2026: Parses loss from stdout and populates work_item["result"]
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.managers.job_orchestration_manager import JobOrchestrationManager

logger = logging.getLogger("p2p_orchestrator")


async def _try_fetch_npz_from_cluster(
    ai_service_root: Path, config_key: str, npz_path: Path,
) -> Path | None:
    """Try to fetch an NPZ file from another cluster node via rsync.

    Feb 2026: Training fails on nodes that don't have the NPZ file synced
    from the coordinator. This fetches it on-demand before training starts,
    preventing 100% failure rates for configs like hexagonal_2p/3p.
    """
    try:
        # Get leader URL to find coordinator IP
        from app.config.cluster_config import load_cluster_config
        cluster_cfg = load_cluster_config()
        preferred_leader = cluster_cfg._raw_config.get("preferred_leader", "")
        if not preferred_leader:
            return None

        # Find the coordinator's SSH info
        nodes = cluster_cfg._raw_config.get("nodes", {})
        leader_node = nodes.get(preferred_leader, {})
        leader_ip = leader_node.get("tailscale_ip") or leader_node.get("ip")
        ssh_user = leader_node.get("ssh_user", "ubuntu")
        if not leader_ip:
            return None

        remote_path = f"{ssh_user}@{leader_ip}:ringrift/ai-service/data/training/{config_key}.npz"
        logger.info(f"Fetching NPZ from coordinator: {remote_path}")

        proc = await asyncio.create_subprocess_exec(
            "rsync", "-az", "--timeout=60",
            remote_path, str(npz_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

        if proc.returncode == 0 and npz_path.exists() and npz_path.stat().st_size > 1024:
            logger.info(
                f"Fetched {config_key}.npz from coordinator: "
                f"{npz_path.stat().st_size / 1e6:.1f}MB"
            )
            return npz_path
        else:
            err = stderr.decode()[:200] if stderr else ""
            logger.warning(f"Failed to fetch NPZ from coordinator: rc={proc.returncode} {err}")
            return None
    except Exception as e:
        logger.debug(f"NPZ fetch from cluster failed for {config_key}: {e}")
        return None


async def _try_local_jsonl_export(
    ai_service_root: Path, config_key: str, board_type: str, num_players: int,
) -> Path | None:
    """Try to convert local JSONL selfplay data to NPZ for training.

    Feb 2026: Breaks the circular dependency where:
    - Coordinator needs selfplay data to export NPZ
    - GPU nodes need NPZ to train
    - But selfplay data stays on GPU nodes

    By converting JSONL→NPZ locally, GPU nodes can train on their own
    selfplay data without waiting for the coordinator export cycle.
    """
    import glob

    # Find local JSONL files for this config
    board_norm = board_type.replace("hexagonal", "hex")
    jsonl_dir = ai_service_root / "data" / "selfplay" / "p2p_gpu" / f"{board_norm}_{num_players}p"
    if not jsonl_dir.exists():
        return None

    jsonl_files = glob.glob(str(jsonl_dir / "**" / "*.jsonl"), recursive=True)
    if not jsonl_files:
        return None

    # Only export if we have meaningful data (>= 5 JSONL files)
    if len(jsonl_files) < 5:
        logger.info(f"Only {len(jsonl_files)} JSONL files for {config_key}, skipping local export")
        return None

    output_path = ai_service_root / "data" / "training" / f"{config_key}.npz"
    script_path = ai_service_root / "scripts" / "jsonl_to_npz.py"
    if not script_path.exists():
        logger.warning(f"jsonl_to_npz.py not found at {script_path}")
        return None

    logger.info(f"Converting {len(jsonl_files)} local JSONL files to NPZ for {config_key}")
    cmd = [
        sys.executable, str(script_path),
        "--input-dir", str(jsonl_dir),
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--output", str(output_path),
        "--gpu-selfplay",  # GPU selfplay format (simplified moves)
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ai_service_root),
            env={**__import__("os").environ, "PYTHONPATH": str(ai_service_root)},
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode == 0 and output_path.exists():
            logger.info(f"JSONL→NPZ export succeeded for {config_key}: {output_path}")
            return output_path
        else:
            logger.warning(
                f"JSONL→NPZ export failed for {config_key} (rc={proc.returncode}): "
                f"{stderr.decode()[:500]}"
            )
    except asyncio.TimeoutError:
        logger.warning(f"JSONL→NPZ export timed out for {config_key}")
    except Exception as e:
        logger.warning(f"JSONL→NPZ export error for {config_key}: {e}")
    return None


async def execute_training_work(
    work_item: dict[str, Any],
    config: dict[str, Any],
    node_id: str,
    ringrift_path: str | Path,
    job_orchestration: "JobOrchestrationManager | None" = None,
) -> bool:
    """Execute a training work item as a subprocess.

    Args:
        work_item: Full work item dict (modified in-place to add result data).
        config: Work config sub-dict (board_type, num_players, etc.).
        node_id: This node's identifier.
        ringrift_path: Path to ai-service root (used as subprocess cwd).
        job_orchestration: Optional manager for recording execution metrics.

    Returns:
        True on success, False on failure.
    """
    work_id = work_item.get("work_id", "")

    # Prevent coordinator from running training locally
    from scripts.p2p.managers.work_discovery_manager import _is_training_enabled_for_node
    if not _is_training_enabled_for_node():
        logger.info(f"Skipping training work {work_id}: training_enabled=false for this node")
        return True  # "handled" (just skipped)

    board_type = config.get("board_type", "square8")
    num_players = config.get("num_players", 2)
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 256)
    learning_rate = config.get("learning_rate", 3e-4)
    # Feb 24, 2026: Use preferred architecture per board type as fallback
    try:
        from app.config.thresholds import get_preferred_architecture
        _default_arch = get_preferred_architecture(board_type)
    except ImportError:
        _default_arch = "v2"
    model_version = config.get("model_version") or _default_arch

    config_key = f"{board_type}_{num_players}p"

    # Save to candidate_ instead of canonical_ to prevent overwriting the
    # production model before evaluation confirms improvement.
    if model_version and model_version != "v2":
        model_filename = f"candidate_{config_key}_{model_version}.pth"
    else:
        model_filename = f"candidate_{config_key}.pth"

    # Resolve paths for training data and initial weights.
    # Without --data-path, train.py relies on flaky catalog discovery.
    # Without --init-weights, training starts from random → loss 5-9 (useless).
    ai_service_root = Path(ringrift_path)
    npz_path = ai_service_root / f"data/training/{config_key}.npz"
    canonical_path = ai_service_root / f"models/canonical_{config_key}.pth"

    cmd = [
        sys.executable, "-m", "app.training.train",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--model-version", model_version,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--save-path", f"models/{model_filename}",
        "--allow-stale-data",
        "--max-data-age-hours", "168",  # 7 days tolerance
    ]

    # Validate training data exists before launching subprocess.
    # Without --data-path, train.py relies on flaky catalog discovery that
    # often fails on GPU nodes (missing DBs, stale indexes).
    if npz_path.exists():
        # Reject suspiciously small NPZ files (likely corrupt or partial transfer)
        npz_size = npz_path.stat().st_size
        if npz_size < 1024:
            logger.error(
                f"Training data too small: {npz_path} ({npz_size} bytes). "
                f"Cannot train {config_key}."
            )
            work_item["error"] = f"npz_too_small:{npz_size}bytes:{config_key}"
            return False
        # Feb 2026: Validate NPZ structure before launching training subprocess.
        # Catches corruption from interrupted exports/syncs (3 corrupt files in one session).
        try:
            from app.coordination.npz_validation import quick_npz_check
            _ok, _err = quick_npz_check(npz_path)
            if not _ok:
                logger.error(f"Training data corrupt: {npz_path}: {_err}")
                work_item["error"] = f"npz_corrupt:{config_key}:{_err}"
                return False
        except ImportError:
            pass  # Validation module not available on this node
        cmd.extend(["--data-path", str(npz_path)])
    else:
        # Feb 2026: NPZ not available locally. Try to fetch from another node.
        # 1. Try rsync from coordinator (fastest for large files)
        # 2. Fall back to local JSONL export
        fetched_npz = await _try_fetch_npz_from_cluster(
            ai_service_root, config_key, npz_path
        )
        if fetched_npz and fetched_npz.exists() and fetched_npz.stat().st_size > 1024:
            logger.info(f"Fetched NPZ from cluster: {fetched_npz}")
            cmd.extend(["--data-path", str(fetched_npz)])
        else:
            # Fall back to converting local JSONL selfplay data to NPZ
            jsonl_npz = await _try_local_jsonl_export(ai_service_root, config_key, board_type, num_players)
            if jsonl_npz and jsonl_npz.exists() and jsonl_npz.stat().st_size > 1024:
                logger.info(f"Using locally-exported JSONL→NPZ: {jsonl_npz}")
                cmd.extend(["--data-path", str(jsonl_npz)])
            else:
                logger.error(
                    f"Training data not found: {npz_path}. "
                    f"No local JSONL data available either. Cannot train {config_key}."
                )
                work_item["error"] = f"no_training_data:{config_key}"
                return False

    # Validate init weights exist before launching subprocess.
    # Without --init-weights, training starts from random (loss 5-9 = useless).
    # Only allow from-random during bootstrap when we have very few games.
    #
    # Feb 24, 2026: Skip init_weights when architecture is changing (e.g., v2→v5-heavy).
    # train.py auto-adapts to match init_weights architecture, so passing a v2 model
    # with --model-version v5-heavy would silently revert to v2 training.
    _skip_init = False
    if canonical_path.exists():
        try:
            import torch as _torch
            _ckpt = _torch.load(canonical_path, map_location="cpu", weights_only=True)
            _meta = _ckpt.get("_versioning_metadata", {})
            # Feb 26, 2026: The actual key is "architecture_version", not
            # "model_version". Using the wrong key always returned the default
            # "v2", masking architecture mismatches during transitions.
            _canonical_version = (
                _meta.get("architecture_version")
                or _meta.get("model_version")
                or "v2"
            )
            # Normalize version strings: "v2.0.0" -> "v2"
            if _canonical_version.startswith("v2"):
                _canonical_version = "v2"
            elif _canonical_version.startswith("v5-heavy"):
                _canonical_version = "v5-heavy"
            if _canonical_version != model_version:
                logger.info(
                    f"Architecture upgrade: canonical={_canonical_version}, "
                    f"target={model_version}. Skipping init_weights for {config_key}."
                )
                _skip_init = True
        except Exception:
            pass  # Can't detect, assume compatible
    if canonical_path.exists() and not _skip_init:
        cmd.extend(["--init-weights", str(canonical_path)])
    elif not canonical_path.exists():
        logger.warning(
            f"No canonical model at {canonical_path} — training from scratch"
        )
        # Refuse from-random training when we have significant data,
        # since it wastes GPU hours producing a weak model.
        game_count = config.get("game_count", 0)
        if game_count and game_count > 100:
            logger.error(
                f"Refusing from-random training for {config_key} with "
                f"{game_count} games. A canonical model should exist by now."
            )
            work_item["error"] = f"refusing_from_random:{config_key}:games={game_count}"
            return False

    logger.info(
        f"Executing training work {work_id}: {config_key} with {model_version} "
        f"(epochs={epochs}, batch={batch_size}, "
        f"data={'found' if npz_path.exists() else 'MISSING'}, "
        f"init_weights={'found' if canonical_path.exists() else 'MISSING'})"
    )

    # Await training subprocess and capture results.
    # Previously used asyncio.create_task() (fire-and-forget) which
    # returned True immediately, causing loss=0.0000 and empty model_path.
    # Timeout: 2 hours max. Training typically completes in 2-30 min,
    # but DataLoader/thread hangs caused 12+ hour zombie processes.
    TRAINING_TIMEOUT_SECONDS = 7200  # 2 hours
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(ringrift_path)),
            env={**os.environ, "PYTHONPATH": str(ai_service_root)},
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=TRAINING_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Training subprocess timed out after {TRAINING_TIMEOUT_SECONDS}s "
                f"for {config_key}/{model_version} (work_id={work_id}). Killing."
            )
            proc.kill()
            await proc.wait()
            # Even though it timed out, the model file may already be saved.
            # Check if the model file exists and report partial success.
            model_file = Path(ringrift_path) / "models" / model_filename
            if model_file.exists():
                logger.info(
                    f"Model file exists despite timeout: {model_file} "
                    f"({model_file.stat().st_size / 1e6:.1f}MB). Reporting success."
                )
                work_item["result"] = {
                    "model_path": f"models/{model_filename}",
                    "final_loss": 0.0,
                    "training_samples": 0,
                    "config_key": config_key,
                    "model_version": model_version,
                    "timed_out": True,
                }
                return True
            work_item["error"] = f"training_timeout:{TRAINING_TIMEOUT_SECONDS}s:{config_key}"
            return False
        output = stdout.decode() if stdout else ""

        if proc.returncode == 0:
            # Parse training output for loss, sample count, and game count.
            # Feb 28, 2026: Prefer the structured TRAINING_SUMMARY line that
            # train.py prints at completion. Fallback to regex for older versions.
            final_loss = 0.0
            training_samples = 0
            training_games = 0

            # First try: structured summary line (reliable)
            for line in reversed(output.splitlines()):
                if line.startswith("TRAINING_SUMMARY:"):
                    summary_match = re.search(
                        r'loss=([0-9.]+)\s+samples=(\d+)\s+games=(\d+)', line
                    )
                    if summary_match:
                        final_loss = float(summary_match.group(1))
                        training_samples = int(summary_match.group(2))
                        training_games = int(summary_match.group(3))
                    break

            # Fallback: regex search (for older train.py versions)
            if final_loss == 0.0:
                for line in reversed(output.splitlines()):
                    line_lower = line.lower()
                    if final_loss == 0.0 and "loss" in line_lower and "=" in line_lower:
                        loss_match = re.search(r'loss[=:\s]+([0-9]+\.?[0-9]*)', line_lower)
                        if loss_match:
                            final_loss = float(loss_match.group(1))
                    if final_loss > 0:
                        break

            if training_games == 0 and training_samples > 0:
                avg_moves = 100 if board_type in ("square19", "hexagonal") else 40
                training_games = max(1, training_samples // avg_moves)

            model_path = f"models/{model_filename}"
            candidate_path = Path(ringrift_path) / model_path
            logger.info(
                f"Training completed successfully: {config_key}/{model_version} "
                f"(work_id={work_id}, loss={final_loss:.4f}, samples={training_samples})"
            )

            # Populate work_item result so report_work_result sends real data.
            # Include candidate model path and size so the coordinator can sync
            # the model from this node after work completion.
            work_item["result"] = {
                "model_path": model_path,
                "final_loss": final_loss,
                "training_samples": training_samples,
                "training_games": training_games,
                "config_key": config_key,
                "model_version": model_version,
            }
            if candidate_path.exists():
                work_item["result"]["candidate_model_path"] = str(candidate_path)
                work_item["result"]["candidate_model_size"] = candidate_path.stat().st_size

            # Emit training completed event
            try:
                from app.distributed.data_events import DataEventType
                from app.coordination.event_router import emit_event
                emit_event(DataEventType.TRAINING_COMPLETED, {
                    "config_key": config_key,
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_version": model_version,
                    "model_path": model_path,
                    "final_loss": final_loss,
                    "training_samples": training_samples,
                    "training_games": training_games,
                    "work_id": work_id,
                })
            except ImportError:
                pass
            return True
        else:
            truncated = output[:2000] if output else "no output"
            logger.error(
                f"Training failed: {config_key}/{model_version}: "
                f"returncode={proc.returncode}, output={truncated}"
            )
            work_item["error"] = f"subprocess_failed:rc={proc.returncode}:{config_key}"
            return False
    except Exception as e:
        logger.exception(f"Training subprocess error for {config_key}: {e}")
        work_item["error"] = f"training_exception:{config_key}:{e}"
        return False
