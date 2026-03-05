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


async def _try_push_candidate_to_s3(
    local_path: str, filename: str, config_key: str,
) -> bool:
    """Push candidate model to S3 after training completes.

    Mar 2026: Closes the critical gap where candidate models were stranded
    on workers with no automated path back to the coordinator. The coordinator's
    auto_promotion_daemon polls S3 for new candidates.
    """
    bucket = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    s3_uri = f"s3://{bucket}/consolidated/models/{filename}"
    logger.info(f"Pushing candidate to S3: {local_path} -> {s3_uri}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp", local_path, s3_uri,
            "--region", os.environ.get("AWS_REGION", "us-east-1"),
            "--storage-class", "STANDARD_IA",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode == 0:
            size_mb = Path(local_path).stat().st_size / 1e6
            logger.info(f"S3 candidate push OK: {config_key} ({size_mb:.1f}MB)")
            return True
        err = stderr.decode()[:200] if stderr else ""
        logger.warning(f"S3 candidate push failed for {config_key}: {err}")
        return False
    except Exception as e:
        logger.debug(f"S3 candidate push error for {config_key}: {e}")
        return False


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
        leader_node = cluster_cfg.hosts_raw.get(preferred_leader, {})
        leader_ip = leader_node.get("tailscale_ip") or leader_node.get("ssh_host")
        ssh_user = leader_node.get("ssh_user", "ubuntu")
        ringrift_path = leader_node.get("ringrift_path", "~/ringrift/ai-service")
        if not leader_ip:
            return None

        remote_path = f"{ssh_user}@{leader_ip}:{ringrift_path}/data/training/{config_key}.npz"
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


async def _try_fetch_npz_from_s3(config_key: str, output_path: str) -> bool:
    """Try to fetch an NPZ file from the consolidated S3 bucket.

    Phase 4: S3 as primary data repository. Inserted between rsync-from-coordinator
    and local-JSONL-fallback in the NPZ resolution order.

    Args:
        config_key: e.g. "hex8_2p"
        output_path: Local path to write the NPZ file to.

    Returns:
        True on success, False on failure.
    """
    bucket = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    s3_path = f"s3://{bucket}/consolidated/training/{config_key}.npz"
    logger.info(f"Attempting S3 fetch for NPZ: {s3_path} -> {output_path}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp", s3_path, output_path,
            "--region", os.environ.get("AWS_REGION", "us-east-1"),
            "--cli-read-timeout", "300",
            "--cli-connect-timeout", "30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1024:
            size_mb = Path(output_path).stat().st_size / 1e6
            logger.info(f"S3 NPZ fetch succeeded: {config_key}.npz ({size_mb:.1f}MB)")
            return True
        else:
            err = stderr.decode()[:300] if stderr else ""
            logger.warning(f"S3 NPZ fetch failed for {config_key}: rc={proc.returncode} {err}")
            return False
    except asyncio.TimeoutError:
        logger.warning(f"S3 NPZ fetch timed out for {config_key}")
        return False
    except Exception as e:
        logger.debug(f"S3 NPZ fetch error for {config_key}: {e}")
        return False


async def _try_fetch_model_from_s3(model_filename: str, output_path: str) -> bool:
    """Try to fetch a model checkpoint from the consolidated S3 bucket.

    Phase 4: S3 as primary data repository. Inserted between rsync-from-coordinator
    and random-init-fallback in the model resolution order.

    Args:
        model_filename: e.g. "canonical_hex8_2p.pth"
        output_path: Local path to write the model file to.

    Returns:
        True on success, False on failure.
    """
    bucket = os.environ.get("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
    s3_path = f"s3://{bucket}/consolidated/models/{model_filename}"
    logger.info(f"Attempting S3 fetch for model: {s3_path} -> {output_path}")
    try:
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp", s3_path, output_path,
            "--region", os.environ.get("AWS_REGION", "us-east-1"),
            "--cli-read-timeout", "300",
            "--cli-connect-timeout", "30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        if proc.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1024:
            size_mb = Path(output_path).stat().st_size / 1e6
            logger.info(f"S3 model fetch succeeded: {model_filename} ({size_mb:.1f}MB)")
            return True
        else:
            err = stderr.decode()[:300] if stderr else ""
            logger.warning(f"S3 model fetch failed for {model_filename}: rc={proc.returncode} {err}")
            return False
    except asyncio.TimeoutError:
        logger.warning(f"S3 model fetch timed out for {model_filename}")
        return False
    except Exception as e:
        logger.debug(f"S3 model fetch error for {model_filename}: {e}")
        return False


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


def _partial_architecture_transfer(
    source_path: Path,
    target_version: str,
    config_key: str,
    board_type: str,
    num_players: int,
) -> Path | None:
    """Transfer compatible weights from a source checkpoint to a new architecture.

    When upgrading architectures (e.g., v2 -> v5-heavy), many conv/residual block
    weights share the same spatial dimensions and can be directly transferred.
    Only incompatible layers (e.g., input conv with different channel counts,
    heuristic encoder) are randomly initialized.

    This is better than training from random because the transferred residual
    blocks already encode useful spatial patterns learned from prior training.

    Args:
        source_path: Path to the source checkpoint (old architecture).
        target_version: Target architecture version string (e.g., "v5-heavy").
        config_key: Configuration key (e.g., "hex8_2p").
        board_type: Board type string (e.g., "hex8", "square8").
        num_players: Number of players.

    Returns:
        Path to the saved transfer model, or None on failure.
    """
    try:
        import torch

        # Map version string to memory_tier for model_factory
        version_to_tier = {
            "v2": "high",
            "v3": "v3-high",
            "v4": "v4",
            "v5-heavy": "v5",
            "v5-heavy-large": "v5-heavy-large",
            "v5-heavy-xl": "v5-heavy-xl",
        }
        target_tier = version_to_tier.get(target_version)
        if not target_tier:
            logger.warning(
                f"No tier mapping for target_version={target_version}. "
                f"Cannot perform partial transfer for {config_key}."
            )
            return None

        # Load source checkpoint
        logger.info(
            f"Partial architecture transfer: loading source {source_path} "
            f"for {config_key} (target={target_version})"
        )
        source_ckpt = torch.load(source_path, map_location="cpu", weights_only=True)
        if "state_dict" in source_ckpt:
            source_state = source_ckpt["state_dict"]
        elif "model_state_dict" in source_ckpt:
            source_state = source_ckpt["model_state_dict"]
        else:
            # Checkpoint may be a raw state_dict (filter out metadata keys)
            source_state = {
                k: v for k, v in source_ckpt.items()
                if isinstance(v, torch.Tensor)
            }

        if not source_state:
            logger.warning(f"No state_dict found in source checkpoint {source_path}")
            return None

        # Create target model with the new architecture
        from app.models import BoardType
        from app.ai.neural_net.model_factory import create_model_for_board

        board_type_enum = BoardType(board_type)
        target_model = create_model_for_board(
            board_type=board_type_enum,
            memory_tier=target_tier,
            num_players=num_players,
        )

        # Get target state dict to compare shapes
        target_state = target_model.state_dict()

        # Transfer compatible weights using strict=False
        # This loads any keys that exist in both source and target with matching shapes,
        # and leaves everything else at its random initialization.
        transferred_keys = []
        skipped_keys_shape = []
        skipped_keys_missing = []

        for key in target_state:
            if key in source_state:
                if source_state[key].shape == target_state[key].shape:
                    target_state[key] = source_state[key]
                    transferred_keys.append(key)
                else:
                    skipped_keys_shape.append(
                        f"{key}: source={list(source_state[key].shape)} "
                        f"target={list(target_state[key].shape)}"
                    )
            else:
                skipped_keys_missing.append(key)

        if not transferred_keys:
            logger.warning(
                f"No compatible weights found between source and target for {config_key}. "
                f"Source has {len(source_state)} keys, target has {len(target_state)} keys."
            )
            return None

        # Load the partially-transferred state dict
        target_model.load_state_dict(target_state, strict=True)

        # Save the transfer model
        transfer_filename = f"{config_key}_{target_version}_transfer.pth"
        transfer_path = source_path.parent / transfer_filename

        # Save in the same checkpoint format as the training pipeline expects
        transfer_ckpt = {
            "state_dict": target_model.state_dict(),
            "_versioning_metadata": {
                "architecture_version": target_version,
                "transfer_source": str(source_path.name),
                "transferred_keys": len(transferred_keys),
                "skipped_shape_mismatch": len(skipped_keys_shape),
                "skipped_missing": len(skipped_keys_missing),
            },
        }
        torch.save(transfer_ckpt, transfer_path)

        total_target = len(target_state)
        logger.info(
            f"Partial architecture transfer complete for {config_key}: "
            f"{len(transferred_keys)}/{total_target} layers transferred, "
            f"{len(skipped_keys_shape)} shape mismatches, "
            f"{len(skipped_keys_missing)} new layers (randomly initialized). "
            f"Saved to {transfer_path}"
        )

        # Log details at debug level for troubleshooting
        if skipped_keys_shape:
            logger.debug(
                f"Shape-mismatched layers for {config_key}: "
                + "; ".join(skipped_keys_shape[:10])
                + (f" ... and {len(skipped_keys_shape) - 10} more"
                   if len(skipped_keys_shape) > 10 else "")
            )
        if skipped_keys_missing:
            logger.debug(
                f"New layers (random init) for {config_key}: "
                + "; ".join(skipped_keys_missing[:10])
                + (f" ... and {len(skipped_keys_missing) - 10} more"
                   if len(skipped_keys_missing) > 10 else "")
            )

        return transfer_path

    except Exception as e:
        logger.warning(
            f"Partial architecture transfer failed for {config_key} "
            f"({source_path} -> {target_version}): {e}"
        )
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
        work_item["error"] = f"training_disabled:{node_id}"
        return False  # Return False so work item gets reassigned to a GPU node

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
    # Feb 28, 2026: Minimum NPZ size thresholds. A stale local NPZ with only
    # 68 samples (few KB) produces garbage models (loss 4+). Worker nodes may
    # have old copies that pass the exists() check but are far too small for
    # meaningful training. If below threshold, fetch a fresh copy.
    MIN_NPZ_SIZE_BYTES = 5 * 1024 * 1024  # 5MB — even smallest configs have 40MB+

    if npz_path.exists():
        npz_size = npz_path.stat().st_size
        if npz_size < 1024:
            logger.error(
                f"Training data too small: {npz_path} ({npz_size} bytes). "
                f"Cannot train {config_key}."
            )
            work_item["error"] = f"npz_too_small:{npz_size}bytes:{config_key}"
            return False
        # If local NPZ exists but is suspiciously small, try fetching a fresh copy
        if npz_size < MIN_NPZ_SIZE_BYTES:
            logger.warning(
                f"Local NPZ small ({npz_size / 1e6:.1f}MB < {MIN_NPZ_SIZE_BYTES / 1e6:.0f}MB). "
                f"Fetching fresh copy for {config_key}."
            )
            fetched = await _try_fetch_npz_from_cluster(
                ai_service_root, config_key, npz_path
            )
            if fetched and fetched.exists() and fetched.stat().st_size > npz_size:
                logger.info(
                    f"Replaced stale NPZ ({npz_size / 1e6:.1f}MB) with fresh "
                    f"({fetched.stat().st_size / 1e6:.1f}MB) for {config_key}"
                )
            elif await _try_fetch_npz_from_s3(config_key, str(npz_path)):
                # Phase 4: S3 fallback for small/stale NPZ
                new_size = npz_path.stat().st_size if npz_path.exists() else 0
                if new_size > npz_size:
                    logger.info(
                        f"Replaced stale NPZ ({npz_size / 1e6:.1f}MB) with S3 copy "
                        f"({new_size / 1e6:.1f}MB) for {config_key}"
                    )
        # Validate NPZ structure before launching training subprocess.
        try:
            from app.coordination.npz_validation import quick_npz_check
            _ok, _err = quick_npz_check(npz_path)
            if not _ok:
                logger.warning(
                    f"Local NPZ corrupt: {npz_path}: {_err}. "
                    f"Fetching fresh copy for {config_key}."
                )
                # Mar 4, 2026: Try to replace corrupt file before failing.
                # Before this fix, corrupt NPZs caused permanent training failures.
                _fetched = await _try_fetch_npz_from_cluster(
                    ai_service_root, config_key, npz_path
                )
                if not _fetched:
                    await _try_fetch_npz_from_s3(config_key, str(npz_path))
                # Re-validate after fetch attempt
                _ok2, _err2 = quick_npz_check(npz_path)
                if not _ok2:
                    logger.error(f"Training data still corrupt after re-fetch: {_err2}")
                    work_item["error"] = f"npz_corrupt:{config_key}:{_err2}"
                    return False
                logger.info(f"Successfully replaced corrupt NPZ for {config_key}")
        except ImportError:
            pass  # Validation module not available on this node
        cmd.extend(["--data-path", str(npz_path)])
    else:
        # Feb 2026: NPZ not available locally. Try to fetch from another node.
        # Resolution order: 1. rsync from coordinator -> 2. S3 fetch -> 3. local JSONL
        fetched_npz = await _try_fetch_npz_from_cluster(
            ai_service_root, config_key, npz_path
        )
        if fetched_npz and fetched_npz.exists() and fetched_npz.stat().st_size > 1024:
            logger.info(f"Fetched NPZ from cluster: {fetched_npz}")
            cmd.extend(["--data-path", str(fetched_npz)])
        elif await _try_fetch_npz_from_s3(config_key, str(npz_path)):
            # Phase 4: S3 as primary data repository fallback
            logger.info(f"Fetched NPZ from S3: {npz_path}")
            cmd.extend(["--data-path", str(npz_path)])
        else:
            # Fall back to converting local JSONL selfplay data to NPZ
            jsonl_npz = await _try_local_jsonl_export(ai_service_root, config_key, board_type, num_players)
            if jsonl_npz and jsonl_npz.exists() and jsonl_npz.stat().st_size > 1024:
                logger.info(f"Using locally-exported JSONL->NPZ: {jsonl_npz}")
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
    # Feb 24, 2026: Detect architecture mismatch (e.g., v2→v5-heavy).
    # train.py auto-adapts to match init_weights architecture, so passing a v2 model
    # with --model-version v5-heavy would silently revert to v2 training.
    # Mar 2026: Instead of skipping init_weights entirely, perform partial weight
    # transfer — conv/residual blocks share spatial dimensions between v2 and v5-heavy,
    # so transferring compatible weights is better than training from random.
    _skip_init = False
    _transfer_path: Path | None = None
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
                    f"target={model_version}. Attempting partial weight transfer for {config_key}."
                )
                _transfer_path = _partial_architecture_transfer(
                    source_path=canonical_path,
                    target_version=model_version,
                    config_key=config_key,
                    board_type=board_type,
                    num_players=num_players,
                )
                if _transfer_path:
                    logger.info(
                        f"Using partial transfer model as init_weights for {config_key}: "
                        f"{_transfer_path}"
                    )
                else:
                    logger.warning(
                        f"Partial transfer failed for {config_key}. "
                        f"Falling through to from-scratch logic."
                    )
                _skip_init = True
        except Exception:
            pass  # Can't detect, assume compatible
    if _transfer_path:
        cmd.extend(["--init-weights", str(_transfer_path)])
    elif canonical_path.exists() and not _skip_init:
        cmd.extend(["--init-weights", str(canonical_path)])
    elif not canonical_path.exists():
        # Phase 4: Try fetching canonical model from S3 before giving up
        canonical_filename = canonical_path.name
        if await _try_fetch_model_from_s3(canonical_filename, str(canonical_path)):
            logger.info(f"Fetched canonical model from S3: {canonical_path}")
            cmd.extend(["--init-weights", str(canonical_path)])
        else:
            logger.warning(
                f"No canonical model at {canonical_path} (local or S3) — training from scratch"
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
    # Mar 5, 2026: Board-aware timeouts. Large boards (square19=361 cells,
    # hexagonal=469 cells) with 50 epochs can exceed 2h on GH200s.
    # 681 consecutive Lambda timeouts were caused by 2h being too short.
    _LARGE_BOARDS = ("square19", "hexagonal")
    TRAINING_TIMEOUT_SECONDS = 14400 if board_type in _LARGE_BOARDS else 7200
    try:
        # Mar 4, 2026: expandable_segments reduces VRAM fragmentation on nodes
        # where selfplay processes already hold large VRAM allocations.
        _training_env = {
            **os.environ,
            "PYTHONPATH": str(ai_service_root),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RINGRIFT_ALLOW_PENDING_GATE": "true",
        }
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(ringrift_path)),
            env=_training_env,
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
                # Mar 5, 2026: Estimate training_samples from the NPZ file used
                # for training. Previously reported 0, causing ALL 2,414+ generations
                # to record training_samples=0 in generation_tracking.db.
                # Extract actual --data-path from cmd (may differ from npz_path
                # when fetched from cluster or converted from JSONL).
                _timeout_samples = 0
                _timeout_loss = 0.0
                _actual_data_path = npz_path
                try:
                    _dp_idx = cmd.index("--data-path")
                    _actual_data_path = Path(cmd[_dp_idx + 1])
                except (ValueError, IndexError):
                    pass
                try:
                    import numpy as _np
                    _npz_data = _np.load(str(_actual_data_path), mmap_mode='r')
                    if 'boards' in _npz_data:
                        _timeout_samples = len(_npz_data['boards'])
                    elif 'states' in _npz_data:
                        _timeout_samples = len(_npz_data['states'])
                    _npz_data.close()
                except Exception as _e:
                    logger.debug(f"Could not read NPZ for sample count: {_e}")
                _timeout_games = 0
                if _timeout_samples > 0:
                    avg_moves = 100 if board_type in _LARGE_BOARDS else 40
                    _timeout_games = max(1, _timeout_samples // avg_moves)
                logger.info(
                    f"Model file exists despite timeout: {model_file} "
                    f"({model_file.stat().st_size / 1e6:.1f}MB). "
                    f"Reporting success with estimated samples={_timeout_samples}."
                )
                work_item["result"] = {
                    "model_path": f"models/{model_filename}",
                    "final_loss": _timeout_loss,
                    "training_samples": _timeout_samples,
                    "training_games": _timeout_games,
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

            # Mar 2026: Push candidate model to S3 so coordinator can discover it.
            # Workers produce candidates locally but the coordinator needs them for
            # evaluation/promotion. S3 bridges the gap without direct SSH.
            if candidate_path.exists():
                await _try_push_candidate_to_s3(
                    str(candidate_path), model_filename, config_key
                )

            return True
        else:
            truncated = output[:2000] if output else "no output"
            logger.error(
                f"Training failed: {config_key}/{model_version}: "
                f"returncode={proc.returncode}, output={truncated}"
            )
            # Mar 4, 2026: Include OOM/CUDA keywords in error string so the
            # coordinator's retryable check can detect transient GPU failures.
            # Previously, "subprocess_failed:rc=1" was never retried because
            # it didn't contain "cuda" or "out of memory".
            if "out of memory" in output.lower() or "outofmemoryerror" in output.lower():
                work_item["error"] = f"cuda_oom:rc={proc.returncode}:{config_key}"
            elif "cuda" in output.lower() and proc.returncode == 1:
                work_item["error"] = f"cuda_error:rc={proc.returncode}:{config_key}"
            else:
                work_item["error"] = f"subprocess_failed:rc={proc.returncode}:{config_key}"
            return False
    except Exception as e:
        logger.exception(f"Training subprocess error for {config_key}: {e}")
        work_item["error"] = f"training_exception:{config_key}:{e}"
        return False
