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
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.p2p.managers.job_orchestration_manager import JobOrchestrationManager

logger = logging.getLogger("p2p_orchestrator")


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
    model_version = config.get("model_version", "v2")

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

    # Point training at the NPZ file so it doesn't need catalog discovery
    if npz_path.exists():
        cmd.extend(["--data-path", str(npz_path)])
    else:
        logger.warning(
            f"Training data not found: {npz_path} — "
            f"train.py will attempt catalog discovery"
        )

    # Continue from canonical model weights instead of random initialization
    if canonical_path.exists():
        cmd.extend(["--init-weights", str(canonical_path)])
    else:
        logger.warning(
            f"No canonical model at {canonical_path} — training from scratch"
        )

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
            return False
        output = stdout.decode() if stdout else ""

        if proc.returncode == 0:
            # Parse training output for loss and sample count.
            # Search all lines (not just until first loss match) so we
            # can find both metrics regardless of line order.
            final_loss = 0.0
            training_samples = 0
            for line in reversed(output.splitlines()):
                line_lower = line.lower()
                if final_loss == 0.0 and "loss" in line_lower and "=" in line_lower:
                    loss_match = re.search(r'loss[=:\s]+([0-9]+\.?[0-9]*)', line_lower)
                    if loss_match:
                        final_loss = float(loss_match.group(1))
                if training_samples == 0 and "samples" in line_lower:
                    samples_match = re.search(r'(\d+)\s*samples', line_lower)
                    if samples_match:
                        training_samples = int(samples_match.group(1))
                if final_loss > 0 and training_samples > 0:
                    break

            model_path = f"models/{model_filename}"
            logger.info(
                f"Training completed successfully: {config_key}/{model_version} "
                f"(work_id={work_id}, loss={final_loss:.4f}, samples={training_samples})"
            )

            # Populate work_item result so report_work_result sends real data
            work_item["result"] = {
                "model_path": model_path,
                "final_loss": final_loss,
                "training_samples": training_samples,
                "config_key": config_key,
                "model_version": model_version,
            }

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
            return False
    except Exception as e:
        logger.exception(f"Training subprocess error for {config_key}: {e}")
        return False
