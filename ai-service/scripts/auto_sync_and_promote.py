#!/usr/bin/env python3
"""Auto-sync models from cluster and promote if they pass gauntlet.

Runs as a background daemon on the coordinator.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/auto_sync_promote.log"),
    ]
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
SYNC_INTERVAL = 300  # 5 minutes
CLUSTER_NODES = [
    ("nebius-h100-3", "ubuntu", "89.169.111.139", 22, "~/ringrift/ai-service"),
    ("nebius-backbone-1", "ubuntu", "89.169.112.47", 22, "~/ringrift/ai-service"),
    ("vultr-a100", "root", "208.167.249.164", 22, "/root/ringrift/ai-service"),
]


def sync_models():
    """Sync models from all cluster nodes."""
    synced = []
    for name, user, host, port, remote_path in CLUSTER_NODES:
        logger.info(f"Syncing from {name}...")
        try:
            result = subprocess.run(
                [
                    "rsync", "-avz",
                    "-e", f"ssh -i ~/.ssh/id_cluster -p {port} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                    f"{user}@{host}:{remote_path}/models/*_cluster.pth",
                    str(MODELS_DIR) + "/",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                # Parse rsync output for synced files
                for line in result.stdout.split('\n'):
                    if '_cluster.pth' in line and not line.startswith('sending'):
                        synced.append(line.strip())
                logger.info(f"  ✅ {name} sync complete")
            else:
                logger.warning(f"  ⚠️ {name}: {result.stderr[:100]}")
        except Exception as e:
            logger.error(f"  ❌ {name} error: {e}")
    return synced


def get_config_from_model(model_path: str) -> tuple:
    """Extract board_type and num_players from model filename."""
    name = Path(model_path).stem
    # Format: hex8_2p_cluster or square8_3p_cluster
    parts = name.replace("_cluster", "").split("_")
    if len(parts) >= 2:
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))
        return board_type, num_players
    return None, None


def promote_model(model_path: Path):
    """Promote a cluster model to canonical if it passes gauntlet."""
    board_type, num_players = get_config_from_model(str(model_path))
    if not board_type:
        logger.warning(f"Could not parse config from {model_path}")
        return False
    
    config_key = f"{board_type}_{num_players}p"
    canonical_path = MODELS_DIR / f"canonical_{config_key}.pth"
    
    logger.info(f"Evaluating {model_path} for promotion...")
    
    # Run quick gauntlet (10 games vs random, 10 vs heuristic)
    try:
        result = subprocess.run(
            [
                sys.executable, "scripts/auto_promote.py",
                "--gauntlet",
                "--model", str(model_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", "10",
            ],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).parent.parent),
            env={**os.environ, "PYTHONPATH": "."},
        )
        
        if "PROMOTED" in result.stdout or result.returncode == 0:
            # Copy to canonical
            import shutil
            shutil.copy(model_path, canonical_path)
            logger.info(f"  ✅ Promoted {config_key} to canonical")
            return True
        else:
            logger.info(f"  ⚠️ {config_key} did not pass gauntlet")
            return False
    except Exception as e:
        logger.error(f"  ❌ Gauntlet error: {e}")
        return False


async def main():
    """Main sync and promote loop."""
    logger.info("Starting auto-sync-and-promote daemon")
    
    promoted_models = set()
    
    while True:
        try:
            # Sync models from cluster
            synced = sync_models()
            
            # Check for new models to evaluate
            for model_file in MODELS_DIR.glob("*_cluster.pth"):
                if model_file.name not in promoted_models:
                    # Check if model is complete (file not being written)
                    size1 = model_file.stat().st_size
                    await asyncio.sleep(2)
                    size2 = model_file.stat().st_size
                    
                    if size1 == size2 and size1 > 1000000:  # >1MB and stable
                        if promote_model(model_file):
                            promoted_models.add(model_file.name)
            
            logger.info(f"Sleeping {SYNC_INTERVAL}s until next sync...")
            await asyncio.sleep(SYNC_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
