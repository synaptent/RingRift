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

# Import dynamic gauntlet games threshold (December 2025)
# Higher player counts need more games for statistical significance
try:
    from app.config.thresholds import get_gauntlet_games_per_opponent
except ImportError:
    # Fallback if running outside proper Python path
    def get_gauntlet_games_per_opponent(num_players: int = 2) -> int:
        """Fallback: 50 for 2p, 75 for 3p, 100 for 4p."""
        if num_players >= 4:
            return 100
        if num_players == 3:
            return 75
        return 50

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
    """Extract board_type and num_players from model filename.

    Handles patterns like:
    - hex8_2p_cluster.pth
    - square8_3p_trained_cluster.pth
    - hexagonal_4p_gh200_trained.pth
    - canonical_hex8_2p.pth
    - ringrift_best_square8_4p.pth

    Returns (board_type, num_players) or (None, None) if not parseable.
    """
    import re
    name = Path(model_path).stem

    # Primary pattern: board_type followed by _Np (player count)
    # Board types: hex8, square8, square19, hexagonal
    # Player counts: 2p, 3p, 4p
    match = re.search(r'(hex8|square8|square19|hexagonal)_(\d+)p', name)
    if match:
        board_type = match.group(1)
        try:
            num_players = int(match.group(2))
            if num_players in (2, 3, 4):
                return board_type, num_players
        except (ValueError, TypeError):
            pass

    # Fallback: try extracting from canonical/ringrift_best naming
    # e.g., canonical_hex8_2p, ringrift_best_square8_4p
    match = re.search(r'(?:canonical|ringrift_best)_(hex8|square8|square19|hexagonal)_(\d+)p', name)
    if match:
        board_type = match.group(1)
        try:
            num_players = int(match.group(2))
            if num_players in (2, 3, 4):
                return board_type, num_players
        except (ValueError, TypeError):
            pass

    # Log for debugging but don't crash
    logger.debug(f"Could not parse config from filename: {name}")
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

    # December 2025: Use dynamic gauntlet games based on player count
    # 3p/4p have higher variance, requiring more games for statistical significance
    gauntlet_games = get_gauntlet_games_per_opponent(num_players)
    logger.info(f"  Running gauntlet with {gauntlet_games} games/opponent ({num_players}p config)")

    try:
        result = subprocess.run(
            [
                sys.executable, "scripts/auto_promote.py",
                "--gauntlet",
                "--model", str(model_path),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--games", str(gauntlet_games),
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
                    try:
                        # Check if model is complete (file not being written)
                        size1 = model_file.stat().st_size
                        await asyncio.sleep(2)
                        size2 = model_file.stat().st_size

                        if size1 == size2 and size1 > 1000000:  # >1MB and stable
                            if promote_model(model_file):
                                promoted_models.add(model_file.name)
                    except FileNotFoundError:
                        logger.debug(f"Model file removed during processing: {model_file}")
                    except OSError as e:
                        logger.warning(f"Error processing {model_file}: {e}")
            
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
