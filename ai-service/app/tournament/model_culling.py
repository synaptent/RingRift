"""Per-config model culling to manage model proliferation.

When the number of models for a specific config (e.g., square8_2p) exceeds
a threshold (default 100), this module archives the bottom 50% based on
Elo rating, keeping the top half.

This prevents unbounded model growth while preserving the best performers.

After culling, automatically triggers cluster-wide Elo sync to propagate
the archive markers to all nodes.

Usage:
    from app.tournament.model_culling import ModelCullingController

    culler = ModelCullingController(elo_db_path, model_dir)

    # Check and cull if needed
    result = culler.check_and_cull("square8_2p")
    print(f"Culled {result.culled} models, kept {result.kept}")
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# All 12 config keys
CONFIG_KEYS = [
    "hex8_2p", "hex8_3p", "hex8_4p",
    "square8_2p", "square8_3p", "square8_4p",
    "square19_2p", "square19_3p", "square19_4p",
    "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
]


@dataclass
class ModelInfo:
    """Information about a model for culling decisions."""

    model_id: str
    filename: str
    elo: float
    games_played: int
    config_key: str


@dataclass
class CullResult:
    """Result of a culling operation."""

    config_key: str
    culled: int
    kept: int
    archived_models: list[str]
    preserved_models: list[str]
    timestamp: float


class ModelCullingController:
    """Controls per-config model culling to keep top half.

    When model count exceeds CULL_THRESHOLD for a config, archives bottom
    50% of models by Elo rating. Models are moved to archived/ subdirectory
    and marked in the database.

    Protections:
    - Always keeps at least MIN_KEEP models
    - Never culls baselines (top 4 by Elo)
    - Never culls models with < MIN_GAMES_FOR_CULL games (high uncertainty)
    - Archives to recoverable location, doesn't delete
    """

    CULL_THRESHOLD = 100  # Trigger culling when > 100 models
    KEEP_FRACTION = 0.50  # Keep top 50%
    MIN_KEEP = 25         # Always keep at least 25 models
    MIN_GAMES_FOR_CULL = 20  # Don't cull models with < 20 games (high uncertainty)

    def __init__(
        self,
        elo_db_path: Path,
        model_dir: Path,
        cull_threshold: int = 100,
        keep_fraction: float = 0.25,
    ):
        """Initialize culling controller.

        Args:
            elo_db_path: Path to unified Elo database
            model_dir: Directory containing model .pth files
            cull_threshold: Number of models before culling triggers
            keep_fraction: Fraction of models to keep (0.25 = top 25%)
        """
        # Convert to Path objects if strings were passed
        self.elo_db_path = Path(elo_db_path) if isinstance(elo_db_path, str) else elo_db_path
        self.model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
        self.CULL_THRESHOLD = cull_threshold
        self.KEEP_FRACTION = keep_fraction

        # Track last cull time per config
        self._last_cull: dict[str, float] = {}
        self._cull_cooldown = 3600  # 1 hour between culls per config

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.elo_db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_archive_columns(self) -> None:
        """Ensure elo_ratings has archive tracking columns."""
        conn = self._get_db_connection()
        try:
            # Check existing columns
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}

            # Add archived_at if missing
            if "archived_at" not in columns:
                conn.execute("""
                    ALTER TABLE elo_ratings ADD COLUMN archived_at REAL
                """)

            # Add archive_reason if missing
            if "archive_reason" not in columns:
                conn.execute("""
                    ALTER TABLE elo_ratings ADD COLUMN archive_reason TEXT
                """)

            conn.commit()
        except sqlite3.OperationalError as e:
            # Column might already exist in some edge cases
            if "duplicate column" not in str(e).lower():
                raise
        finally:
            conn.close()

    def get_models_for_config(self, config_key: str) -> list[ModelInfo]:
        """Get all models for a config, sorted by Elo descending.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            List of ModelInfo, highest Elo first
        """
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        conn = self._get_db_connection()
        try:
            # Determine column name
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            # Get non-archived models
            archived_filter = ""
            if "archived_at" in columns:
                archived_filter = "AND (archived_at IS NULL OR archived_at = 0)"

            cursor = conn.execute(f"""
                SELECT {id_col} as model_id, rating, games_played
                FROM elo_ratings
                WHERE board_type = ? AND num_players = ?
                {archived_filter}
                ORDER BY rating DESC
            """, (board_type, num_players))

            models = []
            for row in cursor.fetchall():
                # Construct filename from model_id
                model_id = row["model_id"]
                filename = f"{model_id}.pth" if not model_id.endswith(".pth") else model_id

                models.append(ModelInfo(
                    model_id=model_id,
                    filename=filename,
                    elo=row["rating"],
                    games_played=row["games_played"],
                    config_key=config_key,
                ))

            return models
        finally:
            conn.close()

    def count_models(self, config_key: str) -> int:
        """Count models for a config."""
        return len(self.get_models_for_config(config_key))

    def needs_culling(self, config_key: str) -> bool:
        """Check if config needs culling.

        Args:
            config_key: Config to check

        Returns:
            True if model count > threshold and cooldown elapsed
        """
        # Check cooldown
        last = self._last_cull.get(config_key, 0)
        if time.time() - last < self._cull_cooldown:
            return False

        return self.count_models(config_key) > self.CULL_THRESHOLD

    def check_and_cull(self, config_key: str) -> CullResult:
        """Check if culling needed and perform if so.

        Args:
            config_key: Config to check/cull

        Returns:
            CullResult with operation details
        """
        models = self.get_models_for_config(config_key)
        model_count = len(models)

        # Not enough to cull
        if model_count <= self.CULL_THRESHOLD:
            return CullResult(
                config_key=config_key,
                culled=0,
                kept=model_count,
                archived_models=[],
                preserved_models=[m.model_id for m in models],
                timestamp=time.time(),
            )

        # Calculate how many to keep
        keep_count = max(
            self.MIN_KEEP,
            int(model_count * self.KEEP_FRACTION)
        )

        # Split into keep and cull lists
        to_keep = models[:keep_count]
        candidates_for_cull = models[keep_count:]

        # Filter out models with high uncertainty (too few games)
        # These models haven't been properly evaluated yet
        to_cull = [
            m for m in candidates_for_cull
            if m.games_played >= self.MIN_GAMES_FOR_CULL
        ]
        protected_by_uncertainty = [
            m for m in candidates_for_cull
            if m.games_played < self.MIN_GAMES_FOR_CULL
        ]

        if protected_by_uncertainty:
            logger.info(
                f"[Culling] {config_key}: Protecting {len(protected_by_uncertainty)} "
                f"models with < {self.MIN_GAMES_FOR_CULL} games (high uncertainty)"
            )

        logger.info(
            f"[Culling] {config_key}: {model_count} models, "
            f"keeping {len(to_keep)}, culling {len(to_cull)}"
        )

        # Archive culled models
        archived = []
        for model in to_cull:
            success = self._archive_model(model, config_key)
            if success:
                archived.append(model.model_id)

        # Update last cull time
        self._last_cull[config_key] = time.time()

        # Export cull manifest so sync respects the culled models
        if archived:
            self.export_cull_manifest()
            # Trigger cluster sync to propagate archive markers
            self._trigger_cluster_sync(config_key, len(archived))

        # Count all kept models: top quartile + uncertainty-protected
        all_kept = to_keep + protected_by_uncertainty

        return CullResult(
            config_key=config_key,
            culled=len(archived),
            kept=len(all_kept),
            archived_models=archived,
            preserved_models=[m.model_id for m in all_kept],
            timestamp=time.time(),
        )

    def _trigger_cluster_sync(self, config_key: str, culled_count: int) -> None:
        """Trigger cluster-wide Elo sync after culling.

        Pushes the updated Elo database (with archive markers) to all
        reachable nodes in the cluster.
        """
        logger.info(
            f"[Culling] Triggering cluster sync after archiving {culled_count} models "
            f"for {config_key}"
        )

        # Try using the elo_db_sync script
        sync_script = Path(__file__).parent.parent.parent / "scripts" / "elo_db_sync.py"
        if sync_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(sync_script), "--mode", "cluster-sync"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(sync_script.parent.parent),
                    env={**os.environ, "PYTHONPATH": str(sync_script.parent.parent)}
                )
                if result.returncode == 0:
                    logger.info(f"[Culling] Cluster sync completed: {result.stdout.strip()[-200:]}")
                else:
                    logger.warning(f"[Culling] Cluster sync failed: {result.stderr.strip()[-200:]}")
            except subprocess.TimeoutExpired:
                logger.warning("[Culling] Cluster sync timed out after 120s")
            except Exception as e:
                logger.warning(f"[Culling] Cluster sync error: {e}")
        else:
            # Fallback: push directly to coordinator
            try:
                from app.sync.cluster_hosts import get_coordinator_address
                coord_ip, coord_port = get_coordinator_address()

                # Read new matches since last sync and push
                url = f"http://{coord_ip}:{coord_port}/status"
                with urllib.request.urlopen(url, timeout=10) as resp:
                    status = json.loads(resp.read().decode())
                    logger.info(f"[Culling] Coordinator status: {status.get('matches', 0)} matches")
            except Exception as e:
                logger.debug(f"[Culling] Could not reach coordinator: {e}")

    def _archive_model(self, model: ModelInfo, config_key: str) -> bool:
        """Archive a single model.

        Moves model file to archived/ subdirectory and updates database.

        Args:
            model: Model to archive
            config_key: Config this model belongs to

        Returns:
            True if archive succeeded
        """
        self._ensure_archive_columns()

        # Create archive directory
        archive_dir = self.model_dir / "archived" / config_key
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Find and move model file
        src = self.model_dir / model.filename
        alt_src = self.model_dir / f"{model.model_id}.pth"

        if src.exists():
            dst = archive_dir / model.filename
            try:
                shutil.move(str(src), str(dst))
                logger.debug(f"[Culling] Archived {src} -> {dst}")
            except Exception as e:
                logger.warning(f"[Culling] Failed to move {src}: {e}")
        elif alt_src.exists():
            dst = archive_dir / f"{model.model_id}.pth"
            try:
                shutil.move(str(alt_src), str(dst))
                logger.debug(f"[Culling] Archived {alt_src} -> {dst}")
            except Exception as e:
                logger.warning(f"[Culling] Failed to move {alt_src}: {e}")
        else:
            # File not found - might already be archived or path different
            logger.debug(f"[Culling] Model file not found: {model.filename}")

        # Update database regardless of file move
        # (mark as archived even if file was already gone)
        self._mark_archived(model, config_key)

        return True  # Always return True since we updated DB

    def _mark_archived(self, model: ModelInfo, config_key: str) -> None:
        """Mark model as archived in database."""
        parts = config_key.split("_")
        board_type = parts[0]
        num_players = int(parts[1].replace("p", ""))

        conn = self._get_db_connection()
        try:
            # Determine column name
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}
            id_col = "model_id" if "model_id" in columns else "participant_id"

            conn.execute(f"""
                UPDATE elo_ratings
                SET archived_at = ?, archive_reason = ?
                WHERE {id_col} = ? AND board_type = ? AND num_players = ?
            """, (
                time.time(),
                f"culled_below_top_{int(self.KEEP_FRACTION * 100)}pct",
                model.model_id,
                board_type,
                num_players,
            ))
            conn.commit()
        finally:
            conn.close()

    def get_protected_model_set(self, config_key: str) -> set[str]:
        """Get the set of model IDs (composite and bare) protected from archival.

        Queries all non-archived elo_ratings for this config, extracts the base
        model (nn_id part), takes the MAX rating across all harnesses per base
        model, then protects the top 50%.

        Args:
            config_key: Config like "square8_2p"

        Returns:
            Set of all participant_ids (composite and bare) for protected models
        """
        models = self.get_models_for_config(config_key)
        if not models:
            return set()

        # Group by base nn_id (strip harness/config_hash suffix)
        from collections import defaultdict
        base_models: dict[str, list[ModelInfo]] = defaultdict(list)
        for m in models:
            # Extract nn_id from composite ID: "nn_id:harness:config" -> "nn_id"
            if ":" in m.model_id:
                nn_id = m.model_id.split(":")[0]
            else:
                nn_id = m.model_id
            base_models[nn_id].append(m)

        # For each base model, take its MAX rating across all harnesses
        base_best: list[tuple[str, float, list[str]]] = []
        for nn_id, entries in base_models.items():
            best_elo = max(e.elo for e in entries)
            all_ids = [e.model_id for e in entries]
            base_best.append((nn_id, best_elo, all_ids))

        # Sort by best rating descending
        base_best.sort(key=lambda x: x[1], reverse=True)

        # Protect top 50%
        keep_count = max(1, int(len(base_best) * self.KEEP_FRACTION))
        protected = set()
        for nn_id, _, all_ids in base_best[:keep_count]:
            protected.add(nn_id)  # bare nn_id
            for pid in all_ids:
                protected.add(pid)  # all composite variants

        return protected

    def cull_all_configs(self) -> dict[str, CullResult]:
        """Check and cull all 9 configs.

        Returns:
            Dict mapping config_key to CullResult
        """
        results = {}
        for config_key in CONFIG_KEYS:
            if self.needs_culling(config_key):
                results[config_key] = self.check_and_cull(config_key)
            else:
                # Return status without culling
                count = self.count_models(config_key)
                results[config_key] = CullResult(
                    config_key=config_key,
                    culled=0,
                    kept=count,
                    archived_models=[],
                    preserved_models=[],
                    timestamp=time.time(),
                )
        return results

    def get_archive_stats(self) -> dict[str, dict]:
        """Get statistics about archived models.

        Returns:
            Dict with per-config archive counts and total
        """
        conn = self._get_db_connection()
        try:
            # Check if archived_at column exists
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}

            if "archived_at" not in columns:
                return {"total_archived": 0, "by_config": {}}

            cursor = conn.execute("""
                SELECT board_type, num_players, COUNT(*) as count
                FROM elo_ratings
                WHERE archived_at IS NOT NULL AND archived_at > 0
                GROUP BY board_type, num_players
            """)

            by_config = {}
            total = 0
            for row in cursor.fetchall():
                config_key = f"{row['board_type']}_{row['num_players']}p"
                by_config[config_key] = row["count"]
                total += row["count"]

            return {"total_archived": total, "by_config": by_config}
        finally:
            conn.close()


    def export_cull_manifest(self) -> Path:
        """Export list of archived model IDs to a manifest file.

        This manifest is used by sync_models.py to skip re-syncing culled models.
        The manifest is synced between nodes before model sync.

        Returns:
            Path to the manifest file
        """
        manifest_path = self.model_dir / "cull_manifest.json"

        conn = self._get_db_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(elo_ratings)")
            columns = {row[1] for row in cursor.fetchall()}

            if "archived_at" not in columns:
                # No archived models
                manifest = {"archived_models": [], "updated_at": time.time()}
            else:
                id_col = "model_id" if "model_id" in columns else "participant_id"
                cursor = conn.execute(f"""
                    SELECT {id_col}, board_type, num_players, archived_at, archive_reason
                    FROM elo_ratings
                    WHERE archived_at IS NOT NULL AND archived_at > 0
                """)

                archived = []
                for row in cursor.fetchall():
                    archived.append({
                        "model_id": row[0],
                        "board_type": row[1],
                        "num_players": row[2],
                        "archived_at": row[3],
                        "reason": row[4],
                    })

                manifest = {
                    "archived_models": archived,
                    "archived_ids": [m["model_id"] for m in archived],
                    "updated_at": time.time(),
                    "total_count": len(archived),
                }

            # Write manifest
            import json
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"[Culling] Exported cull manifest with {len(manifest.get('archived_ids', []))} models")
            return manifest_path

        finally:
            conn.close()

    def load_cull_manifest(self) -> set[str]:
        """Load archived model IDs from manifest file.

        Returns:
            Set of archived model IDs
        """
        manifest_path = self.model_dir / "cull_manifest.json"
        if not manifest_path.exists():
            return set()

        try:
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            return set(manifest.get("archived_ids", []))
        except Exception as e:
            logger.warning(f"[Culling] Failed to load cull manifest: {e}")
            return set()


def get_culling_controller(
    elo_db_path: Path | None = None,
    model_dir: Path | None = None,
) -> ModelCullingController:
    """Get culling controller instance.

    Args:
        elo_db_path: Path to Elo database (default: data/unified_elo.db)
        model_dir: Path to models directory (default: data/models)

    Returns:
        Configured ModelCullingController instance
    """
    project_root = Path(__file__).parent.parent.parent

    if elo_db_path is None:
        elo_db_path = project_root / "data" / "unified_elo.db"
    if model_dir is None:
        model_dir = project_root / "data" / "models"

    return ModelCullingController(elo_db_path, model_dir)
