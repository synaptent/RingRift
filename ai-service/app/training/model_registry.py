"""
Automated Model Registry for RingRift AI.

.. deprecated:: December 2025
    For new code, prefer importing from :mod:`app.training.unified_model_store`
    which provides a simpler API with automatic event emission::

        # Preferred (unified API):
        from app.training.unified_model_store import (
            get_model_store,
            register_model,
            get_production_model,
            promote_model,
        )

        # Or via package:
        from app.training import (
            UnifiedModelStore,
            get_model_store,
            ModelInfo,
        )

    This module remains available for direct access to ModelRegistry internals,
    validation tracking, AutoPromoter, and database operations.

Provides comprehensive model version tracking, metadata storage,
promotion workflows, and comparison tools.

This module handles the MODEL LIFECYCLE:
- Track models from development → staging → production → archived
- Store training configurations and performance metrics
- Support promotion workflows with comparison tools
- Query models by stage, performance, or board type

Works with model_versioning.py which handles CHECKPOINT INTEGRITY:
- Architecture version validation
- SHA256 checksums for weight verification
- Migration from legacy checkpoint formats

Typical usage:
    from app.training.model_registry import ModelRegistry, ModelStage
    from app.training.model_versioning import save_versioned_checkpoint

    # Register a new model after training
    registry = ModelRegistry()
    model_id = registry.register_model(
        board_type="square8",
        num_players=2,
        stage=ModelStage.DEVELOPMENT,
    )

    # Save checkpoint with integrity verification
    save_versioned_checkpoint(model, path, model_id=model_id)

    # Later: promote to production after evaluation
    registry.promote_model(model_id, ModelStage.PRODUCTION)
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Centralized database utilities (December 2025)
try:
    from app.distributed.db_utils import get_db_connection
except ImportError:
    get_db_connection = None

# Import PromotionCriteria for unified thresholds (avoid circular import)
if TYPE_CHECKING:
    from app.training.promotion_controller import PromotionCriteria

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"      # Initial training
    STAGING = "staging"              # Under evaluation
    PRODUCTION = "production"        # Deployed for inference
    ARCHIVED = "archived"            # Retired models
    REJECTED = "rejected"            # Failed evaluation


class ValidationStatus(Enum):
    """Model validation status for automated validation loop."""
    PENDING = "pending"              # Needs validation
    QUEUED = "queued"                # Validation work item queued
    RUNNING = "running"              # Validation in progress
    PASSED = "passed"                # Validation successful
    FAILED = "failed"                # Validation failed
    SKIPPED = "skipped"              # Validation not required


class ModelType(Enum):
    """Types of models in the registry."""
    POLICY_VALUE = "policy_value"    # Main game-playing model
    ENSEMBLE = "ensemble"            # Ensemble of models
    COMPRESSED = "compressed"        # Quantized/pruned model
    EXPERIMENTAL = "experimental"    # Research models
    HEURISTIC = "heuristic"          # CMA-ES optimized heuristic weights


@dataclass
class ModelMetrics:
    """Performance metrics for a model."""
    elo: float | None = None
    elo_uncertainty: float | None = None
    win_rate: float | None = None
    draw_rate: float | None = None
    games_played: int = 0
    avg_move_time_ms: float | None = None
    policy_accuracy: float | None = None
    value_mse: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'ModelMetrics':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration for a model."""
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    optimizer: str = "adam"
    architecture: str = "resnet"
    num_residual_blocks: int = 10
    num_filters: int = 128
    augmentation_enabled: bool = True
    curriculum_stage: str | None = None
    parent_model_id: str | None = None
    training_data_hash: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'TrainingConfig':
        known_fields = set(cls.__dataclass_fields__.keys())
        known = {k: v for k, v in d.items() if k in known_fields and k != 'extra_config'}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        known['extra_config'] = d.get('extra_config', {})
        known['extra_config'].update(extra)
        return cls(**known)


@dataclass
class ModelVersion:
    """A registered model version."""
    model_id: str
    version: int
    name: str
    model_type: ModelType
    stage: ModelStage
    created_at: datetime
    updated_at: datetime
    file_path: str
    file_hash: str
    file_size_bytes: int
    metrics: ModelMetrics
    training_config: TrainingConfig
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['model_type'] = self.model_type.value
        d['stage'] = self.stage.value
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d


class RegistryDatabase:
    """SQLite database for model registry."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            if get_db_connection is not None:
                self._local.conn = get_db_connection(self.db_path)
            else:
                self._local.conn = sqlite3.connect(str(self.db_path), timeout=30)
                self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        if get_db_connection is not None:
            conn = get_db_connection(self.db_path)
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    metrics_json TEXT,
                    training_config_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    UNIQUE(model_id, version)
                );

                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    UNIQUE(model_id, version, tag)
                );

                CREATE TABLE IF NOT EXISTS stage_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    from_stage TEXT,
                    to_stage TEXT NOT NULL,
                    reason TEXT,
                    transitioned_by TEXT,
                    transitioned_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id)
                );

                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_a_id TEXT NOT NULL,
                    model_a_version INTEGER NOT NULL,
                    model_b_id TEXT NOT NULL,
                    model_b_version INTEGER NOT NULL,
                    games_played INTEGER NOT NULL,
                    model_a_wins INTEGER NOT NULL,
                    model_b_wins INTEGER NOT NULL,
                    draws INTEGER NOT NULL,
                    elo_diff REAL,
                    compared_at TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    work_id TEXT,
                    baselines TEXT,
                    games_per_matchup INTEGER DEFAULT 50,
                    results_json TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    UNIQUE(model_id, version)
                );

                CREATE INDEX IF NOT EXISTS idx_versions_model ON versions(model_id);
                CREATE INDEX IF NOT EXISTS idx_versions_stage ON versions(stage);
                CREATE INDEX IF NOT EXISTS idx_tags_model ON tags(model_id, version);
                CREATE INDEX IF NOT EXISTS idx_transitions_model ON stage_transitions(model_id, version);
                CREATE INDEX IF NOT EXISTS idx_validations_status ON validations(status);
            """)
            conn.commit()
        finally:
            conn.close()

    def create_model(self, model_id: str, name: str, model_type: ModelType,
                     description: str = "") -> None:
        """Create a new model entry."""
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO models (model_id, name, description, model_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_id, name, description, model_type.value, now, now))
        self.conn.commit()

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        cursor = self.conn.execute(
            "SELECT 1 FROM models WHERE model_id = ?", (model_id,)
        )
        return cursor.fetchone() is not None

    def get_next_version(self, model_id: str) -> int:
        """Get the next version number for a model."""
        cursor = self.conn.execute(
            "SELECT MAX(version) FROM versions WHERE model_id = ?", (model_id,)
        )
        row = cursor.fetchone()
        return (row[0] or 0) + 1

    def create_version(self, model_id: str, version: int, stage: ModelStage,
                       file_path: str, file_hash: str, file_size: int,
                       metrics: ModelMetrics, training_config: TrainingConfig) -> None:
        """Create a new version entry."""
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO versions
            (model_id, version, stage, file_path, file_hash, file_size_bytes,
             metrics_json, training_config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id, version, stage.value, file_path, file_hash, file_size,
            json.dumps(metrics.to_dict()), json.dumps(training_config.to_dict()),
            now, now
        ))

        # Record stage transition
        self.conn.execute("""
            INSERT INTO stage_transitions
            (model_id, version, from_stage, to_stage, reason, transitioned_at)
            VALUES (?, ?, NULL, ?, 'Initial registration', ?)
        """, (model_id, version, stage.value, now))

        self.conn.commit()

    def update_stage(self, model_id: str, version: int, new_stage: ModelStage,
                     reason: str = "", transitioned_by: str = "system") -> None:
        """Update the stage of a model version."""
        now = datetime.now().isoformat()

        # Get current stage
        cursor = self.conn.execute(
            "SELECT stage FROM versions WHERE model_id = ? AND version = ?",
            (model_id, version)
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Version {model_id}:{version} not found")

        old_stage = row['stage']

        # Update stage
        self.conn.execute("""
            UPDATE versions SET stage = ?, updated_at = ?
            WHERE model_id = ? AND version = ?
        """, (new_stage.value, now, model_id, version))

        # Record transition
        self.conn.execute("""
            INSERT INTO stage_transitions
            (model_id, version, from_stage, to_stage, reason, transitioned_by, transitioned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_id, version, old_stage, new_stage.value, reason, transitioned_by, now))

        self.conn.commit()

    def update_metrics(self, model_id: str, version: int, metrics: ModelMetrics) -> None:
        """Update metrics for a model version."""
        now = datetime.now().isoformat()
        self.conn.execute("""
            UPDATE versions SET metrics_json = ?, updated_at = ?
            WHERE model_id = ? AND version = ?
        """, (json.dumps(metrics.to_dict()), now, model_id, version))
        self.conn.commit()

    def add_tag(self, model_id: str, version: int, tag: str) -> None:
        """Add a tag to a model version."""
        try:
            self.conn.execute("""
                INSERT INTO tags (model_id, version, tag) VALUES (?, ?, ?)
            """, (model_id, version, tag))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # Tag already exists

    def remove_tag(self, model_id: str, version: int, tag: str) -> None:
        """Remove a tag from a model version."""
        self.conn.execute("""
            DELETE FROM tags WHERE model_id = ? AND version = ? AND tag = ?
        """, (model_id, version, tag))
        self.conn.commit()

    def get_version(self, model_id: str, version: int) -> dict[str, Any] | None:
        """Get a specific version."""
        cursor = self.conn.execute("""
            SELECT v.*, m.name, m.description, m.model_type
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.model_id = ? AND v.version = ?
        """, (model_id, version))
        row = cursor.fetchone()
        if not row:
            return None

        # Get tags
        tag_cursor = self.conn.execute(
            "SELECT tag FROM tags WHERE model_id = ? AND version = ?",
            (model_id, version)
        )
        tags = [r['tag'] for r in tag_cursor.fetchall()]

        return {
            'model_id': row['model_id'],
            'version': row['version'],
            'name': row['name'],
            'description': row['description'],
            'model_type': row['model_type'],
            'stage': row['stage'],
            'file_path': row['file_path'],
            'file_hash': row['file_hash'],
            'file_size_bytes': row['file_size_bytes'],
            'metrics': json.loads(row['metrics_json']) if row['metrics_json'] else {},
            'training_config': json.loads(row['training_config_json']) if row['training_config_json'] else {},
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'tags': tags
        }

    def get_versions_by_stage(self, stage: ModelStage) -> list[dict[str, Any]]:
        """Get all versions in a specific stage."""
        cursor = self.conn.execute("""
            SELECT v.*, m.name, m.model_type
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.stage = ?
            ORDER BY v.updated_at DESC
        """, (stage.value,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'model_id': row['model_id'],
                'version': row['version'],
                'name': row['name'],
                'model_type': row['model_type'],
                'stage': row['stage'],
                'file_path': row['file_path'],
                'metrics': json.loads(row['metrics_json']) if row['metrics_json'] else {},
                'updated_at': row['updated_at']
            })
        return results

    def get_latest_production(self, model_type: ModelType | None = None) -> dict[str, Any] | None:
        """Get the latest production model."""
        query = """
            SELECT v.*, m.name, m.model_type
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.stage = 'production'
        """
        params = []
        if model_type:
            query += " AND m.model_type = ?"
            params.append(model_type.value)
        query += " ORDER BY v.updated_at DESC LIMIT 1"

        cursor = self.conn.execute(query, params)
        row = cursor.fetchone()
        if not row:
            return None

        return {
            'model_id': row['model_id'],
            'version': row['version'],
            'name': row['name'],
            'model_type': row['model_type'],
            'file_path': row['file_path'],
            'metrics': json.loads(row['metrics_json']) if row['metrics_json'] else {}
        }

    def search_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Search for models by tag."""
        cursor = self.conn.execute("""
            SELECT v.model_id, v.version, m.name, v.stage, v.file_path
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            JOIN tags t ON v.model_id = t.model_id AND v.version = t.version
            WHERE t.tag = ?
        """, (tag,))
        return [dict(row) for row in cursor.fetchall()]

    def record_comparison(self, model_a_id: str, model_a_version: int,
                          model_b_id: str, model_b_version: int,
                          games: int, a_wins: int, b_wins: int, draws: int,
                          elo_diff: float | None = None, notes: str = "") -> None:
        """Record a model comparison result."""
        now = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO comparisons
            (model_a_id, model_a_version, model_b_id, model_b_version,
             games_played, model_a_wins, model_b_wins, draws, elo_diff, compared_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_a_id, model_a_version, model_b_id, model_b_version,
            games, a_wins, b_wins, draws, elo_diff, now, notes
        ))
        self.conn.commit()

    def get_stage_history(self, model_id: str, version: int) -> list[dict[str, Any]]:
        """Get stage transition history for a version."""
        cursor = self.conn.execute("""
            SELECT * FROM stage_transitions
            WHERE model_id = ? AND version = ?
            ORDER BY transitioned_at ASC
        """, (model_id, version))
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # VALIDATION TRACKING
    # =========================================================================

    def create_validation(
        self,
        model_id: str,
        version: int,
        baselines: list[str] | None = None,
        games_per_matchup: int = 50,
    ) -> None:
        """Create a validation entry for a model version."""
        now = datetime.now().isoformat()
        baselines_str = json.dumps(baselines) if baselines else None
        try:
            self.conn.execute("""
                INSERT INTO validations
                (model_id, version, status, baselines, games_per_matchup, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
            """, (model_id, version, baselines_str, games_per_matchup, now, now))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Already exists, update instead
            self.conn.execute("""
                UPDATE validations
                SET status = 'pending', baselines = ?, games_per_matchup = ?, updated_at = ?
                WHERE model_id = ? AND version = ?
            """, (baselines_str, games_per_matchup, now, model_id, version))
            self.conn.commit()

    def get_validation_status(self, model_id: str, version: int) -> str | None:
        """Get validation status for a model version."""
        cursor = self.conn.execute(
            "SELECT status FROM validations WHERE model_id = ? AND version = ?",
            (model_id, version)
        )
        row = cursor.fetchone()
        return row['status'] if row else None

    def update_validation_status(
        self,
        model_id: str,
        version: int,
        status: str,
        work_id: str | None = None,
        results: dict[str, Any] | None = None,
    ) -> None:
        """Update validation status for a model version."""
        now = datetime.now().isoformat()

        updates = ["status = ?", "updated_at = ?"]
        params = [status, now]

        if work_id is not None:
            updates.append("work_id = ?")
            params.append(work_id)

        if results is not None:
            updates.append("results_json = ?")
            params.append(json.dumps(results))

        if status == "running":
            updates.append("started_at = ?")
            params.append(now)
        elif status in ("passed", "failed"):
            updates.append("completed_at = ?")
            params.append(now)

        params.extend([model_id, version])
        self.conn.execute(f"""
            UPDATE validations
            SET {", ".join(updates)}
            WHERE model_id = ? AND version = ?
        """, params)
        self.conn.commit()

    def get_models_needing_validation(self) -> list[dict[str, Any]]:
        """Get all models that need validation (status = pending)."""
        cursor = self.conn.execute("""
            SELECT v.model_id, v.version, v.file_path, v.stage,
                   val.status as validation_status, val.baselines, val.games_per_matchup,
                   m.name, m.model_type
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            LEFT JOIN validations val ON v.model_id = val.model_id AND v.version = val.version
            WHERE val.status = 'pending'
            ORDER BY v.updated_at DESC
        """)
        results = []
        for row in cursor.fetchall():
            results.append({
                'model_id': row['model_id'],
                'version': row['version'],
                'file_path': row['file_path'],
                'stage': row['stage'],
                'name': row['name'],
                'model_type': row['model_type'],
                'validation_status': row['validation_status'],
                'baselines': json.loads(row['baselines']) if row['baselines'] else [],
                'games_per_matchup': row['games_per_matchup'],
            })
        return results

    def get_models_without_validation(self) -> list[dict[str, Any]]:
        """Get all models that don't have a validation entry yet."""
        cursor = self.conn.execute("""
            SELECT v.model_id, v.version, v.file_path, v.stage, m.name, m.model_type
            FROM versions v
            JOIN models m ON v.model_id = m.model_id
            LEFT JOIN validations val ON v.model_id = val.model_id AND v.version = val.version
            WHERE val.id IS NULL
            AND v.stage IN ('development', 'staging')
            ORDER BY v.created_at DESC
        """)
        results = []
        for row in cursor.fetchall():
            results.append({
                'model_id': row['model_id'],
                'version': row['version'],
                'file_path': row['file_path'],
                'stage': row['stage'],
                'name': row['name'],
                'model_type': row['model_type'],
            })
        return results

    def get_validation(self, model_id: str, version: int) -> dict[str, Any] | None:
        """Get full validation record for a model version."""
        cursor = self.conn.execute("""
            SELECT * FROM validations WHERE model_id = ? AND version = ?
        """, (model_id, version))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'model_id': row['model_id'],
            'version': row['version'],
            'status': row['status'],
            'work_id': row['work_id'],
            'baselines': json.loads(row['baselines']) if row['baselines'] else [],
            'games_per_matchup': row['games_per_matchup'],
            'results': json.loads(row['results_json']) if row['results_json'] else None,
            'started_at': row['started_at'],
            'completed_at': row['completed_at'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
        }


class ModelRegistry:
    """
    Main interface for the model registry.

    Handles model storage, versioning, and lifecycle management.
    """

    # Default registry directory
    DEFAULT_REGISTRY_DIR = Path("data/model_registry")

    def __init__(self, registry_dir: Path | None = None, model_storage_dir: Path | None = None):
        self.registry_dir = Path(registry_dir) if registry_dir else self.DEFAULT_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.model_storage_dir = model_storage_dir or (self.registry_dir / "models")
        self.model_storage_dir.mkdir(parents=True, exist_ok=True)

        self.db = RegistryDatabase(self.registry_dir / "registry.db")

        logger.info(f"Model registry initialized at {registry_dir}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _copy_to_storage(self, source_path: Path, model_id: str, version: int) -> Path:
        """Copy model file to registry storage."""
        dest_dir = self.model_storage_dir / model_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        suffix = source_path.suffix or '.pt'
        dest_path = dest_dir / f"v{version}{suffix}"

        shutil.copy2(source_path, dest_path)
        return dest_path

    def register_model(
        self,
        name: str,
        model_path: Path,
        model_type: ModelType = ModelType.POLICY_VALUE,
        description: str = "",
        metrics: ModelMetrics | None = None,
        training_config: TrainingConfig | None = None,
        tags: list[str] | None = None,
        initial_stage: ModelStage = ModelStage.DEVELOPMENT,
        model_id: str | None = None
    ) -> tuple[str, int]:
        """
        Register a new model or new version of existing model.

        Returns: (model_id, version)
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate or use provided model_id
        if model_id is None:
            # Generate ID from name
            model_id = name.lower().replace(' ', '_').replace('-', '_')
            model_id = ''.join(c for c in model_id if c.isalnum() or c == '_')

        # Create model entry if new
        if not self.db.model_exists(model_id):
            self.db.create_model(model_id, name, model_type, description)

        # Get next version
        version = self.db.get_next_version(model_id)

        # Compute file hash and size
        file_hash = self._compute_file_hash(model_path)
        file_size = model_path.stat().st_size

        # Copy to storage
        storage_path = self._copy_to_storage(model_path, model_id, version)

        # Create version entry
        self.db.create_version(
            model_id=model_id,
            version=version,
            stage=initial_stage,
            file_path=str(storage_path),
            file_hash=file_hash,
            file_size=file_size,
            metrics=metrics or ModelMetrics(),
            training_config=training_config or TrainingConfig()
        )

        # Add tags
        if tags:
            for tag in tags:
                self.db.add_tag(model_id, version, tag)

        # Auto-create validation entry for trackable models
        if model_type == ModelType.POLICY_VALUE and initial_stage in (ModelStage.DEVELOPMENT, ModelStage.STAGING):
            self.db.create_validation(model_id, version)
            logger.debug(f"Created validation entry for {model_id}:v{version}")

        logger.info(f"Registered {model_id}:v{version} ({model_type.value})")
        return model_id, version

    def promote(self, model_id: str, version: int, to_stage: ModelStage,
                reason: str = "", promoted_by: str = "system") -> None:
        """Promote a model to a new stage."""
        # Validate transition
        current = self.db.get_version(model_id, version)
        if not current:
            raise ValueError(f"Version {model_id}:{version} not found")

        current_stage = ModelStage(current['stage'])

        # Define allowed transitions
        allowed_transitions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED, ModelStage.REJECTED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED, ModelStage.REJECTED],
            ModelStage.PRODUCTION: [ModelStage.ARCHIVED, ModelStage.STAGING],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],
            ModelStage.REJECTED: [ModelStage.DEVELOPMENT]
        }

        if to_stage not in allowed_transitions.get(current_stage, []):
            raise ValueError(
                f"Cannot transition from {current_stage.value} to {to_stage.value}"
            )

        # If promoting to production, demote current production model
        if to_stage == ModelStage.PRODUCTION:
            current_prod = self.db.get_latest_production()
            if current_prod and (current_prod['model_id'] != model_id or
                                 current_prod['version'] != version):
                self.db.update_stage(
                    current_prod['model_id'],
                    current_prod['version'],
                    ModelStage.ARCHIVED,
                    f"Replaced by {model_id}:v{version}",
                    promoted_by
                )

        self.db.update_stage(model_id, version, to_stage, reason, promoted_by)
        logger.info(f"Promoted {model_id}:v{version} to {to_stage.value}")

    def update_metrics(self, model_id: str, version: int, metrics: ModelMetrics) -> None:
        """Update metrics for a model version."""
        self.db.update_metrics(model_id, version, metrics)

    def get_model(self, model_id: str, version: int | None = None) -> ModelVersion | None:
        """
        Get a model version.

        If version is None, returns the latest version.
        """
        if version is None:
            version = self.db.get_next_version(model_id) - 1
            if version < 1:
                return None

        data = self.db.get_version(model_id, version)
        if not data:
            return None

        return ModelVersion(
            model_id=data['model_id'],
            version=data['version'],
            name=data['name'],
            model_type=ModelType(data['model_type']),
            stage=ModelStage(data['stage']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            file_size_bytes=data['file_size_bytes'],
            metrics=ModelMetrics.from_dict(data['metrics']),
            training_config=TrainingConfig.from_dict(data['training_config']),
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )

    def get_production_model(self, model_type: ModelType | None = None) -> ModelVersion | None:
        """Get the current production model."""
        data = self.db.get_latest_production(model_type)
        if not data:
            return None
        return self.get_model(data['model_id'], data['version'])

    def list_models(self, stage: ModelStage | None = None,
                    model_type: ModelType | None = None) -> list[dict[str, Any]]:
        """List all models, optionally filtered."""
        if stage:
            models = self.db.get_versions_by_stage(stage)
        else:
            # Get all models
            cursor = self.db.conn.execute("""
                SELECT v.model_id, v.version, m.name, m.model_type, v.stage,
                       v.file_path, v.metrics_json, v.updated_at
                FROM versions v
                JOIN models m ON v.model_id = m.model_id
                ORDER BY v.updated_at DESC
            """)
            models = []
            for row in cursor.fetchall():
                models.append({
                    'model_id': row['model_id'],
                    'version': row['version'],
                    'name': row['name'],
                    'model_type': row['model_type'],
                    'stage': row['stage'],
                    'file_path': row['file_path'],
                    'metrics': json.loads(row['metrics_json']) if row['metrics_json'] else {},
                    'updated_at': row['updated_at']
                })

        if model_type:
            models = [m for m in models if m['model_type'] == model_type.value]

        return models

    def compare_models(
        self,
        model_a: tuple[str, int],
        model_b: tuple[str, int],
        games: int,
        a_wins: int,
        b_wins: int,
        draws: int,
        elo_diff: float | None = None,
        notes: str = ""
    ) -> dict[str, Any]:
        """
        Record and return a comparison between two models.
        """
        self.db.record_comparison(
            model_a[0], model_a[1],
            model_b[0], model_b[1],
            games, a_wins, b_wins, draws,
            elo_diff, notes
        )

        total = a_wins + b_wins + draws
        return {
            'model_a': f"{model_a[0]}:v{model_a[1]}",
            'model_b': f"{model_b[0]}:v{model_b[1]}",
            'games': games,
            'a_win_rate': a_wins / total if total > 0 else 0,
            'b_win_rate': b_wins / total if total > 0 else 0,
            'draw_rate': draws / total if total > 0 else 0,
            'elo_diff': elo_diff
        }

    def add_tag(self, model_id: str, version: int, tag: str) -> None:
        """Add a tag to a model version."""
        self.db.add_tag(model_id, version, tag)

    def search_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Search for models by tag."""
        return self.db.search_by_tag(tag)

    def get_stage_history(self, model_id: str, version: int) -> list[dict[str, Any]]:
        """Get the stage transition history for a model."""
        return self.db.get_stage_history(model_id, version)

    def export_model(self, model_id: str, version: int, dest_path: Path) -> Path:
        """Export a model to a specified location."""
        model = self.get_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id}:v{version} not found")

        dest_path = Path(dest_path)

        # Copy model file
        shutil.copy2(model.file_path, dest_path)

        # Create metadata file
        metadata = model.to_dict()
        metadata_path = dest_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Exported {model_id}:v{version} to {dest_path}")
        return dest_path

    def import_model(self, model_path: Path, metadata_path: Path | None = None,
                     model_id: str | None = None) -> tuple[str, int]:
        """Import a model from an exported file."""
        model_path = Path(model_path)

        # Load metadata if available
        if metadata_path is None:
            metadata_path = model_path.with_suffix('.json')

        metrics = None
        training_config = None
        name = model_id or model_path.stem
        description = ""
        tags = []
        model_type = ModelType.POLICY_VALUE

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            name = metadata.get('name', name)
            description = metadata.get('description', '')
            tags = metadata.get('tags', [])
            model_type = ModelType(metadata.get('model_type', 'policy_value'))

            if 'metrics' in metadata:
                metrics = ModelMetrics.from_dict(metadata['metrics'])
            if 'training_config' in metadata:
                training_config = TrainingConfig.from_dict(metadata['training_config'])

        return self.register_model(
            name=name,
            model_path=model_path,
            model_type=model_type,
            description=description,
            metrics=metrics,
            training_config=training_config,
            tags=tags,
            model_id=model_id
        )

    # =========================================================================
    # VALIDATION TRACKING CONVENIENCE METHODS
    # =========================================================================

    def get_models_needing_validation(self) -> list[dict[str, Any]]:
        """Get all models with pending validation status."""
        return self.db.get_models_needing_validation()

    def get_unvalidated_models(self) -> list[dict[str, Any]]:
        """Get all models without any validation entry."""
        return self.db.get_models_without_validation()

    def create_validation(
        self,
        model_id: str,
        version: int,
        baselines: list[str] | None = None,
        games_per_matchup: int = 50,
    ) -> None:
        """Create or reset a validation entry for a model."""
        self.db.create_validation(model_id, version, baselines, games_per_matchup)

    def set_validation_queued(
        self,
        model_id: str,
        version: int,
        work_id: str,
    ) -> None:
        """Mark a model's validation as queued with the work item ID."""
        self.db.update_validation_status(model_id, version, "queued", work_id=work_id)

    def set_validation_running(self, model_id: str, version: int) -> None:
        """Mark a model's validation as currently running."""
        self.db.update_validation_status(model_id, version, "running")

    def set_validation_passed(
        self,
        model_id: str,
        version: int,
        results: dict[str, Any] | None = None,
    ) -> None:
        """Mark a model's validation as passed."""
        self.db.update_validation_status(model_id, version, "passed", results=results)

    def set_validation_failed(
        self,
        model_id: str,
        version: int,
        results: dict[str, Any] | None = None,
    ) -> None:
        """Mark a model's validation as failed."""
        self.db.update_validation_status(model_id, version, "failed", results=results)

    def get_validation(self, model_id: str, version: int) -> dict[str, Any] | None:
        """Get the validation record for a model."""
        return self.db.get_validation(model_id, version)

    def get_validation_status(self, model_id: str, version: int) -> str | None:
        """Get just the validation status for a model."""
        return self.db.get_validation_status(model_id, version)


class AutoPromoter:
    """
    Automatic model promotion based on performance criteria.

    Uses PromotionCriteria from promotion_controller.py for unified thresholds
    across all promotion systems. Supports backward compatibility with explicit
    threshold overrides.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        criteria: Optional["PromotionCriteria"] = None,
        min_elo_improvement: float | None = None,
        min_games: int | None = None,
        min_win_rate_vs_current: float | None = None,
        auto_archive_after_days: int = 30
    ):
        """Initialize AutoPromoter with unified or custom criteria.

        Args:
            registry: ModelRegistry instance
            criteria: PromotionCriteria from promotion_controller (preferred)
            min_elo_improvement: Override Elo threshold (uses PromotionCriteria default if None)
            min_games: Override games threshold (uses PromotionCriteria default if None)
            min_win_rate_vs_current: Override win rate (uses PromotionCriteria default if None)
            auto_archive_after_days: Days until automatic archival
        """
        self.registry = registry
        self.auto_archive_after_days = auto_archive_after_days

        # Use provided criteria or create default from PromotionCriteria
        if criteria is not None:
            self._criteria = criteria
            self.min_elo_improvement = criteria.min_elo_improvement
            self.min_games = criteria.min_games_played
            self.min_win_rate_vs_current = criteria.min_win_rate
        else:
            try:
                from app.training.promotion_controller import PromotionCriteria
                default_criteria = PromotionCriteria()
                self._criteria = default_criteria
                self.min_elo_improvement = min_elo_improvement if min_elo_improvement is not None else default_criteria.min_elo_improvement
                self.min_games = min_games if min_games is not None else default_criteria.min_games_played
                self.min_win_rate_vs_current = min_win_rate_vs_current if min_win_rate_vs_current is not None else default_criteria.min_win_rate
            except ImportError:
                self._criteria = None
                self.min_elo_improvement = min_elo_improvement if min_elo_improvement is not None else 20.0
                self.min_games = min_games if min_games is not None else 100
                self.min_win_rate_vs_current = min_win_rate_vs_current if min_win_rate_vs_current is not None else 0.52

    def evaluate_for_staging(self, model_id: str, version: int) -> tuple[bool, str]:
        """
        Evaluate if a development model should be promoted to staging.

        Returns: (should_promote, reason)
        """
        model = self.registry.get_model(model_id, version)
        if not model:
            return False, "Model not found"

        if model.stage != ModelStage.DEVELOPMENT:
            return False, f"Model is in {model.stage.value}, not development"

        metrics = model.metrics

        # Check minimum games
        if metrics.games_played < self.min_games // 2:  # Half the requirement for staging
            return False, f"Insufficient games ({metrics.games_played} < {self.min_games // 2})"

        # Check for positive metrics
        if metrics.elo is None:
            return False, "No Elo rating available"

        return True, "Model meets staging criteria"

    def evaluate_for_production(self, model_id: str, version: int) -> tuple[bool, str]:
        """
        Evaluate if a staging model should be promoted to production.

        Returns: (should_promote, reason)
        """
        model = self.registry.get_model(model_id, version)
        if not model:
            return False, "Model not found"

        if model.stage != ModelStage.STAGING:
            return False, f"Model is in {model.stage.value}, not staging"

        metrics = model.metrics

        # Check minimum games
        if metrics.games_played < self.min_games:
            return False, f"Insufficient games ({metrics.games_played} < {self.min_games})"

        # Compare against current production
        current_prod = self.registry.get_production_model()
        if current_prod:
            if metrics.elo is None or current_prod.metrics.elo is None:
                return False, "Cannot compare Elo ratings"

            elo_diff = metrics.elo - current_prod.metrics.elo
            if elo_diff < self.min_elo_improvement:
                return False, f"Elo improvement insufficient ({elo_diff:.1f} < {self.min_elo_improvement})"

            return True, f"Elo improved by {elo_diff:.1f}"
        else:
            # No production model, promote if metrics are reasonable
            return True, "No current production model"

    def auto_promote(self, model_id: str, version: int) -> ModelStage | None:
        """
        Automatically promote model if criteria are met.

        Returns: New stage if promoted, None otherwise
        """
        model = self.registry.get_model(model_id, version)
        if not model:
            return None

        if model.stage == ModelStage.DEVELOPMENT:
            should_promote, reason = self.evaluate_for_staging(model_id, version)
            if should_promote:
                self.registry.promote(model_id, version, ModelStage.STAGING, reason, "auto_promoter")
                return ModelStage.STAGING

        elif model.stage == ModelStage.STAGING:
            should_promote, reason = self.evaluate_for_production(model_id, version)
            if should_promote:
                self.registry.promote(model_id, version, ModelStage.PRODUCTION, reason, "auto_promoter")
                return ModelStage.PRODUCTION

        return None


# Singleton instance
_model_registry: ModelRegistry | None = None


def get_model_registry(registry_dir: Path | None = None) -> ModelRegistry:
    """Get the global model registry singleton.

    Args:
        registry_dir: Registry directory (only used on first call)

    Returns:
        ModelRegistry instance
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry(registry_dir)
    return _model_registry


def reset_model_registry() -> None:
    """Reset the model registry singleton (for testing)."""
    global _model_registry
    _model_registry = None


def main():
    """Example usage of the model registry."""
    import tempfile

    import torch
    import torch.nn as nn

    # Create a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_dir = Path(tmpdir) / "registry"

        # Initialize registry
        registry = ModelRegistry(registry_dir)

        # Create and save a dummy model
        model = DummyModel()
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model.state_dict(), model_path)

        # Register model
        model_id, version = registry.register_model(
            name="RingRift AI",
            model_path=model_path,
            model_type=ModelType.POLICY_VALUE,
            description="Main game-playing model",
            metrics=ModelMetrics(elo=1500, games_played=100),
            training_config=TrainingConfig(
                learning_rate=0.001,
                batch_size=256,
                num_residual_blocks=10
            ),
            tags=["baseline", "v1"]
        )

        print(f"Registered: {model_id}:v{version}")

        # Get model
        registered = registry.get_model(model_id, version)
        print(f"Retrieved: {registered.name} (Stage: {registered.stage.value})")

        # Promote to staging
        registry.promote(model_id, version, ModelStage.STAGING, "Initial evaluation")

        # Update metrics
        registry.update_metrics(model_id, version, ModelMetrics(
            elo=1520,
            games_played=200,
            win_rate=0.55
        ))

        # Promote to production
        registry.promote(model_id, version, ModelStage.PRODUCTION, "Passed evaluation")

        # List production models
        production = registry.list_models(stage=ModelStage.PRODUCTION)
        print(f"Production models: {len(production)}")

        # Get stage history
        history = registry.get_stage_history(model_id, version)
        print(f"Stage transitions: {len(history)}")
        for h in history:
            print(f"  {h['from_stage']} -> {h['to_stage']}: {h['reason']}")

        # Register another version
        model_id2, version2 = registry.register_model(
            name="RingRift AI",
            model_path=model_path,
            model_id=model_id,  # Same model ID, new version
            metrics=ModelMetrics(elo=1550, games_played=150),
            tags=["improved"]
        )
        print(f"Registered new version: {model_id2}:v{version2}")

        # Auto-promotion test
        auto_promoter = AutoPromoter(registry, min_elo_improvement=10)

        # First promote to staging manually
        registry.promote(model_id2, version2, ModelStage.STAGING, "Testing")

        # Try auto-promote to production
        new_stage = auto_promoter.auto_promote(model_id2, version2)
        if new_stage:
            print(f"Auto-promoted to {new_stage.value}")

        # Check that old production model was archived
        old_model = registry.get_model(model_id, version)
        print(f"Old model stage: {old_model.stage.value}")


if __name__ == "__main__":
    main()
