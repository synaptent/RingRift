"""
Unified Model Discovery API for RingRift.

This module provides a single source of truth for model discovery, eliminating
the fragmented discovery logic spread across multiple scripts.

Features:
- Board type detection from sidecar JSON, checkpoint metadata, or filename
- Consistent ModelInfo dataclass returned everywhere
- Automatic sidecar generation for discovered models
- Integration with ELO database for enriched model info

Usage:
    from app.models.discovery import discover_models, ModelInfo

    # Discover all square8 2-player models
    models = discover_models(board_type="square8", num_players=2)
    for model in models:
        print(f"{model.name}: {model.board_type}, elo={model.elo}")

    # Write sidecar for a model
    write_model_sidecar(model_path, board_type="square8", num_players=2)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Standard model directories
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = AI_SERVICE_ROOT / "models"


@dataclass
class ModelInfo:
    """Unified model information used across all components.

    This is the canonical model representation. All discovery functions
    return this dataclass, ensuring consistent behavior.
    """
    path: str
    name: str
    model_type: str  # "nn" or "nnue"
    board_type: str  # "square8", "square19", "hexagonal"
    num_players: int = 2
    elo: Optional[float] = None
    architecture_version: Optional[str] = None
    created_at: Optional[str] = None
    size_bytes: int = 0
    source: str = "filename"  # How board_type was determined: "sidecar", "checkpoint", "filename"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def detect_board_type_from_name(name: str) -> Tuple[Optional[str], Optional[int]]:
    """Detect board type and num_players from model filename.

    Returns:
        Tuple of (board_type, num_players) or (None, None) if unknown.
    """
    name_lower = name.lower()

    # Board type patterns
    board_type = None
    if any(p in name_lower for p in ["sq8", "square8", "_8x8"]):
        board_type = "square8"
    elif any(p in name_lower for p in ["sq19", "square19", "_19x19"]):
        board_type = "square19"
    elif any(p in name_lower for p in ["hex", "hexagonal"]):
        board_type = "hexagonal"
    elif "ringrift_v" in name_lower and "sq" not in name_lower and "hex" not in name_lower:
        # Legacy ringrift_v* models are typically square8
        board_type = "square8"

    # Num players pattern
    num_players = None
    player_match = re.search(r"_(\d)p[_\.]", name_lower)
    if player_match:
        num_players = int(player_match.group(1))
    elif "_2p" in name_lower or "2p_" in name_lower:
        num_players = 2
    elif "_3p" in name_lower or "3p_" in name_lower:
        num_players = 3
    elif "_4p" in name_lower or "4p_" in name_lower:
        num_players = 4
    else:
        num_players = 2  # Default

    return board_type, num_players


def read_model_sidecar(model_path: Path) -> Optional[Dict[str, Any]]:
    """Read sidecar JSON for a model if it exists.

    Sidecar file is {model_path}.json (e.g., model.pth -> model.pth.json)
    """
    sidecar_path = Path(str(model_path) + ".json")
    if sidecar_path.exists():
        try:
            with open(sidecar_path) as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Failed to read sidecar {sidecar_path}: {e}")
    return None


def write_model_sidecar(
    model_path: Path,
    board_type: str,
    num_players: int = 2,
    elo: Optional[float] = None,
    architecture_version: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write sidecar JSON for a model.

    Args:
        model_path: Path to the model file
        board_type: Board type string
        num_players: Number of players
        elo: Optional ELO rating
        architecture_version: Optional architecture version
        extra_metadata: Additional metadata to include

    Returns:
        Path to the written sidecar file
    """
    sidecar_path = Path(str(model_path) + ".json")

    data = {
        "model_path": str(model_path),
        "model_name": model_path.stem,
        "board_type": board_type,
        "num_players": num_players,
        "elo": elo,
        "architecture_version": architecture_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sidecar_version": "1.0",
    }

    if extra_metadata:
        data.update(extra_metadata)

    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.debug(f"Wrote sidecar: {sidecar_path}")
    return sidecar_path


def extract_metadata_from_checkpoint(model_path: Path) -> Optional[Dict[str, Any]]:
    """Extract metadata from a .pth checkpoint without fully loading the model.

    This uses torch.load with weights_only=False to access metadata.
    """
    try:
        # Load only the checkpoint structure, not the weights
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        metadata = {}

        # Check for versioning metadata
        if "_versioning_metadata" in checkpoint:
            vm = checkpoint["_versioning_metadata"]
            metadata["architecture_version"] = vm.get("architecture_version")
            metadata["model_class"] = vm.get("model_class")
            metadata["created_at"] = vm.get("created_at")

            # Extract board_type and num_players from config if present
            config = vm.get("config", {})
            if "board_type" in config:
                metadata["board_type"] = config["board_type"]
            if "num_players" in config:
                metadata["num_players"] = config["num_players"]

        # Check for direct metadata keys
        for key in ["board_type", "num_players", "architecture_version"]:
            if key in checkpoint and key not in metadata:
                metadata[key] = checkpoint[key]

        return metadata if metadata else None

    except Exception as e:
        logger.debug(f"Failed to extract metadata from {model_path}: {e}")
        return None


def get_model_info(model_path: Path, model_type: str = "nn") -> Optional[ModelInfo]:
    """Get ModelInfo for a single model, trying all detection methods.

    Detection order:
    1. Sidecar JSON (fastest, most reliable)
    2. Checkpoint metadata (requires loading, but authoritative)
    3. Filename parsing (fallback)

    Args:
        model_path: Path to the model file
        model_type: "nn" or "nnue"

    Returns:
        ModelInfo or None if board type couldn't be determined
    """
    if not model_path.exists():
        return None

    board_type = None
    num_players = 2
    elo = None
    architecture_version = None
    source = "filename"

    # 1. Try sidecar JSON first
    sidecar = read_model_sidecar(model_path)
    if sidecar:
        board_type = sidecar.get("board_type")
        num_players = sidecar.get("num_players", 2)
        elo = sidecar.get("elo")
        architecture_version = sidecar.get("architecture_version")
        source = "sidecar"

    # 2. Try checkpoint metadata
    if not board_type:
        checkpoint_meta = extract_metadata_from_checkpoint(model_path)
        if checkpoint_meta:
            board_type = checkpoint_meta.get("board_type")
            num_players = checkpoint_meta.get("num_players", 2)
            architecture_version = checkpoint_meta.get("architecture_version")
            source = "checkpoint"

    # 3. Fall back to filename parsing
    if not board_type:
        board_type, parsed_players = detect_board_type_from_name(model_path.stem)
        if parsed_players:
            num_players = parsed_players
        source = "filename"

    if not board_type:
        logger.warning(f"Could not determine board type for {model_path.name}")
        return None

    return ModelInfo(
        path=str(model_path),
        name=model_path.stem,
        model_type=model_type,
        board_type=board_type,
        num_players=num_players,
        elo=elo,
        architecture_version=architecture_version,
        size_bytes=model_path.stat().st_size,
        source=source,
    )


def discover_models(
    models_dir: Optional[Path] = None,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    model_type: Optional[str] = None,
    include_unknown: bool = False,
    generate_sidecars: bool = False,
) -> List[ModelInfo]:
    """Discover all models matching the given criteria.

    This is the canonical discovery function that should be used everywhere.

    Args:
        models_dir: Directory to scan (default: AI_SERVICE_ROOT/models)
        board_type: Filter by board type ("square8", "square19", "hexagonal")
        num_players: Filter by number of players (2, 3, 4)
        model_type: Filter by model type ("nn", "nnue", or None for both)
        include_unknown: Include models with unknown board type
        generate_sidecars: Auto-generate sidecar JSON for models without one

    Returns:
        List of ModelInfo objects for matching models
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir = Path(models_dir)
    results = []

    # Discover NN models (.pth files)
    if model_type in (None, "nn"):
        for f in models_dir.glob("*.pth"):
            if f.stem.startswith("."):
                continue

            info = get_model_info(f, "nn")
            if info is None:
                if include_unknown:
                    # Create placeholder with unknown board type
                    try:
                        size_bytes = f.stat().st_size
                    except (FileNotFoundError, OSError):
                        # File was deleted/moved during discovery
                        continue
                    info = ModelInfo(
                        path=str(f),
                        name=f.stem,
                        model_type="nn",
                        board_type="unknown",
                        size_bytes=size_bytes,
                    )
                else:
                    continue

            # Apply filters
            if board_type and info.board_type != board_type:
                continue
            if num_players and info.num_players != num_players:
                continue

            # Generate sidecar if requested and missing
            if generate_sidecars and info.source == "filename":
                write_model_sidecar(
                    f,
                    board_type=info.board_type,
                    num_players=info.num_players,
                    architecture_version=info.architecture_version,
                )
                info.source = "sidecar"

            results.append(info)

    # Discover NNUE models (.pt files in nnue subdirectory)
    if model_type in (None, "nnue"):
        nnue_dir = models_dir / "nnue"
        if nnue_dir.exists():
            for f in nnue_dir.glob("*.pt"):
                if f.stem.startswith("."):
                    continue

                info = get_model_info(f, "nnue")
                if info is None:
                    if include_unknown:
                        try:
                            size_bytes = f.stat().st_size
                        except (FileNotFoundError, OSError):
                            # File was deleted/moved during discovery
                            continue
                        info = ModelInfo(
                            path=str(f),
                            name=f.stem,
                            model_type="nnue",
                            board_type="unknown",
                            size_bytes=size_bytes,
                        )
                    else:
                        continue

                # Apply filters
                if board_type and info.board_type != board_type:
                    continue
                if num_players and info.num_players != num_players:
                    continue

                if generate_sidecars and info.source == "filename":
                    write_model_sidecar(
                        f,
                        board_type=info.board_type,
                        num_players=info.num_players,
                        architecture_version=info.architecture_version,
                    )
                    info.source = "sidecar"

                results.append(info)

    return results


def generate_all_sidecars(models_dir: Optional[Path] = None, overwrite: bool = False) -> int:
    """Generate sidecar JSON files for all models.

    This is a migration utility to bootstrap sidecars for existing models.

    Args:
        models_dir: Directory to scan
        overwrite: If True, regenerate sidecars even if they exist

    Returns:
        Number of sidecars generated
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir = Path(models_dir)
    count = 0

    for f in models_dir.glob("*.pth"):
        if f.stem.startswith("."):
            continue

        sidecar_path = Path(str(f) + ".json")
        if sidecar_path.exists() and not overwrite:
            continue

        info = get_model_info(f, "nn")
        if info and info.board_type != "unknown":
            write_model_sidecar(
                f,
                board_type=info.board_type,
                num_players=info.num_players,
                elo=info.elo,
                architecture_version=info.architecture_version,
            )
            count += 1
            logger.info(f"Generated sidecar for {f.name}")

    # NNUE models
    nnue_dir = models_dir / "nnue"
    if nnue_dir.exists():
        for f in nnue_dir.glob("*.pt"):
            if f.stem.startswith("."):
                continue

            sidecar_path = Path(str(f) + ".json")
            if sidecar_path.exists() and not overwrite:
                continue

            info = get_model_info(f, "nnue")
            if info and info.board_type != "unknown":
                write_model_sidecar(
                    f,
                    board_type=info.board_type,
                    num_players=info.num_players,
                    elo=info.elo,
                    architecture_version=info.architecture_version,
                )
                count += 1
                logger.info(f"Generated sidecar for {f.name}")

    return count


# CLI for sidecar generation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Discovery and Sidecar Generator")
    parser.add_argument("--generate-sidecars", action="store_true", help="Generate sidecar JSON for all models")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecars")
    parser.add_argument("--list", action="store_true", help="List all discovered models")
    parser.add_argument("--board", type=str, help="Filter by board type")
    parser.add_argument("--players", type=int, help="Filter by num players")
    parser.add_argument("--models-dir", type=str, help="Models directory")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    models_dir = Path(args.models_dir) if args.models_dir else None

    if args.generate_sidecars:
        count = generate_all_sidecars(models_dir, overwrite=args.overwrite)
        print(f"Generated {count} sidecar files")

    if args.list or not args.generate_sidecars:
        models = discover_models(
            models_dir=models_dir,
            board_type=args.board,
            num_players=args.players,
        )

        print(f"\nDiscovered {len(models)} models:\n")
        print(f"{'Name':<50} {'Type':<6} {'Board':<10} {'Players':<8} {'Source':<10}")
        print("-" * 90)

        for m in sorted(models, key=lambda x: (x.board_type, x.name)):
            print(f"{m.name[:50]:<50} {m.model_type:<6} {m.board_type:<10} {m.num_players:<8} {m.source:<10}")
