"""Delivery verification handlers for P2P orchestrator.

Provides endpoints for verifying data deliveries on target nodes,
including file existence and checksum verification.

December 2025: Created as part of Phase 3 infrastructure improvements.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.db_helpers import p2p_db_connection
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import (
    handler_timeout,
    HANDLER_TIMEOUT_DELIVERY,
    HANDLER_TIMEOUT_GOSSIP,
)

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)

__all__ = [
    "DeliveryHandlersMixin",
    "compute_file_checksum",
]


def compute_file_checksum(file_path: Path | str, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hex digest of the file contents
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class DeliveryHandlersMixin(BaseP2PHandler):
    """Mixin providing delivery verification handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    To use, inherit from this mixin in P2POrchestrator and register routes:

        app.router.add_post('/delivery/verify', self.handle_delivery_verify)
        app.router.add_get('/delivery/status/{node_id}', self.handle_delivery_status)
    """

    @handler_timeout(HANDLER_TIMEOUT_DELIVERY)
    async def handle_delivery_verify(
        self: "P2POrchestrator",
        request: web.Request,
    ) -> web.Response:
        """POST /delivery/verify - Verify file received and matches checksum.

        Request body:
        {
            "file_path": "/path/to/file.pth",
            "expected_checksum": "sha256:abc123...",  // optional
            "file_type": "model"  // optional, for type-specific validation
        }

        Response:
        {
            "verified": true,
            "file_path": "/path/to/file.pth",
            "exists": true,
            "actual_checksum": "sha256:abc123...",
            "checksum_match": true,  // only if expected_checksum provided
            "file_size": 12345678,
            "node_id": "runpod-h100"
        }
        """
        try:
            data = await request.json()
            file_path_str = data.get("file_path")
            expected_checksum = data.get("expected_checksum", "")
            file_type = data.get("file_type", "")

            if not file_path_str:
                return self.error_response(
                    "Missing required field: file_path",
                    status=400,
                )

            file_path = Path(file_path_str)

            # Check if file exists
            if not file_path.exists():
                return self.json_response({
                    "verified": False,
                    "file_path": str(file_path),
                    "exists": False,
                    "error": "File not found",
                    "node_id": self.node_id,
                })

            # Get file info
            file_size = file_path.stat().st_size

            # Compute checksum
            try:
                actual_checksum = f"sha256:{compute_file_checksum(file_path)}"
            except (OSError, IOError) as e:
                return self.json_response({
                    "verified": False,
                    "file_path": str(file_path),
                    "exists": True,
                    "file_size": file_size,
                    "error": f"Failed to compute checksum: {e}",
                    "node_id": self.node_id,
                })

            # Compare checksums if expected was provided
            checksum_match = None
            if expected_checksum:
                # Normalize expected checksum format
                if not expected_checksum.startswith("sha256:"):
                    expected_checksum = f"sha256:{expected_checksum}"
                checksum_match = actual_checksum.lower() == expected_checksum.lower()

            # Perform type-specific validation if requested
            type_validation = None
            if file_type:
                type_validation = await self._validate_file_type(file_path, file_type)

            response: dict[str, Any] = {
                "verified": file_path.exists() and (checksum_match is None or checksum_match),
                "file_path": str(file_path),
                "exists": True,
                "actual_checksum": actual_checksum,
                "file_size": file_size,
                "node_id": self.node_id,
            }

            if checksum_match is not None:
                response["checksum_match"] = checksum_match
                response["expected_checksum"] = expected_checksum

            if type_validation is not None:
                response["type_validation"] = type_validation

            return self.json_response(response)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error verifying delivery: {e}")
            return self.error_response(str(e), status=500)

    async def _validate_file_type(
        self: "P2POrchestrator",
        file_path: Path,
        file_type: str,
    ) -> dict[str, Any]:
        """Perform type-specific file validation.

        Args:
            file_path: Path to the file
            file_type: Type of file (model, npz, games)

        Returns:
            Dictionary with validation results
        """
        validation: dict[str, Any] = {"type": file_type, "valid": True}

        try:
            if file_type == "model":
                # Check if it's a valid PyTorch checkpoint
                # Feb 2026: Wrap in to_thread — torch.load blocks event loop 1-5s
                try:
                    import torch

                    def _load_checkpoint() -> dict:
                        return torch.load(file_path, map_location="cpu", weights_only=True)

                    checkpoint = await asyncio.to_thread(_load_checkpoint)
                    validation["has_state_dict"] = "state_dict" in checkpoint
                    validation["has_metadata"] = "metadata" in checkpoint
                except Exception as e:
                    validation["valid"] = False
                    validation["error"] = str(e)

            elif file_type == "npz":
                # Check if it's a valid NPZ file
                # Feb 2026: Wrap in to_thread — np.load blocks event loop 1-3s
                try:
                    import numpy as np

                    def _load_npz() -> dict:
                        with np.load(file_path) as data:
                            return {
                                "arrays": list(data.keys()),
                                "total_samples": data["states"].shape[0] if "states" in data else 0,
                            }

                    npz_info = await asyncio.to_thread(_load_npz)
                    validation["arrays"] = npz_info["arrays"]
                    validation["total_samples"] = npz_info["total_samples"]
                except Exception as e:
                    validation["valid"] = False
                    validation["error"] = str(e)

            elif file_type == "games":
                # Check if it's a valid SQLite database
                try:
                    def _validate_sqlite_db() -> tuple[list[str], int]:
                        """Blocking SQLite validation - runs in thread pool."""
                        with p2p_db_connection(file_path) as conn:
                            cursor = conn.execute(
                                "SELECT name FROM sqlite_master WHERE type='table'"
                            )
                            tables = [row[0] for row in cursor.fetchall()]
                            game_count = 0
                            if "games" in tables:
                                cursor = conn.execute("SELECT COUNT(*) FROM games")
                                game_count = cursor.fetchone()[0]
                            return tables, game_count

                    # Run blocking SQLite in thread pool
                    tables, game_count = await asyncio.to_thread(_validate_sqlite_db)
                    validation["tables"] = tables
                    validation["has_games_table"] = "games" in tables
                    if "games" in tables:
                        validation["game_count"] = game_count
                except Exception as e:
                    validation["valid"] = False
                    validation["error"] = str(e)

        except Exception as e:
            validation["valid"] = False
            validation["error"] = f"Validation error: {e}"

        return validation

    @handler_timeout(HANDLER_TIMEOUT_GOSSIP)
    async def handle_delivery_status(
        self: "P2POrchestrator",
        request: web.Request,
    ) -> web.Response:
        """GET /delivery/status/{node_id} - Get delivery status for a node.

        Returns summary of recent deliveries to this node, if ledger is available.

        Response:
        {
            "node_id": "runpod-h100",
            "total_verified": 15,
            "failure_rate_24h": 0.05,
            "recent_deliveries": [...]
        }
        """
        try:
            node_id = request.match_info.get("node_id", "")

            if not node_id:
                return self.error_response(
                    "Missing node_id parameter",
                    status=400,
                )

            # Try to get delivery ledger
            try:
                from app.coordination.delivery_ledger import get_delivery_ledger
                ledger = get_delivery_ledger()

                status = ledger.get_node_delivery_status(node_id)
                recent = ledger.get_deliveries_for_node(node_id, limit=10)

                return self.json_response({
                    "node_id": node_id,
                    "total_verified": status.get("total_verified", 0),
                    "failure_rate_24h": status.get("failure_rate_24h", 0.0),
                    "status_counts": status.get("status_counts", {}),
                    "recent_deliveries": [d.to_dict() for d in recent],
                })

            except ImportError:
                return self.error_response(
                    "Delivery ledger not available",
                    status=501,
                    details={"node_id": node_id, "available": False},
                )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error getting delivery status: {e}")
            return self.error_response(str(e), status=500)
