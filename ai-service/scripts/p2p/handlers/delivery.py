"""Delivery verification handlers for P2P orchestrator.

Provides endpoints for verifying data deliveries on target nodes,
including file existence and checksum verification.

December 2025: Created as part of Phase 3 infrastructure improvements.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

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
                try:
                    import torch
                    checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
                    validation["has_state_dict"] = "state_dict" in checkpoint
                    validation["has_metadata"] = "metadata" in checkpoint
                except Exception as e:
                    validation["valid"] = False
                    validation["error"] = str(e)

            elif file_type == "npz":
                # Check if it's a valid NPZ file
                try:
                    import numpy as np
                    with np.load(file_path) as data:
                        validation["arrays"] = list(data.keys())
                        validation["total_samples"] = data["states"].shape[0] if "states" in data else 0
                except Exception as e:
                    validation["valid"] = False
                    validation["error"] = str(e)

            elif file_type == "games":
                # Check if it's a valid SQLite database
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(file_path))
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                    validation["tables"] = tables
                    validation["has_games_table"] = "games" in tables
                    if "games" in tables:
                        cursor = conn.execute("SELECT COUNT(*) FROM games")
                        validation["game_count"] = cursor.fetchone()[0]
                    conn.close()
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
