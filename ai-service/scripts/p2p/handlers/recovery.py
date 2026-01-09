"""Recovery HTTP handlers for P2P orchestrator.

January 2026 - P2P Modularization Phase 2b

This mixin provides HTTP handlers for model rollback operations including
status checks, manual rollback execution, and automatic rollback triggers.

Must be mixed into a class that provides:
- self._check_rollback_conditions() -> dict
- self._execute_rollback(config: str, dry_run: bool) -> dict
- self._auto_rollback_check() -> list
"""
from __future__ import annotations

import logging
import os

from aiohttp import web

logger = logging.getLogger(__name__)


class RecoveryHandlersMixin:
    """Mixin providing model rollback HTTP handlers.

    Endpoints:
    - GET /rollback/status - Model rollback status and recommendations
    - POST /rollback/execute - Execute a model rollback
    - POST /rollback/auto - Trigger automatic rollback check for all configs
    """

    async def handle_rollback_status(self, request: web.Request) -> web.Response:
        """GET /rollback/status - Model rollback status and recommendations."""
        try:
            status = await self._check_rollback_conditions()
            return web.json_response(status)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_rollback_execute(self, request: web.Request) -> web.Response:
        """POST /rollback/execute - Execute a model rollback.

        Query params:
            config: Config string like "square8_2p" (required)
            dry_run: If "true", only simulate the rollback (default: false)
        """
        try:
            config = request.query.get("config")
            if not config:
                return web.json_response({"error": "Missing required parameter: config"}, status=400)

            dry_run = request.query.get("dry_run", "").lower() in ("true", "1", "yes")

            result = await self._execute_rollback(config, dry_run=dry_run)
            status_code = 200 if result["success"] else 400
            return web.json_response(result, status=status_code)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_rollback_auto(self, request: web.Request) -> web.Response:
        """POST /rollback/auto - Trigger automatic rollback check for all configs.

        This will check all configs for rollback conditions and execute rollbacks
        for any that meet the criteria.
        """
        try:
            # Temporarily enable auto-rollback for this request
            original_env = os.environ.get("RINGRIFT_AUTO_ROLLBACK", "")
            os.environ["RINGRIFT_AUTO_ROLLBACK"] = "true"

            executed = await self._auto_rollback_check()

            # Restore original env
            if original_env:
                os.environ["RINGRIFT_AUTO_ROLLBACK"] = original_env
            else:
                os.environ.pop("RINGRIFT_AUTO_ROLLBACK", None)

            return web.json_response({
                "executed_rollbacks": executed,
                "count": len(executed),
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
