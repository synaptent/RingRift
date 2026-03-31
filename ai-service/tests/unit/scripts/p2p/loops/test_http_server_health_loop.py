"""Focused tests for HTTP server health loop failure handling."""

from __future__ import annotations

import builtins
import logging

import pytest


@pytest.mark.asyncio
async def test_probe_local_health_fails_closed_when_aiohttp_missing(monkeypatch, caplog):
    """Missing aiohttp should be treated as an unhealthy probe, not success."""
    from scripts.p2p.loops.http_server_health_loop import (
        HttpServerHealthConfig,
        HttpServerHealthLoop,
    )

    loop = HttpServerHealthLoop(
        port=8770,
        config=HttpServerHealthConfig(use_isolated_health_port=False),
    )

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aiohttp":
            raise ImportError("aiohttp not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.ERROR):
        assert await loop._probe_local_health() is False

    assert "failing health probe closed" in caplog.text
