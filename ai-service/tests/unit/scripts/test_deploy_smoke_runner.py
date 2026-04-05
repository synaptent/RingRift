"""Tests for remote deploy smoke-test helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.deploy_smoke_runner import (
    build_remote_deploy_smoke_cmd,
    run_remote_deploy_smoke_test,
    summarize_remote_command_failure,
)


def test_build_remote_deploy_smoke_cmd_includes_expected_commit():
    cmd = build_remote_deploy_smoke_cmd(
        "~/ringrift/ai-service",
        expected_commit="abcdef1234567890",
        venv_activate="source venv/bin/activate",
    )

    assert "cd ~/ringrift/ai-service" in cmd
    assert "source venv/bin/activate" in cmd
    assert "venv/bin/python" in cmd
    assert 'PYTHONPATH=. "$_ringrift_python" scripts/deploy_smoke_test.py' in cmd
    assert "--expected-commit abcdef1234567890" in cmd


def test_build_remote_deploy_smoke_cmd_prefers_project_python_when_no_activate():
    cmd = build_remote_deploy_smoke_cmd(
        "~/Development/RingRift/ai-service",
        expected_commit=None,
        venv_activate=":",
    )

    assert "cd ~/Development/RingRift/ai-service" in cmd
    assert "if [ -x venv/bin/python ]; then _ringrift_python=venv/bin/python;" in cmd
    assert "elif command -v python3.11 >/dev/null 2>&1;" in cmd
    assert 'PYTHONPATH=. "$_ringrift_python" scripts/deploy_smoke_test.py' in cmd


def test_summarize_remote_command_failure_uses_tail_lines():
    summary = summarize_remote_command_failure(
        "line1\nline2\nline3",
        "err1\nerr2",
        max_lines=2,
    )

    assert summary == "err1 | err2"


@pytest.mark.asyncio
async def test_run_remote_deploy_smoke_test_success():
    client = MagicMock()
    client.run_async = AsyncMock(return_value=MagicMock(returncode=0, stdout="ok\n", stderr=""))

    ok, message = await run_remote_deploy_smoke_test(
        client,
        node_name="lambda-gh200-1",
        node_path="~/ringrift/ai-service",
        expected_commit="abcdef1234567890",
    )

    assert ok is True
    assert message == "deploy smoke passed"


@pytest.mark.asyncio
async def test_run_remote_deploy_smoke_test_failure():
    client = MagicMock()
    client.run_async = AsyncMock(
        return_value=MagicMock(
            returncode=1,
            stdout="header\n[FAIL] Imports (0.1s): broken import\nFAILED: 1 issues\n",
            stderr="",
        )
    )

    ok, message = await run_remote_deploy_smoke_test(
        client,
        node_name="lambda-gh200-1",
        node_path="~/ringrift/ai-service",
        expected_commit="abcdef1234567890",
    )

    assert ok is False
    assert "deploy smoke failed on lambda-gh200-1" in message
    assert "broken import" in message
