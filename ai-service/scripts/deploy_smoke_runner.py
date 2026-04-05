"""Helpers for running deploy smoke tests on remote cluster nodes."""

from __future__ import annotations

import shlex
from typing import Any


def build_remote_deploy_smoke_cmd(
    node_path: str,
    *,
    expected_commit: str | None = None,
    venv_activate: str = "source venv/bin/activate",
) -> str:
    """Build the remote shell command for deploy_smoke_test.py."""
    activate = (venv_activate or ":").strip() or ":"
    # Keep node_path unquoted so remote shells can expand leading "~/".
    cmd = (
        f"cd {node_path} && {activate} && "
        "PYTHONPATH=. python3 scripts/deploy_smoke_test.py"
    )
    if expected_commit:
        cmd += f" --expected-commit {shlex.quote(expected_commit)}"
    return cmd


def summarize_remote_command_failure(stdout: str, stderr: str, *, max_lines: int = 12) -> str:
    """Summarize remote command output into a compact failure message."""
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part.strip())
    if not combined:
        return "no output"
    lines = [line.rstrip() for line in combined.splitlines() if line.strip()]
    return " | ".join(lines[-max_lines:])


async def run_remote_deploy_smoke_test(
    client: Any,
    *,
    node_name: str,
    node_path: str,
    expected_commit: str | None = None,
    venv_activate: str = "source venv/bin/activate",
    timeout: float = 300.0,
) -> tuple[bool, str]:
    """Run deploy_smoke_test.py on a remote node."""
    result = await client.run_async(
        build_remote_deploy_smoke_cmd(
            node_path,
            expected_commit=expected_commit,
            venv_activate=venv_activate,
        ),
        timeout=timeout,
    )
    if result.returncode == 0:
        return True, "deploy smoke passed"

    summary = summarize_remote_command_failure(result.stdout, result.stderr)
    return False, f"deploy smoke failed on {node_name}: {summary}"
