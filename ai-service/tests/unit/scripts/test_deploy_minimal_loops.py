"""Contracts for scripts/deploy_minimal_loops.sh.

These tests keep the supported minimal-loop deploy manifest machine-checkable
without SSH access. They intentionally exercise only dry-run behavior and static
manifest parsing.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from app.models import BoardType
from app.training.model_config_contract import get_canonical_model_name


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = AI_SERVICE_ROOT / "scripts" / "deploy_minimal_loops.sh"
EXPECTED_CONFIGS = {"hex8_2p", "square8_2p", "square8_3p", "square8_4p"}


def _script_text() -> str:
    return SCRIPT_PATH.read_text()


def _node_entries() -> list[tuple[str, str, str, str]]:
    match = re.search(r"^NODES=\(\n(?P<body>.*?)\n\)", _script_text(), re.MULTILINE | re.DOTALL)
    assert match, "deploy_minimal_loops.sh must define a NODES=(...) manifest"

    entries: list[tuple[str, str, str, str]] = []
    for raw in re.findall(r'^\s+"([^"]+)"', match.group("body"), re.MULTILINE):
        parts = raw.split("|", 3)
        assert len(parts) == 4, f"node entry must be ip|config|workdir|args: {raw}"
        entries.append((parts[0], parts[1], parts[2], parts[3]))
    return entries


def _arg_value(args: str, flag: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(flag)}\s+(\S+)", args)
    assert match, f"{flag} missing from args: {args}"
    return match.group(1)


def test_dry_run_outputs_supported_configs() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--dry-run"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Selected nodes deployed and restarted." in result.stdout
    assert result.stdout.count("[DRY] Would deploy scripts/minimal_alphazero_loop.py and restart") == 4
    for config in EXPECTED_CONFIGS:
        assert f"=== {config} (" in result.stdout


def test_dry_run_only_filters_single_config() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--dry-run", "--only", "square8_3p"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "=== square8_3p (" in result.stdout
    assert "=== hex8_2p (" not in result.stdout
    assert result.stdout.count("[DRY] Would deploy") == 1


def test_node_manifest_is_valid() -> None:
    entries = _node_entries()
    configs = {config for _ip, config, _workdir, _args in entries}
    ips = [ip for ip, _config, _workdir, _args in entries]

    assert configs == EXPECTED_CONFIGS
    assert len(ips) == len(set(ips))

    for ip, config, workdir, args in entries:
        assert re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", ip), ip
        assert workdir.startswith("data/minimal_loop_")
        assert _arg_value(args, "--board-type") in {"hex8", "square8"}
        assert _arg_value(args, "--num-players") in {"2", "3", "4"}
        assert "--iterations" in args
        assert "--selfplay-budget" in args
        assert "--eval-budget" in args
        assert config == f"{_arg_value(args, '--board-type')}_{_arg_value(args, '--num-players')}p"


def test_config_names_match_canonical_model_names() -> None:
    assert "--model models/canonical_${config}.pth" in _script_text()

    for _ip, config, _workdir, args in _node_entries():
        board_type = BoardType(_arg_value(args, "--board-type"))
        num_players = int(_arg_value(args, "--num-players"))
        assert get_canonical_model_name(board_type, num_players) == f"canonical_{config}.pth"
