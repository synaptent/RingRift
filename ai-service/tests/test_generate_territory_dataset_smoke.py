from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest


# Root of the ai-service package (so that `python -m app...` works reliably)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_generate_territory_dataset_mixed_smoke() -> None:
    """Smoke-test the territory dataset CLI in mixed-engine mode.

    This directly exercises the `python -m app.training.generate_territory_dataset`
    entrypoint with a small configuration and asserts that:
    - the process exits with code 0;
    - stderr does not contain the TerritoryMutator/GameEngine divergence message;
    - the output file is created and non-empty.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "rr-territory-debug.jsonl")

        cmd = [
            sys.executable,
            "-m",
            "app.training.generate_territory_dataset",
            "--num-games",
            "10",
            "--output",
            output_path,
            "--board-type",
            "square8",
            "--engine-mode",
            "mixed",
            "--num-players",
            "2",
            "--max-moves",
            "200",
            "--seed",
            "42",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

        if result.returncode != 0:
            pytest.fail(
                "generate_territory_dataset CLI failed with exit code "
                f"{result.returncode}, stderr={result.stderr}"
            )

        stderr = result.stderr or ""
        assert (
            "TerritoryMutator diverged from GameEngine.apply_move" not in stderr
        ), (
            "Detected TerritoryMutator/GameEngine divergence in "
            "generate_territory_dataset stderr"
        )

        assert os.path.exists(output_path), "Output file was not created"
        assert os.path.getsize(output_path) > 0, "Output file is empty"