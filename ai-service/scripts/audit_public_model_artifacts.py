#!/usr/bin/env python3
"""Audit the public model artifacts used by quick evaluation.

This is intentionally a small release gate rather than a benchmark. It checks
that the model file exists, its .sha256 sidecar matches, the checkpoint loads,
and versioning metadata agrees with the advertised board/player config.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.utils.torch_utils import compute_model_checksum, safe_load_checkpoint


DEFAULT_ARTIFACTS: dict[str, tuple[str, str, int]] = {
    "hex8_2p": ("models/canonical_hex8_2p.pth", "hex8", 2),
    "square8_2p": ("models/canonical_square8_2p.pth", "square8", 2),
}


@dataclass
class ArtifactAudit:
    config_key: str
    path: str
    exists: bool
    sidecar_exists: bool
    file_sha256: str | None
    sidecar_sha256: str | None
    sidecar_matches: bool
    checkpoint_loads: bool
    model_class: str | None
    architecture_version: str | None
    board_type: str | None
    num_players: int | None
    metadata_ok: bool
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and self.sidecar_exists
            and self.sidecar_matches
            and self.checkpoint_loads
            and self.metadata_ok
            and not self.errors
        )


def _read_sidecar(path: Path) -> str | None:
    if not path.exists():
        return None
    line = path.read_text().strip()
    return line.split()[0] if line else None


def _metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    if value not in ("", 0, None):
        return value
    config = metadata.get("config")
    if isinstance(config, dict):
        return config.get(key)
    return value


def _infer_num_players_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    for key in ("value_fc2.weight", "value_fc3.weight", "value_head.3.weight"):
        tensor = state_dict.get(key)
        shape = getattr(tensor, "shape", None)
        if shape and len(shape) >= 1:
            return int(shape[0])
    return None


def audit_artifact(
    root: Path,
    config_key: str,
    rel_path: str,
    board: str,
    players: int,
) -> ArtifactAudit:
    path = root / rel_path
    sidecar = Path(f"{path}.sha256")
    errors: list[str] = []
    warnings: list[str] = []
    file_sha: str | None = None
    sidecar_sha = _read_sidecar(sidecar)
    checkpoint_loads = False
    metadata: dict[str, Any] = {}
    state_dict: dict[str, Any] = {}

    if not path.exists():
        errors.append(f"missing model file: {path}")
    else:
        file_sha = compute_model_checksum(path)

    if sidecar_sha is None:
        errors.append(f"missing checksum sidecar: {sidecar}")

    sidecar_matches = bool(file_sha and sidecar_sha and file_sha == sidecar_sha)
    if file_sha and sidecar_sha and file_sha != sidecar_sha:
        errors.append(
            f"checksum mismatch: expected {sidecar_sha}, actual {file_sha}"
        )

    if path.exists():
        try:
            checkpoint = safe_load_checkpoint(
                path,
                verify_checksum=True,
                map_location="cpu",
            )
            checkpoint_loads = True
            raw_metadata = checkpoint.get("_versioning_metadata", {})
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
            else:
                warnings.append("_versioning_metadata is not a dict")
            raw_state_dict = checkpoint.get("model_state_dict", {})
            if isinstance(raw_state_dict, dict):
                state_dict = raw_state_dict
        except Exception as exc:  # pragma: no cover - message is the release signal
            errors.append(f"checkpoint load failed: {type(exc).__name__}: {exc}")

    board_type = _metadata_value(metadata, "board_type")
    num_players = _metadata_value(metadata, "num_players")
    if num_players in ("", 0, None):
        inferred_players = _infer_num_players_from_state_dict(state_dict)
        if inferred_players is not None:
            warnings.append(
                "metadata num_players is empty; "
                f"using value-head width inference={inferred_players}"
            )
            num_players = inferred_players
    model_class = metadata.get("model_class")
    arch = metadata.get("architecture_version")

    metadata_ok = True
    if board_type != board:
        metadata_ok = False
        errors.append(f"metadata board_type={board_type!r}, expected {board!r}")
    if num_players != players:
        metadata_ok = False
        errors.append(f"metadata num_players={num_players!r}, expected {players!r}")

    for key in ("board_type", "num_players"):
        top_level = metadata.get(key)
        config = metadata.get("config")
        config_value = config.get(key) if isinstance(config, dict) else None
        if top_level in ("", 0, None) and config_value not in ("", 0, None):
            warnings.append(
                f"top-level metadata {key} is empty; using config.{key}={config_value!r}"
            )

    return ArtifactAudit(
        config_key=config_key,
        path=rel_path,
        exists=path.exists(),
        sidecar_exists=sidecar.exists(),
        file_sha256=file_sha,
        sidecar_sha256=sidecar_sha,
        sidecar_matches=sidecar_matches,
        checkpoint_loads=checkpoint_loads,
        model_class=model_class,
        architecture_version=arch,
        board_type=board_type,
        num_players=num_players if isinstance(num_players, int) else None,
        metadata_ok=metadata_ok,
        errors=errors,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="ai-service root directory",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    audits = [
        audit_artifact(args.root, key, rel_path, board, players)
        for key, (rel_path, board, players) in DEFAULT_ARTIFACTS.items()
    ]

    if args.json:
        print(
            json.dumps(
                [asdict(audit) | {"ok": audit.ok} for audit in audits],
                indent=2,
            )
        )
    else:
        for audit in audits:
            status = "OK" if audit.ok else "FAIL"
            print(f"{status} {audit.config_key}: {audit.path}")
            print(f"  sha256: {audit.file_sha256}")
            print(f"  model:  {audit.model_class} {audit.architecture_version}")
            print(f"  config: {audit.board_type}/{audit.num_players}p")
            for warning in audit.warnings:
                print(f"  warning: {warning}")
            for error in audit.errors:
                print(f"  error: {error}")

    return 0 if all(audit.ok for audit in audits) else 1


if __name__ == "__main__":
    sys.exit(main())
