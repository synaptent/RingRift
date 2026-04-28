"""Tests for the public model artifact audit gate."""

from __future__ import annotations

from pathlib import Path

from scripts import audit_public_model_artifacts as audit


def test_audit_accepts_config_metadata_and_matching_sidecar(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "models" / "canonical_square8_2p.pth"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")
    Path(f"{model}.sha256").write_text("abc123  canonical_square8_2p.pth\n")

    monkeypatch.setattr(audit, "compute_model_checksum", lambda _: "abc123")
    monkeypatch.setattr(
        audit,
        "safe_load_checkpoint",
        lambda *_args, **_kwargs: {
            "_versioning_metadata": {
                "architecture_version": "v2.0.0",
                "model_class": "RingRiftCNN_v2",
                "config": {"board_type": "square8", "num_players": 2},
                "board_type": "",
                "num_players": 0,
            },
            "model_state_dict": {},
        },
    )

    result = audit.audit_artifact(
        tmp_path,
        "square8_2p",
        "models/canonical_square8_2p.pth",
        "square8",
        2,
    )

    assert result.ok
    assert result.board_type == "square8"
    assert result.num_players == 2
    assert any("top-level metadata board_type is empty" in w for w in result.warnings)


def test_audit_infers_players_from_value_head_when_metadata_is_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class FakeTensor:
        shape = (2, 128)

    model = tmp_path / "models" / "canonical_hex8_2p.pth"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")
    Path(f"{model}.sha256").write_text("def456  canonical_hex8_2p.pth\n")

    monkeypatch.setattr(audit, "compute_model_checksum", lambda _: "def456")
    monkeypatch.setattr(
        audit,
        "safe_load_checkpoint",
        lambda *_args, **_kwargs: {
            "_versioning_metadata": {
                "architecture_version": "v2.0.0",
                "model_class": "HexNeuralNet_v2",
                "config": {"board_type": "hex8"},
                "board_type": "",
                "num_players": 0,
            },
            "model_state_dict": {"value_fc2.weight": FakeTensor()},
        },
    )

    result = audit.audit_artifact(
        tmp_path,
        "hex8_2p",
        "models/canonical_hex8_2p.pth",
        "hex8",
        2,
    )

    assert result.ok
    assert result.num_players == 2
    assert any("value-head width inference=2" in w for w in result.warnings)


def test_audit_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "models" / "canonical_square8_2p.pth"
    model.parent.mkdir()
    model.write_bytes(b"checkpoint")
    Path(f"{model}.sha256").write_text("expected  canonical_square8_2p.pth\n")

    monkeypatch.setattr(audit, "compute_model_checksum", lambda _: "actual")
    monkeypatch.setattr(audit, "safe_load_checkpoint", lambda *_args, **_kwargs: {})

    result = audit.audit_artifact(
        tmp_path,
        "square8_2p",
        "models/canonical_square8_2p.pth",
        "square8",
        2,
    )

    assert not result.ok
    assert any("checksum mismatch" in error for error in result.errors)
