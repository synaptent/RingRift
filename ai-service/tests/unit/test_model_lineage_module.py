from __future__ import annotations

import importlib

from app import model_lineage


def test_script_wrapper_reexports_model_lineage_api() -> None:
    cli_module = importlib.import_module("scripts.model_lineage")

    assert cli_module.register_model is model_lineage.register_model
    assert cli_module.update_performance is model_lineage.update_performance
    assert cli_module.main is model_lineage.main


def test_model_lineage_register_and_update_use_configurable_db(tmp_path, monkeypatch) -> None:
    lineage_db = tmp_path / "model_lineage.db"
    model_path = tmp_path / "candidate.pth"
    model_path.write_bytes(b"fake checkpoint bytes")
    monkeypatch.setattr(model_lineage, "LINEAGE_DB_PATH", lineage_db)

    model_id = model_lineage.register_model(
        model_path=str(model_path),
        board_type="hex8",
        num_players=2,
        architecture="test-arch",
        tags=["unit"],
    )
    model_lineage.update_performance(model_id, "elo", 1512.5, context="unit")

    metadata = model_lineage.get_model_metadata(model_id)
    assert metadata is not None
    assert metadata.board_type == "hex8"
    assert metadata.num_players == 2
    assert metadata.architecture == "test-arch"
    assert metadata.performance["elo"] == 1512.5
    assert metadata.tags == ["unit"]
