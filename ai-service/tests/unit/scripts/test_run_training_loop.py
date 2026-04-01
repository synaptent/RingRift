"""Focused tests for manual training-loop pipeline triggering."""

from __future__ import annotations

import argparse
import importlib
import sys
from types import SimpleNamespace


def _load_module():
    return importlib.import_module("scripts.run_training_loop")


def test_trigger_manual_pipeline_uses_publish_sync(monkeypatch):
    """Manual trigger must publish synchronously from this sync CLI path."""
    training_loop = _load_module()
    publish_calls = []
    fake_router_module = SimpleNamespace(
        publish_sync=lambda **kwargs: publish_calls.append(kwargs),
        StageEvent=SimpleNamespace(SELFPLAY_COMPLETE="SELFPLAY_COMPLETE"),
    )
    monkeypatch.setitem(sys.modules, "app.coordination.event_router", fake_router_module)

    args = argparse.Namespace(
        board_type="hex8",
        num_players=2,
        dry_run=False,
    )

    assert training_loop.trigger_manual_pipeline(args) is True
    assert len(publish_calls) == 1
    call = publish_calls[0]
    assert call["event_type"] == "SELFPLAY_COMPLETE"
    assert call["source"] == "manual_trigger"
    assert call["payload"]["config_key"] == "hex8_2p"
    assert call["payload"]["board_type"] == "hex8"
    assert call["payload"]["num_players"] == 2
    assert call["payload"]["games_completed"] == 0


def test_trigger_manual_pipeline_dry_run_skips_publish(monkeypatch):
    """Dry-run should not emit any event."""
    training_loop = _load_module()
    publish_calls = []
    fake_router_module = SimpleNamespace(
        publish_sync=lambda **kwargs: publish_calls.append(kwargs),
        StageEvent=SimpleNamespace(SELFPLAY_COMPLETE="SELFPLAY_COMPLETE"),
    )
    monkeypatch.setitem(sys.modules, "app.coordination.event_router", fake_router_module)

    args = argparse.Namespace(
        board_type="square8",
        num_players=4,
        dry_run=True,
    )

    assert training_loop.trigger_manual_pipeline(args) is True
    assert publish_calls == []
