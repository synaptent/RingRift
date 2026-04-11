"""Canonical event-system contracts for the active coordination path.

This contract intentionally targets the supported active event catalog rather
than every historical `DataEventType` value. The enum includes legacy and
one-off operational events that are not part of the reusable supported path.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]
COORDINATION_DIR = AI_SERVICE_ROOT / "app" / "coordination"
P2P_DIR = AI_SERVICE_ROOT / "scripts" / "p2p"
INFRASTRUCTURE_ALLOWLIST = {
    COORDINATION_DIR / "event_router_compat_emitters.py",
}
SUPPORTED_EVENT_CATALOG = (
    "training_completed",
    "evaluation_completed",
    "model_promoted",
    "sync_completed",
    "new_games",
    "curriculum_rebalanced",
    "elo_velocity_changed",
    "training_started",
    "selfplay_complete",
)


def _iter_python_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts or "archive" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def _find_runtime_emit_event_calls(*roots: Path) -> list[tuple[Path, int, str]]:
    """Return runtime call sites that still invoke emit_event directly."""

    violations: list[tuple[Path, int, str]] = []
    for path in _iter_python_files(*roots):
        if path in INFRASTRUCTURE_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "emit_event":
                violations.append((path, node.lineno, "emit_event"))
            elif isinstance(func, ast.Attribute) and func.attr == "emit_event":
                violations.append((path, node.lineno, "attribute.emit_event"))
    return violations


def _scan_file_for_patterns(
    file_path: Path, patterns: list[str]
) -> list[tuple[str, int]]:
    matches: list[tuple[str, int]] = []
    content = file_path.read_text()
    for line_num, line in enumerate(content.splitlines(), start=1):
        for pattern in patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                matches.append((match.group(1), line_num))
    return matches


def _find_emitters() -> dict[str, list[tuple[Path, int]]]:
    emitters: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    emit_patterns = [
        r'emit\(["\']([a-z_]+)["\']',
        r'publish\(["\']([a-z_]+)["\']',
        r'_emit_([a-z_]+)_event',
        r'emit_([a-z_]+)\(',
        r'DataEventType\.([A-Z_]+)\.value',
        r'DataEventType\.([A-Z_]+)',
    ]
    for path in _iter_python_files(COORDINATION_DIR, P2P_DIR):
        if path in INFRASTRUCTURE_ALLOWLIST:
            continue
        for event_name, line_num in _scan_file_for_patterns(path, emit_patterns):
            emitters[event_name.lower()].append((path, line_num))
    return emitters


def _find_subscribers() -> dict[str, list[tuple[Path, int]]]:
    subscribers: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    subscribe_patterns = [
        r'subscribe\(["\']([a-z_]+)["\']',
        r'subscribe_to_event\(["\']([a-z_]+)["\']',
        r'\.subscribe\(["\']([a-z_]+)["\']',
        r'on_event\(["\']([a-z_]+)["\']',
        r'["\']([a-z_]+)["\']\s*:\s*self\._on_',
        r'DataEventType\.([A-Z_]+)\.value.*subscribe',
        r'DataEventType\.([A-Z_]+)[,\)]',
        r'_on_([a-z_]+)\s*\(',
    ]
    for path in _iter_python_files(COORDINATION_DIR, P2P_DIR):
        if path in INFRASTRUCTURE_ALLOWLIST:
            continue
        for event_name, line_num in _scan_file_for_patterns(path, subscribe_patterns):
            subscribers[event_name.lower()].append((path, line_num))
    return subscribers


def _has_supported_match(
    event_name: str,
    catalog_entry: str,
) -> bool:
    synonyms = {
        "sync_completed": {"sync_completed", "data_sync_completed"},
        "new_games": {"new_games", "new_games_available"},
    }
    accepted = synonyms.get(catalog_entry, {catalog_entry})
    return event_name in accepted


def test_no_runtime_emit_event_calls_in_active_coordination_path() -> None:
    """Active coordination/P2P code should emit through the consolidated helper."""

    violations = _find_runtime_emit_event_calls(COORDINATION_DIR, P2P_DIR)
    assert not violations, "\n".join(
        f"{path.relative_to(AI_SERVICE_ROOT)}:{lineno} uses {kind}"
        for path, lineno, kind in violations
    )


def test_supported_event_catalog_has_subscribers() -> None:
    """Every supported-path event should have at least one subscriber."""

    subscribers = _find_subscribers()

    missing = []
    for event_name in SUPPORTED_EVENT_CATALOG:
        found = any(
            _has_supported_match(candidate, event_name)
            for candidate in subscribers
        )
        if not found:
            missing.append(event_name)

    assert not missing, f"Supported events without subscribers: {missing}"


def test_supported_emitted_events_are_not_orphaned() -> None:
    """No supported-path emitted event should be left without a subscriber."""

    emitters = _find_emitters()
    subscribers = _find_subscribers()

    orphaned = []
    for event_name in SUPPORTED_EVENT_CATALOG:
        emitted = any(
            _has_supported_match(candidate, event_name)
            for candidate in emitters
        )
        subscribed = any(
            _has_supported_match(candidate, event_name)
            for candidate in subscribers
        )
        if emitted and not subscribed:
            orphaned.append(event_name)

    assert not orphaned, f"Supported emitted events without subscribers: {orphaned}"
