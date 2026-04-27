"""Sync policy helpers for coordinator disk safety.

The policy defaults to push-only for local rehydration. OWC, S3, and P2P pulls
that write back to the coordinator's internal disk must be explicitly allowed
by family/pattern or by a downstream caller passing an explicit consumer signal.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INTERNAL_WRITE_MIN_FREE_GB = 10.0
DEFAULT_POLICY_REL_PATH = Path("data") / "sync_policy.yaml"
EXPLICIT_PULL_SIGNAL_ENV = "RINGRIFT_SYNC_PULL_CONSUMER_SIGNAL"

GAUNTLET_PATTERNS = (
    "gauntlet_*.db",
    "baseline_calibration_*.db",
    "tournament_*.db",
)


@dataclass(frozen=True)
class PullPolicy:
    """Policy for rehydrating remote data onto internal coordinator storage."""

    default_pull_allowed: bool = False
    require_consumer_signal: bool = True
    pull_allowlist: tuple[str, ...] = ()
    gauntlet_pull_allowed: bool = False


@dataclass(frozen=True)
class SyncPolicy:
    """Disk and direction safety policy for sync daemons."""

    internal_write_min_free_gb: float = DEFAULT_INTERNAL_WRITE_MIN_FREE_GB
    pull: PullPolicy = field(default_factory=PullPolicy)


def _ai_service_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = Path(os.path.join(str(parent), "app", "coordination"))
        if candidate.exists():
            return parent
    return Path.cwd()


def default_policy_path() -> Path:
    override = os.environ.get("RINGRIFT_SYNC_POLICY_PATH")
    if override:
        return Path(override).expanduser()
    return Path(os.path.join(str(_ai_service_root()), str(DEFAULT_POLICY_REL_PATH)))


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Load the small policy YAML subset used by data/sync_policy.yaml.

    PyYAML is used when available. This fallback handles nested dictionaries,
    booleans, numbers, and string lists so tests and production do not gain a
    hard dependency on PyYAML for one safety file.
    """
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        pass

    # JSON is valid enough for emergency/operator overrides.
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass

    data: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, data)]
    current_list_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- ") and current_list_key:
            parent.setdefault(current_list_key, []).append(stripped[2:].strip().strip("'\""))
            continue

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_list_key = None

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            continue
        if value == "[]":
            parent[key] = []
            current_list_key = key
            continue
        if value.lower() in {"true", "false"}:
            parent[key] = value.lower() == "true"
            continue
        try:
            parent[key] = float(value) if "." in value else int(value)
            continue
        except ValueError:
            parent[key] = value.strip("'\"")

    return data


def load_sync_policy(path: Path | None = None) -> SyncPolicy:
    policy_path = path or default_policy_path()
    if not policy_path.exists():
        return SyncPolicy()

    try:
        raw = _simple_yaml_load(policy_path.read_text())
    except OSError as exc:
        logger.warning("Failed to read sync policy %s: %s", policy_path, exc)
        return SyncPolicy()

    pull_raw = raw.get("pull", {}) if isinstance(raw.get("pull"), dict) else {}
    allowlist = pull_raw.get("allowlist", ())
    if isinstance(allowlist, str):
        allowlist = (allowlist,)
    elif not isinstance(allowlist, (list, tuple)):
        allowlist = ()

    return SyncPolicy(
        internal_write_min_free_gb=float(
            raw.get("internal_write_min_free_gb", DEFAULT_INTERNAL_WRITE_MIN_FREE_GB)
        ),
        pull=PullPolicy(
            default_pull_allowed=bool(pull_raw.get("default_allowed", False)),
            require_consumer_signal=bool(pull_raw.get("require_consumer_signal", True)),
            pull_allowlist=tuple(str(item) for item in allowlist),
            gauntlet_pull_allowed=bool(pull_raw.get("gauntlet_allowed", False)),
        ),
    )


def is_evaluation_db_name(path_or_name: str | Path) -> bool:
    name = Path(path_or_name).name
    return any(fnmatch.fnmatch(name, pattern) for pattern in GAUNTLET_PATTERNS)


def is_pull_to_internal_allowed(
    path_or_name: str | Path,
    *,
    family: str = "games",
    consumer_signal: str | None = None,
    policy: SyncPolicy | None = None,
) -> bool:
    """Return whether remote data may be rehydrated to internal storage."""
    resolved = policy or load_sync_policy()
    name = Path(path_or_name).name
    signal = consumer_signal or os.environ.get(EXPLICIT_PULL_SIGNAL_ENV)

    if resolved.pull.require_consumer_signal and not signal:
        return False

    if is_evaluation_db_name(name):
        return resolved.pull.gauntlet_pull_allowed and any(
            fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(family, pattern)
            for pattern in resolved.pull.pull_allowlist
        )

    if any(
        fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(family, pattern)
        for pattern in resolved.pull.pull_allowlist
    ):
        return True

    return resolved.pull.default_pull_allowed


def internal_write_safety_status(
    path: str | Path,
    *,
    min_free_gb: float | None = None,
    policy: SyncPolicy | None = None,
) -> tuple[bool, float, float]:
    """Check whether a write to an internal path has enough free space."""
    resolved = policy or load_sync_policy()
    threshold = (
        float(min_free_gb)
        if min_free_gb is not None
        else resolved.internal_write_min_free_gb
    )
    check_path = Path(path)
    existing = check_path if check_path.exists() else check_path.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(str(existing))
    free_gb = usage.free / (1024**3)
    return free_gb >= threshold, free_gb, threshold


def should_backoff_internal_write(
    path: str | Path,
    *,
    min_free_gb: float | None = None,
    policy: SyncPolicy | None = None,
) -> bool:
    ok, free_gb, threshold = internal_write_safety_status(
        path,
        min_free_gb=min_free_gb,
        policy=policy,
    )
    if not ok:
        logger.warning(
            "sync_backoff_active path=%s free_gb=%.2f threshold_gb=%.2f",
            path,
            free_gb,
            threshold,
        )
    return not ok
