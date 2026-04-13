"""Node workload policy helpers for P2P/training role orchestration.

This module provides a small declarative overlay on top of
``config/distributed_hosts.yaml`` so cluster behavior can be steered without
editing the host inventory directly.

The immediate use case is separating trainer, selfplay-worker, evaluator, and
sync-only nodes while preserving backward compatibility with the legacy role
strings that still exist in distributed host configs.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.cluster_config import load_cluster_config

logger = logging.getLogger(__name__)

DEFAULT_NODE_ROLE_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "node_roles.yaml"

POLICY_ONLY_SELFPLAY_PROFILES = frozenset({"policy-gumbel", "policy-training"})
DISABLED_SELFPLAY_PROFILES = frozenset({"disabled", "none"})
SELFPLAY_WORK_TYPES = frozenset({"selfplay"})
GPU_WORKLOAD_CAPABILITIES = ("training", "cmaes", "gauntlet", "tournament")
EVALUATION_CAPABILITIES = ("gauntlet", "tournament")

_ROLE_MANIFEST_CACHE: dict[str, Any] | None = None
_ROLE_MANIFEST_CACHE_MTIME: float | None = None
_ROLE_MANIFEST_CACHE_PATH: Path | None = None


@dataclass(frozen=True)
class NodeWorkloadPolicy:
    """Resolved workload policy for one node."""

    node_id: str
    role: str
    resolved: bool
    matched_name: str | None = None
    legacy_role: str | None = None
    has_gpu: bool = False
    selfplay_enabled: bool = True
    training_enabled: bool = False
    evaluation_enabled: bool = False
    p2p_enabled: bool = True
    selfplay_profile: str = "balanced"
    job_preference: str = "both"
    allowed_config_keys: tuple[str, ...] = ()
    feeds_trainer: str | None = None


def clear_node_role_manifest_cache() -> None:
    """Reset the cached node role manifest for tests."""

    global _ROLE_MANIFEST_CACHE, _ROLE_MANIFEST_CACHE_MTIME, _ROLE_MANIFEST_CACHE_PATH
    _ROLE_MANIFEST_CACHE = None
    _ROLE_MANIFEST_CACHE_MTIME = None
    _ROLE_MANIFEST_CACHE_PATH = None


def get_local_node_workload_policy(
    *,
    node_id: str | None = None,
    hostname: str | None = None,
    cluster_config_path: str | Path | None = None,
    role_config_path: str | Path | None = None,
) -> NodeWorkloadPolicy:
    """Resolve workload policy for the current node."""

    return get_node_workload_policy(
        node_id=node_id,
        hostname=hostname or socket.gethostname(),
        cluster_config_path=cluster_config_path,
        role_config_path=role_config_path,
    )


def get_node_workload_policy(
    node_id: str | None = None,
    *,
    hostname: str | None = None,
    cluster_config_path: str | Path | None = None,
    role_config_path: str | Path | None = None,
) -> NodeWorkloadPolicy:
    """Resolve workload policy for a node from cluster config and role overlay."""

    normalized_node_id = (node_id or hostname or "unknown").strip() or "unknown"

    try:
        cluster_config = load_cluster_config(_resolve_cluster_config_path(cluster_config_path))
        hosts = cluster_config.hosts_raw
    except Exception as exc:
        logger.debug("Failed to load cluster config for node role resolution: %s", exc)
        hosts = {}

    manifest = _load_node_role_manifest(role_config_path=role_config_path)
    host_name, host_cfg = _match_mapping_entry(hosts, node_id=node_id, hostname=hostname)
    override_name, override_cfg = _match_mapping_entry(
        manifest.get("nodes", {}) if isinstance(manifest, dict) else {},
        node_id=node_id or host_name,
        hostname=hostname,
    )

    resolved = bool(host_cfg or override_cfg)
    has_gpu = _host_has_gpu(host_cfg)
    legacy_role = _normalize_role(host_cfg.get("role")) if host_cfg else None

    base_role = _derive_effective_role(
        declared_role=legacy_role,
        host_cfg=host_cfg,
        has_gpu=has_gpu,
    )
    policy_values = _defaults_for_role(base_role, has_gpu=has_gpu)
    policy_values = _apply_explicit_overrides(policy_values, host_cfg or {})

    if override_cfg:
        override_role = _normalize_role(override_cfg.get("role"))
        if override_role:
            policy_values = _defaults_for_role(override_role, has_gpu=has_gpu)
        policy_values = _apply_explicit_overrides(policy_values, override_cfg)
        if override_role == "selfplay-worker":
            # Manifest-managed selfplay workers use the dedicated
            # ringrift-selfplay-worker.service path. Keep P2P running for sync
            # and health, but do not advertise fallback selfplay capacity.
            policy_values["selfplay_enabled"] = False
            policy_values["selfplay_profile"] = "disabled"

    role = str(policy_values["role"])
    selfplay_enabled = bool(policy_values["selfplay_enabled"])
    training_enabled = bool(policy_values["training_enabled"])
    evaluation_enabled = bool(policy_values["evaluation_enabled"])
    p2p_enabled = bool(policy_values["p2p_enabled"])
    selfplay_profile = str(policy_values["selfplay_profile"]).strip().lower() or "balanced"
    allowed_config_keys = tuple(policy_values["allowed_config_keys"])
    feeds_trainer = policy_values["feeds_trainer"]

    if not selfplay_enabled:
        selfplay_profile = "disabled"

    job_preference = _derive_job_preference(
        role=role,
        has_gpu=has_gpu,
        selfplay_enabled=selfplay_enabled,
        training_enabled=training_enabled,
        evaluation_enabled=evaluation_enabled,
        p2p_enabled=p2p_enabled,
    )

    return NodeWorkloadPolicy(
        node_id=normalized_node_id,
        role=role,
        resolved=resolved,
        matched_name=override_name or host_name,
        legacy_role=legacy_role,
        has_gpu=has_gpu,
        selfplay_enabled=selfplay_enabled,
        training_enabled=training_enabled,
        evaluation_enabled=evaluation_enabled,
        p2p_enabled=p2p_enabled,
        selfplay_profile=selfplay_profile,
        job_preference=job_preference,
        allowed_config_keys=allowed_config_keys,
        feeds_trainer=feeds_trainer,
    )


def policy_allows_config_key(
    policy: NodeWorkloadPolicy,
    config_key: str | None = None,
) -> bool:
    """Return whether a config_key is allowed under a resolved workload policy.

    Unresolved policies fail open so legacy nodes without a role overlay keep
    their historical behavior.
    """

    normalized_config_key = (config_key or "").strip()
    if not normalized_config_key:
        return True
    if not policy.resolved:
        return True
    if not policy.allowed_config_keys:
        return True
    return normalized_config_key in policy.allowed_config_keys


def policy_allows_work_type(
    policy: NodeWorkloadPolicy,
    work_type: str | Any,
    *,
    config_key: str | None = None,
) -> bool:
    """Return whether a workload policy allows a work type for a config.

    This is the manifest-backed gate for manifest-managed fleet behavior. The
    overlay only constrains nodes that resolve through the workload manifest;
    unresolved nodes keep the historical fail-open behavior.
    """

    normalized_work_type = str(getattr(work_type, "value", work_type) or "").strip().lower()
    if not normalized_work_type:
        return True
    if not policy.resolved:
        return True

    if normalized_work_type in SELFPLAY_WORK_TYPES:
        return policy.selfplay_enabled and policy_allows_config_key(
            policy, config_key=config_key
        )

    return True


def node_allows_work_type(
    node_id: str | None = None,
    work_type: str | Any = "",
    *,
    hostname: str | None = None,
    config_key: str | None = None,
    cluster_config_path: str | Path | None = None,
    role_config_path: str | Path | None = None,
) -> bool:
    """Resolve a node policy and check whether a work type is allowed.

    This helper is intentionally lightweight so P2P sender- and receiver-side
    gates can share the same manifest-backed decision.
    """

    policy = get_node_workload_policy(
        node_id=node_id,
        hostname=hostname,
        cluster_config_path=cluster_config_path,
        role_config_path=role_config_path,
    )
    return policy_allows_work_type(policy, work_type, config_key=config_key)


def capabilities_for_policy(
    policy: NodeWorkloadPolicy,
    *,
    has_gpu: bool | None = None,
    memory_gb: int = 0,
) -> list[str]:
    """Translate a workload policy into advertised P2P capabilities.

    Manifest-managed nodes are constrained by the resolved policy. Unresolved
    nodes keep the historical fail-open capability set for backward
    compatibility.
    """

    effective_has_gpu = policy.has_gpu if has_gpu is None else bool(has_gpu)

    if not policy.p2p_enabled:
        return []

    if not policy.resolved:
        capabilities = ["selfplay"]
        if effective_has_gpu:
            capabilities.extend(GPU_WORKLOAD_CAPABILITIES)
        if memory_gb >= 64:
            capabilities.append("large_boards")
        return capabilities

    capabilities: list[str] = []
    if policy.selfplay_enabled:
        capabilities.append("selfplay")

    if policy.training_enabled:
        capabilities.append("training")
        if effective_has_gpu:
            capabilities.extend(cap for cap in GPU_WORKLOAD_CAPABILITIES if cap != "training")
    elif policy.evaluation_enabled:
        capabilities.extend(EVALUATION_CAPABILITIES)

    if memory_gb >= 64 and capabilities:
        capabilities.append("large_boards")

    deduped: list[str] = []
    seen: set[str] = set()
    for capability in capabilities:
        if capability and capability not in seen:
            deduped.append(capability)
            seen.add(capability)
    return deduped


def _resolve_cluster_config_path(
    cluster_config_path: str | Path | None,
) -> str | Path | None:
    if cluster_config_path is not None:
        return cluster_config_path
    override = os.environ.get("RINGRIFT_CLUSTER_CONFIG_PATH", "").strip()
    if override:
        return override
    return None


def _resolve_role_config_path(
    role_config_path: str | Path | None,
) -> Path:
    if role_config_path is not None:
        return Path(role_config_path)
    override = os.environ.get("RINGRIFT_NODE_ROLE_CONFIG", "").strip()
    if override:
        return Path(override)
    return DEFAULT_NODE_ROLE_CONFIG_PATH


def _load_node_role_manifest(
    *,
    role_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load node role manifest from disk with a small mtime cache."""

    global _ROLE_MANIFEST_CACHE, _ROLE_MANIFEST_CACHE_MTIME, _ROLE_MANIFEST_CACHE_PATH

    path = _resolve_role_config_path(role_config_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    if (
        _ROLE_MANIFEST_CACHE is not None
        and _ROLE_MANIFEST_CACHE_PATH == path
        and _ROLE_MANIFEST_CACHE_MTIME == mtime
    ):
        return _ROLE_MANIFEST_CACHE

    try:
        import yaml

        with open(path) as handle:
            manifest = yaml.safe_load(handle) or {}
    except Exception as exc:
        logger.warning("Failed to load node role manifest %s: %s", path, exc)
        return {}

    if not isinstance(manifest, dict):
        logger.warning("Node role manifest %s is not a mapping; ignoring it", path)
        return {}

    _ROLE_MANIFEST_CACHE = manifest
    _ROLE_MANIFEST_CACHE_MTIME = mtime
    _ROLE_MANIFEST_CACHE_PATH = path
    return manifest


def _match_mapping_entry(
    entries: dict[str, Any],
    *,
    node_id: str | None = None,
    hostname: str | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    if not entries:
        return None, None

    raw_candidates = [value for value in (node_id, hostname) if value]
    normalized_candidates = [_normalize_node_name(value) for value in raw_candidates if value]

    for raw in raw_candidates:
        if raw in entries and isinstance(entries[raw], dict):
            return raw, entries[raw]

    for entry_name, entry_cfg in entries.items():
        if not isinstance(entry_cfg, dict):
            continue
        normalized_entry = _normalize_node_name(entry_name)
        for normalized_candidate in normalized_candidates:
            if not normalized_candidate:
                continue
            if (
                normalized_candidate == normalized_entry
                or normalized_candidate in normalized_entry
                or normalized_entry in normalized_candidate
            ):
                return entry_name, entry_cfg
            if "macbook" in normalized_candidate and normalized_entry in ("macstudio", "localmac"):
                return entry_name, entry_cfg

    return None, None


def _normalize_node_name(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "").replace("_", "")


def _normalize_role(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    return role or None


def _host_has_gpu(host_cfg: dict[str, Any] | None) -> bool:
    if not host_cfg:
        return False
    gpu_name = str(host_cfg.get("gpu", "") or "").strip().lower()
    if gpu_name and gpu_name != "none":
        return True
    try:
        return int(host_cfg.get("gpu_vram_gb", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _derive_effective_role(
    *,
    declared_role: str | None,
    host_cfg: dict[str, Any] | None,
    has_gpu: bool,
) -> str:
    if declared_role == "trainer":
        return "trainer"
    if declared_role == "selfplay-worker":
        return "selfplay-worker"
    if declared_role == "evaluator":
        return "evaluator"
    if declared_role == "sync-only":
        return "sync-only"
    if declared_role == "hybrid-trainer":
        return "hybrid-trainer"
    if declared_role in ("coordinator", "proxy"):
        return "sync-only"
    if declared_role in ("gpu_selfplay", "cpu_selfplay"):
        return "selfplay-worker"
    if declared_role == "gpu_training_primary":
        return "trainer"
    if declared_role in ("gpu_training_selfplay", "gpu_both", "training_backup"):
        return "hybrid-trainer"

    selfplay_enabled = _coerce_bool(
        (host_cfg or {}).get("selfplay_enabled"),
        default=has_gpu,
    )
    training_enabled = _coerce_bool(
        (host_cfg or {}).get("training_enabled"),
        default=False,
    )
    evaluation_enabled = _coerce_bool(
        (host_cfg or {}).get("evaluation_enabled"),
        default=False,
    )

    if evaluation_enabled and not selfplay_enabled and not training_enabled:
        return "evaluator"
    if training_enabled and selfplay_enabled:
        return "hybrid-trainer"
    if training_enabled:
        return "trainer"
    if selfplay_enabled:
        return "selfplay-worker"
    return "sync-only"


def _defaults_for_role(role: str, *, has_gpu: bool) -> dict[str, Any]:
    normalized_role = _normalize_role(role) or "sync-only"

    if normalized_role == "trainer":
        return {
            "role": "trainer",
            "selfplay_enabled": False,
            "training_enabled": True,
            "evaluation_enabled": False,
            "p2p_enabled": True,
            "selfplay_profile": "disabled",
            "allowed_config_keys": (),
            "feeds_trainer": None,
        }

    if normalized_role == "selfplay-worker":
        return {
            "role": "selfplay-worker",
            "selfplay_enabled": True,
            "training_enabled": False,
            "evaluation_enabled": False,
            "p2p_enabled": True,
            "selfplay_profile": "policy-gumbel" if has_gpu else "balanced",
            "allowed_config_keys": (),
            "feeds_trainer": None,
        }

    if normalized_role == "evaluator":
        return {
            "role": "evaluator",
            "selfplay_enabled": False,
            "training_enabled": False,
            "evaluation_enabled": True,
            "p2p_enabled": True,
            "selfplay_profile": "disabled",
            "allowed_config_keys": (),
            "feeds_trainer": None,
        }

    if normalized_role == "hybrid-trainer":
        return {
            "role": "hybrid-trainer",
            "selfplay_enabled": True,
            "training_enabled": True,
            "evaluation_enabled": False,
            "p2p_enabled": True,
            "selfplay_profile": "balanced",
            "allowed_config_keys": (),
            "feeds_trainer": None,
        }

    return {
        "role": "sync-only",
        "selfplay_enabled": False,
        "training_enabled": False,
        "evaluation_enabled": False,
        "p2p_enabled": True,
        "selfplay_profile": "disabled",
        "allowed_config_keys": (),
        "feeds_trainer": None,
    }


def _apply_explicit_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    result = dict(base)

    for key in ("selfplay_enabled", "training_enabled", "evaluation_enabled", "p2p_enabled"):
        if key in overrides:
            result[key] = _coerce_bool(overrides.get(key), default=bool(result[key]))

    if "selfplay_profile" in overrides and overrides.get("selfplay_profile") is not None:
        result["selfplay_profile"] = str(overrides["selfplay_profile"]).strip().lower()

    allowed_configs = overrides.get("allowed_configs") or overrides.get("assigned_configs") or []
    if "target_config" in overrides and not allowed_configs:
        allowed_configs = [overrides["target_config"]]
    if allowed_configs:
        result["allowed_config_keys"] = tuple(str(value) for value in allowed_configs if value)

    if "feeds_trainer" in overrides:
        feeds_trainer = str(overrides["feeds_trainer"]).strip()
        result["feeds_trainer"] = feeds_trainer or None

    return result


def _derive_job_preference(
    *,
    role: str,
    has_gpu: bool,
    selfplay_enabled: bool,
    training_enabled: bool,
    evaluation_enabled: bool,
    p2p_enabled: bool,
) -> str:
    if not p2p_enabled and not selfplay_enabled and not training_enabled:
        return "disabled"
    if role == "sync-only" and not selfplay_enabled and not training_enabled:
        return "disabled"
    if evaluation_enabled and not selfplay_enabled and not training_enabled:
        return "evaluation_only"
    if training_enabled and not selfplay_enabled:
        return "training_only"
    if selfplay_enabled and not training_enabled:
        return "gpu_only" if has_gpu else "cpu_only"
    if selfplay_enabled and training_enabled:
        return "both"
    return "disabled"


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
