#!/usr/bin/env python3
"""Validate the checked-in training fleet docs without touching live nodes.

This is a local credibility/reliability preflight. It cross-checks the
orientation manifest, role manifest, canary deploy script, systemd unit files,
and runbook text so a fresh operator can trust the checked-in fleet story before
any SSH or deployment command is run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_SERVICE_ROOT.parent
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "data" / "training_fleet_manifest.json"
DEFAULT_RUNBOOK = REPO_ROOT / "docs" / "operations" / "TRAINING_FLEET_RUNBOOK.md"
NODE_ROLES = AI_SERVICE_ROOT / "config" / "node_roles.yaml"
DEPLOY_MINIMAL_LOOPS = AI_SERVICE_ROOT / "scripts" / "deploy_minimal_loops.sh"

TARGET_CONFIG_RE = re.compile(r"^(?P<board>.+)_(?P<players>[2-4])p$")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{_repo_relative(path)} must contain a YAML mapping")
    return loaded


def _parse_target_config(config: str) -> tuple[str, int]:
    match = TARGET_CONFIG_RE.fullmatch(config)
    if not match:
        raise ValueError(f"target config must look like '<board>_<players>p': {config}")
    return match.group("board"), int(match.group("players"))


def _parse_minimal_loop_entries(script_path: Path) -> dict[str, dict[str, str]]:
    text = script_path.read_text(encoding="utf-8")
    match = re.search(r"^NODES=\(\n(?P<body>.*?)\n\)", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("deploy_minimal_loops.sh must define a NODES=(...) manifest")

    entries: dict[str, dict[str, str]] = {}
    for raw in re.findall(r'^\s+"([^"]+)"', match.group("body"), re.MULTILINE):
        parts = raw.split("|", 3)
        if len(parts) != 4:
            raise ValueError(f"minimal-loop node entry must be ip|config|workdir|args: {raw}")
        ip, config, work_dir, args = parts
        entries[config] = {"known_host": ip, "work_dir": work_dir, "args": args}
    return entries


def _service_files_from_manifest(manifest: dict[str, Any]) -> list[Path]:
    service_paths: list[Path] = []
    for raw_path in manifest.get("source_files", []):
        path = REPO_ROOT / raw_path
        if path.suffix == ".service":
            service_paths.append(path)
    return service_paths


def validate(manifest_path: Path = DEFAULT_MANIFEST, runbook_path: Path = DEFAULT_RUNBOOK) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    try:
        manifest = _load_json(manifest_path)
        roles = _load_yaml(NODE_ROLES)
        deploy_entries = _parse_minimal_loop_entries(DEPLOY_MINIMAL_LOOPS)
    except Exception as exc:
        return {"ok": False, "checks": checks, "errors": [str(exc)], "warnings": warnings}

    runbook = runbook_path.read_text(encoding="utf-8") if runbook_path.exists() else ""
    role_nodes = roles.get("nodes", {})
    manifest_nodes = {node.get("name"): node for node in manifest.get("nodes", [])}

    for raw_path in manifest.get("source_files", []):
        checks += 1
        if not (REPO_ROOT / raw_path).exists():
            errors.append(f"manifest source file is missing: {raw_path}")

    runtime_inventory = manifest.get("runtime_inventory", {})
    for raw_path in runtime_inventory.get("tracked_templates", []):
        checks += 1
        if not (REPO_ROOT / raw_path).exists():
            errors.append(f"runtime inventory template is missing: {raw_path}")
    checks += 1
    if runtime_inventory.get("private_file") != "ai-service/config/distributed_hosts.yaml":
        errors.append("runtime inventory must document ai-service/config/distributed_hosts.yaml as private")

    checks += 1
    missing_manifest_nodes = sorted(set(role_nodes) - set(manifest_nodes))
    if missing_manifest_nodes:
        errors.append("node_roles.yaml nodes missing from manifest: " + ", ".join(missing_manifest_nodes))

    for node_name, role_spec in role_nodes.items():
        checks += 1
        manifest_node = manifest_nodes.get(node_name)
        if not manifest_node:
            continue

        role = role_spec.get("role")
        if manifest_node.get("role") != role:
            errors.append(f"{node_name}: role mismatch manifest={manifest_node.get('role')} node_roles={role}")

        if role == "trainer":
            config = role_spec.get("target_config")
            board_type, players = _parse_target_config(str(config))
            if manifest_node.get("target_config") != config:
                errors.append(f"{node_name}: target_config mismatch for trainer")
            if manifest_node.get("board_type") != board_type:
                errors.append(f"{node_name}: board_type should be {board_type}")
            if manifest_node.get("num_players") != players:
                errors.append(f"{node_name}: num_players should be {players}")
        elif role == "selfplay-worker":
            for key in ("target_config", "feeds_trainer", "selfplay_profile"):
                if manifest_node.get(key) != role_spec.get(key):
                    errors.append(f"{node_name}: {key} mismatch for selfplay worker")
        elif role == "evaluator":
            expected = sorted(role_spec.get("assigned_configs", []))
            actual = sorted(manifest_node.get("assigned_configs", []))
            if actual != expected:
                errors.append(f"{node_name}: assigned_configs mismatch for evaluator")

    canary_configs = {
        node["target_config"]: node
        for node in manifest.get("nodes", [])
        if node.get("known_host_source") == "deploy_minimal_loops.sh"
    }
    canary_configs.update({node["target_config"]: node for node in manifest.get("script_only_canaries", [])})
    for config, manifest_node in sorted(canary_configs.items()):
        checks += 1
        deploy_entry = deploy_entries.get(config)
        if not deploy_entry:
            errors.append(f"{config}: missing from deploy_minimal_loops.sh NODES")
            continue
        if deploy_entry["known_host"] != manifest_node.get("known_host"):
            errors.append(f"{config}: known_host mismatch with deploy_minimal_loops.sh")
        if deploy_entry["work_dir"] != manifest_node.get("work_dir"):
            errors.append(f"{config}: work_dir mismatch with deploy_minimal_loops.sh")

    for service_file in _service_files_from_manifest(manifest):
        checks += 1
        text = service_file.read_text(encoding="utf-8")
        if "Restart=always" not in text:
            errors.append(f"{_repo_relative(service_file)} must document Restart=always")
        if "WantedBy=multi-user.target" not in text:
            errors.append(f"{_repo_relative(service_file)} must be installable at multi-user.target")

    required_runbook_phrases = (
        "docs/data/training_fleet_manifest.json",
        "ai-service/config/distributed_hosts.yaml",
        "deploy_minimal_loops.sh",
        "deploy_training_service.sh",
        "not boot-persistent",
        "Restart=always",
        "validate_training_fleet_docs.py",
    )
    for phrase in required_runbook_phrases:
        checks += 1
        if phrase not in runbook:
            errors.append(f"runbook missing required phrase: {phrase}")

    role_configs = {spec.get("target_config") for spec in role_nodes.values() if spec.get("target_config")}
    script_only_configs = {node.get("target_config") for node in manifest.get("script_only_canaries", [])}
    extra_deploy_configs = set(deploy_entries) - canary_configs.keys()
    if extra_deploy_configs:
        warnings.append(
            "deploy_minimal_loops.sh has configs not represented in manifest: "
            + ", ".join(sorted(extra_deploy_configs))
        )
    if script_only_configs & role_configs:
        warnings.append(
            "script_only_canaries overlap role-assigned configs: "
            + ", ".join(sorted(script_only_configs & role_configs))
        )

    return {"ok": not errors, "checks": checks, "errors": errors, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runbook", type=Path, default=DEFAULT_RUNBOOK)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable validation output")
    args = parser.parse_args(argv)

    result = validate(manifest_path=args.manifest, runbook_path=args.runbook)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "ok" if result["ok"] else "failed"
        print(f"training fleet docs validation: {status} ({result['checks']} checks)")
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
