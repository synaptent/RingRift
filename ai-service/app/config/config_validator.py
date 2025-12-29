#!/usr/bin/env python3
"""Configuration validation for RingRift AI service.

Validates all configuration files at startup to catch errors early before
any expensive operations begin. This prevents runtime failures due to
misconfiguration.

Usage:
    from app.config.config_validator import (
        ConfigValidator,
        validate_all_configs,
        ValidationResult,
    )

    # Validate everything
    result = validate_all_configs()
    if not result.valid:
        for error in result.errors:
            print(f"ERROR: {error}")
        sys.exit(1)

    # Validate specific config
    validator = ConfigValidator()
    result = validator.validate_unified_loop_config()
    if not result.valid:
        print(result.errors)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.utils.paths import AI_SERVICE_ROOT


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    config_name: str
    config_path: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            config_name=f"{self.config_name}, {other.config_name}",
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


class ConfigValidator:
    """Validates RingRift configuration files."""

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or AI_SERVICE_ROOT

    def validate_all(self) -> ValidationResult:
        """Validate all configuration files."""
        results = []

        # Core configs
        results.append(self.validate_unified_loop_config())
        results.append(self.validate_distributed_hosts())
        legacy_hosts = self.validate_remote_hosts()
        if legacy_hosts.config_path:
            results.append(legacy_hosts)
        results.append(self.validate_hyperparameters())
        results.append(self.validate_resource_limits())

        # Optional configs
        elo_alerts = self.validate_elo_alerts()
        if elo_alerts.config_path:  # Only include if file exists
            results.append(elo_alerts)

        # Combine all results
        if not results:
            return ValidationResult(valid=True, config_name="all")

        combined = results[0]
        for r in results[1:]:
            combined = combined.merge(r)

        return combined

    def validate_unified_loop_config(self) -> ValidationResult:
        """Validate unified_loop.yaml configuration."""
        config_path = self.base_path / "config" / "unified_loop.yaml"
        errors = []
        warnings = []

        if not config_path.exists():
            return ValidationResult(
                valid=False,
                config_name="unified_loop.yaml",
                config_path=str(config_path),
                errors=[f"Config file not found: {config_path}"],
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if not config:
                errors.append("Config file is empty")
                return ValidationResult(
                    valid=False,
                    config_name="unified_loop.yaml",
                    config_path=str(config_path),
                    errors=errors,
                )

            # Validate required sections
            required_sections = ["data_ingestion", "evaluation", "training", "promotion"]
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing required section: {section}")

            # Validate data_ingestion
            if "data_ingestion" in config:
                di = config["data_ingestion"]
                if di.get("poll_interval_seconds", 0) < 10:
                    warnings.append("data_ingestion.poll_interval_seconds < 10 may cause high load")
                if di.get("poll_interval_seconds", 0) > 3600:
                    warnings.append("data_ingestion.poll_interval_seconds > 3600 may cause stale data")

            # Validate evaluation
            if "evaluation" in config:
                ev = config["evaluation"]
                if ev.get("shadow_interval_seconds", 0) < 60:
                    warnings.append("evaluation.shadow_interval_seconds < 60 may cause excessive evaluations")
                games_per_match = ev.get("games_per_shadow_match", 4)
                if games_per_match < 2:
                    errors.append("evaluation.games_per_shadow_match must be >= 2")
                if games_per_match > 50:
                    warnings.append("evaluation.games_per_shadow_match > 50 may cause slow evaluations")

            # Validate training
            if "training" in config:
                tr = config["training"]
                min_games = tr.get("min_games_for_training", 0)
                if min_games < 100:
                    warnings.append("training.min_games_for_training < 100 may lead to poor models")
                if min_games > 100000:
                    warnings.append("training.min_games_for_training > 100000 may delay training too much")

            # Validate promotion
            if "promotion" in config:
                pr = config["promotion"]
                elo_threshold = pr.get("elo_threshold", 0)
                if elo_threshold < 10:
                    warnings.append("promotion.elo_threshold < 10 may promote weak models")
                if elo_threshold > 200:
                    warnings.append("promotion.elo_threshold > 200 may be too conservative")

        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        except (KeyError, TypeError, AttributeError, OSError) as e:
            # Config access/file errors: missing keys, wrong types, file I/O
            errors.append(f"Config validation error: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="unified_loop.yaml",
            config_path=str(config_path),
            errors=errors,
            warnings=warnings,
        )

    def validate_distributed_hosts(self) -> ValidationResult:
        """Validate distributed_hosts.yaml configuration."""
        config_path = self.base_path / "config" / "distributed_hosts.yaml"
        errors = []
        warnings = []

        if not config_path.exists():
            return ValidationResult(
                valid=False,
                config_name="distributed_hosts.yaml",
                config_path=str(config_path),
                errors=[f"Config file not found: {config_path}"],
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if not config:
                warnings.append("Config file is empty")
                return ValidationResult(
                    valid=False,
                    config_name="distributed_hosts.yaml",
                    config_path=str(config_path),
                    warnings=warnings,
                )

            hosts = config.get("hosts", {})
            if not hosts:
                errors.append("No hosts defined under 'hosts'")

            for name, info in hosts.items():
                ssh_host = info.get("tailscale_ip") or info.get("ssh_host", "")
                if not ssh_host:
                    errors.append(f"Host '{name}' missing ssh_host/tailscale_ip")
                if not info.get("ssh_user"):
                    warnings.append(f"Host '{name}' missing ssh_user (defaulting to 'ubuntu')")

                if ssh_host and not self._is_valid_host(ssh_host):
                    warnings.append(f"Host '{name}' has invalid ssh_host: {ssh_host}")

                memory_gb = info.get("memory_gb", 0)
                if memory_gb and memory_gb < 16:
                    warnings.append(f"Host '{name}' has low memory: {memory_gb}GB")

        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        except (KeyError, TypeError, AttributeError, OSError) as e:
            # Config access/file errors: missing keys, wrong types, file I/O
            errors.append(f"Config validation error: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="distributed_hosts.yaml",
            config_path=str(config_path),
            errors=errors,
            warnings=warnings,
        )

    def validate_remote_hosts(self) -> ValidationResult:
        """Validate remote_hosts.yaml configuration (deprecated)."""
        config_path = self.base_path / "config" / "remote_hosts.yaml"
        if not config_path.exists():
            return ValidationResult(
                valid=True,
                config_name="remote_hosts.yaml",
                config_path=None,
            )
        return ValidationResult(
            valid=True,
            config_name="remote_hosts.yaml",
            config_path=str(config_path),
            warnings=[
                "remote_hosts.yaml is deprecated; use distributed_hosts.yaml instead",
            ],
        )

    def validate_hyperparameters(self) -> ValidationResult:
        """Validate hyperparameters.json configuration."""
        config_path = self.base_path / "config" / "hyperparameters.json"
        errors = []
        warnings = []

        if not config_path.exists():
            return ValidationResult(
                valid=False,
                config_name="hyperparameters.json",
                config_path=str(config_path),
                errors=[f"Config file not found: {config_path}"],
            )

        try:
            with open(config_path) as f:
                config = json.load(f)

            # Validate configs section
            configs = config.get("configs", {})
            if not configs:
                warnings.append("No board configurations defined")

            for config_key, params in configs.items():
                # Validate config key format
                if not re.match(r"^[a-z]+\d*_\d+p$", config_key):
                    warnings.append(f"Config key '{config_key}' doesn't match expected format (e.g., 'square8_2p')")

                # Validate learning rate
                lr = params.get("learning_rate", 0)
                if lr > 0:
                    if lr < 1e-6:
                        warnings.append(f"{config_key}: learning_rate {lr} is very low")
                    if lr > 1e-2:
                        warnings.append(f"{config_key}: learning_rate {lr} is very high")

                # Validate batch size
                batch_size = params.get("batch_size", 0)
                if batch_size > 0:
                    if batch_size < 32:
                        warnings.append(f"{config_key}: batch_size {batch_size} is small")
                    if batch_size > 4096:
                        warnings.append(f"{config_key}: batch_size {batch_size} may cause OOM")

                # Validate exploration parameters
                cpuct = params.get("c_puct", 0)
                if cpuct > 0:
                    if cpuct < 0.5:
                        warnings.append(f"{config_key}: c_puct {cpuct} may underexplore")
                    if cpuct > 5.0:
                        warnings.append(f"{config_key}: c_puct {cpuct} may overexplore")

        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="hyperparameters.json",
            config_path=str(config_path),
            errors=errors,
            warnings=warnings,
        )

    def validate_elo_alerts(self) -> ValidationResult:
        """Validate elo_alerts.yml configuration."""
        config_path = self.base_path / "config" / "elo_alerts.yml"
        errors = []
        warnings = []

        if not config_path.exists():
            return ValidationResult(
                valid=True,
                config_name="elo_alerts.yml",
                config_path=None,  # Signal optional file not present
                warnings=[],
            )

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            if config:
                # Validate alert thresholds
                thresholds = config.get("thresholds", {})
                for name, value in thresholds.items():
                    if isinstance(value, (int, float)) and value < 0:
                        errors.append(f"Threshold '{name}' cannot be negative: {value}")

        except yaml.YAMLError as e:
            errors.append(f"YAML parse error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="elo_alerts.yml",
            config_path=str(config_path),
            errors=errors,
            warnings=warnings,
        )

    def validate_database_paths(self) -> ValidationResult:
        """Validate that required database directories exist and are writable."""
        errors = []
        warnings = []

        db_paths = [
            self.base_path / "data" / "games",
            self.base_path / "data" / "coordination",
            self.base_path / "logs" / "unified_loop",
        ]

        for db_path in db_paths:
            if not db_path.exists():
                try:
                    db_path.mkdir(parents=True, exist_ok=True)
                    warnings.append(f"Created missing directory: {db_path}")
                except PermissionError:
                    errors.append(f"Cannot create directory: {db_path}")
            elif not os.access(db_path, os.W_OK):
                errors.append(f"Directory not writable: {db_path}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="database_paths",
            errors=errors,
            warnings=warnings,
        )

    def validate_resource_limits(self) -> ValidationResult:
        """Validate resource limit configurations across modules.

        Ensures consistent 80% utilization limits are enforced.
        """
        errors = []
        warnings = []

        # Check unified_loop.yaml safety section
        config_path = self.base_path / "config" / "unified_loop.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                # Check safety thresholds
                safety = config.get("safety", {})

                # Memory limit check (should not exceed 80% of available)
                min_mem = safety.get("min_memory_gb", 0)
                if min_mem > 0 and min_mem < 16:
                    warnings.append(f"safety.min_memory_gb={min_mem}GB is low - consider 32GB+ for production")

                # Overfitting threshold (should be reasonable)
                overfit = safety.get("overfit_threshold", 0.15)
                if overfit > 0.3:
                    warnings.append(f"safety.overfit_threshold={overfit} is high - may allow overfitted models")

                # Check safeguards section for process limits
                safeguards = config.get("safeguards", {})

                max_python = safeguards.get("max_python_processes_per_host", 0)
                if max_python > 100:
                    warnings.append(f"safeguards.max_python_processes_per_host={max_python} is high - may cause resource exhaustion")

                max_selfplay = safeguards.get("max_selfplay_processes", 0)
                if max_selfplay > 100:
                    warnings.append(f"safeguards.max_selfplay_processes={max_selfplay} is high")

                # Check resource limit consistency with resource_guard
                # Keep thresholds aligned with the canonical resource_guard limits.
                try:
                    from app.utils.resource_guard import LIMITS
                    if LIMITS:
                        # Verify consistency
                        if LIMITS.CPU_MAX_PERCENT != 80.0:
                            warnings.append(f"resource_guard.CPU_MAX_PERCENT={LIMITS.CPU_MAX_PERCENT} != 80%")
                        if LIMITS.MEMORY_MAX_PERCENT != 90.0:
                            warnings.append(f"resource_guard.MEMORY_MAX_PERCENT={LIMITS.MEMORY_MAX_PERCENT} != 90%")
                        if LIMITS.GPU_MAX_PERCENT != 80.0:
                            warnings.append(f"resource_guard.GPU_MAX_PERCENT={LIMITS.GPU_MAX_PERCENT} != 80%")
                        if LIMITS.DISK_MAX_PERCENT != 95.0:
                            warnings.append(f"resource_guard.DISK_MAX_PERCENT={LIMITS.DISK_MAX_PERCENT} != 95%")
                except ImportError:
                    warnings.append("resource_guard module not available - cannot verify limits")

            except yaml.YAMLError as e:
                errors.append(f"YAML parse error in unified_loop.yaml: {e}")
            except Exception as e:
                errors.append(f"Error reading unified_loop.yaml: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            config_name="resource_limits",
            errors=errors,
            warnings=warnings,
        )

    def _is_valid_host(self, host: str) -> bool:
        """Check if host is a valid IP or hostname."""
        # IP address pattern
        ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
        if re.match(ip_pattern, host):
            parts = host.split(".")
            return all(0 <= int(p) <= 255 for p in parts)

        # Hostname pattern
        hostname_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9\-\.]*[a-zA-Z0-9])?$"
        return bool(re.match(hostname_pattern, host))


def validate_all_configs(base_path: Path | None = None) -> ValidationResult:
    """Convenience function to validate all configurations."""
    validator = ConfigValidator(base_path)
    return validator.validate_all()


def validate_startup() -> bool:
    """Validate configurations at startup. Returns True if valid."""
    result = validate_all_configs()

    if result.warnings:
        for warning in result.warnings:
            print(f"[ConfigValidator] WARNING: {warning}")

    if not result.valid:
        for error in result.errors:
            print(f"[ConfigValidator] ERROR: {error}")
        return False

    print(f"[ConfigValidator] All configurations valid ({len(result.warnings)} warnings)")
    return True


if __name__ == "__main__":
    import sys

    result = validate_all_configs()

    print("=== Configuration Validation ===\n")

    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  ⚠️  {w}")
        print()

    if result.errors:
        print("ERRORS:")
        for e in result.errors:
            print(f"  ✗ {e}")
        print()

    if result.valid:
        print("✓ All configurations valid")
        sys.exit(0)
    else:
        print("✗ Configuration validation failed")
        sys.exit(1)
