"""Tests for app.config.config_validator - Configuration Validation.

This module tests the configuration validation system including:
- ValidationResult dataclass
- ConfigValidator class
- validate_all_configs function
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.config.config_validator import (
    ConfigValidator,
    ValidationResult,
    validate_all_configs,
)


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Should create valid result."""
        result = ValidationResult(valid=True, config_name="test.yaml")
        assert result.valid is True
        assert result.config_name == "test.yaml"
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self):
        """Should create invalid result with errors."""
        result = ValidationResult(
            valid=False,
            config_name="test.yaml",
            errors=["Error 1", "Error 2"],
        )
        assert result.valid is False
        assert len(result.errors) == 2

    def test_result_with_warnings(self):
        """Should allow warnings on valid result."""
        result = ValidationResult(
            valid=True,
            config_name="test.yaml",
            warnings=["Warning 1"],
        )
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_merge_two_valid(self):
        """Should merge two valid results."""
        r1 = ValidationResult(valid=True, config_name="a.yaml")
        r2 = ValidationResult(valid=True, config_name="b.yaml")
        merged = r1.merge(r2)
        assert merged.valid is True
        assert "a.yaml" in merged.config_name
        assert "b.yaml" in merged.config_name

    def test_merge_valid_and_invalid(self):
        """Should result in invalid when merging valid and invalid."""
        r1 = ValidationResult(valid=True, config_name="a.yaml")
        r2 = ValidationResult(valid=False, config_name="b.yaml", errors=["error"])
        merged = r1.merge(r2)
        assert merged.valid is False

    def test_merge_combines_errors(self):
        """Should combine errors from both results."""
        r1 = ValidationResult(valid=False, config_name="a.yaml", errors=["e1"])
        r2 = ValidationResult(valid=False, config_name="b.yaml", errors=["e2"])
        merged = r1.merge(r2)
        assert len(merged.errors) == 2
        assert "e1" in merged.errors
        assert "e2" in merged.errors

    def test_merge_combines_warnings(self):
        """Should combine warnings from both results."""
        r1 = ValidationResult(valid=True, config_name="a.yaml", warnings=["w1"])
        r2 = ValidationResult(valid=True, config_name="b.yaml", warnings=["w2"])
        merged = r1.merge(r2)
        assert len(merged.warnings) == 2

    def test_config_path_optional(self):
        """Should allow config_path to be None."""
        result = ValidationResult(valid=True, config_name="test")
        assert result.config_path is None


# =============================================================================
# ConfigValidator Tests
# =============================================================================


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_init_default_path(self):
        """Should use default base path."""
        validator = ConfigValidator()
        assert validator.base_path is not None

    def test_init_custom_path(self):
        """Should accept custom base path."""
        custom_path = Path("/custom/path")
        validator = ConfigValidator(base_path=custom_path)
        assert validator.base_path == custom_path

    def test_validate_missing_unified_loop(self):
        """Should report error for missing unified_loop.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is False
            assert any("not found" in e for e in result.errors)

    def test_validate_empty_unified_loop(self):
        """Should report error for empty config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "unified_loop.yaml"
            config_file.write_text("")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is False
            assert any("empty" in e.lower() for e in result.errors)

    def test_validate_valid_unified_loop(self):
        """Should pass for valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "unified_loop.yaml"
            config_file.write_text("""
data_ingestion:
  poll_interval_seconds: 300
evaluation:
  shadow_interval_seconds: 600
  games_per_shadow_match: 10
training:
  min_games_for_training: 1000
promotion:
  elo_threshold: 50
""")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is True
            assert len(result.errors) == 0

    def test_validate_missing_required_sections(self):
        """Should report error for missing required sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "unified_loop.yaml"
            config_file.write_text("""
data_ingestion:
  poll_interval_seconds: 300
""")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is False
            assert any("evaluation" in e for e in result.errors)
            assert any("training" in e for e in result.errors)
            assert any("promotion" in e for e in result.errors)

    def test_validate_warns_on_low_poll_interval(self):
        """Should warn on low poll interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "unified_loop.yaml"
            config_file.write_text("""
data_ingestion:
  poll_interval_seconds: 5
evaluation:
  games_per_shadow_match: 10
training:
  min_games_for_training: 1000
promotion:
  elo_threshold: 50
""")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert any("poll_interval" in w and "high load" in w for w in result.warnings)

    def test_validate_errors_on_low_games_per_match(self):
        """Should error on games_per_shadow_match < 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_file = config_dir / "unified_loop.yaml"
            config_file.write_text("""
data_ingestion:
  poll_interval_seconds: 300
evaluation:
  games_per_shadow_match: 1
training:
  min_games_for_training: 1000
promotion:
  elo_threshold: 50
""")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is False
            assert any("games_per_shadow_match" in e for e in result.errors)

    def test_validate_remote_hosts_optional(self):
        """Should not fail if remote_hosts.yaml is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_remote_hosts()
            assert result.valid is True

    def test_validate_all_with_missing_configs(self):
        """Should validate all configs and report issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_all()
            # Missing unified_loop.yaml should cause failure
            assert result.valid is False

    def test_validate_all_can_skip_unified_loop_for_inference_only(self):
        """Inference-only validation should not require unified_loop.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            (config_dir / "distributed_hosts.yaml").write_text(
                """
hosts:
  test-node:
    tailscale_ip: 100.64.0.1
    ssh_user: ubuntu
"""
            )
            (config_dir / "hyperparameters.json").write_text('{"configs": {}}')

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_all(include_unified_loop=False)
            assert result.valid is True
            assert all("unified_loop" not in error for error in result.errors)


# =============================================================================
# validate_all_configs Tests
# =============================================================================


class TestValidateAllConfigs:
    """Tests for validate_all_configs function."""

    def test_returns_validation_result(self):
        """Should return a ValidationResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_all_configs(base_path=Path(tmpdir))
            assert isinstance(result, ValidationResult)

    def test_default_path_is_used(self):
        """Should use default path when none provided."""
        # This will validate the actual config files
        result = validate_all_configs()
        assert isinstance(result, ValidationResult)
        # Result validity depends on actual config state

    def test_wrapper_supports_skipping_unified_loop(self):
        """Wrapper should expose the inference-only skip flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            (config_dir / "distributed_hosts.yaml").write_text(
                """
hosts:
  test-node:
    tailscale_ip: 100.64.0.1
    ssh_user: ubuntu
"""
            )
            (config_dir / "hyperparameters.json").write_text('{"configs": {}}')

            result = validate_all_configs(
                base_path=Path(tmpdir),
                include_unified_loop=False,
            )
            assert result.valid is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestConfigValidatorIntegration:
    """Integration tests for ConfigValidator."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            # Create valid unified_loop.yaml
            (config_dir / "unified_loop.yaml").write_text("""
data_ingestion:
  poll_interval_seconds: 300
  max_concurrent_syncs: 5
evaluation:
  shadow_interval_seconds: 600
  games_per_shadow_match: 10
training:
  min_games_for_training: 1000
  epochs: 20
promotion:
  elo_threshold: 50
  min_games_evaluated: 100
""")

            validator = ConfigValidator(base_path=Path(tmpdir))

            # Validate unified_loop
            ul_result = validator.validate_unified_loop_config()
            assert ul_result.valid is True

            # Validate hyperparameters - it's optional and may look for .json
            # so we just check that it returns a ValidationResult
            hp_result = validator.validate_hyperparameters()
            assert isinstance(hp_result, ValidationResult)

    def test_yaml_parse_error_handled(self):
        """Should handle YAML parse errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()

            # Create invalid YAML
            (config_dir / "unified_loop.yaml").write_text("""
invalid: yaml: content:
  - not: valid
  indentation problems here
""")

            validator = ConfigValidator(base_path=Path(tmpdir))
            result = validator.validate_unified_loop_config()
            assert result.valid is False
            assert any("YAML" in e or "parse" in e.lower() for e in result.errors)
