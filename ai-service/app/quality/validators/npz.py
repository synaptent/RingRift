"""NPZ Validator.

Validates NPZ training data files for integrity and quality.

December 30, 2025: Created as part of Priority 3.4 consolidation effort.
Migrates validation logic from app/training/data_quality.py.

Usage:
    from app.quality.validators.npz import NpzValidator

    validator = NpzValidator()
    result = validator.validate("data/training/hex8_2p.npz")
    if not result.is_valid:
        print(f"Errors: {result.errors}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.quality.types import ValidationResult
from app.quality.validators.base import PathValidator, ValidatorConfig

__all__ = [
    "NpzValidator",
    "NpzValidatorConfig",
]

logger = logging.getLogger(__name__)


def safe_load_npz(path: Path) -> dict[str, np.ndarray]:
    """Safely load NPZ file.

    Args:
        path: Path to NPZ file

    Returns:
        Dictionary of arrays
    """
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@dataclass
class NpzValidatorConfig(ValidatorConfig):
    """Configuration for NPZ validation.

    Attributes:
        min_samples: Minimum number of samples required
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for infinite values
        check_policy: Whether to validate policy arrays
        value_range: Expected value range for targets
        policy_sum_tolerance: Tolerance for policy sum validation
    """

    min_samples: int = 0
    check_nan: bool = True
    check_inf: bool = True
    check_policy: bool = True
    value_range: tuple[float, float] = (-1.0, 1.0)
    policy_sum_tolerance: float = 0.01


class NpzValidator(PathValidator):
    """Validator for NPZ training data files.

    Checks:
    - File exists and is readable
    - Required arrays present (features, values)
    - Array shapes are consistent
    - Data types are valid
    - No NaN/Inf values (optional)
    - Policy arrays sum correctly (optional)
    """

    VALIDATOR_NAME = "npz"
    VALIDATOR_VERSION = "1.0.0"

    # Required arrays in NPZ file
    REQUIRED_ARRAYS = {"features", "values"}

    # Optional arrays
    OPTIONAL_ARRAYS = {"policy_indices", "policy_values", "globals", "weights"}

    def __init__(self, config: NpzValidatorConfig | None = None):
        """Initialize the NPZ validator.

        Args:
            config: Validator configuration
        """
        self._npz_config = config or NpzValidatorConfig()
        super().__init__(config=self._npz_config)

    def _validate_file(self, path: Path) -> ValidationResult:
        """Validate NPZ file contents."""
        result = ValidationResult(is_valid=True)

        try:
            data = safe_load_npz(path)

            # Check required arrays
            array_errors = self._check_required_arrays(data)
            for error in array_errors:
                result.add_error(error)

            if not result.is_valid:
                return result

            # Validate array shapes
            shape_errors = self._check_shapes(data)
            for error in shape_errors:
                result.add_error(error)

            if not result.is_valid:
                return result

            # Check data quality
            quality_errors, warnings = self._check_data_quality(data)
            for error in quality_errors:
                result.add_error(error)
            for warning in warnings:
                result.add_warning(warning)

            # Check policy arrays
            if self._npz_config.check_policy:
                policy_errors = self._check_policy_arrays(data)
                for error in policy_errors:
                    result.add_error(error)

            # Get metadata
            if result.is_valid:
                result.metadata.update(self._get_metadata(data))

        except Exception as e:
            result.add_error(f"NPZ validation error: {e}")

        return result

    def _check_required_arrays(self, data: dict[str, np.ndarray]) -> list[str]:
        """Check required arrays are present."""
        errors = []

        available = set(data.keys())
        missing = self.REQUIRED_ARRAYS - available

        if missing:
            errors.append(f"Missing required arrays: {missing}")

        return errors

    def _check_shapes(self, data: dict[str, np.ndarray]) -> list[str]:
        """Check array shapes are valid and consistent."""
        errors = []

        features = data["features"]
        values = data["values"]

        # Features should be 4D: (N, C, H, W)
        if features.ndim != 4:
            errors.append(f"Features should be 4D, got {features.ndim}D")

        # Values should be 1D or 2D: (N,) or (N, num_players)
        if values.ndim not in (1, 2):
            errors.append(f"Values should be 1D or 2D, got {values.ndim}D")

        # Sample counts should match
        if len(features) != len(values):
            errors.append(
                f"Sample count mismatch: features={len(features)}, "
                f"values={len(values)}"
            )

        # Check minimum samples
        if len(features) < self._npz_config.min_samples:
            errors.append(
                f"Insufficient samples: {len(features)} < "
                f"{self._npz_config.min_samples}"
            )

        return errors

    def _check_data_quality(
        self, data: dict[str, np.ndarray]
    ) -> tuple[list[str], list[str]]:
        """Check data quality (NaN, Inf, value ranges)."""
        errors = []
        warnings = []

        features = data["features"]
        values = data["values"]

        # Check for NaN
        if self._npz_config.check_nan:
            if np.any(np.isnan(features)):
                errors.append("Features contain NaN values")
            if np.any(np.isnan(values)):
                errors.append("Values contain NaN values")

        # Check for Inf
        if self._npz_config.check_inf:
            if np.any(np.isinf(features)):
                errors.append("Features contain infinite values")
            if np.any(np.isinf(values)):
                errors.append("Values contain infinite values")

        # Check value range
        min_val, max_val = self._npz_config.value_range
        if np.any(values < min_val) or np.any(values > max_val):
            warnings.append(
                f"Values outside expected range [{min_val}, {max_val}]"
            )

        return errors, warnings

    def _check_policy_arrays(self, data: dict[str, np.ndarray]) -> list[str]:
        """Check policy array validity."""
        errors = []

        if "policy_indices" in data:
            if "policy_values" not in data:
                errors.append(
                    "policy_indices present but policy_values missing"
                )
                return errors

            policy_indices = data["policy_indices"]
            policy_values = data["policy_values"]

            # Check lengths match features
            if len(policy_indices) != len(data["features"]):
                errors.append("Policy data length mismatch with features")

            if len(policy_values) != len(data["features"]):
                errors.append("Policy values length mismatch with features")

            # Check policy sums (sample a subset for efficiency)
            if len(policy_values) > 0:
                sample_size = min(100, len(policy_values))
                sample_indices = np.random.choice(
                    len(policy_values), sample_size, replace=False
                )

                for idx in sample_indices:
                    policy_sum = np.sum(policy_values[idx])
                    if abs(policy_sum - 1.0) > self._npz_config.policy_sum_tolerance:
                        errors.append(
                            f"Policy sum deviation: {policy_sum:.4f} != 1.0"
                        )
                        break  # Just report one error

        return errors

    def _get_metadata(self, data: dict[str, np.ndarray]) -> dict[str, Any]:
        """Extract NPZ metadata."""
        features = data["features"]
        values = data["values"]

        metadata: dict[str, Any] = {
            "num_samples": len(features),
            "feature_shape": features.shape,
            "feature_dtype": str(features.dtype),
            "value_shape": values.shape,
            "value_dtype": str(values.dtype),
            "arrays_present": list(data.keys()),
        }

        # Add statistics
        metadata["feature_stats"] = {
            "mean": float(np.mean(features)),
            "std": float(np.std(features)),
            "min": float(np.min(features)),
            "max": float(np.max(features)),
        }

        metadata["value_stats"] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        return metadata

    def get_file_stats(self, path: str | Path) -> dict[str, Any]:
        """Get detailed NPZ file statistics.

        Args:
            path: Path to NPZ file

        Returns:
            Dictionary with file statistics
        """
        path = Path(path)
        stats: dict[str, Any] = {"path": str(path), "valid": False}

        if not path.exists():
            stats["error"] = "File not found"
            return stats

        try:
            data = safe_load_npz(path)

            features = data.get("features")
            if features is not None:
                stats["num_samples"] = len(features)
                stats["feature_shape"] = features.shape
                stats["num_channels"] = features.shape[1] if features.ndim > 1 else 0

                # Per-channel statistics
                if features.ndim == 4:
                    channel_stats = []
                    for c in range(min(features.shape[1], 10)):  # Limit to first 10
                        channel = features[:, c, :, :]
                        channel_stats.append({
                            "channel": c,
                            "mean": float(np.mean(channel)),
                            "std": float(np.std(channel)),
                            "sparsity": float(np.mean(channel == 0)),
                        })
                    stats["channel_stats"] = channel_stats

            values = data.get("values")
            if values is not None:
                stats["value_range"] = [float(np.min(values)), float(np.max(values))]
                stats["value_mean"] = float(np.mean(values))

            stats["arrays"] = list(data.keys())
            stats["valid"] = True

        except Exception as e:
            stats["error"] = str(e)

        return stats
