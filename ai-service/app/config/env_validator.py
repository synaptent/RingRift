"""Environment Variable Validator for RingRift AI Service.

Validates all required environment variables at startup to catch
misconfiguration early before expensive operations begin.

Usage:
    from app.config.env_validator import validate_environment, EnvironmentError

    # Validate at startup
    try:
        validate_environment()
    except EnvironmentError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Or check without raising
    result = check_environment()
    if not result.valid:
        for error in result.errors:
            print(error)

Environment Variable Categories:
    - Required: Must be set (validation fails if missing)
    - Recommended: Should be set (warning if missing)
    - Optional: Nice to have (no warning if missing)

December 2025: Created as part of Phase 8 Configuration Consolidation
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EnvVarPriority(Enum):
    """Priority level for environment variables."""
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


@dataclass
class EnvVar:
    """Definition of an environment variable with validation rules."""
    name: str
    priority: EnvVarPriority = EnvVarPriority.OPTIONAL
    default: str | None = None
    description: str = ""
    validator: Callable[[str], bool] | None = None
    validator_error: str | None = None

    @property
    def required(self) -> bool:
        return self.priority == EnvVarPriority.REQUIRED

    def get_value(self) -> str | None:
        """Get the value from environment or default."""
        return os.environ.get(self.name, self.default)

    def is_set(self) -> bool:
        """Check if the variable is set in environment."""
        return self.name in os.environ


@dataclass
class ValidationResult:
    """Result of environment validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    values: dict[str, str | None] = field(default_factory=dict)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one."""
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            values={**self.values, **other.values},
        )


class EnvironmentError(Exception):
    """Raised when environment validation fails."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = "Environment validation failed:\n"
        for error in self.errors:
            msg += f"  - {error}\n"
        return msg


# =============================================================================
# Validators
# =============================================================================

def is_valid_path(value: str) -> bool:
    """Validate that the value is a valid path (can be created)."""
    try:
        path = Path(value)
        # Check if parent exists or can be created
        if path.exists():
            return True
        parent = path.parent
        return parent.exists() or parent == path
    except Exception:
        return False


def is_valid_url(value: str) -> bool:
    """Validate that the value is a valid URL."""
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return url_pattern.match(value) is not None


def is_valid_log_level(value: str) -> bool:
    """Validate that the value is a valid log level."""
    return value.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def is_positive_int(value: str) -> bool:
    """Validate that the value is a positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False


def is_non_negative_int(value: str) -> bool:
    """Validate that the value is a non-negative integer."""
    try:
        return int(value) >= 0
    except ValueError:
        return False


def is_float_range(min_val: float, max_val: float) -> Callable[[str], bool]:
    """Create a validator for float values in a range."""
    def validator(value: str) -> bool:
        try:
            f = float(value)
            return min_val <= f <= max_val
        except ValueError:
            return False
    return validator


def is_boolean(value: str) -> bool:
    """Validate that the value is a boolean-like string."""
    return value.lower() in ("true", "false", "1", "0", "yes", "no")


# =============================================================================
# Environment Variable Definitions
# =============================================================================

# Core path variables
DATA_DIR = EnvVar(
    name="RINGRIFT_DATA_DIR",
    priority=EnvVarPriority.RECOMMENDED,
    default="data",
    description="Base directory for all data (games, training, models)",
    validator=is_valid_path,
    validator_error="Must be a valid directory path",
)

MODEL_DIR = EnvVar(
    name="RINGRIFT_MODEL_DIR",
    priority=EnvVarPriority.RECOMMENDED,
    default="models",
    description="Directory for trained model checkpoints",
    validator=is_valid_path,
    validator_error="Must be a valid directory path",
)

LOG_DIR = EnvVar(
    name="RINGRIFT_LOG_DIR",
    priority=EnvVarPriority.OPTIONAL,
    default="logs",
    description="Directory for log files",
    validator=is_valid_path,
    validator_error="Must be a valid directory path",
)

# Configuration paths
CONFIG_PATH = EnvVar(
    name="RINGRIFT_CONFIG_PATH",
    priority=EnvVarPriority.OPTIONAL,
    default="config/unified_loop.yaml",
    description="Path to main configuration file",
)

ELO_DB = EnvVar(
    name="RINGRIFT_ELO_DB",
    priority=EnvVarPriority.OPTIONAL,
    default="data/unified_elo.db",
    description="Path to Elo database",
)

# Logging and debugging
LOG_LEVEL = EnvVar(
    name="RINGRIFT_LOG_LEVEL",
    priority=EnvVarPriority.OPTIONAL,
    default="INFO",
    description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    validator=is_valid_log_level,
    validator_error="Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
)

TRACE_DEBUG = EnvVar(
    name="RINGRIFT_TRACE_DEBUG",
    priority=EnvVarPriority.OPTIONAL,
    default="0",
    description="Enable trace-level debugging (0/1)",
    validator=is_boolean,
    validator_error="Must be 0 or 1",
)

# Training thresholds
TRAINING_THRESHOLD = EnvVar(
    name="RINGRIFT_TRAINING_THRESHOLD",
    priority=EnvVarPriority.OPTIONAL,
    default="500",
    description="Number of games to trigger training",
    validator=is_positive_int,
    validator_error="Must be a positive integer",
)

MIN_GAMES_FOR_TRAINING = EnvVar(
    name="RINGRIFT_MIN_GAMES_FOR_TRAINING",
    priority=EnvVarPriority.OPTIONAL,
    default="500",
    description="Legacy: minimum games for training (use RINGRIFT_TRAINING_THRESHOLD)",
    validator=is_positive_int,
    validator_error="Must be a positive integer",
)

# P2P Cluster
P2P_PORT = EnvVar(
    name="RINGRIFT_P2P_PORT",
    priority=EnvVarPriority.OPTIONAL,
    default="8770",
    description="Port for P2P orchestrator",
    validator=is_positive_int,
    validator_error="Must be a valid port number",
)

CLUSTER_AUTH_TOKEN = EnvVar(
    name="RINGRIFT_CLUSTER_AUTH_TOKEN",
    priority=EnvVarPriority.OPTIONAL,
    default=None,
    description="Authentication token for cluster operations",
)

# Feature flags
SKIP_SHADOW_CONTRACTS = EnvVar(
    name="RINGRIFT_SKIP_SHADOW_CONTRACTS",
    priority=EnvVarPriority.OPTIONAL,
    default="true",
    description="Skip shadow contract validation (performance default)",
    validator=is_boolean,
    validator_error="Must be true or false",
)

PARITY_VALIDATION = EnvVar(
    name="RINGRIFT_PARITY_VALIDATION",
    priority=EnvVarPriority.OPTIONAL,
    default="off",
    description="Parity validation mode: off, warn, strict",
)

# Process management
JOB_GRACE_PERIOD = EnvVar(
    name="RINGRIFT_JOB_GRACE_PERIOD",
    priority=EnvVarPriority.OPTIONAL,
    default="60",
    description="Seconds to wait before SIGKILL after SIGTERM",
    validator=is_non_negative_int,
    validator_error="Must be a non-negative integer",
)

GPU_IDLE_THRESHOLD = EnvVar(
    name="RINGRIFT_GPU_IDLE_THRESHOLD",
    priority=EnvVarPriority.OPTIONAL,
    default="600",
    description="Seconds of GPU idle before killing stuck processes",
    validator=is_positive_int,
    validator_error="Must be a positive integer",
)

# All defined environment variables
ALL_ENV_VARS: list[EnvVar] = [
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    CONFIG_PATH,
    ELO_DB,
    LOG_LEVEL,
    TRACE_DEBUG,
    TRAINING_THRESHOLD,
    MIN_GAMES_FOR_TRAINING,
    P2P_PORT,
    CLUSTER_AUTH_TOKEN,
    SKIP_SHADOW_CONTRACTS,
    PARITY_VALIDATION,
    JOB_GRACE_PERIOD,
    GPU_IDLE_THRESHOLD,
]


# =============================================================================
# Validation Functions
# =============================================================================

def check_environment(
    vars_to_check: list[EnvVar] | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Check environment variables without raising exceptions.

    Args:
        vars_to_check: List of EnvVar to validate (defaults to ALL_ENV_VARS)
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with errors, warnings, and values
    """
    vars_to_check = vars_to_check or ALL_ENV_VARS
    errors: list[str] = []
    warnings: list[str] = []
    values: dict[str, str | None] = {}

    for env_var in vars_to_check:
        value = env_var.get_value()
        values[env_var.name] = value

        # Check if required variable is missing
        if env_var.required and value is None:
            errors.append(
                f"Missing required environment variable: {env_var.name}"
                + (f" - {env_var.description}" if env_var.description else "")
            )
            continue

        # Check if recommended variable is missing
        if env_var.priority == EnvVarPriority.RECOMMENDED and not env_var.is_set():
            msg = (
                f"Recommended environment variable not set: {env_var.name}"
                + (f" (using default: {env_var.default})" if env_var.default else "")
            )
            if strict:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Validate value if present and validator exists
        if value is not None and env_var.validator:
            if not env_var.validator(value):
                error_msg = (
                    f"Invalid value for {env_var.name}: '{value}'"
                    + (f" - {env_var.validator_error}" if env_var.validator_error else "")
                )
                errors.append(error_msg)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        values=values,
    )


def validate_environment(
    strict: bool = False,
    raise_on_warning: bool = False,
) -> dict[str, str | None]:
    """Validate environment and raise on errors.

    Args:
        strict: If True, treat recommended vars as required
        raise_on_warning: If True, raise on warnings too

    Returns:
        Dictionary of environment variable values

    Raises:
        EnvironmentError: If validation fails
    """
    result = check_environment(strict=strict)

    # Log warnings
    for warning in result.warnings:
        logger.warning(warning)

    # Check for errors
    if not result.valid:
        raise EnvironmentError(result.errors, result.warnings)

    # Optionally treat warnings as errors
    if raise_on_warning and result.warnings:
        raise EnvironmentError(result.warnings, [])

    return result.values


def validate_startup() -> bool:
    """Validate environment at startup. Returns True if valid.

    This is a convenience function that logs errors/warnings
    and returns a boolean for use in startup scripts.
    """
    result = check_environment()

    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"[EnvValidator] {warning}")

    if not result.valid:
        for error in result.errors:
            logger.error(f"[EnvValidator] {error}")
        return False

    logger.info(f"[EnvValidator] Environment valid ({len(result.warnings)} warnings)")
    return True


def get_env_value(var: EnvVar, allow_missing: bool = True) -> str | None:
    """Get a specific environment variable value with validation.

    Args:
        var: The EnvVar definition
        allow_missing: If False, raise if required var is missing

    Returns:
        The value or None

    Raises:
        EnvironmentError: If required var is missing and allow_missing=False
    """
    value = var.get_value()

    if value is None and var.required and not allow_missing:
        raise EnvironmentError(
            [f"Missing required environment variable: {var.name}"]
        )

    if value is not None and var.validator and not var.validator(value):
        raise EnvironmentError(
            [f"Invalid value for {var.name}: {value}"]
        )

    return value


def print_env_documentation() -> None:
    """Print documentation for all environment variables."""
    print("RingRift Environment Variables")
    print("=" * 60)
    print()

    # Group by priority
    for priority in EnvVarPriority:
        vars_in_priority = [v for v in ALL_ENV_VARS if v.priority == priority]
        if not vars_in_priority:
            continue

        print(f"### {priority.value.upper()} ###")
        print()

        for var in vars_in_priority:
            print(f"  {var.name}")
            if var.description:
                print(f"    Description: {var.description}")
            if var.default:
                print(f"    Default: {var.default}")
            if var.validator_error:
                print(f"    Validation: {var.validator_error}")
            print()


# =============================================================================
# Module-level convenience exports
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(DATA_DIR.get_value() or "data")


def get_model_dir() -> Path:
    """Get the model directory path."""
    return Path(MODEL_DIR.get_value() or "models")


def get_log_dir() -> Path:
    """Get the log directory path."""
    return Path(LOG_DIR.get_value() or "logs")


def get_log_level() -> str:
    """Get the configured log level."""
    return (LOG_LEVEL.get_value() or "INFO").upper()


def is_trace_debug_enabled() -> bool:
    """Check if trace debugging is enabled."""
    value = TRACE_DEBUG.get_value()
    return value is not None and value.lower() in ("1", "true", "yes")


if __name__ == "__main__":
    import sys

    if "--docs" in sys.argv:
        print_env_documentation()
        sys.exit(0)

    result = check_environment()

    print("=== Environment Validation ===\n")

    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(f"  WARNING: {w}")
        print()

    if result.errors:
        print("ERRORS:")
        for e in result.errors:
            print(f"  ERROR: {e}")
        print()

    if result.valid:
        print("Environment valid")
        sys.exit(0)
    else:
        print("Environment validation failed")
        sys.exit(1)
