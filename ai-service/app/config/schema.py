"""Configuration Schema and Validation for RingRift AI Service.

Provides schema-based configuration validation with:
- Type checking and coercion
- Range and constraint validation
- Clear error messages with paths
- Default value handling
- Environment variable interpolation

Usage:
    from app.config.schema import (
        Schema,
        Field,
        validate_config,
    )

    # Define schema
    training_schema = Schema({
        "learning_rate": Field(float, min_val=0.0, max_val=1.0, default=0.001),
        "batch_size": Field(int, min_val=1, default=256),
        "epochs": Field(int, min_val=1, required=True),
        "optimizer": Field(str, choices=["adam", "sgd", "adamw"], default="adam"),
    })

    # Validate config
    config = {"epochs": 50}
    validated = training_schema.validate(config)
    # Returns: {"learning_rate": 0.001, "batch_size": 256, "epochs": 50, "optimizer": "adam"}
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BoolField",
    "DictField",
    "EnumField",
    "Field",
    "FloatField",
    "IntField",
    "ListField",
    "PathField",
    "Schema",
    "SchemaError",
    # Field types
    "StringField",
    "ValidationError",
    "validate_config",
]

T = TypeVar("T")


# =============================================================================
# Exceptions
# =============================================================================

class SchemaError(Exception):
    """Error in schema definition."""


@dataclass
class ValidationError(Exception):
    """Error during config validation."""
    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        if self.value is not None:
            return f"{self.path}: {self.message} (got {self.value!r})"
        return f"{self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validation with errors."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    validated: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise exception if validation failed."""
        if not self.valid:
            error_msgs = "\n  ".join(str(e) for e in self.errors)
            raise ValueError(f"Configuration validation failed:\n  {error_msgs}")


# =============================================================================
# Field Definitions
# =============================================================================

@dataclass
class Field(Generic[T]):
    """Field definition for configuration schema.

    Attributes:
        type_: Expected Python type
        required: Whether field is required
        default: Default value if not provided
        default_factory: Factory for default value
        min_val: Minimum value (for numbers)
        max_val: Maximum value (for numbers)
        min_length: Minimum length (for strings/lists)
        max_length: Maximum length (for strings/lists)
        choices: Allowed values
        pattern: Regex pattern (for strings)
        validator: Custom validation function
        coerce: Whether to coerce types
        env_var: Environment variable to read from
        description: Field description
    """
    type_: type[T]
    required: bool = False
    default: T | None = None
    default_factory: Callable[[], T] | None = None
    min_val: float | None = None
    max_val: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    choices: list[T] | None = None
    pattern: str | None = None
    validator: Callable[[T], bool] | None = None
    coerce: bool = True
    env_var: str | None = None
    description: str = ""

    def __post_init__(self):
        self._compiled_pattern: Pattern | None = None
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern)

    def get_default(self) -> T | None:
        """Get the default value."""
        if self.default_factory is not None:
            return self.default_factory()
        return self.default

    def has_default(self) -> bool:
        """Check if field has a default value."""
        return self.default is not None or self.default_factory is not None

    def validate(self, value: Any, path: str = "") -> tuple[T, list[ValidationError]]:
        """Validate a value against this field.

        Args:
            value: Value to validate
            path: Path for error messages

        Returns:
            Tuple of (validated_value, errors)
        """
        errors: list[ValidationError] = []

        # Check for None/missing
        if value is None:
            if self.required:
                errors.append(ValidationError(path, "Required field is missing"))
                return None, errors  # type: ignore
            return self.get_default(), errors  # type: ignore

        # Type coercion/checking
        validated_value = value
        if not isinstance(value, self.type_):
            if self.coerce:
                try:
                    validated_value = self._coerce_value(value)
                except (ValueError, TypeError) as e:
                    errors.append(ValidationError(
                        path,
                        f"Cannot convert to {self.type_.__name__}: {e}",
                        value,
                    ))
                    return value, errors
            else:
                errors.append(ValidationError(
                    path,
                    f"Expected {self.type_.__name__}, got {type(value).__name__}",
                    value,
                ))
                return value, errors

        # Range validation
        if self.min_val is not None and validated_value < self.min_val:
            errors.append(ValidationError(
                path,
                f"Value must be >= {self.min_val}",
                validated_value,
            ))

        if self.max_val is not None and validated_value > self.max_val:
            errors.append(ValidationError(
                path,
                f"Value must be <= {self.max_val}",
                validated_value,
            ))

        # Length validation
        if hasattr(validated_value, '__len__'):
            length = len(validated_value)
            if self.min_length is not None and length < self.min_length:
                errors.append(ValidationError(
                    path,
                    f"Length must be >= {self.min_length}",
                    validated_value,
                ))
            if self.max_length is not None and length > self.max_length:
                errors.append(ValidationError(
                    path,
                    f"Length must be <= {self.max_length}",
                    validated_value,
                ))

        # Choices validation
        if self.choices is not None and validated_value not in self.choices:
            errors.append(ValidationError(
                path,
                f"Must be one of: {self.choices}",
                validated_value,
            ))

        # Pattern validation
        if self._compiled_pattern is not None:
            if not isinstance(validated_value, str):
                errors.append(ValidationError(
                    path,
                    "Pattern validation requires string",
                    validated_value,
                ))
            elif not self._compiled_pattern.match(validated_value):
                errors.append(ValidationError(
                    path,
                    f"Must match pattern: {self.pattern}",
                    validated_value,
                ))

        # Custom validator
        if self.validator is not None:
            try:
                if not self.validator(validated_value):
                    errors.append(ValidationError(
                        path,
                        "Custom validation failed",
                        validated_value,
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    path,
                    f"Validator raised error: {e}",
                    validated_value,
                ))

        return validated_value, errors

    def _coerce_value(self, value: Any) -> T:
        """Coerce value to the expected type."""
        if self.type_ == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")  # type: ignore
            return bool(value)  # type: ignore

        if self.type_ == int:
            return int(float(value))  # type: ignore

        if self.type_ == float:
            return float(value)  # type: ignore

        if self.type_ == str:
            return str(value)  # type: ignore

        if self.type_ == Path:
            return Path(value)  # type: ignore

        return self.type_(value)


# Convenience field types
def StringField(
    required: bool = False,
    default: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    choices: list[str] | None = None,
    env_var: str | None = None,
    description: str = "",
) -> Field[str]:
    """Create a string field."""
    return Field(
        type_=str,
        required=required,
        default=default,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        choices=choices,
        env_var=env_var,
        description=description,
    )


def IntField(
    required: bool = False,
    default: int | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
    env_var: str | None = None,
    description: str = "",
) -> Field[int]:
    """Create an integer field."""
    return Field(
        type_=int,
        required=required,
        default=default,
        min_val=min_val,
        max_val=max_val,
        env_var=env_var,
        description=description,
    )


def FloatField(
    required: bool = False,
    default: float | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    env_var: str | None = None,
    description: str = "",
) -> Field[float]:
    """Create a float field."""
    return Field(
        type_=float,
        required=required,
        default=default,
        min_val=min_val,
        max_val=max_val,
        env_var=env_var,
        description=description,
    )


def BoolField(
    required: bool = False,
    default: bool | None = None,
    env_var: str | None = None,
    description: str = "",
) -> Field[bool]:
    """Create a boolean field."""
    return Field(
        type_=bool,
        required=required,
        default=default,
        env_var=env_var,
        description=description,
    )


def PathField(
    required: bool = False,
    default: Path | None = None,
    must_exist: bool = False,
    env_var: str | None = None,
    description: str = "",
) -> Field[Path]:
    """Create a path field."""
    def _exists_validator(p: Path) -> bool:
        return p.exists()

    return Field(
        type_=Path,
        required=required,
        default=default,
        validator=_exists_validator if must_exist else None,
        env_var=env_var,
        description=description,
    )


def ListField(
    item_type: type[T] = Any,
    required: bool = False,
    default: list[T] | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    description: str = "",
) -> Field[list[T]]:
    """Create a list field."""
    return Field(
        type_=list,
        required=required,
        default=default,
        default_factory=list if default is None and not required else None,
        min_length=min_length,
        max_length=max_length,
        description=description,
    )


def DictField(
    required: bool = False,
    default: dict[str, Any] | None = None,
    description: str = "",
) -> Field[dict[str, Any]]:
    """Create a dict field."""
    return Field(
        type_=dict,
        required=required,
        default=default,
        default_factory=dict if default is None and not required else None,
        description=description,
    )


def EnumField(
    enum_class: type[Enum],
    required: bool = False,
    default: Enum | None = None,
    description: str = "",
) -> Field[Enum]:
    """Create an enum field."""
    return Field(
        type_=enum_class,
        required=required,
        default=default,
        choices=list(enum_class),
        description=description,
    )


# =============================================================================
# Schema
# =============================================================================

class Schema:
    """Configuration schema for validation.

    Example:
        schema = Schema({
            "name": StringField(required=True, min_length=1),
            "count": IntField(default=10, min_val=1),
            "nested": Schema({
                "enabled": BoolField(default=True),
            }),
        })

        result = schema.validate({"name": "test"})
        # {"name": "test", "count": 10, "nested": {"enabled": True}}
    """

    def __init__(
        self,
        fields: dict[str, Union[Field, Schema]],
        strict: bool = False,
        allow_extra: bool = True,
    ):
        """Initialize schema.

        Args:
            fields: Field definitions
            strict: Raise on first error
            allow_extra: Allow extra fields not in schema
        """
        self._fields = fields
        self._strict = strict
        self._allow_extra = allow_extra

    def validate(
        self,
        config: dict[str, Any],
        path: str = "",
    ) -> ValidationResult:
        """Validate a configuration dictionary.

        Args:
            config: Configuration to validate
            path: Base path for error messages

        Returns:
            ValidationResult with validated config and errors
        """
        errors: list[ValidationError] = []
        validated: dict[str, Any] = {}

        # Process env vars first
        config = self._apply_env_vars(config)

        # Validate each field
        for name, field_def in self._fields.items():
            field_path = f"{path}.{name}" if path else name
            value = config.get(name)

            if isinstance(field_def, Schema):
                # Nested schema
                nested_config = value if value is not None else {}
                if not isinstance(nested_config, dict):
                    errors.append(ValidationError(
                        field_path,
                        "Expected object for nested schema",
                        nested_config,
                    ))
                    continue

                nested_result = field_def.validate(nested_config, field_path)
                errors.extend(nested_result.errors)
                validated[name] = nested_result.validated

            else:
                # Regular field
                validated_value, field_errors = field_def.validate(value, field_path)
                errors.extend(field_errors)

                if validated_value is not None or field_def.has_default():
                    validated[name] = validated_value

            if self._strict and errors:
                break

        # Check for extra fields
        if not self._allow_extra:
            extra = set(config.keys()) - set(self._fields.keys())
            for key in extra:
                errors.append(ValidationError(
                    f"{path}.{key}" if path else key,
                    "Unknown field",
                ))

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            validated=validated,
        )

    def _apply_env_vars(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides."""
        result = dict(config)

        for name, field_def in self._fields.items():
            if isinstance(field_def, Field) and field_def.env_var:
                env_value = os.environ.get(field_def.env_var)
                if env_value is not None:
                    result[name] = env_value

        return result

    def get_defaults(self) -> dict[str, Any]:
        """Get all default values."""
        defaults: dict[str, Any] = {}

        for name, field_def in self._fields.items():
            if isinstance(field_def, Schema):
                defaults[name] = field_def.get_defaults()
            elif field_def.has_default():
                defaults[name] = field_def.get_default()

        return defaults

    def get_documentation(self) -> dict[str, Any]:
        """Generate documentation for the schema."""
        docs: dict[str, Any] = {}

        for name, field_def in self._fields.items():
            if isinstance(field_def, Schema):
                docs[name] = {
                    "type": "object",
                    "fields": field_def.get_documentation(),
                }
            else:
                doc = {
                    "type": field_def.type_.__name__,
                    "required": field_def.required,
                }
                if field_def.has_default():
                    doc["default"] = field_def.get_default()
                if field_def.description:
                    doc["description"] = field_def.description
                if field_def.choices:
                    doc["choices"] = field_def.choices
                if field_def.min_val is not None:
                    doc["min"] = field_def.min_val
                if field_def.max_val is not None:
                    doc["max"] = field_def.max_val
                if field_def.env_var:
                    doc["env_var"] = field_def.env_var

                docs[name] = doc

        return docs


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_config(
    config: dict[str, Any],
    schema: Schema,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate a configuration against a schema.

    Args:
        config: Configuration to validate
        schema: Schema to validate against
        raise_on_error: Whether to raise on validation errors

    Returns:
        Validated configuration with defaults applied

    Raises:
        ValueError: If validation fails and raise_on_error=True
    """
    result = schema.validate(config)

    if not result.valid and raise_on_error:
        result.raise_if_invalid()

    return result.validated


# =============================================================================
# Pre-built Schemas
# =============================================================================

# Training configuration schema
TRAINING_SCHEMA = Schema({
    "learning_rate": FloatField(default=0.001, min_val=0.0, max_val=1.0,
                                env_var="TRAINING_LR"),
    "batch_size": IntField(default=256, min_val=1, max_val=65536,
                           env_var="TRAINING_BATCH_SIZE"),
    "epochs": IntField(default=100, min_val=1),
    "weight_decay": FloatField(default=0.0001, min_val=0.0),
    "dropout": FloatField(default=0.1, min_val=0.0, max_val=1.0),
    "optimizer": StringField(default="adam", choices=["adam", "sgd", "adamw"]),
    "gradient_clip": FloatField(default=1.0, min_val=0.0),
    "validation_split": FloatField(default=0.15, min_val=0.0, max_val=0.5),
})

# Selfplay configuration schema
SELFPLAY_SCHEMA = Schema({
    "games_per_batch": IntField(default=64, min_val=1),
    "parallel_games": IntField(default=8, min_val=1),
    "mcts_simulations": IntField(default=800, min_val=100),
    "temperature": FloatField(default=1.0, min_val=0.0),
    "temperature_threshold": IntField(default=30, min_val=0),
    "dirichlet_alpha": FloatField(default=0.3, min_val=0.0),
    "dirichlet_epsilon": FloatField(default=0.25, min_val=0.0, max_val=1.0),
})

# Evaluation configuration schema
EVALUATION_SCHEMA = Schema({
    "shadow_games": IntField(default=20, min_val=1),
    "full_tournament_games": IntField(default=100, min_val=10),
    "promotion_threshold": FloatField(default=0.55, min_val=0.5, max_val=1.0),
    "elo_k_factor": IntField(default=32, min_val=1),
})
