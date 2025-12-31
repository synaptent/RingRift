"""Quality Validators Package.

Provides data validation implementations following the ValidationResult
pattern from app.quality.types.

December 30, 2025: Created as part of Priority 3.4 consolidation effort.

Available validators:
    - BaseValidator: Abstract base class for validators
    - DatabaseValidator: Game database validation
    - NpzValidator: NPZ training data validation
"""

from app.quality.validators.base import BaseValidator, ValidatorConfig
from app.quality.validators.database import DatabaseValidator, DatabaseValidatorConfig
from app.quality.validators.npz import NpzValidator, NpzValidatorConfig

__all__ = [
    # Base classes
    "BaseValidator",
    "ValidatorConfig",
    # Database validator
    "DatabaseValidator",
    "DatabaseValidatorConfig",
    # NPZ validator
    "NpzValidator",
    "NpzValidatorConfig",
]
