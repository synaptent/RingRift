# Validation Module

Data validation utilities for the RingRift AI service.

## Overview

This module provides standardized validation:

- Common validators (range, pattern, enum)
- Validation result handling
- Pydantic integration helpers
- Domain-specific validators

## Key Components

### Basic Validation

```python
from app.validation import validate, in_range, is_positive, is_not_empty

# Simple validation
result = validate(value, in_range(0, 100))
if not result:
    print(result.errors)

# Multiple validators
result = validate(
    config,
    is_not_empty,
    has_keys("host", "port"),
)
```

### Common Validators

```python
from app.validation import (
    in_range,
    matches_pattern,
    is_positive,
    is_not_empty,
    is_one_of,
    has_length,
)

# Range check
validate(age, in_range(0, 120))

# Regex pattern
validate(email, matches_pattern(r"^[\w.]+@[\w.]+$"))

# Positive number
validate(count, is_positive)

# Non-empty string/list
validate(name, is_not_empty)

# Enum values
validate(status, is_one_of("pending", "running", "completed"))

# Length constraint
validate(items, has_length(min=1, max=100))
```

### Chained Validation

```python
from app.validation import validate, each_value, each_key

# Validate dict values
result = validate(
    config,
    has_keys("host", "port", "timeout"),
    each_value(is_not_empty),
)

# Validate list items
result = validate(
    scores,
    each_item(in_range(0, 1)),
)
```

### Domain Validators

```python
from app.validation import (
    is_valid_board_type,
    is_valid_model_path,
    is_valid_elo,
    is_valid_config_key,
)

# Board type validation
validate(board, is_valid_board_type)  # hex8, square8, etc.

# Model path exists and is valid
validate(path, is_valid_model_path)

# ELO rating in reasonable range
validate(elo, is_valid_elo)  # 0-3000
```

### Validation Results

```python
from app.validation import ValidationResult, ValidationError

result = validate(data, my_validator)

if result.is_valid:
    proceed(data)
else:
    for error in result.errors:
        print(f"Field: {error.field}, Message: {error.message}")

# Raise on invalid
try:
    result.raise_if_invalid()
except ValidationError as e:
    handle_error(e)
```

### Custom Validators

```python
from app.validation import Validator

class IsValidGameId(Validator):
    def validate(self, value) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult.invalid("Must be string")
        if not value.startswith("game_"):
            return ValidationResult.invalid("Must start with 'game_'")
        return ValidationResult.valid()

# Use custom validator
validate(game_id, IsValidGameId())
```

### Pydantic Integration

```python
from app.validation import pydantic_validator

# Create Pydantic-compatible validator
@pydantic_validator
def validate_port(v):
    if not 1 <= v <= 65535:
        raise ValueError("Port must be 1-65535")
    return v

# Use in Pydantic model
class Config(BaseModel):
    port: int

    _validate_port = validator("port", allow_reuse=True)(validate_port)
```

## Error Handling

```python
from app.validation import ValidationError

try:
    validate(config, strict=True)  # Raises on failure
except ValidationError as e:
    print(f"Validation failed: {e}")
    for field, errors in e.field_errors.items():
        print(f"  {field}: {errors}")
```
