# Validation Module

Data validation utilities for the RingRift AI service.

## Overview

This module provides standardized validation:

- Common validators (range, pattern, enum)
- Validation result handling
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
    has_length,
    is_non_negative,
    is_instance,
)

# Range check
validate(age, in_range(0, 120))

# Regex pattern
validate(email, matches_pattern(r"^[\w.]+@[\w.]+$"))

# Positive number
validate(count, is_positive)

# Non-empty string/list
validate(name, is_not_empty)

# Length constraint
validate(items, has_length(min_len=1, max_len=100))

# Type / numeric constraints
validate(timeout, is_instance(int), is_non_negative)
```

### Chained Validation

```python
from app.validation import each_item, has_keys, validate, validate_all

# Validate required config keys
result = validate(config, has_keys("host", "port", "timeout"))

# Validate list items
result = validate(
    scores,
    each_item(in_range(0, 1)),
)

# Validate every config key in a list
result = validate_all(
    config_keys,
    is_not_empty,
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
    print(result.errors)
    raise ValidationError(result.error_message, result.field)
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

### Batch Validation

```python
from app.validation import validate_all, is_valid_model_path

result = validate_all(model_paths, is_valid_model_path)
if not result:
    print(result.errors)
```

## Error Handling

```python
from app.validation import ValidationError

result = validate(config, has_keys("host", "port"))
if not result:
    raise ValidationError(result.error_message, result.field)
```
