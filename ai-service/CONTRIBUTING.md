# Contributing to RingRift AI Service

Thank you for your interest in contributing to RingRift! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Architecture Overview](#architecture-overview)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

### Prerequisites

- Python 3.10+ (Docker/CI uses 3.11)
- Git
- (Optional) NVIDIA GPU with CUDA for training

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/RingRift.git
cd RingRift/ai-service
```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

### 3. Install Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
# Run the service locally
uvicorn app.main:app --reload --port 8001

# Run tests
pytest tests/ -v
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-mcts-parallelization`
- `fix/nnue-memory-leak`
- `docs/update-training-guide`
- `refactor/simplify-data-loader`

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:

```
feat(training): add curriculum learning support
fix(nnue): resolve memory leak in batch inference
docs(readme): update cluster setup instructions
```

## Testing

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_heuristic_ai.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names
- Include docstrings explaining what's being tested

```python
def test_heuristic_ai_selects_valid_move():
    """HeuristicAI should always return a legal move."""
    ai = HeuristicAI(difficulty=2)
    game_state = create_test_game_state()

    move = ai.select_move(game_state, player_number=1)

    assert move is not None
    assert is_valid_move(game_state, move)
```

## Pull Request Process

### Before Submitting

1. **Run tests**: `pytest tests/ -v`
2. **Check linting**: `ruff check .`
3. **Format code**: `ruff format .`
4. **Update documentation** if needed

### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Tests pass locally
- [ ] New tests added (if applicable)

## Checklist

- [ ] Code follows project style
- [ ] Self-reviewed my code
- [ ] Commented complex code
- [ ] Updated documentation
```

### Review Process

1. Create PR against `main` branch
2. Ensure CI passes
3. Request review from maintainers
4. Address feedback
5. Squash and merge when approved

## Code Style

### Python Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Key Conventions

- Use type hints for function signatures
- Document public functions with docstrings
- Keep functions focused and small
- Prefer composition over inheritance
- Use meaningful variable names

### Unused Variables and Parameters

When a function parameter is intentionally unused (e.g., for API compatibility or future use), use one of these patterns:

```python
# Option 1: Prefix with underscore (preferred for parameters)
def my_function(used_param: int, _unused_param: str = "default") -> None:
    print(used_param)

# Option 2: Assign to _ inside function (preferred by linter)
def my_function(used_param: int, unused_param: str = "default") -> None:
    _ = unused_param  # Explicitly mark as intentionally unused
    print(used_param)
```

The linter prefers Option 2, but both are acceptable. Never remove documented parameters as they may be part of the public API.

### Dead Code Detection

We use [vulture](https://github.com/jendrikseipp/vulture) to detect dead code:

```bash
# Check for dead code with 100% confidence
vulture app/ scripts/ --min-confidence 100
```

This runs automatically in CI. To mark code as intentionally unused:

- Prefix unused parameters with `_`
- Use `_ = variable` for unused locals

```python
def train_model(
    config: TrainConfig,
    data_path: Path,
    output_dir: Path,
) -> TrainingResult:
    """
    Train an NNUE policy model.

    Args:
        config: Training configuration
        data_path: Path to training data
        output_dir: Directory for checkpoints

    Returns:
        TrainingResult with metrics and model path
    """
    ...
```

## Architecture Overview

### Directory Structure

```
ai-service/
├── app/                    # Main application code
│   ├── ai/                 # AI implementations (NNUE, MCTS, etc.)
│   ├── training/           # Training pipeline
│   ├── rules/              # Game rules engine
│   └── models/             # Pydantic models
├── scripts/                # CLI tools and utilities
├── tests/                  # Test suite
├── config/                 # Configuration files
├── docs/                   # Documentation
└── models/                 # Trained model checkpoints
```

### Key Components

| Component                | Description                   |
| ------------------------ | ----------------------------- |
| `app/ai/nnue_policy.py`  | NNUE policy network           |
| `app/ai/mcts_ai.py`      | Monte Carlo Tree Search       |
| `app/training/train.py`  | Main training loop            |
| `scripts/master_loop.py` | Self-improvement orchestrator |

### Adding a New AI Type

1. Create `app/ai/your_ai.py` extending `BaseAI`
2. Implement `select_move()` and `evaluate_position()`
3. Register in `app/main.py` difficulty ladder
4. Add tests in `tests/test_your_ai.py`
5. Update documentation

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing!
