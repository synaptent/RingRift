# CLI Module

Command-line interface utilities for RingRift scripts.

## Overview

This module provides standardized patterns for CLI scripts:

- Argument parsing helpers
- Script runner with common setup
- Output formatting and tables
- Progress display

## Key Components

### `ScriptRunner` - Script Framework

```python
from app.cli import ScriptRunner

def main():
    runner = ScriptRunner(
        name="my_script",
        description="Processes game data",
    )

    # Add arguments
    runner.add_argument("--config", required=True, help="Config file path")
    runner.add_argument("--verbose", action="store_true")

    # Parse and run
    args = runner.parse_args()

    with runner.run_context():
        # Automatic logging setup, timing, error handling
        process_data(args.config)

if __name__ == "__main__":
    main()
```

### `setup_script` - Simple Setup

```python
from app.cli import setup_script

# Quick one-liner setup
args, logger = setup_script(
    name="simple_script",
    description="Does something simple",
    args=[
        ("--input", {"required": True}),
        ("--output", {"default": "output.txt"}),
    ],
)

logger.info(f"Processing {args.input}")
```

### Output Utilities

```python
from app.cli import (
    print_status,
    print_success,
    print_error,
    print_table,
    print_progress,
)

# Status messages with colors
print_status("Processing...", status="info")
print_success("Completed successfully!")
print_error("Something went wrong")

# Tables
print_table(
    headers=["Model", "Accuracy", "ELO"],
    rows=[
        ["hex8_2p_v1", "72.3%", "1050"],
        ["hex8_2p_v2", "76.1%", "1120"],
    ],
)

# Progress updates
for i, item in enumerate(items):
    print_progress(f"Processing item {i+1}/{len(items)}")
```

### `ProgressBar` - Progress Tracking

```python
from app.cli import ProgressBar

with ProgressBar(total=1000, desc="Training") as pbar:
    for batch in batches:
        train_batch(batch)
        pbar.update(batch_size)
        pbar.set_postfix(loss=current_loss)
```

## Common Arguments

The `add_common_args` helper adds frequently-used options:

```python
from app.cli import add_common_args

parser = argparse.ArgumentParser()
add_common_args(parser)  # Adds --verbose, --dry-run, --config

# Also available: add_board_args, add_training_args
```

## Standard Script Template

```python
#!/usr/bin/env python3
"""Script description here."""

from app.cli import ScriptRunner, print_status, print_success

def main():
    runner = ScriptRunner(
        name="my_script",
        description=__doc__,
    )
    runner.add_argument("--board-type", choices=["hex8", "square8"])
    runner.add_argument("--num-games", type=int, default=100)

    args = runner.parse_args()

    with runner.run_context():
        print_status(f"Running with {args.num_games} games")

        # ... do work ...

        print_success("Done!")

if __name__ == "__main__":
    main()
```

## Features

### Automatic Setup

- Logging configuration
- Signal handling (Ctrl+C)
- Timing and duration reporting
- Error handling with tracebacks

### Output Formatting

- Colored terminal output
- Table formatting
- Progress bars with ETA
- Status indicators

### Integration

- Works with `app.core.shutdown` for graceful shutdown
- Integrates with `app.core.logging_config`
