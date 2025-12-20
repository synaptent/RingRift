"""Generic state persistence utilities for daemon scripts.

Provides utilities for saving and loading persistent state to JSON files,
with support for dataclasses and custom serialization.

Usage:
    from scripts.lib.state_manager import StateManager, StatePersistence
    from dataclasses import dataclass, asdict

    @dataclass
    class MyState:
        last_run: float = 0.0
        count: int = 0

    # Simple usage with StateManager
    manager = StateManager(Path("state.json"), MyState)
    state = manager.load()
    state.count += 1
    manager.save(state)

    # Or use the mixin for dataclasses
    @dataclass
    class MyState(StatePersistence):
        last_run: float = 0.0
        count: int = 0

    state = MyState.load_from_file(Path("state.json"))
    state.count += 1
    state.save_to_file(Path("state.json"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union
from collections.abc import Callable

from scripts.lib.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class StateManager(Generic[T]):
    """Generic state manager for loading/saving state to JSON files.

    Supports:
    - Dataclasses (using asdict for serialization)
    - Classes with to_dict/from_dict methods
    - Plain dictionaries

    Example:
        @dataclass
        class DaemonState:
            last_run: float = 0.0
            processed_count: int = 0

        manager = StateManager(Path("daemon_state.json"), DaemonState)
        state = manager.load()
        state.processed_count += 1
        manager.save(state)
    """

    def __init__(
        self,
        state_file: Union[str, Path],
        state_class: type[T],
        default_factory: Callable[[], T] | None = None,
    ):
        """Initialize state manager.

        Args:
            state_file: Path to the JSON state file
            state_class: The class to use for state (dataclass or class with to_dict/from_dict)
            default_factory: Optional factory function for creating default state
        """
        self.state_file = Path(state_file)
        self.state_class = state_class
        self.default_factory = default_factory or state_class

    def load(self) -> T:
        """Load state from file, returning default if not found or invalid.

        Returns:
            The loaded state or a new default state
        """
        if not self.state_file.exists():
            return self.default_factory()

        try:
            with open(self.state_file) as f:
                data = json.load(f)

            # Handle classes with from_dict method
            if hasattr(self.state_class, "from_dict"):
                return self.state_class.from_dict(data)

            # Handle dataclasses
            if is_dataclass(self.state_class):
                return self.state_class(**data)

            # Handle dict type
            if self.state_class is dict:
                return data

            # Fallback: try direct construction
            return self.state_class(**data)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in state file {self.state_file}: {e}")
            return self.default_factory()
        except TypeError as e:
            logger.warning(f"Failed to construct state from {self.state_file}: {e}")
            return self.default_factory()
        except Exception as e:
            logger.warning(f"Failed to load state from {self.state_file}: {e}")
            return self.default_factory()

    def save(self, state: T) -> bool:
        """Save state to file.

        Args:
            state: The state object to save

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Serialize state
            if hasattr(state, "to_dict"):
                data = state.to_dict()
            elif is_dataclass(state):
                data = asdict(state)
            elif isinstance(state, dict):
                data = state
            else:
                raise TypeError(f"Cannot serialize state of type {type(state)}")

            # Write atomically (write to temp, then rename)
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            temp_file.replace(self.state_file)
            return True

        except Exception as e:
            logger.error(f"Failed to save state to {self.state_file}: {e}")
            return False

    def update(self, update_fn: Callable[[T], T]) -> T:
        """Load state, apply update function, and save.

        Args:
            update_fn: Function that takes current state and returns updated state

        Returns:
            The updated state
        """
        state = self.load()
        state = update_fn(state)
        self.save(state)
        return state

    def exists(self) -> bool:
        """Check if state file exists."""
        return self.state_file.exists()

    def delete(self) -> bool:
        """Delete state file if it exists.

        Returns:
            True if file was deleted, False if it didn't exist
        """
        if self.state_file.exists():
            self.state_file.unlink()
            return True
        return False


class StatePersistence:
    """Mixin class for dataclasses that need state persistence.

    Add this as a base class to enable load_from_file/save_to_file methods.

    Example:
        @dataclass
        class MyState(StatePersistence):
            count: int = 0
            last_update: float = 0.0

        state = MyState.load_from_file(Path("state.json"))
        state.count += 1
        state.save_to_file(Path("state.json"))
    """

    @classmethod
    def load_from_file(cls, path: Union[str, Path]) -> "StatePersistence":
        """Load state from JSON file.

        Args:
            path: Path to the state file

        Returns:
            Instance of the class, or default if file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)

            if hasattr(cls, "from_dict"):
                return cls.from_dict(data)

            return cls(**data)

        except Exception as e:
            logger.warning(f"Failed to load state from {path}: {e}")
            return cls()

    def save_to_file(self, path: Union[str, Path]) -> bool:
        """Save state to JSON file.

        Args:
            path: Path to the state file

        Returns:
            True if save succeeded
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if hasattr(self, "to_dict"):
                data = self.to_dict()
            elif is_dataclass(self):
                data = asdict(self)
            else:
                raise TypeError(f"Cannot serialize {type(self)}")

            temp_file = path.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            temp_file.replace(path)
            return True

        except Exception as e:
            logger.error(f"Failed to save state to {path}: {e}")
            return False


def load_json_state(
    path: Union[str, Path],
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load state from a JSON file, returning default if not found.

    Simple function for scripts that use plain dictionaries.

    Args:
        path: Path to the JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        The loaded dictionary or default
    """
    path = Path(path)
    if default is None:
        default = {}

    if not path.exists():
        return default.copy()

    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load state from {path}: {e}")
        return default.copy()


def save_json_state(
    path: Union[str, Path],
    state: dict[str, Any],
) -> bool:
    """Save state dictionary to a JSON file.

    Args:
        path: Path to the JSON file
        state: Dictionary to save

    Returns:
        True if save succeeded
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        temp_file = path.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        temp_file.replace(path)
        return True

    except Exception as e:
        logger.error(f"Failed to save state to {path}: {e}")
        return False


__all__ = [
    "StateManager",
    "StatePersistence",
    "load_json_state",
    "save_json_state",
]
