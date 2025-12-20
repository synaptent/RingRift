"""Tests for scripts/lib/state_manager.py module.

Tests cover:
- StateManager for generic state persistence
- StatePersistence mixin for dataclasses
- load_json_state/save_json_state helper functions
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pytest

from scripts.lib.state_manager import (
    StateManager,
    StatePersistence,
    load_json_state,
    save_json_state,
)


@dataclass
class SimpleState:
    """Simple dataclass for testing."""
    count: int = 0
    name: str = "default"


@dataclass
class ComplexState:
    """Complex dataclass with nested structures."""
    values: list[int] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    last_run: float = 0.0


@dataclass
class StateWithMethods(StatePersistence):
    """State class using StatePersistence mixin."""
    count: int = 0
    active: bool = True


class StateWithCustomSerialization:
    """Class with custom to_dict/from_dict methods."""

    def __init__(self, value: int = 0, label: str = ""):
        self.value = value
        self.label = label

    def to_dict(self) -> dict:
        return {"value": self.value, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict) -> "StateWithCustomSerialization":
        return cls(value=data.get("value", 0), label=data.get("label", ""))


class TestStateManager:
    """Tests for StateManager class."""

    def test_load_nonexistent_returns_default(self, tmp_path):
        """Test loading from nonexistent file returns default."""
        state_file = tmp_path / "nonexistent.json"
        manager = StateManager(state_file, SimpleState)

        state = manager.load()

        assert isinstance(state, SimpleState)
        assert state.count == 0
        assert state.name == "default"

    def test_save_and_load_dataclass(self, tmp_path):
        """Test saving and loading a dataclass."""
        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, SimpleState)

        # Save state
        state = SimpleState(count=42, name="test")
        assert manager.save(state)

        # Load and verify
        loaded = manager.load()
        assert loaded.count == 42
        assert loaded.name == "test"

    def test_save_and_load_complex_dataclass(self, tmp_path):
        """Test saving and loading complex nested structures."""
        state_file = tmp_path / "complex.json"
        manager = StateManager(state_file, ComplexState)

        state = ComplexState(
            values=[1, 2, 3],
            metadata={"key": "value"},
            last_run=12345.67,
        )
        manager.save(state)

        loaded = manager.load()
        assert loaded.values == [1, 2, 3]
        assert loaded.metadata == {"key": "value"}
        assert loaded.last_run == 12345.67

    def test_save_and_load_custom_serialization(self, tmp_path):
        """Test saving and loading class with to_dict/from_dict."""
        state_file = tmp_path / "custom.json"
        manager = StateManager(state_file, StateWithCustomSerialization)

        state = StateWithCustomSerialization(value=100, label="test")
        manager.save(state)

        loaded = manager.load()
        assert loaded.value == 100
        assert loaded.label == "test"

    def test_save_and_load_dict(self, tmp_path):
        """Test saving and loading plain dictionaries."""
        state_file = tmp_path / "dict.json"
        manager = StateManager(state_file, dict)

        state = {"key": "value", "count": 42}
        manager.save(state)

        loaded = manager.load()
        assert loaded == {"key": "value", "count": 42}

    def test_load_invalid_json_returns_default(self, tmp_path):
        """Test loading invalid JSON returns default."""
        state_file = tmp_path / "invalid.json"
        state_file.write_text("not valid json{")

        manager = StateManager(state_file, SimpleState)
        state = manager.load()

        assert isinstance(state, SimpleState)
        assert state.count == 0

    def test_load_incompatible_data_returns_default(self, tmp_path):
        """Test loading incompatible data returns default."""
        state_file = tmp_path / "incompatible.json"
        state_file.write_text('{"unknown_field": 123}')

        manager = StateManager(state_file, SimpleState)
        state = manager.load()

        # Should still load with defaults for missing fields
        assert isinstance(state, SimpleState)

    def test_update(self, tmp_path):
        """Test update function."""
        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, SimpleState)

        # Initial save
        manager.save(SimpleState(count=10))

        # Update
        def increment(state):
            state.count += 1
            return state

        result = manager.update(increment)

        assert result.count == 11

        # Verify it was saved
        loaded = manager.load()
        assert loaded.count == 11

    def test_exists(self, tmp_path):
        """Test exists method."""
        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, SimpleState)

        assert not manager.exists()

        manager.save(SimpleState())

        assert manager.exists()

    def test_delete(self, tmp_path):
        """Test delete method."""
        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, SimpleState)

        # Delete nonexistent returns False
        assert not manager.delete()

        # Save and delete
        manager.save(SimpleState())
        assert manager.exists()

        assert manager.delete()
        assert not manager.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Test save creates parent directories."""
        state_file = tmp_path / "nested" / "deep" / "state.json"
        manager = StateManager(state_file, SimpleState)

        assert manager.save(SimpleState(count=1))
        assert state_file.exists()

    def test_custom_default_factory(self, tmp_path):
        """Test custom default factory."""
        state_file = tmp_path / "state.json"

        def custom_default():
            return SimpleState(count=999, name="custom")

        manager = StateManager(state_file, SimpleState, default_factory=custom_default)
        state = manager.load()

        assert state.count == 999
        assert state.name == "custom"


class TestStatePersistence:
    """Tests for StatePersistence mixin."""

    def test_load_from_file_nonexistent(self, tmp_path):
        """Test loading from nonexistent file."""
        state_file = tmp_path / "nonexistent.json"
        state = StateWithMethods.load_from_file(state_file)

        assert isinstance(state, StateWithMethods)
        assert state.count == 0
        assert state.active is True

    def test_save_and_load(self, tmp_path):
        """Test saving and loading with mixin."""
        state_file = tmp_path / "state.json"

        state = StateWithMethods(count=42, active=False)
        assert state.save_to_file(state_file)

        loaded = StateWithMethods.load_from_file(state_file)
        assert loaded.count == 42
        assert loaded.active is False

    def test_creates_parent_directories(self, tmp_path):
        """Test save creates parent directories."""
        state_file = tmp_path / "nested" / "state.json"

        state = StateWithMethods(count=1)
        assert state.save_to_file(state_file)
        assert state_file.exists()


class TestLoadJsonState:
    """Tests for load_json_state function."""

    def test_load_nonexistent_returns_default(self, tmp_path):
        """Test loading nonexistent file returns default."""
        state_file = tmp_path / "nonexistent.json"

        result = load_json_state(state_file)
        assert result == {}

        result = load_json_state(state_file, default={"key": "value"})
        assert result == {"key": "value"}

    def test_load_existing(self, tmp_path):
        """Test loading existing file."""
        state_file = tmp_path / "state.json"
        state_file.write_text('{"count": 42}')

        result = load_json_state(state_file)
        assert result == {"count": 42}

    def test_load_invalid_returns_default(self, tmp_path):
        """Test loading invalid JSON returns default."""
        state_file = tmp_path / "invalid.json"
        state_file.write_text("not json")

        result = load_json_state(state_file, default={"fallback": True})
        assert result == {"fallback": True}

    def test_default_is_copied(self, tmp_path):
        """Test that default is copied, not shared."""
        state_file = tmp_path / "nonexistent.json"
        default = {"count": 0}

        result1 = load_json_state(state_file, default=default)
        result1["count"] = 1

        result2 = load_json_state(state_file, default=default)
        assert result2["count"] == 0  # Original default unchanged


class TestSaveJsonState:
    """Tests for save_json_state function."""

    def test_save_creates_file(self, tmp_path):
        """Test saving creates file."""
        state_file = tmp_path / "state.json"

        assert save_json_state(state_file, {"key": "value"})
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)
        assert data == {"key": "value"}

    def test_save_creates_directories(self, tmp_path):
        """Test save creates parent directories."""
        state_file = tmp_path / "nested" / "deep" / "state.json"

        assert save_json_state(state_file, {"key": "value"})
        assert state_file.exists()

    def test_save_handles_non_serializable(self, tmp_path):
        """Test save handles non-serializable types using default=str."""
        state_file = tmp_path / "state.json"

        # Path objects aren't JSON serializable by default
        assert save_json_state(state_file, {"path": Path("/some/path")})
        assert state_file.exists()


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_save_is_atomic(self, tmp_path):
        """Test that save uses atomic write pattern."""
        state_file = tmp_path / "state.json"
        manager = StateManager(state_file, SimpleState)

        # Save initial state
        manager.save(SimpleState(count=1))

        # No temp file should remain
        temp_file = state_file.with_suffix(".tmp")
        assert not temp_file.exists()

        # State file should exist
        assert state_file.exists()
