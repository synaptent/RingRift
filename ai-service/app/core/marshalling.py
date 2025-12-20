"""Marshalling/Serialization Utilities for RingRift AI Service.

Provides unified serialization patterns for:
- Dataclasses to/from dict/JSON
- NumPy arrays
- PyTorch tensors
- Custom types with registrable codecs

Usage:
    from app.core.marshalling import serialize, deserialize, Serializable

    # Direct serialization
    data = serialize(my_dataclass)
    obj = deserialize(data, MyDataclass)

    # Serializable mixin
    class MyConfig(Serializable):
        name: str
        value: int

    config = MyConfig(name="test", value=42)
    json_str = config.to_json()
    restored = MyConfig.from_json(json_str)
"""

from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    get_type_hints,
)
from uuid import UUID

__all__ = [
    "Codec",
    "Serializable",
    "SerializationError",
    "deserialize",
    "from_json",
    "register_codec",
    "serialize",
    "to_json",
]

T = TypeVar("T")


class SerializationError(Exception):
    """Error during serialization or deserialization."""


# =============================================================================
# Codec System
# =============================================================================

class Codec(ABC, Generic[T]):
    """Abstract codec for custom type serialization."""

    @property
    @abstractmethod
    def type_name(self) -> str:
        """Unique name for this type in serialized form."""

    @property
    @abstractmethod
    def python_type(self) -> type[T]:
        """The Python type this codec handles."""

    @abstractmethod
    def encode(self, value: T) -> Any:
        """Encode value to JSON-serializable form."""

    @abstractmethod
    def decode(self, data: Any) -> T:
        """Decode value from JSON-serializable form."""


class CodecRegistry:
    """Registry for type codecs."""

    def __init__(self):
        self._by_type: dict[type, Codec] = {}
        self._by_name: dict[str, Codec] = {}

    def register(self, codec: Codec) -> None:
        """Register a codec."""
        self._by_type[codec.python_type] = codec
        self._by_name[codec.type_name] = codec

    def get_by_type(self, type_: type) -> Codec | None:
        """Get codec for a Python type."""
        return self._by_type.get(type_)

    def get_by_name(self, name: str) -> Codec | None:
        """Get codec by type name."""
        return self._by_name.get(name)

    def has_type(self, type_: type) -> bool:
        """Check if type has a codec."""
        return type_ in self._by_type


# Global registry
_registry = CodecRegistry()


def register_codec(codec: Codec) -> None:
    """Register a custom codec globally.

    Args:
        codec: Codec instance to register

    Example:
        class MyTypeCodec(Codec[MyType]):
            @property
            def type_name(self) -> str:
                return "my_type"

            @property
            def python_type(self) -> Type[MyType]:
                return MyType

            def encode(self, value: MyType) -> Any:
                return value.to_dict()

            def decode(self, data: Any) -> MyType:
                return MyType.from_dict(data)

        register_codec(MyTypeCodec())
    """
    _registry.register(codec)


# =============================================================================
# Built-in Codecs
# =============================================================================

class DateTimeCodec(Codec[datetime]):
    """Codec for datetime objects."""

    @property
    def type_name(self) -> str:
        return "datetime"

    @property
    def python_type(self) -> type[datetime]:
        return datetime

    def encode(self, value: datetime) -> str:
        return value.isoformat()

    def decode(self, data: str) -> datetime:
        return datetime.fromisoformat(data)


class DateCodec(Codec[date]):
    """Codec for date objects."""

    @property
    def type_name(self) -> str:
        return "date"

    @property
    def python_type(self) -> type[date]:
        return date

    def encode(self, value: date) -> str:
        return value.isoformat()

    def decode(self, data: str) -> date:
        return date.fromisoformat(data)


class TimeDeltaCodec(Codec[timedelta]):
    """Codec for timedelta objects."""

    @property
    def type_name(self) -> str:
        return "timedelta"

    @property
    def python_type(self) -> type[timedelta]:
        return timedelta

    def encode(self, value: timedelta) -> float:
        return value.total_seconds()

    def decode(self, data: float) -> timedelta:
        return timedelta(seconds=data)


class UUIDCodec(Codec[UUID]):
    """Codec for UUID objects."""

    @property
    def type_name(self) -> str:
        return "uuid"

    @property
    def python_type(self) -> type[UUID]:
        return UUID

    def encode(self, value: UUID) -> str:
        return str(value)

    def decode(self, data: str) -> UUID:
        return UUID(data)


class PathCodec(Codec[Path]):
    """Codec for Path objects."""

    @property
    def type_name(self) -> str:
        return "path"

    @property
    def python_type(self) -> type[Path]:
        return Path

    def encode(self, value: Path) -> str:
        return str(value)

    def decode(self, data: str) -> Path:
        return Path(data)


class BytesCodec(Codec[bytes]):
    """Codec for bytes objects using base64."""

    @property
    def type_name(self) -> str:
        return "bytes"

    @property
    def python_type(self) -> type[bytes]:
        return bytes

    def encode(self, value: bytes) -> str:
        return b64encode(value).decode("ascii")

    def decode(self, data: str) -> bytes:
        return b64decode(data.encode("ascii"))


class SetCodec(Codec[set]):
    """Codec for set objects."""

    @property
    def type_name(self) -> str:
        return "set"

    @property
    def python_type(self) -> type[set]:
        return set

    def encode(self, value: set) -> list:
        return list(value)

    def decode(self, data: list) -> set:
        return set(data)


class FrozenSetCodec(Codec[frozenset]):
    """Codec for frozenset objects."""

    @property
    def type_name(self) -> str:
        return "frozenset"

    @property
    def python_type(self) -> type[frozenset]:
        return frozenset

    def encode(self, value: frozenset) -> list:
        return list(value)

    def decode(self, data: list) -> frozenset:
        return frozenset(data)


# Register built-in codecs
_registry.register(DateTimeCodec())
_registry.register(DateCodec())
_registry.register(TimeDeltaCodec())
_registry.register(UUIDCodec())
_registry.register(PathCodec())
_registry.register(BytesCodec())
_registry.register(SetCodec())
_registry.register(FrozenSetCodec())


# =============================================================================
# NumPy/Torch Codecs (registered if available)
# =============================================================================

try:
    import numpy as np

    class NumpyArrayCodec(Codec[np.ndarray]):
        """Codec for NumPy arrays."""

        @property
        def type_name(self) -> str:
            return "ndarray"

        @property
        def python_type(self) -> type[np.ndarray]:
            return np.ndarray

        def encode(self, value: np.ndarray) -> dict[str, Any]:
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": b64encode(value.tobytes()).decode("ascii"),
            }

        def decode(self, data: dict[str, Any]) -> np.ndarray:
            arr = np.frombuffer(
                b64decode(data["data"].encode("ascii")),
                dtype=np.dtype(data["dtype"]),
            )
            return arr.reshape(data["shape"])

    _registry.register(NumpyArrayCodec())
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch

    class TorchTensorCodec(Codec[torch.Tensor]):
        """Codec for PyTorch tensors."""

        @property
        def type_name(self) -> str:
            return "tensor"

        @property
        def python_type(self) -> type[torch.Tensor]:
            return torch.Tensor

        def encode(self, value: torch.Tensor) -> dict[str, Any]:
            np_arr = value.detach().cpu().numpy()
            return {
                "dtype": str(np_arr.dtype),
                "shape": list(np_arr.shape),
                "data": b64encode(np_arr.tobytes()).decode("ascii"),
            }

        def decode(self, data: dict[str, Any]) -> torch.Tensor:
            import numpy as np
            arr = np.frombuffer(
                b64decode(data["data"].encode("ascii")),
                dtype=np.dtype(data["dtype"]),
            )
            return torch.from_numpy(arr.reshape(data["shape"]))

    _registry.register(TorchTensorCodec())
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# =============================================================================
# Core Serialization Functions
# =============================================================================

def _serialize_value(value: Any, include_type: bool = False) -> Any:
    """Serialize a single value."""
    if value is None:
        return None

    # Primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Enum
    if isinstance(value, Enum):
        if include_type:
            return {"__type__": "enum", "__class__": f"{type(value).__module__}.{type(value).__name__}", "value": value.value}
        return value.value

    # Check registered codecs
    codec = _registry.get_by_type(type(value))
    if codec:
        if include_type:
            return {"__type__": codec.type_name, "value": codec.encode(value)}
        return codec.encode(value)

    # Dataclass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {}
        for field in dataclasses.fields(value):
            field_value = getattr(value, field.name)
            result[field.name] = _serialize_value(field_value, include_type)
        if include_type:
            return {"__type__": "dataclass", "__class__": f"{type(value).__module__}.{type(value).__name__}", "value": result}
        return result

    # Dict
    if isinstance(value, dict):
        return {
            k: _serialize_value(v, include_type)
            for k, v in value.items()
        }

    # List/Tuple
    if isinstance(value, (list, tuple)):
        result = [_serialize_value(v, include_type) for v in value]
        if isinstance(value, tuple) and include_type:
            return {"__type__": "tuple", "value": result}
        return result

    # Set/Frozenset
    if isinstance(value, (set, frozenset)):
        return list(value)

    # Fallback: try to_dict method
    if hasattr(value, "to_dict"):
        return value.to_dict()

    # Last resort: str
    return str(value)


def _deserialize_value(
    data: Any,
    target_type: type | None = None,
) -> Any:
    """Deserialize a single value."""
    if data is None:
        return None

    # Check for type hint
    if isinstance(data, dict) and "__type__" in data:
        type_name = data["__type__"]

        if type_name == "tuple":
            return tuple(_deserialize_value(v) for v in data["value"])

        if type_name == "enum":
            # Import the enum class and recreate
            module_class = data["__class__"]
            module_name, class_name = module_class.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            enum_class = getattr(module, class_name)
            return enum_class(data["value"])

        if type_name == "dataclass":
            module_class = data["__class__"]
            module_name, class_name = module_class.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            dc_class = getattr(module, class_name)
            return _deserialize_dataclass(data["value"], dc_class)

        codec = _registry.get_by_name(type_name)
        if codec:
            return codec.decode(data["value"])

    # Target type provided
    if target_type is not None:
        return _deserialize_to_type(data, target_type)

    # Primitives
    if isinstance(data, (str, int, float, bool)):
        return data

    # Dict
    if isinstance(data, dict):
        return {k: _deserialize_value(v) for k, v in data.items()}

    # List
    if isinstance(data, list):
        return [_deserialize_value(v) for v in data]

    return data


def _deserialize_to_type(data: Any, target_type: type[T]) -> T:
    """Deserialize data to a specific type."""
    # Handle None
    if data is None:
        return None  # type: ignore

    # Handle Optional
    origin = getattr(target_type, "__origin__", None)
    if origin is Union:
        args = target_type.__args__
        # Check for Optional (Union[X, None])
        if type(None) in args:
            if data is None:
                return None  # type: ignore
            # Try non-None types
            for arg in args:
                if arg is not type(None):
                    try:
                        return _deserialize_to_type(data, arg)
                    except Exception:
                        continue

    # Primitives
    if target_type in (str, int, float, bool):
        return target_type(data)

    # Enum
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(data)

    # Codec
    codec = _registry.get_by_type(target_type)
    if codec:
        return codec.decode(data)

    # Dataclass
    if dataclasses.is_dataclass(target_type):
        return _deserialize_dataclass(data, target_type)

    # List
    if origin is list:
        item_type = target_type.__args__[0] if hasattr(target_type, "__args__") else Any
        return [_deserialize_to_type(v, item_type) for v in data]

    # Dict
    if origin is dict:
        _key_type, value_type = target_type.__args__ if hasattr(target_type, "__args__") else (Any, Any)
        return {k: _deserialize_to_type(v, value_type) for k, v in data.items()}

    # Set
    if origin is set:
        item_type = target_type.__args__[0] if hasattr(target_type, "__args__") else Any
        return {_deserialize_to_type(v, item_type) for v in data}

    # Tuple
    if origin is tuple:
        if hasattr(target_type, "__args__"):
            return tuple(
                _deserialize_to_type(v, t)
                for v, t in zip(data, target_type.__args__, strict=False)
            )
        return tuple(data)

    # Fallback
    return data


def _deserialize_dataclass(data: dict[str, Any], cls: type[T]) -> T:
    """Deserialize a dict to a dataclass."""
    if not dataclasses.is_dataclass(cls):
        raise SerializationError(f"{cls} is not a dataclass")

    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name in data:
            field_type = hints.get(field.name, Any)
            kwargs[field.name] = _deserialize_to_type(data[field.name], field_type)

    return cls(**kwargs)


# =============================================================================
# Public API
# =============================================================================

def serialize(
    obj: Any,
    include_type_info: bool = False,
) -> Any:
    """Serialize an object to JSON-compatible form.

    Args:
        obj: Object to serialize
        include_type_info: Include type hints for reconstruction

    Returns:
        JSON-serializable data structure

    Example:
        data = serialize(my_dataclass)
        json_str = json.dumps(data)
    """
    return _serialize_value(obj, include_type_info)


def deserialize(
    data: Any,
    target_type: type[T] | None = None,
) -> T:
    """Deserialize data to an object.

    Args:
        data: JSON-compatible data
        target_type: Expected type (for dataclass reconstruction)

    Returns:
        Deserialized object

    Example:
        data = json.loads(json_str)
        obj = deserialize(data, MyDataclass)
    """
    return _deserialize_value(data, target_type)


def to_json(
    obj: Any,
    pretty: bool = False,
    include_type_info: bool = False,
) -> str:
    """Serialize object directly to JSON string.

    Args:
        obj: Object to serialize
        pretty: Pretty-print with indentation
        include_type_info: Include type hints

    Returns:
        JSON string
    """
    data = serialize(obj, include_type_info)
    if pretty:
        return json.dumps(data, indent=2, sort_keys=True)
    return json.dumps(data)


def from_json(
    json_str: str,
    target_type: type[T] | None = None,
) -> T:
    """Deserialize JSON string to object.

    Args:
        json_str: JSON string
        target_type: Expected type

    Returns:
        Deserialized object
    """
    data = json.loads(json_str)
    return deserialize(data, target_type)


# =============================================================================
# Serializable Mixin
# =============================================================================

class Serializable:
    """Mixin for serializable classes.

    Add this to dataclasses for convenient serialization methods.

    Example:
        @dataclass
        class Config(Serializable):
            name: str
            value: int

        config = Config(name="test", value=42)
        json_str = config.to_json()
        restored = Config.from_json(json_str)
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return serialize(self)

    def to_json(self, pretty: bool = False) -> str:
        """Serialize to JSON string."""
        return to_json(self, pretty=pretty)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize from dictionary."""
        return deserialize(data, cls)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Deserialize from JSON string."""
        return from_json(json_str, cls)
