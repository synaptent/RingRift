"""JSON serialization utilities.

This module provides enhanced JSON encoding with automatic handling of
common Python types that aren't natively JSON serializable.

Usage:
    from app.utils.json_utils import dumps, JSONEncoder

    # Serialize with automatic datetime/Path/Enum handling
    data = {"timestamp": datetime.now(), "path": Path("/tmp")}
    json_str = dumps(data)

    # Use with json.dump()
    with open("file.json", "w") as f:
        json.dump(data, f, cls=JSONEncoder)
"""

from __future__ import annotations

import json
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID

__all__ = [
    "JSONEncoder",
    "dumps",
    "dump",
    "pretty_dumps",
    "json_default",
    "load_json",
    "save_json",
]


class JSONEncoder(json.JSONEncoder):
    """Extended JSON encoder that handles common Python types.

    Handles:
    - datetime, date -> ISO format strings
    - timedelta -> total seconds (float)
    - Path -> string path
    - Enum -> value
    - UUID -> string
    - sets -> lists
    - bytes -> base64 string
    - objects with to_dict() or dict() methods
    """

    def default(self, obj: Any) -> Any:
        # Datetime types
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()

        # Path
        if isinstance(obj, Path):
            return str(obj)

        # Enum
        if isinstance(obj, Enum):
            return obj.value

        # UUID
        if isinstance(obj, UUID):
            return str(obj)

        # Sets
        if isinstance(obj, (set, frozenset)):
            return list(obj)

        # Bytes
        if isinstance(obj, bytes):
            import base64
            return base64.b64encode(obj).decode('ascii')

        # Objects with serialization methods
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'model_dump'):  # Pydantic v2
            return obj.model_dump()
        if hasattr(obj, 'dict'):  # Pydantic v1
            return obj.dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__

        return super().default(obj)


def dumps(
    obj: Any,
    *,
    indent: Optional[int] = None,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    **kwargs: Any,
) -> str:
    """Serialize obj to a JSON string with enhanced type handling.

    This is a drop-in replacement for json.dumps() that automatically
    handles datetime, Path, Enum, and other common types.

    Args:
        obj: Object to serialize
        indent: Indentation level for pretty-printing
        sort_keys: Whether to sort dictionary keys
        ensure_ascii: Whether to escape non-ASCII characters
        **kwargs: Additional arguments passed to json.dumps()

    Returns:
        JSON string representation
    """
    return json.dumps(
        obj,
        cls=JSONEncoder,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        **kwargs,
    )


def dump(
    obj: Any,
    fp: Any,
    *,
    indent: Optional[int] = None,
    sort_keys: bool = False,
    ensure_ascii: bool = False,
    **kwargs: Any,
) -> None:
    """Serialize obj to a JSON file with enhanced type handling.

    This is a drop-in replacement for json.dump() that automatically
    handles datetime, Path, Enum, and other common types.

    Args:
        obj: Object to serialize
        fp: File-like object to write to
        indent: Indentation level for pretty-printing
        sort_keys: Whether to sort dictionary keys
        ensure_ascii: Whether to escape non-ASCII characters
        **kwargs: Additional arguments passed to json.dump()
    """
    json.dump(
        obj,
        fp,
        cls=JSONEncoder,
        indent=indent,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        **kwargs,
    )


def pretty_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize obj to a pretty-printed JSON string.

    Convenience wrapper for dumps() with indent=2.
    """
    return dumps(obj, indent=2, **kwargs)


def json_default(obj: Any) -> Any:
    """Default function for json.dumps() that handles common types.

    Use this with json.dumps(data, default=json_default) when you
    don't want to use the full JSONEncoder class.

    Args:
        obj: Object that isn't directly JSON serializable

    Returns:
        JSON-serializable representation

    Raises:
        TypeError: If object cannot be serialized
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('ascii')
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'dict'):
        return obj.dict()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_json(path: Union[str, Path], default: Any = None) -> Any:
    """Load JSON from a file path.

    Args:
        path: Path to the JSON file
        default: Value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data, or default if file not found/invalid
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    atomic: bool = True,
    **kwargs: Any,
) -> None:
    """Save data as JSON to a file path.

    Args:
        path: Path to write the JSON file
        data: Data to serialize
        indent: Indentation level for pretty-printing (default 2)
        atomic: If True, write to temp file first then rename (safer)
        **kwargs: Additional arguments passed to dump()
    """
    import tempfile
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        # Write to temp file first, then rename (atomic on POSIX)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                dump(data, f, indent=indent, **kwargs)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    else:
        with path.open("w", encoding="utf-8") as f:
            dump(data, f, indent=indent, **kwargs)
