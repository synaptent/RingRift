#!/usr/bin/env python3
"""Generate event payload schemas from codebase analysis.

Sprint 14 (January 2026): Auto-generate payload schemas for undocumented events
by scanning emit() calls across the codebase.

Usage:
    python scripts/generate_event_schemas.py --output docs/EVENT_PAYLOAD_SCHEMAS_AUTO.md
    python scripts/generate_event_schemas.py --output docs/event_schemas.yaml --format yaml
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PayloadField:
    """A field in an event payload."""
    name: str
    type_hint: str = "Any"
    required: bool = True
    description: str = ""
    example: str = ""


@dataclass
class EventSchema:
    """Schema for a single event type."""
    name: str
    value: str  # The string value (e.g., "training_completed")
    category: str = "uncategorized"
    description: str = ""
    emitters: list[str] = field(default_factory=list)
    subscribers: list[str] = field(default_factory=list)
    payload_fields: list[PayloadField] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)


def get_all_event_types() -> dict[str, str]:
    """Get all DataEventType members as name -> value mapping."""
    try:
        from app.distributed.data_events import DataEventType
        return {member.name: member.value for member in DataEventType}
    except ImportError:
        from app.coordination.event_router import DataEventType
        return {member.name: member.value for member in DataEventType}


def categorize_event(event_name: str) -> str:
    """Categorize an event by its name pattern."""
    name = event_name.upper()

    # Training pipeline
    if any(kw in name for kw in ["TRAINING", "TRAIN"]):
        return "training"
    if any(kw in name for kw in ["SELFPLAY", "GAME"]):
        return "selfplay"
    if any(kw in name for kw in ["EVALUATION", "GAUNTLET", "ELO"]):
        return "evaluation"
    if any(kw in name for kw in ["MODEL", "CHECKPOINT"]):
        return "model"
    if any(kw in name for kw in ["NPZ", "EXPORT", "DATA"]):
        return "data"
    if any(kw in name for kw in ["SYNC", "DISTRIBUTION", "REPLICATION"]):
        return "sync"

    # Infrastructure
    if any(kw in name for kw in ["CURRICULUM"]):
        return "curriculum"
    if any(kw in name for kw in ["HEALTH", "ERROR", "FAIL", "RECOVER"]):
        return "health"
    if any(kw in name for kw in ["NODE", "CLUSTER", "PEER", "LEADER", "QUORUM"]):
        return "p2p"
    if any(kw in name for kw in ["CIRCUIT", "BACKPRESSURE", "THROTTLE"]):
        return "resilience"
    if any(kw in name for kw in ["PROGRESS", "STALL", "REGRESSION"]):
        return "monitoring"
    if any(kw in name for kw in ["QUALITY"]):
        return "quality"
    if any(kw in name for kw in ["DAEMON", "COORDINATOR"]):
        return "daemon"

    return "other"


def find_emit_calls(file_path: Path) -> list[tuple[str, dict[str, Any], int]]:
    """Find emit() calls in a Python file and extract event type + payload.

    Returns list of (event_name, payload_dict, line_number).
    """
    results = []
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return results

    # Pattern 1: emit_event(DataEventType.NAME, payload) or emit_data_event(...)
    # Pattern 2: subscribe(DataEventType.NAME, handler)
    # Pattern 3: emit("event_name", payload)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check for emit function calls
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in ("emit_event", "emit_data_event", "emit", "_emit_event"):
                if len(node.args) >= 1:
                    event_arg = node.args[0]
                    event_name = _extract_event_name(event_arg)
                    if event_name:
                        payload = _extract_payload_structure(node.args[1] if len(node.args) > 1 else None)
                        results.append((event_name, payload, node.lineno))

    return results


def _extract_event_name(node: ast.AST) -> str | None:
    """Extract event name from AST node."""
    if isinstance(node, ast.Attribute):
        # DataEventType.NAME
        if isinstance(node.value, ast.Name) and node.value.id == "DataEventType":
            return node.attr
        if isinstance(node.value, ast.Attribute):
            return node.attr
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        # String literal "event_name"
        return node.value.upper().replace("-", "_")
    return None


def _extract_payload_structure(node: ast.AST | None) -> dict[str, str]:
    """Extract payload field names and guess types from AST dict node."""
    if node is None:
        return {}

    fields = {}
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                field_name = key.value
                field_type = _guess_type(value)
                fields[field_name] = field_type

    return fields


def _guess_type(node: ast.AST) -> str:
    """Guess the type of a value from AST."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return "string"
        elif isinstance(node.value, bool):
            return "boolean"
        elif isinstance(node.value, int):
            return "integer"
        elif isinstance(node.value, float):
            return "number"
    elif isinstance(node, ast.List):
        return "array"
    elif isinstance(node, ast.Dict):
        return "object"
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "time":
                return "timestamp"
            if node.func.attr == "isoformat":
                return "datetime"
        if isinstance(node.func, ast.Name):
            if node.func.id in ("time", "time.time"):
                return "timestamp"
    elif isinstance(node, ast.Name):
        name = node.id.lower()
        if "path" in name:
            return "string (path)"
        if "count" in name or "num" in name:
            return "integer"
        if "time" in name or "timestamp" in name:
            return "timestamp"
        if "elo" in name:
            return "number"
    return "any"


def scan_codebase_for_events(root: Path) -> dict[str, EventSchema]:
    """Scan entire codebase for event emissions and subscriptions."""
    event_types = get_all_event_types()
    schemas: dict[str, EventSchema] = {}

    # Initialize schemas for all known events
    for name, value in event_types.items():
        schemas[name] = EventSchema(
            name=name,
            value=value,
            category=categorize_event(name),
        )

    # Scan Python files
    python_files = list(root.rglob("*.py"))
    python_files = [f for f in python_files if "archive" not in str(f) and "__pycache__" not in str(f)]

    logger.info(f"Scanning {len(python_files)} Python files...")

    payload_observations: dict[str, list[dict[str, str]]] = defaultdict(list)
    emitter_files: dict[str, set[str]] = defaultdict(set)

    for file_path in python_files:
        relative_path = str(file_path.relative_to(root))
        emit_calls = find_emit_calls(file_path)

        for event_name, payload, line_no in emit_calls:
            # Normalize event name
            event_key = event_name.upper().replace("-", "_")

            if event_key in schemas:
                emitter_files[event_key].add(f"{relative_path}:{line_no}")
                if payload:
                    payload_observations[event_key].append(payload)

    # Merge payload observations
    for event_name, observations in payload_observations.items():
        if event_name in schemas:
            merged_fields = {}
            for obs in observations:
                for field_name, field_type in obs.items():
                    if field_name not in merged_fields:
                        merged_fields[field_name] = field_type
                    # Keep more specific type
                    elif field_type != "any" and merged_fields[field_name] == "any":
                        merged_fields[field_name] = field_type

            schemas[event_name].payload_fields = [
                PayloadField(name=name, type_hint=type_hint)
                for name, type_hint in sorted(merged_fields.items())
            ]

    # Add emitter info
    for event_name, files in emitter_files.items():
        if event_name in schemas:
            schemas[event_name].emitters = sorted(files)[:5]  # Top 5 emitters

    return schemas


def generate_markdown(schemas: dict[str, EventSchema], documented: set[str]) -> str:
    """Generate markdown documentation for event schemas."""
    lines = [
        "# Event Payload Schemas (Auto-Generated)",
        "",
        f"Generated: January 2026 (Sprint 14)",
        "",
        f"Total events: {len(schemas)}",
        f"Previously documented: {len(documented)}",
        f"Newly documented: {len(schemas) - len(documented)}",
        "",
        "---",
        "",
    ]

    # Group by category
    by_category: dict[str, list[EventSchema]] = defaultdict(list)
    for schema in schemas.values():
        by_category[schema.category].append(schema)

    # Category order
    category_order = [
        "training", "selfplay", "evaluation", "model", "data", "sync",
        "curriculum", "quality", "health", "p2p", "resilience",
        "monitoring", "daemon", "other"
    ]

    for category in category_order:
        if category not in by_category:
            continue

        events = sorted(by_category[category], key=lambda s: s.name)
        lines.append(f"## {category.title()} Events ({len(events)})")
        lines.append("")

        for schema in events:
            is_new = schema.name not in documented
            marker = " 🆕" if is_new else ""

            lines.append(f"### {schema.name}{marker}")
            lines.append("")
            lines.append(f"**Value**: `{schema.value}`")
            lines.append("")

            if schema.payload_fields:
                lines.append("**Payload Fields**:")
                lines.append("")
                lines.append("| Field | Type |")
                lines.append("|-------|------|")
                for field in schema.payload_fields:
                    lines.append(f"| `{field.name}` | {field.type_hint} |")
                lines.append("")
            else:
                lines.append("**Payload**: (no fields detected)")
                lines.append("")

            if schema.emitters:
                lines.append("**Emitters**:")
                for emitter in schema.emitters[:3]:
                    lines.append(f"- `{emitter}`")
                lines.append("")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def generate_yaml(schemas: dict[str, EventSchema]) -> str:
    """Generate YAML schema file."""
    import yaml

    output = {
        "version": "1.0",
        "generated": "January 2026 (Sprint 14)",
        "total_events": len(schemas),
        "events": {}
    }

    for name, schema in sorted(schemas.items()):
        event_data = {
            "value": schema.value,
            "category": schema.category,
        }

        if schema.payload_fields:
            event_data["payload"] = {
                field.name: {"type": field.type_hint}
                for field in schema.payload_fields
            }

        if schema.emitters:
            event_data["emitters"] = schema.emitters[:3]

        output["events"][name] = event_data

    return yaml.dump(output, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_documented_events() -> set[str]:
    """Get set of already-documented event names from EVENT_PAYLOAD_SCHEMAS.md."""
    docs_path = ROOT / "docs" / "EVENT_PAYLOAD_SCHEMAS.md"
    if not docs_path.exists():
        return set()

    content = docs_path.read_text()
    pattern = re.compile(r'^#{2,3}\s+([A-Z][A-Z0-9_]+)', re.MULTILINE)
    return set(pattern.findall(content))


def main():
    parser = argparse.ArgumentParser(description="Generate event payload schemas")
    parser.add_argument(
        "--output", "-o",
        default="docs/EVENT_PAYLOAD_SCHEMAS_AUTO.md",
        help="Output file path"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "yaml", "json"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get already-documented events
    documented = get_documented_events()
    logger.info(f"Found {len(documented)} previously documented events")

    # Scan codebase
    schemas = scan_codebase_for_events(ROOT)
    logger.info(f"Found {len(schemas)} total events")

    # Count events with payload fields
    with_payload = sum(1 for s in schemas.values() if s.payload_fields)
    logger.info(f"Events with detected payload fields: {with_payload}")

    # Generate output
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "markdown":
        content = generate_markdown(schemas, documented)
    elif args.format == "yaml":
        content = generate_yaml(schemas)
    else:
        content = json.dumps(
            {name: {
                "value": s.value,
                "category": s.category,
                "fields": [f.name for f in s.payload_fields]
            } for name, s in schemas.items()},
            indent=2
        )

    output_path.write_text(content)
    logger.info(f"Written to {output_path}")

    # Summary
    new_events = len(schemas) - len(documented)
    print(f"\nSummary:")
    print(f"  Total events: {len(schemas)}")
    print(f"  Previously documented: {len(documented)}")
    print(f"  Newly documented: {new_events}")
    print(f"  Events with payload: {with_payload}")


if __name__ == "__main__":
    main()
