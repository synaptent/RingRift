#!/usr/bin/env python3
"""Generate event system reference documentation.

December 2025: Auto-generates documentation showing all events, their emitters,
and subscribers by analyzing the codebase.

Usage:
    python scripts/generate_event_reference.py
    python scripts/generate_event_reference.py --output docs/EVENT_REFERENCE_AUTO.md
    python scripts/generate_event_reference.py --json > event_reference.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class EventInfo:
    """Information about an event type."""

    name: str
    emitters: list[tuple[str, int]] = field(default_factory=list)  # (file, line)
    subscribers: list[tuple[str, int, str]] = field(default_factory=list)  # (file, line, handler_name)
    category: str = "unknown"

    @property
    def is_orphan(self) -> bool:
        """Check if event has no subscribers."""
        return len(self.emitters) > 0 and len(self.subscribers) == 0

    @property
    def is_dead(self) -> bool:
        """Check if event has no emitters."""
        return len(self.emitters) == 0 and len(self.subscribers) > 0


# Event categories by prefix/pattern
EVENT_CATEGORIES = {
    "TRAINING_": "Training",
    "EVALUATION_": "Evaluation",
    "SELFPLAY_": "Selfplay",
    "SYNC_": "Sync",
    "DATA_": "Data",
    "MODEL_": "Model",
    "NODE_": "Node",
    "CLUSTER_": "Cluster",
    "HEALTH_": "Health",
    "DAEMON_": "Daemon",
    "QUALITY_": "Quality",
    "CURRICULUM_": "Curriculum",
    "PROMOTION_": "Promotion",
    "REGRESSION_": "Regression",
    "WORK_": "Work Queue",
    "TASK_": "Task",
    "HOST_": "Host",
    "LEADER_": "Leadership",
    "P2P_": "P2P",
    "BACKPRESSURE_": "Backpressure",
    "REPAIR_": "Repair",
    "RESOURCE_": "Resource",
    "IDLE_": "Resource",
    "CHECKPOINT_": "Checkpoint",
    "ELO_": "Elo",
    "PBT_": "PBT",
    "NAS_": "NAS",
    "PER_": "PER",
}


def categorize_event(event_name: str) -> str:
    """Determine event category from name."""
    for prefix, category in EVENT_CATEGORIES.items():
        if event_name.startswith(prefix):
            return category
    return "Other"


class EventEmitterVisitor(ast.NodeVisitor):
    """Find event emission points in code."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.emitters: list[tuple[str, int]] = []  # (event_name, line)

    def visit_Call(self, node: ast.Call) -> None:
        # Look for emit/publish/emit_sync patterns
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ("emit", "emit_sync", "publish", "emit_event", "safe_emit_event"):
                event_name = self._extract_event_name(node.args)
                if event_name:
                    self.emitters.append((event_name, node.lineno))

        # Look for emit_* function calls like emit_training_complete()
        if isinstance(node.func, ast.Name):
            if node.func.id.startswith("emit_"):
                # Convert emit_training_complete to TRAINING_COMPLETE
                event_name = node.func.id[5:].upper()
                self.emitters.append((event_name, node.lineno))

        self.generic_visit(node)

    def _extract_event_name(self, args: list[ast.expr]) -> str | None:
        if not args:
            return None
        first_arg = args[0]

        # String literal
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value

        # Attribute access like DataEventType.TRAINING_COMPLETED
        if isinstance(first_arg, ast.Attribute):
            return first_arg.attr

        # .value access like DataEventType.TRAINING_COMPLETED.value
        if isinstance(first_arg, ast.Attribute) and first_arg.attr == "value":
            if isinstance(first_arg.value, ast.Attribute):
                return first_arg.value.attr

        return None


class EventSubscriberVisitor(ast.NodeVisitor):
    """Find event subscription points in code."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.subscribers: list[tuple[str, int, str]] = []  # (event_name, line, handler_name)

    def visit_Call(self, node: ast.Call) -> None:
        # Look for subscribe patterns
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ("subscribe", "on", "add_handler", "register_handler"):
                event_name = self._extract_event_name(node.args)
                handler_name = self._extract_handler_name(node.args)
                if event_name:
                    self.subscribers.append((event_name, node.lineno, handler_name))

        self.generic_visit(node)

    def _extract_event_name(self, args: list[ast.expr]) -> str | None:
        if not args:
            return None
        first_arg = args[0]

        # String literal
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value

        # Attribute access like DataEventType.TRAINING_COMPLETED
        if isinstance(first_arg, ast.Attribute):
            return first_arg.attr

        return None

    def _extract_handler_name(self, args: list[ast.expr]) -> str:
        if len(args) < 2:
            return "unknown"
        handler_arg = args[1]

        if isinstance(handler_arg, ast.Attribute):
            return handler_arg.attr
        if isinstance(handler_arg, ast.Name):
            return handler_arg.id

        return "unknown"


def scan_file_for_events(file_path: Path) -> tuple[list[tuple[str, int]], list[tuple[str, int, str]]]:
    """Scan a file for event emitters and subscribers."""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        emitter_visitor = EventEmitterVisitor(str(file_path))
        emitter_visitor.visit(tree)

        subscriber_visitor = EventSubscriberVisitor(str(file_path))
        subscriber_visitor.visit(tree)

        return emitter_visitor.emitters, subscriber_visitor.subscribers

    except (SyntaxError, UnicodeDecodeError):
        return [], []


def scan_for_event_enum(root: Path) -> dict[str, str]:
    """Find DataEventType enum definitions."""
    events: dict[str, str] = {}

    for py_file in root.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            # Look for enum definitions like TRAINING_COMPLETED = "training_completed"
            pattern = r'(\w+)\s*=\s*["\'](\w+)["\']'
            for match in re.finditer(pattern, source):
                enum_name = match.group(1)
                value = match.group(2)
                if enum_name.isupper() and "_" in enum_name:
                    events[enum_name] = value
                    events[value] = enum_name  # Also map value to name
        except (UnicodeDecodeError, OSError):
            continue

    return events


def build_event_registry(root: Path) -> dict[str, EventInfo]:
    """Build complete event registry from codebase."""
    registry: dict[str, EventInfo] = defaultdict(EventInfo)

    for py_file in root.rglob("*.py"):
        if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
            continue

        emitters, subscribers = scan_file_for_events(py_file)

        short_path = str(py_file)
        if "ai-service/" in short_path:
            short_path = short_path[short_path.index("ai-service/") + 11:]

        for event_name, line in emitters:
            normalized = event_name.upper().replace("-", "_")
            if normalized not in registry:
                registry[normalized] = EventInfo(name=normalized)
            registry[normalized].emitters.append((short_path, line))
            registry[normalized].category = categorize_event(normalized)

        for event_name, line, handler in subscribers:
            normalized = event_name.upper().replace("-", "_")
            if normalized not in registry:
                registry[normalized] = EventInfo(name=normalized)
            registry[normalized].subscribers.append((short_path, line, handler))
            registry[normalized].category = categorize_event(normalized)

    return dict(registry)


def format_markdown(registry: dict[str, EventInfo]) -> str:
    """Format registry as markdown documentation."""
    lines = [
        "# Event System Reference",
        "",
        "*Auto-generated by `scripts/generate_event_reference.py`*",
        "",
        f"**Total Events**: {len(registry)}",
        "",
    ]

    # Group by category
    by_category: dict[str, list[EventInfo]] = defaultdict(list)
    for event in registry.values():
        by_category[event.category].append(event)

    # Summary statistics
    orphans = [e for e in registry.values() if e.is_orphan]
    dead = [e for e in registry.values() if e.is_dead]

    lines.extend([
        "## Summary",
        "",
        f"- **Events with emitters and subscribers**: {len(registry) - len(orphans) - len(dead)}",
        f"- **Orphan events** (emitted but no subscribers): {len(orphans)}",
        f"- **Dead events** (subscribers but no emitters): {len(dead)}",
        "",
    ])

    # Orphan events warning
    if orphans:
        lines.extend([
            "### Orphan Events (No Subscribers)",
            "",
            "These events are emitted but have no subscribers:",
            "",
        ])
        for e in sorted(orphans, key=lambda x: x.name):
            emitter_files = ", ".join(set(f for f, _ in e.emitters[:3]))
            lines.append(f"- `{e.name}` - emitted in: {emitter_files}")
        lines.append("")

    # Dead events warning
    if dead:
        lines.extend([
            "### Dead Events (No Emitters)",
            "",
            "These events have subscribers but are never emitted:",
            "",
        ])
        for e in sorted(dead, key=lambda x: x.name):
            sub_files = ", ".join(set(f for f, _, _ in e.subscribers[:3]))
            lines.append(f"- `{e.name}` - subscribed in: {sub_files}")
        lines.append("")

    # Events by category
    for category in sorted(by_category.keys()):
        events = sorted(by_category[category], key=lambda x: x.name)
        lines.extend([
            f"## {category} Events ({len(events)})",
            "",
            "| Event | Emitters | Subscribers |",
            "|-------|----------|-------------|",
        ])

        for event in events:
            emitter_count = len(event.emitters)
            sub_count = len(event.subscribers)
            emitter_str = f"{emitter_count} location{'s' if emitter_count != 1 else ''}"
            sub_str = f"{sub_count} handler{'s' if sub_count != 1 else ''}"

            # Mark orphans/dead
            status = ""
            if event.is_orphan:
                status = " :warning:"
            elif event.is_dead:
                status = " :x:"

            lines.append(f"| `{event.name}`{status} | {emitter_str} | {sub_str} |")

        lines.append("")

    # Detailed event information
    lines.extend([
        "## Event Details",
        "",
    ])

    for event in sorted(registry.values(), key=lambda x: x.name):
        if not event.emitters and not event.subscribers:
            continue

        lines.extend([
            f"### {event.name}",
            "",
            f"**Category**: {event.category}",
            "",
        ])

        if event.emitters:
            lines.append("**Emitters**:")
            for file_path, line in event.emitters[:5]:
                lines.append(f"- `{file_path}:{line}`")
            if len(event.emitters) > 5:
                lines.append(f"- ... and {len(event.emitters) - 5} more")
            lines.append("")

        if event.subscribers:
            lines.append("**Subscribers**:")
            for file_path, line, handler in event.subscribers[:5]:
                lines.append(f"- `{file_path}:{line}` - `{handler}()`")
            if len(event.subscribers) > 5:
                lines.append(f"- ... and {len(event.subscribers) - 5} more")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate event system reference documentation"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to scan (default: current dir)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    registry = build_event_registry(args.root)

    if args.json:
        output = {
            "total_events": len(registry),
            "events": {
                name: {
                    "category": info.category,
                    "emitters": [{"file": f, "line": l} for f, l in info.emitters],
                    "subscribers": [{"file": f, "line": l, "handler": h} for f, l, h in info.subscribers],
                    "is_orphan": info.is_orphan,
                    "is_dead": info.is_dead,
                }
                for name, info in sorted(registry.items())
            },
        }
        content = json.dumps(output, indent=2)
    else:
        content = format_markdown(registry)

    if args.output:
        args.output.write_text(content)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
