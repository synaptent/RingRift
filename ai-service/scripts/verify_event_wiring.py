#!/usr/bin/env python3
"""Verify all DataEventType events have emitters and subscribers.

This script scans the codebase to identify:
- Events that are emitted but have no subscribers (orphan emitters)
- Events that have subscribers but are never emitted (orphan subscribers)

Usage:
    python scripts/verify_event_wiring.py

December 2025
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.distributed.data_events import DataEventType


def scan_file(filepath: Path) -> tuple[set[str], set[str]]:
    """Scan a Python file for event emissions and subscriptions.

    Returns:
        (emitted_events, subscribed_events)
    """
    emitted: set[str] = set()
    subscribed: set[str] = set()

    try:
        content = filepath.read_text()
    except (OSError, UnicodeDecodeError):
        return emitted, subscribed

    # Patterns for event emission
    # emit_* functions
    emit_pattern = re.compile(r'\bemit_(\w+)\s*\(', re.IGNORECASE)
    # DataEventType.* in emit context
    emit_type_pattern = re.compile(r'\bemit[^(]*\(\s*DataEventType\.(\w+)', re.IGNORECASE)
    # _emit_event or emit_event with DataEventType
    emit_generic_pattern = re.compile(r'emit_event\s*\(\s*["\']?DataEventType\.(\w+)', re.IGNORECASE)
    emit_generic_str_pattern = re.compile(r'emit_event\s*\(\s*["\'](\w+)["\']', re.IGNORECASE)
    # publish with event type
    publish_pattern = re.compile(r'\bpublish\s*\(\s*DataEventType\.(\w+)', re.IGNORECASE)
    publish_str_pattern = re.compile(r'\bpublish\s*\(\s*["\'](\w+)["\']', re.IGNORECASE)

    # Patterns for event subscription
    # subscribe with DataEventType
    subscribe_pattern = re.compile(r'\bsubscribe\s*\(\s*DataEventType\.(\w+)', re.IGNORECASE)
    subscribe_str_pattern = re.compile(r'\bsubscribe\s*\(\s*["\'](\w+)["\']', re.IGNORECASE)
    # on_* handlers
    on_handler_pattern = re.compile(r'\b(?:async\s+)?def\s+_?on_(\w+)\s*\(', re.IGNORECASE)
    # _get_subscriptions or _get_event_subscriptions return values
    subscription_dict_pattern = re.compile(r'["\'](\w+)["\']\s*:\s*self\._on_', re.IGNORECASE)

    # Scan for emissions
    for match in emit_pattern.finditer(content):
        event_name = match.group(1).upper()
        emitted.add(event_name)

    for match in emit_type_pattern.finditer(content):
        emitted.add(match.group(1).upper())

    for match in emit_generic_pattern.finditer(content):
        emitted.add(match.group(1).upper())

    for match in emit_generic_str_pattern.finditer(content):
        emitted.add(match.group(1).upper())

    for match in publish_pattern.finditer(content):
        emitted.add(match.group(1).upper())

    for match in publish_str_pattern.finditer(content):
        emitted.add(match.group(1).upper())

    # Scan for subscriptions
    for match in subscribe_pattern.finditer(content):
        subscribed.add(match.group(1).upper())

    for match in subscribe_str_pattern.finditer(content):
        subscribed.add(match.group(1).upper())

    for match in subscription_dict_pattern.finditer(content):
        subscribed.add(match.group(1).upper())

    return emitted, subscribed


def scan_codebase(root_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Scan codebase for event emissions and subscriptions.

    Returns:
        (emitters_by_event, subscribers_by_event)
    """
    emitters: dict[str, list[str]] = defaultdict(list)
    subscribers: dict[str, list[str]] = defaultdict(list)

    # Directories to scan
    scan_dirs = [
        root_dir / "app",
        root_dir / "scripts",
    ]

    # Skip patterns
    skip_patterns = [
        "__pycache__",
        ".pyc",
        "test_",
        "_test.py",
        "archive/",
        "deprecated_",
    ]

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue

        for filepath in scan_dir.rglob("*.py"):
            # Skip test files and archives
            path_str = str(filepath)
            if any(skip in path_str for skip in skip_patterns):
                continue

            emitted, subscribed = scan_file(filepath)

            rel_path = str(filepath.relative_to(root_dir))
            for event in emitted:
                emitters[event].append(rel_path)
            for event in subscribed:
                subscribers[event].append(rel_path)

    return dict(emitters), dict(subscribers)


def main():
    root_dir = Path(__file__).parent.parent

    print("Scanning codebase for event wiring...")
    print()

    # Get all defined event types
    all_events = {e.name for e in DataEventType}

    # Scan for actual usage
    emitters, subscribers = scan_codebase(root_dir)

    # Find issues
    orphan_emitters: list[str] = []
    orphan_subscribers: list[str] = []
    fully_wired: list[str] = []
    never_used: list[str] = []

    for event in sorted(all_events):
        has_emitters = event in emitters
        has_subscribers = event in subscribers

        if has_emitters and has_subscribers:
            fully_wired.append(event)
        elif has_emitters and not has_subscribers:
            orphan_emitters.append(event)
        elif has_subscribers and not has_emitters:
            orphan_subscribers.append(event)
        else:
            never_used.append(event)

    # Report
    print("=" * 60)
    print("EVENT WIRING VERIFICATION REPORT")
    print("=" * 60)
    print()

    print(f"Total event types defined: {len(all_events)}")
    print(f"Fully wired (emitter + subscriber): {len(fully_wired)}")
    print(f"Orphan emitters (no subscribers): {len(orphan_emitters)}")
    print(f"Orphan subscribers (no emitters): {len(orphan_subscribers)}")
    print(f"Never used (no emitter or subscriber): {len(never_used)}")
    print()

    if orphan_emitters:
        print("-" * 60)
        print("ORPHAN EMITTERS (events emitted but never subscribed):")
        print("-" * 60)
        for event in orphan_emitters:
            print(f"  - {event}")
            for file in emitters.get(event, [])[:3]:
                print(f"      emitted in: {file}")
        print()

    if orphan_subscribers:
        print("-" * 60)
        print("ORPHAN SUBSCRIBERS (events subscribed but never emitted):")
        print("-" * 60)
        for event in orphan_subscribers:
            print(f"  - {event}")
            for file in subscribers.get(event, [])[:3]:
                print(f"      subscribed in: {file}")
        print()

    if never_used:
        print("-" * 60)
        print("NEVER USED (event types with no wiring):")
        print("-" * 60)
        for event in never_used[:20]:  # Limit output
            print(f"  - {event}")
        if len(never_used) > 20:
            print(f"  ... and {len(never_used) - 20} more")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_issues = len(orphan_emitters) + len(orphan_subscribers)
    if total_issues == 0:
        print("✅ All events are properly wired!")
        return 0
    else:
        print(f"⚠️  Found {total_issues} wiring issues to investigate")
        print()
        print("To fix orphan emitters: Add subscribers in coordination modules")
        print("To fix orphan subscribers: Verify emitters exist or remove dead code")
        return 1


if __name__ == "__main__":
    sys.exit(main())
