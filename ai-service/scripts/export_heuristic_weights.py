#!/usr/bin/env python3
"""
Export heuristic weights from Python to JSON format for TypeScript.

Usage:
    python scripts/export_heuristic_weights.py [--profile ID] [--output PATH]

Examples:
    # Export the balanced profile (default)
    python scripts/export_heuristic_weights.py

    # Export a specific profile
    python scripts/export_heuristic_weights.py --profile aggressive

    # Export all profiles to a directory
    python scripts/export_heuristic_weights.py --all --output ./weights/

    # Export to custom file
    python scripts/export_heuristic_weights.py --output ./weights.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.heuristic_weights import (  # noqa: E402
    HEURISTIC_WEIGHT_PROFILES,
    HeuristicWeights,
    load_trained_profiles_if_available,
)


def weights_to_json(weights: HeuristicWeights) -> dict:
    """Convert HeuristicWeights TypedDict to plain dict for JSON."""
    return dict(weights)


def export_profile(profile_id: str, output_path: Path) -> None:
    """Export a single profile to a JSON file."""
    # Try loading trained profiles first (from JSON override files)
    load_trained_profiles_if_available()

    if profile_id not in HEURISTIC_WEIGHT_PROFILES:
        available = list(HEURISTIC_WEIGHT_PROFILES.keys())
        msg = f"Unknown profile '{profile_id}'. Available: {available}"
        raise ValueError(msg)

    weights = HEURISTIC_WEIGHT_PROFILES[profile_id]
    json_data = {
        "profile_id": profile_id,
        "weights": weights_to_json(weights),
        "metadata": {
            "exported_from": "python",
            "format_version": "1.0",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Exported '{profile_id}' to {output_path}")


def export_all_profiles(output_dir: Path) -> None:
    """Export all profiles to a directory."""
    # Try loading trained profiles first
    load_trained_profiles_if_available()

    output_dir.mkdir(parents=True, exist_ok=True)

    for profile_id in HEURISTIC_WEIGHT_PROFILES:
        output_path = output_dir / f"{profile_id}.json"
        export_profile(profile_id, output_path)

    # Also export a combined file
    combined_path = output_dir / "all_profiles.json"
    combined_data = {
        "profiles": {pid: weights_to_json(weights) for pid, weights in HEURISTIC_WEIGHT_PROFILES.items()},
        "metadata": {
            "exported_from": "python",
            "format_version": "1.0",
        },
    }

    with open(combined_path, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(f"Exported combined profiles to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Export heuristic weights to JSON for TypeScript")
    parser.add_argument(
        "--profile",
        type=str,
        default="heuristic_v1_balanced",
        help="Profile ID to export (default: heuristic_v1_balanced)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (file or directory with --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all profiles to a directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available profile IDs and exit",
    )

    args = parser.parse_args()

    # Load any trained overrides
    load_trained_profiles_if_available()

    if args.list:
        print("Available profile IDs:")
        for pid in HEURISTIC_WEIGHT_PROFILES:
            print(f"  - {pid}")
        return

    if args.all:
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path("./exported_weights")
        export_all_profiles(output_dir)
    else:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"./exported_weights/{args.profile}.json")

        export_profile(args.profile, output_path)


if __name__ == "__main__":
    main()
