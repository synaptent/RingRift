#!/usr/bin/env python
"""Analyze weight sensitivity test results and classify weights for CMA-ES.

This script reads the output from `run_weight_sensitivity_test.py` and
classifies weights according to the strategy defined in TODO.md:

- **Strong positive signal (>55% win rate):** Keep with positive sign
- **Strong negative signal (<45% win rate):** Keep but **invert sign** for CMA-ES
  (these indicate features that hurt when over-weighted but help when negated)
- **Noise band (45-55% win rate):** Candidate for pruning or zero-initialization

Usage:
    python scripts/analyze_weight_sensitivity.py \
        --input logs/axis_aligned/sensitivity_results.json \
        --output logs/axis_aligned/cmaes_seed_weights.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_weights import BASE_V1_BALANCED_WEIGHTS, HEURISTIC_WEIGHT_KEYS


@dataclass
class WeightClassification:
    """Classification result for a single weight."""

    weight_name: str
    win_rate: float
    original_value: float
    classification: str  # 'positive', 'negative', 'noise'
    cmaes_seed_value: float
    action: str  # Description of what to do


def classify_weight(
    weight_name: str,
    win_rate: float,
    original_value: float,
    positive_threshold: float = 0.55,
    negative_threshold: float = 0.45,
) -> WeightClassification:
    """Classify a weight based on its win rate.

    Args:
        weight_name: Name of the weight
        win_rate: Win rate from sensitivity test (0.0 to 1.0)
        original_value: Original value from BASE_V1_BALANCED_WEIGHTS
        positive_threshold: Win rate threshold for positive signal (default 0.55)
        negative_threshold: Win rate threshold for negative signal (default 0.45)

    Returns:
        WeightClassification with classification and recommended CMA-ES seed value
    """
    if win_rate >= positive_threshold:
        # Strong positive signal - keep with positive sign
        return WeightClassification(
            weight_name=weight_name,
            win_rate=win_rate,
            original_value=original_value,
            classification="positive",
            cmaes_seed_value=abs(original_value),
            action=f"Keep positive (WR {win_rate:.0%} >= {positive_threshold:.0%})",
        )
    elif win_rate <= negative_threshold:
        # Strong negative signal - invert sign for CMA-ES
        # The feature hurts when over-weighted but may help when negated
        return WeightClassification(
            weight_name=weight_name,
            win_rate=win_rate,
            original_value=original_value,
            classification="negative",
            cmaes_seed_value=-abs(original_value),
            action=f"Invert sign (WR {win_rate:.0%} <= {negative_threshold:.0%}) - feature hurts when overweighted",
        )
    else:
        # Noise band - candidate for pruning or zero-initialization
        return WeightClassification(
            weight_name=weight_name,
            win_rate=win_rate,
            original_value=original_value,
            classification="noise",
            cmaes_seed_value=0.0,
            action=f"Zero/prune (WR {win_rate:.0%} in noise band [{negative_threshold:.0%}, {positive_threshold:.0%}])",
        )


def analyze_sensitivity_results(
    results_path: str,
    positive_threshold: float = 0.55,
    negative_threshold: float = 0.45,
) -> List[WeightClassification]:
    """Analyze sensitivity test results and classify all weights.

    Args:
        results_path: Path to sensitivity test JSON output
        positive_threshold: Win rate threshold for positive signal
        negative_threshold: Win rate threshold for negative signal

    Returns:
        List of WeightClassification objects sorted by win rate (descending)
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    classifications: List[WeightClassification] = []

    for result in results:
        weight_name = result["weight"]
        win_rate = result["win_rate"]
        original_value = BASE_V1_BALANCED_WEIGHTS.get(weight_name, 1.0)

        classification = classify_weight(
            weight_name=weight_name,
            win_rate=win_rate,
            original_value=original_value,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
        )
        classifications.append(classification)

    # Sort by win rate descending
    classifications.sort(key=lambda c: c.win_rate, reverse=True)
    return classifications


def generate_cmaes_seed_weights(
    classifications: List[WeightClassification],
    include_noise: bool = False,
) -> Dict[str, float]:
    """Generate CMA-ES seed weights from classifications.

    Args:
        classifications: List of weight classifications
        include_noise: If True, include noise-band weights at 0.0;
                      if False, use a small value (0.1) to allow exploration

    Returns:
        Dictionary of weight name -> CMA-ES seed value
    """
    weights: Dict[str, float] = {}

    for c in classifications:
        if c.classification == "noise" and not include_noise:
            # For noise band, use small non-zero value to allow exploration
            weights[c.weight_name] = 0.1 * (1.0 if c.original_value >= 0 else -1.0)
        else:
            weights[c.weight_name] = c.cmaes_seed_value

    # Ensure all keys from HEURISTIC_WEIGHT_KEYS are present
    for key in HEURISTIC_WEIGHT_KEYS:
        if key not in weights:
            # Weight not in results - use original value
            weights[key] = BASE_V1_BALANCED_WEIGHTS.get(key, 0.0)

    return weights


def print_classification_report(classifications: List[WeightClassification]) -> None:
    """Print a formatted classification report."""
    print("\n" + "=" * 80)
    print("WEIGHT CLASSIFICATION REPORT")
    print("=" * 80)

    positive = [c for c in classifications if c.classification == "positive"]
    negative = [c for c in classifications if c.classification == "negative"]
    noise = [c for c in classifications if c.classification == "noise"]

    print(f"\n📈 STRONG POSITIVE SIGNAL ({len(positive)} weights) - Keep with positive sign:")
    print("-" * 80)
    for c in positive:
        print(f"  {c.weight_name:<45} WR: {c.win_rate:>6.1%} -> seed: {c.cmaes_seed_value:>8.2f}")

    print(f"\n📉 STRONG NEGATIVE SIGNAL ({len(negative)} weights) - Invert sign for CMA-ES:")
    print("-" * 80)
    for c in negative:
        print(f"  {c.weight_name:<45} WR: {c.win_rate:>6.1%} -> seed: {c.cmaes_seed_value:>8.2f} (inverted)")

    print(f"\n📊 NOISE BAND ({len(noise)} weights) - Candidate for pruning/zero:")
    print("-" * 80)
    for c in noise:
        print(f"  {c.weight_name:<45} WR: {c.win_rate:>6.1%} -> seed: {c.cmaes_seed_value:>8.2f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Strong positive signal: {len(positive):>3} weights (keep positive)")
    print(f"  Strong negative signal: {len(negative):>3} weights (invert for CMA-ES)")
    print(f"  Noise band:            {len(noise):>3} weights (prune/zero)")
    print(f"  Total:                 {len(classifications):>3} weights")


def main():
    parser = argparse.ArgumentParser(description="Analyze weight sensitivity results and classify for CMA-ES")
    parser.add_argument("--input", type=str, required=True, help="Path to sensitivity test results JSON file")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/axis_aligned/cmaes_seed_weights.json",
        help="Output path for CMA-ES seed weights JSON",
    )
    parser.add_argument(
        "--positive-threshold", type=float, default=0.55, help="Win rate threshold for positive signal (default: 0.55)"
    )
    parser.add_argument(
        "--negative-threshold", type=float, default=0.45, help="Win rate threshold for negative signal (default: 0.45)"
    )
    parser.add_argument(
        "--include-noise-zeros",
        action="store_true",
        help="Include noise-band weights as exactly 0.0 (default: use small values for exploration)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Load and analyze results
    print(f"Loading sensitivity results from: {args.input}")
    classifications = analyze_sensitivity_results(
        args.input,
        positive_threshold=args.positive_threshold,
        negative_threshold=args.negative_threshold,
    )

    # Print report
    print_classification_report(classifications)

    # Generate CMA-ES seed weights
    cmaes_weights = generate_cmaes_seed_weights(
        classifications,
        include_noise=args.include_noise_zeros,
    )

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load original input metadata
    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    output_data = {
        "meta": {
            "source": args.input,
            "positive_threshold": args.positive_threshold,
            "negative_threshold": args.negative_threshold,
            "include_noise_zeros": args.include_noise_zeros,
            "source_config": input_data.get("config", {}),
        },
        "weights": cmaes_weights,
        "classifications": {
            "positive": [c.weight_name for c in classifications if c.classification == "positive"],
            "negative": [c.weight_name for c in classifications if c.classification == "negative"],
            "noise": [c.weight_name for c in classifications if c.classification == "noise"],
        },
        "details": [
            {
                "weight": c.weight_name,
                "win_rate": c.win_rate,
                "original_value": c.original_value,
                "classification": c.classification,
                "cmaes_seed_value": c.cmaes_seed_value,
                "action": c.action,
            }
            for c in classifications
        ],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    print(f"\nCMA-ES seed weights saved to: {args.output}")
    print("\nTo use with CMA-ES optimization:")
    print(f"  python scripts/run_cmaes_optimization.py --baseline {args.output}")


if __name__ == "__main__":
    main()
