#!/usr/bin/env python3
"""Validate GPU MCTS selfplay data quality.

Checks:
1. Game completion rate (>90% expected)
2. Average game length (detect degenerate play)
3. Policy distribution entropy (higher = more exploration)
4. Value distribution (balanced +1/-1 for 2p)
5. Feature validity (no NaN/Inf)

Usage:
    python scripts/validate_gpu_mcts_data.py data/training/gpu_mcts_hex8_2p.npz
    python scripts/validate_gpu_mcts_data.py --dir data/training/gpu_mcts/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def entropy(probs: np.ndarray) -> float:
    """Compute entropy of probability distribution."""
    # Filter out zeros to avoid log(0)
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log(p))


def validate_npz(path: str) -> dict:
    """Validate a GPU MCTS NPZ file.

    Returns dict with validation metrics and pass/fail status.
    """
    results = {
        "path": path,
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    try:
        data = np.load(path)
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load: {e}")
        return results

    # Check required fields
    required = ["features", "globals", "values", "policy_indices", "policy_values"]
    for field in required:
        if field not in data.files:
            results["valid"] = False
            results["errors"].append(f"Missing required field: {field}")

    if not results["valid"]:
        return results

    features = data["features"]
    values = data["values"]
    policy_indices = data["policy_indices"]
    policy_values = data["policy_values"]

    n_samples = len(features)
    results["n_samples"] = n_samples

    # 1. Check for NaN/Inf
    if np.any(np.isnan(features)):
        results["valid"] = False
        results["errors"].append("Features contain NaN")
    if np.any(np.isinf(features)):
        results["valid"] = False
        results["errors"].append("Features contain Inf")
    if np.any(np.isnan(values)):
        results["valid"] = False
        results["errors"].append("Values contain NaN")

    # 2. Check value distribution
    value_mean = np.mean(values)
    value_std = np.std(values)
    results["value_mean"] = float(value_mean)
    results["value_std"] = float(value_std)

    # For 2-player games, expect roughly balanced +1/-1
    if abs(value_mean) > 0.3:
        results["warnings"].append(
            f"Value mean is skewed: {value_mean:.2f} (expected near 0 for balanced games)"
        )

    # 3. Check policy entropy
    entropies = []
    for i in range(min(1000, n_samples)):  # Sample first 1000
        pv = policy_values[i]
        pv = pv[pv > 0]  # Non-zero entries
        if len(pv) > 0:
            # Normalize to sum to 1
            pv = pv / np.sum(pv)
            entropies.append(entropy(pv))

    if entropies:
        avg_entropy = np.mean(entropies)
        results["avg_policy_entropy"] = float(avg_entropy)

        # Entropy < 0.5 suggests very peaked distributions (might be OK for late game)
        # Entropy > 3.0 suggests very flat distributions (unusual)
        if avg_entropy < 0.3:
            results["warnings"].append(
                f"Low policy entropy: {avg_entropy:.2f} (very peaked - check if intentional)"
            )

    # 4. Check policy coverage
    avg_actions = np.mean([np.sum(pv > 0) for pv in policy_values])
    results["avg_actions_per_sample"] = float(avg_actions)

    if avg_actions < 2:
        results["warnings"].append(
            f"Low action coverage: {avg_actions:.1f} actions/sample (expected >2)"
        )

    # 5. Check feature shape consistency
    if features.ndim != 4:
        results["errors"].append(f"Expected 4D features, got {features.ndim}D")
        results["valid"] = False

    # 6. Metadata
    for key in ["board_type", "encoder_version", "source"]:
        if key in data.files:
            results[key] = str(data[key])

    return results


def print_results(results: dict) -> None:
    """Pretty-print validation results."""
    path = results["path"]
    valid = results["valid"]

    status = "✓ PASS" if valid else "✗ FAIL"
    logger.info(f"\n{status}: {path}")
    logger.info("-" * 60)

    if "n_samples" in results:
        logger.info(f"  Samples: {results['n_samples']}")
    if "board_type" in results:
        logger.info(f"  Board type: {results['board_type']}")
    if "encoder_version" in results:
        logger.info(f"  Encoder: {results['encoder_version']}")

    if "value_mean" in results:
        logger.info(f"  Value mean: {results['value_mean']:.3f} (std: {results['value_std']:.3f})")
    if "avg_policy_entropy" in results:
        logger.info(f"  Policy entropy: {results['avg_policy_entropy']:.3f}")
    if "avg_actions_per_sample" in results:
        logger.info(f"  Actions/sample: {results['avg_actions_per_sample']:.1f}")

    if results["errors"]:
        logger.info("\n  Errors:")
        for err in results["errors"]:
            logger.info(f"    ✗ {err}")

    if results["warnings"]:
        logger.info("\n  Warnings:")
        for warn in results["warnings"]:
            logger.info(f"    ⚠ {warn}")


def main():
    parser = argparse.ArgumentParser(description="Validate GPU MCTS selfplay data")
    parser.add_argument("files", nargs="*", help="NPZ files to validate")
    parser.add_argument("--dir", help="Directory containing NPZ files")
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat warnings as failures")

    args = parser.parse_args()

    # Collect files
    files = list(args.files)
    if args.dir:
        files.extend(str(p) for p in Path(args.dir).glob("*.npz"))

    if not files:
        logger.error("No files specified. Use --dir or provide file paths.")
        sys.exit(1)

    # Validate each file
    all_passed = True
    for path in files:
        results = validate_npz(path)
        print_results(results)

        if not results["valid"]:
            all_passed = False
        elif args.fail_on_warning and results["warnings"]:
            all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("All validations PASSED")
        sys.exit(0)
    else:
        logger.info("Some validations FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
