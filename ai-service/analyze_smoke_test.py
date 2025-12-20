#!/usr/bin/env python3
"""Analyze Phase 1 smoke test results."""

import json
import sys

print("=" * 70)
print("PHASE 1 SMOKE TEST ANALYSIS")
print("=" * 70)

# Load generation data
gen1_path = "logs/cmaes/smoke_test/runs/cmaes_20251201_172946/generations/generation_001.json"
gen2_path = "logs/cmaes/smoke_test/runs/cmaes_20251201_172946/generations/generation_002.json"

try:
    with open(gen1_path) as f:
        gen1 = json.load(f)
    with open(gen2_path) as f:
        gen2 = json.load(f)
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Could not load generation files: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("GENERATION 1 STATS")
print("=" * 70)
print(f"  Mean fitness:    {gen1['mean_fitness']:.4f}")
print(f"  Std fitness:     {gen1['std_fitness']:.4f} {'⚠️ ZERO VARIANCE!' if gen1['std_fitness'] == 0 else ''}")
print(f"  Max fitness:     {gen1['max_fitness']:.4f}")
print(f"  Population size: {gen1['population_size']}")
print(f"  Games per eval:  {gen1['games_per_eval']}")

print("\n" + "=" * 70)
print("GENERATION 2 STATS")
print("=" * 70)
print(f"  Mean fitness:    {gen2['mean_fitness']:.4f}")
print(f"  Std fitness:     {gen2['std_fitness']:.4f} {'⚠️ ZERO VARIANCE!' if gen2['std_fitness'] == 0 else ''}")
print(f"  Max fitness:     {gen2['max_fitness']:.4f}")
print(f"  Population size: {gen2['population_size']}")
print(f"  Games per eval:  {gen2['games_per_eval']}")

print("\n" + "=" * 70)
print("CRITICAL METRIC EVALUATION")
print("=" * 70)

# Decision logic based on std_fitness
if gen2['std_fitness'] > 0:
    print("\n✅ PASS: Non-zero fitness variance detected")
    print("   Training has gradient - safe to proceed to Phase 2")
    decision = "PASS"
else:
    print("\n❌ FAIL: Zero fitness variance in both generations")
    print("   No optimization signal detected")
    print("\n   DIAGNOSIS:")
    print("   - All candidates in population produced IDENTICAL fitness")
    print("   - This indicates NO GRADIENT for optimization")
    print("   - Possible causes:")
    print("     1. Population size too small (2 candidates)")
    print("     2. Games per eval too small (4 games)")
    print("     3. Evaluation lacks diversity/randomness")
    print("     4. Bug in fitness calculation")
    print("     5. Weight perturbations not affecting gameplay")
    decision = "FAIL"

print("\n" + "=" * 70)
print(f"FINAL DECISION: {decision}")
print("=" * 70)

sys.exit(0 if decision == "PASS" else 1)
