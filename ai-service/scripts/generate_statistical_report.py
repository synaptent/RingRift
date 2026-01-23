#!/usr/bin/env python3
"""
Statistical Analysis Report Generator for AI Evaluation Results

This script reads all JSON result files from ai-service/results/ and generates
a comprehensive statistical analysis report including:
- Win rates with 95% Wilson score confidence intervals
- P-values for pairwise comparisons (vs 50% baseline)
- Effect sizes (Cohen's h for proportion differences)
- Statistical power analysis
- Ranking of AI implementations
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.utils.progress_reporter import ProgressReporter


@dataclass
class MatchupResult:
    """Represents the result of a matchup between two AI implementations."""

    player1: str
    player2: str
    player1_wins: int
    player2_wins: int
    draws: int
    total_games: int
    win_rate: float
    win_rate_ci: tuple[float, float]
    avg_game_length: float
    avg_game_length_std: float
    victory_types: dict[str, int]
    source_file: str


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.

    The Wilson score interval is more accurate than the normal approximation,
    especially for small sample sizes and proportions near 0 or 1.
    """
    if trials == 0:
        return (0.0, 1.0)

    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)

    p_hat = successes / trials
    denominator = 1 + z**2 / trials

    center = (p_hat + z**2 / (2 * trials)) / denominator
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * trials)) / trials) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (round(lower, 4), round(upper, 4))


def binomial_test_pvalue(successes: int, trials: int, null_prob: float = 0.5) -> float:
    """
    Calculate two-tailed p-value for binomial test.

    Tests null hypothesis that true probability = null_prob.
    Uses exact binomial probabilities.
    """
    if trials == 0:
        return 1.0

    # Calculate binomial coefficient and probabilities
    def binomial_coef(n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def binomial_prob(n: int, k: int, p: float) -> float:
        return binomial_coef(n, k) * (p**k) * ((1 - p) ** (n - k))

    # Calculate probability of observed outcome under null
    observed_prob = binomial_prob(trials, successes, null_prob)

    # Two-tailed: sum probabilities of outcomes as extreme or more extreme
    p_value = 0.0
    for k in range(trials + 1):
        prob_k = binomial_prob(trials, k, null_prob)
        if prob_k <= observed_prob + 1e-10:  # Small epsilon for floating point
            p_value += prob_k

    return min(1.0, round(p_value, 6))


def fishers_exact_test(a: int, b: int, c: int, d: int) -> float:
    """
    Fisher's exact test for 2x2 contingency table.

    Table:
              Outcome1  Outcome2
    Group1       a         b
    Group2       c         d

    Returns two-tailed p-value.
    """
    n = a + b + c + d

    def hypergeom_prob(a: int, b: int, c: int, d: int) -> float:
        """Calculate hypergeometric probability for given cell values."""
        try:
            numerator = math.factorial(a + b) * math.factorial(c + d) * math.factorial(a + c) * math.factorial(b + d)
            denominator = (
                math.factorial(a) * math.factorial(b) * math.factorial(c) * math.factorial(d) * math.factorial(n)
            )
            return numerator / denominator
        except (ValueError, ZeroDivisionError):
            return 0.0

    # Get probability of observed table
    p_observed = hypergeom_prob(a, b, c, d)

    # Calculate all possible tables with same margins and sum probabilities <= observed
    row1_total = a + b
    row2_total = c + d
    col1_total = a + c

    p_value = 0.0
    for a_new in range(max(0, col1_total - row2_total), min(row1_total, col1_total) + 1):
        b_new = row1_total - a_new
        c_new = col1_total - a_new
        d_new = row2_total - c_new

        if b_new >= 0 and c_new >= 0 and d_new >= 0:
            p_table = hypergeom_prob(a_new, b_new, c_new, d_new)
            if p_table <= p_observed + 1e-10:
                p_value += p_table

    return min(1.0, round(p_value, 6))


def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for difference between two proportions.

    h = φ1 - φ2, where φ = 2 * arcsin(sqrt(p))

    Interpretation:
    - |h| < 0.2: negligible
    - 0.2 <= |h| < 0.5: small
    - 0.5 <= |h| < 0.8: medium
    - |h| >= 0.8: large
    """
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return round(phi1 - phi2, 4)


def effect_size_interpretation(h: float) -> str:
    """Interpret Cohen's h effect size."""
    abs_h = abs(h)
    if abs_h < 0.2:
        return "negligible"
    elif abs_h < 0.5:
        return "small"
    elif abs_h < 0.8:
        return "medium"
    else:
        return "large"


def statistical_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """
    Approximate statistical power for comparing proportion to 0.5.

    Uses normal approximation for binomial test.
    Power = P(reject H0 | H1 true)
    """
    if n == 0:
        return 0.0

    # Standard error under null (p=0.5)
    se_null = math.sqrt(0.5 * 0.5 / n)

    # Critical value for two-tailed test
    z_alpha = 1.96 if alpha == 0.05 else 1.645

    # True proportion based on effect size (assume comparing to 0.5)
    # Cohen's h = 2*arcsin(sqrt(p)) - 2*arcsin(sqrt(0.5))
    # Solve for p: p = sin((h + 2*arcsin(sqrt(0.5))) / 2)^2
    phi_null = 2 * math.asin(math.sqrt(0.5))
    phi_alt = phi_null + effect_size
    p_alt = math.sin(phi_alt / 2) ** 2
    p_alt = max(0.01, min(0.99, p_alt))  # Clamp to valid range

    # Standard error under alternative
    se_alt = math.sqrt(p_alt * (1 - p_alt) / n)

    if se_alt == 0:
        return 1.0 if abs(p_alt - 0.5) > 0.01 else 0.0

    # Z-score for the alternative mean at the critical boundary
    z_power = (abs(p_alt - 0.5) - z_alpha * se_null) / se_alt

    # Approximate power using standard normal CDF
    # Using error function approximation
    def norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    return round(norm_cdf(z_power), 4)


def load_result_file(filepath: Path) -> MatchupResult | None:
    """Load and parse a single result JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        config = data.get("config", {})
        results = data.get("results", {})

        player1 = config.get("player1", "unknown")
        player2 = config.get("player2", "unknown")

        player1_wins = results.get("player1_wins", 0)
        player2_wins = results.get("player2_wins", 0)
        draws = results.get("draws", 0)
        total_games = player1_wins + player2_wins + draws

        win_rate = results.get("player1_win_rate", player1_wins / total_games if total_games > 0 else 0)

        # Use existing CI if available, otherwise compute
        ci = results.get("player1_win_rate_ci95")
        if ci and len(ci) == 2:
            win_rate_ci = (ci[0], ci[1])
        else:
            win_rate_ci = wilson_score_interval(player1_wins, total_games)

        return MatchupResult(
            player1=player1,
            player2=player2,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            draws=draws,
            total_games=total_games,
            win_rate=win_rate,
            win_rate_ci=win_rate_ci,
            avg_game_length=results.get("avg_game_length", 0),
            avg_game_length_std=results.get("avg_game_length_std", 0),
            victory_types=results.get("victory_types", {}),
            source_file=filepath.name,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return None


def load_all_results(results_dir: Path) -> list[MatchupResult]:
    """Load all result files from the results directory.

    Uses ProgressReporter to provide throttled visibility when scanning
    large result directories so long-running report generation jobs do not
    go silent.
    """
    results: list[MatchupResult] = []

    # Materialize the file list up front so we know the total for reporting.
    files = list(results_dir.glob("*.json"))
    total_files = len(files)

    reporter: ProgressReporter | None = None
    if total_files > 0:
        reporter = ProgressReporter(
            total_units=total_files,
            unit_name="files",
            context_label=f"stat-report | {results_dir}",
        )

    for idx, filepath in enumerate(files, start=1):
        result = load_result_file(filepath)
        if result:
            results.append(result)

        if reporter is not None:
            reporter.update(completed=idx)

    if reporter is not None:
        reporter.finish(
            extra_metrics={
                "parsed": len(results),
                "skipped": total_files - len(results),
            }
        )

    return results


def analyze_vs_random(results: list[MatchupResult]) -> dict:
    """Analyze all matchups against random baseline."""
    vs_random = {}

    for r in results:
        # Find matchups where one player is "random"
        if r.player2 == "random" and r.player1 != "random":
            ai_name = r.player1
            wins = r.player1_wins
            losses = r.player2_wins
            total = r.total_games
            win_rate = r.win_rate
        elif r.player1 == "random" and r.player2 != "random":
            # Note: In this case, we want player2's win rate
            ai_name = r.player2
            wins = r.player2_wins  # AI wins (as player 2)
            losses = r.player1_wins  # AI losses
            total = r.total_games
            win_rate = wins / total if total > 0 else 0
        else:
            continue

        if ai_name not in vs_random or total > vs_random[ai_name]["total_games"]:
            # P-value: testing if win_rate > 50%
            p_value = binomial_test_pvalue(wins, total, 0.5)

            # Effect size vs random (50%)
            effect = cohens_h(win_rate, 0.5)

            # Power analysis
            power = statistical_power(effect, total)

            vs_random[ai_name] = {
                "wins": wins,
                "losses": losses,
                "draws": r.draws,
                "total_games": total,
                "win_rate": win_rate,
                "win_rate_ci": wilson_score_interval(wins, total),
                "p_value": p_value,
                "effect_size": effect,
                "effect_interpretation": effect_size_interpretation(effect),
                "statistical_power": power,
                "significant": p_value < 0.05,
                "source_file": r.source_file,
            }

    return vs_random


def analyze_pairwise(results: list[MatchupResult]) -> dict:
    """Analyze pairwise comparisons between AIs (excluding random)."""
    pairwise = {}

    for r in results:
        # Skip random vs random
        if r.player1 == "random" and r.player2 == "random":
            continue
        # Skip vs random matchups (handled separately)
        if r.player1 == "random" or r.player2 == "random":
            continue

        key = f"{r.player1}_vs_{r.player2}"

        # P-value: testing if significantly different from 50%
        p_value = binomial_test_pvalue(r.player1_wins, r.total_games, 0.5)

        # Effect size
        effect = cohens_h(r.win_rate, 0.5)

        pairwise[key] = {
            "player1": r.player1,
            "player2": r.player2,
            "player1_wins": r.player1_wins,
            "player2_wins": r.player2_wins,
            "draws": r.draws,
            "total_games": r.total_games,
            "player1_win_rate": r.win_rate,
            "player1_win_rate_ci": r.win_rate_ci,
            "p_value": p_value,
            "effect_size": effect,
            "effect_interpretation": effect_size_interpretation(effect),
            "significant": p_value < 0.05,
            "source_file": r.source_file,
        }

    return pairwise


def calculate_cross_comparison(vs_random: dict) -> dict:
    """Calculate statistical comparisons between different AIs based on their vs-random performance."""
    comparisons = {}

    ai_names = list(vs_random.keys())

    for i, ai1 in enumerate(ai_names):
        for ai2 in ai_names[i + 1 :]:
            data1 = vs_random[ai1]
            data2 = vs_random[ai2]

            # Fisher's exact test for comparing two AIs
            # 2x2 table: AI wins vs losses for each
            a = data1["wins"]  # AI1 wins
            b = data1["losses"]  # AI1 losses
            c = data2["wins"]  # AI2 wins
            d = data2["losses"]  # AI2 losses

            p_value = fishers_exact_test(a, b, c, d)

            # Effect size between the two win rates
            effect = cohens_h(data1["win_rate"], data2["win_rate"])

            key = f"{ai1}_vs_{ai2}_indirect"
            comparisons[key] = {
                "ai1": ai1,
                "ai2": ai2,
                "ai1_win_rate_vs_random": data1["win_rate"],
                "ai2_win_rate_vs_random": data2["win_rate"],
                "difference": data1["win_rate"] - data2["win_rate"],
                "p_value": p_value,
                "effect_size": effect,
                "effect_interpretation": effect_size_interpretation(effect),
                "significant": p_value < 0.05,
                "comparison_type": "indirect_via_random",
            }

    return comparisons


def rank_ais(vs_random: dict, pairwise: dict) -> list[dict]:
    """Rank AIs based on performance metrics."""
    scores = {}

    # Primary score: win rate vs random
    for ai_name, data in vs_random.items():
        scores[ai_name] = {
            "name": ai_name,
            "win_rate_vs_random": data["win_rate"],
            "significant_vs_random": data["significant"],
            "effect_vs_random": data["effect_size"],
            "head_to_head_wins": 0,
            "head_to_head_losses": 0,
            "head_to_head_ties": 0,
        }

    # Secondary: head-to-head results
    for _key, data in pairwise.items():
        p1, p2 = data["player1"], data["player2"]

        if p1 not in scores:
            scores[p1] = {
                "name": p1,
                "win_rate_vs_random": 0,
                "significant_vs_random": False,
                "effect_vs_random": 0,
                "head_to_head_wins": 0,
                "head_to_head_losses": 0,
                "head_to_head_ties": 0,
            }
        if p2 not in scores:
            scores[p2] = {
                "name": p2,
                "win_rate_vs_random": 0,
                "significant_vs_random": False,
                "effect_vs_random": 0,
                "head_to_head_wins": 0,
                "head_to_head_losses": 0,
                "head_to_head_ties": 0,
            }

        if data["significant"]:
            if data["player1_win_rate"] > 0.5:
                scores[p1]["head_to_head_wins"] += 1
                scores[p2]["head_to_head_losses"] += 1
            else:
                scores[p2]["head_to_head_wins"] += 1
                scores[p1]["head_to_head_losses"] += 1
        else:
            scores[p1]["head_to_head_ties"] += 1
            scores[p2]["head_to_head_ties"] += 1

    # Sort by: 1) win rate vs random, 2) head-to-head wins, 3) effect size
    ranked = sorted(
        scores.values(),
        key=lambda x: (x["win_rate_vs_random"], x["head_to_head_wins"], x["effect_vs_random"]),
        reverse=True,
    )

    # Add rank
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1

    return ranked


def generate_report(results_dir: Path, output_path: Path) -> dict:
    """Generate comprehensive statistical analysis report."""

    print(f"Loading results from {results_dir}...")
    all_results = load_all_results(results_dir)
    print(f"Loaded {len(all_results)} result files")

    # Perform analyses
    vs_random = analyze_vs_random(all_results)
    pairwise = analyze_pairwise(all_results)
    cross_comparison = calculate_cross_comparison(vs_random)
    ranking = rank_ais(vs_random, pairwise)

    # Build report
    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            "results_directory": str(results_dir),
            "files_analyzed": [r.source_file for r in all_results],
            "statistical_methods": {
                "confidence_intervals": "Wilson score interval (95%)",
                "significance_test": "Binomial exact test (two-tailed)",
                "pairwise_comparison": "Fisher's exact test",
                "effect_size": "Cohen's h",
                "significance_threshold": 0.05,
            },
        },
        "summary": {
            "total_matchups_analyzed": len(all_results),
            "ais_evaluated": list(vs_random.keys()),
            "best_performer": ranking[0]["name"] if ranking else None,
        },
        "vs_random_baseline": vs_random,
        "pairwise_comparisons": pairwise,
        "cross_comparisons": cross_comparison,
        "ranking": ranking,
        "key_findings": {"cmaes_vs_baseline": None, "neural_network_performance": None, "training_effectiveness": None},
    }

    # Extract key findings
    # CMA-ES vs Baseline
    for key, data in pairwise.items():
        if "baseline_heuristic" in key and "cmaes_heuristic" in key:
            report["key_findings"]["cmaes_vs_baseline"] = {
                "result": "no_significant_difference" if not data["significant"] else "significant_difference",
                "p_value": data["p_value"],
                "effect_size": data["effect_size"],
                "effect_interpretation": data["effect_interpretation"],
                "conclusion": (
                    "CMA-ES optimization did NOT provide statistically significant improvement over baseline"
                    if not data["significant"]
                    else f"CMA-ES {'improved' if data['player1_win_rate'] < 0.5 else 'worse than'} baseline with {data['effect_interpretation']} effect"
                ),
                "raw_data": data,
            }

    # Neural network performance
    if "neural_network" in vs_random:
        nn_data = vs_random["neural_network"]
        baseline_data = vs_random.get("baseline_heuristic", {})

        report["key_findings"]["neural_network_performance"] = {
            "win_rate_vs_random": nn_data["win_rate"],
            "significant_vs_random": nn_data["significant"],
            "p_value_vs_random": nn_data["p_value"],
            "comparison_to_baseline": (
                nn_data["win_rate"] - baseline_data.get("win_rate", 0) if baseline_data else None
            ),
            "conclusion": (
                f"Neural network achieves {nn_data['win_rate']:.0%} win rate vs random "
                f"({'statistically significant' if nn_data['significant'] else 'not statistically significant'})"
            ),
        }

    # Training effectiveness
    nn_vs_heuristic_direct = None
    for key, data in pairwise.items():
        if "neural_network" in key:
            nn_vs_heuristic_direct = data
            break

    if nn_vs_heuristic_direct:
        report["key_findings"]["training_effectiveness"] = {
            "neural_network_vs_heuristics": {
                "matchup": f"{nn_vs_heuristic_direct['player1']} vs {nn_vs_heuristic_direct['player2']}",
                "result_win_rate": nn_vs_heuristic_direct["player1_win_rate"],
                "significant": nn_vs_heuristic_direct["significant"],
                "p_value": nn_vs_heuristic_direct["p_value"],
            },
            "conclusion": (
                "Neural network training shows limited effectiveness compared to hand-tuned heuristics. "
                "Heuristic approaches remain stronger for this game."
            ),
        }

    # Save report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {output_path}")
    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the report."""

    print("\n" + "=" * 70)
    print("          AI STATISTICAL ANALYSIS REPORT")
    print("=" * 70)
    print(f"Generated: {report['metadata']['generated_at']}")
    print(f"Files analyzed: {len(report['metadata']['files_analyzed'])}")
    print()

    # 1. VS RANDOM BASELINE
    print("=" * 70)
    print("1. PERFORMANCE VS RANDOM BASELINE")
    print("   (Null hypothesis: win_rate = 50%)")
    print("-" * 70)

    vs_random = report["vs_random_baseline"]
    sorted_vs_random = sorted(vs_random.items(), key=lambda x: x[1]["win_rate"], reverse=True)

    for ai_name, data in sorted_vs_random:
        ci_low, ci_high = data["win_rate_ci"]
        sig_str = "✓ SIGNIFICANT" if data["significant"] else "✗ not significant"
        print(f"   {ai_name}:")
        print(f"      Win Rate: {data['win_rate']:.0%} [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"      p-value: {data['p_value']:.4f} ({sig_str})")
        print(f"      Effect size: {data['effect_size']:.3f} ({data['effect_interpretation']})")
        print(f"      Statistical power: {data['statistical_power']:.1%}")
        print()

    # 2. PAIRWISE COMPARISONS
    print("=" * 70)
    print("2. PAIRWISE COMPARISONS (Direct Head-to-Head)")
    print("-" * 70)

    pairwise = report["pairwise_comparisons"]
    for _key, data in pairwise.items():
        sig_str = "SIGNIFICANT" if data["significant"] else "not significant"
        ci_low, ci_high = data["player1_win_rate_ci"]
        print(f"   {data['player1']} vs {data['player2']}:")
        print(f"      {data['player1']} win rate: {data['player1_win_rate']:.0%} [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"      p-value: {data['p_value']:.4f} ({sig_str})")
        print(f"      Effect size: {data['effect_size']:.3f} ({data['effect_interpretation']})")
        print()

    # 3. EFFECT SIZES SUMMARY
    print("=" * 70)
    print("3. EFFECT SIZES SUMMARY")
    print("-" * 70)

    for ai_name, data in sorted_vs_random:
        print(f"   Cohen's h for {ai_name} vs Random: {data['effect_size']:.3f} ({data['effect_interpretation']})")
    print()

    # 4. RANKING
    print("=" * 70)
    print("4. AI RANKING (by strength)")
    print("-" * 70)

    ranking = report["ranking"]
    for entry in ranking:
        sig_marker = "**" if entry.get("significant_vs_random", False) else ""
        print(f"   {entry['rank']}. {entry['name']}{sig_marker}")
        print(f"      Win rate vs random: {entry['win_rate_vs_random']:.0%}")
        print(
            f"      Head-to-head: {entry['head_to_head_wins']}W-{entry['head_to_head_losses']}L-{entry['head_to_head_ties']}T"
        )
        print()

    # 5. KEY FINDINGS
    print("=" * 70)
    print("5. KEY FINDINGS")
    print("-" * 70)

    findings = report["key_findings"]

    if findings.get("cmaes_vs_baseline"):
        f = findings["cmaes_vs_baseline"]
        print("   CMA-ES vs Baseline Heuristic:")
        print(f"      Result: {f['result'].replace('_', ' ')}")
        print(f"      p-value: {f['p_value']:.4f}")
        print(f"      Effect size: {f['effect_size']:.3f} ({f['effect_interpretation']})")
        print(f"      Conclusion: {f['conclusion']}")
        print()

    if findings.get("neural_network_performance"):
        f = findings["neural_network_performance"]
        print("   Neural Network Performance:")
        print(f"      Win rate vs random: {f['win_rate_vs_random']:.0%}")
        print(f"      Significant: {'Yes' if f['significant_vs_random'] else 'No'} (p={f['p_value_vs_random']:.4f})")
        if f["comparison_to_baseline"] is not None:
            print(f"      vs Baseline: {f['comparison_to_baseline']:+.0%} difference")
        print(f"      Conclusion: {f['conclusion']}")
        print()

    if findings.get("training_effectiveness"):
        f = findings["training_effectiveness"]
        print("   Training Effectiveness:")
        print(f"      {f['conclusion']}")
        print()

    # 6. STATISTICAL POWER ANALYSIS
    print("=" * 70)
    print("6. STATISTICAL POWER ANALYSIS")
    print("-" * 70)
    print("   Sample size: 20 games per matchup")
    print("   Significance level: α = 0.05")
    print()
    print("   Power to detect various effect sizes:")
    for h, desc in [(0.3, "small"), (0.5, "medium"), (0.8, "large")]:
        power = statistical_power(h, 20)
        print(f"      {desc} effect (h={h}): {power:.0%} power")
    print()
    print("   Recommendation: For 80% power to detect medium effects,")
    print("   increase sample size to approximately 50 games per matchup.")

    print("\n" + "=" * 70)
    print("                    END OF REPORT")
    print("=" * 70)


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    output_path = results_dir / "statistical_analysis_report.json"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    # Generate and print report
    report = generate_report(results_dir, output_path)
    print_summary(report)

    return 0


if __name__ == "__main__":
    exit(main())
