"""Evaluation Module for RingRift AI.

This module provides tools for evaluating AI models:

1. **Benchmark Suite** (benchmark_suite.py):
   - Inference speed benchmarks
   - Memory usage benchmarks
   - Policy/value accuracy benchmarks
   - Tactical pattern recognition
   - MCTS search quality
   - Robustness testing

2. **Human Evaluation** (human_eval.py):
   - Move quality ratings from human experts
   - A/B model preference comparisons
   - Position annotations for training improvement
   - Web interface for collecting evaluations
   - Analysis tools for aggregating human feedback

Usage:
    # Benchmark a model
    from app.evaluation import BenchmarkSuite, create_default_suite

    suite = create_default_suite()
    result = suite.run(model, model_id="my_model_v1")
    print(f"Aggregate score: {result.compute_aggregate_score():.4f}")

    # Human evaluation (requires sqlite database)
    from app.evaluation import (
        EvaluationDatabase,
        HumanEvalServer,
        TaskGenerator,
    )

    db = EvaluationDatabase(Path("evaluations.db"))
    generator = TaskGenerator(db)
    task = generator.create_move_quality_task(
        model_id="my_model",
        position_data={"board": [...], "phase": "capture"},
        ai_move=42,
    )

Integration Points:
    - Quality Pipeline: Benchmark results can inform quality scoring weights
    - Training Pipeline: Human feedback generates training signals
    - Tournament System: Model comparison via human preference
    - CI/CD: Automated benchmark regression testing

See Also:
    - scripts/benchmark_*.py for CLI benchmark runners
    - scripts/run_model_elo_tournament.py for Elo evaluation
"""

from app.evaluation.benchmark_suite import (
    Benchmark,
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    InferenceBenchmark,
    MCTSBenchmark,
    MemoryBenchmark,
    PolicyAccuracyBenchmark,
    RobustnessBenchmark,
    TacticalBenchmark,
    ValueAccuracyBenchmark,
    create_default_suite,
)
from app.evaluation.human_eval import (
    EvaluationAnalyzer,
    EvaluationDatabase,
    EvaluationResponse,
    EvaluationTask,
    EvaluationType,
    EvaluatorProfile,
    HumanEvalServer,
    MoveQuality,
    TaskGenerator,
)

__all__ = [
    # Benchmark classes
    "Benchmark",
    "BenchmarkCategory",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkSuiteResult",
    "InferenceBenchmark",
    "MCTSBenchmark",
    "MemoryBenchmark",
    "PolicyAccuracyBenchmark",
    "RobustnessBenchmark",
    "TacticalBenchmark",
    "ValueAccuracyBenchmark",
    "create_default_suite",
    # Human evaluation classes
    "EvaluationAnalyzer",
    "EvaluationDatabase",
    "EvaluationResponse",
    "EvaluationTask",
    "EvaluationType",
    "EvaluatorProfile",
    "HumanEvalServer",
    "MoveQuality",
    "TaskGenerator",
]
