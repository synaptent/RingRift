"""Focused tests for app.evaluation package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_evaluation_surface() -> None:
    module = importlib.import_module("app.evaluation")

    expected = [
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

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
