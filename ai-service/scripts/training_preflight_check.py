#!/usr/bin/env python3
"""
RingRift Training Preflight Check

Comprehensive verification that all training infrastructure components
are ready and functioning correctly before running training.

Usage:
    cd ai-service
    python scripts/training_preflight_check.py
"""

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Ensure we can import from app/
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    name: str
    passed: bool
    message: str = ""
    error: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class CheckCategory:
    """Category of related checks."""

    name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)


class PreflightChecker:
    """Runs all preflight checks and reports results."""

    def __init__(self):
        self.categories: List[CheckCategory] = []

    def add_check(self, category_name: str, result: CheckResult) -> None:
        """Add a check result to a category."""
        for cat in self.categories:
            if cat.name == category_name:
                cat.checks.append(result)
                return
        new_cat = CheckCategory(name=category_name, checks=[result])
        self.categories.append(new_cat)

    def run_all_checks(self) -> bool:
        """Run all preflight checks. Returns True if all pass."""
        self._check_environment()
        self._check_data_generation()
        self._check_neural_networks()
        self._check_training_infrastructure()
        self._check_memory_management()
        self._check_optimization_utilities()
        self._run_smoke_tests()

        return all(cat.all_passed for cat in self.categories)

    def _check_environment(self) -> None:
        """Check environment and dependencies."""
        cat = "Environment"

        # Python version
        py_version = sys.version_info
        py_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
        if py_version >= (3, 10):
            self.add_check(
                cat,
                CheckResult(
                    name="Python version",
                    passed=True,
                    message=py_str,
                ),
            )
        else:
            self.add_check(
                cat,
                CheckResult(
                    name="Python version",
                    passed=False,
                    message=py_str,
                    error=f"Python 3.10+ required, got {py_str}",
                    suggestion="Upgrade to Python 3.10 or later",
                ),
            )

        # PyTorch
        try:
            import torch

            self.add_check(
                cat,
                CheckResult(
                    name="PyTorch version",
                    passed=True,
                    message=torch.__version__,
                ),
            )
        except ImportError as e:
            self.add_check(
                cat,
                CheckResult(
                    name="PyTorch version",
                    passed=False,
                    error=str(e),
                    suggestion="Install PyTorch: pip install torch",
                ),
            )

        # CUDA availability (optional)
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.add_check(
                    cat,
                    CheckResult(
                        name="CUDA available",
                        passed=True,
                        message=f"Yes ({device_name})",
                    ),
                )
            else:
                self.add_check(
                    cat,
                    CheckResult(
                        name="CUDA available",
                        passed=True,
                        message="No (CPU training only)",
                    ),
                )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="CUDA available",
                    passed=True,
                    message=f"Check failed: {e}",
                ),
            )

        # Required packages
        required_packages = ["torch", "numpy", "psutil"]
        optional_packages = ["cma"]
        all_ok = True
        missing = []

        for pkg in required_packages:
            try:
                __import__(pkg)
            except ImportError:
                all_ok = False
                missing.append(pkg)

        for pkg in optional_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(f"{pkg} (optional)")

        if all_ok:
            msg = "OK"
            if any("optional" in m for m in missing):
                optional_missing = [m for m in missing if "optional" in m]
                msg = f"OK (missing optional: {', '.join(optional_missing)})"
            self.add_check(
                cat,
                CheckResult(
                    name="Required packages",
                    passed=True,
                    message=msg,
                ),
            )
        else:
            self.add_check(
                cat,
                CheckResult(
                    name="Required packages",
                    passed=False,
                    error=f"Missing: {', '.join(missing)}",
                    suggestion="Install missing packages: pip install "
                    + " ".join(m.split()[0] for m in missing if "optional" not in m),
                ),
            )

    def _check_data_generation(self) -> None:
        """Check data generation components."""
        cat = "Data Generation"

        # Rules engine
        try:
            from app.rules.default_engine import DefaultRulesEngine

            DefaultRulesEngine()  # Verify instantiation works
            self.add_check(
                cat,
                CheckResult(
                    name="Rules engine",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Rules engine",
                    passed=False,
                    error=str(e),
                    suggestion="Ensure you're running from ai-service directory",
                ),
            )

        # Self-play game (short test)
        try:
            from app.training.env import RingRiftEnv
            from app.models import BoardType

            env = RingRiftEnv(board_type=BoardType.SQUARE8, max_moves=10)
            env.reset(seed=42)
            moves_played = 0

            for _ in range(10):
                legal = env.legal_moves()
                if not legal:
                    break
                # Pick first legal move
                move = legal[0]
                _, _, done, _ = env.step(move)
                moves_played += 1
                if done:
                    break

            self.add_check(
                cat,
                CheckResult(
                    name="Self-play game",
                    passed=True,
                    message=f"OK ({moves_played} moves played)",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Self-play game",
                    passed=False,
                    error=str(e),
                    suggestion="Check game engine and RingRiftEnv implementation",
                ),
            )

        # Generate data module
        try:
            from app.training import generate_data  # noqa: F401
            from app.training.generate_data import (  # noqa: F401
                create_initial_state,
            )

            self.add_check(
                cat,
                CheckResult(
                    name="generate_data module",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="generate_data module",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/training/generate_data.py",
                ),
            )

        # Canonical parity gate for Square-8 self-play data.
        #
        # We treat canonical_square8.db as the primary training source for
        # Square-8; before running serious training, its parity summary must
        # show:
        #   - games_with_semantic_divergence == 0
        #   - games_with_structural_issues == 0
        #   - total_games_checked > 0
        try:
            from pathlib import Path

            root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            summary_path = root / "parity_summary.canonical_square8.json"

            if not summary_path.exists():
                self.add_check(
                    cat,
                    CheckResult(
                        name="Canonical square8 parity summary",
                        passed=False,
                        error="parity_summary.canonical_square8.json not found",
                        suggestion=(
                            "Run scripts/check_ts_python_replay_parity.py "
                            "--db data/games/canonical_square8.db and keep the "
                            "summary JSON alongside TRAINING_DATA_REGISTRY.md "
                            "before training on canonical_square8.db."
                        ),
                    ),
                )
            else:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)

                sem = int(summary.get("games_with_semantic_divergence", 1))
                struct = int(summary.get("games_with_structural_issues", 1))
                total = int(summary.get("total_games_checked", 0))

                if sem == 0 and struct == 0 and total > 0:
                    self.add_check(
                        cat,
                        CheckResult(
                            name="Canonical square8 parity summary",
                            passed=True,
                            message=f"OK ({total} games parity-checked, no semantic divergences)",
                        ),
                    )
                else:
                    self.add_check(
                        cat,
                        CheckResult(
                            name="Canonical square8 parity summary",
                            passed=False,
                            error=(
                                f"Parity summary reports semantic_divergences={sem}, "
                                f"structural_issues={struct}, total_games_checked={total}"
                            ),
                            suggestion=(
                                "Investigate TS↔Python replay parity for "
                                "canonical_square8.db and rerun the parity gate "
                                "before training."
                            ),
                        ),
                    )
        except Exception as e:  # pragma: no cover - defensive
            self.add_check(
                cat,
                CheckResult(
                    name="Canonical square8 parity summary",
                    passed=False,
                    error=f"Failed to load parity summary: {e}",
                ),
            )

    def _check_neural_networks(self) -> None:
        """Check neural network components."""
        cat = "Neural Networks"

        # RingRiftCNN_v2 instantiation
        try:
            from app.ai.neural_net import RingRiftCNN_v2

            net = RingRiftCNN_v2(board_size=8)
            self.add_check(
                cat,
                CheckResult(
                    name="RingRiftCNN_v2 instantiation",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="RingRiftCNN_v2 instantiation",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/ai/neural_net.py for RingRiftCNN_v2 class",
                ),
            )

        # RingRiftCNN_v2 forward pass
        try:
            import torch
            from app.ai.neural_net import RingRiftCNN_v2

            net = RingRiftCNN_v2(board_size=8)
            # Create dummy input: batch=1, channels=40 (10*4), 8x8 board
            # Plus global features (20) - must match model's global_features parameter
            batch_size = 2
            in_channels = net.total_in_channels  # 40 by default
            dummy_spatial = torch.randn(batch_size, in_channels, 8, 8)
            dummy_global = torch.randn(batch_size, net.global_features)

            with torch.no_grad():
                # Network returns (value, policy)
                value, policy = net(dummy_spatial, dummy_global)

            # Check output shapes (multi-player value head)
            assert value.shape == (batch_size, net.num_players), f"Value shape mismatch: {value.shape} != {(batch_size, net.num_players)}"
            assert policy.shape == (batch_size, net.policy_size), f"Policy shape mismatch: {policy.shape} != {(batch_size, net.policy_size)}"

            # Check for NaN
            assert not torch.isnan(value).any(), "Value contains NaN"
            assert not torch.isnan(policy).any(), "Policy contains NaN"

            self.add_check(
                cat,
                CheckResult(
                    name="RingRiftCNN_v2 forward pass",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="RingRiftCNN_v2 forward pass",
                    passed=False,
                    error=str(e),
                    suggestion="Check neural network architecture",
                ),
            )

        # Weight initialization check
        try:
            import torch
            from app.ai.neural_net import RingRiftCNN_v2

            net = RingRiftCNN_v2(board_size=8)
            has_nan = False
            unreasonable_magnitude = False

            for name, param in net.named_parameters():
                if torch.isnan(param).any():
                    has_nan = True
                    break
                if param.abs().max() > 100:
                    unreasonable_magnitude = True

            if has_nan:
                self.add_check(
                    cat,
                    CheckResult(
                        name="Weight initialization",
                        passed=False,
                        error="Weights contain NaN values",
                        suggestion="Check weight initialization",
                    ),
                )
            elif unreasonable_magnitude:
                self.add_check(
                    cat,
                    CheckResult(
                        name="Weight initialization",
                        passed=True,
                        message="OK (some large weights detected)",
                    ),
                )
            else:
                self.add_check(
                    cat,
                    CheckResult(
                        name="Weight initialization",
                        passed=True,
                        message="OK",
                    ),
                )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Weight initialization",
                    passed=False,
                    error=str(e),
                ),
            )

    def _check_training_infrastructure(self) -> None:
        """Check training infrastructure components."""
        cat = "Training Infrastructure"

        # EarlyStopping class
        try:
            from app.training.train import EarlyStopping

            EarlyStopping(patience=5)  # Verify instantiation works
            self.add_check(
                cat,
                CheckResult(
                    name="EarlyStopping class",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="EarlyStopping class",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/training/train.py",
                ),
            )

        # Checkpoint saving
        try:
            from app.training.train import (  # noqa: F401
                save_checkpoint,
                load_checkpoint,
            )

            self.add_check(
                cat,
                CheckResult(
                    name="Checkpoint saving",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Checkpoint saving",
                    passed=False,
                    error=str(e),
                    suggestion="Check save_checkpoint/load_checkpoint in train.py",
                ),
            )

        # LR schedulers
        try:
            from app.training.train import get_warmup_scheduler  # noqa: F401

            self.add_check(
                cat,
                CheckResult(
                    name="LR schedulers",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="LR schedulers",
                    passed=False,
                    error=str(e),
                    suggestion="Check get_warmup_scheduler in train.py",
                ),
            )

        # Distributed utilities
        try:
            from app.training.distributed import (  # noqa: F401
                setup_distributed,
                cleanup_distributed,
                is_main_process,
                get_distributed_sampler,
                wrap_model_ddp,
            )

            self.add_check(
                cat,
                CheckResult(
                    name="Distributed utilities",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Distributed utilities",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/training/distributed.py",
                ),
            )

    def _check_memory_management(self) -> None:
        """Check memory management components."""
        cat = "Memory Management"

        # MemoryConfig
        try:
            from app.utils.memory_config import MemoryConfig

            config = MemoryConfig()
            default_gb = config.max_memory_gb
            self.add_check(
                cat,
                CheckResult(
                    name="MemoryConfig",
                    passed=True,
                    message=f"OK ({default_gb} GB default)",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="MemoryConfig",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/utils/memory_config.py",
                ),
            )

        # BoundedTranspositionTable
        try:
            from app.ai.bounded_transposition_table import (
                BoundedTranspositionTable,
            )

            tt = BoundedTranspositionTable(max_entries=1000)
            tt.put("test_key", {"value": 42})
            retrieved = tt.get("test_key")
            assert retrieved is not None, "Failed to retrieve"
            self.add_check(
                cat,
                CheckResult(
                    name="BoundedTranspositionTable",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="BoundedTranspositionTable",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/ai/bounded_transposition_table.py",
                ),
            )

    def _check_optimization_utilities(self) -> None:
        """Check optimization utilities."""
        cat = "Optimization Utilities"

        # CMA-ES
        try:
            import cma  # noqa: F401

            self.add_check(
                cat,
                CheckResult(
                    name="CMA-ES",
                    passed=True,
                    message="OK",
                ),
            )
        except ImportError:
            self.add_check(
                cat,
                CheckResult(
                    name="CMA-ES",
                    passed=True,
                    message="Not installed (optional)",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="CMA-ES",
                    passed=False,
                    error=str(e),
                ),
            )

        # Hex augmentation
        try:
            from app.training.hex_augmentation import (  # noqa: F401
                HexSymmetryTransform,
            )

            self.add_check(
                cat,
                CheckResult(
                    name="Hex augmentation",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Hex augmentation",
                    passed=False,
                    error=str(e),
                    suggestion="Check app/training/hex_augmentation.py",
                ),
            )

        # Parallel self-play utilities check (just imports)
        try:
            # Check that key components exist
            from app.ai.descent_ai import DescentAI  # noqa: F401
            from app.ai.heuristic_ai import HeuristicAI  # noqa: F401

            self.add_check(
                cat,
                CheckResult(
                    name="Parallel self-play",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Parallel self-play",
                    passed=False,
                    error=str(e),
                    suggestion="Check AI implementations in app/ai/",
                ),
            )

    def _run_smoke_tests(self) -> None:
        """Run quick smoke tests."""
        cat = "Smoke Test"

        # Generate data
        states = []
        try:
            from app.training.env import RingRiftEnv
            from app.models import BoardType

            env = RingRiftEnv(board_type=BoardType.SQUARE8, max_moves=50)

            for game_idx in range(5):
                env.reset(seed=game_idx)
                game_states = [env.state]
                for _ in range(20):
                    legal = env.legal_moves()
                    if not legal:
                        break
                    move = legal[0]
                    _, _, done, _ = env.step(move)
                    game_states.append(env.state)
                    if done:
                        break
                states.extend(game_states)

            self.add_check(
                cat,
                CheckResult(
                    name="Generate data",
                    passed=True,
                    message=f"OK ({len(states)} states from 5 games)",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Generate data",
                    passed=False,
                    error=str(e),
                ),
            )
            return  # Can't continue without data

        # Save and load data
        try:
            import numpy as np

            # Create dummy training data
            num_samples = min(len(states), 10)
            dummy_features = np.random.randn(num_samples, 40, 8, 8).astype(np.float32)
            dummy_globals = np.random.randn(num_samples, 10).astype(np.float32)
            dummy_values = np.random.randn(num_samples).astype(np.float32)

            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
                temp_path = f.name
                np.savez(
                    temp_path,
                    features=dummy_features,
                    globals=dummy_globals,
                    values=dummy_values,
                )

            # Load and verify
            loaded = np.load(temp_path)
            assert loaded["features"].shape == dummy_features.shape
            assert loaded["globals"].shape == dummy_globals.shape
            assert loaded["values"].shape == dummy_values.shape

            os.unlink(temp_path)

            self.add_check(
                cat,
                CheckResult(
                    name="Load data",
                    passed=True,
                    message="OK",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Load data",
                    passed=False,
                    error=str(e),
                ),
            )

        # Training step
        try:
            import torch
            import torch.nn as nn
            from app.ai.neural_net import RingRiftCNN_v2

            net = RingRiftCNN_v2(board_size=8)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            # Create dummy batch
            batch_size = 4
            in_channels = net.total_in_channels
            dummy_spatial = torch.randn(batch_size, in_channels, 8, 8)
            dummy_global = torch.randn(batch_size, net.global_features)
            target_value = torch.randn(batch_size, 1)

            # Forward pass
            net.train()
            optimizer.zero_grad()
            # Network returns (value, policy)
            value, policy = net(dummy_spatial, dummy_global)

            # Compute loss (value head only for simplicity)
            loss = loss_fn(value, target_value)

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if not (torch.isfinite(torch.tensor(loss_val)) and not torch.isnan(torch.tensor(loss_val))):
                raise ValueError(f"Loss is not finite: {loss_val}")

            self.add_check(
                cat,
                CheckResult(
                    name="Training step",
                    passed=True,
                    message=f"OK (loss={loss_val:.4f})",
                ),
            )
        except Exception as e:
            self.add_check(
                cat,
                CheckResult(
                    name="Training step",
                    passed=False,
                    error=str(e),
                    suggestion="Check neural network training loop",
                ),
            )

    def print_report(self) -> None:
        """Print the preflight check report."""
        print("\n=== RingRift Training Preflight Check ===\n")

        total_passed = 0
        total_checks = 0

        for cat in self.categories:
            print(f"[{cat.name}]")
            for check in cat.checks:
                total_checks += 1
                if check.passed:
                    total_passed += 1
                    symbol = "✓"
                    msg = check.message or "OK"
                    print(f"  {symbol} {check.name}: {msg}")
                else:
                    symbol = "✗"
                    print(f"  {symbol} {check.name}: FAILED")
                    if check.error:
                        print(f"    Error: {check.error}")
                    if check.suggestion:
                        print(f"    Suggestion: {check.suggestion}")
            print()

        # Summary
        all_passed = total_passed == total_checks
        if all_passed:
            print("=== PREFLIGHT CHECK PASSED ===")
            print(f"All {total_checks} checks passed. Ready for training.")
        else:
            print("=== PREFLIGHT CHECK FAILED ===")
            failed = total_checks - total_passed
            print(
                f"{failed} of {total_checks} checks failed. "
                "Please fix issues before training."
            )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the training preflight checker."""
    parser = argparse.ArgumentParser(
        description=(
            "Run RingRift training preflight checks, including canonical "
            "replay DB gating based on TRAINING_DATA_REGISTRY.md."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Path to a training pipeline config JSON that may reference "
            "replay DBs (for example tier_training_pipeline.square8_2p.json). "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=str(AI_SERVICE_ROOT / "TRAINING_DATA_REGISTRY.md"),
        help=(
            "Path to TRAINING_DATA_REGISTRY.md. Defaults to the copy under "
            "ai-service/."
        ),
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help=(
            "Allow non-canonical DBs as training sources. This is intended "
            "only for explicitly marked legacy/experimental runs."
        ),
    )
    return parser.parse_args(argv)


def _discover_db_paths_from_config(config_path: Path) -> List[Path]:
    """Discover *.db paths referenced anywhere in a JSON config."""
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    found: List[str] = []

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, str):
            if ".db" in node:
                found.append(node)

    _walk(data)

    db_paths: List[Path] = []
    for raw in found:
        raw_str = raw.strip()
        if not raw_str:
            continue
        if os.path.isabs(raw_str):
            db_paths.append(Path(raw_str))
        else:
            db_paths.append(AI_SERVICE_ROOT / raw_str)

    seen: Dict[Path, bool] = {}
    unique: List[Path] = []
    for path in db_paths:
        key = path.resolve()
        if key in seen:
            continue
        seen[key] = True
        unique.append(path)
    return unique


def _run_canonical_sources_preflight(args: argparse.Namespace) -> bool:
    """Run canonical replay DB validation based on training configs.

    Returns True when either:

    - No DB paths are discovered, or
    - All discovered DBs are canonical according to the registry and gate
      artefacts, or
    - Problems are found but --allow-legacy is set.
    """
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = AI_SERVICE_ROOT / registry_path

    db_paths: List[Path] = []
    for cfg in args.config or []:
        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = AI_SERVICE_ROOT / cfg_path
        if not cfg_path.exists():
            print(
                f"[canonical-source-error] Config file not found: {cfg_path}",
                file=sys.stderr,
            )
            return False
        db_paths.extend(_discover_db_paths_from_config(cfg_path))

    if not db_paths:
        # Nothing to validate; treat this as OK but make it explicit so
        # callers know that no replay DBs were discovered.
        print(
            "No replay DB paths discovered in training configs; "
            "skipping canonical source validation."
        )
        return True

    # Import lazily to avoid hard dependency when running preflight in
    # isolation from the broader training stack.
    from scripts.validate_canonical_training_sources import (  # type: ignore
        validate_canonical_sources,
    )

    result = validate_canonical_sources(registry_path, db_paths)
    if result.get("ok"):
        print("Canonical training sources validated:")
        for checked in result.get("checked", []):
            print(f"  {checked}")
        return True

    # Surface all issues with a stable prefix so CI and tooling can grep.
    for issue in result.get("problems", []):
        print(f"[canonical-source-error] {issue}", file=sys.stderr)

    if getattr(args, "allow_legacy", False):
        print(
            "[canonical-sources] allow-legacy flag set; continuing "
            "despite canonical-source errors.",
            file=sys.stderr,
        )
        return True

    return False


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point. Returns 0 on success, 1 on failure."""
    args = _parse_args(argv)

    # Run canonical replay DB gating first so we fail fast when training
    # configs reference non-canonical or un-gated replay databases.
    if not _run_canonical_sources_preflight(args):
        return 1

    checker = PreflightChecker()
    all_passed = checker.run_all_checks()
    checker.print_report()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
