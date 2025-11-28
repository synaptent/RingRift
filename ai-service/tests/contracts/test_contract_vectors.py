"""
Contract Test Vector Runner for Python Rules Engine

This module runs the canonical test vectors from
tests/fixtures/contract-vectors/v2/ against the Python rules engine
to validate cross-language parity with TypeScript.

Test vectors are JSON files that define:
- Input state + move
- Expected output state assertions

The runner:
1. Loads test vectors from the shared fixtures directory
2. Deserializes input state into Python GameState
3. Applies the move using GameEngine.apply_move()
4. Validates output against assertions

Usage:
    pytest ai-service/tests/contracts/test_contract_vectors.py -v
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from app.game_engine import GameEngine
from app.rules.serialization import (
    TestVector,
    TestVectorBundle,
    compute_collapsed_count,
    compute_marker_count,
    compute_s_invariant,
    compute_stack_count,
)

# ============================================================================
# Test vector loading
# ============================================================================

# Path to shared test vectors (relative to project root)
VECTORS_DIR = Path(__file__).parent.parent.parent.parent / \
    "tests" / "fixtures" / "contract-vectors" / "v2"

# Categories of test vectors
VECTOR_CATEGORIES = [
    "placement",
    "movement",
    "capture",
    "line_detection",
    "territory",
    "chain_capture",
]


def load_vector_file(category: str) -> Optional[TestVectorBundle]:
    """Load test vectors for a specific category."""
    filepath = VECTORS_DIR / f"{category}.vectors.json"
    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TestVectorBundle(data)


def load_all_vectors() -> List[TestVector]:
    """Load all test vectors from all categories."""
    all_vectors: List[TestVector] = []

    for category in VECTOR_CATEGORIES:
        bundle = load_vector_file(category)
        if bundle:
            all_vectors.extend(bundle.vectors)

    return all_vectors


def get_vector_ids() -> List[str]:
    """Get all vector IDs for parametrization."""
    vectors = load_all_vectors()
    return [v.id for v in vectors]


def get_vector_by_id(vector_id: str) -> Optional[TestVector]:
    """Get a specific vector by ID."""
    vectors = load_all_vectors()
    for v in vectors:
        if v.id == vector_id:
            return v
    return None


# ============================================================================
# Assertion validation
# ============================================================================


class ValidationResult:
    """Result of validating a test vector."""

    def __init__(self, vector_id: str):
        self.vector_id = vector_id
        self.passed = True
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def add_failure(self, message: str) -> None:
        """Add a failure message."""
        self.passed = False
        self.failures.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        result = f"[{status}] {self.vector_id}"
        if self.failures:
            result += "\n  Failures:"
            for f in self.failures:
                result += f"\n    - {f}"
        if self.warnings:
            result += "\n  Warnings:"
            for w in self.warnings:
                result += f"\n    - {w}"
        return result


def validate_assertions(
    vector: TestVector,
    result_state: Any,
    initial_s: int,
) -> ValidationResult:
    """Validate test vector assertions against the result state.

    Args:
        vector: The test vector with assertions
        result_state: The GameState after applying the move
        initial_s: The S-invariant value before the move

    Returns:
        ValidationResult with pass/fail status and details
    """
    validation = ValidationResult(vector.id)
    assertions = vector.assertions

    # Current player assertion
    if assertions.current_player is not None:
        actual = result_state.current_player
        expected = assertions.current_player
        if actual != expected:
            validation.add_failure(
                f"currentPlayer: expected {expected}, got {actual}"
            )

    # Current phase assertion
    if assertions.current_phase is not None:
        actual = result_state.current_phase.value
        expected = assertions.current_phase
        if actual != expected:
            validation.add_failure(
                f"currentPhase: expected '{expected}', got '{actual}'"
            )

    # Game status assertion
    if assertions.game_status is not None:
        actual = result_state.game_status.value
        expected = assertions.game_status
        if actual != expected:
            validation.add_failure(
                f"gameStatus: expected '{expected}', got '{actual}'"
            )

    # Stack count assertion
    if assertions.stack_count is not None:
        actual = compute_stack_count(result_state)
        expected = assertions.stack_count
        if actual != expected:
            validation.add_failure(
                f"stackCount: expected {expected}, got {actual}"
            )

    # Marker count assertion
    if assertions.marker_count is not None:
        actual = compute_marker_count(result_state)
        expected = assertions.marker_count
        if actual != expected:
            validation.add_failure(
                f"markerCount: expected {expected}, got {actual}"
            )

    # Collapsed count assertion
    if assertions.collapsed_count is not None:
        actual = compute_collapsed_count(result_state)
        expected = assertions.collapsed_count
        if actual != expected:
            validation.add_failure(
                f"collapsedCount: expected {expected}, got {actual}"
            )

    # S-invariant delta assertion
    if assertions.s_invariant_delta is not None:
        final_s = compute_s_invariant(result_state)
        actual_delta = final_s - initial_s
        expected = assertions.s_invariant_delta
        if actual_delta != expected:
            validation.add_failure(
                f"sInvariantDelta: expected {expected}, "
                f"got {actual_delta} (S: {initial_s} -> {final_s})"
            )

    return validation


# ============================================================================
# Test execution
# ============================================================================


def execute_vector(vector: TestVector) -> ValidationResult:
    """Execute a single test vector and return validation result.

    Args:
        vector: The test vector to execute

    Returns:
        ValidationResult with pass/fail status and details
    """
    # Compute initial S-invariant before applying move
    initial_s = compute_s_invariant(vector.input_state)

    # Apply the move using GameEngine
    try:
        result_state = GameEngine.apply_move(
            vector.input_state,
            vector.input_move,
        )
    except Exception as e:
        validation = ValidationResult(vector.id)
        validation.add_failure(f"Exception during apply_move: {e}")
        return validation

    # Validate assertions
    return validate_assertions(vector, result_state, initial_s)


# ============================================================================
# Pytest test cases
# ============================================================================


class TestContractVectors:
    """Contract test vector suite."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear GameEngine cache before each test."""
        GameEngine.clear_cache()

    def test_vectors_directory_exists(self):
        """Verify test vectors directory is accessible."""
        assert VECTORS_DIR.exists(), (
            f"Test vectors directory not found: {VECTORS_DIR}"
        )

    def test_all_categories_have_vectors(self):
        """Verify all vector categories have files."""
        for category in VECTOR_CATEGORIES:
            filepath = VECTORS_DIR / f"{category}.vectors.json"
            assert filepath.exists(), (
                f"Missing vector file: {category}.vectors.json"
            )

    def test_vector_count(self):
        """Verify expected number of test vectors."""
        vectors = load_all_vectors()
        # Phase 3 created 12 vectors (3 placement, 3 movement,
        # 2 capture, 2 line_detection, 2 territory)
        assert len(vectors) >= 12, (
            f"Expected at least 12 vectors, found {len(vectors)}"
        )


# Parametrized test for all vectors
@pytest.fixture(params=get_vector_ids(), ids=lambda x: x)
def vector(request) -> TestVector:
    """Fixture providing each test vector."""
    v = get_vector_by_id(request.param)
    if v is None:
        pytest.skip(f"Vector not found: {request.param}")
    return v


def test_contract_vector(vector: TestVector):
    """Execute a contract test vector against Python engine.

    This test:
    1. Loads the input state and move from the vector
    2. Applies the move using GameEngine.apply_move()
    3. Validates the result state against expected assertions

    If the test fails, it indicates a parity divergence between
    the Python and TypeScript engines that should be investigated.
    """
    GameEngine.clear_cache()
    result = execute_vector(vector)

    if not result.passed:
        # Build detailed failure message
        failure_details = "\n".join(result.failures)
        pytest.fail(
            f"Contract vector {vector.id} failed:\n{failure_details}"
        )


# ============================================================================
# Direct execution for debugging
# ============================================================================


def run_all_vectors_directly() -> Dict[str, Any]:
    """Run all vectors and return summary (for debugging)."""
    vectors = load_all_vectors()
    results: Dict[str, Any] = {
        "total": len(vectors),
        "passed": 0,
        "failed": 0,
        "failures": [],
    }

    for vector in vectors:
        GameEngine.clear_cache()
        validation = execute_vector(vector)

        if validation.passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "id": vector.id,
                "messages": validation.failures,
            })

        print(validation)

    return results


if __name__ == "__main__":
    print(f"Loading vectors from: {VECTORS_DIR}")
    print("-" * 60)

    summary = run_all_vectors_directly()

    print("-" * 60)
    print(
        f"Summary: {summary['passed']}/{summary['total']} passed, "
        f"{summary['failed']} failed"
    )

    if summary["failures"]:
        print("\nFailed vectors:")
        for failure in summary["failures"]:
            print(f"  - {failure['id']}")
            for msg in failure["messages"]:
                print(f"      {msg}")