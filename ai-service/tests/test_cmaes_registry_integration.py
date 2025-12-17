"""Tests for CMA-ES model registry integration.

Tests registration of CMA-ES optimization results in the model registry,
including metrics tracking, auto-promotion, and weight loading.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.training.cmaes_registry_integration import (
    register_cmaes_result,
    get_best_heuristic_model,
    load_heuristic_weights_from_registry,
    list_cmaes_runs,
    CMAESRunConfig,
)
from app.training.model_registry import (
    ModelRegistry,
    ModelStage,
    ModelType,
)


class TestCMAESRunConfig:
    """Test CMAESRunConfig dataclass."""

    def test_config_to_dict(self):
        """Test serialization to dict."""
        config = CMAESRunConfig(
            population_size=20,
            sigma=0.5,
            generations=50,
            games_per_eval=10,
            board_type="square8",
            num_players=2,
            opponent_mode="baseline-only",
            run_id="test_run_001",
        )

        d = config.to_dict()

        assert d["population_size"] == 20
        assert d["sigma"] == 0.5
        assert d["generations"] == 50
        assert d["board_type"] == "square8"
        assert d["num_players"] == 2
        assert d["run_id"] == "test_run_001"


class TestRegisterCMAESResult:
    """Test CMA-ES result registration."""

    def setup_method(self):
        """Create temporary directories and test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"

        # Create a dummy weights file
        self.weights_path = Path(self.temp_dir) / "test_weights.json"
        self.weights_data = {
            "weights": {
                "material": 1.0,
                "position": 0.5,
                "mobility": 0.3,
            },
            "timestamp": "2024-01-01T00:00:00",
            "generation": 50,
            "fitness": 0.85,
        }
        with open(self.weights_path, "w") as f:
            json.dump(self.weights_data, f)

    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_basic(self):
        """Test basic CMA-ES result registration."""
        model_id, version = register_cmaes_result(
            weights_path=self.weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.85,
            generation=50,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        assert model_id == "heuristic_square8_2p"
        assert version == 1

        # Verify in registry
        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)

        assert model is not None
        assert model.model_type == ModelType.HEURISTIC
        assert model.stage == ModelStage.DEVELOPMENT
        assert model.metrics.win_rate == 0.85

    def test_register_with_cmaes_config(self):
        """Test registration with CMA-ES hyperparameters."""
        model_id, version = register_cmaes_result(
            weights_path=self.weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.75,
            generation=100,
            cmaes_config={
                "population_size": 30,
                "sigma": 0.3,
                "generations": 100,
                "games_per_eval": 20,
                "opponent_mode": "mixed",
                "run_id": "test_run",
            },
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)

        # Check training config extra_config
        extra = model.training_config.extra_config
        assert extra["cmaes_population_size"] == 30
        assert extra["cmaes_sigma"] == 0.3
        assert extra["cmaes_games_per_eval"] == 20
        assert extra["optimizer_type"] == "cmaes"
        assert extra["final_generation"] == 100

    def test_register_multiple_versions(self):
        """Test registering multiple versions of same config."""
        # First version
        model_id1, v1 = register_cmaes_result(
            weights_path=self.weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.70,
            generation=50,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        # Create new weights file for second version
        weights_path2 = Path(self.temp_dir) / "test_weights_v2.json"
        with open(weights_path2, "w") as f:
            json.dump({
                "weights": {"material": 1.2, "position": 0.6},
                "fitness": 0.80,
            }, f)

        # Second version
        model_id2, v2 = register_cmaes_result(
            weights_path=weights_path2,
            board_type="square8",
            num_players=2,
            fitness=0.80,
            generation=100,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        assert model_id1 == model_id2
        assert v2 == 2

    def test_auto_promotion(self):
        """Test auto-promotion to staging on fitness improvement."""
        # First version with lower fitness
        register_cmaes_result(
            weights_path=self.weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.60,
            generation=50,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        # Create new weights file
        weights_path2 = Path(self.temp_dir) / "better_weights.json"
        with open(weights_path2, "w") as f:
            json.dump({"weights": {}, "fitness": 0.75}, f)

        # Second version with higher fitness - should auto-promote
        model_id, version = register_cmaes_result(
            weights_path=weights_path2,
            board_type="square8",
            num_players=2,
            fitness=0.75,
            generation=100,
            registry_dir=self.registry_dir,
            auto_promote=True,
            min_fitness_improvement=0.02,
        )

        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)

        # Should be promoted to staging due to fitness improvement
        assert model.stage == ModelStage.STAGING

    def test_tags_added(self):
        """Test that appropriate tags are added."""
        model_id, version = register_cmaes_result(
            weights_path=self.weights_path,
            board_type="hex8",
            num_players=3,
            fitness=0.65,
            generation=25,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        registry = ModelRegistry(self.registry_dir)
        model = registry.get_model(model_id, version)

        assert "board:hex8" in model.tags
        assert "players:3" in model.tags
        assert "generation:25" in model.tags
        assert "cmaes" in model.tags


class TestGetBestHeuristicModel:
    """Test best heuristic model retrieval."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_best_model(self):
        """Test retrieving best model by fitness."""
        # Create weights files
        for i, fitness in enumerate([0.60, 0.75, 0.70]):
            weights_path = Path(self.temp_dir) / f"weights_{i}.json"
            with open(weights_path, "w") as f:
                json.dump({"weights": {}, "fitness": fitness}, f)

            register_cmaes_result(
                weights_path=weights_path,
                board_type="square8",
                num_players=2,
                fitness=fitness,
                generation=i * 25,
                registry_dir=self.registry_dir,
                auto_promote=False,
            )

        best = get_best_heuristic_model(
            board_type="square8",
            num_players=2,
            registry_dir=self.registry_dir,
        )

        assert best is not None
        assert best["metrics"]["win_rate"] == 0.75

    def test_no_models(self):
        """Test when no models exist."""
        result = get_best_heuristic_model(
            board_type="square8",
            num_players=2,
            registry_dir=self.registry_dir,
        )

        assert result is None


class TestLoadHeuristicWeights:
    """Test loading weights from registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"

        self.test_weights = {
            "material": 1.5,
            "position": 0.8,
            "mobility": 0.4,
        }

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_weights(self):
        """Test loading weights from registered model."""
        weights_path = Path(self.temp_dir) / "weights.json"
        with open(weights_path, "w") as f:
            json.dump({"weights": self.test_weights, "fitness": 0.80}, f)

        register_cmaes_result(
            weights_path=weights_path,
            board_type="square8",
            num_players=2,
            fitness=0.80,
            generation=50,
            registry_dir=self.registry_dir,
            auto_promote=False,
        )

        loaded = load_heuristic_weights_from_registry(
            board_type="square8",
            num_players=2,
            registry_dir=self.registry_dir,
            stage="development",
        )

        assert loaded is not None
        assert loaded["material"] == 1.5
        assert loaded["position"] == 0.8


class TestListCMAESRuns:
    """Test listing CMA-ES runs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_dir = Path(self.temp_dir) / "registry"

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_runs(self):
        """Test listing CMA-ES runs."""
        # Create multiple runs
        for i in range(5):
            weights_path = Path(self.temp_dir) / f"weights_{i}.json"
            with open(weights_path, "w") as f:
                json.dump({"weights": {}, "fitness": 0.5 + i * 0.1}, f)

            register_cmaes_result(
                weights_path=weights_path,
                board_type="square8",
                num_players=2,
                fitness=0.5 + i * 0.1,
                generation=i * 10,
                registry_dir=self.registry_dir,
                auto_promote=False,
            )

        runs = list_cmaes_runs(
            board_type="square8",
            num_players=2,
            registry_dir=self.registry_dir,
            limit=3,
        )

        assert len(runs) == 3

    def test_filter_by_board(self):
        """Test filtering runs by board type."""
        for board in ["square8", "hex8"]:
            weights_path = Path(self.temp_dir) / f"weights_{board}.json"
            with open(weights_path, "w") as f:
                json.dump({"weights": {}}, f)

            register_cmaes_result(
                weights_path=weights_path,
                board_type=board,
                num_players=2,
                fitness=0.70,
                generation=50,
                registry_dir=self.registry_dir,
                auto_promote=False,
            )

        square_runs = list_cmaes_runs(
            board_type="square8",
            registry_dir=self.registry_dir,
        )
        hex_runs = list_cmaes_runs(
            board_type="hex8",
            registry_dir=self.registry_dir,
        )

        assert len(square_runs) == 1
        assert len(hex_runs) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
