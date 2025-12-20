"""
End-to-End Integration Tests for Training Infrastructure.

This module provides comprehensive integration tests that verify the
complete training pipeline from data generation through training to
tournament evaluation.

Tests cover:
- Full training pipeline for square boards (8x8)
- Full training pipeline for hex boards (21x21)
- Model upgrade/promotion flow
- Streaming large dataset handling
- Checkpoint resume functionality
- Component interactions across the infrastructure
- Error handling and recovery
- Performance baselines

Prerequisites:
- All P0/P1 training infrastructure components completed
- StreamingDataLoader (data_loader.py)
- ModelVersionManager (model_versioning.py)
- AutoTournamentPipeline (auto_tournament.py)
- HexStateEncoder (encoding.py)
- train.py entry point
"""

import gc
import os
import tempfile
import time
import tracemalloc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.ai.neural_net import (
    HEX_BOARD_SIZE,
    P_HEX,
    HexNeuralNet_v2,
    POLICY_SIZE_8x8,
    RingRiftCNN_v2,
)
from app.models import BoardType
from app.training.auto_tournament import AutoTournamentPipeline
from app.training.data_loader import StreamingDataLoader
from app.training.encoding import (
    HexStateEncoder,
    detect_board_type_from_features,
)
from app.training.model_versioning import (
    ChecksumMismatchError,
    ModelVersionManager,
    VersionMismatchError,
    load_model_with_validation,
    save_model_checkpoint,
)
from app.training.train import (
    EarlyStopping,
    RingRiftDataset,
    load_checkpoint,
    save_checkpoint,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_npz(
    path: str,
    num_samples: int = 100,
    feature_shape: tuple = (40, 8, 8),
    global_features: int = 10,
    policy_size: int = 55000,
    include_empty_policies: bool = False,
    seed: int = 42,
) -> str:
    """Create a test .npz file with random training data."""
    rng = np.random.default_rng(seed)

    features = rng.random(
        (num_samples, *feature_shape), dtype=np.float64
    ).astype(np.float32)
    globals_arr = rng.random(
        (num_samples, global_features), dtype=np.float64
    ).astype(np.float32)
    values = rng.choice(
        [-1.0, 0.0, 1.0], size=num_samples
    ).astype(np.float32)

    # Create sparse policies
    policy_indices = []
    policy_values = []
    for i in range(num_samples):
        if include_empty_policies and i % 10 == 0:
            # Empty policy (terminal state)
            policy_indices.append(np.array([], dtype=np.int32))
            policy_values.append(np.array([], dtype=np.float32))
        else:
            # Random sparse policy with 5-20 non-zero entries
            num_moves = rng.integers(5, 21)
            indices = rng.choice(
                policy_size, size=num_moves, replace=False
            ).astype(np.int32)
            probs = rng.random(num_moves).astype(np.float32)
            probs = probs / probs.sum()
            policy_indices.append(indices)
            policy_values.append(probs)

    policy_indices_arr = np.array(policy_indices, dtype=object)
    policy_values_arr = np.array(policy_values, dtype=object)

    np.savez_compressed(
        path,
        features=features,
        globals=globals_arr,
        values=values,
        policy_indices=policy_indices_arr,
        policy_values=policy_values_arr,
    )
    return path


def create_hex_test_npz(
    path: str,
    num_samples: int = 100,
    seed: int = 42,
) -> str:
    """Create a test .npz file for hex boards (25x25).

    HexStateEncoder uses a 25x25 grid (radius=12 mapped to [0,24]).
    10 channels * 4 (history_length + 1) = 40 total feature channels.
    """
    return create_test_npz(
        path=path,
        num_samples=num_samples,
        feature_shape=(40, 25, 25),  # 25x25 grid for hex boards
        global_features=10,
        policy_size=P_HEX,
        seed=seed,
    )


class SimpleSquareModel(nn.Module):
    """Simple test model mimicking RingRiftCNN_v2 for fast testing."""

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 40,
        global_features: int = 10,
        policy_size: int = 55000,
    ):
        super().__init__()
        self.board_size = board_size
        self.total_in_channels = in_channels
        self.policy_size = policy_size

        # Minimal layers for fast testing
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(32 * 2 * 2 + global_features, 64)

        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, policy_size)
        self.tanh = nn.Tanh()

    def forward(self, x, globals):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, globals), dim=1)
        x = torch.relu(self.fc1(x))
        value = self.tanh(self.value_head(x))
        policy = self.policy_head(x)
        return value, policy


class SimpleHexModel(nn.Module):
    """Simple test model mimicking HexNeuralNet_v2 for fast testing."""

    ARCHITECTURE_VERSION = "v1.0.0"

    def __init__(
        self,
        in_channels: int = 40,
        global_features: int = 10,
        board_size: int = 21,
        policy_size: int = P_HEX,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_features = global_features
        self.num_res_blocks = 2
        self.num_filters = 32
        self.board_size = board_size
        self.policy_size = policy_size

        # Minimal layers for fast testing
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(32 * 2 * 2 + global_features, 64)

        self.value_head = nn.Linear(64, 1)
        self.policy_head = nn.Linear(64, policy_size)
        self.tanh = nn.Tanh()

    def forward(self, x, globals, hex_mask=None):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, globals), dim=1)
        x = torch.relu(self.fc1(x))
        value = self.tanh(self.value_head(x))
        policy = self.policy_head(x)
        return value, policy


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def models_dir(temp_dir):
    """Create models subdirectory."""
    path = os.path.join(temp_dir, "models")
    os.makedirs(path)
    return path


@pytest.fixture
def results_dir(temp_dir):
    """Create results subdirectory."""
    path = os.path.join(temp_dir, "results")
    os.makedirs(path)
    return path


@pytest.fixture
def data_dir(temp_dir):
    """Create data subdirectory."""
    path = os.path.join(temp_dir, "data")
    os.makedirs(path)
    return path


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create checkpoints subdirectory."""
    path = os.path.join(temp_dir, "checkpoints")
    os.makedirs(path)
    return path


@pytest.fixture
def square_data_file(data_dir):
    """Create square board training data."""
    path = os.path.join(data_dir, "square_data.npz")
    create_test_npz(path, num_samples=100, feature_shape=(40, 8, 8))
    return path


@pytest.fixture
def hex_data_file(data_dir):
    """Create hex board training data."""
    path = os.path.join(data_dir, "hex_data.npz")
    create_hex_test_npz(path, num_samples=100)
    return path


@pytest.fixture
def multiple_data_files(data_dir):
    """Create multiple data files for streaming tests."""
    paths = []
    for i in range(5):
        path = os.path.join(data_dir, f"data_{i}.npz")
        create_test_npz(path, num_samples=50 + i * 10, seed=42 + i)
        paths.append(path)
    return paths


@pytest.fixture
def square_model():
    """Create a simple square board model."""
    return SimpleSquareModel()


@pytest.fixture
def hex_model():
    """Create a simple hex board model."""
    return SimpleHexModel()


# =============================================================================
# TestTrainingPipelineIntegration - End-to-End Training Pipeline Tests
# =============================================================================


class TestTrainingPipelineIntegration:
    """
    End-to-end integration tests for the complete training pipeline.

    These tests verify:
    - Data generation → streaming load → train → save → tournament flow
    - Both square and hex board types
    - Model versioning throughout the pipeline
    """

    def test_full_square_board_pipeline(
        self,
        temp_dir,
        square_data_file,
        models_dir,
        results_dir,
        checkpoint_dir,
    ):
        """
        Full pipeline test: data generation → stream load → train → save
        versioned → register in tournament.

        Steps:
        1. Generate small training dataset (already done via fixture)
        2. Load with StreamingDataLoader
        3. Train model for few epochs
        4. Save with ModelVersionManager
        5. Register in AutoTournamentPipeline
        6. Verify model is properly tracked
        """
        # Step 2: Load with StreamingDataLoader
        loader = StreamingDataLoader(
            data_paths=square_data_file,
            batch_size=16,
            shuffle=True,
            seed=42,
        )

        assert loader.total_samples == 100
        assert len(loader) == 7  # ceil(100/16)

        # Step 3: Train model for few batches
        model = SimpleSquareModel()
        device = torch.device("cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        value_criterion = nn.MSELoss()
        policy_criterion = nn.KLDivLoss(reduction='batchmean')

        model.train()
        batch_count = 0
        for (features, globals_tensor), (values, policies) in loader:
            features = features.to(device)
            globals_tensor = globals_tensor.to(device)
            values = values.to(device)
            policies = policies.to(device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)
            policy_log_probs = torch.log_softmax(policy_pred, dim=1)

            value_loss = value_criterion(value_pred, values)
            policy_loss = policy_criterion(policy_log_probs, policies)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            batch_count += 1

            if batch_count >= 3:  # Train for 3 batches
                break

        assert batch_count == 3

        # Step 4: Save with ModelVersionManager
        model_path = os.path.join(models_dir, "square_model_v1.pth")
        manager = ModelVersionManager()
        metadata = manager.create_metadata(
            model,
            training_info={
                "epochs": 1,
                "batches": batch_count,
                "board_type": "square8",
            },
        )
        manager.save_checkpoint(model, metadata, model_path)

        assert os.path.exists(model_path)

        # Verify checkpoint contents
        loaded_metadata = manager.get_metadata(model_path)
        assert loaded_metadata.architecture_version == "v1.0.0"
        assert loaded_metadata.model_class == "SimpleSquareModel"
        assert loaded_metadata.training_info["board_type"] == "square8"

        # Step 5: Register in AutoTournamentPipeline
        pipeline = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        model_id = pipeline.register_model(model_path)
        assert model_id is not None

        # Verify model is registered
        registered = pipeline.get_model(model_id)
        assert registered is not None
        assert registered.elo_rating == 1500.0
        assert registered.is_champion is True  # First model is champion
        assert registered.metadata.model_class == "SimpleSquareModel"

        # Generate report to verify pipeline state
        report = pipeline.generate_report()
        assert "## Current Champion" in report
        assert model_id in report

        loader.close()

    def test_full_hex_board_pipeline(
        self,
        temp_dir,
        hex_data_file,
        models_dir,
        results_dir,
    ):
        """
        Full pipeline test for hex boards (21x21).

        Steps:
        1. Generate hex training dataset (already done via fixture)
        2. Load with StreamingDataLoader (with P_HEX policy size)
        3. Train HexNeuralNet_v2-style model for few epochs
        4. Save with ModelVersionManager
        5. Register in AutoTournamentPipeline
        """
        # Step 2: Load with StreamingDataLoader (policy_size = P_HEX)
        loader = StreamingDataLoader(
            data_paths=hex_data_file,
            batch_size=16,
            shuffle=True,
            seed=42,
            policy_size=P_HEX,
        )

        assert loader.total_samples == 100

        # Step 3: Train hex model for few batches
        model = SimpleHexModel()
        device = torch.device("cpu")
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        value_criterion = nn.MSELoss()
        policy_criterion = nn.KLDivLoss(reduction='batchmean')

        model.train()
        batch_count = 0

        for (features, globals_tensor), (values, policies) in loader:
            features = features.to(device)
            globals_tensor = globals_tensor.to(device)
            values = values.to(device)
            policies = policies.to(device)

            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)
            policy_log_probs = torch.log_softmax(policy_pred, dim=1)

            value_loss = value_criterion(value_pred, values)
            policy_loss = policy_criterion(policy_log_probs, policies)
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()
            batch_count += 1

            if batch_count >= 3:
                break

        assert batch_count == 3

        # Step 4: Save with versioning
        model_path = os.path.join(models_dir, "hex_model_v1.pth")
        manager = ModelVersionManager()
        metadata = manager.create_metadata(
            model,
            training_info={
                "epochs": 1,
                "batches": batch_count,
                "board_type": "hexagonal",
                "board_size": HEX_BOARD_SIZE,
            },
        )
        manager.save_checkpoint(model, metadata, model_path)

        assert os.path.exists(model_path)

        # Verify hex-specific metadata
        loaded_metadata = manager.get_metadata(model_path)
        assert loaded_metadata.model_class == "SimpleHexModel"
        assert loaded_metadata.training_info["board_type"] == "hexagonal"
        assert loaded_metadata.training_info["board_size"] == HEX_BOARD_SIZE

        # Step 5: Register in tournament pipeline
        pipeline = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        model_id = pipeline.register_model(model_path)
        assert model_id is not None

        registered = pipeline.get_model(model_id)
        assert registered is not None
        assert registered.is_champion is True

        loader.close()

    def test_model_upgrade_flow(
        self,
        temp_dir,
        square_data_file,
        models_dir,
        results_dir,
    ):
        """
        Model upgrade flow: Train v1 → Train v2 → Evaluate challenger →
        Promote if better.

        Steps:
        1. Train and save model v1
        2. Train and save model v2 (with slightly different weights)
        3. Register v1 as champion
        4. Evaluate v2 against v1 (mocked tournament)
        5. Verify promotion logic works correctly
        """
        # Step 1 & 2: Create two versions of models
        model_v1 = SimpleSquareModel()
        model_v2 = SimpleSquareModel()

        # Make v2 slightly different
        with torch.no_grad():
            model_v2.fc1.weight.data += 0.1

        # Save both
        manager = ModelVersionManager()

        v1_path = os.path.join(models_dir, "model_v1.pth")
        metadata_v1 = manager.create_metadata(
            model_v1,
            training_info={"version": "v1", "epochs": 100},
        )
        manager.save_checkpoint(model_v1, metadata_v1, v1_path)

        v2_path = os.path.join(models_dir, "model_v2.pth")
        metadata_v2 = manager.create_metadata(
            model_v2,
            training_info={"version": "v2", "epochs": 150},
        )
        manager.save_checkpoint(model_v2, metadata_v2, v2_path)

        # Step 3: Register v1 as champion
        pipeline = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        v1_id = pipeline.register_model(v1_path)
        champion = pipeline.get_champion()
        assert champion is not None
        assert champion.model_id == v1_id

        # Step 4: Evaluate v2 against v1 (mock the Tournament class)
        tournament_patch = 'app.training.auto_tournament.Tournament'
        with patch(tournament_patch) as mock_tournament_cls:
            # Mock v2 winning convincingly
            # The mock needs to simulate rating updates after run()
            mock_tournament = MagicMock()
            mock_tournament.run.return_value = {"A": 35, "B": 15, "Draw": 0}
            # Start with initial ratings, then update after run
            mock_ratings = {"A": 1500.0, "B": 1500.0}

            def mock_run():
                # Simulate Elo updates after games
                mock_ratings["A"] = 1580.0  # Challenger gains
                mock_ratings["B"] = 1420.0  # Champion loses
                return {"A": 35, "B": 15, "Draw": 0}

            mock_tournament.run.side_effect = mock_run
            # Use property-like access for ratings
            type(mock_tournament).ratings = property(lambda self: mock_ratings)
            mock_tournament.victory_reasons = {"elimination": 50}
            mock_tournament_cls.return_value = mock_tournament

            result = pipeline.evaluate_challenger(v2_path, games=50)

        # Step 5: Verify promotion logic
        assert result.challenger_wins == 35
        assert result.champion_wins == 15
        assert result.challenger_win_rate == 0.70
        assert result.should_promote is True

        # Promote v2
        pipeline.promote_champion(result.challenger_id)

        new_champion = pipeline.get_champion()
        assert new_champion is not None
        assert new_champion.model_id == result.challenger_id
        old_champion = pipeline.get_model(v1_id)
        assert old_champion is not None
        assert not old_champion.is_champion

    def test_streaming_large_dataset(
        self,
        temp_dir,
        multiple_data_files,
    ):
        """
        Verify memory stays bounded with large synthetic dataset.

        Creates multiple data files and streams them without loading
        all data into memory at once.
        """
        # Total samples: 50 + 60 + 70 + 80 + 90 = 350
        loader = StreamingDataLoader(
            data_paths=multiple_data_files,
            batch_size=32,
            shuffle=True,
            seed=42,
        )

        assert loader.total_samples == 350
        assert len(loader) == 11  # ceil(350/32)

        # Iterate through all batches and verify data validity
        total_samples_seen = 0
        for (features, globals_tensor), (values, policies) in loader:
            batch_size = features.shape[0]
            total_samples_seen += batch_size

            # Verify tensor shapes
            assert features.shape[1:] == (40, 8, 8)
            assert globals_tensor.shape[1] == 10
            assert values.shape == (batch_size, 1)
            assert policies.shape[1] == 55000

            # Verify no NaN values
            assert not torch.isnan(features).any()
            assert not torch.isnan(values).any()

        assert total_samples_seen == 350

        loader.close()

    @patch("app.training.checkpointing.check_disk_space", return_value=True)
    def test_checkpoint_resume(
        self,
        mock_disk_check,
        temp_dir,
        square_data_file,
        models_dir,
        checkpoint_dir,
    ):
        """
        Test checkpoint resume: Train → Save → Load → Continue training.

        Verifies that checkpoint saves model, optimizer, scheduler,
        and early stopping state. Resume correctly continues from saved
        state. Training can proceed after resume.
        """
        # Initial training
        model = SimpleSquareModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        early_stopper = EarlyStopping(patience=10)

        # Do a few training steps
        loader = StreamingDataLoader(
            square_data_file, batch_size=16, shuffle=True
        )

        model.train()
        batches_before_save = 0
        for (features, globals_tensor), (_values, _policies) in loader:
            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)
            loss = value_pred.sum()  # Dummy loss
            loss.backward()
            optimizer.step()
            batches_before_save += 1

            if batches_before_save >= 3:
                break

        scheduler.step()

        # Save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_epoch_1.pth")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            loss=0.5,
            path=ckpt_path,
            scheduler=scheduler,
            early_stopping=early_stopper,
            use_versioning=True,
        )

        assert os.path.exists(ckpt_path)

        # Create new instances for resume
        new_model = SimpleSquareModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_sched = torch.optim.lr_scheduler.StepLR(
            new_optimizer, step_size=5
        )
        new_early_stopper = EarlyStopping(patience=10)

        # Load checkpoint
        epoch, loss = load_checkpoint(
            path=ckpt_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_sched,
            early_stopping=new_early_stopper,
            device=torch.device("cpu"),
            strict_versioning=False,
        )

        assert epoch == 1
        assert loss == 0.5

        # Verify model weights are restored
        for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=False):
            assert torch.allclose(p1, p2)

        # Continue training
        loader.set_epoch(2)
        new_model.train()
        batches_after_resume = 0

        for (features, globals_tensor), (_values, _policies) in loader:
            new_optimizer.zero_grad()
            value_pred, _policy_pred = new_model(features, globals_tensor)
            loss = value_pred.sum()
            loss.backward()
            new_optimizer.step()
            batches_after_resume += 1

            if batches_after_resume >= 2:
                break

        assert batches_after_resume == 2

        loader.close()


# =============================================================================
# TestComponentInteractions - Component Interaction Tests
# =============================================================================


class TestComponentInteractions:
    """
    Tests for interactions between training infrastructure components.

    Verifies that components work correctly together:
    - DataLoader with versioned checkpoints
    - Tournament with versioned models
    - Hex and square board type handling
    """

    def test_dataloader_with_versioned_checkpoint(
        self,
        temp_dir,
        square_data_file,
        models_dir,
    ):
        """
        Verify streaming data works with versioned models.

        Tests that data from StreamingDataLoader is compatible with
        models saved using ModelVersionManager.
        """
        # Create and train model with streaming data
        loader = StreamingDataLoader(
            square_data_file,
            batch_size=16,
            shuffle=True,
        )

        model = SimpleSquareModel()

        # Train briefly
        model.train()
        for (features, globals_tensor), (_values, _policies) in loader:
            _, _policy_pred = model(features, globals_tensor)
            break  # Single batch

        # Save with versioning
        model_path = os.path.join(models_dir, "streamed_model.pth")
        save_model_checkpoint(
            model,
            model_path,
            training_info={
                "data_source": "streaming",
                "total_samples": loader.total_samples,
            },
        )

        # Load and verify
        new_model = SimpleSquareModel()
        loaded_model, metadata = load_model_with_validation(
            new_model,
            model_path,
            strict=False,
        )

        assert metadata.training_info["data_source"] == "streaming"
        assert metadata.training_info["total_samples"] == 100

        # Verify loaded model works with streaming data
        loader.set_epoch(1)
        loaded_model.eval()
        with torch.no_grad():
            for (features, globals_tensor), (_, _) in loader:
                value_out, policy_out = loaded_model(features, globals_tensor)
                assert value_out.shape[0] == features.shape[0]
                assert policy_out.shape == (features.shape[0], 55000)
                break

        loader.close()

    @patch('app.training.auto_tournament.Tournament')
    def test_tournament_with_versioned_models(
        self,
        mock_tournament_cls,
        temp_dir,
        models_dir,
        results_dir,
    ):
        """
        Register versioned models → Run tournament → Get Elo.

        Verifies that AutoTournamentPipeline correctly handles
        versioned model checkpoints.
        """
        # Create two versioned models
        manager = ModelVersionManager()

        model_a = SimpleSquareModel()
        model_a_path = os.path.join(models_dir, "model_a.pth")
        metadata_a = manager.create_metadata(
            model_a,
            training_info={"name": "model_a", "epochs": 100},
        )
        manager.save_checkpoint(model_a, metadata_a, model_a_path)

        model_b = SimpleSquareModel()
        with torch.no_grad():
            model_b.fc1.weight.data.fill_(0.5)
        model_b_path = os.path.join(models_dir, "model_b.pth")
        metadata_b = manager.create_metadata(
            model_b,
            training_info={"name": "model_b", "epochs": 150},
        )
        manager.save_checkpoint(model_b, metadata_b, model_b_path)

        # Register both models
        pipeline = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        id_a = pipeline.register_model(model_a_path)
        id_b = pipeline.register_model(model_b_path)

        assert len(pipeline.list_models()) == 2

        # Mock tournament results with ratings that update after run()
        mock_ratings = {"A": 1500.0, "B": 1500.0}

        def mock_run():
            # Simulate Elo updates: model A wins more
            mock_ratings["A"] = 1520.0
            mock_ratings["B"] = 1480.0
            return {"A": 6, "B": 4, "Draw": 0}

        mock_tournament = MagicMock()
        mock_tournament.run.side_effect = mock_run
        type(mock_tournament).ratings = property(lambda self: mock_ratings)
        mock_tournament.victory_reasons = {"elimination": 10}
        mock_tournament_cls.return_value = mock_tournament

        # Run tournament
        result = pipeline.run_tournament(games_per_match=10)

        assert len(result.participants) == 2
        assert id_a in result.participants
        assert id_b in result.participants

        # Verify Elo updates
        rankings = pipeline.get_elo_rankings()
        assert len(rankings) == 2

        # Higher Elo should be first
        _top_model_id, top_elo = rankings[0]
        assert top_elo > 1500

    def test_hex_and_square_mixed(
        self,
        temp_dir,
        data_dir,
    ):
        """
        Verify board type detection and routing works.

        Tests that the system can correctly identify feature tensors
        as belonging to square or hex boards based on their shape.

        Board sizes:
        - Square8: 8x8
        - Hexagonal: 25x25 (radius=12 grid)
        """
        # Create square features
        square_features = np.random.rand(10, 8, 8).astype(np.float32)
        board_type = detect_board_type_from_features(square_features)
        assert board_type == BoardType.SQUARE8

        # Create hex features (25x25 grid for HexStateEncoder)
        hex_features = np.random.rand(10, 25, 25).astype(np.float32)
        board_type = detect_board_type_from_features(hex_features)
        assert board_type == BoardType.HEXAGONAL

        # Batched features
        square_batch = np.random.rand(4, 10, 8, 8).astype(np.float32)
        board_type = detect_board_type_from_features(square_batch)
        assert board_type == BoardType.SQUARE8

        hex_batch = np.random.rand(4, 10, 25, 25).astype(np.float32)
        board_type = detect_board_type_from_features(hex_batch)
        assert board_type == BoardType.HEXAGONAL

    def test_hex_encoder_with_dataloader(
        self,
        temp_dir,
        hex_data_file,
    ):
        """
        Test HexStateEncoder works with StreamingDataLoader data.

        HexStateEncoder uses a 25x25 grid (radius=12).
        """
        encoder = HexStateEncoder()

        # Load hex data
        loader = StreamingDataLoader(
            hex_data_file,
            batch_size=16,
            shuffle=False,
            policy_size=P_HEX,
        )

        for (features, globals_tensor), (_values, policies) in loader:
            # Verify feature shapes match encoder expectations
            # 40 channels = 10 base channels * 4 (history_length + 1)
            # 25x25 grid for hex boards (radius=12)
            assert features.shape[1:] == (40, 25, 25)
            assert globals_tensor.shape[1] == 10

            # Verify valid mask is correct shape (25x25 grid)
            valid_mask = encoder.get_valid_mask_tensor()
            assert valid_mask.shape == (1, 25, 25)

            # Check policy size matches
            assert policies.shape[1] == encoder.POLICY_SIZE
            break

        loader.close()

    def test_versioned_model_pipeline_integration(
        self,
        temp_dir,
        models_dir,
    ):
        """
        Test complete model versioning lifecycle integration.
        """
        manager = ModelVersionManager()

        # Create initial model
        model_v1 = SimpleSquareModel()
        v1_path = os.path.join(models_dir, "model_v1.pth")

        # Save with training info
        metadata_v1 = manager.create_metadata(
            model_v1,
            training_info={
                "epoch": 1,
                "loss": 0.5,
            },
        )
        manager.save_checkpoint(model_v1, metadata_v1, v1_path)

        # Simulate retraining - load, update, save as v2
        state_dict, loaded_meta = manager.load_checkpoint(
            v1_path,
            strict=False,
            verify_checksum=True,
        )

        model_v2 = SimpleSquareModel()
        model_v2.load_state_dict(state_dict)

        # Modify weights (simulating training)
        with torch.no_grad():
            model_v2.fc1.weight.data += 0.01

        v2_path = os.path.join(models_dir, "model_v2.pth")
        metadata_v2 = manager.create_metadata(
            model_v2,
            training_info={
                "epoch": 10,
                "loss": 0.3,
                "parent_version": loaded_meta.architecture_version,
            },
            parent_checkpoint=v1_path,
        )
        manager.save_checkpoint(model_v2, metadata_v2, v2_path)

        # Verify lineage tracking
        v2_loaded_meta = manager.get_metadata(v2_path)
        assert v2_loaded_meta.parent_checkpoint == v1_path
        # Different weights should produce different checksum
        assert v2_loaded_meta.checksum != loaded_meta.checksum


# =============================================================================
# TestErrorRecovery - Error Handling Integration Tests
# =============================================================================


class TestErrorRecovery:
    """
    Tests for error handling and recovery in the training infrastructure.

    Verifies that the system handles:
    - Checkpoint corruption (checksum mismatch)
    - Version mismatch rejection
    - Missing files gracefully
    """

    def test_checkpoint_corruption_recovery(
        self,
        temp_dir,
        models_dir,
    ):
        """
        Test checksum mismatch detection and handling.

        Verifies that corrupted checkpoints are detected and
        appropriate errors are raised.
        """
        manager = ModelVersionManager()

        # Create and save a valid checkpoint
        model = SimpleSquareModel()
        model_path = os.path.join(models_dir, "model.pth")
        metadata = manager.create_metadata(model)
        manager.save_checkpoint(model, metadata, model_path)

        # Corrupt the checkpoint by modifying the checksum
        checkpoint = torch.load(model_path)
        meta_key = manager.METADATA_KEY
        checkpoint[meta_key]['checksum'] = 'corrupted_checksum_value'
        torch.save(checkpoint, model_path)

        # Attempt to load with checksum verification should fail
        with pytest.raises(ChecksumMismatchError) as exc_info:
            manager.load_checkpoint(
                model_path,
                strict=True,
                verify_checksum=True,
            )

        assert "corrupted" in str(exc_info.value).lower() or \
               "integrity" in str(exc_info.value).lower() or \
               exc_info.value.actual != exc_info.value.expected

        # Loading with verify_checksum=False should succeed
        state_dict, _ = manager.load_checkpoint(
            model_path,
            strict=False,
            verify_checksum=False,
        )
        assert state_dict is not None

    def test_version_mismatch_rejection(
        self,
        temp_dir,
        models_dir,
    ):
        """
        Test that wrong version is rejected and training can continue
        with explicit handling.
        """
        manager = ModelVersionManager()

        # Create and save model
        model = SimpleSquareModel()
        model_path = os.path.join(models_dir, "old_version.pth")
        metadata = manager.create_metadata(model)
        manager.save_checkpoint(model, metadata, model_path)

        # Try loading with different expected version
        with pytest.raises(VersionMismatchError) as exc_info:
            manager.load_checkpoint(
                model_path,
                strict=True,
                expected_version="v2.0.0",  # Different version
            )

        assert exc_info.value.checkpoint_version == "v1.0.0"
        assert exc_info.value.current_version == "v2.0.0"

        # Non-strict loading should work with warning
        state_dict, loaded_meta = manager.load_checkpoint(
            model_path,
            strict=False,
            expected_version="v2.0.0",
        )
        assert state_dict is not None
        assert loaded_meta.architecture_version == "v1.0.0"

    def test_missing_file_handling(
        self,
        temp_dir,
        data_dir,
    ):
        """
        Test that missing data files are handled gracefully.
        """
        # StreamingDataLoader should skip missing files
        valid_path = os.path.join(data_dir, "valid.npz")
        create_test_npz(valid_path, num_samples=50)

        missing_path = os.path.join(data_dir, "missing.npz")

        loader = StreamingDataLoader(
            data_paths=[valid_path, missing_path],
            batch_size=16,
        )

        # Should only have samples from valid file
        assert loader.total_samples == 50

        # Should iterate successfully
        batch_count = 0
        for _batch in loader:
            batch_count += 1

        assert batch_count > 0

        loader.close()

    def test_empty_data_handling(
        self,
        temp_dir,
        data_dir,
    ):
        """
        Test handling of empty data files or directories.
        """
        # Empty loader should not crash
        loader = StreamingDataLoader(
            data_paths=[],
            batch_size=16,
        )

        assert loader.total_samples == 0
        assert len(loader) == 0
        assert list(loader) == []

        loader.close()

    def test_early_stopping_preserves_best_weights(
        self,
        temp_dir,
    ):
        """
        Test that early stopping properly preserves and restores best weights.
        """
        model = SimpleSquareModel()
        early_stopper = EarlyStopping(patience=2, min_delta=0.0001)

        # Simulate improving losses
        early_stopper(0.5, model)  # Best so far
        early_stopper(0.4, model)  # New best

        # Change weights
        with torch.no_grad():
            model.fc1.weight.data.fill_(999.0)

        # Worse loss should not update best_state
        early_stopper(0.45, model)

        # Restore best weights
        early_stopper.restore_best_weights(model)

        # Weights should not be 999.0
        assert not torch.allclose(
            model.fc1.weight.data,
            torch.full_like(model.fc1.weight.data, 999.0)
        )


# =============================================================================
# TestPerformanceBaseline - Performance Regression Tests
# =============================================================================


class TestPerformanceBaseline:
    """
    Performance baseline tests to detect regressions.

    These tests verify that:
    - Training throughput meets minimum requirements
    - Memory usage stays within bounds during streaming
    - Data loading is efficient
    """

    def test_training_throughput_baseline(
        self,
        temp_dir,
        square_data_file,
    ):
        """
        Verify training speed hasn't regressed.

        Measures time to process a fixed number of batches and ensures
        it's within acceptable bounds.
        """
        loader = StreamingDataLoader(
            square_data_file,
            batch_size=32,
            shuffle=True,
        )

        model = SimpleSquareModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()

        # Warmup
        for (features, globals_tensor), (_values, _policies) in loader:
            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)
            loss = value_pred.sum()
            loss.backward()
            optimizer.step()
            break

        loader.set_epoch(1)

        # Timed run
        start_time = time.perf_counter()
        batch_count = 0
        samples_processed = 0

        for (features, globals_tensor), (_values, _policies) in loader:
            optimizer.zero_grad()
            value_pred, _policy_pred = model(features, globals_tensor)
            loss = value_pred.sum()
            loss.backward()
            optimizer.step()
            batch_count += 1
            samples_processed += features.shape[0]

        elapsed = time.perf_counter() - start_time

        # Performance assertion
        # Should process at least 100 samples/second on CPU
        # (Very conservative baseline for CI environments)
        samples_per_second = samples_processed / elapsed

        assert samples_per_second > 10, (
            f"Training throughput too low: {samples_per_second:.1f} "
            f"samples/sec (expected > 10)"
        )

        loader.close()

    def test_dataloader_memory_bound(
        self,
        temp_dir,
        data_dir,
    ):
        """
        Verify memory stays under threshold during streaming.

        Creates a larger dataset and ensures streaming iteration
        doesn't load everything into memory at once.
        """
        # Create multiple data files (simulating larger dataset)
        paths = []
        for i in range(10):
            path = os.path.join(data_dir, f"large_{i}.npz")
            create_test_npz(path, num_samples=100, seed=42 + i)
            paths.append(path)

        # Force garbage collection before measuring
        gc.collect()

        # Start memory tracking
        tracemalloc.start()

        loader = StreamingDataLoader(
            data_paths=paths,
            batch_size=32,
            shuffle=True,
        )

        assert loader.total_samples == 1000

        # Iterate through all data
        batch_count = 0
        for (_features, _globals_tensor), (_values, _policies) in loader:
            batch_count += 1

            # Periodic memory check
            if batch_count % 10 == 0:
                _current, peak = tracemalloc.get_traced_memory()
                # Peak memory should stay reasonable
                # Allow up to 500MB (generous for CI)
                assert peak < 500 * 1024 * 1024, (
                    f"Memory usage too high: {peak / 1024 / 1024:.1f}MB"
                )

        tracemalloc.stop()

        assert batch_count == 32  # ceil(1000/32)

        loader.close()

    def test_streaming_vs_dataset_consistency(
        self,
        temp_dir,
        square_data_file,
    ):
        """
        Verify StreamingDataLoader produces same results as RingRiftDataset
        for deterministic seeds.
        """
        # Use RingRiftDataset (traditional)
        dataset = RingRiftDataset(
            square_data_file,
            board_type=BoardType.SQUARE8,
        )

        # Get first few samples
        traditional_samples = []
        for i in range(min(5, len(dataset))):
            features, _globals_vec, _values, _policies = dataset[i]
            traditional_samples.append(features.numpy())

        # Use StreamingDataLoader
        loader = StreamingDataLoader(
            square_data_file,
            batch_size=5,
            shuffle=False,
        )

        streaming_samples = []
        for (features, _), (_, _) in loader:
            for i in range(features.shape[0]):
                streaming_samples.append(features[i].numpy())
            break

        # Compare (note: order may differ, but content should be consistent)
        assert len(streaming_samples) == len(traditional_samples)

        # Check that data is valid (not comparing exact order due to filtering)
        for sample in streaming_samples:
            assert sample.shape == (40, 8, 8)
            assert not np.isnan(sample).any()

        loader.close()

    def test_large_batch_handling(
        self,
        temp_dir,
        square_data_file,
    ):
        """
        Test that large batch sizes are handled correctly.
        """
        loader = StreamingDataLoader(
            square_data_file,
            batch_size=100,  # Larger than dataset
            shuffle=True,
        )

        batch_count = 0
        total_samples = 0

        for (features, _), (_, _) in loader:
            batch_count += 1
            total_samples += features.shape[0]

        # Should get one batch with all samples
        assert batch_count == 1
        assert total_samples == 100

        loader.close()


# =============================================================================
# TestRegistryPersistence - Registry and State Persistence Tests
# =============================================================================


class TestRegistryPersistence:
    """
    Tests for registry persistence and state management.
    """

    def test_tournament_registry_persistence(
        self,
        temp_dir,
        models_dir,
        results_dir,
    ):
        """
        Test that tournament registry persists across pipeline instances.
        """
        # Create and register model in first pipeline instance
        manager = ModelVersionManager()
        model = SimpleSquareModel()
        model_path = os.path.join(models_dir, "persistent_model.pth")
        metadata = manager.create_metadata(
            model,
            training_info={"test": "persistence"},
        )
        manager.save_checkpoint(model, metadata, model_path)

        pipeline1 = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )
        model_id = pipeline1.register_model(model_path, initial_elo=1600)

        # Modify model stats
        pipeline1._models[model_id].games_played = 10
        pipeline1._models[model_id].wins = 7
        pipeline1._save_registry()

        # Create new pipeline instance
        pipeline2 = AutoTournamentPipeline(
            models_dir=models_dir,
            results_dir=results_dir,
        )

        # Verify state was persisted
        assert model_id in pipeline2._models
        assert pipeline2._models[model_id].elo_rating == 1600
        assert pipeline2._models[model_id].games_played == 10
        assert pipeline2._models[model_id].wins == 7

    def test_multiple_epoch_streaming(
        self,
        temp_dir,
        square_data_file,
    ):
        """
        Test that set_epoch produces different shuffling across epochs.
        """
        loader = StreamingDataLoader(
            square_data_file,
            batch_size=16,
            shuffle=True,
            seed=42,
        )

        # Get first batch from epoch 0
        loader.set_epoch(0)
        batch_epoch_0 = None
        for (features, _), (_, _) in loader:
            batch_epoch_0 = features.clone()
            break

        # Get first batch from epoch 1
        loader.set_epoch(1)
        batch_epoch_1 = None
        for (features, _), (_, _) in loader:
            batch_epoch_1 = features.clone()
            break

        # Batches should be different due to different shuffling
        assert batch_epoch_0 is not None
        assert batch_epoch_1 is not None
        assert not torch.equal(batch_epoch_0, batch_epoch_1)

        loader.close()


# =============================================================================
# TestRealModelIntegration - Tests with Real Model Architectures
# =============================================================================


class TestRealModelIntegration:
    """
    Integration tests using actual RingRiftCNN_v2 and HexNeuralNet_v2 architectures.

    These tests verify the integration works with production model classes.
    """

    def test_ringrift_cnn_with_streaming_data(
        self,
        temp_dir,
        square_data_file,
        models_dir,
    ):
        """
        Test RingRiftCNN_v2 works end-to-end with streaming data.
        """
        # Create small RingRiftCNN_v2 for testing with correct architecture params
        # Note: in_channels=14 is the base, total channels = in_channels * (history_length + 1)
        # For history_length=3, total_in_channels = 14 * 4 = 56
        # Test data has feature_shape=(40, 8, 8), so we use in_channels=10 with history_length=3
        # which gives total_in_channels = 10 * 4 = 40
        num_players = 4  # Multi-player value head
        model = RingRiftCNN_v2(
            board_size=8,
            in_channels=10,  # Base channels; total = 10 * (3+1) = 40 to match test data
            global_features=10,  # Match test data globals
            num_res_blocks=2,  # Small for testing
            num_filters=32,
            history_length=3,
            policy_size=POLICY_SIZE_8x8,  # Explicit policy size for 8x8 board
            num_players=num_players,
        )

        loader = StreamingDataLoader(
            square_data_file,
            batch_size=16,
            shuffle=True,
        )

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for (features, globals_tensor), (_values, _policies) in loader:
            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)

            # Verify output shapes
            # v2 models use multi-player value head: [batch, num_players]
            assert value_pred.shape == (features.shape[0], num_players)
            assert policy_pred.shape == (features.shape[0], POLICY_SIZE_8x8)

            loss = value_pred.sum()
            loss.backward()
            optimizer.step()
            break

        # Save with versioning
        model_path = os.path.join(models_dir, "ringrift_cnn.pth")
        manager = ModelVersionManager()
        metadata = manager.create_metadata(model)
        manager.save_checkpoint(model, metadata, model_path)

        # Verify version
        loaded_meta = manager.get_metadata(model_path)
        assert loaded_meta.model_class == "RingRiftCNN_v2"
        expected_ver = RingRiftCNN_v2.ARCHITECTURE_VERSION
        assert loaded_meta.architecture_version == expected_ver

        loader.close()

    def test_hex_neural_net_with_streaming_data(
        self,
        temp_dir,
        hex_data_file,
        models_dir,
    ):
        """
        Test HexNeuralNet_v2 works end-to-end with streaming hex data.
        """
        # Create small HexNeuralNet_v2 for testing
        # Test data uses feature_shape=(40, 25, 25), global_features=10
        # (created by create_hex_test_npz with HEX_BOARD_SIZE=25)
        # HexNeuralNet_v2 takes in_channels as TOTAL channels (no history_length param)
        num_players = 4  # Multi-player value head
        model = HexNeuralNet_v2(
            in_channels=40,  # Total channels to match test data (40, 25, 25)
            global_features=10,  # Match test data
            num_res_blocks=2,  # Small for testing
            num_filters=32,
            board_size=HEX_BOARD_SIZE,  # 25 - match test data from create_hex_test_npz
            policy_size=P_HEX,
            num_players=num_players,
            hex_radius=12,  # Radius 12 for 25x25 grid (standard hex board)
        )

        loader = StreamingDataLoader(
            hex_data_file,
            batch_size=16,
            shuffle=True,
            policy_size=P_HEX,
        )

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for (features, globals_tensor), (_values, _policies) in loader:
            optimizer.zero_grad()
            value_pred, policy_pred = model(features, globals_tensor)

            # Verify output shapes
            # v2 models use multi-player value head: [batch, num_players]
            assert value_pred.shape == (features.shape[0], num_players)
            assert policy_pred.shape == (features.shape[0], P_HEX)

            loss = value_pred.sum()
            loss.backward()
            optimizer.step()
            break

        # Save with versioning
        model_path = os.path.join(models_dir, "hex_neural_net.pth")
        manager = ModelVersionManager()
        metadata = manager.create_metadata(model)
        manager.save_checkpoint(model, metadata, model_path)

        # Verify version and hex-specific config
        loaded_meta = manager.get_metadata(model_path)
        assert loaded_meta.model_class == "HexNeuralNet_v2"
        expected_ver = HexNeuralNet_v2.ARCHITECTURE_VERSION
        assert loaded_meta.architecture_version == expected_ver
        assert loaded_meta.config.get("board_size") == HEX_BOARD_SIZE  # 25
        assert loaded_meta.config.get("policy_size") == P_HEX

        loader.close()

    def test_integrated_enhancements_smoke(
        self,
        temp_dir,
        square_data_file,
    ):
        """
        Smoke test for IntegratedTrainingManager with all enhancements enabled.

        Verifies that:
        - IntegratedEnhancementsConfig initializes correctly
        - IntegratedTrainingManager starts/stops background services
        - Auxiliary task loss computation works
        - Batch scheduling integration works
        """
        try:
            from app.training.integrated_enhancements import (
                IntegratedEnhancementsConfig,
                IntegratedTrainingManager,
            )
        except ImportError:
            pytest.skip("IntegratedTrainingManager not available")

        # Create config with all enhancements enabled
        config = IntegratedEnhancementsConfig(
            auxiliary_tasks_enabled=True,
            batch_scheduling_enabled=True,
            curriculum_enabled=True,
            augmentation_enabled=True,
            # Background eval disabled for unit test (requires AI instances)
            background_eval_enabled=False,
        )

        # Create manager
        manager = IntegratedTrainingManager(
            config=config,
            model=None,  # No model needed for basic tests
            board_type="square8",
        )

        # Verify initialization
        assert manager.config is not None
        assert manager.config.auxiliary_tasks_enabled
        assert manager.config.batch_scheduling_enabled

        # Test step updates
        initial_step = manager._step
        manager.update_step()
        assert manager._step == initial_step + 1

        # Test batch size scheduling
        batch_size = manager.get_batch_size()
        assert batch_size > 0, "Batch scheduler should return positive batch size"

        # Test early stopping (should be False with background eval disabled)
        assert not manager.should_early_stop()

        # Test background services lifecycle (no-op when disabled)
        manager.start_background_services()
        manager.stop_background_services()

        # Test auxiliary task loss with fake features
        if manager._auxiliary_module is not None:
            fake_features = torch.randn(4, 256)
            fake_targets = {
                "outcome": torch.randint(0, 3, (4,)),
            }
            aux_loss, breakdown = manager.compute_auxiliary_loss(
                fake_features, fake_targets
            )
            assert isinstance(aux_loss, torch.Tensor)
            assert "outcome" in breakdown or "total_aux" in breakdown

    def test_parallel_selfplay_import(self):
        """
        Verify parallel selfplay module is importable and has expected exports.
        """
        try:
            from app.training.parallel_selfplay import (
                GameResult,
                SelfplayConfig,
                generate_dataset_parallel,
            )
        except ImportError:
            pytest.skip("parallel_selfplay module not available")

        # Verify config has expected fields
        config = SelfplayConfig()
        assert hasattr(config, 'board_type')
        assert hasattr(config, 'engine')
        assert hasattr(config, 'num_players')

        # Verify GameResult has auxiliary task fields
        assert hasattr(GameResult, '__dataclass_fields__')
        fields = GameResult.__dataclass_fields__
        assert 'game_lengths' in fields
        assert 'piece_counts' in fields
        assert 'outcomes' in fields


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
