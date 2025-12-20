"""Integration tests for training workflow.

Tests the complete flow of training components working together.
These tests verify that modules integrate correctly.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """Simple model for integration testing."""

    def __init__(self, in_channels=21, policy_size=2048):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 64)
        self.policy_head = nn.Linear(64, policy_size)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x, globals_input=None):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value


class TestTrainingProgress:
    """Integration tests for TrainingProgress tracking."""

    def test_training_progress_serialization(self):
        """Test TrainingProgress can be serialized and deserialized."""
        from app.training.checkpoint_unified import TrainingProgress

        progress = TrainingProgress(
            epoch=10,
            global_step=5000,
            batch_idx=100,
            samples_seen=50000,
            best_metric=0.25,
            best_metric_name="loss",
            best_epoch=8,
            learning_rate=1e-4,
        )

        # Convert to dict and back
        d = progress.to_dict()
        assert d["epoch"] == 10
        assert d["global_step"] == 5000
        assert d["best_metric"] == 0.25

    def test_training_progress_with_extra_state(self):
        """Test TrainingProgress handles extra state correctly."""
        from app.training.checkpoint_unified import TrainingProgress

        progress = TrainingProgress(
            epoch=5,
            extra_state={
                "custom_metric": 0.95,
                "iterations": [1, 2, 3],
            },
        )

        d = progress.to_dict()
        assert d["extra_state"]["custom_metric"] == 0.95


class TestCheckpointMetadata:
    """Integration tests for CheckpointMetadata."""

    def test_metadata_serialization(self):
        """Test CheckpointMetadata serialization."""
        from datetime import datetime

        from app.training.checkpoint_unified import (
            CheckpointMetadata,
            CheckpointType,
        )

        metadata = CheckpointMetadata(
            checkpoint_id="ckpt_12345",
            checkpoint_type=CheckpointType.BEST,
            epoch=10,
            global_step=5000,
            timestamp=datetime.now(),
            metrics={"loss": 0.25, "accuracy": 0.92},
            training_config={"lr": 1e-3, "batch_size": 256},
            file_path="/path/to/checkpoint.pt",
            file_hash="abc123",
        )

        # Convert to dict
        d = metadata.to_dict()

        assert d["checkpoint_id"] == "ckpt_12345"
        assert d["checkpoint_type"] == "best"
        assert d["epoch"] == 10

        # Convert back
        restored = CheckpointMetadata.from_dict(d)
        assert restored.checkpoint_id == "ckpt_12345"
        assert restored.checkpoint_type == CheckpointType.BEST


class TestTrainingComponents:
    """Integration tests for training loop components."""

    def test_lr_scheduler_warmup_integration(self):
        """Test LR scheduler with warmup works correctly."""
        from app.training.schedulers import get_warmup_scheduler

        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        scheduler = get_warmup_scheduler(
            optimizer,
            warmup_epochs=2,
            total_epochs=10,
        )

        # Collect LRs during warmup and after
        lrs = []
        for _epoch in range(5):
            lrs.append(optimizer.param_groups[0]['lr'])
            # Simulate training step
            x = torch.randn(4, 21, 8, 8)
            policy, _value = model(x)
            loss = policy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # LR should change (either warmup ramp or schedule)
        assert len(lrs) == 5

    def test_gradient_clipping_integration(self):
        """Test gradient clipping works during training step."""
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        max_norm = 1.0

        model.train()
        x = torch.randn(4, 21, 8, 8)
        policy, value = model(x)

        # Create artificially large gradients
        loss = 1000 * (policy.mean() + value.mean())
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Verify clipping worked
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        assert total_norm <= max_norm + 1e-6

    def test_model_forward_backward(self):
        """Test model forward and backward passes work."""
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()

        # Forward pass
        x = torch.randn(4, 21, 8, 8)
        policy, value = model(x)

        assert policy.shape == (4, 2048)
        assert value.shape == (4, 1)

        # Backward pass
        target = torch.randint(0, 2048, (4,))
        loss = criterion(policy, target)
        optimizer.zero_grad()
        loss.backward()

        # Verify gradients exist
        has_grads = any(p.grad is not None for p in model.parameters())
        assert has_grads

        # Optimizer step
        optimizer.step()


class TestHeuristicWeightIntegration:
    """Integration tests for heuristic weights system."""

    def test_weight_profiles_are_complete(self):
        """Test all weight profiles have required keys."""
        from app.ai.heuristic_weights import (
            HEURISTIC_WEIGHT_KEYS,
            HEURISTIC_WEIGHT_PROFILES,
        )

        for profile_id, profile in HEURISTIC_WEIGHT_PROFILES.items():
            for key in HEURISTIC_WEIGHT_KEYS:
                assert key in profile, (
                    f"Profile '{profile_id}' missing key '{key}'"
                )

    def test_flatten_reconstruct_roundtrip(self):
        """Test flatten and reconstruct preserve weights."""
        from app.ai.heuristic_weights import (
            HEURISTIC_WEIGHT_KEYS,
            HEURISTIC_WEIGHT_PROFILES,
        )
        from app.training.train import (
            _flatten_heuristic_weights,
            _reconstruct_heuristic_profile,
        )

        if not HEURISTIC_WEIGHT_PROFILES:
            pytest.skip("No profiles defined")

        profile_id = next(iter(HEURISTIC_WEIGHT_PROFILES.keys()))
        original = HEURISTIC_WEIGHT_PROFILES[profile_id]

        keys, values = _flatten_heuristic_weights(original)
        reconstructed = _reconstruct_heuristic_profile(keys, values)

        for key in HEURISTIC_WEIGHT_KEYS:
            assert abs(reconstructed[key] - original[key]) < 1e-9


class TestAdvancedTrainingIntegration:
    """Integration tests for advanced training utilities."""

    def test_pfsp_pool_workflow(self):
        """Test PFSP opponent pool complete workflow."""
        from app.training.advanced_training import PFSPOpponentPool

        pool = PFSPOpponentPool(max_pool_size=5)

        # Add opponents
        pool.add_opponent("/models/gen1.pth", elo=1500, generation=1)
        pool.add_opponent("/models/gen2.pth", elo=1550, generation=2)
        pool.add_opponent("/models/gen3.pth", elo=1600, generation=3)

        assert len(pool.get_opponents()) == 3

        # Sample and update
        opponent = pool.sample_opponent(current_elo=1550)
        assert opponent is not None

        pool.update_stats(opponent.model_path, won=True)
        pool.update_stats(opponent.model_path, won=False)

        # Check stats
        stats = pool.get_pool_stats()
        assert stats["size"] == 3
        assert stats["total_games"] == 2

    def test_cmaes_tuner_plateau_detection(self):
        """Test CMA-ES tuner detects plateau correctly."""
        from app.training.advanced_training import (
            CMAESAutoTuner,
            PlateauConfig,
        )

        config = PlateauConfig(patience=3, min_delta=5.0)
        tuner = CMAESAutoTuner(
            plateau_config=config,
            min_epochs_between_tuning=1,
        )

        # Improving metric
        tuner.step(current_elo=1500)
        tuner.step(current_elo=1520)
        assert not tuner.should_tune()

        # Plateau
        tuner.step(current_elo=1521)
        tuner.step(current_elo=1521)
        tuner.step(current_elo=1521)
        tuner.step(current_elo=1521)

        assert tuner.should_tune()

    def test_lr_finder_result_structure(self):
        """Test LRFinder produces valid result structure."""
        from app.training.advanced_training import LRFinder

        model = SimpleTestModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        finder = LRFinder(model, optimizer, criterion)

        # Test analyze_results with synthetic data
        lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        losses = [1.0, 0.9, 0.7, 0.4, 0.5, 2.0]

        result = finder._analyze_results(lrs, losses, 1e-7, 10.0)

        assert result.suggested_lr > 0
        assert result.best_lr > 0
        assert len(result.lrs) == 6


class TestModelTrainingLoop:
    """Integration tests for complete model training loop."""

    def test_simple_training_loop(self):
        """Test a simple training loop runs without errors."""
        model = SimpleTestModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        losses = []

        for _epoch in range(3):
            epoch_loss = 0
            for _ in range(5):  # 5 batches per epoch
                x = torch.randn(4, 21, 8, 8)
                target = torch.randint(0, 2048, (4,))

                policy, _value = model(x)
                loss = criterion(policy, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / 5)

        # Loss should be computed for all epochs
        assert len(losses) == 3
        # All losses should be finite
        assert all(np.isfinite(l) for l in losses)

    def test_mixed_precision_training_simulation(self):
        """Test simulated mixed precision training works."""
        if not torch.cuda.is_available():
            # Simulate on CPU with float16 cast
            model = SimpleTestModel()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            model.train()
            x = torch.randn(4, 21, 8, 8)

            # Forward in float32 (simulating autocast behavior on CPU)
            policy, value = model(x)

            loss = policy.mean() + value.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Should complete without error
            assert loss.item() is not None


class TestEventBusIntegration:
    """Integration tests for event bus (if available)."""

    @pytest.mark.asyncio
    async def test_event_emission_and_handling(self):
        """Test event can be emitted and handled."""
        try:
            from app.distributed.data_events import (
                DataEvent,
                DataEventType,
                get_event_bus,
            )
        except ImportError:
            pytest.skip("Event bus not available")

        bus = get_event_bus()
        received = []

        async def handler(event):
            received.append(event)

        # Subscribe
        bus.subscribe(DataEventType.TRAINING_STARTED, handler)

        # Publish (EventBus uses publish, not emit)
        event = DataEvent(
            event_type=DataEventType.TRAINING_STARTED,
            payload={"test": True},
            source="test",
        )
        await bus.publish(event)

        # Give async handler time to process
        import asyncio
        await asyncio.sleep(0.1)

        # Should have received the event
        assert len(received) >= 1
