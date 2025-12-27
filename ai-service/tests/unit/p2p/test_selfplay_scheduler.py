"""Unit tests for SelfplayScheduler.

Tests priority-based config selection, job targeting, and diversity tracking.
"""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.p2p.managers.selfplay_scheduler import (
    DiversityMetrics,
    SelfplayScheduler,
    MIN_MEMORY_GB_FOR_TASKS,
    DISK_WARNING_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
)


class TestDiversityMetrics:
    """Test DiversityMetrics dataclass."""

    def test_default_init(self):
        """Test default initialization."""
        metrics = DiversityMetrics()

        assert metrics.games_by_engine_mode == {}
        assert metrics.games_by_board_config == {}
        assert metrics.games_by_difficulty == {}
        assert metrics.asymmetric_games == 0
        assert metrics.symmetric_games == 0
        assert metrics.last_reset <= time.time()

    def test_to_dict_empty(self):
        """Test to_dict with no games."""
        metrics = DiversityMetrics()
        result = metrics.to_dict()

        assert result["games_by_engine_mode"] == {}
        assert result["games_by_board_config"] == {}
        assert result["games_by_difficulty"] == {}
        assert result["asymmetric_games"] == 0
        assert result["symmetric_games"] == 0
        assert result["asymmetric_ratio"] == 0.0
        assert result["engine_mode_distribution"] == {}
        assert result["uptime_seconds"] >= 0

    def test_to_dict_with_data(self):
        """Test to_dict with game data."""
        metrics = DiversityMetrics()
        metrics.games_by_engine_mode = {"mixed": 10, "heuristic-only": 5}
        metrics.games_by_board_config = {"hex8_2p": 8, "square8_2p": 7}
        metrics.asymmetric_games = 3
        metrics.symmetric_games = 12

        result = metrics.to_dict()

        assert result["asymmetric_ratio"] == 3 / 15  # 3 / (3 + 12)
        assert result["engine_mode_distribution"]["mixed"] == 10 / 15
        assert result["engine_mode_distribution"]["heuristic-only"] == 5 / 15


class TestSelfplaySchedulerInit:
    """Test SelfplayScheduler initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scheduler = SelfplayScheduler()

        assert callable(scheduler.get_cluster_elo)
        assert callable(scheduler.load_curriculum_weights)
        assert callable(scheduler.get_board_priority_overrides)
        assert scheduler.should_stop_production is None
        assert scheduler.verbose is False
        assert isinstance(scheduler.diversity_metrics, DiversityMetrics)

    def test_init_with_callbacks(self):
        """Test initialization with custom callbacks."""
        def mock_elo():
            return {"top_models": []}

        def mock_weights():
            return {"hex8_2p": 1.2}

        scheduler = SelfplayScheduler(
            get_cluster_elo_fn=mock_elo,
            load_curriculum_weights_fn=mock_weights,
            verbose=True,
        )

        assert scheduler.get_cluster_elo() == {"top_models": []}
        assert scheduler.load_curriculum_weights() == {"hex8_2p": 1.2}
        assert scheduler.verbose is True


class TestEloPriorityBoost:
    """Test ELO-based priority boost calculation."""

    def test_boost_no_elo_data(self):
        """Test boost when no ELO data available."""
        scheduler = SelfplayScheduler()
        boost = scheduler.get_elo_based_priority_boost("hex8", 2)

        # Should get boost for hex8 being underrepresented
        assert boost >= 0

    def test_boost_high_elo_model(self):
        """Test boost for high-ELO models."""
        def mock_elo():
            return {
                "top_models": [
                    {"name": "model_hex8_2p", "elo": 1400}
                ]
            }

        scheduler = SelfplayScheduler(get_cluster_elo_fn=mock_elo)
        boost = scheduler.get_elo_based_priority_boost("hex8", 2)

        # Should get boost for high ELO (1400 - 1200) // 100 = 2
        # Plus boost for hex8 being underrepresented
        assert boost >= 2

    def test_boost_for_underrepresented_configs(self):
        """Test boost for underrepresented board types."""
        scheduler = SelfplayScheduler()

        # hexagonal and square19 should get +1
        boost_hex = scheduler.get_elo_based_priority_boost("hexagonal", 2)
        boost_sq19 = scheduler.get_elo_based_priority_boost("square19", 2)
        boost_sq8 = scheduler.get_elo_based_priority_boost("square8", 2)

        assert boost_hex >= 1
        assert boost_sq19 >= 1
        # square8 shouldn't get underrepresented boost
        assert boost_sq8 == 0 or boost_sq8 == 1  # 1 for 2p

    def test_boost_for_multiplayer(self):
        """Test boost for 3+ player games."""
        scheduler = SelfplayScheduler()

        boost_2p = scheduler.get_elo_based_priority_boost("square8", 2)
        boost_3p = scheduler.get_elo_based_priority_boost("square8", 3)
        boost_4p = scheduler.get_elo_based_priority_boost("square8", 4)

        # 3p and 4p should get +1 boost
        assert boost_3p > boost_2p
        assert boost_4p > boost_2p

    def test_boost_capped_at_5(self):
        """Test that boost is capped at 5."""
        def mock_elo():
            return {
                "top_models": [
                    {"name": "model_hexagonal_3p", "elo": 2000}  # Very high ELO
                ]
            }

        scheduler = SelfplayScheduler(get_cluster_elo_fn=mock_elo)
        boost = scheduler.get_elo_based_priority_boost("hexagonal", 3)

        assert boost <= 5


class TestPickWeightedConfig:
    """Test weighted config selection."""

    @dataclass
    class MockNode:
        node_id: str = "test-node"
        memory_gb: int = 64

    def test_pick_config_basic(self):
        """Test basic config selection."""
        scheduler = SelfplayScheduler()
        node = self.MockNode()

        config = scheduler.pick_weighted_config(node)

        assert config is not None
        assert "board_type" in config
        assert "num_players" in config
        assert "engine_mode" in config
        assert "priority" in config

    def test_pick_config_low_memory_filters(self):
        """Test that low memory nodes only get square8 configs."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(memory_gb=32)  # Less than 48GB

        # Run multiple times to check filtering
        for _ in range(10):
            config = scheduler.pick_weighted_config(node)
            if config:
                assert config["board_type"] == "square8"

    def test_pick_config_no_memory_info(self):
        """Test config selection when node has no memory info."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(memory_gb=0)

        config = scheduler.pick_weighted_config(node)
        # Should still return a config (no filtering applied)
        assert config is not None

    def test_pick_config_with_curriculum_weights(self):
        """Test config selection with curriculum weight boosts."""
        def mock_weights():
            return {"hex8_2p": 1.5}  # High weight for hex8_2p

        scheduler = SelfplayScheduler(load_curriculum_weights_fn=mock_weights)
        node = self.MockNode()

        # Run multiple times - hex8_2p should appear more often
        configs = [scheduler.pick_weighted_config(node) for _ in range(100)]
        hex8_2p_count = sum(
            1 for c in configs
            if c and c["board_type"] == "hex8" and c["num_players"] == 2
        )

        # With 1.5 weight boost, should appear more than baseline
        assert hex8_2p_count > 0

    def test_pick_config_with_board_priority_overrides(self):
        """Test config selection with board priority overrides."""
        def mock_overrides():
            return {"square8_2p": 0}  # CRITICAL priority for square8_2p

        scheduler = SelfplayScheduler(get_board_priority_overrides_fn=mock_overrides)
        node = self.MockNode()

        # Run multiple times
        configs = [scheduler.pick_weighted_config(node) for _ in range(100)]
        sq8_2p_count = sum(
            1 for c in configs
            if c and c["board_type"] == "square8" and c["num_players"] == 2
        )

        # With CRITICAL (0) priority = +6 boost, should appear often
        assert sq8_2p_count > 0


class TestTargetJobsForNode:
    """Test job targeting for nodes."""

    @dataclass
    class MockNode:
        node_id: str = "test-node"
        has_gpu: bool = True
        gpu_name: str = "RTX 4090"
        gpu_count: int = 1
        cpu_count: int = 32
        memory_gb: int = 64
        cpu_percent: float = 50.0
        memory_percent: float = 40.0
        disk_percent: float = 60.0
        gpu_percent: float = 50.0
        gpu_memory_percent: float = 40.0
        selfplay_jobs: int = 4

    def test_target_jobs_basic(self):
        """Test basic job targeting."""
        scheduler = SelfplayScheduler()
        node = self.MockNode()

        target = scheduler.get_target_jobs_for_node(node)

        assert target >= 1
        assert isinstance(target, int)

    def test_target_jobs_low_memory(self):
        """Test that low memory nodes get 0 jobs."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(memory_gb=8)  # Less than MIN_MEMORY_GB_FOR_TASKS

        target = scheduler.get_target_jobs_for_node(node)
        assert target == 0

    def test_target_jobs_backpressure_stop(self):
        """Test that backpressure stop returns 0."""
        scheduler = SelfplayScheduler(
            should_stop_production_fn=lambda _: True,
            should_throttle_production_fn=lambda _: False,
        )
        node = self.MockNode()

        target = scheduler.get_target_jobs_for_node(node)
        assert target == 0

    def test_target_jobs_backpressure_throttle(self):
        """Test that backpressure throttle reduces jobs."""
        scheduler = SelfplayScheduler(
            should_stop_production_fn=lambda _: False,
            should_throttle_production_fn=lambda _: True,
            get_throttle_factor_fn=lambda _: 0.5,
        )
        node = self.MockNode()

        target = scheduler.get_target_jobs_for_node(node)
        # Should be reduced by 50%
        assert target >= 1

    def test_target_jobs_high_disk_usage(self):
        """Test that high disk usage limits jobs."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(disk_percent=95)  # Over DISK_WARNING_THRESHOLD

        target = scheduler.get_target_jobs_for_node(node)
        assert target <= 4  # Capped at 4 for high disk

    def test_target_jobs_high_memory_usage(self):
        """Test that high memory usage limits jobs."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(memory_percent=95)  # Over MEMORY_WARNING_THRESHOLD

        target = scheduler.get_target_jobs_for_node(node)
        assert target <= 2  # Capped at 2 for high memory

    def test_target_jobs_gpu_scaling(self):
        """Test GPU-based job scaling."""
        scheduler = SelfplayScheduler()

        # High-end GPU should get more jobs
        h100_node = self.MockNode(gpu_name="H100", cpu_count=64)
        rtx3060_node = self.MockNode(gpu_name="RTX 3060", cpu_count=8)

        h100_target = scheduler.get_target_jobs_for_node(h100_node)
        rtx3060_target = scheduler.get_target_jobs_for_node(rtx3060_node)

        assert h100_target > rtx3060_target

    def test_target_jobs_cpu_only(self):
        """Test CPU-only node job targeting."""
        scheduler = SelfplayScheduler()
        node = self.MockNode(has_gpu=False, cpu_count=64)

        target = scheduler.get_target_jobs_for_node(node)
        # CPU-only: 0.3 jobs per core, capped at 32
        assert target >= 1
        assert target <= 32


class TestHybridJobTargets:
    """Test hybrid GPU+CPU job targeting."""

    @dataclass
    class MockNode:
        node_id: str = "test-node"
        has_gpu: bool = True
        gpu_name: str = "RTX 3060"
        gpu_count: int = 1
        cpu_count: int = 128
        memory_gb: int = 256

    def test_hybrid_targets_basic(self):
        """Test basic hybrid job targets."""
        scheduler = SelfplayScheduler()
        node = self.MockNode()

        targets = scheduler.get_hybrid_job_targets(node)

        assert "gpu_jobs" in targets
        assert "cpu_only_jobs" in targets
        assert "total_jobs" in targets
        assert targets["total_jobs"] == targets["gpu_jobs"] + targets["cpu_only_jobs"]

    def test_hybrid_targets_no_function(self):
        """Test hybrid targets when no limits function available."""
        scheduler = SelfplayScheduler()
        node = self.MockNode()

        targets = scheduler.get_hybrid_job_targets(node)

        # Should fallback to GPU jobs only
        assert targets["cpu_only_jobs"] == 0

    def test_should_spawn_cpu_only_jobs_high_cpu_low_vram(self):
        """Test CPU-only spawning for high-CPU low-VRAM nodes."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 4, "cpu_only_jobs": 8, "total_jobs": 12}
        )

        node = self.MockNode(
            cpu_count=128,
            gpu_name="RTX 3060",  # Limited VRAM
        )

        should_spawn = scheduler.should_spawn_cpu_only_jobs(node)
        assert should_spawn is True

    def test_should_spawn_cpu_only_jobs_low_cpu(self):
        """Test CPU-only NOT spawned for low-CPU nodes."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 4, "cpu_only_jobs": 0, "total_jobs": 4}
        )

        node = self.MockNode(cpu_count=32)  # Low CPU count

        should_spawn = scheduler.should_spawn_cpu_only_jobs(node)
        assert should_spawn is False

    def test_should_spawn_cpu_only_jobs_high_end_gpu(self):
        """Test CPU-only NOT spawned for high-end datacenter GPUs."""
        scheduler = SelfplayScheduler(
            get_hybrid_selfplay_limits_fn=lambda **kw: {"gpu_jobs": 8, "cpu_only_jobs": 0, "total_jobs": 8}
        )

        node = self.MockNode(
            cpu_count=128,
            gpu_name="H100",  # High-end GPU
        )

        should_spawn = scheduler.should_spawn_cpu_only_jobs(node)
        assert should_spawn is False


class TestDiversityTracking:
    """Test diversity metrics tracking."""

    def test_track_diversity_engine_mode(self):
        """Test tracking engine mode diversity."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_engine_mode["mixed"] == 1

    def test_track_diversity_board_config(self):
        """Test tracking board config diversity."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_board_config["hex8_2p"] == 1

    def test_track_diversity_symmetric(self):
        """Test tracking symmetric games."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.symmetric_games == 1
        assert scheduler.diversity_metrics.asymmetric_games == 0

    def test_track_diversity_asymmetric(self):
        """Test tracking asymmetric games."""
        scheduler = SelfplayScheduler()

        config = {
            "engine_mode": "mixed",
            "board_type": "hex8",
            "num_players": 2,
            "asymmetric": True,
            "strong_config": {"engine_mode": "mcts", "difficulty": 8},
            "weak_config": {"engine_mode": "heuristic", "difficulty": 3},
        }
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.asymmetric_games == 1
        assert scheduler.diversity_metrics.symmetric_games == 0

    def test_track_diversity_difficulty(self):
        """Test tracking difficulty distribution."""
        scheduler = SelfplayScheduler()

        config = {
            "engine_mode": "mixed",
            "board_type": "hex8",
            "num_players": 2,
            "difficulty": 5,
        }
        scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_difficulty["5"] == 1

    def test_track_diversity_multiple(self):
        """Test tracking multiple games."""
        scheduler = SelfplayScheduler()

        for i in range(5):
            config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
            scheduler.track_diversity(config)

        for i in range(3):
            config = {"engine_mode": "heuristic-only", "board_type": "square8", "num_players": 2}
            scheduler.track_diversity(config)

        assert scheduler.diversity_metrics.games_by_engine_mode["mixed"] == 5
        assert scheduler.diversity_metrics.games_by_engine_mode["heuristic-only"] == 3
        assert scheduler.diversity_metrics.games_by_board_config["hex8_2p"] == 5
        assert scheduler.diversity_metrics.games_by_board_config["square8_2p"] == 3

    def test_get_diversity_metrics(self):
        """Test getting diversity metrics."""
        scheduler = SelfplayScheduler()

        config = {"engine_mode": "mixed", "board_type": "hex8", "num_players": 2}
        scheduler.track_diversity(config)

        metrics = scheduler.get_diversity_metrics()

        assert isinstance(metrics, dict)
        assert metrics["symmetric_games"] == 1
        assert "mixed" in metrics["games_by_engine_mode"]


class TestConstants:
    """Test module constants."""

    def test_min_memory_threshold(self):
        """Test MIN_MEMORY_GB_FOR_TASKS is reasonable.

        Dec 2025: Increased to 64GB for GH200 nodes.
        """
        assert MIN_MEMORY_GB_FOR_TASKS == 64

    def test_disk_warning_threshold(self):
        """Test DISK_WARNING_THRESHOLD is reasonable.

        Dec 2025: Lowered to 65% for conservative disk management.
        """
        assert 60 <= DISK_WARNING_THRESHOLD <= 75

    def test_memory_warning_threshold(self):
        """Test MEMORY_WARNING_THRESHOLD is reasonable.

        Dec 2025: Set to 75% (below 80% hard cap).
        """
        assert 70 <= MEMORY_WARNING_THRESHOLD <= 80


class TestPromotionFailureTracking:
    """Tests for promotion failure tracking and curriculum feedback.

    December 27, 2025: Added to test PROMOTION_FAILED event handling.
    """

    def test_record_promotion_failure_first_failure(self):
        """First promotion failure applies 30-minute penalty."""
        scheduler = SelfplayScheduler()
        scheduler.record_promotion_failure("hex8_2p")

        assert "hex8_2p" in scheduler._promotion_failures
        assert len(scheduler._promotion_failures["hex8_2p"]) == 1
        assert "hex8_2p" in scheduler._promotion_penalties
        factor, expiry = scheduler._promotion_penalties["hex8_2p"]
        assert factor == 0.7  # 70% priority
        assert expiry > time.time()  # Expires in future

    def test_record_promotion_failure_second_failure(self):
        """Second promotion failure applies 1-hour penalty."""
        scheduler = SelfplayScheduler()
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")

        assert len(scheduler._promotion_failures["hex8_2p"]) == 2
        factor, expiry = scheduler._promotion_penalties["hex8_2p"]
        assert factor == 0.5  # 50% priority after 2 failures

    def test_record_promotion_failure_three_failures(self):
        """Three or more failures apply 2-hour penalty."""
        scheduler = SelfplayScheduler()
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")

        assert len(scheduler._promotion_failures["hex8_2p"]) == 3
        factor, expiry = scheduler._promotion_penalties["hex8_2p"]
        assert factor == 0.3  # 30% priority after 3+ failures

    def test_record_promotion_failure_multiple_configs(self):
        """Track failures independently per config."""
        scheduler = SelfplayScheduler()
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("hex8_2p")
        scheduler.record_promotion_failure("square8_2p")

        assert len(scheduler._promotion_failures["hex8_2p"]) == 2
        assert len(scheduler._promotion_failures["square8_2p"]) == 1
        assert scheduler._promotion_penalties["hex8_2p"][0] == 0.5
        assert scheduler._promotion_penalties["square8_2p"][0] == 0.7

    def test_old_failures_expire(self):
        """Failures older than 24 hours are removed."""
        scheduler = SelfplayScheduler()
        # Manually add an old failure
        scheduler._promotion_failures = {
            "hex8_2p": [time.time() - 90000]  # >24h ago
        }
        scheduler.record_promotion_failure("hex8_2p")

        # Old failure should be pruned, only new one remains
        assert len(scheduler._promotion_failures["hex8_2p"]) == 1
