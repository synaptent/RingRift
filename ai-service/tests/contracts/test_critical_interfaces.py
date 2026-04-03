"""Contract tests for critical interfaces.

These are NOT full unit tests — they verify that functions/classes exist
with the expected signatures. Catches API mismatches (renamed methods,
changed signatures) before they silently break the pipeline.

Each test takes <0.1s. Total suite runs in <5s.
"""
from __future__ import annotations

import inspect
import pytest


class TestEloServiceInterface:
    """EloService.get_rating() was called without board_type/num_players,
    causing 112 errors/sec. This contract ensures the signature stays stable."""

    def test_get_rating_signature(self):
        from app.training.elo_service import EloService
        sig = inspect.signature(EloService.get_rating)
        params = list(sig.parameters.keys())
        assert "participant_id" in params
        assert "board_type" in params
        assert "num_players" in params


class TestTrainingExecutorInterface:
    """execute_training_work was called with wrong args. Verify signature."""

    def test_execute_training_work_exists(self):
        from scripts.p2p.work_executors.training_executor import execute_training_work
        assert callable(execute_training_work)

    def test_execute_training_work_signature(self):
        from scripts.p2p.work_executors.training_executor import execute_training_work
        sig = inspect.signature(execute_training_work)
        params = list(sig.parameters.keys())
        assert "work_item" in params
        assert "config" in params
        assert "node_id" in params
        assert "ringrift_path" in params


class TestHandlerBaseInterface:
    """HandlerBase is the parent of 100+ daemons. Changes here break everything."""

    def test_handler_base_has_run_cycle(self):
        from app.coordination.handler_base import HandlerBase
        assert hasattr(HandlerBase, "_run_cycle")

    def test_handler_base_has_config_property(self):
        from app.coordination.handler_base import HandlerBase
        assert isinstance(inspect.getattr_static(HandlerBase, "config"), property)

    def test_handler_base_has_record_error(self):
        from app.coordination.handler_base import HandlerBase
        assert hasattr(HandlerBase, "_record_error")

    def test_handler_base_has_safe_create_task(self):
        from app.coordination.handler_base import HandlerBase
        assert hasattr(HandlerBase, "_safe_create_task")


class TestEventRouterInterface:
    """Event router is the central nervous system."""

    def test_emit_event_exists(self):
        from app.coordination.event_router import emit_event
        assert callable(emit_event)

    def test_subscribe_exists(self):
        from app.coordination.event_router import subscribe
        assert callable(subscribe)

    def test_publish_sync_exists(self):
        from app.coordination.event_router import publish_sync
        assert callable(publish_sync)


class TestTrainingInterface:
    """Training pipeline critical path."""

    def test_train_module_importable(self):
        import app.training.train
        assert hasattr(app.training.train, "train_model")

    def test_make_env_exists(self):
        from app.training.env import make_env, TrainingEnvConfig
        assert callable(make_env)

    def test_gumbel_mcts_ai_exists(self):
        from app.ai.gumbel_mcts_ai import GumbelMCTSAI
        assert callable(GumbelMCTSAI)


class TestClusterConfigInterface:
    """Cluster config is used by every daemon for node discovery."""

    def test_get_cluster_nodes(self):
        from app.config.cluster_config import get_cluster_nodes
        assert callable(get_cluster_nodes)

    def test_cluster_node_has_is_active(self):
        from app.config.cluster_config import ClusterNode
        assert hasattr(ClusterNode, "is_active")

    def test_cluster_node_has_ssh_key(self):
        """ssh_key was missing from node monitor, causing all SSH to fail."""
        from app.config.cluster_config import ClusterNode
        # Verify the attribute exists in the class definition
        import dataclasses
        if dataclasses.is_dataclass(ClusterNode):
            fields = {f.name for f in dataclasses.fields(ClusterNode)}
            assert "ssh_key" in fields


class TestPromotionInterface:
    """Promotion daemon gates model promotion."""

    def test_auto_promotion_daemon_importable(self):
        from app.coordination.auto_promotion_daemon import AutoPromotionDaemon
        assert callable(AutoPromotionDaemon)

    def test_promotion_config_has_threshold(self):
        from app.coordination.auto_promotion_daemon import AutoPromotionConfig
        cfg = AutoPromotionConfig()
        assert hasattr(cfg, "min_win_rate_vs_canonical")
        assert hasattr(cfg, "head_to_head_games")


class TestS3SyncInterface:
    """S3 sync daemon config was None, causing 124 errors/sec."""

    def test_s3_sync_daemon_has_config(self):
        from app.coordination.s3_sync_daemon import S3SyncDaemon, S3SyncConfig
        daemon = S3SyncDaemon.__new__(S3SyncDaemon)
        # Verify the config property exists (even if not initialized)
        assert hasattr(S3SyncDaemon, "config")
