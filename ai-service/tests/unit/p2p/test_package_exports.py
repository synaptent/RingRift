"""Focused tests for app.p2p package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_p2p_surface() -> None:
    module = importlib.import_module("app.p2p")

    expected = [
        "DEFAULT_PORT",
        "GPU_POWER_RANKINGS",
        "HEARTBEAT_INTERVAL",
        "LEADER_LEASE_DURATION",
        "PEER_TIMEOUT",
        "JobStatus",
        "JobType",
        "NodeHealth",
        "NodeRole",
        "P2PConfig",
        "get_p2p_config",
        "TrainingThresholds",
        "calculate_training_priority",
        "should_trigger_training",
        "WebhookConfig",
        "send_webhook_notification",
        "CONSENSUS_MODE",
        "MEMBERSHIP_MODE",
        "HybridCoordinator",
        "HybridStatus",
        "create_hybrid_coordinator",
        "SWIM_AVAILABLE",
        "HybridMembershipManager",
        "SwimConfig",
        "SwimMembershipManager",
        "PYSYNCOBJ_AVAILABLE",
        "RAFT_ENABLED",
        "RaftWorkItem",
        "ReplicatedJobAssignments",
        "ReplicatedWorkQueue",
        "create_replicated_job_assignments",
        "create_replicated_work_queue",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
