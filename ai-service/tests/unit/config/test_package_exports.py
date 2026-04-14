"""Focused tests for app.config package exports."""

from __future__ import annotations

import importlib


def test_package_dir_lists_declared_config_surface() -> None:
    module = importlib.import_module("app.config")

    expected = [
        "AutoSyncConfig",
        "ClusterConfig",
        "SyncRoutingConfig",
        "filter_hosts_by_provider",
        "filter_hosts_by_role",
        "filter_hosts_by_status",
        "get_auto_sync_config",
        "get_host_bandwidth_limit",
        "get_host_provider",
        "get_p2p_voters",
        "get_priority_sync_targets",
        "get_ready_hosts",
        "get_sync_routing",
        "get_underserved_configs",
        "is_host_sync_excluded",
        "load_cluster_config",
        "DATA_SERVER_PORT",
        "GOSSIP_PORT",
        "HEALTH_CHECK_PORT",
        "METRICS_PORT",
        "P2P_DEFAULT_PORT",
        "get_data_server_url",
        "get_health_check_url",
        "get_local_p2p_url",
        "get_p2p_base_url",
        "get_p2p_status_url",
        "ELO_DROP_ROLLBACK",
        "ELO_IMPROVEMENT_PROMOTE",
        "ELO_K_FACTOR",
        "HAS_UNIFIED_CONFIG",
        "INITIAL_ELO_RATING",
        "MIN_GAMES_FOR_ELO",
        "MIN_GAMES_PROMOTE",
        "MIN_GAMES_REGRESSION",
        "TRAINING_MAX_CONCURRENT",
        "TRAINING_MIN_INTERVAL_SECONDS",
        "TRAINING_STALENESS_HOURS",
        "TRAINING_TRIGGER_GAMES",
        "WIN_RATE_DROP_ROLLBACK",
        "CMAESConfig",
        "ConfigLoadError",
        "ConfigLoader",
        "ConfigSource",
        "ConfigValidator",
        "DataLoadingConfig",
        "IntegratedEnhancementsConfig",
        "NeuralNetConfig",
        "QualityConfig",
        "SelfPlayConfig",
        "UnifiedConfig",
        "ValidationResult",
        "create_training_manager",
        "env_override",
        "get_config",
        "get_training_threshold",
        "load_config",
        "merge_configs",
        "save_config",
        "validate_all_configs",
        "validate_config",
        "validate_startup",
    ]

    assert module.__all__ == expected
    assert len(module.__all__) == len(set(module.__all__))

    for name in expected:
        assert hasattr(module, name)
        assert name in dir(module)
