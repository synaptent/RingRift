from __future__ import annotations

from pathlib import Path

from app.coordination.sync_policy import (
    SyncPolicy,
    PullPolicy,
    is_pull_to_internal_allowed,
    is_evaluation_db_name,
    load_sync_policy,
)


def test_default_policy_blocks_gauntlet_rehydration_without_signal() -> None:
    policy = SyncPolicy()

    assert is_evaluation_db_name("gauntlet_square19_3p.db")
    assert not is_pull_to_internal_allowed(
        "gauntlet_square19_3p.db",
        family="games",
        policy=policy,
    )


def test_non_gauntlet_pull_requires_allowlist_and_signal() -> None:
    policy = SyncPolicy(
        pull=PullPolicy(
            require_consumer_signal=True,
            pull_allowlist=("training", "*.npz"),
        )
    )

    assert not is_pull_to_internal_allowed(
        "hex8_2p.npz",
        family="training",
        policy=policy,
    )
    assert is_pull_to_internal_allowed(
        "hex8_2p.npz",
        family="training",
        consumer_signal="trainer:hex8_2p",
        policy=policy,
    )


def test_gauntlet_requires_explicit_gauntlet_allowance() -> None:
    policy = SyncPolicy(
        pull=PullPolicy(
            require_consumer_signal=True,
            pull_allowlist=("gauntlet_*.db",),
            gauntlet_pull_allowed=True,
        )
    )

    assert is_pull_to_internal_allowed(
        Path("data/games/gauntlet_hex8_2p.db"),
        family="games",
        consumer_signal="manual-gauntlet-import",
        policy=policy,
    )


def test_load_policy_yaml_subset(tmp_path: Path) -> None:
    policy_path = tmp_path / "sync_policy.yaml"
    policy_path.write_text(
        "\n".join(
            [
                "internal_write_min_free_gb: 12",
                "pull:",
                "  default_allowed: false",
                "  require_consumer_signal: true",
                "  gauntlet_allowed: false",
                "  allowlist:",
                "    - '*.npz'",
            ]
        )
    )

    policy = load_sync_policy(policy_path)

    assert policy.internal_write_min_free_gb == 12
    assert policy.pull.pull_allowlist == ("*.npz",)
