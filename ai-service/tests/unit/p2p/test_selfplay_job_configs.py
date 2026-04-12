from __future__ import annotations

from scripts.p2p.config.selfplay_job_configs import (
    coerce_config_to_profile,
    config_key_for,
    get_filtered_configs,
)


class TestSelfplayJobConfigs:
    def test_policy_gumbel_profile_filters_to_policy_bearing_configs(self) -> None:
        configs = get_filtered_configs(selfplay_profile="policy-gumbel")

        assert configs
        assert {cfg["engine_mode"] for cfg in configs} == {"gumbel-mcts"}

    def test_allowed_config_keys_filter_is_applied(self) -> None:
        configs = get_filtered_configs(
            selfplay_profile="policy-gumbel",
            allowed_config_keys=["hex8_2p", "square8_2p"],
        )

        assert configs
        assert {config_key_for(cfg) for cfg in configs} == {"hex8_2p", "square8_2p"}

    def test_policy_profile_coerces_mixed_mode_back_to_gumbel(self) -> None:
        config = {
            "board_type": "hex8",
            "num_players": 2,
            "engine_mode": "mixed-opponents",
        }

        coerced = coerce_config_to_profile(
            config,
            selfplay_profile="policy-gumbel",
            allowed_config_keys=["hex8_2p"],
        )

        assert coerced is not None
        assert coerced["board_type"] == "hex8"
        assert coerced["num_players"] == 2
        assert coerced["engine_mode"] == "gumbel-mcts"

    def test_allowed_config_keys_can_reject_unassigned_config(self) -> None:
        config = {
            "board_type": "square8",
            "num_players": 3,
            "engine_mode": "gumbel-mcts",
        }

        coerced = coerce_config_to_profile(
            config,
            selfplay_profile="policy-gumbel",
            allowed_config_keys=["hex8_2p"],
        )

        assert coerced is None
