"""Unit tests for selfplay_config module.

Tests the unified configuration system for selfplay generation,
including EngineMode enum, SelfplayConfig dataclass, argument
parsing, and curriculum configuration templates.

December 2025: Created for P1 test coverage priority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.training.selfplay_config import (
    CURRICULUM_STAGES,
    CPU_COMPATIBLE_ENGINE_MODES,
    ENGINE_MODE_ALIASES,
    GPU_REQUIRED_ENGINE_MODES,
    MIXED_ENGINE_MODES,
    CurriculumStage,
    EngineMode,
    OutputFormat,
    SelfplayConfig,
    create_argument_parser,
    engine_mode_is_cpu_compatible,
    engine_mode_requires_gpu,
    get_all_configs_curriculum,
    get_curriculum_config,
    get_default_config,
    get_full_curriculum,
    get_production_config,
    list_curriculum_stages,
    normalize_engine_mode,
    parse_selfplay_args,
)


# =============================================================================
# EngineMode Enum Tests
# =============================================================================


class TestEngineMode:
    """Tests for EngineMode enum."""

    def test_engine_mode_values(self):
        """Should have expected engine mode values."""
        assert EngineMode.HEURISTIC.value == "heuristic-only"
        assert EngineMode.GUMBEL_MCTS.value == "gumbel-mcts"
        assert EngineMode.MCTS.value == "mcts"
        assert EngineMode.RANDOM.value == "random"

    def test_engine_mode_count(self):
        """Should have expected number of engine modes."""
        # Core modes + experimental + GNN
        assert len(EngineMode) >= 15

    def test_string_enum(self):
        """EngineMode should be a string enum."""
        assert isinstance(EngineMode.GUMBEL_MCTS, str)
        assert EngineMode.GUMBEL_MCTS == "gumbel-mcts"

    def test_deprecated_modes_exist(self):
        """Deprecated modes should still be available."""
        assert EngineMode.GMO.value == "gmo"
        assert EngineMode.EBMO.value == "ebmo"
        assert EngineMode.IG_GMO.value == "ig-gmo"
        assert EngineMode.CAGE.value == "cage"

    def test_gnn_modes_exist(self):
        """GNN-based modes should be available."""
        assert EngineMode.GNN.value == "gnn"
        assert EngineMode.HYBRID.value == "hybrid"


class TestEngineModeAliases:
    """Tests for ENGINE_MODE_ALIASES dictionary."""

    def test_alias_mapping(self):
        """Should map aliases to canonical values."""
        assert ENGINE_MODE_ALIASES["gumbel_mcts"] == "gumbel-mcts"
        assert ENGINE_MODE_ALIASES["gumbel"] == "gumbel-mcts"
        assert ENGINE_MODE_ALIASES["policy_only"] == "policy-only"

    def test_underscore_hyphen_normalization(self):
        """Should handle both underscores and hyphens."""
        assert ENGINE_MODE_ALIASES["nn_minimax"] == "nn-minimax"
        assert ENGINE_MODE_ALIASES["nn_descent"] == "nn-descent"

    def test_shorthand_aliases(self):
        """Should have shorthand aliases."""
        assert ENGINE_MODE_ALIASES["descent"] == "descent-only"
        assert ENGINE_MODE_ALIASES["heuristic"] == "heuristic-only"

    def test_gnn_aliases(self):
        """Should have GNN-related aliases."""
        assert ENGINE_MODE_ALIASES["gnn-policy"] == "gnn"
        assert ENGINE_MODE_ALIASES["hybrid-gnn"] == "hybrid"
        assert ENGINE_MODE_ALIASES["cnn-gnn"] == "hybrid"


class TestNormalizeEngineMode:
    """Tests for normalize_engine_mode function."""

    def test_normalize_alias(self):
        """Should normalize aliases to canonical values."""
        assert normalize_engine_mode("gumbel") == "gumbel-mcts"
        assert normalize_engine_mode("GUMBEL") == "gumbel-mcts"
        assert normalize_engine_mode("  gumbel  ") == "gumbel-mcts"

    def test_normalize_canonical(self):
        """Should return canonical values unchanged."""
        assert normalize_engine_mode("gumbel-mcts") == "gumbel-mcts"
        assert normalize_engine_mode("heuristic-only") == "heuristic-only"

    def test_normalize_unknown(self):
        """Should return unknown values lowercased."""
        assert normalize_engine_mode("unknown-mode") == "unknown-mode"
        assert normalize_engine_mode("CUSTOM") == "custom"


# =============================================================================
# GPU Requirement Tests
# =============================================================================


class TestGPURequirements:
    """Tests for GPU requirement classification."""

    def test_gpu_required_modes(self):
        """Should correctly classify GPU-required modes."""
        assert EngineMode.GUMBEL_MCTS in GPU_REQUIRED_ENGINE_MODES
        assert EngineMode.MCTS in GPU_REQUIRED_ENGINE_MODES
        assert EngineMode.NNUE_GUIDED in GPU_REQUIRED_ENGINE_MODES
        assert EngineMode.POLICY_ONLY in GPU_REQUIRED_ENGINE_MODES

    def test_cpu_compatible_modes(self):
        """Should correctly classify CPU-compatible modes."""
        assert EngineMode.HEURISTIC in CPU_COMPATIBLE_ENGINE_MODES
        assert EngineMode.RANDOM in CPU_COMPATIBLE_ENGINE_MODES
        assert EngineMode.DESCENT_ONLY in CPU_COMPATIBLE_ENGINE_MODES

    def test_mixed_modes(self):
        """Should correctly classify mixed modes."""
        assert EngineMode.MIXED in MIXED_ENGINE_MODES
        assert EngineMode.DIVERSE in MIXED_ENGINE_MODES

    def test_sets_are_disjoint(self):
        """GPU and CPU mode sets should be disjoint."""
        gpu_cpu_overlap = GPU_REQUIRED_ENGINE_MODES & CPU_COMPATIBLE_ENGINE_MODES
        assert len(gpu_cpu_overlap) == 0

    def test_engine_mode_requires_gpu_enum(self):
        """engine_mode_requires_gpu should work with enum."""
        assert engine_mode_requires_gpu(EngineMode.GUMBEL_MCTS) is True
        assert engine_mode_requires_gpu(EngineMode.HEURISTIC) is False

    def test_engine_mode_requires_gpu_string(self):
        """engine_mode_requires_gpu should work with string."""
        assert engine_mode_requires_gpu("gumbel-mcts") is True
        assert engine_mode_requires_gpu("heuristic-only") is False
        assert engine_mode_requires_gpu("gumbel") is True  # Alias

    def test_engine_mode_requires_gpu_unknown(self):
        """Unknown modes should default to GPU required."""
        assert engine_mode_requires_gpu("unknown-mode") is True

    def test_engine_mode_is_cpu_compatible_enum(self):
        """engine_mode_is_cpu_compatible should work with enum."""
        assert engine_mode_is_cpu_compatible(EngineMode.HEURISTIC) is True
        assert engine_mode_is_cpu_compatible(EngineMode.MIXED) is True
        assert engine_mode_is_cpu_compatible(EngineMode.GUMBEL_MCTS) is False

    def test_engine_mode_is_cpu_compatible_string(self):
        """engine_mode_is_cpu_compatible should work with string."""
        assert engine_mode_is_cpu_compatible("heuristic-only") is True
        assert engine_mode_is_cpu_compatible("mixed") is True
        assert engine_mode_is_cpu_compatible("gumbel-mcts") is False

    def test_engine_mode_is_cpu_compatible_unknown(self):
        """Unknown modes should not be CPU compatible."""
        assert engine_mode_is_cpu_compatible("unknown-mode") is False


# =============================================================================
# OutputFormat Enum Tests
# =============================================================================


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_output_format_values(self):
        """Should have expected format values."""
        assert OutputFormat.JSONL.value == "jsonl"
        assert OutputFormat.DB.value == "db"
        assert OutputFormat.NPZ.value == "npz"

    def test_output_format_count(self):
        """Should have 3 output formats."""
        assert len(OutputFormat) == 3


# =============================================================================
# SelfplayConfig Tests
# =============================================================================


class TestSelfplayConfig:
    """Tests for SelfplayConfig dataclass."""

    def test_default_creation(self):
        """Should create with default values."""
        config = SelfplayConfig()
        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.num_games == 1000
        assert config.engine_mode == EngineMode.NNUE_GUIDED

    def test_custom_creation(self):
        """Should accept custom values."""
        config = SelfplayConfig(
            board_type="hexagonal",
            num_players=4,
            num_games=500,
            engine_mode=EngineMode.GUMBEL_MCTS,
        )
        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.num_games == 500
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_board_type_normalization(self):
        """Should normalize board type aliases."""
        config = SelfplayConfig(board_type="hex")
        assert config.board_type == "hexagonal"

        config = SelfplayConfig(board_type="sq8")
        assert config.board_type == "square8"

        config = SelfplayConfig(board_type="full_hex")
        assert config.board_type == "hexagonal"

    def test_engine_mode_string_conversion(self):
        """Should convert string engine mode to enum."""
        config = SelfplayConfig(engine_mode="gumbel-mcts")
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

        config = SelfplayConfig(engine_mode="gumbel")
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_output_format_string_conversion(self):
        """Should convert string output format to enum."""
        config = SelfplayConfig(output_format="jsonl")
        assert config.output_format == OutputFormat.JSONL

    def test_config_key_property(self):
        """Should generate correct config key."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        assert config.config_key == "hex8_2p"

        config = SelfplayConfig(board_type="hexagonal", num_players=4)
        assert config.config_key == "hexagonal_4p"

    def test_board_type_enum_property(self):
        """Should return correct BoardType enum."""
        from app.models import BoardType

        config = SelfplayConfig(board_type="hex8")
        assert config.board_type_enum == BoardType.HEX8

        config = SelfplayConfig(board_type="square19")
        assert config.board_type_enum == BoardType.SQUARE19

    def test_default_output_dir(self):
        """Should set default output directory."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        assert config.output_dir == "data/selfplay/hex8_2p"

    def test_default_record_db(self):
        """Should set default record database path."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        assert config.record_db == "data/games/hex8_2p.db"

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = SelfplayConfig(board_type="hex8", num_players=2)
        d = config.to_dict()

        assert d["board_type"] == "hex8"
        assert d["num_players"] == 2
        assert d["engine_mode"] == "nnue-guided"
        assert d["config_key"] == "hex8_2p"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "board_type": "hexagonal",
            "num_players": 4,
            "num_games": 500,
            "engine_mode": "gumbel-mcts",
            "output_format": "db",
        }
        config = SelfplayConfig.from_dict(data)

        assert config.board_type == "hexagonal"
        assert config.num_players == 4
        assert config.num_games == 500
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_get_effective_budget_explicit(self):
        """Should return explicit budget when set."""
        config = SelfplayConfig(simulation_budget=1000)
        assert config.get_effective_budget() == 1000

    def test_get_effective_budget_elo_adaptive(self):
        """Should use Elo-adaptive budget when model_elo is set."""
        # Test with model_elo set - should use adaptive budget calculation
        config = SelfplayConfig(model_elo=1450, training_epoch=50)
        budget = config.get_effective_budget()
        # Elo-adaptive budget should be > 0 and reasonable
        assert budget > 0
        assert budget <= 1600  # Max budget is GUMBEL_BUDGET_ULTIMATE

    def test_get_effective_budget_difficulty(self):
        """Should use difficulty-based budget when difficulty is set."""
        # Test with difficulty set - should use difficulty-based budget
        config = SelfplayConfig(difficulty=8)
        budget = config.get_effective_budget()
        # Difficulty-based budget should be > 0 and reasonable
        assert budget > 0
        assert budget <= 1600  # Max budget is GUMBEL_BUDGET_ULTIMATE

    def test_resource_settings(self):
        """Should have resource settings."""
        config = SelfplayConfig(
            num_workers=4,
            batch_size=512,
            use_gpu=True,
            gpu_device=1,
        )
        assert config.num_workers == 4
        assert config.batch_size == 512
        assert config.use_gpu is True
        assert config.gpu_device == 1

    def test_recording_options(self):
        """Should have recording options."""
        config = SelfplayConfig(
            store_history_entries=False,
            lean_db=True,
            snapshot_interval=50,
        )
        assert config.store_history_entries is False
        assert config.lean_db is True
        assert config.snapshot_interval == 50

    def test_extra_options(self):
        """Should support extra options dict."""
        config = SelfplayConfig(extra_options={"custom_key": "value"})
        assert config.extra_options["custom_key"] == "value"


# =============================================================================
# Argument Parser Tests
# =============================================================================


class TestCreateArgumentParser:
    """Tests for create_argument_parser function."""

    def test_creates_parser(self):
        """Should create argument parser."""
        parser = create_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_custom_description(self):
        """Should accept custom description."""
        parser = create_argument_parser(description="Custom description")
        assert parser.description == "Custom description"

    def test_core_game_arguments(self):
        """Should have core game arguments."""
        parser = create_argument_parser()
        args = parser.parse_args(["--board", "hex8", "--num-players", "4", "--num-games", "500"])

        assert args.board == "hex8"
        assert args.num_players == 4
        assert args.num_games == 500

    def test_engine_arguments(self):
        """Should have engine arguments."""
        parser = create_argument_parser()
        args = parser.parse_args([
            "--engine-mode", "gumbel-mcts",
            "--search-depth", "5",
            "--mcts-simulations", "1600",
            "--temperature", "0.5",
        ])

        assert args.engine_mode == "gumbel-mcts"
        assert args.search_depth == 5
        assert args.mcts_simulations == 1600
        assert args.temperature == 0.5

    def test_gpu_arguments(self):
        """Should have GPU arguments when included."""
        parser = create_argument_parser(include_gpu=True)
        args = parser.parse_args(["--no-gpu", "--gpu-device", "1"])

        assert args.no_gpu is True
        assert args.gpu_device == 1

    def test_no_gpu_arguments(self):
        """Should not have GPU arguments when excluded."""
        parser = create_argument_parser(include_gpu=False)
        args = parser.parse_args([])

        assert not hasattr(args, "no_gpu")
        assert not hasattr(args, "gpu_device")

    def test_ramdrive_arguments(self):
        """Should have ramdrive arguments when included."""
        parser = create_argument_parser(include_ramdrive=True)
        args = parser.parse_args(["--use-ramdrive", "--ramdrive-path", "/tmp/ram"])

        assert args.use_ramdrive is True
        assert args.ramdrive_path == "/tmp/ram"

    def test_pfsp_argument(self):
        """Should have PFSP disable argument."""
        parser = create_argument_parser()
        args = parser.parse_args(["--disable-pfsp"])
        assert args.disable_pfsp is True

    def test_mixed_opponents_arguments(self):
        """Should have mixed opponents arguments."""
        parser = create_argument_parser()
        args = parser.parse_args([
            "--mixed-opponents",
            "--opponent-mix", "random:0.2,heuristic:0.5,mcts:0.3",
        ])

        assert args.mixed_opponents is True
        assert args.opponent_mix == "random:0.2,heuristic:0.5,mcts:0.3"

    def test_pipeline_events_argument(self):
        """Should have pipeline events argument."""
        parser = create_argument_parser()
        args = parser.parse_args(["--emit-pipeline-events"])
        assert args.emit_pipeline_events is True


class TestParseSelfplayArgs:
    """Tests for parse_selfplay_args function."""

    def test_basic_parsing(self):
        """Should parse basic arguments."""
        config = parse_selfplay_args([
            "--board", "hex8",
            "--num-players", "2",
            "--num-games", "100",
        ])

        assert config.board_type == "hex8"
        assert config.num_players == 2
        assert config.num_games == 100

    def test_engine_mode_parsing(self):
        """Should parse engine mode."""
        config = parse_selfplay_args(["--engine-mode", "gumbel-mcts"])
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_pfsp_enabled_by_default(self):
        """PFSP should be enabled by default."""
        config = parse_selfplay_args([])
        assert config.use_pfsp is True

    def test_pfsp_disabled(self):
        """Should disable PFSP when flag is set."""
        config = parse_selfplay_args(["--disable-pfsp"])
        assert config.use_pfsp is False

    def test_mixed_opponents_sets_engine_mode(self):
        """Mixed opponents flag should set engine mode to MIXED."""
        config = parse_selfplay_args(["--mixed-opponents"])
        assert config.engine_mode == EngineMode.MIXED
        assert config.mixed_opponents is True

    def test_opponent_mix_parsing(self):
        """Should parse opponent mix string."""
        config = parse_selfplay_args([
            "--mixed-opponents",
            "--opponent-mix", "random:0.2,heuristic:0.5,mcts:0.3",
        ])

        assert config.opponent_mix == {
            "random": 0.2,
            "heuristic": 0.5,
            "mcts": 0.3,
        }

    def test_invalid_opponent_mix_logs_warning(self):
        """Should handle invalid opponent mix gracefully."""
        # Invalid format should not raise, just log warning
        config = parse_selfplay_args([
            "--mixed-opponents",
            "--opponent-mix", "invalid_format",
        ])
        assert config.opponent_mix is None  # Falls back to None


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience configuration functions."""

    def test_get_default_config(self):
        """Should return default config."""
        config = get_default_config()
        assert config.board_type == "square8"
        assert config.num_players == 2

    def test_get_default_config_custom(self):
        """Should accept custom board type and players."""
        config = get_default_config("hexagonal", 4)
        assert config.board_type == "hexagonal"
        assert config.num_players == 4

    def test_get_production_config(self):
        """Should return production config."""
        config = get_production_config("hex8", 2)
        assert config.board_type == "hex8"
        assert config.num_players == 2
        assert config.num_games == 100000
        assert config.engine_mode == EngineMode.DIVERSE
        assert config.mcts_simulations == 1600


# =============================================================================
# Curriculum Tests
# =============================================================================


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_curriculum_stage_creation(self):
        """Should create curriculum stage."""
        stage = CurriculumStage(
            name="test",
            engine_mode=EngineMode.GUMBEL_MCTS,
            temperature=0.5,
            mcts_simulations=800,
            search_depth=4,
            games_per_config=1000,
        )
        assert stage.name == "test"
        assert stage.engine_mode == EngineMode.GUMBEL_MCTS

    def test_curriculum_stage_defaults(self):
        """Should have default values."""
        stage = CurriculumStage(
            name="test",
            engine_mode=EngineMode.HEURISTIC,
            temperature=1.0,
            mcts_simulations=0,
            search_depth=1,
            games_per_config=100,
        )
        assert stage.random_opening_moves == 0
        assert stage.description == ""


class TestCurriculumStages:
    """Tests for CURRICULUM_STAGES dictionary."""

    def test_curriculum_stages_exist(self):
        """Should have expected curriculum stages."""
        assert "explore_random" in CURRICULUM_STAGES
        assert "explore_weak" in CURRICULUM_STAGES
        assert "moderate_mcts" in CURRICULUM_STAGES
        assert "strong_gumbel" in CURRICULUM_STAGES
        assert "strong_full" in CURRICULUM_STAGES

    def test_curriculum_stages_structure(self):
        """Stages should have required fields."""
        for name, stage in CURRICULUM_STAGES.items():
            assert isinstance(stage, CurriculumStage)
            assert stage.name == name
            assert isinstance(stage.engine_mode, EngineMode)
            assert stage.games_per_config > 0

    def test_robust_diverse_stage(self):
        """Should have robust_diverse stage for mixed training."""
        assert "robust_diverse" in CURRICULUM_STAGES
        stage = CURRICULUM_STAGES["robust_diverse"]
        assert stage.engine_mode == EngineMode.MIXED


class TestGetCurriculumConfig:
    """Tests for get_curriculum_config function."""

    def test_get_curriculum_config_by_name(self):
        """Should get config by stage name."""
        config = get_curriculum_config("strong_gumbel", "hex8", 2)
        assert config.engine_mode == EngineMode.GUMBEL_MCTS
        assert config.board_type == "hex8"
        assert config.num_players == 2

    def test_get_curriculum_config_by_stage(self):
        """Should get config by stage object."""
        stage = CURRICULUM_STAGES["explore_random"]
        config = get_curriculum_config(stage, "square19", 4)
        assert config.engine_mode == EngineMode.RANDOM
        assert config.board_type == "square19"
        assert config.num_players == 4

    def test_get_curriculum_config_invalid_stage(self):
        """Should raise for invalid stage name."""
        with pytest.raises(ValueError, match="Unknown curriculum stage"):
            get_curriculum_config("invalid_stage")

    def test_curriculum_config_source(self):
        """Should set source based on stage name."""
        config = get_curriculum_config("strong_gumbel")
        assert config.source == "curriculum_strong_gumbel"


class TestGetFullCurriculum:
    """Tests for get_full_curriculum function."""

    def test_get_full_curriculum_default(self):
        """Should return full curriculum progression."""
        configs = get_full_curriculum("hex8", 2)
        assert len(configs) == 6  # Default has 6 stages
        assert configs[0].engine_mode == EngineMode.RANDOM
        assert configs[-1].engine_mode == EngineMode.GUMBEL_MCTS

    def test_get_full_curriculum_custom_stages(self):
        """Should accept custom stage list."""
        configs = get_full_curriculum("hex8", 2, stages=["explore_random", "strong_full"])
        assert len(configs) == 2
        assert configs[0].engine_mode == EngineMode.RANDOM
        assert configs[1].engine_mode == EngineMode.GUMBEL_MCTS


class TestGetAllConfigsCurriculum:
    """Tests for get_all_configs_curriculum function."""

    def test_get_all_configs_curriculum_count(self):
        """Should return configs for all board/player combinations."""
        configs = get_all_configs_curriculum()
        # 3 board types * 4 player counts * 6 stages = 72 configs
        # Note: Using 3,4 players but square8/square19/hexagonal
        # Actually it's 3 boards * 3 player counts (2,3,4) * 6 stages = 54
        # Let me check the implementation...
        # The function uses player_counts = [2, 3, 4] which is 3 options
        # and board_types = ["square8", "square19", "hexagonal"] which is 3 boards
        # So: 3 * 3 * 6 = 54 configs
        assert len(configs) == 54

    def test_get_all_configs_curriculum_diversity(self):
        """Should cover all board types."""
        configs = get_all_configs_curriculum(stages=["strong_gumbel"])
        board_types = {c.board_type for c in configs}
        player_counts = {c.num_players for c in configs}

        assert "square8" in board_types
        assert "square19" in board_types
        assert "hexagonal" in board_types
        assert 2 in player_counts
        assert 3 in player_counts
        assert 4 in player_counts


class TestListCurriculumStages:
    """Tests for list_curriculum_stages function."""

    def test_list_curriculum_stages(self):
        """Should return stage descriptions."""
        stages = list_curriculum_stages()
        assert isinstance(stages, dict)
        assert "explore_random" in stages
        assert "random" in stages["explore_random"].lower()  # Description mentions random


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_board_type_case_insensitive(self):
        """Board type should be case insensitive."""
        config = SelfplayConfig(board_type="HEX8")
        assert config.board_type == "hex8"

    def test_empty_extra_options(self):
        """Should handle empty extra options."""
        config = SelfplayConfig()
        assert config.extra_options == {}

    def test_all_board_aliases(self):
        """Should normalize all board type aliases."""
        aliases = {
            "hex": "hexagonal",
            "hex8": "hex8",
            "square": "square8",
            "sq8": "square8",
            "sq19": "square19",
            "full_hex": "hexagonal",
            "full-hex": "hexagonal",
            "fullhex": "hexagonal",
            "hex24": "hexagonal",
        }
        for alias, expected in aliases.items():
            config = SelfplayConfig(board_type=alias)
            assert config.board_type == expected, f"Failed for alias '{alias}'"

    def test_simulation_budget_none(self):
        """Should handle None simulation budget."""
        config = SelfplayConfig(simulation_budget=None, difficulty=None, model_elo=None)
        # Should fall back to default
        budget = config.get_effective_budget()
        assert budget > 0

    def test_config_key_consistency(self):
        """Config key should be consistent after normalization."""
        config1 = SelfplayConfig(board_type="hex", num_players=2)
        config2 = SelfplayConfig(board_type="hexagonal", num_players=2)
        assert config1.config_key == config2.config_key


class TestArgumentParserEdgeCases:
    """Tests for argument parser edge cases."""

    def test_short_flags(self):
        """Should handle short flags."""
        config = parse_selfplay_args(["-p", "4", "-n", "500", "-e", "mcts"])
        assert config.num_players == 4
        assert config.num_games == 500
        assert config.engine_mode == EngineMode.MCTS

    def test_engine_mode_alias_in_args(self):
        """Should handle engine mode aliases in arguments."""
        config = parse_selfplay_args(["--engine-mode", "gumbel"])
        assert config.engine_mode == EngineMode.GUMBEL_MCTS

    def test_defaults_applied(self):
        """Should apply defaults for missing arguments."""
        config = parse_selfplay_args([])
        assert config.board_type == "square8"
        assert config.num_players == 2
        assert config.engine_mode == EngineMode.NNUE_GUIDED

    def test_elo_adaptive_arguments(self):
        """Should parse Elo-adaptive budget arguments."""
        config = parse_selfplay_args([
            "--model-elo", "1500",
            "--training-epoch", "100",
        ])
        assert config.model_elo == 1500
        assert config.training_epoch == 100
