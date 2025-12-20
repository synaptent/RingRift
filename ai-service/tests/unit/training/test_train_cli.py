"""Tests for app.training.train_cli module."""

import pytest

from app.training.train_cli import parse_args


class TestParseArgs:
    """Tests for parse_args function."""

    def test_default_values(self):
        """Test that defaults are set correctly."""
        args = parse_args([])

        assert args.config is None
        assert args.data_path is None
        assert args.save_path is None
        assert args.epochs is None
        assert args.batch_size is None
        assert args.learning_rate is None
        assert args.seed is None
        assert args.checkpoint_dir == "checkpoints"
        assert args.checkpoint_interval == 5
        assert args.distributed is False

    def test_data_path(self):
        """Test --data-path argument."""
        args = parse_args(["--data-path", "/path/to/data.npz"])
        assert args.data_path == "/path/to/data.npz"

    def test_save_path(self):
        """Test --save-path argument."""
        args = parse_args(["--save-path", "/path/to/model.pth"])
        assert args.save_path == "/path/to/model.pth"

    def test_epochs(self):
        """Test --epochs argument."""
        args = parse_args(["--epochs", "100"])
        assert args.epochs == 100

    def test_batch_size(self):
        """Test --batch-size argument."""
        args = parse_args(["--batch-size", "64"])
        assert args.batch_size == 64

    def test_learning_rate(self):
        """Test --learning-rate argument."""
        args = parse_args(["--learning-rate", "0.001"])
        assert args.learning_rate == 0.001

    def test_seed(self):
        """Test --seed argument."""
        args = parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_board_type(self):
        """Test --board-type argument."""
        for board_type in ["square8", "square19", "hex8", "hexagonal"]:
            args = parse_args(["--board-type", board_type])
            assert args.board_type == board_type

    def test_model_version(self):
        """Test --model-version argument."""
        for version in ["v2", "v2_lite", "v3", "v3_lite", "v4"]:
            args = parse_args(["--model-version", version])
            assert args.model_version == version

    def test_distributed_flag(self):
        """Test --distributed flag."""
        args = parse_args(["--distributed"])
        assert args.distributed is True

    def test_local_rank(self):
        """Test --local-rank argument."""
        args = parse_args(["--local-rank", "0"])
        assert args.local_rank == 0

    def test_early_stopping(self):
        """Test --early-stopping-patience argument."""
        args = parse_args(["--early-stopping-patience", "10"])
        assert args.early_stopping_patience == 10

    def test_checkpoint_args(self):
        """Test checkpoint-related arguments."""
        args = parse_args([
            "--checkpoint-dir", "/path/to/checkpoints",
            "--checkpoint-interval", "10",
            "--resume", "/path/to/checkpoint.pt"
        ])
        assert args.checkpoint_dir == "/path/to/checkpoints"
        assert args.checkpoint_interval == 10
        assert args.resume == "/path/to/checkpoint.pt"

    def test_lr_scheduler(self):
        """Test --lr-scheduler argument."""
        for scheduler in ["cosine", "step", "plateau", "warmrestart"]:
            args = parse_args(["--lr-scheduler", scheduler])
            assert args.lr_scheduler == scheduler

    def test_warmup_epochs(self):
        """Test --warmup-epochs argument."""
        args = parse_args(["--warmup-epochs", "5"])
        assert args.warmup_epochs == 5

    def test_multi_player(self):
        """Test --multi-player and --num-players arguments."""
        args = parse_args(["--multi-player", "--num-players", "4"])
        assert args.multi_player is True
        assert args.num_players == 4

    def test_augmentation_flag(self):
        """Test --augment-hex-symmetry flag."""
        args = parse_args(["--augment-hex-symmetry"])
        assert args.augment_hex_symmetry is True

    def test_sampling_weights(self):
        """Test --sampling-weights argument."""
        for weight in ["uniform", "recency", "policy_entropy"]:
            args = parse_args(["--sampling-weights", weight])
            assert args.sampling_weights == weight

    def test_hot_data_buffer_args(self):
        """Test hot data buffer arguments."""
        args = parse_args([
            "--use-hot-data-buffer",
            "--hot-buffer-size", "5000",
            "--hot-buffer-mix-ratio", "0.5"
        ])
        assert args.use_hot_data_buffer is True
        assert args.hot_buffer_size == 5000
        assert args.hot_buffer_mix_ratio == 0.5

    def test_integrated_enhancements(self):
        """Test integrated enhancement flags."""
        args = parse_args([
            "--use-integrated-enhancements",
            "--enable-curriculum",
            "--enable-augmentation",
            "--enable-elo-weighting",
        ])
        assert args.use_integrated_enhancements is True
        assert args.enable_curriculum is True
        assert args.enable_augmentation is True
        assert args.enable_elo_weighting is True

    def test_fault_tolerance_args(self):
        """Test fault tolerance arguments."""
        args = parse_args([
            "--disable-circuit-breaker",
            "--disable-anomaly-detection",
            "--gradient-clip-mode", "fixed",
            "--gradient-clip-max-norm", "2.0",
        ])
        assert args.disable_circuit_breaker is True
        assert args.disable_anomaly_detection is True
        assert args.gradient_clip_mode == "fixed"
        assert args.gradient_clip_max_norm == 2.0

    def test_cmaes_args(self):
        """Test CMA-ES heuristic optimization arguments."""
        args = parse_args([
            "--cmaes-heuristic",
            "--cmaes-generations", "30",
            "--cmaes-population-size", "32",
        ])
        assert args.cmaes_heuristic is True
        assert args.cmaes_generations == 30
        assert args.cmaes_population_size == 32

    def test_curriculum_args(self):
        """Test curriculum training arguments."""
        args = parse_args([
            "--curriculum",
            "--curriculum-generations", "15",
            "--curriculum-games-per-gen", "500",
        ])
        assert args.curriculum is True
        assert args.curriculum_generations == 15
        assert args.curriculum_games_per_gen == 500

    def test_regularization_args(self):
        """Test regularization arguments."""
        args = parse_args([
            "--dropout", "0.1",
            "--policy-label-smoothing", "0.05",
        ])
        assert args.dropout == 0.1
        assert args.policy_label_smoothing == 0.05

    def test_config_file(self):
        """Test --config argument."""
        args = parse_args(["--config", "/path/to/config.yaml"])
        assert args.config == "/path/to/config.yaml"


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with app.training.train."""

    def test_import_from_train(self):
        """Test that parse_args can be imported from train module."""
        from app.training.train import parse_args as parse_args_train

        # Should work the same
        args = parse_args_train(["--epochs", "10"])
        assert args.epochs == 10

    def test_main_import_from_train(self):
        """Test that main can be imported from train module."""
        from app.training.train import main

        # Should be callable
        assert callable(main)
