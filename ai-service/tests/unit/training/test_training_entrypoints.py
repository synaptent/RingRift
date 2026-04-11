"""Tests for the thin training entrypoint wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.training.config import TrainConfig
from app.training.train_config import FullTrainingConfig
from app.training.training_entrypoints import train_from_file, train_with_config


def test_train_from_file_delegates_to_train_model(tmp_path: Path) -> None:
    config = TrainConfig()
    output_path = tmp_path / "model.pth"

    with patch(
        "app.training.train.train_model",
        return_value={
            "best_val_loss": 0.25,
            "epochs_completed": 3,
            "epoch_losses": [0.5, 0.25],
        },
    ) as mock_train_model:
        result = train_from_file(
            data_path="dataset.npz",
            output_path=str(output_path),
            config=config,
            initial_model_path="initial.pth",
        )

    assert result["total"] == 0.25
    assert result["epochs_completed"] == 3
    assert result["epoch_losses"] == [0.5, 0.25]
    mock_train_model.assert_called_once()
    assert mock_train_model.call_args.kwargs["data_path"] == "dataset.npz"
    assert mock_train_model.call_args.kwargs["save_path"] == str(output_path)
    assert mock_train_model.call_args.kwargs["resume_path"] == "initial.pth"


def test_train_with_config_builds_train_config_and_delegates(tmp_path: Path) -> None:
    full_config = FullTrainingConfig()
    full_config.board_type = "hex8"
    full_config.num_players = 2
    full_config.epochs = 7
    full_config.batch_size = 64
    full_config.learning_rate = 0.0005
    full_config.data.data_path = "training_data.npz"
    full_config.checkpoint.save_path = str(tmp_path / "out.pth")

    with patch("app.training.train.train_model", return_value={"status": "ok"}) as mock_train_model:
        result = train_with_config(full_config)

    assert result == {"status": "ok"}
    mock_train_model.assert_called_once()
    train_config = mock_train_model.call_args.kwargs["config"]
    assert isinstance(train_config, TrainConfig)
    assert str(train_config.board_type).lower().endswith("hex8")
    assert train_config.num_players == 2
    assert train_config.epochs_per_iter == 7
    assert train_config.batch_size == 64
    assert train_config.learning_rate == 0.0005
    assert mock_train_model.call_args.kwargs["data_path"] == "training_data.npz"
    assert mock_train_model.call_args.kwargs["save_path"] == str(tmp_path / "out.pth")
