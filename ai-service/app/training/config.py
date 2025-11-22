from dataclasses import dataclass
from app.models import BoardType


@dataclass
class TrainConfig:
    """Configuration for training run"""
    board_type: BoardType = BoardType.SQUARE8
    episodes_per_iter: int = 4
    epochs_per_iter: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    history_length: int = 3
    seed: int = 42
    max_moves_per_game: int = 200
    k_elo: int = 32
    policy_weight: float = 1.0
    
    # Paths
    data_dir: str = "ai-service/app/training/data"
    model_dir: str = "ai-service/app/models"
    log_dir: str = "ai-service/app/logs"
    
    def __post_init__(self):
        import os
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)