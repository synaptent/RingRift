# Integration Module

Pipeline integration components connecting training, evaluation, promotion, and P2P coordination.

## Overview

`app.integration` is a thin root facade over the main coordination submodules:

- Pipeline feedback: `PipelineFeedbackController`, `FeedbackAction`, `FeedbackSignal`, `FeedbackSignalRouter`
- Model lifecycle: `ModelLifecycleManager`, `LifecycleConfig`
- P2P cluster integration: `P2PIntegrationManager`, `P2PIntegrationConfig`
- Evaluation-to-curriculum bridge: `EvaluationCurriculumBridge`

For richer types that are not re-exported at the package root, import from the canonical submodule directly. Common examples are `EvaluationResult` and `LifecycleStage` from `app.integration.model_lifecycle`.

## Supported Root Imports

```python
from app.integration import (
    EvaluationCurriculumBridge,
    FeedbackAction,
    FeedbackSignal,
    FeedbackSignalRouter,
    LifecycleConfig,
    ModelLifecycleManager,
    P2PIntegrationConfig,
    P2PIntegrationManager,
    PipelineFeedbackController,
    create_evaluation_bridge,
    create_feedback_controller,
    create_feedback_router,
    create_lifecycle_manager,
)
```

## Key Workflows

### Pipeline Feedback

```python
from pathlib import Path

from app.integration import (
    FeedbackAction,
    create_feedback_controller,
    create_feedback_router,
)

controller = create_feedback_controller(Path("."))
router = create_feedback_router()


async def handle_cmaes(signal):
    return True


router.register_handler(FeedbackAction.TRIGGER_CMAES, handle_cmaes, name="cmaes")

await controller.on_stage_complete(
    "evaluation",
    {
        "config_key": "hex8_2p",
        "elo": 1530.0,
        "win_rate": 0.42,
        "games_played": 40,
    },
)

summary = controller.get_state_summary()
pending = controller.get_pending_actions()
```

### Model Lifecycle

```python
from pathlib import Path

from app.integration import LifecycleConfig, ModelLifecycleManager
from app.integration.model_lifecycle import EvaluationResult

manager = ModelLifecycleManager(
    LifecycleConfig(
        registry_dir="data/model_registry",
        model_storage_dir="data/models",
    )
)

model_id, version = await manager.register_model(
    name="hex8_2p_v3",
    model_path=Path("data/models/hex8_2p_v3.pth"),
    training_config={"board_type": "hex8", "num_players": 2},
    tags=["candidate"],
)

await manager.submit_evaluation(
    model_id,
    version,
    EvaluationResult(
        model_id=model_id,
        version=version,
        elo=1545.0,
        games_played=60,
        win_rate=0.56,
    ),
)
```

### P2P Integration

```python
from app.integration import P2PIntegrationConfig, P2PIntegrationManager

manager = P2PIntegrationManager(
    P2PIntegrationConfig(
        p2p_base_url="http://localhost:8770",
        target_selfplay_games_per_hour=1000,
    )
)

await manager.start()

cycle_status = await manager.start_improvement_cycle(
    phases=["selfplay", "training", "evaluation"]
)
training_status = await manager.trigger_training(wait_for_completion=False)

await manager.stop()
```

### Evaluation → Curriculum Bridge

```python
from pathlib import Path

from app.integration import create_evaluation_bridge, create_feedback_controller

feedback = create_feedback_controller(Path("."))
bridge = create_evaluation_bridge(feedback_controller=feedback)

await feedback.on_stage_complete(
    "evaluation",
    {
        "config_key": "square8_3p",
        "elo": 1534.9,
        "win_rate": 0.20,
        "games_played": 40,
    },
)
```

## See Also

- `app.integration.pipeline_feedback` for controller internals and signal routing details
- `app.integration.model_lifecycle` for `EvaluationResult`, `LifecycleStage`, and promotion logic
- `app.integration.p2p_integration` for bridge internals and cluster operations
- `app.p2p` for the lower-level cluster protocol adapters
- `app.training` for the training pipeline itself
