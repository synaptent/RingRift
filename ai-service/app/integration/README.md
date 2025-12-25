# Integration Module

Pipeline integration components connecting training, evaluation, and optimization.

## Overview

This module provides integration components that connect training, evaluation, and optimization into a cohesive self-improvement system:

- Pipeline feedback loops
- Model lifecycle management
- P2P cluster integration

## Key Components

### Pipeline Feedback

```python
from app.integration import PipelineFeedback, FeedbackType

# Create feedback loop
feedback = PipelineFeedback()

# Record training result
feedback.record(
    feedback_type=FeedbackType.TRAINING_COMPLETE,
    model_id="hex8_2p_v3",
    metrics={"loss": 0.025, "accuracy": 0.76},
)

# Get recommendations for next training cycle
recommendations = feedback.get_recommendations()
```

### Model Lifecycle

```python
from app.integration import ModelLifecycle, ModelStage

# Track model through stages
lifecycle = ModelLifecycle()

# Register new model
lifecycle.register_model(
    model_id="hex8_2p_v3",
    stage=ModelStage.TRAINING,
)

# Promote after evaluation
lifecycle.promote_model(
    model_id="hex8_2p_v3",
    stage=ModelStage.EVALUATION,
    elo_delta=45,
)

# Move to production
lifecycle.promote_to_production("hex8_2p_v3")
```

### P2P Integration

```python
from app.integration import P2PIntegration

# Connect to cluster
p2p = P2PIntegration(port=8770)

# Submit training job
job_id = p2p.submit_job(
    job_type="training",
    config={
        "board_type": "hex8",
        "num_players": 2,
        "epochs": 50,
    },
)

# Monitor progress
status = p2p.get_job_status(job_id)
```

## Feedback Types

| Type                 | Description                |
| -------------------- | -------------------------- |
| `TRAINING_COMPLETE`  | Training run finished      |
| `EVALUATION_RESULT`  | Model evaluation completed |
| `PROMOTION_DECISION` | Model promoted or rejected |
| `PIPELINE_ERROR`     | Error in pipeline step     |

## Model Stages

| Stage        | Description                     |
| ------------ | ------------------------------- |
| `TRAINING`   | Model being trained             |
| `EVALUATION` | Model under evaluation          |
| `STAGING`    | Approved for production testing |
| `PRODUCTION` | Active production model         |
| `RETIRED`    | Superseded by newer model       |

## Integration Flow

```
Selfplay → Training → Evaluation → Promotion → Production
    ↑                                    |
    └────── Feedback Loop ←──────────────┘
```

## See Also

- `app.p2p` - P2P orchestration
- `app.evaluation` - Benchmark framework
- `app.training` - Training pipeline
