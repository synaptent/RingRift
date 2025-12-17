# AI Service Documentation

Technical documentation for the RingRift AI training service.

## Quick Links

| Document                                                       | Description                    |
| -------------------------------------------------------------- | ------------------------------ |
| [Developer Guide](DEVELOPER_GUIDE.md)                          | Getting started for developers |
| [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md) | System architecture            |
| [Training Pipeline](training/TRAINING_PIPELINE.md)             | Training pipeline overview     |

## Documentation Structure

### [architecture/](architecture/)

System design and component architecture.

- Neural network architecture
- GPU pipeline design
- Coordination systems
- Platform-specific (MPS, CUDA)

### [training/](training/)

Training pipeline and methodology.

- Training features and internals
- Optimization techniques
- Curriculum learning
- Model promotion and Elo

### [infrastructure/](infrastructure/)

Cluster setup, operations, and cloud infrastructure.

- Cluster setup and operations
- Vast.ai integration
- P2P orchestration
- Resource management

### [algorithms/](algorithms/)

AI algorithms and search methods.

- Gumbel MCTS
- MCTS integration
- Hex board augmentation
- NNUE policy training

### [specs/](specs/)

Game notation and data format specifications.

- Game notation spec
- Game record format
- Replay database design
- Make/unmake design

### [roadmaps/](roadmaps/)

Active development roadmaps and plans.

- GPU pipeline roadmap
- Consolidation roadmap
- Integration migration plan

### [archive/](archive/)

Historical documentation and status reports.

## Root Level Docs

| Document                                                | Description                |
| ------------------------------------------------------- | -------------------------- |
| [DEVELOPER_GUIDE](DEVELOPER_GUIDE.md)                   | Developer onboarding       |
| [HEX_ARTIFACTS_DEPRECATED](HEX_ARTIFACTS_DEPRECATED.md) | Deprecated hex data notice |

## See Also

- [Main docs/](../../docs/) - Product documentation
- [TRAINING_DATA_REGISTRY](../TRAINING_DATA_REGISTRY.md) - Training data inventory
- [RULES_CANONICAL_SPEC](../../RULES_CANONICAL_SPEC.md) - Canonical rules
