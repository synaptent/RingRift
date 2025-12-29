# AI Service Documentation

Technical documentation for the RingRift AI training service.

## Quick Links

| Document                                                       | Description                   |
| -------------------------------------------------------------- | ----------------------------- |
| [Quick Start](QUICK_START.md)                                  | Get up and running quickly    |
| [Developer Guide](DEVELOPER_GUIDE.md)                          | Coding patterns and standards |
| [Config Reference](CONFIG_REFERENCE.md)                        | All configuration options     |
| [API Reference](API_REFERENCE.md)                              | FastAPI endpoints             |
| [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md) | System architecture           |
| [Training Pipeline](training/TRAINING_PIPELINE.md)             | Training pipeline overview    |

## P2P + Sync SSoT

- Cluster inventory + voters: `ai-service/config/distributed_hosts.yaml` (`p2p_voters`, per-host `p2p_enabled`)
- Canonical P2P start: `PYTHONPATH=. venv/bin/python scripts/p2p_orchestrator.py --node-id <name> --port 8770 --peers <url-list>`
- Data sync service: `PYTHONPATH=. venv/bin/python scripts/unified_data_sync.py --watchdog --http-port 8765`
- External storage path: `/Volumes/RingRift-Data` on mac-studio via `allowed_external_storage`

## Documentation Structure

### [architecture/](architecture)

System design and component architecture.

- Neural network architecture
- GPU pipeline design (see also [GPU_VECTORIZATION.md](GPU_VECTORIZATION.md))
- Coordination systems (see also [COORDINATION_ARCHITECTURE.md](COORDINATION_ARCHITECTURE.md))
- Platform-specific (MPS, CUDA)

### [training/](training)

Training pipeline and methodology.

- Training features and internals
- Optimization techniques
- Curriculum learning
- Model promotion and Elo

### [infrastructure/](infrastructure)

Cluster setup, operations, and cloud infrastructure.

- Cluster setup and operations
- Vast.ai integration
- P2P orchestration
- Resource management

### [algorithms/](algorithms)

AI algorithms and search methods.

- Gumbel MCTS
- MCTS integration
- Hex board augmentation
- NNUE policy training

### [specs/](specs)

Game notation and data format specifications.

- Game notation spec
- Game record format
- Replay database design
- Make/unmake design

### [roadmaps/](roadmaps)

Active development roadmaps and plans.

- GPU pipeline roadmap
- Consolidation roadmap
- Integration migration plan

### [runbooks/](runbooks)

Operational runbooks for incident response.

### [audits/](audits)

Generated audits and code health reports.

- [Circular Dependency Map](audits/CIRCULAR_DEPENDENCY_MAP.md)

- Cluster health critical
- Coordinator errors
- Sync host issues

### [archive/](archive)

Historical documentation and status reports.

### Historical Reports

- `ai-service/docs/archive/status_reports/README.md` - Point-in-time operational reports (Dec 2025)

## Root Level Docs

| Document                                                              | Description                      |
| --------------------------------------------------------------------- | -------------------------------- |
| [QUICK_START](QUICK_START.md)                                         | Quick start guide                |
| [DEVELOPER_GUIDE](DEVELOPER_GUIDE.md)                                 | Coding patterns and standards    |
| [CONFIG_REFERENCE](CONFIG_REFERENCE.md)                               | Complete config reference        |
| [GPU_VECTORIZATION](GPU_VECTORIZATION.md)                             | GPU module architecture & limits |
| [COORDINATION_ARCHITECTURE](COORDINATION_ARCHITECTURE.md)             | Event system & coordination      |
| [P2P_HANDLERS](P2P_HANDLERS.md)                                       | P2P orchestrator handler mixins  |
| [EVENT_CATALOG](EVENT_CATALOG.md)                                     | Event types reference            |
| [sync_architecture](sync_architecture.md)                             | Canonical sync architecture      |
| [CONSOLIDATION_STATUS_2025_12_19](CONSOLIDATION_STATUS_2025_12_19.md) | Historical consolidation status  |
| [HEX_ARTIFACTS_DEPRECATED](HEX_ARTIFACTS_DEPRECATED.md)               | Deprecated hex data notice       |

## See Also

- [Main docs/](../../docs) - Product documentation
- `ai-service/TRAINING_DATA_REGISTRY.md` - Training data inventory and provenance
- [RULES_CANONICAL_SPEC](../../RULES_CANONICAL_SPEC.md) - Canonical rules
