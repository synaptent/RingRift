# Architecture Overview

This is the entry point for the current `ai-service` architecture.

The supported system is narrower than the historical codebase. Read the documents in this order if you want the current picture rather than the December 2025 planning history.

## Current Runtime

The active fleet is role-based:

- 4 trainer nodes run [`minimal_alphazero_loop.py`](/Users/armand/Development/RingRift/ai-service/scripts/minimal_alphazero_loop.py)
- 2 selfplay-worker nodes run [`policy_selfplay_worker.py`](/Users/armand/Development/RingRift/ai-service/scripts/policy_selfplay_worker.py)
- 1 evaluator node runs evaluation services
- all nodes keep P2P active for model sync, health, and coordination

The deployment contract is systemd-managed and role-aware through [`deploy_training_service.sh`](/Users/armand/Development/RingRift/ai-service/scripts/deploy_training_service.sh).

## Core Data Flow

1. Canonical rules live in the TypeScript engine under [`src/shared/engine`](/Users/armand/Development/RingRift/src/shared/engine).
2. Python mirrors those rules for replay, training, and parity.
3. Trainers generate canonical `iter_*.npz` artifacts.
4. Selfplay workers generate policy-bearing JSONL, then ingest it into supplemental NPZ shards.
5. Supplemental shards land in trainer supplemental directories without breaking the trainer’s lexical `iter_*.npz` window.

For the current autonomy/runtime lane, the important operational fact is:

- worker shard landing is proven end-to-end
- trainer merge is expected during the next trainer combine step

## Recommended Reading Order

### System shape

- [DAEMON_SYSTEM_ARCHITECTURE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/DAEMON_SYSTEM_ARCHITECTURE.md)
- [P2P_ORCHESTRATOR_ARCHITECTURE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/P2P_ORCHESTRATOR_ARCHITECTURE.md)
- [TRAINING_LOOPS.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/TRAINING_LOOPS.md)
- [UNIFIED_DATA_PLANE_DESIGN.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/UNIFIED_DATA_PLANE_DESIGN.md)

### Coordination and eventing

- [COORDINATION_SYSTEM.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/COORDINATION_SYSTEM.md)
- [DAEMON_LIFECYCLE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/DAEMON_LIFECYCLE.md)
- [HEALTH_MONITORING.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/HEALTH_MONITORING.md)
- [EVENT_SYSTEM_AUDIT.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/EVENT_SYSTEM_AUDIT.md)

### AI and model path

- [NEURAL_AI_ARCHITECTURE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/NEURAL_AI_ARCHITECTURE.md)
- [MODEL_REGISTRY.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/MODEL_REGISTRY.md)
- [GPU_MODULES_ARCHITECTURE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/GPU_MODULES_ARCHITECTURE.md)

### Sync and distribution

- [SYNC_INFRASTRUCTURE_ARCHITECTURE.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/SYNC_INFRASTRUCTURE_ARCHITECTURE.md)
- [P2P_ORCHESTRATION.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/P2P_ORCHESTRATION.md)
- [RESOURCE_MANAGEMENT.md](/Users/armand/Development/RingRift/ai-service/docs/architecture/RESOURCE_MANAGEMENT.md)

## Historical Material

Many planning and roadmap documents under [`docs/planning`](/Users/armand/Development/RingRift/ai-service/docs/planning), [`docs/roadmaps`](/Users/armand/Development/RingRift/ai-service/docs/roadmaps), and [`docs/archive`](/Users/armand/Development/RingRift/ai-service/docs/archive) are useful historical context, but they do not define the current supported runtime.

Use them for archaeology, not as the source of truth for the live system.
