# Architecture Overview

This directory mixes current-state architecture docs, active roadmaps, and historical decomposition plans. Start here, then branch into the smallest document that answers your question.

## Start Here

- [RULES_ENGINE_ARCHITECTURE.md](/docs/architecture/RULES_ENGINE_ARCHITECTURE.md)
  Current structure of the TypeScript rules engine, which remains the executable source of truth.

- [CANONICAL_ENGINE_API.md](/docs/architecture/CANONICAL_ENGINE_API.md)
  Canonical engine-facing contracts and boundaries.

- [AI_ARCHITECTURE.md](/docs/architecture/AI_ARCHITECTURE.md)
  High-level Python AI-service architecture and model/training context.

- [MINIMAL_LOOP_CONTRACT.md](/docs/architecture/MINIMAL_LOOP_CONTRACT.md)
  Supported trainer-loop contract for the current minimal-loop fleet.

- [PRODUCTION_VS_TRAINING.md](/docs/architecture/PRODUCTION_VS_TRAINING.md)
  Separation between live product/runtime paths and training infrastructure.

## Current Runtime And Coordination

- [EVENT_SYSTEM.md](/docs/architecture/EVENT_SYSTEM.md)
  Event bus and routing model.

- [SYNC_ARCHITECTURE.md](/docs/architecture/SYNC_ARCHITECTURE.md)
  Sync/distribution architecture and data movement.

- [PHASE_ORCHESTRATION_ARCHITECTURE.md](/docs/architecture/PHASE_ORCHESTRATION_ARCHITECTURE.md)
  Turn and phase orchestration model.

- [TOPOLOGY_MODES.md](/docs/architecture/TOPOLOGY_MODES.md)
  Deployment/topology modes across environments.

- [MODULE_RESPONSIBILITIES.md](/docs/architecture/MODULE_RESPONSIBILITIES.md)
  Useful orientation map for major module ownership.

## Data, APIs, And Persistence

- [DATABASE_SCHEMA.md](/docs/architecture/DATABASE_SCHEMA.md)
  Persistent schema reference.

- [API_REFERENCE.md](/docs/architecture/API_REFERENCE.md)
  Backend API reference.

- [WEBSOCKET_API.md](/docs/architecture/WEBSOCKET_API.md)
  Realtime protocol reference.

## Active Roadmaps

- [PART4_QUALITY_ROADMAP.md](/docs/architecture/PART4_QUALITY_ROADMAP.md)
  Current code-quality and autonomy cleanup roadmap.

- [PART3_INFRASTRUCTURE_ROADMAP.md](/docs/architecture/PART3_INFRASTRUCTURE_ROADMAP.md)
  Infrastructure roadmap context that still informs the fleet/runtime path.

- [NEXT_STEPS.md](/docs/architecture/NEXT_STEPS.md)
  Near-term follow-up work.

## Historical Design Work

These are useful for context, but they are not the current source of truth by default:

- `*_PLAN.md`
- `*_PROPOSAL.md`
- `*_STUDY.md`
- `*_AUDIT.md`
- `*_RECOMMENDATIONS.md`

Examples:

- [CLIENT_DECOMPOSITION_PLAN.md](/docs/architecture/CLIENT_DECOMPOSITION_PLAN.md)
- [SERVER_DECOMPOSITION_PLAN.md](/docs/architecture/SERVER_DECOMPOSITION_PLAN.md)
- [TURN_ORCHESTRATOR_MODULARIZATION_STUDY.md](/docs/architecture/TURN_ORCHESTRATOR_MODULARIZATION_STUDY.md)
- [ARCHITECTURAL_DEBT_ASSESSMENT.md](/docs/architecture/ARCHITECTURAL_DEBT_ASSESSMENT.md)

Use these when you need background on why the current structure exists, not when you need the current contract.
