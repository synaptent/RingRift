# RingRift AI Service Docs

This directory contains deeper technical documentation for the Python side of RingRift.

If you are new to the project, do not start here. Start with:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
3. [QUICKSTART.md](/Users/armand/Development/RingRift/QUICKSTART.md)
4. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
5. [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
6. [ai-service/README.md](/Users/armand/Development/RingRift/ai-service/README.md)

This doc hub is for readers who already know they need the AI-service internals.

## Supported Entry Points

### Inference service

- [../README.md](/Users/armand/Development/RingRift/ai-service/README.md)
- [API_REFERENCE.md](/Users/armand/Development/RingRift/ai-service/docs/API_REFERENCE.md)
- [CONFIG_REFERENCE.md](/Users/armand/Development/RingRift/ai-service/docs/CONFIG_REFERENCE.md)

### Training and reproducible experiments

- [../README.md](/Users/armand/Development/RingRift/ai-service/README.md)
- [../scripts/README.md](/Users/armand/Development/RingRift/ai-service/scripts/README.md)
- [training/TRAINING_FEATURES.md](/Users/armand/Development/RingRift/ai-service/docs/training/TRAINING_FEATURES.md)
- [roadmaps/GPU_PIPELINE_ROADMAP.md](/Users/armand/Development/RingRift/ai-service/docs/roadmaps/GPU_PIPELINE_ROADMAP.md)
- [../TRAINING_DATA_REGISTRY.md](/Users/armand/Development/RingRift/ai-service/TRAINING_DATA_REGISTRY.md)

### Parity and canonical data trust

- [../../docs/PARITY_RUNBOOK.md](/Users/armand/Development/RingRift/docs/PARITY_RUNBOOK.md)
- [../../docs/rules/PYTHON_PARITY_REQUIREMENTS.md](/Users/armand/Development/RingRift/docs/rules/PYTHON_PARITY_REQUIREMENTS.md)
- [../../docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md](/Users/armand/Development/RingRift/docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md)

## Directory Guide

### `architecture/`

Design docs for the AI-service internals, GPU pipelines, and system decomposition.

### `training/`

Training pipeline details, feature notes, and implementation-specific references.

### `algorithms/`

Algorithm-specific documentation for search methods and experimental model variants.

### `specs/`

Data, replay, and format specifications used by the Python training and replay stack.

### `roadmaps/`

Planning documents for AI-service evolution. Useful context, but not always current implementation truth.

### `runbooks/`

Operational runbooks for AI-service incidents and maintenance.

### `infrastructure/`

Cluster and environment documentation. Useful for operators, not the first stop for understanding the supported training path.

### `archive/`

Historical notes and superseded docs.

## What To Treat As Canonical

Within the AI service, the main rules are:

- TypeScript rules behavior is authoritative.
- Python mirrors it for inference, replay validation, and training.
- Canonical replay data and parity checks matter more than historical convenience.

If you need the actual rules source of truth, go back to:

- [RULES_CANONICAL_SPEC.md](/Users/armand/Development/RingRift/RULES_CANONICAL_SPEC.md)
- [docs/rules/COMPLETE_RULES.md](/Users/armand/Development/RingRift/docs/rules/COMPLETE_RULES.md)
- [src/shared/engine](/Users/armand/Development/RingRift/src/shared/engine)

## Bottom Line

The shortest trustworthy AI-service path is:

1. top-level project docs
2. [ai-service/README.md](/Users/armand/Development/RingRift/ai-service/README.md)
3. parity docs
4. training docs
5. deeper architecture and infrastructure references only as needed
