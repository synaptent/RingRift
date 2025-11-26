# RingRift Documentation Index

**Start Here** for a guide to the project's documentation and structure.

## 🚀 Quick Links

- **Setup & Run:** [QUICKSTART.md](../QUICKSTART.md:1) - How to install and run the project.
- **Current Status:** [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md:1) - What works, what doesn't, and verified code status.
- **Roadmap:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:1) - Future plans and milestones.
- **Rules Engine:** [RULES_ENGINE_ARCHITECTURE.md](../RULES_ENGINE_ARCHITECTURE.md:1) – Complete rules engine architecture including orchestration layer.
- **AI, Rules & Training:** [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md:1) – High-level AI/rules/training architecture; [docs/AI_TRAINING_AND_DATASETS.md](./AI_TRAINING_AND_DATASETS.md:1) – AI service training & dataset pipelines; [docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1) – incident report and fix for the TerritoryMutator divergence.

## 📖 Rules & Design

- **Complete Rules:** [ringrift_complete_rules.md](../ringrift_complete_rules.md) - The authoritative rulebook.
- **Compact Rules:** [ringrift_compact_rules.md](../ringrift_compact_rules.md) - Implementation-focused summary.
- **Known Issues:** [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Current bugs and gaps.

### Supplementary Rules Documentation

- [docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md](./supplementary/RULES_CONSISTENCY_EDGE_CASES.md) - Edge case analysis and handling.
- [docs/supplementary/RULES_RULESET_CLARIFICATIONS.md](./supplementary/RULES_RULESET_CLARIFICATIONS.md) - Clarifications for ambiguous rules.
- [docs/supplementary/AI_IMPROVEMENT_BACKLOG.md](./supplementary/AI_IMPROVEMENT_BACKLOG.md) - Backlog for AI improvements.

### Architecture Remediation Reports (NEW)

The following documents record the 4-phase architecture remediation completed November 2025:

- [docs/drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md](./drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md) - Original consolidation design document.
- [docs/drafts/PHASE1_REMEDIATION_PLAN.md](./drafts/PHASE1_REMEDIATION_PLAN.md) - Production hardening tasks.
- [docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md](./drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md) - Backend and sandbox adapter migration.
- [docs/drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md](./drafts/PHASE4_PYTHON_CONTRACT_TEST_REPORT.md) - Python contract test runner with 100% parity.

### Canonical Orchestrator (NEW)

- [src/shared/engine/orchestration/README.md](../src/shared/engine/orchestration/README.md) - Orchestrator usage and API documentation.
- [docs/CANONICAL_ENGINE_API.md](./CANONICAL_ENGINE_API.md) - Canonical engine public API specification.
- [docs/MODULE_RESPONSIBILITIES.md](./MODULE_RESPONSIBILITIES.md) - Module catalog for shared engine.

### Contract Testing (NEW)

- **Contract Schemas:** [src/shared/engine/contracts/](../src/shared/engine/contracts/) - Contract types and serialization.
- **Test Vectors:** [tests/fixtures/contract-vectors/v2/](../tests/fixtures/contract-vectors/v2/) - 12 test vectors across 5 categories.
- **TS Runner:** [tests/contracts/contractVectorRunner.test.ts](../tests/contracts/contractVectorRunner.test.ts) - TypeScript contract test runner.
- **Python Runner:** [ai-service/tests/contracts/test_contract_vectors.py](../ai-service/tests/contracts/test_contract_vectors.py) - Python contract test runner.

### Drafts

## 🏗️ Project Structure

RingRift is a monorepo-style project with three main components:

1.  **Backend (`src/server/`)**: Node.js/Express/TypeScript. Handles game state, WebSockets, and API.
    - **Engine:** `src/server/game/` (GameEngine, RuleEngine).
2.  **Frontend (`src/client/`)**: React/TypeScript/Vite.
    - **Sandbox:** `src/client/sandbox/` (Client-side engine for testing/prototyping).
3.  **AI Service (`ai-service/`)**: Python/FastAPI.
    - Provides AI moves and heuristics.

## 📡 API Documentation

- **API Reference:** [docs/API_REFERENCE.md](./API_REFERENCE.md:1) – REST API overview, endpoints, error codes, and examples.
- **Interactive Docs:** `/api/docs` – Swagger UI for live API exploration (when server is running).
- **OpenAPI Spec:** `/api/docs.json` – Raw OpenAPI 3.0 specification.

## 🧪 Testing

- **Guide:** [tests/README.md](../tests/README.md)
- **Parity:** We maintain parity between the Backend Engine and the Client Sandbox Engine. See `tests/unit/Backend_vs_Sandbox.*` for parity tests.

## 🤝 Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Guidelines for contributing code.

## 🛠 Operations & Configuration

- **Environment variables reference:** [docs/ENVIRONMENT_VARIABLES.md](./ENVIRONMENT_VARIABLES.md:1) – complete reference for all configuration options, defaults, and validation rules.
- **Database operations & migrations:** [docs/OPERATIONS_DB.md](./OPERATIONS_DB.md:1)
- **Deployment requirements:** [docs/DEPLOYMENT_REQUIREMENTS.md](./DEPLOYMENT_REQUIREMENTS.md:1) – infrastructure requirements per environment.
- **Environment setup & staging stack:** [QUICKSTART.md](../QUICKSTART.md:71)

## 📋 Deployment Runbooks

Step-by-step operational procedures for deploying and managing RingRift. See [docs/runbooks/INDEX.md](./runbooks/INDEX.md) for the complete index.

- **Initial Deployment:** [docs/runbooks/DEPLOYMENT_INITIAL.md](./runbooks/DEPLOYMENT_INITIAL.md) – first-time environment setup.
- **Routine Deployment:** [docs/runbooks/DEPLOYMENT_ROUTINE.md](./runbooks/DEPLOYMENT_ROUTINE.md) – standard release procedures.
- **Rollback Procedures:** [docs/runbooks/DEPLOYMENT_ROLLBACK.md](./runbooks/DEPLOYMENT_ROLLBACK.md) – how to revert to previous versions.
- **Scaling Procedures:** [docs/runbooks/DEPLOYMENT_SCALING.md](./runbooks/DEPLOYMENT_SCALING.md) – how to scale services up/down.
- **Database Migrations:** [docs/runbooks/DATABASE_MIGRATION.md](./runbooks/DATABASE_MIGRATION.md) – Prisma migration procedures.

## 🚨 Incident Response

Procedures for responding to alerts and incidents. See [docs/incidents/INDEX.md](./incidents/INDEX.md) for the complete index with alert-to-guide mapping.

- **Initial Triage:** [docs/incidents/TRIAGE_GUIDE.md](./incidents/TRIAGE_GUIDE.md) – first steps when an alert fires.
- **Availability Incidents:** [docs/incidents/AVAILABILITY.md](./incidents/AVAILABILITY.md) – database down, high error rates, service degradation.
- **Latency Incidents:** [docs/incidents/LATENCY.md](./incidents/LATENCY.md) – response time degradation, slow queries.
- **Resource Incidents:** [docs/incidents/RESOURCES.md](./incidents/RESOURCES.md) – memory, CPU, event loop issues.
- **AI Service Incidents:** [docs/incidents/AI_SERVICE.md](./incidents/AI_SERVICE.md) – AI failures, high fallback rates.
- **Security Incidents:** [docs/incidents/SECURITY.md](./incidents/SECURITY.md) – rate limiting, suspicious activity.
- **Post-Mortem Template:** [docs/incidents/POST_MORTEM_TEMPLATE.md](./incidents/POST_MORTEM_TEMPLATE.md) – post-incident review template.
- **Alerting Thresholds:** [docs/ALERTING_THRESHOLDS.md](./ALERTING_THRESHOLDS.md) – complete alert configuration and rationale.

## ⚙️ Performance & Scalability

- **SLOs & load scenarios:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:155) - Performance & Scalability (P-01) section.

## 🔐 Security & Threat Model

- **Secrets management:** [docs/SECRETS_MANAGEMENT.md](./SECRETS_MANAGEMENT.md:1) – complete guide to secrets handling including inventory, rotation procedures, production requirements, and best practices.
- **Threat model & hardening plan:** [docs/SECURITY_THREAT_MODEL.md](./SECURITY_THREAT_MODEL.md:1) – S‑05 assets, trust boundaries, attacker profiles, major threat surfaces, existing controls, gaps, and the prioritized security backlog.
- **Supply chain & CI/CD safeguards (S‑05.F):** [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](./SUPPLY_CHAIN_AND_CI_SECURITY.md:1) – supply-chain & CI/CD threat overview, current controls vs gaps, and the S‑05.F.x implementation tracks for dependency, CI, Docker, and secret-management hardening.
- **Data retention, privacy & user data (S‑05.E):** [docs/DATA_LIFECYCLE_AND_PRIVACY.md](./DATA_LIFECYCLE_AND_PRIVACY.md:1) – data inventory, retention/anonymization policies, and account deletion/export workflows, designed for incremental implementation.
- **Roadmap summary:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:155) – 🔐 Security & Threat Model (S‑05) section pointing to the canonical threat model and data lifecycle documents.
