# RingRift Documentation Index

**Start Here** for a guide to the project's documentation and structure.

## 🚀 Quick Links

- **Setup & Run:** [QUICKSTART.md](../QUICKSTART.md:1) - How to install and run the project.
- **Current Status:** [CURRENT_STATE_ASSESSMENT.md](../CURRENT_STATE_ASSESSMENT.md:1) - What works, what doesn't, and verified code status.
- **Roadmap:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:1) - Future plans and milestones.
- **AI, Rules & Training:** [AI_ARCHITECTURE.md](../AI_ARCHITECTURE.md:1) – High-level AI/rules/training architecture; [docs/AI_TRAINING_AND_DATASETS.md](./AI_TRAINING_AND_DATASETS.md:1) – AI service training & dataset pipelines; [docs/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md](./INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md:1) – incident report and fix for the TerritoryMutator divergence.

## 📖 Rules & Design

- **Complete Rules:** [ringrift_complete_rules.md](../ringrift_complete_rules.md) - The authoritative rulebook.
- **Compact Rules:** [ringrift_compact_rules.md](../ringrift_compact_rules.md) - Implementation-focused summary.
- **Known Issues:** [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) - Current bugs and gaps.

## 🏗️ Project Structure

RingRift is a monorepo-style project with three main components:

1.  **Backend (`src/server/`)**: Node.js/Express/TypeScript. Handles game state, WebSockets, and API.
    - **Engine:** `src/server/game/` (GameEngine, RuleEngine).
2.  **Frontend (`src/client/`)**: React/TypeScript/Vite.
    - **Sandbox:** `src/client/sandbox/` (Client-side engine for testing/prototyping).
3.  **AI Service (`ai-service/`)**: Python/FastAPI.
    - Provides AI moves and heuristics.

## 🧪 Testing

- **Guide:** [tests/README.md](../tests/README.md)
- **Parity:** We maintain parity between the Backend Engine and the Client Sandbox Engine. See `tests/unit/Backend_vs_Sandbox.*` for parity tests.

## 🤝 Contributing

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Guidelines for contributing code.

## 🛠 Operations & Database

- **Database operations & migrations:** [docs/OPERATIONS_DB.md](./OPERATIONS_DB.md:1)
- **Environment setup & staging stack:** [QUICKSTART.md](../QUICKSTART.md:71)

## ⚙️ Performance & Scalability

- **SLOs & load scenarios:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:155) - Performance & Scalability (P-01) section.

## 🔐 Security & Threat Model

- **Threat model & hardening plan:** [docs/SECURITY_THREAT_MODEL.md](./SECURITY_THREAT_MODEL.md:1) – S‑05 assets, trust boundaries, attacker profiles, major threat surfaces, existing controls, gaps, and the prioritized security backlog.
- **Supply chain & CI/CD safeguards (S‑05.F):** [docs/SUPPLY_CHAIN_AND_CI_SECURITY.md](./SUPPLY_CHAIN_AND_CI_SECURITY.md:1) – supply-chain & CI/CD threat overview, current controls vs gaps, and the S‑05.F.x implementation tracks for dependency, CI, Docker, and secret-management hardening.
- **Data retention, privacy & user data (S‑05.E):** [docs/DATA_LIFECYCLE_AND_PRIVACY.md](./DATA_LIFECYCLE_AND_PRIVACY.md:1) – data inventory, retention/anonymization policies, and account deletion/export workflows, designed for incremental implementation.
- **Roadmap summary:** [STRATEGIC_ROADMAP.md](../STRATEGIC_ROADMAP.md:155) – 🔐 Security & Threat Model (S‑05) section pointing to the canonical threat model and data lifecycle documents.
