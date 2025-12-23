# Phase 1 Remediation Plan

**Version:** 1.0  
**Created:** November 25, 2025  
**Status:** Planning  
**Scope:** Post-Phase 0 bug fixes and production hardening

---

## 1. Executive Summary

### 1.1 Overview

Phase 1 addresses two categories of work:

1. **Deferred Phase 0 Bug Fixes (Category 1):** Issues identified during Phase 0 that were deferred to maintain focus on critical infrastructure stabilization
2. **Production Hardening (Category 2):** Security, error handling, observability, configuration, and documentation improvements required before production launch

### 1.2 Goals

- Resolve all deferred engine and test infrastructure bugs
- Establish production-ready security posture
- Implement standardized error handling and graceful degradation
- Deploy comprehensive observability infrastructure
- Harden configuration and secrets management
- Complete operational documentation

### 1.3 Phase 0 Completion Reference

| Task ID        | Description                              | Status      |
| -------------- | ---------------------------------------- | ----------- |
| P0-NODE-001    | Node Version Standardization             | ✅ Complete |
| P0-TEST-001    | Jest Test Profile Separation             | ✅ Complete |
| P0-PYTHON-001  | Fix Python ACTIVE-No-Move Invariant      | ✅ Complete |
| P0-PARITY-LINE | Multi-line processing parity verified    | ✅ Complete |
| P0-RULES-001   | Fix Chain Capture Continuation Surface   | ✅ Complete |
| P0-RULES-002   | Fix Overlength Line Reward Semantics     | ✅ Complete |
| P0-TEST-002    | Align RefactoredEngine Test Expectations | ✅ Complete |
| P0-E2E-001     | Fix Playwright E2E Configuration         | ✅ Complete |
| P0-E2E-002     | Create Minimal E2E Happy Path Suite      | ✅ Complete |
| P0-TEST-003    | Fix Lobby Realtime Integration Tests     | ✅ Complete |

---

## 2. Dependency Map

### 2.1 Dependency Summary Table

| Task ID     | Blocked By                | Blocks                 |
| ----------- | ------------------------- | ---------------------- |
| P1-TERR-001 | P0-RULES-001              | P1-FAQ-001             |
| P1-FAQ-001  | P0-RULES-002, P1-TERR-001 | -                      |
| P1-REG-001  | P0-TEST-001               | P1-FAQ-001             |
| P1-JEST-001 | P0-TEST-001               | -                      |
| P1-SEC-001  | -                         | P1-ERR-001             |
| P1-SEC-002  | -                         | P1-ERR-002, P1-CFG-002 |
| P1-ERR-001  | P1-SEC-001                | P1-OBS-001             |
| P1-OBS-001  | P1-ERR-001                | P1-OBS-003, P1-DOC-002 |
| P1-CFG-001  | -                         | P1-CFG-002, P1-CFG-003 |
| P1-DOC-001  | P1-SEC-001, P1-ERR-001    | P1-DOC-002             |

---

## 3. Category 1: Deferred Phase 0 Bug Fixes

### P1-TERR-001: Territory Hex Processing Engine Bug

**Task ID:** P1-TERR-001  
**Type:** Engine Bug  
**Priority:** P0 (Critical)  
**Estimated Complexity:** High

#### Root Cause Hypothesis

The territory processing logic on hexagonal boards has edge cases related to:

1. Cube coordinate handling in disconnection detection
2. Border marker position calculation for hex geometries
3. Interaction between line collapses and territory region processing

#### Proposed Fix Approach

1. Review hex coordinate handling in territory detection helpers
2. Validate border marker calculations for hex geometry
3. Add targeted unit tests for hex-specific scenarios
4. Apply corrections to shared geometry helpers
5. Run parity tests across both GameEngine and ClientSandboxEngine

#### Affected Test Files

- `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`
- `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`
- `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`

#### Acceptance Criteria

- [ ] All tests in GameEngine.territoryDisconnection.hex.test.ts pass
- [ ] All tests in ClientSandboxEngine.territoryDisconnection.hex.test.ts pass
- [ ] Parity tests confirm identical behavior between engines
- [ ] No regressions in square board territory processing

---

### P1-FAQ-001: FAQ Behavior Divergence

**Task ID:** P1-FAQ-001  
**Type:** Parity Issue  
**Priority:** P1 (High)  
**Estimated Complexity:** Very High

#### Root Cause Hypothesis

FAQ scenario tests may exhibit divergences from expected behavior due to:

1. Edge cases in movement/capture validation
2. Subtle differences between rules documentation and implementation
3. Test expectations needing alignment with canonical rules

#### Proposed Fix Approach

1. Audit all FAQ test suites and catalog failures
2. Categorize divergences into implementation bugs, test errors, or doc ambiguities
3. Create fix PRs per category to minimize regression risk
4. Update rules documentation for any clarifications
5. Add regression guards with targeted test coverage

#### Affected Test Files

- `tests/scenarios/FAQ_Q01_Q06.test.ts`
- `tests/scenarios/FAQ_Q07_Q08.test.ts`
- `tests/scenarios/FAQ_Q09_Q14.test.ts`
- `tests/scenarios/FAQ_Q15.test.ts`
- `tests/scenarios/FAQ_Q16_Q18.test.ts`
- `tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`
- `tests/scenarios/FAQ_Q22_Q23.test.ts`

#### Acceptance Criteria

- [ ] All FAQ scenario tests pass (Q01-Q24)
- [ ] No divergences between backend GameEngine and ClientSandboxEngine
- [ ] Rules documentation updated for any clarified behaviors

---

### P1-REG-001: Region Order Choice Engine Bug

**Task ID:** P1-REG-001  
**Type:** Engine Bug  
**Priority:** P1 (High)  
**Estimated Complexity:** Medium

#### Root Cause Hypothesis

Issues in player interaction flow for multiple disconnected regions:

1. `moveId` generation and mapping for territory processing moves
2. WebSocket event emission timing for player_choice_required

#### Proposed Fix Approach

1. Review PlayerInteractionManager integration
2. Validate moveId generation format consistency
3. Test real geometry scenarios with reduced mocking
4. Fix WebSocket timing issues
5. Add sandbox parity tests

#### Affected Test Files

- `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`
- `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`

#### Acceptance Criteria

- [ ] Region order choice integration test passes
- [ ] WebSocket player_choice_required events emit correct payload
- [ ] Sandbox engine exhibits identical behavior

---

### P1-JEST-001: import.meta.env Jest Compatibility

**Task ID:** P1-JEST-001  
**Type:** Infrastructure  
**Priority:** P2 (Medium)  
**Estimated Complexity:** Low

#### Root Cause Hypothesis

Jest's CommonJS module system does not natively support `import.meta.env` (Vite-specific ESM feature), affecting client-side component testing.

#### Proposed Fix Approach

1. Create Jest transformation or module mock for import.meta.env
2. Update jest.config.js for ESM/Vite construct handling
3. Add setupFiles entry in tests/setup-jsdom.ts
4. Document pattern in tests/README.md

#### Affected Test Files

- `tests/unit/ErrorBoundary.test.tsx` (to be created)
- `tests/setup-jsdom.ts`
- `jest.config.js`

#### Acceptance Criteria

- [ ] import.meta.env references work in Jest test environment
- [ ] ErrorBoundary component has test coverage
- [ ] Pattern documented for future client-side tests

---

## 4. Category 2: Phase 1 Production Hardening

### 4.1 Security Hardening (P1-SEC-XXX)

#### P1-SEC-001: Input Validation Schemas

**Task ID:** P1-SEC-001  
**Description:** Extend Zod validation schemas to cover all HTTP and WebSocket inputs  
**Priority:** P0 (Critical)  
**Dependencies:** None  
**Blocks:** P1-ERR-001  
**Estimated Complexity:** Medium

##### Current State

- Partial Zod schemas exist in `src/shared/validation/schemas.ts`
- WebSocket schemas in `src/shared/validation/websocketSchemas.ts`
- Some HTTP routes rely on TypeScript types without runtime validation

##### Proposed Implementation

1. Audit all HTTP endpoints and map to existing validation schemas
2. Create missing schemas for query parameters and headers
3. Add `validateRequest` Express middleware
4. Wire middleware into all routes
5. Export validation schemas for client-side use

##### Definition of Done

- [ ] All HTTP endpoints have explicit Zod validation
- [ ] All WebSocket events validated via WebSocketPayloadSchemas
- [ ] Validation errors return structured 400 responses
- [ ] No routes accept unvalidated user input

---

#### P1-SEC-002: Authentication Flow Improvements

**Task ID:** P1-SEC-002  
**Description:** Harden authentication flows with token rotation and revocation enhancements  
**Priority:** P0 (Critical)  
**Dependencies:** None  
**Blocks:** P1-ERR-002, P1-CFG-002  
**Estimated Complexity:** High

##### Current State

- JWT access tokens with tokenVersion for revocation
- Refresh tokens stored in database
- Basic token extraction from headers, cookies, query params

##### Proposed Implementation

1. Implement refresh token rotation on every use
2. Add token family tracking to detect token theft
3. Implement forced logout admin endpoint
4. Add suspicious login detection and flagging
5. Create auth flow documentation with diagrams
6. Add auth event logging for audit trail

##### Definition of Done

- [ ] Refresh tokens rotate on every use
- [ ] Reused refresh tokens trigger family revocation
- [ ] Admin can force logout any user
- [ ] Suspicious login patterns logged with warning level
- [ ] Auth documentation includes token lifecycle diagram

---

#### P1-SEC-003: Rate Limiting Enhancement

**Task ID:** P1-SEC-003  
**Description:** Extend rate limiting to cover all abuse-sensitive endpoints  
**Priority:** P1 (High)  
**Dependencies:** None  
**Estimated Complexity:** Medium

##### Current State

- Redis-backed rate limiters for api, auth, game, websocket, game creation
- Graceful degradation when Redis unavailable

##### Proposed Implementation

1. Add per-user game creation quotas
2. Add chat/message rate limiting for future features
3. Add failed login lockout logic
4. Add WebSocket message rate limiting per connection
5. Move all limits to environment variables
6. Add X-RateLimit-\* headers to responses

##### Definition of Done

- [ ] All abuse-sensitive endpoints have rate limiting
- [ ] Rate limits configurable via environment variables
- [ ] Rate limit headers included in responses
- [ ] Lockout logic works for repeated failed logins

---

#### P1-SEC-004: CSRF/XSS Protection

**Task ID:** P1-SEC-004  
**Description:** Implement CSRF protection and verify XSS mitigations  
**Priority:** P1 (High)  
**Dependencies:** None  
**Estimated Complexity:** Medium

##### Current State

- SPA architecture reduces CSRF surface (API uses Bearer tokens)
- React JSX escaping provides baseline XSS protection

##### Proposed Implementation

1. Audit state-changing endpoints for CSRF vulnerability
2. Implement CSRF tokens for cookie-based sessions
3. Implement strict Content-Security-Policy headers
4. Audit user-generated content for proper sanitization
5. Add security headers: HSTS, X-Content-Type-Options, X-Frame-Options
6. Create centralized security header middleware

##### Definition of Done

- [ ] CSRF protection for state-changing operations using cookies
- [ ] CSP headers configured and documented
- [ ] All security headers present in responses
- [ ] XSS audit completed for user-generated content

---

### 4.2 Error Handling (P1-ERR-XXX)

#### P1-ERR-001: Error Response Standardization

**Task ID:** P1-ERR-001  
**Description:** Standardize all error responses to consistent structured format  
**Priority:** P0 (Critical)  
**Dependencies:** P1-SEC-001  
**Blocks:** P1-OBS-001  
**Estimated Complexity:** Medium

##### Current State

- Centralized error handler with typed AppError
- Handles Zod, Auth, JWT, and AI service errors
- Logs errors with request context

##### Proposed Implementation

1. Define canonical error envelope with code, message, details, requestId, timestamp
2. Create error codes enum consolidating all codes
3. Update errorHandler to ensure canonical format
4. Add client error mapping to user-friendly messages
5. Document all error codes in OpenAPI spec
6. Remove stack traces in production responses

##### Definition of Done

- [ ] All error responses match canonical format
- [ ] Error codes documented in OpenAPI spec
- [ ] No stack traces in production responses
- [ ] Request ID included in all error responses

---

#### P1-ERR-002: Graceful Degradation Patterns

**Task ID:** P1-ERR-002  
**Description:** Implement graceful degradation for all external dependencies  
**Priority:** P1 (High)  
**Dependencies:** P1-SEC-002  
**Estimated Complexity:** High

##### Current State

- AI service has fallback chain (Remote → Local Heuristic → Random)
- Some graceful handling for Redis unavailability

##### Proposed Implementation

1. Define degradation modes: Full service, Degraded AI, Degraded persistence, Degraded auth
2. Implement health-based degradation auto-detection
3. Add degradation indicators in response headers
4. Implement graceful reconnection for dependency recovery
5. Add degradation alerts
6. Document exactly what works in each degradation mode

##### Definition of Done

- [ ] All external dependencies have degradation handling
- [ ] Service continues operating in degraded mode
- [ ] Health endpoint reports dependency status
- [ ] Automatic recovery when dependencies return

---

#### P1-ERR-003: Retry Logic

**Task ID:** P1-ERR-003  
**Description:** Implement standardized retry logic for transient failures  
**Priority:** P2 (Medium)  
**Dependencies:** None  
**Estimated Complexity:** Medium

##### Current State

- AI service client has basic timeout handling
- No standardized retry pattern across services

##### Proposed Implementation

1. Create withRetry utility for async operations
2. Implement exponential backoff with jitter
3. Define registry of retryable error codes
4. Add retry telemetry and logging
5. Wire into AI service, database, and Redis operations
6. Make retry behavior configurable

##### Definition of Done

- [ ] Retry utility available for all async operations
- [ ] Exponential backoff with jitter implemented
- [ ] Retryable vs non-retryable errors documented
- [ ] Retry metrics captured

---

#### P1-ERR-004: Circuit Breaker Implementation

**Task ID:** P1-ERR-004  
**Description:** Implement circuit breaker pattern for external services  
**Priority:** P2 (Medium)  
**Dependencies:** P1-ERR-003  
**Estimated Complexity:** High

##### Current State

- Circuit breaker mentioned in docs but not implemented
- External service failures can cascade

##### Proposed Implementation

1. Implement CircuitBreaker class with execute method and state tracking
2. Define states: Closed (normal), Open (fail fast), Half-open (test recovery)
3. Add state persistence in Redis
4. Implement per-service breakers for AI service and database
5. Add breaker metrics tracking state transitions
6. Create visibility into circuit states via health endpoint

##### Definition of Done

- [ ] Circuit breaker available for all external services
- [ ] State transitions occur based on failure rates
- [ ] Half-open state allows gradual recovery
- [ ] Circuit breaker state visible in health endpoint

---

### 4.3 Observability (P1-OBS-XXX)

#### P1-OBS-001: Logging Standards and Structured Format

**Task ID:** P1-OBS-001  
**Description:** Establish consistent structured logging standards across all services  
**Priority:** P0 (Critical)  
**Dependencies:** P1-ERR-001  
**Blocks:** P1-OBS-003, P1-DOC-002  
**Estimated Complexity:** Medium

##### Current State

- Winston logger with JSON format in `src/server/utils/logger.ts`
- Request context support via withRequestContext
- Email redaction via redactEmail helper

##### Proposed Implementation

1. Define canonical log schema with timestamp, level, service, requestId, userId, gameId, message, context, error
2. Standardize log levels: ERROR (user-impacting), WARN (degraded), INFO (lifecycle), DEBUG (dev only)
3. Propagate correlation IDs through all layers
4. Implement context enrichment with auto-attached identifiers
5. Add log sanitization ensuring no PII beyond redacted forms
6. Update Python AI service logging to match schema

##### Definition of Done

- [ ] All logs follow standardized schema
- [ ] Correlation IDs present in all requests
- [ ] No PII in logs beyond redacted forms
- [ ] Python AI service logs match TypeScript schema

---

#### P1-OBS-002: Health Check Endpoints

**Task ID:** P1-OBS-002  
**Description:** Implement comprehensive health check endpoints for all services  
**Priority:** P0 (Critical)  
**Dependencies:** None  
**Blocks:** P1-DOC-002  
**Estimated Complexity:** Medium

##### Current State

- Basic /health endpoint exists
- Limited dependency health reporting

##### Proposed Implementation

1. Implement /health/live endpoint for Kubernetes liveness probes
2. Implement /health/ready endpoint for readiness probes
3. Add dependency checks: Database, Redis, AI Service
4. Return structured health response with component status
5. Add health check timeouts to prevent cascade failures
6. Create AI service health endpoint with model status

##### Definition of Done

- [ ] /health/live endpoint returns 200 when process healthy
- [ ] /health/ready endpoint checks all dependencies
- [ ] Structured health response includes component status
- [ ] Health check timeouts prevent blocking
- [ ] AI service health includes model availability

---

#### P1-OBS-003: Metrics Collection

**Task ID:** P1-OBS-003  
**Description:** Implement application metrics collection and export  
**Priority:** P1 (High)  
**Dependencies:** P1-OBS-001  
**Estimated Complexity:** High

##### Current State

- Python AI service has basic Prometheus metrics in `ai-service/app/metrics.py`
- Node.js backend lacks metrics implementation

##### Proposed Implementation

1. Add prom-client to Node.js backend
2. Implement standard metrics: request duration, request count, error rate
3. Add business metrics: games created, moves per second, AI predictions
4. Create /metrics endpoint for Prometheus scraping
5. Add custom metrics for circuit breaker and retry states
6. Document metrics and their meanings

##### Definition of Done

- [ ] Node.js backend exports Prometheus metrics
- [ ] Standard HTTP metrics implemented
- [ ] Business-specific metrics tracked
- [ ] /metrics endpoint available for scraping
- [ ] Metrics documented

---

#### P1-OBS-004: Alerting Thresholds

**Task ID:** P1-OBS-004  
**Description:** Define alerting thresholds and notification rules  
**Priority:** P1 (High)  
**Dependencies:** P1-OBS-003  
**Estimated Complexity:** Medium

##### Current State

- No alerting infrastructure defined
- Manual monitoring required

##### Proposed Implementation

1. Define SLIs: availability, latency, error rate
2. Set SLO targets: 99.5% availability, p95 latency < 500ms
3. Create alert rules for SLO breaches
4. Define escalation policies
5. Create runbook links for each alert
6. Document alert thresholds and response procedures

##### Definition of Done

- [ ] SLIs and SLOs documented
- [ ] Alert rules created for all SLOs
- [ ] Escalation policies defined
- [ ] Each alert links to runbook

---

### 4.4 Configuration (P1-CFG-XXX)

#### P1-CFG-001: Environment Variable Management

**Task ID:** P1-CFG-001  
**Description:** Standardize environment variable management and validation  
**Priority:** P1 (High)  
**Dependencies:** None  
**Blocks:** P1-CFG-002, P1-CFG-003  
**Estimated Complexity:** Medium

##### Current State

- Zod-validated config in `src/server/config.ts`
- Placeholder secret detection for production
- Some variables spread across multiple files

##### Proposed Implementation

1. Consolidate all env vars into single config module
2. Add comprehensive Zod validation for all variables
3. Implement config namespacing by domain (database, auth, ai, etc.)
4. Add config documentation with all available variables
5. Create config validation CLI command
6. Ensure sensible defaults for development

##### Definition of Done

- [ ] All env vars consolidated in config module
- [ ] Comprehensive Zod validation in place
- [ ] Config namespaced by domain
- [ ] All variables documented with defaults
- [ ] Config validation command available

---

#### P1-CFG-002: Secrets Handling Best Practices

**Task ID:** P1-CFG-002  
**Description:** Implement secure secrets management patterns  
**Priority:** P0 (Critical)  
**Dependencies:** P1-SEC-002, P1-CFG-001  
**Estimated Complexity:** Medium

##### Current State

- Secrets in environment variables
- Placeholder detection prevents production start with dummy values
- No secrets rotation mechanism

##### Proposed Implementation

1. Document approved secrets sources (env vars, secrets manager)
2. Implement secrets masking in logs and error messages
3. Add secrets rotation support for database credentials
4. Create secrets audit logging
5. Document secure secrets deployment process
6. Add startup validation for all required secrets

##### Definition of Done

- [ ] Secrets never appear in logs or error messages
- [ ] Secrets rotation documented and supported
- [ ] Secrets audit logging implemented
- [ ] Deployment process documented
- [ ] Startup fails fast on missing secrets

---

#### P1-CFG-003: Feature Flags

**Task ID:** P1-CFG-003  
**Description:** Implement feature flag system for controlled rollouts  
**Priority:** P2 (Medium)  
**Dependencies:** P1-CFG-001  
**Estimated Complexity:** Medium

##### Current State

- No feature flag system
- Features deployed all-or-nothing

##### Proposed Implementation

1. Evaluate feature flag options (env vars, LaunchDarkly, custom)
2. Implement basic feature flag module
3. Add feature flags for: new AI models, experimental rules, UI features
4. Create feature flag admin interface or CLI
5. Implement percentage rollout support
6. Add feature flag telemetry

##### Definition of Done

- [ ] Feature flag module implemented
- [ ] Initial flags defined and documented
- [ ] Percentage rollout supported
- [ ] Admin interface for flag management
- [ ] Feature flag usage logged

---

#### P1-CFG-004: Deployment Configuration Validation

**Task ID:** P1-CFG-004  
**Description:** Add deployment-time configuration validation  
**Priority:** P1 (High)  
**Dependencies:** P1-CFG-001  
**Estimated Complexity:** Low

##### Current State

- Partial validation at startup
- Docker Compose files define some defaults

##### Proposed Implementation

1. Create deployment config schema
2. Validate Docker Compose environment against schema
3. Add CI check for config completeness
4. Create environment comparison tool
5. Document required vs optional config per environment
6. Add config drift detection

##### Definition of Done

- [ ] Deployment config schema defined
- [ ] CI validates config completeness
- [ ] Environment comparison tool available
- [ ] Config requirements documented per environment

---

### 4.5 Documentation (P1-DOC-XXX)

#### P1-DOC-001: OpenAPI/Swagger API Documentation

**Task ID:** P1-DOC-001  
**Description:** Create comprehensive OpenAPI 3.0 API documentation  
**Priority:** P1 (High)  
**Dependencies:** P1-SEC-001, P1-ERR-001  
**Blocks:** P1-DOC-002  
**Estimated Complexity:** Medium

##### Current State

- No OpenAPI specification
- API documented in various markdown files
- Route definitions in code lack formal documentation

##### Proposed Implementation

1. Create OpenAPI 3.0 specification file
2. Document all HTTP endpoints with request/response schemas
3. Include authentication requirements
4. Add error response schemas
5. Generate Swagger UI endpoint for interactive docs
6. Add CI check for spec completeness

##### Definition of Done

- [ ] OpenAPI 3.0 specification complete
- [ ] All endpoints documented with schemas
- [ ] Swagger UI available at /api-docs
- [ ] CI validates spec against code
- [ ] Error responses documented

---

#### P1-DOC-002: Deployment Runbooks

**Task ID:** P1-DOC-002  
**Description:** Create operational runbooks for deployment and maintenance  
**Priority:** P1 (High)  
**Dependencies:** P1-OBS-001, P1-OBS-002, P1-DOC-001  
**Estimated Complexity:** Medium

##### Current State

- Basic deployment info in README
- Some operations documented in docs/OPERATIONS_DB.md

##### Proposed Implementation

1. Create deployment runbook with step-by-step procedures
2. Document rollback procedures
3. Create database migration runbook
4. Document scaling procedures
5. Create troubleshooting guide with common issues
6. Add runbook for each alert

##### Definition of Done

- [ ] Deployment runbook complete
- [ ] Rollback procedures documented and tested
- [ ] Database migration runbook available
- [ ] Troubleshooting guide covers common issues
- [ ] Each alert has linked runbook

---

#### P1-DOC-003: Incident Response Guides

**Task ID:** P1-DOC-003  
**Description:** Create incident response procedures and templates  
**Priority:** P1 (High)  
**Dependencies:** None  
**Estimated Complexity:** Medium

##### Current State

- No formal incident response documentation
- Existing incident doc for territory mutator divergence

##### Proposed Implementation

1. Create incident response template
2. Document severity levels and response times
3. Create playbooks for common incident types
4. Document post-incident review process
5. Define communication templates
6. Create incident tracking workflow

##### Definition of Done

- [ ] Incident response process documented
- [ ] Severity levels defined with SLAs
- [ ] Playbooks for common incidents available
- [ ] Post-incident review template created
- [ ] Communication templates ready

---

#### P1-DOC-004: Architecture Decision Records (ADRs)

**Task ID:** P1-DOC-004  
**Description:** Establish ADR process and document key decisions  
**Priority:** P2 (Medium)  
**Dependencies:** None  
**Estimated Complexity:** Low

##### Current State

- Key decisions scattered across various docs
- No formal ADR process

##### Proposed Implementation

1. Create ADR template following standard format
2. Document existing key decisions retroactively
3. Create ADRs for: engine architecture, state management, AI integration
4. Establish ADR review process
5. Link ADRs from relevant code
6. Create ADR index

##### Definition of Done

- [ ] ADR template established
- [ ] Key existing decisions documented
- [ ] ADR review process defined
- [ ] ADR index available
- [ ] New decisions require ADR

---

## 5. Priority Matrix

### 5.1 By Priority Level

#### P0 - Critical (Must complete before production)

| Task ID     | Description                    | Complexity | Category       |
| ----------- | ------------------------------ | ---------- | -------------- |
| P1-TERR-001 | Territory Hex Processing       | High       | Bug Fix        |
| P1-SEC-001  | Input Validation Schemas       | Medium     | Security       |
| P1-SEC-002  | Auth Flow Improvements         | High       | Security       |
| P1-ERR-001  | Error Response Standardization | Medium     | Error Handling |
| P1-OBS-001  | Logging Standards              | Medium     | Observability  |
| P1-OBS-002  | Health Check Endpoints         | Medium     | Observability  |
| P1-CFG-002  | Secrets Handling               | Medium     | Configuration  |

#### P1 - High (Should complete before production)

| Task ID    | Description                  | Complexity | Category       |
| ---------- | ---------------------------- | ---------- | -------------- |
| P1-FAQ-001 | FAQ Behavior Divergence      | Very High  | Bug Fix        |
| P1-REG-001 | Region Order Choice          | Medium     | Bug Fix        |
| P1-SEC-003 | Rate Limiting Enhancement    | Medium     | Security       |
| P1-SEC-004 | CSRF/XSS Protection          | Medium     | Security       |
| P1-ERR-002 | Graceful Degradation         | High       | Error Handling |
| P1-OBS-003 | Metrics Collection           | High       | Observability  |
| P1-OBS-004 | Alerting Thresholds          | Medium     | Observability  |
| P1-CFG-001 | Env Variable Management      | Medium     | Configuration  |
| P1-CFG-004 | Deployment Config Validation | Low        | Configuration  |
| P1-DOC-001 | OpenAPI Documentation        | Medium     | Documentation  |
| P1-DOC-002 | Deployment Runbooks          | Medium     | Documentation  |
| P1-DOC-003 | Incident Response Guides     | Medium     | Documentation  |

#### P2 - Medium (Can follow production launch)

| Task ID     | Description                   | Complexity | Category       |
| ----------- | ----------------------------- | ---------- | -------------- |
| P1-JEST-001 | import.meta.env Jest Compat   | Low        | Bug Fix        |
| P1-ERR-003  | Retry Logic                   | Medium     | Error Handling |
| P1-ERR-004  | Circuit Breaker               | High       | Error Handling |
| P1-CFG-003  | Feature Flags                 | Medium     | Configuration  |
| P1-DOC-004  | Architecture Decision Records | Low        | Documentation  |

### 5.2 By Complexity

| Complexity | Tasks                                                                                                                                                                              |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Very High  | P1-FAQ-001                                                                                                                                                                         |
| High       | P1-TERR-001, P1-SEC-002, P1-ERR-002, P1-OBS-003, P1-ERR-004                                                                                                                        |
| Medium     | P1-SEC-001, P1-SEC-003, P1-SEC-004, P1-ERR-001, P1-ERR-003, P1-OBS-001, P1-OBS-002, P1-OBS-004, P1-CFG-001, P1-CFG-002, P1-CFG-003, P1-REG-001, P1-DOC-001, P1-DOC-002, P1-DOC-003 |
| Low        | P1-JEST-001, P1-CFG-004, P1-DOC-004                                                                                                                                                |

---

## 6. Execution Order

### 6.1 Recommended Implementation Sequence

#### Sprint 1: Foundation (Week 1-2)

1. **P1-SEC-001** - Input Validation Schemas (unlocks P1-ERR-001)
2. **P1-OBS-002** - Health Check Endpoints (no dependencies)
3. **P1-CFG-001** - Environment Variable Management (unlocks P1-CFG-002, P1-CFG-003)
4. **P1-DOC-003** - Incident Response Guides (no dependencies)

#### Sprint 2: Core Hardening (Week 3-4)

1. **P1-SEC-002** - Auth Flow Improvements (unlocks P1-ERR-002, P1-CFG-002)
2. **P1-ERR-001** - Error Response Standardization (unlocks P1-OBS-001)
3. **P1-CFG-002** - Secrets Handling (critical for production)
4. **P1-TERR-001** - Territory Hex Processing (critical bug fix)

#### Sprint 3: Observability & Security (Week 5-6)

1. **P1-OBS-001** - Logging Standards (unlocks P1-OBS-003)
2. **P1-SEC-003** - Rate Limiting Enhancement
3. **P1-SEC-004** - CSRF/XSS Protection
4. **P1-DOC-001** - OpenAPI Documentation

#### Sprint 4: Bug Fixes & Metrics (Week 7-8)

1. **P1-REG-001** - Region Order Choice
2. **P1-OBS-003** - Metrics Collection
3. **P1-OBS-004** - Alerting Thresholds
4. **P1-ERR-002** - Graceful Degradation

#### Sprint 5: Documentation & Polish (Week 9-10)

1. **P1-FAQ-001** - FAQ Behavior Divergence (largest task)
2. **P1-DOC-002** - Deployment Runbooks
3. **P1-CFG-004** - Deployment Config Validation

#### Post-Launch (Week 11+)

1. **P1-JEST-001** - import.meta.env Jest Compatibility
2. **P1-ERR-003** - Retry Logic
3. **P1-ERR-004** - Circuit Breaker
4. **P1-CFG-003** - Feature Flags
5. **P1-DOC-004** - Architecture Decision Records

### 6.2 Critical Path

The critical path for production readiness:

```
P1-SEC-001 → P1-ERR-001 → P1-OBS-001 → P1-OBS-003 → P1-OBS-004
     ↓
P1-DOC-001 → P1-DOC-002
```

### 6.3 Parallel Work Streams

Tasks that can proceed in parallel:

- **Security Stream:** P1-SEC-003, P1-SEC-004
- **Bug Fix Stream:** P1-TERR-001, P1-REG-001
- **Config Stream:** P1-CFG-002, P1-CFG-004
- **Documentation Stream:** P1-DOC-003, P1-DOC-004

---

## 7. Task ID Summary

### Category 1: Deferred Bug Fixes (4 tasks)

| Task ID     | Description                         | Priority |
| ----------- | ----------------------------------- | -------- |
| P1-TERR-001 | Territory Hex Processing Engine Bug | P0       |
| P1-FAQ-001  | FAQ Behavior Divergence             | P1       |
| P1-REG-001  | Region Order Choice Engine Bug      | P1       |
| P1-JEST-001 | import.meta.env Jest Compatibility  | P2       |

### Category 2: Production Hardening (17 tasks)

#### Security (4 tasks)

| Task ID    | Description                      | Priority |
| ---------- | -------------------------------- | -------- |
| P1-SEC-001 | Input Validation Schemas         | P0       |
| P1-SEC-002 | Authentication Flow Improvements | P0       |
| P1-SEC-003 | Rate Limiting Enhancement        | P1       |
| P1-SEC-004 | CSRF/XSS Protection              | P1       |

#### Error Handling (4 tasks)

| Task ID    | Description                    | Priority |
| ---------- | ------------------------------ | -------- |
| P1-ERR-001 | Error Response Standardization | P0       |
| P1-ERR-002 | Graceful Degradation Patterns  | P1       |
| P1-ERR-003 | Retry Logic                    | P2       |
| P1-ERR-004 | Circuit Breaker Implementation | P2       |

#### Observability (4 tasks)

| Task ID    | Description            | Priority |
| ---------- | ---------------------- | -------- |
| P1-OBS-001 | Logging Standards      | P0       |
| P1-OBS-002 | Health Check Endpoints | P0       |
| P1-OBS-003 | Metrics Collection     | P1       |
| P1-OBS-004 | Alerting Thresholds    | P1       |

#### Configuration (4 tasks)

| Task ID    | Description                         | Priority |
| ---------- | ----------------------------------- | -------- |
| P1-CFG-001 | Environment Variable Management     | P1       |
| P1-CFG-002 | Secrets Handling Best Practices     | P0       |
| P1-CFG-003 | Feature Flags                       | P2       |
| P1-CFG-004 | Deployment Configuration Validation | P1       |

#### Documentation (4 tasks)

| Task ID    | Description                       | Priority |
| ---------- | --------------------------------- | -------- |
| P1-DOC-001 | OpenAPI/Swagger API Documentation | P1       |
| P1-DOC-002 | Deployment Runbooks               | P1       |
| P1-DOC-003 | Incident Response Guides          | P1       |
| P1-DOC-004 | Architecture Decision Records     | P2       |

---

## Appendix A: Reference Documents

- [`FINAL_ARCHITECT_REPORT.md`](FINAL_ARCHITECT_REPORT.md) - Architecture assessment
- [[`../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md`](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md)](../docs/archive/historical/CURRENT_STATE_ASSESSMENT.md) - Implementation status
- [`STRATEGIC_ROADMAP.md`](STRATEGIC_ROADMAP.md) - Project roadmap
- [`docs/SECURITY_THREAT_MODEL.md`](docs/SECURITY_THREAT_MODEL.md) - Security documentation
- [`docs/DATA_LIFECYCLE_AND_PRIVACY.md`](docs/DATA_LIFECYCLE_AND_PRIVACY.md) - Data handling
- [`docs/SUPPLY_CHAIN_AND_CI_SECURITY.md`](docs/SUPPLY_CHAIN_AND_CI_SECURITY.md) - CI/CD security
- [`docs/OPERATIONS_DB.md`](docs/OPERATIONS_DB.md) - Database operations
