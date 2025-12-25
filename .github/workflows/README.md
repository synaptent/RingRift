# CI/CD Pipeline Documentation

## Overview

RingRift uses a comprehensive CI/CD pipeline to ensure code quality, automated testing, and reliable deployments. The pipeline runs on every push and pull request to the `main` and `develop` branches.

## Pipeline Stages

### 1. Lint (Code Quality Check)

**Purpose**: Ensures code follows consistent style guidelines and best practices.

**What it does**:

- Runs ESLint on all TypeScript files
- Checks for code quality issues, potential bugs, and style violations
- Fails the build if any linting errors are found

**Command**: `npm run lint`

**Fix locally**: `npm run lint:fix`

### 2. Type Check (TypeScript Validation)

**Purpose**: Validates TypeScript types across the entire codebase.

**What it does**:

- Type checks server code (`tsconfig.server.json`)
- Type checks client code (`tsconfig.client.json`)
- Ensures no type errors exist
- Does not emit JavaScript files (--noEmit flag)

**Commands**:

- Server: `npx tsc --noEmit -p tsconfig.server.json`
- Client: `npx tsc --noEmit -p tsconfig.client.json`

### 3. Test (Automated Testing)

**Purpose**: Runs the full test suite with coverage reporting.

**What it does**:

- Executes all unit and integration tests
- Generates code coverage reports
- Uploads coverage to Codecov (optional)
- Archives coverage reports as artifacts
- Fails if coverage thresholds (80%) are not met

**Command**: `npm run test:ci`

**Coverage Artifacts**:

- Available in GitHub Actions for 7 days
- Downloadable HTML report
- LCOV format for external tools

### 4. Build (Production Build)

**Purpose**: Ensures the application can be built successfully.

**What it does**:

- Compiles TypeScript server code
- Builds optimized client bundle with Vite
- Archives build artifacts
- Only runs if lint, typecheck, and test pass

**Commands**:

- `npm run build:server`
- `npm run build:client`

**Build Artifacts**:

- Available in GitHub Actions for 7 days
- Contains compiled server and client code

### 5. Quality Gate (Final Check)

**Purpose**: Final approval step ensuring all quality checks passed.

**What it does**:

- Confirms all previous stages succeeded
- Provides clear success message

## Pre-commit Hooks

Git pre-commit hooks run automatically before every commit to catch issues early.

### What runs on pre-commit:

1. **Lint-Staged**
   - Runs ESLint with auto-fix on staged `.ts` and `.tsx` files
   - Runs Prettier formatting on staged files
   - Only processes files being committed

2. **Type Check**
   - Validates TypeScript types for server and client
   - Ensures no type errors before commit

3. **Related Tests**
   - Runs tests related to changed files
   - Uses `--bail` to stop on first failure
   - Fast feedback loop

### Skipping Pre-commit Hooks

**Not recommended**, but in emergencies:

```bash
git commit --no-verify -m "Emergency fix"
```

## Local Development Workflow

### Before Committing

```bash
# Run all quality checks locally
npm run lint          # Check code style
npm run lint:fix      # Fix code style issues
npm test             # Run all tests
npm run test:coverage # Run tests with coverage
npm run build        # Ensure build works
```

### Recommended Git Flow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... edit files ...

# 3. Stage changes
git add .

# 4. Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# 5. Push to remote
git push origin feature/my-feature

# 6. Create pull request
# CI/CD pipeline runs automatically
```

## CI/CD Best Practices

### 1. Write Tests First (TDD)

- Create tests before implementing features
- Aim for 80%+ code coverage
- Test edge cases and error scenarios

### 2. Fix Linting Issues

- Don't commit code with linting errors
- Use `npm run lint:fix` to auto-fix issues
- Configure your editor for real-time linting

### 3. Type Safety

- Avoid `any` types
- Use proper TypeScript types
- Fix type errors, don't bypass them

### 4. Small, Focused Commits

- One feature/fix per commit
- Clear, descriptive commit messages
- Follow conventional commits (feat, fix, docs, etc.)

### 5. Green Build

- Ensure all CI checks pass before merging
- Don't merge failing PR

s

- Fix issues promptly

## Troubleshooting CI/CD Issues

### Lint Failures

**Common causes**:

- Unused variables
- Missing semicolons
- Inconsistent formatting

**Fix**:

```bash
npm run lint:fix
git add .
git commit --amend --no-edit
```

### Type Check Failures

**Common causes**:

- Type mismatches
- Missing type definitions
- Incorrect imports

**Fix**:

- Review TypeScript errors carefully
- Add proper types
- Use type assertions only when necessary

### Test Failures

**Common causes**:

- Race conditions
- Incorrect mocks
- Environment issues

**Fix**:

```bash
npm test -- --verbose
npm test -- tests/unit/failing-test.test.ts
```

### Build Failures

**Common causes**:

- Missing dependencies
- Environment variables
- Module resolution issues

**Fix**:

```bash
npm ci                 # Clean install
rm -rf node_modules    # Nuclear option
npm install
npm run build
```

## Coverage Reports

### Viewing Coverage Locally

```bash
npm run test:coverage
open coverage/lcov-report/index.html
```

### Coverage Thresholds

- **Branches**: 80%
- **Functions**: 80%
- **Lines**: 80%
- **Statements**: 80%

### Improving Coverage

1. Identify uncovered code in the HTML report
2. Write tests for uncovered functions/branches
3. Run `npm run test:coverage:watch` for live feedback

## GitHub Actions Configuration

### Workflow File

Location: `.github/workflows/ci.yml`

### Node.js Version

Currently using Node.js 18 (LTS). Update in workflow file if needed:

```yaml
- name: Setup Node.js
  uses: actions/setup-node@v4
  with:
    node-version: '18'
```

### Caching

npm dependencies are cached automatically to speed up builds:

```yaml
with:
  cache: 'npm'
```

### Parallelization

Jobs run in parallel when possible:

- `lint` and `typecheck` run simultaneously
- `test` runs independently
- `build` runs after lint, typecheck, and test complete

## Environment Variables

### Required for CI

No environment variables required for basic CI/CD.

### Optional Integrations

- `CODECOV_TOKEN`: For Codecov integration
- Add to GitHub repository secrets if using Codecov

## Maintenance

### Updating Dependencies

```bash
# Check for outdated packages
npm outdated

# Update all dependencies
npm update

# Update specific package
npm install package@latest --save-dev

# After updates, ensure CI passes
npm run test:ci
```

### Updating GitHub Actions

Check for updates to GitHub Actions:

- `actions/checkout`
- `actions/setup-node`
- `codecov/codecov-action`
- `actions/upload-artifact`

## Monitoring

### Build Status

View build status:

- GitHub repository → Actions tab
- Pull request checks
- Commit status badges

### Build History

- Actions tab shows all workflow runs
- Filter by branch, status, or workflow
- Download logs for debugging

## Parity CI Gate (`parity-ci.yml`)

### Purpose

Automatically verifies TS↔Python engine parity on PRs that modify rules/engine code. Blocks merges when semantic divergences are detected.

### What Gets Verified

1. **Replay Parity**: Games from `canonical_square8.db` are replayed through both TS and Python engines
2. **State Comparison**: At each move, compares `currentPlayer`, `currentPhase`, `gameStatus`, and `stateHash`
3. **Divergence Detection**: Any mismatch is reported and causes the workflow to fail

### Trigger Conditions

The workflow runs when PRs modify:

- `src/shared/engine/**` - TypeScript engine
- `src/client/sandbox/**` - TypeScript sandbox adapter
- `ai-service/app/rules/**` - Python rules implementation
- `ai-service/app/game_engine.py` - Python GameEngine
- `ai-service/app/db/game_replay.py` - Python replay DB
- Parity scripts and canonical databases

### Running Locally

```bash
# Replay parity check (CI-equivalent)
cd ai-service
PYTHONPATH=. python scripts/check_ts_python_replay_parity.py \
  --db data/games/canonical_square8.db \
  --fail-on-divergence

# Contract vectors + plateau snapshots
PYTHONPATH=. python scripts/run_parity_healthcheck.py --fail-on-mismatch
```

### Related Documentation

- [`docs/PARITY_VERIFICATION_RUNBOOK.md`](../../docs/runbooks/PARITY_VERIFICATION_RUNBOOK.md) - Full parity verification documentation
- [`ai-service/TRAINING_DATA_REGISTRY.md`](../../ai-service/TRAINING_DATA_REGISTRY.md) - Canonical data inventory
- [`AGENTS.md`](../../AGENTS.md) - Section 4 (TS↔Python Parity)

---

## SLO Gate Workflow (`slo-gate.yml`)

### Purpose

Implements automated SLO (Service Level Objective) validation gates for staging deployments and production promotion decisions. Provides automated quality gates based on load test results.

### Trigger Methods

1. **Manual Dispatch** - On-demand validation via GitHub Actions UI
2. **Workflow Call** - Reusable workflow from deployment pipelines
3. **Scheduled** - Nightly production readiness assessment (3:00 AM UTC)

### Gate Types

| Gate Type              | Purpose                            | When to Use                  |
| ---------------------- | ---------------------------------- | ---------------------------- |
| `staging-promotion`    | Validate staging before production | After staging deployment     |
| `production-readiness` | Full production SLO check          | Before production deployment |
| `smoke-test`           | Quick validation                   | Local testing, quick checks  |

### Gate Statuses

| Status        | Meaning                             | CI Behavior              |
| ------------- | ----------------------------------- | ------------------------ |
| `APPROVED`    | All critical + high SLOs passed     | Continues pipeline       |
| `CONDITIONAL` | Critical passed, some high breached | Requires manual approval |
| `BLOCKED`     | One or more critical SLOs breached  | Fails pipeline           |

### Key SLOs Gated

**Critical (Zero Tolerance):**

- Service Availability: ≥99.9%
- Error Rate: ≤0.5% (production) / ≤1% (staging)
- Contract Failures: 0
- Lifecycle Mismatches: 0

**High Priority:**

- API Latency p95: <500ms (production) / <800ms (staging)
- Move Latency E2E p95: <200ms (production) / <300ms (staging)
- AI Response Time p95: <1000ms (production) / <1500ms (staging)

### Running Locally

```bash
# Run SLO gate check on existing results
npm run slo:gate -- --results-file tests/load/results/baseline.json --env staging

# Run full verification pipeline
npm run slo:check

# Run with production thresholds
npm run slo:gate:production -- --results-file tests/load/results/baseline.json --fail-on-breach
```

### Workflow Integration

```yaml
# Use as a reusable workflow
jobs:
  slo-validation:
    uses: ./.github/workflows/slo-gate.yml
    with:
      environment: staging
      gate_type: staging-promotion
```

### Related Documentation

- [`docs/operations/SLO_GATE_CI_RUNBOOK.md`](../../docs/operations/SLO_GATE_CI_RUNBOOK.md) - Full runbook
- [`docs/production/PRODUCTION_VALIDATION_GATE.md`](../../docs/production/PRODUCTION_VALIDATION_GATE.md) - Validation checklist
- [`docs/planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md`](../../docs/planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md) - SLO analysis

---

## Future Enhancements

Planned CI/CD improvements:

1. **Deployment Automation**
   - Auto-deploy to staging on `develop` branch
   - Auto-deploy to production on `main` branch releases

2. **Performance Testing**
   - Lighthouse CI for frontend performance
   - Load testing for backend APIs

3. **Security Scanning**
   - Dependency vulnerability scanning
   - SAST (Static Application Security Testing)

4. **Release Automation**
   - Semantic versioning
   - Automated changelog generation
   - GitHub releases

## Monitoring Configuration Validation

### Purpose

Validates Prometheus and Alertmanager configurations to catch misconfigurations before deployment. This prevents invalid configs from causing monitoring outages.

### What Gets Validated

1. **YAML Syntax**: Basic syntax validation using Python's yaml parser
2. **Prometheus Config**: Using official `promtool check config`
3. **Alert Rules**: Using official `promtool check rules` (validates PromQL expressions)
4. **Alertmanager Config**: Using official `amtool check-config`
5. **Common Issues**: Checks for hardcoded secrets, tabs in YAML, etc.

### Trigger Conditions

The validation workflow runs when:

- Files in `monitoring/**` are modified
- The validation script `scripts/validate-monitoring-configs.sh` is modified

### Running Locally

```bash
# Using npm script (recommended)
npm run validate:monitoring

# Or run the script directly
./scripts/validate-monitoring-configs.sh

# Options
./scripts/validate-monitoring-configs.sh --help       # Show help
./scripts/validate-monitoring-configs.sh --verbose    # Verbose output
./scripts/validate-monitoring-configs.sh --yaml-only  # YAML only (no Docker)
./scripts/validate-monitoring-configs.sh --docker-only # Requires Docker
```

### Files Validated

| File                                       | Tool Used | Checks                          |
| ------------------------------------------ | --------- | ------------------------------- |
| `monitoring/prometheus/prometheus.yml`     | promtool  | Config syntax, scrape configs   |
| `monitoring/prometheus/alerts.yml`         | promtool  | PromQL syntax, rule format      |
| `monitoring/alertmanager/alertmanager.yml` | amtool    | Routing, receivers, inhibitions |

### Requirements

- **Docker**: Required for full validation (uses official Prometheus/Alertmanager images)
- **Python 3 + pyyaml**: Fallback for basic YAML validation if Docker unavailable

---

**Last Updated**: November 27, 2025
**Maintainer**: RingRift Development Team
