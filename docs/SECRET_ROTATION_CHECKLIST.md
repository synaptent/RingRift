# Secret Rotation Checklist

This document tracks secrets that were committed to version control and need to be rotated.

## Secrets Found in Git History

### 1. `.env.staging` (commit `547bbb6`)

**Committed**: Nov 24, 2025
**Secrets exposed**:

- `JWT_SECRET` - Staging JWT signing secret (64-char base64 string)
- `JWT_REFRESH_SECRET` - Staging JWT refresh secret (64-char base64 string)
- `GRAFANA_PASSWORD` - Staging Grafana admin password (`StagingGrafana2024`)
- `DB_PASSWORD` - Staging database password (`password`)
- `RATE_LIMIT_BYPASS_TOKEN` - Staging rate limit bypass token
- `POSTGRES_PASSWORD` - Staging PostgreSQL password (`password`)

### 2. `monitoring/prometheus.yml` (multiple commits)

**Committed**: Multiple times between Nov 2025 - Feb 2026
**Secrets exposed**:

- Tailscale IPs for all cluster nodes (100.x.x.x addresses)
- Node hostnames and roles
- Internal network topology

**Key commits**:

- `9bf4dd86f` - chore(monitoring): add cluster node targets to Prometheus config
- `c4cf9844e` - chore: clean up prometheus targets and fix test mock nesting
- `9aad8ada1` - chore(monitoring): update prometheus targets for current cluster
- `f20de016e` - chore(monitoring): update prometheus targets for current cluster state
- `0a6ab2270` - refactor(monitoring): update prometheus targets to use Tailscale IPs

## Remediation Steps Completed

1. [x] Added `.env.production` to `.gitignore`
2. [x] Verified `.env` and `.env.staging` are already gitignored
3. [x] Verified `monitoring/prometheus.yml` is gitignored
4. [x] Verified `ai-service/config/distributed_hosts.yaml` is gitignored
5. [x] Removed `monitoring/prometheus.yml` from git tracking (`git rm --cached`)
6. [x] Created sanitized example files:
   - `.env.example` (updated with missing keys)
   - `.env.staging.example` (already existed with placeholders)
   - `.env.production.example` (already existed with placeholders)
   - `monitoring/prometheus.yml.example`
   - `ai-service/config/distributed_hosts.yaml.example`

## Git History Cleanup (NOT YET DONE)

The following command will rewrite git history to remove secret files. **Do NOT run this without coordinating with all contributors**, as it rewrites commit hashes.

```bash
# Install git-filter-repo if not already installed
# pip install git-filter-repo
# or: brew install git-filter-repo

# Preview what would be removed (dry run)
git filter-repo --path .env.staging --path monitoring/prometheus.yml --invert-paths --dry-run

# Actually rewrite history (DESTRUCTIVE - rewrites all commit hashes)
git filter-repo --path .env.staging --path monitoring/prometheus.yml --invert-paths --force

# After rewriting, force push to remote
# WARNING: All collaborators must re-clone after this
# git push origin --force --all
# git push origin --force --tags
```

## Credentials That Need Rotation

### Priority 1 - Rotate Immediately

These credentials were committed with real values:

| Credential                 | File           | Action Required                                |
| -------------------------- | -------------- | ---------------------------------------------- |
| Staging JWT Secret         | `.env.staging` | Generate new secret: `openssl rand -base64 48` |
| Staging JWT Refresh Secret | `.env.staging` | Generate new secret: `openssl rand -base64 48` |
| Staging Grafana Password   | `.env.staging` | Change in Grafana admin settings               |

### Priority 2 - Rotate When Convenient

These were committed with weak/default values (not high-risk if only used locally):

| Credential                | File           | Action Required                                              |
| ------------------------- | -------------- | ------------------------------------------------------------ |
| Staging DB Password       | `.env.staging` | Was `password` - change if staging DB is internet-accessible |
| Staging Postgres Password | `.env.staging` | Was `password` - same as above                               |

### Priority 3 - Network Topology (Low Risk)

Tailscale IPs are on a private mesh network and not directly routable from the internet, but exposing the topology aids reconnaissance:

| Data                      | File                        | Action                                  |
| ------------------------- | --------------------------- | --------------------------------------- |
| Tailscale IPs (100.x.x.x) | `monitoring/prometheus.yml` | Rotate Tailscale node keys if concerned |
| Node hostnames/roles      | `monitoring/prometheus.yml` | Informational only                      |

## Verification Steps

After rotating credentials:

1. [ ] Staging JWT secrets rotated - verify staging login still works
2. [ ] Staging Grafana password changed - verify Grafana access
3. [ ] Git history cleaned with `git-filter-repo` (if decided)
4. [ ] All collaborators re-cloned (if history was rewritten)
5. [ ] Run `git log --all -p -- .env.staging monitoring/prometheus.yml` to confirm removal
