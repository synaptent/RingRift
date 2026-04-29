# Canonical Model Provenance Gap (2026-04-28)

## TL;DR

Six of twelve `canonical_*.pth` checkpoints currently disagree with the
`.sha256` sidecars that are tracked in git. The actual model files are
gitignored (`ai-service/**/*.pth`), so the sidecar is the only persistent
record of which weights were intended to be canonical.

The mismatch is a release-artifact issue, not a training quality issue: the
on-disk weights were overwritten by the live training pipeline after the
sidecar was committed.

Run `python3 scripts/check_canonical_models.py` from a clean checkout to see
the current verification matrix.

## What is on disk vs. what the sidecar says

| Sidecar                               | Status   | Sidecar SHA[:12] | On-disk SHA[:12] |
| ------------------------------------- | -------- | ---------------- | ---------------- |
| `canonical_hex8_2p.pth.sha256`        | OK       | `f10abdee718d`   | `f10abdee718d`   |
| `canonical_hex8_3p.pth.sha256`        | OK       | `31d1c5c5ff55`   | `31d1c5c5ff55`   |
| `canonical_hex8_4p.pth.sha256`        | OK       | `1ebd27ef77e2`   | `1ebd27ef77e2`   |
| `canonical_hexagonal_2p.pth.sha256`   | OK       | `bfda04ca2dd8`   | `bfda04ca2dd8`   |
| `canonical_hexagonal_3p.pth.sha256`   | OK       | `ffd0701b4f5d`   | `ffd0701b4f5d`   |
| `canonical_square19_3p_v5.pth.sha256` | OK       | `94d9914baaa4`   | `94d9914baaa4`   |
| `canonical_square8_4p.pth.sha256`     | OK       | `8006153dab97`   | `8006153dab97`   |
| `canonical_hexagonal_4p.pth.sha256`   | MISMATCH | `2a4e1f825ccf`   | `7fb3c666145d`   |
| `canonical_square19_2p.pth.sha256`    | MISMATCH | `ff6256193b0a`   | `97948dbbd09e`   |
| `canonical_square19_3p.pth.sha256`    | MISMATCH | `9e4a5e203e5f`   | `94d9914baaa4`   |
| `canonical_square19_4p.pth.sha256`    | MISMATCH | `811ea6b80cff`   | `18203696bf88`   |
| `canonical_square8_2p.pth.sha256`     | MISMATCH | `62f8ae65ea1a`   | `7c1cf2a996da`   |
| `canonical_square8_3p.pth.sha256`     | MISMATCH | `c5f0a217de4d`   | `22ab9849c178`   |

(Plus ~50 additional historical/timestamped sidecars whose `.pth` is no longer
on disk — these are recoverable evidence of past training milestones, not
release blockers.)

## Why this happened

1. The original training pipeline produced canonical weights and committed
   the matching `.sha256` sidecar at some point during early 2026.
2. The cluster training pipeline kept producing improved checkpoints and
   wrote them over `models/canonical_*.pth` in place. Each node-local
   `.pth.sha256` was regenerated after the rewrite, but the
   _committed_ sidecar in git was never updated to match.
3. Because `ai-service/**/*.pth` is gitignored, git history cannot tell us
   which run produced the original 6 of these files, and there is no
   signed pointer back to S3.

## What is and is not blocked

- **Not blocked**: training and eval _inside the cluster_. Cluster nodes
  load whatever `.pth` is on disk and rely on the embedded
  `app.training.model_versioning` integrity check, not the sidecar.
- **Blocked**: outsider-facing public evaluation, because
  `scripts/run_quick_eval.sh` deliberately fails on sidecar mismatch.
  Today only `canonical_hex8_2p.pth` clears that gate cleanly, which is
  why it is the current quick-eval default.

## Recovery options (in priority order)

1. **Treat the on-disk weights as canonical and regenerate sidecars from
   provenance evidence**.
   This is the cheapest path. For each MISMATCH:
   - confirm via `app.training.model_versioning` metadata that the on-disk
     `.pth` is the intended best model for that config;
   - record the source training run (commit, NPZ inputs, evaluation Elo)
     in `models/<config>.provenance.json`;
   - regenerate `.sha256` and commit both files together with a clear
     message that says "rotate sidecar to current trained weights".

2. **Restore the original sidecar checkpoints from S3**.
   The S3 bucket `ringrift-models-20251214` has historically held
   consolidated artifacts. From local-mac the bucket is currently
   unreachable because `~/.aws/credentials` was deleted during the
   2026-04-27 AWS root-key rotation. Restoring credentials with the
   scoped `ringrift-cluster` IAM user would let us search the bucket
   for an artifact whose SHA-256 matches the committed sidecar.

3. **Drop sidecars whose models are unrecoverable**.
   For checkpoints where neither the on-disk file nor the S3 artifact
   matches the committed sidecar, the responsible action is to remove
   the stale `.sha256` so the verification utility cannot give a false
   "should match but does not" reading.

## Why option 1 is recommended for now

The on-disk weights _are_ the live weights producing the published Elo
numbers. The committed sidecars correspond to artifacts that were
overwritten months ago and are not what any user actually evaluates today.

Rotating sidecars to the current weights is honest about what the project
is shipping. Recovery from S3 is a clean-narrative option that can be done
later, when AWS credentials are reissued at IAM-user scope.

## What the verification utilities provide

There are two complementary tools:

1. `scripts/check_canonical_models.py` — broad sweep, dependency-free. From a
   clean checkout it lists every `canonical_*.pth.sha256` sidecar, recomputes
   the corresponding `.pth` hash, and prints OK / MISSING-PTH / MISMATCH per
   row. Exit code is 1 if any mismatch is detected. Suitable as a CI gate
   before publishing a release.

   ```bash
   python3 scripts/check_canonical_models.py            # human view
   python3 scripts/check_canonical_models.py --quiet    # only show issues
   ```

2. `ai-service/scripts/audit_public_model_artifacts.py` — deep release gate
   for the specific public artifacts the README advertises. Verifies that
   each public model not only matches its sidecar but also loads through
   `safe_load_checkpoint` and reports metadata consistent with its
   advertised board/player config.

Use the broad sweep to find drift; use the deep audit to gate what gets
published.
