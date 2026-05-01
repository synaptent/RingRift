# RingRift Reviewer Guide

This is the shortest path for a cold outside reviewer to evaluate RingRift
without getting lost in historical training and operations surfaces.

## Review Thesis

RingRift should be judged first on its supported path:

1. A novel deterministic board game that is playable in the browser.
2. A canonical TypeScript rules engine reused by product hosts.
3. A Python AI and replay-parity mirror for training and inference.
4. A narrow minimal-loop training harness that produced the checked-in results.

The most credible current story is not "every experimental lane is strong." It
is that the supported two-player training path can improve small-board models,
and that the project has unusually explicit contract tests and evidence for
silent AlphaZero failure modes.

Everything else is secondary until the supported path is understood.

## Ten-Minute Review Path

If you want to see the product before reading, run:

```bash
npm install
npm run play
```

This opens the no-account Human-vs-AI sandbox at
`http://localhost:5173/sandbox?preset=hex8-1h-1ai`. It is the fastest way to
verify that RingRift is a playable game, not only a rules and training codebase.

Read these in order:

1. [README.md](/README.md)
2. [QUICKSTART.md](/QUICKSTART.md)
3. [docs/RESULTS.md](/docs/RESULTS.md)
4. [docs/REPRODUCIBILITY.md](/docs/REPRODUCIBILITY.md)
5. [docs/ARCHITECTURE_OVERVIEW.md](/docs/ARCHITECTURE_OVERVIEW.md)
6. [docs/LESSONS_LEARNED.md](/docs/LESSONS_LEARNED.md)
7. [docs/LESSONS_LEARNED_2026-04.md](/docs/LESSONS_LEARNED_2026-04.md) - dated
   training-debugging addendum; useful for process review, not a headline
   results source.
8. [docs/research/SILENT_ALPHAZERO_FAILURES.md](/docs/research/SILENT_ALPHAZERO_FAILURES.md) -
   reusable AlphaZero failure catalog with tests and evidence pointers.
9. [docs/research/FV3_QUALITY_GATE_RESUME_NOTE_2026-04-29.md](/docs/research/FV3_QUALITY_GATE_RESUME_NOTE_2026-04-29.md) -
   live-run evidence where staged eval said promote but a resume-limited
   quality-gate sample blocked iter 12 before iter 13 clean-promoted.

Then inspect these code surfaces:

1. [src/shared/engine](/src/shared/engine) - canonical rules engine.
2. [src/client](/src/client) - playable React client.
3. [src/server](/src/server) - backend host and WebSocket surface.
4. [ai-service/app/rules](/ai-service/app/rules) - Python rules contracts.
5. [ai-service/scripts/minimal_alphazero_loop.py](/ai-service/scripts/minimal_alphazero_loop.py) - supported training loop.

For AI and training code, use [docs/data/ai_surface_manifest.json](/docs/data/ai_surface_manifest.json)
as the explicit supported-vs-experimental boundary.

## Evidence Boundary

The public result claims are intentionally narrower than live operator context.

- Checked-in claims live in [docs/data/results_snapshot.json](/docs/data/results_snapshot.json).
- Claim provenance lives in [docs/data/results_evidence_manifest.json](/docs/data/results_evidence_manifest.json).
- Larger artifacts such as checkpoints, full metrics logs, and training NPZ files live outside git and are referenced from [docs/REPRODUCIBILITY.md](/docs/REPRODUCIBILITY.md).
- Live cluster updates should not be promoted into public claims until their artifacts are mirrored or explicitly recorded in the evidence manifest.
- The 2026-04-29 fv3 quality-gate resume note is not a headline result; it is
  a checked-in failure-analysis artifact with copied metrics.
- The `gh200-8` v4 multiplayer retry is an active diagnostic, not a
  multiplayer-strength claim. The 2026-04-30 fixed-seat and padded-target
  fixes prove that the retry can train and evaluate through the intended
  mechanical path, but early post-fix eval was weak. Treat that lane as
  operational evidence until its verdict and seat-balance analysis are
  preserved.
- The `gh200-12` square8_3p recovery similarly proves that max-slot
  multiplayer targets can train against fixed-seat heads after the
  2026-04-30 loss fix. It should not be presented as a strength result unless
  the post-fix eval trajectory supports that claim.

This boundary is deliberate. It is better for RingRift to underclaim than to ask
a reviewer to trust chat logs or oral history.

## Trust Commands

Run these before deeper review:

```bash
python3 scripts/check_github_workflows.py
python3 scripts/check_reviewer_surface.py
python3 scripts/check_ai_surface.py
python3 scripts/build_reviewer_packet.py --clean
bash scripts/check_supported_path.sh
npm run test:coverage:rules-critical
npm run test:coverage:training-contracts
npm run build
```

For Python-only training and parity work:

```bash
cd ai-service
PYTHONPATH=. python3 -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```

For rules/product work:

```bash
npm run test:p0-robustness
```

## What To Ignore At First

Do not start with the whole repository tree. It contains years of experiments,
cluster automation, diagnostics, and archived plans.

Treat these as secondary until needed:

- `archive/`
- `docs/archive/`
- broad cluster orchestration under `ai-service/scripts`
- large compatibility shims under `ai-service/app/ai` and `ai-service/app/game_engine`
- internal planning notes under `docs/planning`

The important distinction is not "delete everything else." It is "do not confuse
support infrastructure with the proof path."

## Current Reviewer Risks

The project is strongest on rules correctness, parity discipline, and honest
result caveats. It is weakest where a reviewer has to separate supported code
from historical surface area.

Highest-signal improvement areas:

1. Keep the public evidence pack current from machine-readable snapshots.
2. Keep new fv3 and reliability-fix artifacts mirrored into evidence manifests
   before changing public result claims.
3. Continue shrinking or archiving non-supported scripts.
4. Extend non-zero coverage thresholds beyond the current TypeScript rules engine and Python training-contract ratchets.
5. Tighten Python typing for parity and checkpoint-contract modules.
6. Package result artifacts so the headline claims can be audited without private cluster access.

## Reviewer Surface Manifest

The machine-readable map for this guide is
[docs/data/reviewer_surface_manifest.json](/docs/data/reviewer_surface_manifest.json).
It is validated by [scripts/check_reviewer_surface.py](/scripts/check_reviewer_surface.py).
The AI/training support boundary is
[docs/data/ai_surface_manifest.json](/docs/data/ai_surface_manifest.json) and is
validated by [scripts/check_ai_surface.py](/scripts/check_ai_surface.py).
To copy the supported docs, result snapshots, evidence manifest, and result
visuals into a compact local packet, run
[scripts/build_reviewer_packet.py](/scripts/build_reviewer_packet.py).
