# fv3 Quality-Gate Resume Note - 2026-04-29

This note corrects the first-pass interpretation of the `gh200-14` fv3
reference lane. It is not a new public Elo headline. It is a reviewer-facing
failure-analysis artifact with copied metrics.

Raw evidence copied from the live node:

- [`docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.metrics.jsonl`](../data/training_runs/2026-04-29/fv3_reference_gh200-14.metrics.jsonl)
- [`docs/data/training_runs/2026-04-29/summary.csv`](../data/training_runs/2026-04-29/summary.csv)
- [`docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.iter012_resume_backfill.json`](../data/training_runs/2026-04-29/fv3_reference_gh200-14.iter012_resume_backfill.json)
- [`docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.iter013_final.json`](../data/training_runs/2026-04-29/fv3_reference_gh200-14.iter013_final.json)

## What Changed

The 2026-04-28 fv3 snapshot said the reference lane reached `1586.8` Elo after
promotions at iters 6 and 7, then rejected iters 8, 9, and 10. That supported
a plausible plateau interpretation.

The later metrics show a sharper story:

| Iter | Eval result                        | Staged decision | Promoted | Quality verdict                                  |
| ---: | ---------------------------------- | --------------- | -------- | ------------------------------------------------ |
|   11 | `193/207/0`, `48.25%`, `400` games | reject          | false    | passed, with seat warning                        |
|   12 | `36/14/0`, `72.0%`, `50` games     | promote         | false    | critical `MODE_COLLAPSE` on `1/1` opening sample |
|   13 | `38/12/0`, `76.0%`, `50` games     | promote         | true     | passed                                           |

The iter 13 row changes the conclusion. The lane did not confirm a clean
mode-collapse failure. It clean-promoted at stage 1, with value-head health,
seat fairness, and behavioral-diversity checks passing.

## Corrected Interpretation

The strongest current diagnosis is that iter 12 exposed a quality-gate resume
coverage bug, not a game-design flaw and not a collapsed policy. The live
`training.log` confirmed the critical detail:
`eval resume: 49 games already played (cand=36 best=13 draws=0)`.

Evidence:

- Direct empty-board policy probes on the iter 7 best, iter 12 candidate, and
  iter 13 candidate showed healthy policy entropy: top action around `2.5%`,
  entropy around `0.52` of max, and roughly `80` effective actions.
- Iter 12 self-play was healthy: `100` games, about `74` moves on average, `35`
  unique first moves, and the top opening only `8/100`.
- The iter 12 quality warning said `100%` of games shared the same opening, but
  the denominator was `1/1`. That means the tracker had only one post-resume
  opening sequence, while staged evaluation counted `50` game outcomes.
- `staged_evaluate()` can resume from `iter_NNN_eval.json` with only outcome
  counts restored. Its own docstring notes that pre-restart move-level tracker
  data is not recovered, so behavioral diversity can reflect only the
  post-resume suffix.
- Iter 13 then reached a real stage-1 promotion and passed the quality gate:
  `38/12/0`, `76.0%`, estimated Elo `1787.0`, `quality gate passed`.

## Root Cause

There are two distinct counters in the current metrics:

- evaluation outcome count: candidate wins, best wins, draws, games played
- quality-gate move-level sample count: opening sequences, chosen moves, legal
  moves, root values

After an eval resume, the first counter can be complete while the second is
partial. The quality-gate details currently report `games_tracked=50`, but they
do not also report how many games contributed opening sequences. That made the
iter 12 row easy to misread as "50 games all shared one opening" instead of
"one post-resume opening sample repeated once."

## Fix Status

The resume fix is committed as `4e1b7e20e`
(`fix(coordination): persist quality-gate tracker state across eval resume`) and
has been deployed to the live fleet. Running loops that started before the
deploy will pick it up on their next cooperative restart. Legacy checkpoints
without tracker state are backward-compatible: the new code marks move/value
tracking partial instead of allowing a partial suffix to issue critical
behavioral or value-head verdicts.

1. Persist quality-tracker state in the eval checkpoint.
   - Added `QualityGateTracker.to_checkpoint()` and
     `QualityGateTracker.load_checkpoint()` in
     `ai-service/scripts/lib/model_quality_gate.py`.
   - Store `_openings`, `_unique_moves_chosen`, `_unique_legal_moves_seen`,
     `_values`, `_nonfinite_value_samples`, `_game_count`, seat counts, and the
     self-play seat baseline in `iter_NNN_eval.json`.
   - Restore that state before replaying any remaining games.

2. Make partial tracker coverage explicit and non-critical.
   - Added `opening_games_tracked`, `opening_sample_coverage`, and
     `move_tracking_complete` to behavioral-diversity details.
   - If a legacy checkpoint is resumed without tracker state, mark move-level
     tracking partial and skip critical behavioral/value-head gates for that
     iteration instead of blocking promotion on a partial suffix.
   - Keep seat fairness active because resumed checkpoints already replay
     per-seat outcomes.

3. Add regression tests.
   - `test_small_opening_sample_does_not_trigger_mode_collapse` covers the
     `50` games / `1` opening-sample failure.
   - `test_staged_evaluate_resume_restores_quality_tracker_state` covers
     checkpoint save/load.
   - `test_staged_evaluate_legacy_resume_marks_tracker_partial` covers legacy
     checkpoints that lack tracker state.

## Operator Decision

Do not reassign `gh200-14` away from fv3 on the basis of iter 12. Iter 13
clean-promoted. Let the reference lane run to the next clean milestone while
the v4 multiplayer retry runs as a separate reviewer-value lane.

## Backfill Classification

The iter 12 backfill is documented rather than rerun. A fresh rerun would
consume an active GH200 lane, while the copied metrics already contain the
control-vs-treatment evidence:

- Iter 12 was strength-positive: `36/14/0`, `72.0%`, staged decision
  `promote`.
- Iter 12 was not promoted because the legacy quality gate used a partial
  denominator: `MODE_COLLAPSE` on `1/1` opening sample.
- Iter 13 was the treatment/control check: another strength-positive candidate,
  `38/12/0`, `76.0%`, and a clean quality-gate pass.

The structured backfill artifact is
[`docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.iter012_resume_backfill.json`](../data/training_runs/2026-04-29/fv3_reference_gh200-14.iter012_resume_backfill.json).

## Follow-on v4 Retry

`gh200-8` seed A finished `0/7` promotions and was retired after preserving a
compact evidence bundle. The host was then repurposed for the v4 multiplayer
retry in a fresh workdir:

- workdir: `data/minimal_loop_hex8_3p_v4_retry_gh2008_20260429`
- model: `models/canonical_hex8_3p_v4_retry_20260429.pth`
- launch commit: `4e1b7e20e`
- required flags present: `--model-version v4`, `--num-players 3`,
  `--policy-weight 0.8`, `--value-weight 2.0`,
  `--rank-dist-weight 0.05`, `--gradient-clip-max-norm 0.5`

Seed A evidence copied into the repo:
[`docs/data/training_runs/2026-04-29/seed_a_gh200-8_0of7/gh200-8_seed_a_0of7_20260429T1545Z_evidence.tgz`](../data/training_runs/2026-04-29/seed_a_gh200-8_0of7/gh200-8_seed_a_0of7_20260429T1545Z_evidence.tgz).

## Seed-Lane Audit

A read-only audit of the fv3 seed lanes found no additional
`MODE_COLLAPSE`-blocked promotions.

| Lane                | Rows checked | Latest checked row                                             | Finding                                  |
| ------------------- | -----------: | -------------------------------------------------------------- | ---------------------------------------- |
| seed A (`gh200-8`)  |            5 | iter 5 reject, `46.0%`, quality gate passed with seat warning  | no mode-collapse false block             |
| seed B (`gh200-11`) |            3 | iter 3 promote, `50.75%`, quality gate passed                  | low opening sample (`3`) but no block    |
| seed C (`gh200-13`) |            5 | iter 5 reject, `48.0%`, quality gate passed                    | iter 4 had no opening stats but no block |
| seed D (`gh200-10`) |            4 | iter 4 promote, `53.0%`, quality gate passed with seat warning | no mode-collapse false block             |

These lanes still benefit from the resume fix because several logs show
mid-eval resumes, but only `gh200-14` iter 12 is currently known to have lost a
promotion to the old partial-sample behavior.

## Reviewer Takeaway

The interesting project claim is narrower and stronger than "more GPU made Elo
go up." It is:

> RingRift has enough instrumentation to discover when a quality gate itself is
> producing an unsafe conclusion, and enough copied metrics to correct the
> public interpretation quickly.
