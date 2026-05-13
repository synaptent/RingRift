# Elo Rating Drift Audit - 2026-05-12

Scope: `hex8_2p` v5-heavy + fv3 reference lane, especially the May 12
`gh200-14` iter 47 frontier.

## Current Verdict

The May 12 frontier is solid as an internally consistent promotion-ladder
result:

- iter 47 promoted at stage 1, `33-17` over `50` eval games (`66.0%`).
- The quality gate passed with no warnings.
- `best.pth` and `candidate_047.pth` have identical SHA256:
  `54c4bdb7cf58c28a1d93d64a95c4d6218869b3ecf8da31e4173551ba079e18cf`.
- The trainer recorded `estimated_elo=2583.9` and `total_promotions=19`.

The right public wording is therefore: **estimated promotion-ladder Elo**.
It is good evidence of iterative improvement against the previous best, but it
is not yet a globally calibrated Elo rating.

## What Is Solid

- The 2-player Elo delta formula is internally coherent for these runs:
  promotion deltas are computed from candidate win rate versus the current best.
- Resume behavior recomputes progress from metrics history, so the headline is
  not just mutable `progress.json` state.
- The May 12 evidence bundle includes exact metrics, progress, process command,
  git head, and matching checkpoint hashes.
- The independent `seed_d` replica reaching `2193.4` Elo with `20` promotions
  reduces the risk that fv3 is a one-seed accident.

## Where Drift Can Enter

The absolute value can drift because the rating is a chained promotion ladder:

- Each promotion is measured only against the immediately previous best, not a
  stable external pool.
- Stage-1 promotions use `50` games, so `60-66%` wins can carry substantial
  sampling error compared with `400`-game stage-4 verdicts.
- Self-play can overfit to the current policy distribution. A better ladder
  successor may not improve equally against older checkpoints, seed replicas, or
  fixed baselines.
- Seat and opening distributions are checked, but a passing quality gate is not
  the same as a calibrated rating pool.
- The number should not be compared directly to chess-style public Elo, or to
  non-identical RingRift experiment families, without an anchor gauntlet.

## Recommended Anchor Gauntlet

To turn the headline into a stronger calibrated claim, run a fixed-checkpoint
rating pool:

- Current `gh200-14` best (`2583.9` ladder Elo).
- Prior reference frontiers: `2327.8`, `2241.7`, `2028.3`, and `1979.8`.
- `seed_d` best (`2193.4`) as an independent replica.
- Fixed non-neural baselines already used for public context: random,
  heuristic, and MCTS-medium.

Use symmetric seats and at least `400` games per pairing; `1000+` is preferable
for the closest pairs. Estimate a pool rating with a Bradley-Terry, BayesElo, or
equivalent pairwise model, and keep the promotion-ladder Elo as a separate
training-progress metric.

The supported executable path is:

```bash
cd ai-service
PYTHONPATH=. RINGRIFT_DISABLE_TORCH_COMPILE=1 \
python scripts/run_anchor_gauntlet.py \
  --board-type hex8 \
  --num-players 2 \
  --model-version v5-heavy \
  --feature-version 3 \
  --model frontier=models/canonical_hex8_2p_v5_heavy_fv3.pth \
  --model iter42=/path/to/candidate_042.pth \
  --model iter34=/path/to/candidate_034.pth \
  --model iter20=/path/to/candidate_020.pth \
  --model seed_d=/path/to/seed_d_best.pth \
  --baseline heuristic=heuristic \
  --baseline random=random \
  --fixed-rating heuristic=1500 \
  --games 400 \
  --budget 128 \
  --output data/elo_calibration/hex8_2p_fv3_anchor_gauntlet.json \
  --resume
```

The output is a fixed-pool calibration artifact, not a replacement for the
promotion-ladder metric. If the calibrated frontier differs materially from the
ladder Elo, public docs should report both numbers with their source labels.

## Publication Guidance

- Keep `2583.9` as the current best **estimated promotion-ladder Elo**.
- Do not describe it as a globally calibrated Elo until the anchor gauntlet
  exists.
- Public comparisons should emphasize the durable facts: `19` promotions,
  `33-17` iter 47 promotion, matching checkpoint hashes, and seed_d replication.
