# fv3 Replication Snapshot - 2026-04-28

This is a point-in-time snapshot of the active Lambda GH200 lanes. It is meant
to support outside review without requiring access to live nodes.

Update 2026-04-29: the `gh200-14` reference-lane plateau interpretation in this
snapshot is incomplete. Later copied metrics show iter 12 had a staged
`promote` decision at `72.0%` over `50` games but was blocked after a `49/50`
eval resume left the quality gate with a one-game opening sample; iter 13 then
clean-promoted at `76.0%`. Keep this file as the 2026-04-28 snapshot, and use
[`FV3_QUALITY_GATE_RESUME_NOTE_2026-04-29.md`](FV3_QUALITY_GATE_RESUME_NOTE_2026-04-29.md)
for the updated interpretation.

Raw metrics are checked in under
[`docs/data/training_runs/2026-04-28`](../data/training_runs/2026-04-28/).
The compact machine-readable summary is
[`summary.csv`](../data/training_runs/2026-04-28/summary.csv).

## Headline

- `sq8_3p` is the strongest current productive lane: `6` promotions by iter 42,
  estimated Elo `1717.6`, and the latest iter promoted at `36.0%` in a
  3-player evaluation.
- `fv3` replication is mechanically validated but plateauing early. Across the
  four seed lanes in this snapshot, completed seed iters are `2/11` promotions.
- The fv3 reference lane (`gh200-14`) reached `1586.8` Elo after promotions at
  iters 6 and 7, then rejected iters 8, 9, and 10. That supports a real plateau
  interpretation rather than a single noisy reject.

## Active Lane Snapshot

| Run                        | Completed iters | Promotions |    Elo | Latest result             | Latest value std |
| -------------------------- | --------------: | ---------: | -----: | ------------------------- | ---------------: |
| fv3 reference (`gh200-14`) |              10 |          2 | 1586.8 | reject, 45.9%, 194 games  |              n/a |
| fv3 seed A (`gh200-8`)     |               3 |          0 | 1500.0 | reject, 47.0%, 200 games  |           0.1022 |
| fv3 seed B (`gh200-11`)    |               2 |          0 | 1500.0 | reject, 47.0%, 200 games  |           0.2759 |
| fv3 seed C (`gh200-13`)    |               3 |          0 | 1500.0 | reject, 48.0%, 200 games  |           0.1073 |
| fv3 seed D (`gh200-10`)    |               3 |          2 | 1527.8 | reject, 45.0%, 100 games  |           0.3190 |
| sq8_3p (`gh200-12`)        |              36 |          6 | 1717.6 | promote, 36.0%, 400 games |           0.3225 |

## Interpretation

The fv3 result is not a clean success story. It is useful because it separates
mechanical health from strength improvement:

- The training graph is alive: completed probes have non-collapsed value heads.
- The evaluation ladder is producing real decisions across independent seeds.
- The seed lanes mostly reject near the threshold, while one seed promoted twice
  before hitting the same plateau pattern.

The supported public claim should stay narrow:

> RingRift self-play improvement is reproducible on supported 2-player and
> selected small-board lanes; fv3 multiplayer replication currently shows a
> real plateau rather than robust solved-strength progress.

## Next Data Milestone

Let the current fv3 lanes reach a clean stopping point before changing
architecture again:

- seed A: complete iter 4
- seed B: complete iter 3
- seed C: complete iter 4
- seed D: complete iter 4
- reference: complete iter 11

After that, regenerate this snapshot and plot:

- Elo by iteration
- promotion count by lane
- final evaluation win rate by lane
- value-head standard deviation by lane

Do not count the terminated `gh200-9` v4 diagnostic as fv3 or v4 model-quality
evidence; it was a host-stability failure.
