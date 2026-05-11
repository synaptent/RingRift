# RingRift Research Snapshot

This is the shortest shareable summary of the RingRift training project as of May 11, 2026.

## What RingRift Is

RingRift is a deterministic abstract strategy game plus a research codebase for training neural agents on a new ruleset. The repository contains:

- a playable web app
- a canonical TypeScript rules engine
- a Python AI and replay-parity mirror
- a narrow AlphaZero-style self-play loop used to produce the current results

## What Has Been Demonstrated

The project now has credible evidence of iterative neural-network improvement on more than one configuration.

| Config       | Best Reported Elo | Promotions | Interpretation                                                       |
| ------------ | ----------------: | ---------: | -------------------------------------------------------------------- |
| `hex8_2p`    |          `2327.8` |       `16` | Strongest result; v5-heavy + fv3 reference lane crossed 2300 Elo     |
| `square8_2p` |          `1782.0` |        `5` | Second clean 2-player proof now has two consecutive `62%` promotions |
| `square8_3p` |          `1534.9` |        `1` | Useful multiplayer signal, but still weak evidence                   |

The core research claim is no longer "can the pipeline run at all?" It is now: the RingRift self-play training loop can produce stronger models over time on at least two supported configurations, with one weaker multiplayer signal that is not yet strong enough to generalize from.

## Why The Results Are Credible

The reported April 2026 results only became defensible after several important fixes:

- TypeScript ↔ Python replay parity was tightened
- incorrect hex territory-threshold logic was fixed in the Python mutable state mirror
- structural stalemate resolution and winner selection were corrected
- the experiment harness gained true fixed-LR support end-to-end
- multiplayer evaluation was corrected to rotate exactly one candidate seat per game
- trainer/selfplay role separation stopped P2P GPU contention from polluting the supported training path

In other words, the current evidence is post-fix evidence, not a continuation of the earlier buggy harness.

## What Is Still In Flight

The project is not finished.

- the older `hex8_2p` v3-family line plateaued at `1979.8`, and the newer v5-heavy + fv3 reference lane has now broken above it to `2327.8`; the seed_d replica has also reached `2193.4`
- `hex8_3p` finally produced a first clean result, but it was a `35%` reject
- `square8_3p` still needs another clean promotion before it should count as persuasive multiplayer evidence
- `square8_4p` and the larger-board paths remain unproven
- larger boards and slower multiplayer configurations remain much less mature than the 2-player path
- cluster runs still produce the strongest evidence; local reproduction is useful but smaller in scale

## What To Read First

If you want to evaluate the project quickly, use this order:

1. [README.md](/README.md)
2. [docs/PROJECT_BRIEF.md](/docs/PROJECT_BRIEF.md)
3. [docs/RESULTS.md](/docs/RESULTS.md)
4. [docs/ARCHITECTURE_OVERVIEW.md](/docs/ARCHITECTURE_OVERVIEW.md)
5. [scripts/run_proven_experiment.sh](/scripts/run_proven_experiment.sh)

## Bottom Line

RingRift is now in a presentable state for external readers because there is a clear supported path through the repository and a real result to show at the end of it. The strongest remaining uncertainty is not whether the system works at all. It is how far the same approach extends cleanly beyond `hex8_2p` and `square8_2p`.
