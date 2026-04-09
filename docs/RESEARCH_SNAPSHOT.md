# RingRift Research Snapshot

This is the shortest shareable summary of the RingRift training project as of April 9, 2026.

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
| `hex8_2p`    |          `1967.6` |        `6` | Strongest result; clear iterative improvement from the 1500 baseline |
| `square8_2p` |          `1601.8` |        `2` | Second clean 2-player proof under the corrected experiment harness   |
| `square8_3p` |          `1534.9` |        `1` | Useful multiplayer signal, but weaker than the 2-player evidence     |

The core research claim is no longer "can the pipeline run at all?" It is now: the RingRift self-play training loop can produce stronger models over time on multiple supported configurations.

## Why The Results Are Credible

The reported April 2026 results only became defensible after several important fixes:

- TypeScript ↔ Python replay parity was tightened
- incorrect hex territory-threshold logic was fixed in the Python mutable state mirror
- structural stalemate resolution and winner selection were corrected
- the experiment harness gained true fixed-LR support end-to-end
- multiplayer evaluation was corrected to rotate exactly one candidate seat per game

In other words, the current evidence is post-fix evidence, not a continuation of the earlier buggy harness.

## What Is Still In Flight

The project is not finished.

- `hex8_2p` may be plateauing near 2000 Elo
- `square8_3p` and `square8_4p` are currently being rerun under the corrected seat-fair multiplayer evaluator
- larger boards and slower multiplayer configurations remain much less mature than the 2-player path
- cluster runs still produce the strongest evidence; local reproduction is useful but smaller in scale

## What To Read First

If you want to evaluate the project quickly, use this order:

1. [README.md](/Users/armand/Development/RingRift/README.md)
2. [docs/PROJECT_BRIEF.md](/Users/armand/Development/RingRift/docs/PROJECT_BRIEF.md)
3. [docs/RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
4. [docs/ARCHITECTURE_OVERVIEW.md](/Users/armand/Development/RingRift/docs/ARCHITECTURE_OVERVIEW.md)
5. [scripts/run_proven_experiment.sh](/Users/armand/Development/RingRift/scripts/run_proven_experiment.sh)

## Bottom Line

RingRift is now in a presentable state for external readers because there is a clear supported path through the repository and a real result to show at the end of it. The strongest remaining uncertainty is not whether the system works at all. It is how far the same approach extends cleanly to multiplayer and larger-board configurations.
