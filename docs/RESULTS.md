# RingRift Results

This document summarizes the current research evidence from the RingRift self-play training project.

Status is current as of April 9, 2026.

## Headline Results

| Config       | Start Elo | Best Reported Elo | Promotions | Status                                                               |
| ------------ | --------: | ----------------: | ---------: | -------------------------------------------------------------------- |
| `hex8_2p`    |    `1500` |          `1967.6` |        `6` | Strongest result; currently plateaued near 2000                      |
| `square8_2p` |    `1500` |          `1601.8` |        `2` | Clean 2-player square result with recent promotions                  |
| `square8_3p` |    `1500` |          `1534.9` |        `1` | Promising, but multiplayer evidence is weaker than the 2-player runs |

![Headline results snapshot](assets/results/headline_results.svg)

## Why These Results Matter

RingRift was built as a novel deterministic strategy game plus an end-to-end AlphaZero-style training system. The central question was not only whether the system could run, but whether it could produce real iterative neural-network improvement on a nontrivial new game.

The answer is now yes.

The strongest evidence is `hex8_2p`, but `square8_2p` now matters almost as much because it shows the improvement loop is not limited to a single board family.

## Concrete Evidence

### `hex8_2p`

- Best reported Elo: `1967.6`
- Promotions: `6`
- Interpretation: strong iterative improvement from the 1500 baseline to a much stronger checkpoint family

Recent iterations have been clustering around the promotion boundary, which looks more like a real plateau than an infrastructure failure.

### `square8_2p`

This is the clearest recent improvement story in the repo.

Recent progression:

| Iteration | Win Rate |         Eval Games | Result  | Estimated Elo |
| --------- | -------: | -----------------: | ------- | ------------: |
| `27`      |  `52.0%` |   legacy eval path | reject  |      `1500.0` |
| `28`      |  `52.0%` |   legacy eval path | reject  |      `1500.0` |
| `29`      |  `54.5%` | `200` staged games | promote |      `1531.4` |
| `30`      |  `60.0%` |  `50` staged games | promote |      `1601.8` |

![square8_2p progression](assets/results/square8_2p_progression.svg)

Important context:

- iteration `29` and `30` were produced after the minimal loop gained true fixed-LR support end-to-end
- those promotions also used staged evaluation and the cleaned-up experiment harness

### `square8_3p`

- Best reported Elo: `1534.9`
- Promotions: `1`
- Best promotion came at iteration `6` with a `55%` win rate

This is encouraging, but it should be treated more cautiously than the 2-player results because multiplayer evaluation was corrected later to rotate one candidate seat per game fairly.

As of April 9, 2026, the first seat-fair multiplayer reruns for `square8_3p` and `square8_4p` are in flight, but they have not yet produced a new completed iteration worth replacing the headline evidence above.

## What Had To Be Fixed Before These Results Were Trustworthy

The current results only became defensible after several bug families were removed.

### Training-contract and model-selection drift

- square `56`-channel helpers still defaulted to the wrong architecture family in multiple places
- hex heavy channel contracts also drifted across helpers

### Rules and parity issues

- hex territory victory thresholds were wrong in the mutable-state mirror
- structural stalemate resolution was incomplete and biased
- helper-level victory labeling had multiple non-canonical fallbacks

### Experiment-harness issues

- “fixed LR” canaries were not truly fixed end-to-end because the inner training subprocess still hardcoded a scheduler
- multiplayer evaluation incorrectly gave the candidate multiple seats in the same game on some runs

The reported April 2026 results should be understood as post-fix results, not as evidence from the earlier buggy harness.

## Staged Evaluation

The current minimal loop uses staged evaluation with early exit for clear wins and losses.

| Stage | Cumulative Games | Promote If |        Reject If |
| ----- | ---------------: | ---------: | ---------------: |
| 1     |             `50` |    `> 60%` |          `< 42%` |
| 2     |            `100` |    `> 56%` |          `< 46%` |
| 3     |            `200` |    `> 53%` |          `< 48%` |
| 4     |            `400` |  `> 50.1%` | otherwise reject |

This replaced a weaker fixed-size eval regime that made near-threshold runs hard to interpret.

## Limitations

The project is not “finished” in a research sense.

Current limitations:

- `hex8_2p` appears to be plateauing near 2000 Elo
- `square8_3p` is still being revalidated under the seat-fair multiplayer evaluator
- larger boards and some 4-player configs remain much slower and less mature
- the strongest results still come from cluster runs, not commodity local hardware

## Reproducing The Supported Path

The supported public entry point is:

```bash
./scripts/run_proven_experiment.sh hex8_2p
./scripts/run_proven_experiment.sh square8_2p
```

That script launches the same minimal-loop configurations used for the reported results and writes:

- `metrics.jsonl`
- `summary.json`
- `models/best.pth`

under `ai-service/data/proven_experiments/<config>/`.

The checked-in snapshot and SVGs are refreshed from local metrics artifacts with:

```bash
npm run results:refresh
```

That command updates [`docs/data/results_snapshot.json`](/Users/armand/Development/RingRift/docs/data/results_snapshot.json) and regenerates the SVGs under `docs/assets/results/`. By default it searches the standard metrics locations under `ai-service/data/` and leaves existing snapshot values in place for any config that is missing local metrics.

## Bottom Line

RingRift now has credible evidence that its neural self-play training system can produce stronger models over time on more than one configuration.

The repo’s next challenge is no longer “does the training loop work at all?” It is:

1. extend that proof cleanly to the remaining multiplayer and large-board paths
2. keep the supported path understandable and reproducible for other engineers
