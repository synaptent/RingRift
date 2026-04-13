# RingRift Results

This document summarizes the current research evidence from the RingRift self-play training project.

Status is current as of April 13, 2026.

## Headline Results

| Config       | Start Elo | Best Reported Elo | Promotions | Status                                                         |
| ------------ | --------: | ----------------: | ---------: | -------------------------------------------------------------- |
| `hex8_2p`    |    `1500` |          `1979.8` |        `7` | Strongest result; promoted again at iteration `33` on fixed LR |
| `square8_2p` |    `1500` |          `1601.8` |        `2` | Second clean 2P proof under the corrected minimal-loop harness |
| `square8_3p` |    `1500` |          `1534.9` |        `1` | Only current multiplayer promotion; still weak evidence        |
| `square8_4p` |    `1500` |          `1500.0` |        `0` | No proven improvement above baseline                           |

![Headline results snapshot](assets/results/headline_results.svg)

## Why These Results Matter

RingRift was built as a novel deterministic strategy game plus an end-to-end AlphaZero-style training system. The central question was not only whether the system could run, but whether it could produce real iterative neural-network improvement on a nontrivial new game.

The answer is now yes.

The strongest evidence is still `hex8_2p`, and `square8_2p` remains important because it demonstrates that the loop can improve on a second configuration under the same general training stack. The current state is not a universal success story, though: only `2` of `12` configs have strong evidence of improvement, multiplayer remains weak, and larger boards remain unproven.

## Concrete Evidence

### `hex8_2p`

- Best reported Elo: `1979.8`
- Promotions: `7`
- Latest milestone: iteration `33` promoted on the fixed-learning-rate minimal-loop line
- Interpretation: strong iterative improvement from the `1500` baseline to a checkpoint family that is now within one more promotion of the `2000` Elo headline

The older checked-in April 10 snapshot still stops at `1967.6`, but the current live cluster line advanced once more to `1979.8`. That does not change the scientific interpretation: the path is real, but it is also clearly in the plateau regime where uninterrupted runtime matters more than more infrastructure churn.

### `square8_2p`

This remains the clearest recent improvement story after `hex8_2p`.

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
- `square8_2p` remains the second clean proof point for iterative NN improvement even though it is materially weaker than `hex8_2p`

### `square8_3p`

- Best reported Elo: `1534.9`
- Promotions: `1`
- Best promotion came at iteration `6` with a `55%` win rate

This should be treated cautiously. Multiplayer evaluation was corrected later to rotate one candidate seat per game fairly, and the recent April 10 seat-fair evaluations remain poor: `22%`, `24%`, then `22%` candidate win rate in the latest metrics tail.

As of April 13, 2026, the result is still not strong enough to claim robust multiplayer progress. What it does show is that the multiplayer path is no longer blocked by the earlier evaluator bug; it now has one promotion under the corrected threshold and seat-fair regime, but it still needs another clean promotion before it should be treated as persuasive evidence.

### `square8_4p`

- Best reported Elo: `1500.0`
- Promotions: `0`
- Latest completed eval in the status snapshot was roughly `46%`

This configuration has not demonstrated improvement above baseline. It remains scientifically unproven and is not part of the current strongest evidence path.

## April 13 Methodology And Operations State

The most important fact about the April 13 state is that the training claim now depends more on preserving uninterrupted runtime than on changing infrastructure again. The `hex8_2p` path has already demonstrated the core research result. The next value is incremental: get `hex8_2p` over `2000`, get `square8_2p` to promote again, and get one more clean multiplayer promotion.

The active runtime is now organized as a role-based fleet:

- `5` trainers: `gh200-8` (`hex8_2p`), `gh200-9` (`square8_2p`), `gh200-10` (`hex8_3p`), `gh200-12` (`square8_3p`), and `gh200-14` (`square19_2p`)
- `2` selfplay workers: `gh200-11` feeding `hex8_2p`, and `gh200-13` feeding `square8_2p`
- `1` dedicated evaluator: `vultr-a100-20gb`

The reason for this role split is operational, not cosmetic: trainers must spend GPU time on uninterrupted minimal-loop training, while selfplay workers produce supplemental policy data without spawning extra GPU work on trainer nodes.

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
- trainer nodes previously allowed P2P selfplay work to contend for the same GPU, which made loop timing and progress unreliable

The reported April 2026 results should be understood as post-fix results, not as evidence from the earlier buggy harness.

### Hyperparameter and threshold corrections that now matter

- fixed learning rate `5e-5` is now the proven baseline on the supported path
- staged evaluation is the active promotion gate
- multiplayer promotion thresholds were corrected so 3-player and 4-player runs are judged against the right criteria instead of the 2-player defaults

These changes are part of why the `hex8_2p` and `square8_2p` results should be treated as current evidence, while earlier unstable or misconfigured runs should not.

### Supplemental policy-data pipeline

The supplemental policy-data path is now proven far enough to matter operationally:

- selfplay workers generate policy-bearing Gumbel JSONL
- ingestion converts those raw files into supplemental NPZ shards
- those shards land in the target trainers' supplemental directories with metadata manifests

That is the key end-to-end evidence needed for the current role-based architecture. It means workers can produce useful policy-bearing data for the trainers without pretending the main training claim depends only on pure trainer-local selfplay.

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

- `hex8_2p` appears to be plateauing near `2000` Elo
- `square8_2p` has a credible `1601.8` result, but it still needs another promotion to make the second-board result feel robust rather than merely positive
- `square8_3p` has only one multiplayer promotion and still looks weak under the corrected evaluator
- `square8_4p` remains at baseline
- no new iteration had completed since April 11 during the latest operational churn window, so cluster time was spent mostly recovering stability instead of accumulating fresh evidence
- larger boards and other 3-4 player configs remain much slower and less mature
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

RingRift has credible evidence that its neural self-play system can produce stronger models over time on more than one configuration.

That evidence is real, but it is still narrow. The strongest supported claim is:

- `hex8_2p` improved from `1500` to `1979.8`
- `square8_2p` improved from `1500` to `1601.8`
- the corrected minimal-loop stack, fixed-LR baseline, staged evaluator, and role-based fleet are sufficient to keep pushing that line of evidence forward

The next challenge is not proving that the loop can ever improve. It is:

1. keep the supported path reproducible and uninterrupted long enough to finish more iterations
2. get `hex8_2p` over `2000` Elo and `square8_2p` to promote again
3. extend the proof to multiplayer and larger-board paths without overstating weak results
