# RingRift Project Brief

This document is the shortest external-facing explanation of what RingRift is, what the repository currently proves, and which parts of the codebase matter first.

## What RingRift Is

RingRift is both a deterministic abstract strategy game and a research codebase for training neural agents on a novel multiplayer ruleset. The repository contains a playable web application, a canonical TypeScript rules engine, a Python mirror used for AI and replay validation, and a narrow self-play training harness used to produce the current experiment results.

## What The Project Has Proven So Far

As of April 15, 2026, the strongest reported results are:

| Config       | Best Reported Elo | Promotions | Notes                                                                       |
| ------------ | ----------------: | ---------: | --------------------------------------------------------------------------- |
| `hex8_2p`    |          `1979.8` |        `7` | Strongest overall result; current v3 line now looks plateaued near 2000 Elo |
| `square8_2p` |          `1697.3` |        `4` | Clean fixed-LR run is now a materially stronger second proof point          |
| `square8_3p` |          `1534.9` |        `1` | Promising multiplayer evidence, but still too weak to overclaim             |

These results came from longer GH200 cluster runs. Local reproduction is possible through the supported experiment path, but the same scale of improvement still depends on GPU time and multiple iterations.

The supported claim is therefore narrow and defensible: RingRift has shown iterative NN improvement clearly on `hex8_2p` and `square8_2p`, with only an early multiplayer signal so far. The next branch of work is not more orchestration. It is whether a new `hex8_2p` `v4` architecture can break the current plateau.

## The Current Supported Path

If you are evaluating the project as an engineer, follow this path:

1. [README.md](/README.md)
2. [QUICKSTART.md](/QUICKSTART.md)
3. [docs/RESULTS.md](/docs/RESULTS.md)
4. [docs/ARCHITECTURE_OVERVIEW.md](/docs/ARCHITECTURE_OVERVIEW.md)
5. [scripts/run_proven_experiment.sh](/scripts/run_proven_experiment.sh)

The most important code behind that path is:

- [`src/shared/engine`](/src/shared/engine): canonical TypeScript rules engine
- [`ai-service/app`](/ai-service/app): Python AI and parity mirror
- [`ai-service/scripts/minimal_alphazero_loop.py`](/ai-service/scripts/minimal_alphazero_loop.py): supported experiment harness
- [`ai-service/scripts/check_ts_python_replay_parity.py`](/ai-service/scripts/check_ts_python_replay_parity.py): trust boundary for replay parity

## What To Ignore At First

The repository is historically layered. Not every subsystem is equally current.

Do not start with:

- `archive/`
- `docs/archive/`
- the full cluster automation surface under [`ai-service/scripts`](/ai-service/scripts)
- internal assistant memory files such as `CLAUDE.md`

Those areas are real and often useful operationally, but they are not the shortest path to understanding the project’s main claims.

## How To Read The Repository

The project is easiest to understand as four layers:

1. canonical TypeScript rules
2. playable web app
3. Python AI/parity mirror
4. minimal self-play training loop

Everything else is support infrastructure, operations, or historical context until you specifically need it.

## Bottom Line

RingRift is presentable if you judge it by its supported path rather than by every historical script in the tree. The current codebase has a clear rules source of truth, a reproducible experiment entrypoint, and documented evidence for real iterative improvement on multiple configurations.

If you need a version of this story that is more results-focused and easier to share externally, see [docs/RESEARCH_SNAPSHOT.md](/docs/RESEARCH_SNAPSHOT.md).
