# RingRift Lessons Learned

This document is the engineering retrospective for RingRift as a software and
training system, not a marketing summary.

The interesting lessons were not “AI is hard” or “distributed systems are
complex.” They were more specific: how reasonable safety layers combined into
failure loops, how evaluation protocol details changed the interpretation of the
results, and why a narrow proof harness beat the feature-rich stack.

## 1. The Watchdogs Could Cause The Failure They Were Supposed To Prevent

RingRift accumulated multiple restart and liveness layers:

- `ai-service/scripts/master_loop_watchdog.py`
- `ai-service/deploy/sentinel/ringrift_sentinel.c`
- `ai-service/scripts/p2p_watchdog.py`
- `ai-service/scripts/p2p/loops/http_server_health_loop.py`
- `ai-service/scripts/p2p/entrypoint.py`

Each piece was defensible on its own.

- `ringrift_sentinel.c` watches a heartbeat file and restarts the watchdog if
  it goes stale.
- `p2p_watchdog.py` checks the P2P process and prefers `systemctl` for managed
  services.
- `http_server_health_loop.py` detects the “process alive, HTTP dead” zombie
  state and forces a restart after roughly `70` seconds.
- `entrypoint.py` starts an event-loop watchdog that exits after `5`
  consecutive failed probes.

The problem was composition. A transient slowdown or blocked event loop could
trip one watchdog, which triggered a restart, which re-entered startup grace
periods, which looked unhealthy to a different watchdog, which triggered more
restarts. The fleet could spend more time proving it was alive than doing
self-play.

The lesson is simple: a liveness layer is not free just because it is
well-intentioned. Every watchdog is part of the workload. When the stack has
four or five of them, you need a single authority for restart policy and a
clear ownership boundary for “observe only” versus “kill and replace.”

## 2. Process Ownership Matters More Than Process Detection

The P2P layer used broad process detection and cleanup patterns because that was
the easiest way to recover from orphaned workers:

- `ai-service/scripts/p2p_watchdog.py`
- `ai-service/deploy/systemd/ringrift-p2p.service`
- `docs/architecture/PART4_QUALITY_ROADMAP.md`

That worked until the ownership model changed.

`ringrift-p2p.service` still contains the historical scar tissue in comments:

- orphan selfplay processes accumulated across restarts
- `28+` processes times `80` threads pushed nodes toward `LimitNPROC`
- the service needed `KillMode=control-group`

The more subtle failure was the opposite one: a cleanup rule that was too broad
could kill systemd-managed or independently-managed workloads that happened to
match the same process-name pattern. The fix direction in
`PART4_QUALITY_ROADMAP.md` was the right one: keep `ringrift-p2p.service`
enabled, remove the broad `ExecStopPost pkill`, and make role-aware services
own their own children.

The lesson: “find by name and kill” is not orchestration. Once a system has
`systemd`, worker roles, and independent runtimes, the only safe cleanup is
cleanup that respects the control group or service boundary.

## 3. The Minimal Loop Won Because It Had Fewer Hidden Policies

The most important engineering decision in the project was not a model change.
It was keeping a narrow proof harness alive:

- `ai-service/scripts/minimal_alphazero_loop.py`
- `scripts/run_proven_experiment.sh`
- `docs/architecture/MINIMAL_LOOP_CONTRACT.md`
- `docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md`

The larger coordinator/P2P stack had real value:

- supervision
- transfer
- queues
- health
- worker orchestration

But it also carried more implicit behavior around scheduling, restart policy,
data movement, and evaluation wiring. That made it harder to answer a simple
question: “what exactly produced this checkpoint?”

The minimal loop answered that question cleanly. It owned a complete iteration
and wrote explicit artifacts:

- `metrics.jsonl`
- `progress.json`
- `models/best.pth`

That is why the strongest claims in the project come from the minimal loop, not
from the more feature-complete legacy training stack. The smaller system had
less power, but much better epistemics.

## 4. Fixed Learning Rate Beat The Fancy Schedule Because The Fancy Schedule Was Compounding

The hyperparameter breakthrough was not exotic. It was removing accidental
complexity.

Relevant files:

- `docs/REPRODUCIBILITY.md`
- `ai-service/scripts/minimal_alphazero_loop.py`
- `ai-service/app/training/train_cli.py`
- `docs/RESULTS.md`

The critical insight was that “sqrt decay + cosine” was not one schedule. It
was two schedules multiplying each other:

- loop-level decay from the minimal loop
- epoch-level decay from the inner training subprocess

The checked-in project docs record the result plainly:

- `hex8_2p` stalled at `1967.6` on the older line
- switching to fixed `5e-5` produced the later `1979.8` promotion

This was not a grand search-space win. It was a contract bug. The outer loop
believed it was running a “fixed LR” canary while the inner trainer was still
free to apply cosine decay. Once the loop was truly fixed end to end, the line
started moving again.

The lesson: when a training system has more than one scheduler surface, the
default assumption should be that they will interact in a way the operator did
not intend.

## 5. Multiplayer Evaluation Was A Protocol Problem Before It Was A Threshold Problem

The multiplayer weakness in RingRift was not just “3-player is harder.”

Relevant files:

- `docs/RESULTS.md`
- `docs/RESEARCH_SNAPSHOT.md`
- `ai-service/scripts/minimal_alphazero_loop.py`

The project had to correct a more basic problem first: some multiplayer
evaluation runs effectively gave the candidate more than one seat in the same
game. That meant the observed win rate was not measuring the scenario the
project thought it was measuring.

After the evaluator was corrected to rotate exactly one candidate seat per game,
the meaning of the multiplayer numbers changed:

- `square8_3p` still has only one promotion and later rejects at `20%`, `30%`,
  and `24%`
- `hex8_3p` finally completed a first clean result, but it rejected at `35%`

That result is actually useful. It means the system moved from “evaluation was
partly invalid” to “evaluation is valid and the candidate still loses.”

The lesson: a fair evaluator can make a project look worse in the short term.
That is progress, not regression.

## 6. “Self-Play Data” Was Not A Single Thing

One of the most important data discoveries was that P2P self-play output was
not automatically training-ready for policy learning.

Relevant files:

- `ai-service/scripts/generate_gumbel_selfplay.py`
- `ai-service/scripts/ingest_policy_selfplay.py`
- `ai-service/scripts/jsonl_to_npz.py`
- `ai-service/scripts/policy_selfplay_worker.py`
- `docs/architecture/PART4_QUALITY_ROADMAP.md`

The core issue was simple: some self-play traces had useful MCTS visit
distributions and some did not. Treating both as equivalent “self-play data”
silently polluted the policy target path.

The fix was to make policy-bearing data explicit:

- `generate_gumbel_selfplay.py` now emits per-move `policy_target`
- `jsonl_to_npz.py` skips `policy_target=false` moves
- `ingest_policy_selfplay.py` rejects inputs with no policy-bearing records
- the trainer only consumes those worker outputs through a supplemental NPZ lane

That split matters because it turned a fuzzy data lake into a contract:
supplemental worker data had to prove that it contained policy targets before it
was allowed into the training set.

The lesson: “we have more data” is not progress if the labels you need are only
present in part of that data.

## 7. The 17K Python File Count Was Mostly A Workspace Smell, Then A Scope-Control Smell

The raw file-count scare looked catastrophic until it was measured properly.

Relevant files:

- `docs/CODEBASE_QUALITY_PROGRAM.md`
- `ai-service/archive/**`
- `docs/archive/**`

The April 2026 audit found:

- `17,381` raw `*.py` files in the workspace
- `14,015` of them inside local `.venv`
- `3,366` repo-relevant Python files before pruning
- `3,359` repo-relevant Python files after removing `7` dead tracked archive files
- `652` `__pycache__` directories
- `4,826` `*.pyc` files

So the first lesson was quantitative: the repo did not literally contain
17,000 maintained source files.

The second lesson was still uncomfortable. The active tree did contain too many
scripts, daemons, compatibility shims, and historical layers because the repo
had repeatedly rewarded vague “improve the system” automation with additive
surface area. That is how you end up with more watchdogs, more migration notes,
more sidecar scripts, and more archive weight than the core proof path needed.

The lesson: if the prompt is broad and the acceptance criteria are vague,
autonomous tooling will usually add files faster than it removes them.

## 8. What Actually Worked

The parts that produced the best outcomes were surprisingly consistent:

- a single supported proof harness
- explicit artifacts per iteration
- parity checks at the TS↔Python boundary
- staged evaluation with visible thresholds
- narrow package/documentation ratchets
- role-aware deployment instead of broad process heuristics

That is the practical takeaway from the project.

The most successful pattern was not “build the most powerful platform.” It was:

1. shrink the claim
2. make the claim measurable
3. put the measurement on disk
4. remove every hidden behavior that can falsify the interpretation

That pattern produced the results that are still worth showing:

- `hex8_2p`: `1500 -> 1979.8`
- `square8_2p`: `1500 -> 1697.3`

Everything else should be read through that lens.
