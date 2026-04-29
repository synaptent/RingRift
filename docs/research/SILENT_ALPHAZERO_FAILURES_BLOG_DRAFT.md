# Five silent ways an AlphaZero implementation can lie to you

_Draft. ~1,800 words. Audience: ML engineers who have built or maintain
self-play training systems. Not yet edited for publication; structured as a
LessWrong / personal-site post that can be split into a thread or a paper._

---

If you are running self-play training over a long horizon, the most expensive
class of bug is not a crash. It is the bug that lets every iteration finish
on time, with reasonable-looking loss curves, while the actual training graph
optimizes a different objective from the one your config promised.

I run a small AlphaZero-style training cluster for a novel territory game
called RingRift. Over the last two months, five failure modes of this exact
shape showed up in the codebase. Each one let training keep running long
enough that the only way to detect it was a targeted probe or a quality gate
that looked beyond win rate.

This post is the short version of an internal writeup at
`docs/research/SILENT_ALPHAZERO_FAILURES.md` in the RingRift repository, with
file:line citations to the fixes. The point of writing it down outside the
repo is to make the failure modes available to other people building
AlphaZero-style systems, because each of these bugs is easy to reproduce in a
greenfield project.

## Bug 1: A head defined in `__init__` that the forward pass never returns

The cleanest example. `HexNeuralNet_v4` had four layers explicitly named
`rank_dist_fc1`, `rank_dist_fc2`, `rank_dist_fc3`, and `rank_softmax`. Every
saved checkpoint shipped these tensors. The training pipeline was wired to
expect a rank-distribution output for multiplayer games (the head encodes
`P(player i finishes in rank j)`).

But the forward pass returned `(value, policy)`.

Not `(value, policy, rank_dist)`. Two outputs.

Multiplayer training tolerated this because the loss code branched on the
return arity. When the model returned two tensors, the rank-loss term was
silently zero. When the model returned three tensors, the rank-loss term
contributed correctly. There was no warning, no log line, no degraded
metric. The architecture diagram was correct. The state dict was correct.
The training pipeline ran for many iterations producing two-player-style
loss curves that looked sane.

The fix was a four-line addition to `forward()` to actually compute and
return the rank distribution, plus a unit test that exercises the
multiplayer return contract. Total commit footprint: 18 insertions, 6
deletions. The cost of the bug, in terms of invalidated training runs, was
much larger than that.

**Lesson**: every output head deserves a forward-shape unit test. If your
model class has a layer named `something_head`, your test suite should
require that the forward pass returns it.

## Bug 2: A flag that promotes config to behavior, dropped at the boundary

The minimal training loop accepted `--num-players 3` (or 4) on the command
line. It built a subprocess command for `python -m app.training.train` with
all the right arguments — board type, model version, NPZ path, learning rate.

Except `--multi-player`.

The training script defaults to `multi_player=False`. When that flag is
false, the rank-distribution loss term is gated off. So a 3-player run that
was launched correctly at the outer layer would silently degrade to a
two-player loss at the inner layer, even though logs, NPZ metadata, and
model architecture all pointed at "3p" everywhere.

The fix was two lines:

```python
if NUM_PLAYERS > 2:
    cmd.append("--multi-player")
```

The reason it took weeks to find: the bug expressed itself as "multiplayer
configurations don't seem to make Elo progress". That's not a stack trace.
It's a slow research-direction lie.

**Lesson**: any wrapper that builds a CLI command for an inner training
script needs to be unit-tested by capturing the exact command and asserting
it contains the flags the experiment requires. "Did the subprocess run?" is
not enough. "Did the subprocess run with the loss flags this experiment is
meant to optimize?" is the question.

## Bug 3: A migration script that exits successfully without doing the work

`transfer_2p_to_4p.py` was originally written when the model architecture's
last value layer was named `value_fc2`. The script walked the source
checkpoint, found tensors whose name contained `value_fc2` and whose first
dimension was 2, copied them, resized to 4 outputs, and saved a new
checkpoint.

When v4 added `value_fc3` and a separate `rank_dist_fc3` head, the script
kept running fine. It just didn't match anything any more, because v4
checkpoints had `value_fc3` not `value_fc2`. The output checkpoint existed.
The script printed success. The downstream training run loaded the wrong
weights silently.

The fix is more interesting than the bug. It rewrote the matcher as a
predicate against the _intent_ of each head:

```python
def _is_final_value_head_key(key, value, source_players):
    return (
        key.endswith((".weight", ".bias"))
        and any(name in key for name in ("value_fc3", "value_fc2", "value_head"))
        and value.ndim in (1, 2)
        and value.shape[0] == source_players
    )
```

— and added a strict-load step that materializes the result back into the
target architecture, so an incompatible transfer fails at generation time
instead of at training time.

**Lesson**: any script that "transforms a checkpoint" should report exactly
which tensor families it touched, fail loudly if any expected family is
missing, and verify the result loads strict against the intended target
class. Silent no-ops on schema drift are otherwise inevitable.

## Bug 4: An init that compounds to saturation through depth

This one was the most interesting failure to diagnose. Symptoms:

- Training probe printed `value_std=0.000000, mean=±1.0` from epoch 1.
- FP16 autocast emitted non-finite warnings every iter.
- Weight delta L2 between consecutive iters spiked to ~140+, then
  collapsed to noise.
- Standard architecture, parameter count, and forward shape all matched
  expectations.

The architecture was a wide v5-heavy model with a heuristic-feature encoder
that emits FiLM (feature-wise linear modulation) gamma/beta scaling factors.
The intent of the FiLM layer was to behave as identity at init: gamma should
start near 1.0, beta near 0.0, so untrained heuristic encoders contributed
nothing.

The init looked correct on paper. It zero-initialized half of the gamma
weights and one-initialized the other half:

```python
nn.init.ones_(self.film_gamma.weight.data[:, :output_dim // 2])
nn.init.zeros_(self.film_gamma.weight.data[:, output_dim // 2:])
```

The author's logic was: the heuristic encoder output has a known structure,
half of it should pass through as gamma=1, the other half should be
untouched at zero. Combined with bias=0, the expected effective gamma at
init is 1.

This works if the heuristic encoder output is centered at zero. The
RingRift heuristic encoder uses ReLU. So the encoder output is positive,
non-centered, and concentrated. Half-ones in the gamma matrix then produce
a gamma vector that is _not_ 1 — at init it averages around 3.3.

Six SE blocks later, that 3.3-per-block scale compounds. By the time the
input reaches the value head, the magnitudes are already saturated. The
value head saturates to ±1.0 before any gradient step. After that the
network is dead.

Fix: zero-init everything in FiLM, including gamma. Identity is `gamma=1`
mathematically, but `gamma=0` is how you actually get identity through a
ReLU encoder, because the additive bias inside the FiLM application is
what restores `gamma=1` after the heuristic input is processed.

**Lesson**: forward-shape tests don't catch initialization that compounds.
Add a smoke test that requires `value.abs().max() < 0.5` and `policy.std()
< 2.0` on random input at init, and a sensitivity test that perturbs the
conditioning input and requires nonzero output deltas.

Crucially: every checkpoint saved before the init fix is poisoned. The
bad gamma weights are baked in. They cannot be fine-tuned out reliably.

## Bug 5: A quality gate that loses its denominator on resume

The newest example is not a broken forward pass or missing flag. It is a
quality gate that made a promotion look unsafe because its internal sample was
partial after an evaluation resume.

One RingRift fv3 reference run had apparently plateaued: rejects at 47-48%
for several iterations around 1587 Elo. Then iteration 12 suddenly beat the
current best by 36 games to 14, a 72% staged-evaluation win rate. A naive
promotion rule would have accepted the model.

The model was rejected because the quality gate said every tracked game shared
the same opening. The metric row looked damning:
`decision=promote`, `promoted=false`, `quality_gate.passed=false`,
`quality_gate.critical=true`, `MODE_COLLAPSE`.

The clue was hidden in the denominator: `1/1 games`.

The staged evaluator had a resume checkpoint with the 50-game outcome count,
but the quality tracker only had move-level data from the post-resume suffix.
So the row combined two different sample sizes: full outcome evidence and
one-game behavioral-diversity evidence. Iteration 13 then clean-promoted at
38-12 over 50 games with the quality gate passing, which made the iter 12
interpretation much clearer.

**Lesson**: quality gates need their own provenance and coverage fields.
If an evaluation can resume from a checkpoint, either persist the gate's
move-level state too or mark behavioral/value checks as partial and non-critical
for that iteration. A gate that catches bad models is useful. A gate that hides
its sample denominator becomes another silent failure.

RingRift's fix is now committed as `4e1b7e20e`
(`fix(coordination): persist quality-gate tracker state across eval resume`).
For publication, the clean evidence pair is the copied control-vs-treatment
record: iter 12 was a strength-positive promotion blocked by the legacy
partial-sample gate, while iter 13 was a strength-positive promotion that
passed the gate.

## The general class: The experiment name lies

The five specific bugs above are all instances of a more general failure
mode: **the experiment name claims one thing, the actual training graph
or evaluation gate proves another, and both keep running long enough that
nobody notices**.

The mismatch can be at four different layers:

1. **Model contract**: the architecture has a head it never returns, or a
   conditioning path that contributes nothing, or a forward signature that
   silently degrades a feature when called certain ways.

2. **Training contract**: the wrapping CLI doesn't propagate flags into the
   inner training process, or doesn't activate loss terms the experiment
   was supposed to optimize, or runs the inner training loop with stale
   defaults that override the outer config.

3. **Experiment contract**: the metrics and logs do not record what the
   training graph actually computed. There is no way to recover from
   artifacts whether the run that produced them was the run that was
   supposed to be produced.

4. **Evaluation contract**: the promotion gate measures win rate, behavioral
   sanity, and the coverage of the quality checks themselves.

In RingRift, the first three layers each had a real bug at some point
during 2026, and the fourth layer caught a real near-miss. None of them
crashed.

The reusable guardrail is to test contracts at all four layers
independently:

- For the model: forward returns the documented tuple shape for every
  supported `return_features` value, conditioning paths produce nonzero
  output deltas under input perturbation, random-input forward keeps
  `value.abs().max()` and `policy.std()` within sane bounds at init.

- For the training CLI: unit-test the exact subprocess command emitted by
  any wrapper loop or training launcher. Assert `multi_player`,
  `rank_dist`-loss, and per-head loss weights match the experiment
  configuration.

- For the experiment: every artifact (checkpoint, NPZ, log line) should
  let an outsider reconstruct what was actually optimized, not just what
  was intended.

- For evaluation: persist quality-gate verdicts next to win-rate decisions,
  make promotion require both, and include sample coverage for each gate.

## Why I am writing this externally

In a sense, the most useful artifact a long-running self-play training
project can produce is not the strongest model. It is the catalog of
failure modes that didn't crash. The strongest model is, at best, locally
interesting. The catalog is reusable.

Every AlphaZero-style codebase I have seen has at least one of these
contract gaps live in production right now. If you have inherited or
written one and you have not specifically built tests for the model,
training, and experiment contracts as separate things, this post is
mostly an excuse to nudge you into doing it.

The full reproducer index, with commit links and test paths, is at
[`docs/research/SILENT_ALPHAZERO_FAILURES.md`](./SILENT_ALPHAZERO_FAILURES.md)
in the RingRift repository.
