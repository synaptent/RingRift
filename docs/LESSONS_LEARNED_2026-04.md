# RingRift Lessons Learned: April 2026 Training Debugging

Status: internal engineering retrospective.

This file captures lessons from the late-April 2026 training/debugging cycle.
It is not a public results manifest and should not be used to make strength
claims until the linked experiment gates have settled.

## Why This Exists

The project lost time to a sequence of bugs that looked like model weakness but
were often contract failures:

- checkpoints and exported NPZ data disagreed about encoder contracts
- self-heal logic could classify the wrong failure mode
- model transfer scripts produced checkpoints that loaded as the wrong player
  count
- value-target export silently fell back to winner-only labels
- training probes caught dead value heads, but the first probe interpretation
  was too broad

The important lesson is not any single fix. The important lesson is that
training infrastructure needs explicit contracts at every boundary: checkpoint
metadata, feature encoding, JSONL shape, NPZ labels, value-head shape, runtime
loader behavior, and quality-gate interpretation.

## Evidence Status

Validated locally and in CI:

- `8cbc13ba3 fix(training): make gpu selfplay rank targets observable`
  passed the GitHub `Supported Path` gate.
- The converter now logs replay fallback exceptions instead of silently
  replacing rank-aware multiplayer targets with winner-only targets.
- The minimal loop now writes `final_state` into JSONL records so rank-aware
  multiplayer values can be computed from recorded terminal state rather than
  requiring perfect replay.
- Focused tests cover fallback logging, `final_state` target computation, and
  camelCase move metadata parsing.

Validated on cluster before this addendum:

- `c7aa48f92 fix(ai): initialize v5-heavy FiLM as identity` removed the
  immediate v5-heavy value-head saturation seen at fresh initialization.
- `19ceb1ceb fix(training): resize multiplayer transfer rank heads` fixed the
  3p/4p v4 checkpoint head-shape contract. Runtime detection reported correct
  `num_players` after regenerated canonicals.
- v4 2p success replicated across two seeds, which reduced the chance that the
  original v4 2p trajectory was only lucky initialization.

Pending cluster gates:

- Gate 1: live JSONL on gh200-10/13 includes `final_state`.
- Gate 2: exported 3p/4p NPZ `values` include intermediate rank-aware values,
  not only `-1.0` and `1.0`.
- Gate 3: if Gate 2 passes, training probes no longer hit repeated
  `DEAD_VALUE_HEAD`.
- Gate 4: v5-heavy fv3 value health remains stable over multiple iterations,
  not just the first healthy row after the FiLM fix.

Do not update `docs/data/results_evidence_manifest.json` from these cluster
signals until Gates 2 and 3 are known.

## Timeline Of Root Causes

### 1. Encoder Mismatch Was A Contract Problem, Not Just A Bad Checkpoint

The first v5-heavy blocker was an encoder-family mismatch. Runtime fixes
changed the error from a broad architecture mismatch to the narrower form:
`init_weights=v5-heavy, data=v3`.

That refinement mattered. It showed that the loader was no longer confusing
64-channel v5-heavy weights with an older family, but the training path still
treated compatible v5-heavy/v3 data as incompatible. The fix direction was
therefore contract threading, not random checkpoint replacement.

Relevant commits observed in this chain:

- `8ab08483a fix(training): pass feature version into minimal loop training`
- `85a752205 fix(training): honor runtime checkpoint contracts`

Lesson: error strings should get narrower after a good fix. If a fix only makes
the error disappear by skipping the failing path, it is not evidence.

### 2. Placement-Shortcut Removal Was A Useful Negative Result

The fv3 encoder path removed a suspected placement-validity shortcut. That was
a reasonable hypothesis because v5-heavy consumed richer features and showed
seat-correlated value saturation.

The follow-up run showed the dead value-head pathology persisted after the fv3
placement shortcut was removed. That ruled out the shortcut as the primary
cause of the observed v5-heavy collapse.

Lesson: negative results are useful when they are clean. The fv3 attempt
reduced the hypothesis space even though it did not solve the model.

### 3. FiLM Initialization Was A Real v5-heavy Root Cause

The v5-heavy FiLM path was not initialized as an identity transform. That made
the fresh model produce pathological output scale before learning had a chance
to work.

After `c7aa48f92`, the fresh forward-pass smoke changed from saturated or
non-finite behavior to small, finite value/policy outputs. The first live
training row also passed the value-head quality gate.

Lesson: architecture additions need initialization smoke tests. "The model
loads" is too weak; the initial output distribution must be sane before
self-play starts.

### 4. Multiplayer Transfer Needed Strict Post-save Verification

The v4 3p/4p transfer script initially failed to resize all player-dependent
heads. It handled part of the value path but missed actual output layers:

- `value_fc3`: shape should be `[N, 256]`
- `rank_dist_fc3`: shape should be `[N * N, 256]`

The resulting checkpoints could look plausible while still encoding a 2-player
contract. The correct fix was not only resizing more tensors. The correct fix
was post-save verification: build the target-player model and assert strict
load succeeds.

Lesson: checkpoint migration scripts need an executable target-contract check.
Static tensor edits are not enough.

### 5. Winner-only Multiplayer Targets Were A Silent Data Bug

The v4 multiplayer lanes still diverged after the transfer fix. The next
hypothesis was target quality: for 3p/4p games, assigning every non-winner
`-1.0` is too harsh. Runner-up positions should receive intermediate values.

`5bd85a68e` added rank-aware target computation, but the live NPZ still showed
only `-1.0` and `1.0`. The reason was a silent fallback: replay failed for
every game, and the converter quietly returned winner-only targets.

`8cbc13ba3` fixes the observability gap and adds a more reliable source of
truth by writing `final_state` from the minimal loop.

Lesson: silent fallback in a data exporter is dangerous. If the fallback changes
label semantics, it must be logged at warning level with game id, move index,
move type, phase, exception type, and exception message.

### 6. Probe Distribution Matters

One checkpoint probe showed value variance on training-distribution positions,
while the quality gate saw near-constant root values. That was not a
contradiction. It showed the probe and gate were sampling different position
distributions.

The corrected interpretation was narrower: the model was not necessarily dead
globally; it could be saturated on the root/early-game distribution used by the
quality gate.

Lesson: a "value head is dead" diagnosis must name the sampled distribution.
Global collapse and distribution-specific saturation imply different fixes.

## Operating Principles Adopted

### Prefer Stop-and-preserve Over Running Invalid Science

When a lane repeatedly hits training-probe failures and writes no meaningful
metrics, stop it and preserve artifacts. Idle GPU cost is visible and bounded.
Invalid experiments are worse because they produce data that looks active but
cannot answer the question.

### Keep Productive Lanes Untouched

Lanes that are promoting or producing interpretable metrics should not be
restarted just because adjacent experiments are broken. During this cycle,
productive 2p and square lanes were kept running while failing v4 multiplayer
and v5-heavy experiments were isolated.

### Treat "Activity" And "Value" As Different Metrics

The fleet can be fully active while producing no usable evidence. The right
allocation metric is not GPU occupancy alone; it is occupancy times ability to
answer a concrete hypothesis.

### Keep Codex Reserve Capacity During Gates

Broad repository cleanup has real value, but it has opportunity cost. When
cluster gates are minutes to hours away, the highest-value action can be to
stand by in context rather than start unrelated refactors.

### Underclaim Until Evidence Settles

The repo now has reviewer-facing manifests and supported-path docs. Those are
useful only if they stay defensible. Do not promote chat-only cluster
observations into public evidence docs until the data is checked and stable.

## Current Best Next Actions

If Gate 2 fails:

- inspect the new warning logs from `jsonl_to_npz.py`
- fix the exact replay or `final_state` deserialization error
- do not restart long training until NPZ values show intermediate ranks

If Gate 2 passes and Gate 3 fails:

- stop gh200-10/13 again and preserve artifacts
- shift from export debugging to v4 multiplayer training dynamics
- inspect value-loss scale, rank-distribution loss, gradient norms, and early
  target distribution by player/rank

If Gates 2 and 3 pass:

- let v4 3p/4p run long enough to produce first meaningful metrics rows
- then refresh evidence docs once with settled claims
- consider one additional seed only if the first two lanes disagree

Do later, not during active gates:

- shrink the layer-violation allowlist gradually
- fix `.gitignore` handling for tracked `docs/data/*.json`
- update results/evidence manifests after cluster outcomes settle
- continue demo polish only if it answers a named reviewer concern

## Durable Engineering Checks To Add Over Time

- A JSONL fixture with 3p final rankings that converts to NPZ values containing
  `0.0` for runner-up positions.
- A converter assertion or warning when a 3p/4p NPZ contains only two unique
  terminal values across many completed games.
- A checkpoint-transfer test that regenerates 3p/4p v4 canonicals and strict
  loads them into fresh target-player models.
- A fresh-model forward-pass smoke for every architecture family that checks
  finite logits and reasonable value/policy scale.
- A training-probe report that records the sampled position distribution, not
  only aggregate value statistics.

## Reviewer-facing Takeaway

This cycle should improve external confidence only if it is presented honestly:
RingRift did not simply "train better models." It found and fixed several
places where the training system could produce invalid or misleading signals.

That is valuable because it makes future results harder to fake accidentally.
The strongest project-quality story is the discipline: explicit contracts,
strict gates, preserved artifacts, bounded claims, and willingness to stop
expensive lanes when they stop answering the hypothesis.
