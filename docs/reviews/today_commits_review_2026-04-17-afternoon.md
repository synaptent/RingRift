# Today's Commits Independent Review (2026-04-17 afternoon)

## Summary

The C2 persona chain (5aa0fe3be + 186a74bd5 + 1d9393a9d) is clean end-to-end
with the 4-name allow-list locked in all three layers, dual feature flags
default-off, and appropriate defence-in-depth validation — ready to flip. The
v5-heavy channel-alignment fixes (c9a43020c + 5764c2656) are correct and
plug a real crash. The C1 ensemble work (cb8eca501 + 7d692717d), however,
has **one HIGH-severity latent bug in the budget-reduction fix**: the AI
instance cache key does not include `gumbel_simulation_budget`, so the
"reduced-budget primary" collapses onto the already-cached full-budget
primary and the intended CPU-halving never actually happens for the primary
constituent — only for the extras. The accompanying test validates only the
arithmetic of `budget // N`, not the real AI instance's effective budget, so
this slipped the gate.

## Per-commit findings

### 7d692717d — ensemble budget reduction

- **HIGH** — `_ai_cache_key` does **not** include `config.gumbel_simulation_budget`
  among its key fields (see `app/main.py:442-458`). The fix tries to cache the
  reduced-budget primary under a key computed from
  `config.model_copy(update={"gumbel_simulation_budget": reduced_budget})`,
  but since the budget isn't part of the key, `ensemble_primary_cache_key`
  is byte-identical to the full-budget `cache_key`. Concretely, on the first
  D10 request after a restart:
  1. `ai` is built at full budget (400) and cached under `cache_key`.
  2. Ensemble branch computes `ensemble_primary_cache_key` — identical string.
  3. `_get_cached_ai(ensemble_primary_cache_key)` returns the full-budget
     `ai` instance that was just cached.
  4. `ensemble_primary` is therefore the full-budget AI instance, and the
     ensemble runs at **400 sims for the primary + 200 sims for each extra**,
     not 200+200 as intended.
     The commit message claims the fix halves CPU work; in practice it only
     halves it for extras. Wall-clock ≈ max(full-budget sim, reduced-budget sim)
     = roughly the same as today's failing 2×400 case, since the full-budget
     primary's search time still dominates. Whether the 30s timeout is actually
     satisfied depends on how the extras finish before the primary — it likely
     is closer to single-model latency than the pre-fix case, but the fix is
     not doing what the comments say it does. Mitigation options:
     (a) add `gumbel_simulation_budget` to `_ai_cache_key`, or (b) force the
     ensemble path to construct the primary fresh (skip the cache lookup) with
     the reduced-budget config, or (c) rebuild `ai` with the reduced-budget
     config when the ensemble branch triggers. This was also the exact scenario
     the scrutiny question flagged ("Any risk the two get confused, or that
     stale cache entries survive across config changes?").

- **MEDIUM** — Secondary consequence of the key collision: `_put_cached_ai(
ensemble_primary_cache_key, ensemble_primary)` overwrites the cached
  `ai` whenever the fresh-build branch is reached (e.g., after a cache
  eviction). That means a subsequent request on a non-ensemble code path
  (flag off, difficulty out of {9,10}, no extras configured for some
  (board, np)) can retrieve the _reduced-budget_ instance under the
  original `cache_key` and silently serve reduced-budget moves outside
  the ensemble window. Fixing the key also fixes this.

- **LOW** — The parametrized test `TestEnsembleBudgetReduction` recomputes
  the formula `reduced = max(1, full_budget // ensemble_size)` **inline
  inside the test body** rather than invoking the real helper or the
  `/ai/move` path (see
  `ai-service/tests/unit/test_main_ensemble_config.py:156-161`). It
  therefore cannot catch any regression where production code silently
  drops the override (e.g., the cache-key issue above). The regression guard
  the commit message advertises is essentially a tautology.

- **LOW** — `AIConfig.gumbel_simulation_budget` has a pydantic constraint
  `ge=10` (`app/models/core.py:749`). `pydantic.BaseModel.model_copy(update=...)`
  does not revalidate by default, so `reduced_budget=1` (the floor) or
  `reduced_budget` under 10 will slip past validation and land in the
  constituent config. Unlikely to matter in the live ladder (min full
  budget is 64, so floor is ~21 at N=3), but worth noting — if a future
  N=8 ensemble is configured, you could end up with `reduced_budget` under
  the schema's declared minimum and no complaint at copy time.

- **LOW** — Non-Gumbel tiers (MCTS, DESCENT) hit the ensemble branch if
  configured but leave `reduced_budget=None`. In that code path
  `ensemble_primary_cache_key` stays None, so the ensemble primary is
  rebuilt fresh on every request (lines 1595-1608) instead of reusing the
  already-cached `ai`. Trivially fixable by setting `ensemble_primary = ai`
  when `reduced_budget is None`; in the meantime, D9/D10 ensemble for
  non-Gumbel tiers pays per-request tree init cost.

- **NIT** — Log line `logger.info("C1 ensemble vote: size=%d agreement=%d/%d
failures=%d per_model_budget=%s picks=%s", vote.ensemble_size,
vote.agreement_count, vote.ensemble_size, vote.failures, ...)` repeats
  `vote.ensemble_size` twice (once as size, once as denominator). Both
  slots end up identical; the intended "size/total" display collapses.

### cb8eca501 — ensemble serving infrastructure

- **MEDIUM** — `asyncio.to_thread` + `asyncio.wait_for` on a blocking
  `ai.select_move` cannot actually cancel the worker thread when the
  outer timeout fires. On `EnsembleFailure("…timed out…")`, the blocking
  threads continue running tree search until their internal time budget
  exhausts. Under sustained timeout pressure this can exhaust the default
  ThreadPoolExecutor (`min(32, os.cpu_count()+4)` workers) and amplify
  latency across _all_ requests, not just the D9/D10 ensemble branch.
  Pre-existing characteristic of the single-model path too, but the
  ensemble branch multiplies the surface area by N and makes it more
  likely to manifest during the very deploy scenario this code was
  written for. Worth either an executor dedicated to tree search, or
  documented operational awareness.

- **MEDIUM** — `gumbel_budget` on the primary path is used as the source
  of truth for whether the tier is "gumbel-flavoured" (`if gumbel_budget
is not None`), but `ai_type in (AIType.GUMBEL_MCTS, AIType.MCTS,
AIType.DESCENT)` also admits MCTS / Descent, which are _not_ Gumbel.
  The ensemble therefore fires for those tiers too but silently skips
  the budget-reduction override (non-Gumbel has no `gumbel_simulation_budget`).
  Combined with the cache-rebuild-every-time bug above, this is a
  known-live but untuned path. Either tighten the gate to
  `AIType.GUMBEL_MCTS` only, or propagate a generic "search budget" field.

- **MEDIUM** — `select_move_ensemble` has no explicit graceful-shutdown
  behaviour when `asyncio.gather` is cancelled (e.g., the caller itself
  is cancelled before the timeout expires). In particular, if a downstream
  caller cancels the request, the gather is cancelled but the underlying
  `to_thread` futures remain running — same class of leak as the
  timeout one, just driven by cancellation rather than timeout.

- **LOW** — `_ensemble_extra_checkpoints` calls `board_type.value if
hasattr(board_type, "value") else str(board_type)` — for a `BoardType`
  enum, `.value` is something like `"hex8"`, but the key format
  `f"{board_name}_{num_players or 2}p"` silently coerces `num_players=0`
  to 2. Probably fine in practice (0 players is nonsensical), but a
  genuinely missing `num_players` and a pathological `0` are indistinguishable.
  Not a security problem; the caller supplies this.

- **LOW** — JSON env-var parsing is correctly fail-safe (swallowed errors
  return `[]`) so an operator typo can't 5xx the endpoint. **No injection
  risk**: the env var is operator-controlled, and even maliciously
  crafted JSON would only be able to point to a local path; `_create_ai_instance`
  already gates on filesystem access and will log + skip on failure.

- **LOW** — `_repr_move` relies on `getattr(move, "type", None)` and
  `.from_pos` / `.to` attributes. If an AI ever returns a move type with
  the same `type.value` but different extra fields (e.g., `chain_capture`
  with different `capturedStacks`), these are collapsed into the same
  vote bucket. Today's `Move`s are dominated by `place` and `move_stack`
  so this is benign. Document the assumption or extend the key.

- **LOW** — `test_concurrency_latency_is_max_not_sum` (`_SlowAI` sleeps
  0.3s ×3, asserts <0.7s) relies on the default thread pool having at
  least 3 free workers at test time. In a CI worker with many parallel
  pytest processes sharing the same OS, this could flake. Not yet
  observed but worth a watch.

- **NIT** — `EnsembleVoteResult.agreement_fraction` is reported as
  `agreement_count / len(successful_moves)` rather than
  `agreement_count / len(ais)`. That means a vote where 2 of 3 AIs fail
  and the surviving one produces a move returns `agreement_fraction=1.0`
  with `failures=2`. The docstring says "0.0 on single-model fallback"
  but no path actually sets it to 0.0 — this field's semantics are
  a little ambiguous when failures exist.

### 1d9393a9d — persona picker UI + per-seat propagation (C2 phase 3)

- **MEDIUM** — `AIPersonaPicker.tsx` calls `useId()` **after** an early
  `return null` on `if (!enabled && !forceVisible)`. This violates the
  Rules of Hooks. In today's usage (`LobbyPage` gates mounting behind
  `personasFeatureEnabled()` so `enabled` is always true when mounted)
  this is latent — but any future caller that toggles the `featureEnabled`
  or `forceVisible` prop dynamically (tests, Storybook, a future "try
  before you buy" flow) will hit "Rendered more hooks than during the
  previous render" and crash the subtree. Move `const labelId = useId();`
  above the early return.

- **LOW** — `personaIds` on `AiOpponentsConfig` is `(string | undefined)[]`,
  but `CreateGameSchema` only accepts `z.array(z.enum([...]).optional())`.
  If `aiOpponents.count` is 3 and `personaIds` has length 2, the Zod
  schema accepts it; `GameSession` then reads `personaIds[2]`
  (undefined) and silently falls back to the ladder default for seat 3.
  Probably the intended behaviour (per the shared-types comment), but
  there is no length-check nor a unit test pinning this — worth at
  least documenting "length need not match `count`; missing entries
  inherit the ladder default" somewhere the server reviewers will see,
  or adding a Zod `refine(len == count)` once you're ready to treat
  mismatches as client bugs.

- **LOW** — `handleAIQuickPlay` does `option.personaId ?? selectedPersonaId`
  so the preset persona wins over the lobby picker. The picker state
  (`selectedPersonaId`) is persistent across quick-play clicks but is
  never reset if the user backs out and picks a preset with its own
  persona — the invisible-when-off picker can still be set to something
  and then silently ignored by an opinionated preset. UI-only concern,
  probably fine.

- **LOW** — The picker never renders an "unset" choice when the preset
  already carries a persona — there's no way for the player to _override_
  a preset persona back to "use ladder default" from the lobby. Since
  no preset currently carries a `personaId`, this is theoretical, but
  the UX contract is quietly narrower than the `includeNoOverride` prop
  suggests.

- **NIT** — `ALL_PERSONAS` and `PERSONA_COPY` in
  `src/client/config/aiQuickPlay.ts` duplicate the persona list _again_
  — the allow-list is now in four places: `AIServiceClient.ts`
  (`ALLOWED_PERSONA_IDS`), `aiQuickPlay.ts` (`ALL_PERSONAS`),
  `schemas.ts` (`z.enum([...])`), and the server's
  `coercePersonaId` inline tuple. Plus the Python
  `_ALLOWED_PERSONA_IDS`. Cross-layer consistency is locked in by a
  TS-side "mirrors the Python set" test, but client-schema-vs-client-
  config isn't. Centralise into one TS constant or add a contract test.

### 186a74bd5 — thread persona_id through AIEngine + AIServiceClient (C2 phase 2)

- **LOW** — `coercePersonaId` re-declares the allow-list inline
  (`['balanced','aggressive','territorial','defensive']`) instead of
  reusing the already-exported `ALLOWED_PERSONA_IDS` const from the
  same codebase (`src/server/services/AIServiceClient.ts`). Two sources
  of truth in the TS server alone; low cost to unify.

- **LOW** — `MoveRequest.persona_id` on the TS side is typed
  `PersonaId` (the 4-name union), so the TS compiler will reject
  anything else. Good. But `AIProfile.personaId` is typed as `string` —
  the comment calls out why (untrusted client input), and the coerce
  validates it. The asymmetry means static readers have to know the
  validation sits in `AIEngine.createAIFromProfile`; a comment on
  `AIProfile.personaId` pointing at the coerce helper would shorten
  that grep.

- **NIT** — `createAIFromProfile` logs `personaId: config.personaId ?? null`
  but the `AIConfig` field is `personaId?: PersonaId`, so the logged
  value is either a valid PersonaId or `null`. Consistent with the
  existing log style; no issue.

### 5aa0fe3be — /ai/move persona_id accept (C2 phase 1)

- **LOW** — `MoveResponse.persona_id=request.persona_id if persona_applied
else None` correctly falls back to `None` when the server-side flag
  is off (because `_resolve_persona_profile_id` returns None in that
  case, setting `persona_applied=False`). ✅ The question "MoveResponse.
  persona_id echoes the INTENDED persona — does it correctly fall back
  to null when the server-side flag is off?" is answered yes.

- **LOW** — `MoveRequest.validate_persona_id` runs _after_
  `_ALLOWED_PERSONA_IDS` is defined (module-level frozenset below the
  class). Python processes class body before the frozenset is bound,
  but `model_validator(mode="after")` decorators are evaluated at
  instance validation time, not class-definition time, so the lookup
  at `self.persona_id not in _ALLOWED_PERSONA_IDS` sees the final
  frozenset. Not a bug. Worth noting for future maintainers that the
  ordering is intentional.

- **LOW** — Validator error message uses `sorted(_ALLOWED_PERSONA_IDS)`
  which emits a list repr of sorted strings. Fine for logs; clients
  consuming the 422 body would need to parse the message — an
  enumerated-field structured error (pydantic 2 lets you use `Literal`
  on the field type) would be nicer. Non-blocking.

- **NIT** — 187-line test file is named as "41 new cases" in the commit
  message but `grep -c "def test_"` will find substantially fewer test
  functions. Cosmetic.

### c9a43020c — v5-heavy bootstrap in_channels alignment

- **LOW** — `_v5_heavy_in_channels(board_type)` uses
  `if board_type.startswith("hex") or board_type == "hexagonal"`: since
  "hexagonal" starts with "hex", the second clause is dead code.

- **LOW** — `TestBootstrapLoopAlignment` hardcodes the expected channel
  count **in the test body** (`16 * 4` / `14 * 4`) rather than reading
  `_ENCODER_METADATA` from `jsonl_to_npz.py`. The test's stated purpose
  — "bootstrap matches what jsonl_to_npz will emit" — is therefore not
  truly anchored to the jsonl_to_npz side. If someone changes
  jsonl_to_npz's metadata, this test would still pass. Strengthen by
  importing the real encoder metadata dict.

- **LOW** — Helper correctly centralises the channel math, but
  `num_players` is accepted by the caller signature and ignored by the
  helper. `_create_v5_heavy_model(board_type, num_players)` threads
  `num_players` through to `create_v5_heavy_model`; the NPZ side
  doesn't vary by num_players though, so the alignment invariant is
  truly just a function of board. OK.

### 5764c2656 — runtime v5-heavy in_channels from checkpoint

- **LOW** — The peek loads the entire checkpoint via `safe_load_checkpoint`
  just to read `conv1.weight.shape[1]`, then the main load path loads
  it again a few lines below to copy weights into the constructed model.
  For `v5-heavy-large` (~25-35M params) this is a noticeable one-time
  memory + IO cost at model init. Acceptable for a cold-start path, but
  worth a comment about "double load" or keeping the state-dict from
  the peek and reusing it downstream.

- **LOW** — Fallback to `in_channels=40` when `conv1.weight` is not at
  the expected key (e.g., wrapped as `module.conv1.weight` or renamed
  in a future refactor). The commit does call `_strip_module_prefix`
  so the `module.` case is handled, but any other rename (e.g.,
  `backbone.conv1.weight`) falls through to 40 and then crashes at
  strict load — the exact scenario the fix was trying to avoid. The
  warning log is good enough as a safety net for now, but the fallback
  silently choosing 40 could mask a real mismatch.

- **LOW** — No unit test added for this runtime loader path. Coverage
  gap given that this is a production runtime path (all callers go
  through the symlink `app/ai/_neural_net_legacy.py`). Would want at
  least a test that a 64ch checkpoint loads cleanly in the runtime
  flow.

- **NIT** — Exception `# noqa: BLE001` catches `Exception` but re-raises
  none; the comment "never block on metadata peek" justifies it, but
  pylint/pyright will probably still flag it. Fine.

### b714c63e7 — training probe model_version propagation regression test

- **LOW** — The test **does** exercise the real `_inference_probe` code
  path. Concretely: `patch.object(gumbel_module, "GumbelMCTSAI", FakeAI)`
  works because `_inference_probe` does `from app.ai.gumbel_mcts_ai
import GumbelMCTSAI` inside the function body (each call re-resolves),
  so the patch is visible. `FakeAI.__init__` captures the `cfg`, and
  the test asserts `cfg.nn_model_version` was set correctly. ✅ The
  review question "could the mocking bypass it?" is answered no.

- **LOW** — The test relies on `_weight_delta_check` producing a non-zero
  L2 so `result.critical` stays False and the inference probe actually
  runs. Random 4×4 weights are saved to `candidate.pth` and `best.pth`
  — the probability of two random normal tensors being identical is
  effectively zero, so the test is numerically deterministic in practice.
  **However**, the test does not pin the torch RNG (`torch.manual_seed`)
  before `torch.randn(4, 4)`. There's still a vanishingly small chance
  of ties-with-zero, especially if a future `weights_only=True` / dtype
  change rounds values. Pinning the seed would harden this.

- **LOW** — The comment "50x50 tensor" in the scrutiny ask doesn't
  match the actual test (which uses 4×4). Likely a review-ask typo,
  not a code issue — but 4×4 is fine.

- **LOW** — `_loss_convergence_check` is executed against
  `{"last_epoch_line": "Epoch 1, Train Loss: 0.5"}`. The test does
  not assert what this probe decides, only that the inference probe
  ran. If `_loss_convergence_check` ever starts setting `critical=True`
  for a loss of 0.5 at Epoch 1 (e.g., via a stricter convergence rule),
  the inference probe will be skipped and this test will begin to fail
  with a confusing "FakeAI was never constructed" error rather than
  a clear "loss convergence rule changed". Brittle coupling to the
  other probe's policy.

- **NIT** — `_UnsetSentinel` is imported-adjacent and used as a module-
  level sentinel; pytest `@pytest.mark.parametrize` would have been a
  cleaner way to express "the model_version kwarg is omitted"
  (`marks=pytest.mark.skip(...)` or a parametrised `model_version_kwargs`
  dict). Style preference only.

## Cross-cutting concerns

- **Cache-key invariants**: the single most dangerous finding in this
  batch is that `_ai_cache_key` doesn't cover every field in `AIConfig`
  that materially changes AI behaviour. Beyond
  `gumbel_simulation_budget` (this review), `allow_fresh_weights`,
  `personaId`, and any future knobs will have the same silent-reuse
  footgun. Worth a small refactor that derives the cache key from a
  declared "cache-relevant fields" list on `AIConfig` itself, with a
  test that fails when a new field is added without being classified.

- **Four-name allow-list duplication**: the persona name list is now
  redeclared in five places (Python `_ALLOWED_PERSONA_IDS`, TS
  `ALLOWED_PERSONA_IDS`, TS `AIEngine.coercePersonaId`, Zod enum, client
  `ALL_PERSONAS`). One cross-layer contract test exists
  (TS↔Python); a pure TS-internal drift is still possible.

- **Threadpool exhaustion on timeouts**: `asyncio.to_thread` + blocking
  AI code is a persistent latency-amplification risk. Not introduced
  here, but the C1 ensemble work multiplies the surface area N×.

- **Commit message accuracy**: several commit messages state test
  counts that don't match the actual file contents (e.g., "41 new
  cases" for a <20-test file). Not a code problem, but reviewers
  using the counts as a coverage proxy should verify locally.

## Verification you were not asked to do

- Would strongly recommend a follow-up that actually asserts the
  reduced-budget primary runs at the reduced budget — e.g., an
  integration test on `/ai/move` at D10 with ensemble flag on, a
  stub AI class that records the `config.gumbel_simulation_budget`
  at `select_move` time, and an assertion that all N constituents
  used the reduced budget. That would catch the cache-key bug above
  and any future regression of the same class.

- Consider adding a regression test for `_init_v5_heavy_model`'s
  checkpoint peek (5764c2656) against a saved 64ch state_dict to
  close the runtime-side coverage gap noted above.

- Consider seeding torch's RNG in `_run_probes_with_stub_ai` to
  remove the (astronomically small) risk of the two random weight
  tensors colliding and silently skipping the inference probe.

## Reviewer

Droid (Claude Opus 4.7) via factory.ai
