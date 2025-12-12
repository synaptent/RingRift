/**
 * Territory Decision Helpers - Shared territory-processing decision logic
 *
 * @module territoryDecisionHelpers
 *
 * This module provides territory-processing decision enumeration and application.
 * It works in conjunction with TerritoryAggregate and territoryProcessing:
 *
 * Canonical module hierarchy for territory logic:
 * - `./aggregates/TerritoryAggregate.ts` - Core territory detection and processing
 * - `./territoryProcessing.ts` - Region-level processing primitives
 * - `./territoryDetection.ts` - Disconnected region geometry
 * - This module - Decision enumeration and application helpers
 *
 * For new code, prefer importing core geometry from the aggregates when possible.
 * This module focuses on decision surfaces and GameState-level updates.
 */

import type { GameState, Move, Territory, RingStack } from '../types/game';
import { positionToString } from '../types/game';
import { computeNextMoveNumber } from './sharedDecisionHelpers';
import {
  eliminateFromStack,
  isStackEligibleForElimination,
  getRingsToEliminate,
  calculateCapHeight,
  type EliminationContext,
} from './aggregates/EliminationAggregate';
import {
  getProcessableTerritoryRegions,
  filterProcessableTerritoryRegions,
  applyTerritoryRegion,
  canProcessTerritoryRegion,
} from './territoryProcessing';
import type { TerritoryProcessingContext } from './territoryProcessing';

/**
 * Shared helpers for territory-processing decision enumeration and
 * application, including mandatory self-elimination semantics.
 *
 * This module is intended to centralise the remaining non-geometric parts of
 * the territory rules that are currently duplicated across:
 *
 * - Backend:
 *   - [`rules/territoryProcessing.processDisconnectedRegionsForCurrentPlayer`](src/server/game/rules/territoryProcessing.ts:1)
 *   - `GameEngine.applyDecisionMove` branches for `process_territory_region`
 *     and `eliminate_rings_from_stack`.
 * - Sandbox:
 *   - (legacy) `sandboxTerritoryEngine.processDisconnectedRegionsForCurrentPlayerEngine`
 *   - `ClientSandboxEngine.applyCanonicalMoveInternal` territory-processing
 *     branches.
 *
 * Geometry and core region application are already shared via:
 *
 * - [`territoryDetection.findDisconnectedRegions`](src/shared/engine/territoryDetection.ts:1)
 * - [`territoryProcessing.getProcessableTerritoryRegions`](src/shared/engine/territoryProcessing.ts:1)
 * - [`territoryProcessing.applyTerritoryRegion`](src/shared/engine/territoryProcessing.ts:1)
 *
 * The helpers defined here sit one level above those primitives and express
 * the canonical **decision surface** as `Move` instances:
 *
 * - `choose_territory_option` – choose which disconnected region to process.
 *   - Legacy alias: `process_territory_region` (accepted for replay only).
 *   - One move per processable region.
 * - `eliminate_rings_from_stack` – pay the mandatory self-elimination cost
 *   after processing a region (or other elimination-triggering effects).
 *   - One move per eligible stack under the acting player's control.
 *   - Eliminates the **entire cap** (all consecutive top rings of the
 *     controlling colour). For mixed-colour stacks, this exposes buried
 *     rings; for single-colour stacks with height > 1, this removes the
 *     stack entirely.
 *   - **Exception:** Recovery actions use buried ring extraction instead
 *     (handled separately in RecoveryAggregate).
 *
 * Implementations will be introduced in later P0 tasks when backend and
 * sandbox engines are refactored to call into these helpers. For P0 Task #21,
 * the functions are specified and documented but left as design-time stubs.
 */

/**
 * Options that control how territory-processing moves are enumerated.
 */
export interface TerritoryEnumerationOptions {
  /**
   * Whether to derive regions from `state.board.territories` (as populated
   * by host engines when they detect disconnections) or to re-run the shared
   * detector over `state.board` on demand.
   *
   * - 'use_board_cache' – trust `state.board.territories` and filter for
   *   disconnected regions controlled by the acting player.
   * - 'detect_now'      – invoke the shared detector +
   *   `getProcessableTerritoryRegions` and ignore any cached territories.
   *
   * Default: 'use_board_cache'.
   */
  detectionMode?: 'use_board_cache' | 'detect_now';

  /**
   * Optional test-only override list of disconnected regions. When provided,
   * {@link enumerateProcessTerritoryRegionMoves} will skip region detection
   * and instead filter this list via the canonical self-elimination
   * prerequisite. Production callers should omit this; it exists only for
   * rules-layer tests that want to decouple decision semantics from
   * detector geometry.
   */
  testOverrideRegions?: Territory[];
}

/**
 * Enumerate `choose_territory_option` decision moves for the specified
 * player in the current `GameState`.
 *
 * Semantics:
 *
 * - Only disconnected regions that are processable for `player` under the
 *   self-elimination prerequisite are surfaced. This prerequisite is defined
 *   by [`canProcessTerritoryRegion`](src/shared/engine/territoryProcessing.ts:1):
 *   the player must control at least one stack outside the region.
 * - Each returned {@link Move} has:
 *   - `type: 'choose_territory_option'`,
 *   - `player: player`,
 *   - `disconnectedRegions[0]` describing the region geometry, and
 *   - `to` set to a representative position inside the region for UI/debug.
 *
 * Error handling:
 *
 * - The helper is pure and does not mutate `state`.
 * - It may throw when structural invariants are violated (e.g. malformed
 *   `board.territories`) but is expected to be total for states produced by
 *   the canonical engines.
 */

export function enumerateProcessTerritoryRegionMoves(
  state: GameState,
  player: number,
  options?: TerritoryEnumerationOptions
): Move[] {
  const board = state.board;

  // For now both enumeration modes delegate to the canonical shared detector
  // + getProcessableTerritoryRegions so that backend GameEngine, RuleEngine,
  // and the sandbox all share identical region geometry and Q23 gating. Once
  // board.territories is populated consistently across hosts, callers may
  // opt into a stricter 'use_board_cache' interpretation without changing
  // this helper's external behaviour.
  const _mode = options?.detectionMode ?? 'use_board_cache';

  // Touch _mode to satisfy TypeScript unused-variable checks while preserving
  // current semantics. Future variants may branch on this flag to distinguish
  // cached vs on-demand detection, but today both paths share the same code.
  if (_mode === 'detect_now') {
    // No-op for now; detection behaviour is identical for both modes.
  }

  const ctx: TerritoryProcessingContext = { player };
  const overrideRegions = options?.testOverrideRegions;

  const processableRegions =
    overrideRegions && overrideRegions.length > 0
      ? filterProcessableTerritoryRegions(board, overrideRegions, ctx)
      : getProcessableTerritoryRegions(board, ctx);

  if (processableRegions.length === 0) {
    return [];
  }

  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  processableRegions.forEach((region, index) => {
    if (!region.spaces || region.spaces.length === 0) {
      return;
    }

    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : `region-${index}`;

    moves.push({
      id: `process-region-${index}-${regionKey}`,
      type: 'choose_territory_option',
      player,
      to: representative ?? { x: 0, y: 0 },
      disconnectedRegions: [region],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  });

  return moves;
}

/**
 * Result of applying a `process_territory_region` decision.
 *
 * This mirrors the responsibilities of the backend/sandbox territory engines
 * **excluding** the actual self-elimination step, which is modelled as a
 * separate `eliminate_rings_from_stack` decision.
 */
export interface TerritoryProcessApplicationOutcome {
  /**
   * Next GameState after applying the region-processing consequences,
   * including:
   *
   * - elimination of all stacks inside the region, credited to the acting
   *   player;
   * - collapsing of region spaces and their border markers to territory for
   *   the acting player;
   * - updates to `players[n].territorySpaces`, `players[n].eliminatedRings`,
   *   `board.eliminatedRings`, and `totalRingsEliminated`.
   */
  nextState: GameState;

  /**
   * Identifier and geometry of the processed region. Hosts may use this to
   * drive UI (e.g. highlighting) or to enforce "must eliminate from outside
   * the processed region" semantics during self-elimination.
   */
  processedRegionId: string;
  processedRegion: Territory;

  /**
   * True when the rules require a **mandatory self-elimination** step to
   * follow this region processing (compact rules §12.2 / FAQ Q23). In the
   * current rules this is always true when a region is processed, but this
   * flag is included to support potential future variants.
   */
  pendingSelfElimination: boolean;
}

/**
 * Apply a `process_territory_region` move to the given GameState.
 *
 * Responsibilities:
 *
 * - Delegate geometric effects and internal eliminations to the shared
 *   [`applyTerritoryRegion`](src/shared/engine/territoryProcessing.ts:1) helper.
 * - Project the resulting board-level deltas into GameState-level aggregates
 *   (per-player territory counts, eliminated-ring counts, totalRingsEliminated).
 * - Identify the processed region and indicate that a mandatory self-
 *   elimination decision is now required.
 *
 * This helper does **not**:
 *
 * - choose which stack to self-eliminate from; or
 * - apply the self-elimination itself.
 *
 * Those responsibilities are handled by the elimination helpers below.
 */
export function applyProcessTerritoryRegionDecision(
  state: GameState,
  move: Move
): TerritoryProcessApplicationOutcome {
  if (move.type !== 'choose_territory_option' && move.type !== 'process_territory_region') {
    throw new Error(
      `applyProcessTerritoryRegionDecision expected move.type === 'choose_territory_option' (or legacy 'process_territory_region'), got '${move.type}'`
    );
  }

  const player = move.player;

  // Prefer the concrete Territory attached to the Move when present, mirroring
  // how backend GameEngine and ClientSandboxEngine currently construct
  // choose_territory_option moves. This avoids re-running region detection in
  // the common case and keeps traces stable.
  let region: Territory | undefined =
    move.disconnectedRegions && move.disconnectedRegions.length > 0
      ? move.disconnectedRegions[0]
      : undefined;

  // Fallback: re-derive a processable region for this player from the current
  // board using the same helper as enumeration. This is primarily defensive
  // for callers that construct synthetic moves without disconnectedRegions.
  if (!region) {
    const candidates = getProcessableTerritoryRegions(state.board, { player });
    if (candidates.length === 1) {
      region = candidates[0];
    } else if (candidates.length > 1) {
      // Try to match by representative position (either move.to or the id
      // suffix 'process-region-{index}-{regionKey}').
      if (move.to) {
        const repKey = positionToString(move.to);
        region = candidates.find(
          (r) => r.spaces.length > 0 && positionToString(r.spaces[0]) === repKey
        );
      }

      if (!region && move.id && move.id.startsWith('process-region-')) {
        const tail = move.id.slice('process-region-'.length);
        const dashIndex = tail.indexOf('-');
        const regionKeyStr = dashIndex >= 0 ? tail.slice(dashIndex + 1) : tail;
        region = candidates.find(
          (r) => r.spaces.length > 0 && positionToString(r.spaces[0]) === regionKeyStr
        );
      }
    }
  }

  if (!region) {
    // No valid region can be identified for this decision; treat as a
    // no-op at the state level while still returning a well-formed outcome.
    return {
      nextState: state,
      processedRegionId: move.id,
      processedRegion: {
        spaces: [],
        controllingPlayer: player,
        isDisconnected: true,
      },
      pendingSelfElimination: false,
    };
  }

  // Enforce the self-elimination prerequisite defensively at application
  // time as well as during enumeration: the acting player must control at
  // least one stack/cap outside the chosen region (FAQ Q23 / §12.2). When
  // this prerequisite is not satisfied, the move is treated as a no-op.
  if (!canProcessTerritoryRegion(state.board, region, { player })) {
    const representative = region.spaces[0];
    const regionKey = representative ? positionToString(representative) : 'region-0';
    const processedRegionId =
      move.id && move.id.length > 0 ? move.id : `process-region-0-${regionKey}`;

    return {
      nextState: state,
      processedRegionId,
      processedRegion: region,
      pendingSelfElimination: false,
    };
  }

  const ctx: TerritoryProcessingContext = { player };
  const outcome = applyTerritoryRegion(state.board, region, ctx);

  // Project board-level deltas into GameState-level aggregates.
  const territoryGain = outcome.territoryGainedByPlayer[player] ?? 0;
  const internalElims = outcome.eliminatedRingsByPlayer[player] ?? 0;

  let nextPlayers = state.players.map((p) => ({ ...p }));

  if (territoryGain > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === player ? { ...p, territorySpaces: p.territorySpaces + territoryGain } : p
    );
  }

  if (internalElims > 0) {
    nextPlayers = nextPlayers.map((p) =>
      p.playerNumber === player ? { ...p, eliminatedRings: p.eliminatedRings + internalElims } : p
    );
  }

  const nextState: GameState = {
    ...state,
    board: outcome.board,
    players: nextPlayers,
    totalRingsEliminated: state.totalRingsEliminated + internalElims,
  };

  const representative = region.spaces[0];
  const regionKey = representative ? positionToString(representative) : 'region-0';
  const processedRegionId =
    move.id && move.id.length > 0 ? move.id : `process-region-0-${regionKey}`;

  return {
    nextState,
    processedRegionId,
    processedRegion: region,
    // Under current rules, processing any disconnected region always incurs a
    // mandatory self-elimination debt for the acting player (FAQ Q23 / §12.2).
    pendingSelfElimination: true,
  };
}

/**
 * Enumerate `eliminate_rings_from_stack` decision moves for the specified
 * player in the territory-processing context.
 *
 * Primary use cases:
 *
 * - Mandatory self-elimination after processing a disconnected territory
 *   region. This requires eliminating the **entire cap** (all consecutive
 *   top rings of the controlling colour). For mixed-colour stacks, this
 *   exposes buried rings of other colours; for single-colour stacks with
 *   height > 1, this eliminates all rings (removing the stack entirely).
 * - Ring-elimination rewards earned from line-collapses (Option 1 on long
 *   lines), when engines choose to express those rewards explicitly as
 *   `eliminate_rings_from_stack` moves rather than as implicit effects.
 *
 * **Exception:** Recovery actions use buried ring extraction (one ring)
 * instead of entire cap elimination; that logic is handled separately in
 * RecoveryAggregate.
 *
 * Semantics:
 *
 * - For each eligible stack controlled by `player`, produce one Move with:
 *   - `type: 'eliminate_rings_from_stack'`,
 *   - `player: player`,
 *   - `to` set to the stack position, and
 *   - `eliminationFromStack` snapshotting the pre-elimination geometry
 *     (position, capHeight, totalHeight) for parity/debugging.
 * - The exact notion of "eligible" stacks (e.g. whether stacks inside the
 *   processed region are allowed) is parameterised via {@link scope}; see
 *   the design doc for current assumptions and open questions.
 *
 * When no eligible stacks exist, engines must instead eliminate rings from
 * the acting player's hand automatically; that behaviour is host-specific
 * and not modelled as a separate Move.
 */
export interface TerritoryEliminationScope {
  /**
   * Optional identifier of the region that was just processed. When
   * provided, helpers may use this to enforce "must eliminate from outside
   * the processed region" semantics; when omitted, all stacks controlled by
   * the player are considered.
   *
   * **NOTE:** At the time of writing, the existing backend and sandbox
   * engines do not consistently distinguish inside-vs-outside behaviour for
   * self-elimination. This is called out as an open question in
   * [`P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md`](P0_TASK_21_SHARED_HELPER_MODULES_DESIGN.md:1).
   */
  processedRegionId?: string;

  /**
   * Elimination context that controls eligibility and elimination semantics.
   * - 'territory': Only eligible cap targets (multicolor or height > 1) can be
   *   selected; entire cap is eliminated. Per RR-CANON-R145.
   * - 'line': Any controlled stack (including height-1 standalone rings) can be
   *   selected; only ONE ring is eliminated. Per RR-CANON-R122.
   * - 'forced' or undefined: Any controlled stack can be selected; entire cap
   *   is eliminated. Per RR-CANON-R100.
   */
  eliminationContext?: 'line' | 'territory' | 'forced';
}

export function enumerateTerritoryEliminationMoves(
  state: GameState,
  player: number,
  scope?: TerritoryEliminationScope
): Move[] {
  const board = state.board;

  // Touch scope to satisfy TypeScript unused-parameter checks while keeping
  // behaviour identical. Current engines do not branch on processedRegionId;
  // this placeholder makes the parameter observable for the compiler only.
  if (scope && scope.processedRegionId === 'noop') {
    // No-op placeholder; elimination semantics are currently global.
  }

  // In the dedicated territory_processing phase, do not surface explicit
  // self-elimination decisions while any disconnected region remains
  // processable for this player. This preserves the "region-first, then
  // self-elimination" ordering from §12.2 / FAQ Q23 and mirrors the
  // backend RuleEngine / GameEngine gating.
  if (state.currentPhase === 'territory_processing') {
    const remainingRegions = getProcessableTerritoryRegions(board, { player });
    if (remainingRegions.length > 0) {
      return [];
    }
  }

  const stacks: { key: string; stack: RingStack }[] = [];
  for (const [key, stack] of board.stacks.entries()) {
    if (stack.controllingPlayer === player) {
      stacks.push({ key, stack });
    }
  }

  if (stacks.length === 0) {
    return [];
  }

  const nextMoveNumber = computeNextMoveNumber(state);
  const moves: Move[] = [];

  // Determine elimination context - delegates to canonical EliminationAggregate
  const eliminationContext: EliminationContext = scope?.eliminationContext ?? 'territory';

  for (const { key, stack } of stacks) {
    const capHeight = calculateCapHeight(stack.rings);
    if (capHeight <= 0) {
      continue;
    }

    // Delegate eligibility check to canonical EliminationAggregate
    // RR-CANON-R082 / R100 / R122 / R145: Rules handled by isStackEligibleForElimination
    const eligibility = isStackEligibleForElimination(stack, eliminationContext, player);
    if (!eligibility.eligible) {
      continue;
    }

    // Delegate ring count calculation to canonical EliminationAggregate
    const ringsToEliminate = getRingsToEliminate(stack, eliminationContext);

    // NOTE: scope.processedRegionId is reserved for future variants where
    // eliminations may be constrained to outside/inside a particular region.
    // The current engines do not distinguish on that axis, so we ignore it.
    moves.push({
      id: `eliminate-${key}`,
      type: 'eliminate_rings_from_stack',
      player,
      to: stack.position,
      eliminatedRings: [{ player, count: ringsToEliminate }],
      eliminationFromStack: {
        position: stack.position,
        capHeight,
        totalHeight: stack.stackHeight,
      },
      // Tag the context so applyEliminateRingsFromStackDecision knows how many rings to eliminate
      eliminationContext,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: nextMoveNumber,
    } as Move);
  }

  return moves;
}

/**
 * Result of applying an `eliminate_rings_from_stack` decision.
 *
 * This is intentionally minimal: the helper is responsible only for
 * modifying the chosen stack and updating elimination counters. Phase/turn
 * progression (e.g. whether further eliminations are required or whether
 * bookkeeping proceeds to the next phase) remains a host/turn-logic concern.
 */
export interface EliminateRingsFromStackOutcome {
  /** Next GameState after eliminating rings from the chosen stack. */
  nextState: GameState;
}

/**
 * Apply an `eliminate_rings_from_stack` move to the given GameState.
 *
 * Responsibilities:
 *
 * - Remove the cap (consecutive top rings of the controlling colour) from
 *   the chosen stack, mirroring the semantics of the backend and sandbox
 *   self-elimination flows.
 * - Update `board.eliminatedRings`, `players[n].eliminatedRings`, and
 *   `totalRingsEliminated` accordingly.
 * - Remove the stack entirely when all rings are eliminated.
 *
 * This helper intentionally does **not** attempt to infer why the
 * elimination is occurring (territory self-elimination vs line reward vs
 * forced elimination for a blocked player). Hosts remain responsible for
 * tracking that context and for deciding which phase/turn transitions to
 * perform after elimination.
 */
export function applyEliminateRingsFromStackDecision(
  state: GameState,
  move: Move
): EliminateRingsFromStackOutcome {
  if (move.type !== 'eliminate_rings_from_stack') {
    throw new Error(
      `applyEliminateRingsFromStackDecision expected move.type === 'eliminate_rings_from_stack', got '${move.type}'`
    );
  }

  if (!move.to) {
    // No target position – treat as no-op for robustness.
    return { nextState: state };
  }

  const player = move.player;

  // Determine elimination context from move - defaults to 'territory' for backwards compat
  const eliminationContext: EliminationContext =
    (move.eliminationContext as EliminationContext) ?? 'territory';

  // Delegate to canonical EliminationAggregate
  const eliminationResult = eliminateFromStack({
    context: eliminationContext,
    player,
    stackPosition: move.to,
    board: state.board,
  });

  if (!eliminationResult.success) {
    // Invalid or stale target; leave state unchanged defensively.
    return { nextState: state };
  }

  // Update players array (EliminationAggregate only updates board)
  const nextPlayers = state.players.map((p) =>
    p.playerNumber === player
      ? {
          ...p,
          eliminatedRings: p.eliminatedRings + eliminationResult.ringsEliminated,
        }
      : p
  );

  const nextState: GameState = {
    ...state,
    board: eliminationResult.updatedBoard,
    players: nextPlayers,
    totalRingsEliminated: state.totalRingsEliminated + eliminationResult.ringsEliminated,
  };

  return { nextState };
}
