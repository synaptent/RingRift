import type {
  GameState,
  Move,
  Player,
  BoardType,
  TimeControl,
  Position,
  RingStack,
  Territory,
  LineInfo,
  GameHistoryEntry,
} from '../../shared/engine';
import {
  BOARD_CONFIGS,
  getEffectiveLineLengthThreshold,
  positionToString,
  calculateCapHeight,
  computeRingEliminationVictoryThreshold,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
  filterProcessableTerritoryRegions,
  replaceMapContents,
  replaceArrayContents,
  applyTerritoryRegion,
  canProcessTerritoryRegion,
  enumerateProcessTerritoryRegionMoves,
  applyForcedEliminationForPlayer,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  findLinesForPlayer,
  // Capture aggregate helpers (global enumeration)
  enumerateAllCaptureMoves as enumerateAllCaptureMovesAggregate,
  // Movement/placement aggregation helpers
  enumerateSimpleMovesForPlayer,
  enumeratePlacementPositions,
  // LPS helpers
  hasAnyRealAction,
  createLpsTrackingState,
  type LpsTrackingState,
  // Type guards for move narrowing
  isCaptureMove,
  // Swap sides (pie rule) helpers
  shouldOfferSwapSides,
  validateSwapSidesMove,
  applySwapSidesIdentitySwap,
} from '../../shared/engine';
import type {
  GameResult,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  PlayerChoiceResponseFor,
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import { ClockManager } from './managers/ClockManager';
import {
  getChainCaptureContinuationInfo,
  ChainCaptureState,
  updateChainCaptureStateAfterCapture,
} from '../../shared/engine/aggregates/CaptureAggregate';
import {
  PerTurnState,
  advanceGameForCurrentPlayer,
  TurnEngineDeps,
  TurnEngineHooks,
} from './turn/TurnEngine';
import { orchestratorRollout } from '../services/OrchestratorRolloutService';
import { getMetricsService } from '../services/MetricsService';
import {
  TurnEngineAdapter,
  StateAccessor,
  DecisionHandler,
  EventEmitter as AdapterEventEmitter,
  AdapterMoveResult,
} from './turn/TurnEngineAdapter';
import { flagEnabled, debugLog } from '../../shared/utils/envFlags';
import { logger } from '../utils/logger';

/**
 * Backend `GameEngine` host over the shared rules engine.
 *
 * This class is responsible for orchestration, persistence, and
 * interaction/wiring only; **it must not introduce new rules semantics**.
 * All move legality, phase transitions, capture/territory/line/victory
 * semantics should flow through the shared engine helpers/aggregates under
 * `src/shared/engine/**` as documented in
 * `docs/rules/RULES_ENGINE_SURFACE_AUDIT.md` (§0 Rules Entry Surfaces).
 */

/**
 * Internal state for enforcing mandatory chain captures during the capture phase.
 *
 * This is intentionally kept out of the wire-level GameState so we can evolve
 * the representation without breaking clients. It is roughly modeled after the
 * Rust engine's `ChainCaptureState` and is used only inside GameEngine.
 *
 * The concrete shape is shared with the CaptureAggregate module; we keep the
 * Ts* aliases here to preserve existing semantics and comments while
 * centralising the implementation.
 */
type TsChainCaptureState = ChainCaptureState;

// Deterministic identifier helper for moves and choice payloads.
// This deliberately avoids any RNG so that core engine behaviour
// remains fully deterministic (RR‑CANON R190).
function generateUUID(...parts: Array<string | number | undefined>): string {
  return parts
    .filter((part) => part !== undefined)
    .map((part) => String(part))
    .join('|');
}

export class GameEngine {
  private gameState: GameState;
  private boardManager: BoardManager;
  private ruleEngine: RuleEngine;
  private clockManager: ClockManager;
  private interactionManager: PlayerInteractionManager | undefined;
  private debugCheckpointHook: ((label: string, state: GameState) => void) | undefined;
  /**
   * Per-turn placement state: when a ring placement occurs, we track that
   * fact and remember which stack must be moved this turn. This mirrors
   * the sandbox engine's per-turn fields but remains internal to the
   * backend engine.
   */
  private hasPlacedThisTurn: boolean = false;
  private mustMoveFromStackKey: string | undefined;
  /**
   * Internal chain capture state, used to enforce mandatory continuation of
   * captures once started. When defined, only additional overtaking captures
   * from `currentPosition` are legal until no options remain.
   */
  private chainCaptureState: TsChainCaptureState | undefined;
  /**
   * Internal flag used only in move-driven decision phases to indicate
   * that the current player has processed at least one disconnected
   * territory region in the current territory_processing cycle and
   * therefore must perform mandatory self-elimination via an explicit
   * eliminate_rings_from_stack Move. This prevents the backend from
   * surfacing spurious elimination decisions (and auto-applying them
   * in stepAutomaticPhasesForTesting) in territory_processing states
   * where no region has ever been processed.
   */
  private pendingTerritorySelfElimination: boolean = false;
  /**
   * Internal flag used only in move-driven decision phases to indicate
   * that the current player has processed an exact-length or overlength
   * line with Option 1 reward and must now perform mandatory ring
   * elimination via an explicit eliminate_rings_from_stack Move. This
   * prevents the backend from bypassing the unified Move model for line
   * reward eliminations.
   */
  private pendingLineRewardElimination: boolean = false;

  /**
   * Internal helper flag for the 2-player pie rule (swap_sides).
   * We rely primarily on moveHistory shape for gating, but this flag
   * can be used for additional future diagnostics if needed.
   */
  private _swapSidesApplied: boolean = false;

  /**
   * Last-Player-Standing (R172) tracking state. Mirrors sandbox LPS tracking
   * so backend hosts can evaluate LPS victories (including recovery as a
   * real action).
   */
  private _lpsState: LpsTrackingState;

  /** Returns true if the swap sides (pie rule) has been applied in this game. */
  public get swapSidesApplied(): boolean {
    return this._swapSidesApplied;
  }

  /**
   * Returns a lightweight summary of LPS tracking state for client display.
   * Per RR-CANON-R172, LPS victory requires 3 consecutive rounds where only
   * one player has real actions available.
   */
  public getLpsTrackingSummary(): GameState['lpsTracking'] {
    return {
      roundIndex: this._lpsState.roundIndex,
      consecutiveExclusiveRounds: this._lpsState.consecutiveExclusiveRounds,
      consecutiveExclusivePlayer: this._lpsState.consecutiveExclusivePlayer,
    };
  }

  /**
   * When true, the engine operates in replay mode:
   * - Auto-processing of single-option decisions is disabled.
   * - The decision loop is broken immediately when a decision is required.
   * - This allows the replay driver to supply explicit decision moves from
   *   a recording.
   */
  private readonly replayMode: boolean;

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true,
    interactionManager?: PlayerInteractionManager,
    rngSeed?: number,
    rulesOptions?: GameState['rulesOptions'],
    replayMode: boolean = false
  ) {
    this.replayMode = replayMode;
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    this.interactionManager = interactionManager;

    // Initialize clock manager with forfeit callback
    this.clockManager = new ClockManager({
      onTimeout: (playerNumber) => this.forfeitGame(playerNumber.toString()),
    });

    const boardConfig = BOARD_CONFIGS[boardType];
    const effectiveRingsPerPlayer = rulesOptions?.ringsPerPlayer ?? boardConfig.ringsPerPlayer;
    const effectiveLpsRoundsRequired = rulesOptions?.lpsRoundsRequired ?? 2;

    this.gameState = {
      id: gameId,
      boardType,
      ...(typeof rngSeed === 'number' ? { rngSeed } : {}),
      board: this.boardManager.createBoard(),
      players: players.map((p, index) => ({
        ...p,
        playerNumber: index + 1,
        timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
        // Preserve isReady from input (GameSession sets it), only default AI to ready if not set
        isReady: p.isReady ?? p.type === 'ai',
      })),
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl,
      spectators: [],
      gameStatus: 'waiting',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated,
      // Optional per-game rules configuration (e.g., swap rule / pie rule).
      // When omitted, hosts should treat this as "use defaults".
      ...(rulesOptions ? { rulesOptions } : {}),
      maxPlayers: players.length,
      totalRingsInPlay: effectiveRingsPerPlayer * players.length,
      totalRingsEliminated: 0,
      // Per RR-CANON-R061: victoryThreshold = round((2/3) × ownStartingRings + (1/3) × opponentsCombinedStartingRings)
      // Simplified: round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
      victoryThreshold: computeRingEliminationVictoryThreshold(
        effectiveRingsPerPlayer,
        players.length
      ),
      territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1,
      lpsRoundsRequired: effectiveLpsRoundsRequired,
    };

    // Internal no-op hook to keep selected helpers referenced so that
    // ts-node/TypeScript with noUnusedLocals can compile the server in
    // dev without stripping them. This has no behavioural effect.
    this._debugUseInternalHelpers();

    // Initialise LPS tracking state (R172) mirroring sandbox behaviour.
    this._lpsState = createLpsTrackingState();
  }

  /**
   * Test-only helper: register a debug checkpoint hook used by parity and
   * diagnostic harnesses to capture GameState snapshots at key points inside
   * makeMove / decision processing. When unset, all debugCheckpoint calls
   * are no-ops and have zero runtime cost.
   */
  public setDebugCheckpointHook(
    hook: ((label: string, state: GameState) => void) | undefined
  ): void {
    this.debugCheckpointHook = hook;
  }

  /**
   * Apply the pie-rule style colour/seat swap for a 2-player game.
   *
   * Semantics:
   * - Only legal once, for Player 2, at the start of their first
   *   interactive turn after Player 1 has completed a full turn.
   * - Board geometry is unchanged; we simply swap which user occupies
   *   playerNumber 1 vs 2 (rings, territory, clocks, etc.).
   * - Turn order and currentPhase are preserved; after swap, it is still
   *   Player 2's turn in the same phase.
   */
  private async applySwapSidesMove(
    playerNumber: number
  ): Promise<{ success: boolean; error?: string; gameState?: GameState; gameResult?: GameResult }> {
    // Validate via shared helper (handles all eligibility checks)
    const validation = validateSwapSidesMove(this.gameState, playerNumber);
    if (!validation.valid) {
      return {
        success: false,
        error: validation.reason,
        gameState: this.getGameState(),
      };
    }

    // Capture a full snapshot for history before we mutate players.
    const beforeStateForHistory = this.getGameState();

    // Apply identity swap via shared helper. This swaps id/username/type/rating
    // between seats 1 and 2 while keeping seat numbers and board state stable.
    const swappedPlayers = applySwapSidesIdentitySwap(this.gameState.players);

    this.gameState = {
      ...this.gameState,
      players: swappedPlayers,
    };

    this._swapSidesApplied = true;

    // Record a canonical Move for history/traces. Board geometry and
    // phase/player are unchanged; we use a sentinel coordinate for `to`.
    const fullMove: Move = {
      id: generateUUID('swap_sides', this.gameState.id, this.gameState.moveHistory.length + 1),
      type: 'swap_sides',
      player: playerNumber,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1,
    } as Move;

    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = fullMove.timestamp;

    // Swap has no geometric effect, but we still append a history entry so
    // that traces record the seat/colour transition.
    this.appendHistoryEntry(beforeStateForHistory, fullMove);

    return {
      success: true,
      gameState: this.getGameState(),
    };
  }

  /**
   * @deprecated Legacy path removed - this is now a no-op.
   * Orchestrator always uses move-driven phases.
   *
   * Ownership / deprecation:
   * - Still invoked by a number of backend-vs-sandbox parity and scenario suites
   *   (for example: tests/helpers/orchestratorTestUtils.ts, TerritoryDecisions.*,
   *   RefactoredEngineParity, Backend_vs_Sandbox.* parity tests).
   * - New entry points should configure move-driven phases via the shared
   *   TurnEngineAdapter/orchestrator instead of calling this directly.
   */
  public enableMoveDrivenDecisionPhases(): void {
    // No-op: orchestrator is now the only path
  }

  /**
   * @deprecated Legacy path removed - this is now a no-op.
   * Orchestrator adapter is always enabled.
   *
   * Ownership / deprecation:
   * - Used in older test utilities (for example orchestratorTestUtils and
   *   GameEngine.utilityMethods) to make the adapter behaviour explicit.
   * - Orchestrator-backed hosts no longer need to toggle this; new code
   *   should assume the adapter is always enabled.
   */
  public enableOrchestratorAdapter(): void {
    // No-op: orchestrator is now the only path
  }

  /**
   * @deprecated Legacy path removed - this is now a no-op.
   * The legacy GameEngine turn processing pipeline has been removed.
   *
   * Ownership / deprecation:
   * - Exercised only by legacy/compat tests (see GameEngine.utilityMethods).
   * - Kept as a no-op shim until those tests are fully migrated away from
   *   explicit adapter toggling.
   */
  public disableOrchestratorAdapter(): void {
    // No-op: orchestrator is now the only path
  }

  /**
   * @deprecated Legacy path removed - always returns true.
   *
   * Ownership / deprecation:
   * - Verified by GameEngine.utilityMethods tests as part of the
   *   orchestrator-migration cleanup.
   * - Callers that need to know whether orchestrator is active should instead
   *   rely on configuration, not this legacy getter.
   */
  public isOrchestratorAdapterEnabled(): boolean {
    return true;
  }

  /**
   * Create a TurnEngineAdapter instance wired to this GameEngine's state.
   * This is used when orchestrator delegation is enabled.
   */
  private createAdapterForCurrentGame(): TurnEngineAdapter {
    // StateAccessor implementation that reads/writes this.gameState
    const stateAccessor: StateAccessor = {
      getGameState: () => this.gameState,
      updateGameState: (newState: GameState) => {
        // Preserve references to existing board Maps so tests that cache
        // engineAny.gameState before making moves continue to see updates.
        // The orchestrator returns immutable state updates, but the legacy
        // GameEngine tests rely on mutation semantics.
        const existingBoard = this.gameState.board;
        const incomingBoard = newState.board;

        // IMPORTANT: Some orchestrator transitions (e.g. skip_placement) are
        // pure phase changes that *reuse* the existing BoardState instance
        // rather than allocating a fresh board. In those cases incomingBoard
        // and existingBoard are the same object. If we clear the Maps on
        // existingBoard and then iterate incomingBoard.* we would be iterating
        // the very Maps we just cleared, effectively zeroing out all stacks,
        // markers, and collapsed spaces and violating the S-invariant.
        //
        // When the BoardState instance is shared, the orchestrator has already
        // applied any intended board mutations in-place, so we can safely keep
        // the existing board as-is and just update scalar fields on gameState.
        if (incomingBoard === existingBoard) {
          this.gameState = {
            ...newState,
            board: existingBoard,
          };
          return;
        }

        // Update stacks and markers Maps in-place using the orchestrator's board.
        replaceMapContents(existingBoard.stacks, incomingBoard.stacks);
        replaceMapContents(existingBoard.markers, incomingBoard.markers);

        // Update collapsedSpaces Map, preserving monotonic territory semantics.
        //
        // Collapsed spaces represent territory that should never "un-collapse"
        // outside of explicit territory-processing decisions. Invariant-critical
        // tooling (including the orchestrator soak harness) treats
        // board.collapsedSpaces as a monotone component of S:
        //
        //   S = markers + collapsedSpaces + eliminatedRings
        //
        // To defend against any transient drift between hosts and the shared
        // orchestrator, we apply the orchestrator's view *on top of* the
        // existing map rather than replacing it wholesale. This guarantees that
        // previously-collapsed spaces remain collapsed even if a buggy update
        // omits them.
        const previousCollapsed = new Map(existingBoard.collapsedSpaces);

        existingBoard.collapsedSpaces.clear();
        for (const [key, collapsed] of incomingBoard.collapsedSpaces) {
          existingBoard.collapsedSpaces.set(key, collapsed);
        }
        // Re-add any previously-collapsed keys that the new state does not
        // mention, preserving monotonicity.
        for (const [key, owner] of previousCollapsed) {
          if (!existingBoard.collapsedSpaces.has(key)) {
            existingBoard.collapsedSpaces.set(key, owner);
          }
        }

        // Update territories Map and formedLines array in-place
        replaceMapContents(existingBoard.territories, incomingBoard.territories);
        replaceArrayContents(existingBoard.formedLines, incomingBoard.formedLines);

        // Update eliminatedRings object, preserving monotonic elimination
        // accounting between host and orchestrator. The Python rules engine,
        // shared TS core, and soak harness all assume that rings, once
        // eliminated, are never "returned to play" and that
        // board.eliminatedRings[player] is non-decreasing.
        const previousElims = { ...existingBoard.eliminatedRings };

        for (const key of Object.keys(existingBoard.eliminatedRings)) {
          delete existingBoard.eliminatedRings[key as unknown as number];
        }
        for (const [key, value] of Object.entries(incomingBoard.eliminatedRings)) {
          existingBoard.eliminatedRings[key as unknown as number] = value;
        }
        // Enforce per-player monotonicity: never allow a host/orchestrator
        // update to reduce the recorded elimination count for any player.
        for (const [key, prevValue] of Object.entries(previousElims)) {
          const numKey = key as unknown as number;
          const current = existingBoard.eliminatedRings[numKey] ?? 0;
          if (prevValue > current) {
            existingBoard.eliminatedRings[numKey] = prevValue;
          }
        }

        // Update scalar board fields
        existingBoard.size = incomingBoard.size;

        // Update the gameState while preserving the board reference
        this.gameState = {
          ...newState,
          board: existingBoard,
        };
      },
      getPlayerInfo: (playerNumber: number) => {
        const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
        if (!player) {
          return undefined;
        }
        return { type: player.type };
      },
    };

    // DecisionHandler that auto-resolves certain orchestrator decisions when
    // no interactive PlayerInteractionManager is wired. This keeps adapter-
    // driven hosts (including the orchestrator soak harness) from treating
    // elimination/ordering decisions as HOST_REJECTED_MOVE violations while
    // still surfacing unexpected decision types as hard errors.
    const decisionHandler: DecisionHandler = {
      requestDecision: async (decision) => {
        const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

        // When a PlayerInteractionManager is wired and the decision is an
        // elimination_target, route it through the existing ring_elimination
        // PlayerChoice path so humans (and AI via the handler) can choose the
        // target stack explicitly. This covers both territory self-elimination
        // and forced-elimination cases emitted by the orchestrator.
        if (decision.type === 'elimination_target' && this.interactionManager) {
          const interaction = this.requireInteractionManager();

          // Build a RingEliminationChoice whose options map 1:1 onto the
          // orchestrator's eliminate_rings_from_stack Moves via moveId.
          const eliminationMoves = decision.options.filter(
            (m) => m.type === 'eliminate_rings_from_stack' && m.to
          );

          if (eliminationMoves.length === 0) {
            // Fall back to the defensive auto-resolve path below.
          } else {
            // Derive eliminationContext from first move
            const firstMove = eliminationMoves[0];
            const eliminationContext = firstMove?.eliminationContext || 'territory';

            // Generate context-specific prompt
            let prompt: string;
            if (eliminationContext === 'line') {
              prompt =
                'Line reward cost: You must eliminate ONE ring from the top of any stack you control.';
            } else if (eliminationContext === 'recovery') {
              prompt =
                'Recovery territory cost: You must extract ONE buried ring from a stack outside the region.';
            } else if (eliminationContext === 'forced') {
              prompt =
                'Forced elimination: You must eliminate your ENTIRE CAP from a controlled stack.';
            } else {
              prompt =
                'Territory cost: You must eliminate your ENTIRE CAP from an eligible stack outside the region.';
            }

            const choice: RingEliminationChoice = {
              id: generateUUID('ring_elimination', this.gameState.id, decision.player, Date.now()),
              gameId: this.gameState.id,
              playerNumber: decision.player,
              type: 'ring_elimination',
              eliminationContext,
              prompt,
              options: eliminationMoves.map((move) => {
                const pos = move.to as Position;
                const stack = this.boardManager.getStack(pos, this.gameState.board);
                const capHeight =
                  (move.eliminationFromStack && move.eliminationFromStack.capHeight) ||
                  (stack ? stack.capHeight : 1);
                const totalHeight =
                  (move.eliminationFromStack && move.eliminationFromStack.totalHeight) ||
                  (stack ? stack.stackHeight : capHeight || 1);

                // Per RR-CANON-R122 / RR-CANON-R114: line + recovery cost 1 ring; territory/forced costs entire cap.
                const ringsToEliminate =
                  eliminationContext === 'line' || eliminationContext === 'recovery'
                    ? 1
                    : capHeight;

                return {
                  stackPosition: pos,
                  capHeight,
                  totalHeight,
                  ringsToEliminate,
                  moveId: move.id,
                };
              }),
            };

            const response: PlayerChoiceResponseFor<RingEliminationChoice> =
              await interaction.requestChoice(choice);
            const selectedMoveId = response.selectedOption.moveId;

            const chosen =
              eliminationMoves.find((m) => m.id === selectedMoveId) ??
              eliminationMoves.find((m) => {
                const to = m.to as Position | undefined;
                return (
                  to &&
                  to.x === response.selectedOption.stackPosition.x &&
                  to.y === response.selectedOption.stackPosition.y
                );
              }) ??
              eliminationMoves[0];

            debugLog(
              TRACE_DEBUG_ENABLED,
              '[GameEngine.DecisionHandler] resolved elimination_target via choice',
              {
                player: decision.player,
                optionCount: eliminationMoves.length,
                selectedMoveId,
              }
            );

            return chosen;
          }
        }

        // RR-FIX-2026-01-15: Route region_order decisions through the interaction
        // manager so human players can explicitly choose which territory to claim.
        // Without this, territory decisions were auto-selected even during chain
        // captures, causing ring_elimination to surface at the wrong time.
        if (decision.type === 'region_order' && this.interactionManager) {
          const interaction = this.requireInteractionManager();

          // Extract choose_territory_option moves (skip skip_territory_processing for now)
          const territoryMoves = decision.options.filter(
            (m) => m.type === 'choose_territory_option' && m.disconnectedRegions?.length
          );

          if (territoryMoves.length > 0) {
            const choice: RegionOrderChoice = {
              id: generateUUID('region_order', this.gameState.id, decision.player, Date.now()),
              gameId: this.gameState.id,
              playerNumber: decision.player,
              type: 'region_order',
              prompt:
                territoryMoves.length > 1
                  ? `Choose which territory region to claim (${territoryMoves.length} available). Each region costs your entire cap from a stack outside.`
                  : 'Claim this territory region? It will cost your entire cap from a stack outside.',
              options: territoryMoves.map((move, index) => {
                const region = move.disconnectedRegions?.[0];
                const representative = move.to as Position;
                const option: RegionOrderChoice['options'][number] = {
                  regionId: move.id || `region-${index}`,
                  size: region?.spaces?.length ?? 0,
                  representativePosition: representative,
                  moveId: move.id,
                };
                // Only include spaces if defined (exactOptionalPropertyTypes)
                if (region?.spaces) {
                  option.spaces = region.spaces;
                }
                return option;
              }),
            };

            const response: PlayerChoiceResponseFor<RegionOrderChoice> =
              await interaction.requestChoice(choice);
            const selectedMoveId = response.selectedOption.moveId;

            const chosen =
              territoryMoves.find((m) => m.id === selectedMoveId) ??
              territoryMoves.find((m) => {
                const to = m.to as Position | undefined;
                return (
                  to &&
                  to.x === response.selectedOption.representativePosition.x &&
                  to.y === response.selectedOption.representativePosition.y
                );
              }) ??
              territoryMoves[0];

            debugLog(
              TRACE_DEBUG_ENABLED,
              '[GameEngine.DecisionHandler] resolved region_order via choice',
              {
                player: decision.player,
                optionCount: territoryMoves.length,
                selectedMoveId,
              }
            );

            return chosen;
          }
          // If no territory moves (only skip option), fall through to auto-resolve
        }

        // For core move-driven decision types where the backend should behave
        // like an AI host in the absence of a real PlayerInteractionManager,
        // auto-select the first available option. This mirrors the
        // TurnEngineAdapter.autoSelectForAI behaviour and preserves the
        // semantics of the shared orchestrator while avoiding HOST_REJECTED_MOVE
        // errors in soak/diagnostic runs.
        // RR-FIX-2026-01-10: Added 'line_elimination_required' to auto-resolvable types.
        // This decision type is now populated with actual elimination moves.
        const autoResolvableTypes: string[] = [
          'elimination_target',
          'line_order',
          'region_order', // Kept for fallback when no interaction manager / no territory moves
          'line_elimination_required',
        ];

        if (autoResolvableTypes.includes(decision.type)) {
          // Rare defensive case: an elimination_target / ordering decision with
          // zero options indicates a structurally inconsistent state from the
          // orchestrator's perspective (for example, a pending self-elimination
          // but no eligible stacks). For soak/diagnostic hosts we treat this as
          // a no-op decision rather than a hard HOST_REJECTED_MOVE error.
          if (decision.options.length === 0) {
            if (
              decision.type === 'elimination_target' ||
              decision.type === 'line_elimination_required'
            ) {
              debugLog(
                TRACE_DEBUG_ENABLED,
                `[GameEngine.DecisionHandler] ${decision.type} decision with no options; returning no-op elimination move`,
                {
                  type: decision.type,
                  player: decision.player,
                }
              );

              const noopMove: Move = {
                id: `noop-eliminate-${Date.now()}`,
                type: 'eliminate_rings_from_stack',
                player: decision.player,
                // Use a harmless sentinel coordinate that is extremely unlikely
                // to host a real stack; applyEliminateRingsFromStackDecision will
                // treat this as a no-op when no stack exists at this position.
                to: { x: 0, y: 0 },
                eliminationContext:
                  decision.type === 'line_elimination_required' ? 'line' : 'territory',
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: 0,
              };

              return noopMove;
            }

            throw new Error(
              `DecisionHandler.requestDecision received ${decision.type} decision with no options`
            );
          }

          const choice = decision.options[0];

          debugLog(TRACE_DEBUG_ENABLED, '[GameEngine.DecisionHandler] auto-resolving decision', {
            type: decision.type,
            player: decision.player,
            optionCount: decision.options.length,
          });

          return choice;
        }

        // For all other decision types (including any future additions that
        // should be handled explicitly by transports/UI), preserve the previous
        // behaviour and throw so wiring bugs are not silently hidden.
        throw new Error(
          `DecisionHandler.requestDecision called for ${decision.type} - ` +
            `decisions should be handled via explicit Moves in move-driven mode`
        );
      },
    };

    // EventEmitter for adapter events (currently no-op, can be wired to
    // WebSocket notifications in future)
    const eventEmitter: AdapterEventEmitter = {
      emit: (_event: string, _payload?: unknown) => {
        // No-op for now. Future: emit to WebSocket spectators
      },
    };

    // Build deps object - only include debugHook if defined (for exactOptionalPropertyTypes)
    const deps = {
      stateAccessor,
      decisionHandler,
      eventEmitter,
      ...(this.debugCheckpointHook ? { debugHook: this.debugCheckpointHook } : {}),
      replayMode: this.replayMode,
    };

    return new TurnEngineAdapter(deps);
  }

  /**
   * Process a move via the TurnEngineAdapter, delegating rules logic to
   * the shared orchestrator. This is called when useOrchestratorAdapter is true.
   */
  private async processMoveViaAdapter(
    move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
  ): Promise<{
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  }> {
    const beforeStateForHistory = this.getGameState();

    // Create the full move with generated fields
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1,
    };

    // Create the adapter and process the move
    const adapter = this.createAdapterForCurrentGame();

    try {
      const result = await adapter.processMove(fullMove);

      if (!result.success) {
        // DEBUG: Log orchestrator error for debugging Category C failures
        logger.warn('Orchestrator rejected move', {
          component: 'GameEngine.processMoveViaAdapter',
          moveType: fullMove.type,
          player: fullMove.player,
          from: fullMove.from,
          captureTarget: fullMove.captureTarget,
          to: fullMove.to,
          error: result.error,
          currentPhase: this.gameState.currentPhase,
          currentPlayer: this.gameState.currentPlayer,
          stackCount: this.gameState.board.stacks.size,
        });

        // Record unsuccessful orchestrator move for rollout metrics
        getMetricsService().recordOrchestratorMove('orchestrator', 'error');

        return {
          success: false,
          error: result.error || 'Move rejected by orchestrator',
          gameState: this.getGameState(),
        };
      }

      // Record a successful orchestrator operation for rollout/circuit breaker
      orchestratorRollout.recordSuccess();
      getMetricsService().recordOrchestratorMove('orchestrator', 'success');

      // The adapter has already updated this.gameState via stateAccessor.
      // Now handle post-processing that GameEngine is responsible for:

      // Add move to history
      this.gameState.moveHistory.push(fullMove);
      this.gameState.lastMoveAt = new Date();

      // Record structured history entry
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      // Check for game end - victoryResult is set if game is over
      if (result.victoryResult) {
        // Start next player's timer (no-op since game is over, but maintain consistency)
        this.startPlayerTimer(this.gameState.currentPlayer);

        return {
          success: true,
          gameState: this.getGameState(),
          gameResult: result.victoryResult,
        };
      }

      // Handle pending chain capture decision from orchestrator.
      // When processTurnAsync returns with status 'awaiting_decision' for
      // a chain_capture decision type, we need to set up the internal
      // chainCaptureState so that subsequent getValidMoves() returns
      // continue_capture_segment moves and makeMove() accepts them.
      //
      // NOTE: The adapter wraps processTurnAsync which now returns early
      // for chain_capture decisions. We detect this via the returned error
      // containing the pending decision info, OR we can check the result
      // state for chain capture indicators.
      //
      // After a capture move, if chain continuation is available, the
      // orchestrator sets gameState.currentPhase to something (typically
      // stays in movement/capture context). We detect chain continuation
      // by checking if the move was a capture and if further captures
      // are available from the landing position.
      if (
        (fullMove.type === 'overtaking_capture' || fullMove.type === 'continue_capture_segment') &&
        fullMove.to
      ) {
        // Check if chain continuation is available from the landing position
        // using the shared CaptureAggregate which provides a cleaner interface
        const continuationInfo = getChainCaptureContinuationInfo(
          this.gameState,
          fullMove.player,
          fullMove.to
        );

        if (continuationInfo.mustContinue) {
          // Set up chain capture state for GameEngine's internal tracking
          // Capture the cap height of the captured target for proper state
          const targetStack = fullMove.captureTarget
            ? this.boardManager.getStack(fullMove.captureTarget, beforeStateForHistory.board)
            : undefined;
          const capturedCapHeight = targetStack ? targetStack.capHeight : 0;

          this.chainCaptureState = updateChainCaptureStateAfterCapture(
            this.chainCaptureState,
            fullMove,
            capturedCapHeight
          );

          if (this.chainCaptureState) {
            this.chainCaptureState.availableMoves = continuationInfo.availableContinuations;
          }

          // Set phase to chain_capture so getValidMoves returns continue_capture_segment moves
          this.gameState.currentPhase = 'chain_capture';

          // Start player's timer for the chain capture decision
          this.startPlayerTimer(this.gameState.currentPlayer);

          return {
            success: true,
            gameState: this.getGameState(),
          };
        } else {
          // Chain is exhausted; clear any stale chain state
          this.chainCaptureState = undefined;
        }
      }

      // Start next player's timer
      this.startPlayerTimer(this.gameState.currentPlayer);

      return {
        success: true,
        gameState: this.getGameState(),
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err);

      // Record orchestrator error for rollout/circuit breaker tracking
      orchestratorRollout.recordError();
      getMetricsService().recordOrchestratorMove('orchestrator', 'error');

      return {
        success: false,
        error: `Adapter error: ${errorMessage}`,
        gameState: this.getGameState(),
      };
    }
  }

  getGameState(): GameState {
    const state = this.gameState;

    // Deep-clone the board and key collections so tests (especially the
    // AI simulation debug harness) see true pre/post snapshots rather
    // than views that can be mutated via shared Maps.
    const board = state.board;
    const clonedBoard = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };

    const clonedPlayers = state.players.map((p) => ({ ...p }));

    // Normalise totalRingsEliminated for external snapshots so that it
    // always reflects the sum of board.eliminatedRings. This keeps the
    // S-invariant bookkeeping (S = markers + collapsed + eliminated)
    // consistent for parity/debug tooling even if internal counters
    // drift transiently during complex elimination sequences.
    const eliminatedFromBoard = Object.values(clonedBoard.eliminatedRings ?? {}).reduce(
      (sum, value) => sum + value,
      0
    );

    return {
      ...state,
      totalRingsEliminated: eliminatedFromBoard,
      board: clonedBoard,
      moveHistory: [...state.moveHistory],
      history: [...state.history],
      players: clonedPlayers,
      spectators: [...state.spectators],
      // Include LPS tracking summary for client display (RR-CANON-R172)
      lpsTracking: this.getLpsTrackingSummary(),
    };
  }

  /**
   * Helper to ensure that a PlayerInteractionManager is available when
   * attempting to perform any player-facing choice. This keeps the core
   * engine decoupled from transport/UI while still enforcing that choices
   * cannot be resolved silently once integration is enabled.
   */
  private requireInteractionManager(): PlayerInteractionManager {
    if (!this.interactionManager) {
      throw new Error('PlayerInteractionManager is required for player choice operations');
    }
    return this.interactionManager;
  }

  startGame(): boolean {
    // Check if all players are ready
    const allReady = this.gameState.players.every((p) => p.isReady);
    if (!allReady) {
      return false;
    }

    this.gameState.gameStatus = 'active';
    this.gameState.lastMoveAt = new Date();

    // Start the first player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    return true;
  }

  /**
   * Internal no-op hook to keep selected helper methods referenced so that
   * ts-node/TypeScript with noUnusedLocals can compile the server in dev
   * without treating them as dead code. This has no impact on runtime
   * behaviour; it only preserves helpers for parity/debug tooling and
   * future rule-engine extensions.
   */
  private _debugUseInternalHelpers(): void {
    // These void-accesses mark the helpers as "used" from the compiler's
    // perspective without invoking them.
    void this.processLineFormations;
    void this.getValidLineProcessingMoves;
    void this.getValidTerritoryProcessingMoves;

    // Keep selected shared helpers and adapter types referenced so that
    // ts-node/TypeScript with noUnusedLocals can compile backend entry
    // points (including orchestrator soak harnesses) without treating
    // them as dead code. These are intentionally no-ops.
    void filterProcessableTerritoryRegions;

    const _debugAdapterResult: AdapterMoveResult | null = null;
    void _debugAdapterResult;
  }

  /**
   * Append a structured history entry for a canonical move applied to the
   * engine. This is the primary hook used by parity/debug tooling; it is
   * intentionally side-effect-free with respect to core rules logic.
   */
  private appendHistoryEntry(before: GameState, action: Move): void {
    const after = this.getGameState();

    // Raw snapshots based purely on the underlying GameState. These are
    // used as a diagnostic reference, but the history entry itself is
    // normalised to match the geometric board summaries so that
    // progress.collapsed and progress.markers can never silently drift
    // out of sync with the recorded board geometry.
    const rawProgressBefore = computeProgressSnapshot(before);
    const rawProgressAfter = computeProgressSnapshot(after);

    const boardBeforeSummary = summarizeBoard(before.board);
    const boardAfterSummary = summarizeBoard(after.board);

    const progressBefore = {
      markers: boardBeforeSummary.markers.length,
      collapsed: boardBeforeSummary.collapsedSpaces.length,
      eliminated: rawProgressBefore.eliminated,
      S:
        boardBeforeSummary.markers.length +
        boardBeforeSummary.collapsedSpaces.length +
        rawProgressBefore.eliminated,
    };

    const progressAfter = {
      markers: boardAfterSummary.markers.length,
      collapsed: boardAfterSummary.collapsedSpaces.length,
      eliminated: rawProgressAfter.eliminated,
      S:
        boardAfterSummary.markers.length +
        boardAfterSummary.collapsedSpaces.length +
        rawProgressAfter.eliminated,
    };

    // Defensive diagnostic: under trace-debug runs, detect any mismatch
    // between the raw S-invariant counts (from GameState) and the
    // geometry implied by the board summaries. This continues to surface
    // bookkeeping drift for debugging, while the stored history entry is
    // always consistent with its own boardBefore/After summaries.
    const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

    if (TRACE_DEBUG_ENABLED) {
      const collapsedFromBoard = boardAfterSummary.collapsedSpaces.length;
      const markersFromBoard = boardAfterSummary.markers.length;

      // Compare board-level vs player-level elimination accounting so we can
      // spot drift between the S notion used by the soak harness
      // (totalRingsEliminated / board.eliminatedRings) and the orchestrator's
      // internal view (players[].eliminatedRings).
      const eliminatedFromBoardBefore = Object.values(before.board.eliminatedRings ?? {}).reduce(
        (sum, value) => sum + value,
        0
      );
      const eliminatedFromBoardAfter = Object.values(after.board.eliminatedRings ?? {}).reduce(
        (sum, value) => sum + value,
        0
      );
      const eliminatedFromPlayersBefore = before.players.reduce(
        (sum, p) => sum + (p.eliminatedRings ?? 0),
        0
      );
      const eliminatedFromPlayersAfter = after.players.reduce(
        (sum, p) => sum + (p.eliminatedRings ?? 0),
        0
      );

      const sPlayersBefore =
        boardBeforeSummary.markers.length +
        boardBeforeSummary.collapsedSpaces.length +
        eliminatedFromPlayersBefore;
      const sPlayersAfter =
        boardAfterSummary.markers.length +
        boardAfterSummary.collapsedSpaces.length +
        eliminatedFromPlayersAfter;

      // Strict S-invariant trace: log any decrease in the canonical S that the
      // soak harness observes (markers + collapsed + eliminated-from-board)
      // while the game remains active. This is the primary signal we're
      // chasing in orchestrator soak debugging.
      if (
        before.gameStatus === 'active' &&
        after.gameStatus === 'active' &&
        progressAfter.S < progressBefore.S
      ) {
        logger.warn('S-invariant decreased during active game', {
          component: 'GameEngine.appendHistoryEntry',
          invariantType: 'STRICT_S_INVARIANT_DECREASE',
          moveNumber: action.moveNumber,
          actor: action.player,
          phaseBefore: before.currentPhase,
          phaseAfter: after.currentPhase,
          statusBefore: before.gameStatus,
          statusAfter: after.gameStatus,
          progressBefore,
          progressAfter,
          eliminatedFromBoardBefore,
          eliminatedFromBoardAfter,
          eliminatedFromPlayersBefore,
          eliminatedFromPlayersAfter,
          sPlayersBefore,
          sPlayersAfter,
          stateHashBefore: hashGameState(before),
          stateHashAfter: hashGameState(after),
        });
        // Record an orchestrator-related invariant violation for S decreasing.
        getMetricsService().recordOrchestratorInvariantViolation('S_INVARIANT_DECREASED');
      }

      // Detect non-monotone totalRingsEliminated accounting even when S itself
      // does not decrease. This surfaces cases where elimination bookkeeping
      // drifts but markers/collapsed compensate, which is still a strict
      // invariant violation for the orchestrator rollout.
      if (
        before.gameStatus === 'active' &&
        after.gameStatus === 'active' &&
        eliminatedFromBoardAfter < eliminatedFromBoardBefore
      ) {
        logger.warn('Total rings eliminated decreased', {
          component: 'GameEngine.appendHistoryEntry',
          invariantType: 'TOTAL_RINGS_ELIMINATED_DECREASED',
          moveNumber: action.moveNumber,
          actor: action.player,
          phaseBefore: before.currentPhase,
          phaseAfter: after.currentPhase,
          statusBefore: before.gameStatus,
          statusAfter: after.gameStatus,
          eliminatedFromBoardBefore,
          eliminatedFromBoardAfter,
          stateHashBefore: hashGameState(before),
          stateHashAfter: hashGameState(after),
        });
        getMetricsService().recordOrchestratorInvariantViolation(
          'TOTAL_RINGS_ELIMINATED_DECREASED'
        );
      }

      // Log when board-level and player-level elimination accounting diverge
      // even if S itself does not decrease. This helps diagnose cases where
      // one view of S is monotone while the other is not.
      if (eliminatedFromBoardAfter !== eliminatedFromPlayersAfter) {
        logger.debug('S-elimination bookkeeping mismatch', {
          component: 'GameEngine.appendHistoryEntry',
          moveNumber: action.moveNumber,
          actor: action.player,
          phaseBefore: before.currentPhase,
          phaseAfter: after.currentPhase,
          statusBefore: before.gameStatus,
          statusAfter: after.gameStatus,
          eliminatedFromBoardBefore,
          eliminatedFromBoardAfter,
          eliminatedFromPlayersBefore,
          eliminatedFromPlayersAfter,
          sFromBoardBefore: progressBefore.S,
          sFromBoardAfter: progressAfter.S,
          sFromPlayersBefore: sPlayersBefore,
          sFromPlayersAfter: sPlayersAfter,
          stateHashBefore: hashGameState(before),
          stateHashAfter: hashGameState(after),
        });
      }

      if (
        collapsedFromBoard !== rawProgressAfter.collapsed ||
        markersFromBoard !== rawProgressAfter.markers
      ) {
        logger.debug('S-invariant debug mismatch', {
          component: 'GameEngine.appendHistoryEntry',
          moveNumber: action.moveNumber,
          actor: action.player,
          phaseAfter: after.currentPhase,
          statusAfter: after.gameStatus,
          rawProgressAfter,
          countsFromBoardSummary: {
            markers: markersFromBoard,
            collapsed: collapsedFromBoard,
          },
          stateHashAfter: hashGameState(after),
        });
      }
    }

    const entry: GameHistoryEntry = {
      moveNumber: action.moveNumber,
      action,
      actor: action.player,
      phaseBefore: before.currentPhase,
      phaseAfter: after.currentPhase,
      statusBefore: before.gameStatus,
      statusAfter: after.gameStatus,
      progressBefore,
      progressAfter,
      stateHashBefore: hashGameState(before),
      stateHashAfter: hashGameState(after),
      boardBeforeSummary,
      boardAfterSummary,
    };

    const history: GameHistoryEntry[] = [...this.gameState.history, entry];
    this.gameState = {
      ...this.gameState,
      history,
    };
  }

  async makeMove(move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>): Promise<{
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  }> {
    // Special meta-move: swap_sides (pie rule) for 2-player games.
    // This is handled entirely in the backend engine and never routed
    // through the shared orchestrator or Python rules engine.
    if (move.type === 'swap_sides') {
      return this.applySwapSidesMove(move.player);
    }

    // When a chain capture is in progress, only follow-up capture segments
    // chosen as explicit continue_capture_segment moves from the current
    // chain position are legal until no options remain.
    if (this.chainCaptureState) {
      const state = this.chainCaptureState;

      // All moves during a chain must be made by the same player.
      if (move.player !== state.playerNumber) {
        return {
          success: false,
          error: 'Chain capture in progress: only the capturing player may move',
        };
      }

      // During a chain, only continue_capture_segment moves from the current
      // position are allowed. Ending a chain early is not permitted by the
      // rules; the chain ends only when no further captures are available.
      if (
        move.type !== 'continue_capture_segment' ||
        !move.from ||
        positionToString(move.from) !== positionToString(state.currentPosition)
      ) {
        return {
          success: false,
          error: 'Chain capture in progress: must continue capturing with the same stack',
        };
      }
    } else {
      // Defensive: it is not legal to start a chain with a
      // continue_capture_segment move when no chain is active.
      if (move.type === 'continue_capture_segment') {
        return {
          success: false,
          error: 'No chain capture in progress for continue_capture_segment move',
        };
      }
    }

    // Enforce must-move origin when a placement has occurred this turn.
    if (this.mustMoveFromStackKey) {
      const moveFromKey = move.from ? positionToString(move.from) : undefined;

      const isMovementOrCaptureType =
        move.type === 'move_stack' ||
        move.type === 'overtaking_capture' ||
        move.type === 'continue_capture_segment';

      if (isMovementOrCaptureType && (!moveFromKey || moveFromKey !== this.mustMoveFromStackKey)) {
        return {
          success: false,
          error: 'You must move the stack that was just placed or updated this turn',
        };
      }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ORCHESTRATOR ADAPTER DELEGATION
    // ═══════════════════════════════════════════════════════════════════════
    // All move processing is delegated to TurnEngineAdapter which wraps the
    // canonical shared orchestrator (processTurnAsync). The legacy
    // RuleEngine-based pipeline has been removed.
    // ═══════════════════════════════════════════════════════════════════════
    return this.processMoveViaAdapter(move);
  }

  /**
   * Enumerate all valid capture moves from a given position for the
   * specified player by ray-walking in each movement direction and
   * validating each candidate via RuleEngine.
   *
   * This helper delegates to the shared CaptureAggregate which provides
   * the canonical source for chain-continuation options; the unified Move
   * model simply re-labels these as 'continue_capture_segment' during
   * the dedicated 'chain_capture' phase.
   */
  private getCaptureOptionsFromPosition(position: Position, playerNumber: number): Move[] {
    const continuationInfo = getChainCaptureContinuationInfo(
      this.gameState,
      playerNumber,
      position
    );
    return continuationInfo.availableContinuations;
  }

  /**
   * When multiple capture continuations are available from the current
   * chain position, use the generic PlayerChoice system to let the
   * active player choose a direction and landing. This wires the
   * CaptureDirectionChoice type into GameEngine using the
   * chainCaptureState.availableMoves list.
   *
   * NOTE: This helper does not yet automatically apply the chosen move;
   * it simply selects and returns it. Future work can integrate this
   * into a full chain-capture loop once the transport/UI flow is ready.
   */

  /**
   * Enumerate canonical line-processing decision moves for the given player.
   *
   * This is now a thin adapter over the shared line-decision helpers so that
   * backend GameEngine, shared GameEngine, RuleEngine, and sandbox all share
   * a single source of truth for which `process_line` / `choose_line_option`
   * Moves exist in a given GameState.
   *
   * Semantics:
   * - One `process_line` Move per player-owned line.
   * - For each line index, {@link enumerateChooseLineRewardMoves} may expose:
   *   - A collapse-all reward (with no `collapsedMarkers`), and
   *   - Zero or more minimum-collapse contiguous-length-L segments for
   *     overlength lines (each with `collapsedMarkers` populated).
   */
  private getValidLineProcessingMoves(playerNumber: number): Move[] {
    // Base process_line decisions from the shared helper. For phase
    // scheduling we rely on fresh geometry (detect_now) so that a stale
    // formedLines cache can never cause the engine to enter
    // `line_processing` when no canonical lines actually exist for the
    // current player. The helper will still consult the cache internally
    // when it is known to be valid.
    const processMoves = enumerateProcessLineMoves(this.gameState, playerNumber, {
      detectionMode: 'detect_now',
    });

    // Reward decisions are driven per line index using the same detection
    // order as the shared geometry helper.
    const playerLines = findLinesForPlayer(this.gameState.board, playerNumber);
    const rewardMoves: Move[] = [];

    playerLines.forEach((_line, index) => {
      rewardMoves.push(...enumerateChooseLineRewardMoves(this.gameState, playerNumber, index));
    });

    return [...processMoves, ...rewardMoves];
  }

  /**
   /**
    * Enumerate canonical territory-processing decision moves for the
    * current player. This now delegates to the shared
    * {@link enumerateProcessTerritoryRegionMoves} helper so that backend
    * GameEngine, RuleEngine, and sandbox all share a single source of
    * truth for:
    *
    * - Disconnected-region detection.
    * - Q23 self-elimination gating (must control a stack outside the region).
    * - Move ID / payload conventions for choose_territory_option.
    */
  private getValidTerritoryProcessingMoves(playerNumber: number): Move[] {
    return enumerateProcessTerritoryRegionMoves(this.gameState, playerNumber);
  }
  /**
   * DIAGNOSTICS-ONLY (legacy): process all line formations with graduated rewards.
   * Rule Reference: Section 11.2, 11.3
   *
   * For exact required length (4 for 8x8, 5 for 19x19/hex):
   *   - Collapse all markers
   *   - Eliminate one ring from any controlled stack
   *
   * For longer lines (5+ for 8x8, 6+ for 19x19/hex):
   *   - Option 1: Collapse all + eliminate one ring
   *   - Option 2: Collapse required markers only, no elimination
   *
   * Canonical line processing is now handled via applyProcessLineDecision /
   * applyChooseLineRewardDecision in src/shared/engine/lineDecisionHelpers.ts,
   * surfaced through the shared orchestrator and move-driven decision phases.
   *
   * @deprecated Phase 4 legacy path — use lineDecisionHelpers + shared orchestrator instead.
   * This method will be removed once all tests migrate to orchestrator-backed flows.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * Ownership / deprecation:
   * - Called directly by legacy scenario suites (for example
   *   tests/scenarios/RulesMatrix.GameEngine.test.ts) which snapshot the
   *   behaviour of the pre-orchestrator line pipeline.
   * - New code should drive line processing via the shared
   *   applyProcessLineDecision / applyChooseLineRewardDecision helpers and
   *   TurnEngineAdapter instead of using this method.
   * - Once the RulesMatrix and other parity suites have been fully migrated
   *   to move-driven/orchestrator flows, this helper can be removed.
   */

  private async processLineFormations(): Promise<void> {
    const requiredLength = getEffectiveLineLengthThreshold(
      this.gameState.boardType,
      this.gameState.players.length,
      this.gameState.rulesOptions
    );

    // Keep processing until no more (eligible) lines exist
    while (true) {
      const allLines = this.boardManager.findAllLines(this.gameState.board);
      if (allLines.length === 0) break;

      // Only consider lines for the moving player that meet the effective
      // threshold (e.g. 4-in-a-row for 2p 8x8, base length otherwise).
      const playerLines = allLines.filter(
        (line) =>
          line.player === this.gameState.currentPlayer && line.positions.length >= requiredLength
      );
      if (playerLines.length === 0) break;

      let lineToProcess: LineInfo;

      if (!this.interactionManager || playerLines.length === 1) {
        // No interaction manager wired yet, or only one choice: keep current behaviour
        lineToProcess = playerLines[0];
      } else {
        const interaction = this.requireInteractionManager();

        const choice: LineOrderChoice = {
          id: generateUUID(),
          gameId: this.gameState.id,
          playerNumber: this.gameState.currentPlayer,
          type: 'line_order',
          prompt: 'Choose which line to process first',
          options: playerLines.map((line, index) => {
            const lineKey = line.positions.map((p) => positionToString(p)).join('|');
            return {
              lineId: String(index),
              markerPositions: line.positions,
              /**
               * Stable identifier for the canonical 'process_line' Move that
               * would process this line when enumerated via
               * getValidLineProcessingMoves. This lets transports/AI map this
               * choice option directly onto a Move.id.
               */
              moveId: `process-line-${index}-${lineKey}`,
            };
          }),
        };

        const response: PlayerChoiceResponseFor<LineOrderChoice> =
          await interaction.requestChoice(choice);
        const selected = response.selectedOption;
        const index = parseInt(selected.lineId, 10);
        lineToProcess = playerLines[index] ?? playerLines[0];
      }

      await this.processOneLine(lineToProcess, requiredLength);
      // After processing one line, loop will re-evaluate remaining lines
    }
  }

  /**
   * Process a single line formation
   * Rule Reference: Section 11.2
   *
   * @deprecated Phase 4 legacy path — use applyProcessLineDecision / applyChooseLineRewardDecision
   * in src/shared/engine/lineDecisionHelpers.ts via the shared orchestrator instead.
   * This method will be removed once all tests migrate to orchestrator-backed flows.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * Ownership / deprecation:
   * - Internal to the legacy line-processing pipeline; invoked by
   *   processLineFormations() and never called by production hosts.
   * - Behaviour is covered indirectly by RulesMatrix.* and other legacy
   *   parity suites; new tests should prefer exercising the shared
   *   lineDecisionHelpers surface instead.
   */
  private async processOneLine(line: LineInfo, requiredLength: number): Promise<void> {
    const lineLength = line.positions.length;

    if (lineLength === requiredLength) {
      // Exact required length: Must collapse all and eliminate ring/cap
      this.collapseLineMarkers(line.positions, line.player);
      await this.eliminatePlayerRingOrCapWithChoice(line.player);
    } else if (lineLength > requiredLength) {
      // Longer than required: player chooses Option 1 or Option 2 when an
      // interaction manager is available; otherwise, preserve current
      // behaviour and default to Option 2 (collapse minimum only, no elimination).
      if (!this.interactionManager) {
        const markersToCollapse = line.positions.slice(0, requiredLength);
        this.collapseLineMarkers(markersToCollapse, line.player);
        return;
      }

      const interaction = this.requireInteractionManager();

      // Pre-compute canonical decision moves so we can attach moveIds to the choice
      const validMoves = this.getValidLineProcessingMoves(this.gameState.currentPlayer);
      const lineKey = line.positions.map((p) => positionToString(p)).join('|');

      const option1Move = validMoves.find(
        (m) =>
          m.type === 'choose_line_option' &&
          m.id.includes(lineKey) &&
          m.collapsedMarkers?.length === line.positions.length
      );

      const option2Move = validMoves.find(
        (m) =>
          m.type === 'choose_line_option' &&
          m.id.includes(lineKey) &&
          m.collapsedMarkers?.length === requiredLength
      );

      const choice: LineRewardChoice = {
        id: generateUUID(),
        gameId: this.gameState.id,
        playerNumber: this.gameState.currentPlayer,
        type: 'line_reward_option',
        prompt: 'Choose line reward option',
        options: ['option_1_collapse_all_and_eliminate', 'option_2_min_collapse_no_elimination'],
        moveIds: {
          ...(option1Move?.id ? { option_1_collapse_all_and_eliminate: option1Move.id } : {}),
          ...(option2Move?.id ? { option_2_min_collapse_no_elimination: option2Move.id } : {}),
        },
      };

      const response: PlayerChoiceResponseFor<LineRewardChoice> =
        await interaction.requestChoice(choice);
      const selected = response.selectedOption;

      if (selected === 'option_1_collapse_all_and_eliminate') {
        this.collapseLineMarkers(line.positions, line.player);
        await this.eliminatePlayerRingOrCapWithChoice(line.player);
      } else {
        const markersToCollapse = line.positions.slice(0, requiredLength);
        this.collapseLineMarkers(markersToCollapse, line.player);
      }
    }
  }

  /**
   * Collapse marker positions to player's color territory
   * Rule Reference: Section 11.2 - Markers collapse to colored spaces
   */
  private collapseLineMarkers(positions: Position[], player: number): void {
    for (const pos of positions) {
      this.boardManager.setCollapsedSpace(pos, player, this.gameState.board);
    }
    // Update player's territory count
    this.updatePlayerTerritorySpaces(player, positions.length);
  }

  /**
   * Eliminate entire cap from player's controlled stacks
   * Rule Reference: Section 11.2 - Moving player chooses which stack cap to eliminate.
   * For line/territory processing, we eliminate the entire cap (all consecutive top rings
   * of the controlling color). For mixed-colour stacks, this exposes buried rings; for
   * single-colour stacks with height > 1, this eliminates all rings.
   */
  private eliminatePlayerRingOrCap(player: number, stackPosition?: Position): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    if (playerStacks.length === 0) {
      // No stacks to eliminate from, player might have rings in hand
      const playerState = this.gameState.players.find((p) => p.playerNumber === player);
      if (playerState && playerState.ringsInHand > 0) {
        // Eliminate from hand
        playerState.ringsInHand--;
        this.gameState.totalRingsEliminated++;

        // Track eliminated rings in board state
        if (!this.gameState.board.eliminatedRings[player]) {
          this.gameState.board.eliminatedRings[player] = 0;
        }
        this.gameState.board.eliminatedRings[player]++;

        // Update player state
        this.updatePlayerEliminatedRings(player, 1);
      }
      return;
    }

    // If a specific stack position is provided, try to find and use it
    if (stackPosition) {
      const targetKey = positionToString(stackPosition);
      const targetStack = playerStacks.find((s) => positionToString(s.position) === targetKey);
      if (targetStack) {
        this.eliminateFromStack(targetStack, player);
        return;
      }
    }

    // Default behaviour: eliminate from first stack
    const stack = playerStacks[0];
    this.eliminateFromStack(stack, player);
  }

  /**
   * Core elimination logic from a specific stack. Used by both the
   * default elimination path and the choice-based elimination helper.
   */
  private eliminateFromStack(stack: RingStack, player: number): void {
    // Calculate cap height
    const capHeight = calculateCapHeight(stack.rings);

    // Eliminate the entire cap (all consecutive top rings of controlling color)
    const remainingRings = stack.rings.slice(capHeight);

    // Update eliminated rings count
    this.gameState.totalRingsEliminated += capHeight;
    if (!this.gameState.board.eliminatedRings[player]) {
      this.gameState.board.eliminatedRings[player] = 0;
    }
    this.gameState.board.eliminatedRings[player] += capHeight;

    // Update player state
    this.updatePlayerEliminatedRings(player, capHeight);

    if (remainingRings.length > 0) {
      // Update stack with remaining rings
      const newStack: RingStack = {
        ...stack,
        rings: remainingRings,
        stackHeight: remainingRings.length,
        capHeight: calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0],
      };
      this.boardManager.setStack(stack.position, newStack, this.gameState.board);
    } else {
      // Stack is now empty, remove it
      this.boardManager.removeStack(stack.position, this.gameState.board);
    }
  }

  /**
   * Eliminate entire stack cap using the player choice system when available.
   * All controlled stacks are eligible, including height-1 standalone rings
   * (per RR-CANON-R022/R145).
   * Falls back to default behaviour when no interaction manager is wired.
   */
  private async eliminatePlayerRingOrCapWithChoice(player: number): Promise<void> {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    if (playerStacks.length === 0) {
      // Mirror the hand-elimination behaviour from eliminatePlayerRingOrCap
      const playerState = this.gameState.players.find((p) => p.playerNumber === player);
      if (playerState && playerState.ringsInHand > 0) {
        playerState.ringsInHand--;
        this.gameState.totalRingsEliminated++;
        if (!this.gameState.board.eliminatedRings[player]) {
          this.gameState.board.eliminatedRings[player] = 0;
        }
        this.gameState.board.eliminatedRings[player]++;
        this.updatePlayerEliminatedRings(player, 1);
      }
      return;
    }

    if (!this.interactionManager || playerStacks.length === 1) {
      // No manager or only one stack: use default behaviour
      this.eliminatePlayerRingOrCap(player);
      return;
    }

    const interaction = this.requireInteractionManager();

    // This method is called for territory elimination, so context is 'territory'
    const eliminationContext = 'territory' as const;

    const choice: RingEliminationChoice = {
      id: generateUUID(),
      gameId: this.gameState.id,
      playerNumber: player,
      type: 'ring_elimination',
      eliminationContext,
      prompt:
        'Territory cost: You must eliminate your ENTIRE CAP from an eligible stack outside the region.',
      options: playerStacks.map((stack) => {
        const stackKey = positionToString(stack.position);
        return {
          stackPosition: stack.position,
          capHeight: stack.capHeight,
          totalHeight: stack.stackHeight,
          // Territory elimination removes entire cap (RR-CANON-R145)
          ringsToEliminate: stack.capHeight,
          /**
           * Stable identifier for the canonical 'eliminate_rings_from_stack'
           * Move that would eliminate from this stack when enumerated via
           * RuleEngine.getValidMoves during the territory_processing phase.
           * This lets transports/AI map this choice option directly onto a
           * Move.id in the unified Move model.
           */
          moveId: `eliminate-${stackKey}`,
        };
      }),
    };

    const response: PlayerChoiceResponseFor<RingEliminationChoice> =
      await interaction.requestChoice(choice);
    const selected = response.selectedOption;

    const selectedKey = positionToString(selected.stackPosition);
    const chosenStack =
      playerStacks.find((s) => positionToString(s.position) === selectedKey) || playerStacks[0];

    this.eliminateFromStack(chosenStack, player);
  }

  /**
   * Update player's eliminatedRings counter
   */
  private updatePlayerEliminatedRings(playerNumber: number, count: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (player) {
      player.eliminatedRings += count;
    }
  }

  /**
   * Update player's territorySpaces counter
   */
  private updatePlayerTerritorySpaces(playerNumber: number, count: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (player) {
      player.territorySpaces += count;
    }
  }

  /**
   * Check if player can process a disconnected region
   * Rule Reference: Section 12.2 - Self-Elimination Prerequisite
   *
   * Player must have at least one ring/cap outside the region before processing
   */
  private canProcessDisconnectedRegion(region: Territory, player: number): boolean {
    // Thin wrapper around the shared self-elimination prerequisite helper
    // so GameEngine-based flows (legacy automatic and move-driven decision
    // phases) stay aligned with the canonical territoryProcessing module.
    return canProcessTerritoryRegion(this.gameState.board, region, { player });
  }

  /**
   * Core territory-processing helper shared by both legacy
   * choice-driven flows and the move-driven decision model.
   *
   * This applies the geometric consequences of processing a
   * disconnected region (eliminating all rings in the region,
   * collapsing spaces and border markers, and crediting all
   * eliminations/territory to the moving player) but does **not**
   * perform the mandatory self-elimination step. That final
   * Self-elimination is handled via explicit eliminate_rings_from_stack
   * decision Moves surfaced by RuleEngine after this helper completes,
   * so the self-elimination is represented as a canonical Move.
   *
   * @deprecated Phase 4 legacy path — use applyProcessTerritoryRegionDecision
   * in src/shared/engine/territoryDecisionHelpers.ts via the shared orchestrator instead.
   * This method will be removed once all tests migrate to orchestrator-backed flows.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * DO NOT REMOVE - used for parity testing (TerritoryCore.GameEngine_vs_Sandbox.test.ts).
   * See P20.3-2 for deprecation timeline.
   */
  private processDisconnectedRegionCore(region: Territory, movingPlayer: number): void {
    // Delegate the geometric core (internal eliminations + region/border
    // collapse) to the shared engine helper so that backend GameEngine,
    // sandbox engine, and rules-layer tests share a single source of truth
    // for territory semantics.
    const outcome = applyTerritoryRegion(this.gameState.board, region, { player: movingPlayer });

    // Replace the board with the processed clone from the shared helper.
    this.gameState = {
      ...this.gameState,
      board: outcome.board,
    };

    // Apply per-player territory gain at the GameState level. Under current
    // rules all territory gain from disconnected regions is credited to the
    // moving player.
    const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
    if (territoryGain > 0) {
      this.updatePlayerTerritorySpaces(movingPlayer, territoryGain);
    }

    // Apply internal elimination deltas to GameState.totalRingsEliminated and
    // the moving player's eliminatedRings counter. The BoardState-level
    // bookkeeping (board.eliminatedRings) has already been updated by the
    // shared helper.
    const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
    if (internalElims > 0) {
      this.gameState.totalRingsEliminated += internalElims;
      this.updatePlayerEliminatedRings(movingPlayer, internalElims);
    }
  }

  /**
   * Process a single disconnected region
   * Rule Reference: Section 12.2 - Processing steps
   *
   * @deprecated Phase 4 legacy path — use applyProcessTerritoryRegionDecision
   * in src/shared/engine/territoryDecisionHelpers.ts via the shared orchestrator instead.
   * This method will be removed once all tests migrate to orchestrator-backed flows.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * Ownership / deprecation:
   * - Internal to the legacy territory-processing pipeline; invoked by
   *   processDisconnectedRegions() and not used by production hosts.
   * - Covered by legacy territory-disconnection suites; new callers should
   *   use move-driven territory decisions via TurnEngineAdapter.
   */
  private async processOneDisconnectedRegion(
    region: Territory,
    movingPlayer: number
  ): Promise<void> {
    // Apply the geometric/core consequences. Self-elimination is now handled
    // via explicit eliminate_rings_from_stack decisions in move-driven mode.
    this.processDisconnectedRegionCore(region, movingPlayer);
  }
  /**
   * Test-only helper: process all eligible disconnected regions for the
   * current player using the legacy territory-processing pipeline
   * (processOneDisconnectedRegion).
   *
   * This mirrors the behaviour exercised by the RulesMatrix territory
   * scenarios, but is implemented purely
   * in terms of {@link processDisconnectedRegionCore} and the shared
   * territory helpers so it stays aligned with RR‑CANON‑R140–R145 and
   * the TS/Python engines.
   *
   * Production hosts do not call this method; it exists solely for
   * parity/scenario suites and diagnostic harnesses.
   *
   * @deprecated Phase 4 legacy path — use applyProcessTerritoryRegionDecision
   * in src/shared/engine/territoryDecisionHelpers.ts via the shared orchestrator instead.
   * This method will be removed once all tests migrate to orchestrator-backed flows.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * Ownership / deprecation:
   * - Used by legacy parity/scenario suites (for example
   *   tests/scenarios/RulesMatrix.Comprehensive.test.ts) to exercise the
   *   pre-orchestrator territory pipeline.
   * - New territory tests should drive decisions via the shared
   *   territoryDecisionHelpers + TurnEngineAdapter instead of calling this.
   */
  public async processDisconnectedRegions(): Promise<void> {
    const movingPlayer = this.gameState.currentPlayer;

    // Drive the legacy pipeline until no further eligible regions remain
    // for the moving player. After each collapse we recompute the
    // candidate set so that the self-elimination prerequisite (Q23 /
    // RR‑CANON‑R143) is evaluated against the updated board.
    while (this.gameState.gameStatus === 'active') {
      const disconnected: Territory[] =
        this.boardManager.findDisconnectedRegions(this.gameState.board, movingPlayer) ?? [];

      if (disconnected.length === 0) {
        break;
      }

      const eligible = disconnected.filter((region) =>
        this.canProcessDisconnectedRegion(region, movingPlayer)
      );

      if (eligible.length === 0) {
        break;
      }

      // For these legacy tests we process regions in the order returned
      // by BoardManager. Region-order choice semantics (Q20 / RR‑CANON‑R144)
      // are covered by the dedicated move-driven region-order suites.
      const regionToProcess = eligible[0];
      await this.processOneDisconnectedRegion(regionToProcess, movingPlayer);
    }
  }

  /**
   * Advance game through phases according to RingRift rules
   * Rule Reference: Section 4, Section 15.2
   *
   * Phase Flow:
   * 1. ring_placement (optional unless no rings on board)
   * 2. movement (required if able)
   * 3. capture (optional to start, mandatory chaining)
   * 4. line_processing (automatic)
   * 5. territory_processing (automatic)
   * 6. Next player's turn
   */
  private advanceGame(): void {
    const deps: TurnEngineDeps = {
      boardManager: this.boardManager,
      ruleEngine: this.ruleEngine,
    };

    const hooks: TurnEngineHooks = {
      eliminatePlayerRingOrCap: (playerNumber: number, stackPosition?: Position) => {
        this.eliminatePlayerRingOrCap(playerNumber, stackPosition);
      },
      endGame: (winner?: number, reason?: string) => this.endGame(winner, reason),
      getLpsState: () => this._lpsState,
      setLpsState: (next) => {
        this._lpsState = next;
      },
      hasAnyRealActionForPlayer: (pn) => this.hasAnyRealActionForPlayer(pn),
      hasMaterialForPlayer: (pn) => {
        const player = this.gameState.players.find((p) => p.playerNumber === pn);
        if (!player) return false;
        const hasStacks =
          this.gameState.board.stacks.size > 0
            ? Array.from(this.gameState.board.stacks.values()).some(
                (s) => s.controllingPlayer === pn || s.rings.includes(pn)
              )
            : false;
        return player.ringsInHand > 0 || hasStacks;
      },
    };

    const turnStateBefore: PerTurnState = {
      hasPlacedThisTurn: this.hasPlacedThisTurn,
      mustMoveFromStackKey: this.mustMoveFromStackKey,
    };

    const previousPhase = this.gameState.currentPhase;

    const turnStateAfter = advanceGameForCurrentPlayer(
      this.gameState,
      turnStateBefore,
      deps,
      hooks
    );

    this.hasPlacedThisTurn = turnStateAfter.hasPlacedThisTurn;
    this.mustMoveFromStackKey = turnStateAfter.mustMoveFromStackKey;
    if (
      this.mustMoveFromStackKey === undefined &&
      turnStateAfter.mustMoveFromStackKey === undefined
    ) {
      // console.log('[GameEngine] advanceGame: mustMove cleared/remains undefined');
    } else {
      debugLog(
        flagEnabled('RINGRIFT_TRACE_DEBUG'),
        `[GameEngine] advanceGame: mustMove is ${this.mustMoveFromStackKey}`
      );
    }

    // Note: skipping players who have no stacks and no rings in hand is now
    // handled centrally by the shared turnLogic.advanceTurnAndPhase helper
    // (via TurnEngine). We rely on that shared sequencing so backend and
    // sandbox rotate turns using exactly the same material-based skipping
    // rules, avoiding host-specific double-skips.

    // Whenever we leave the territory_processing or line_processing phases
    // for the current player, clear any pending self-elimination flags so
    // the next interactive turn starts with a clean slate.
    if (
      previousPhase === 'territory_processing' &&
      this.gameState.currentPhase !== 'territory_processing'
    ) {
      this.pendingTerritorySelfElimination = false;
    }
    if (previousPhase === 'line_processing' && this.gameState.currentPhase !== 'line_processing') {
      this.pendingLineRewardElimination = false;
    }
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player) return;

    this.clockManager.startTimer({
      playerNumber,
      timeRemaining: player.timeRemaining,
      isAI: player.type === 'ai',
    });
  }

  private stopPlayerTimer(playerNumber: number): void {
    this.clockManager.stopTimer(playerNumber);
  }

  private endGame(
    winner?: number,
    reason?: string
  ): {
    success: boolean;
    gameResult: GameResult;
  } {
    this.gameState.gameStatus = 'completed';
    this.gameState.winner = winner;

    // Normalise all terminal states so that callers never observe a completed
    // game in an internal bookkeeping or interactive phase. The sandbox engine
    // likewise reports terminal states with currentPhase === 'ring_placement',
    // and parity tests compare snapshots after victory. Using a single canonical
    // phase for completed games keeps backend vs sandbox victory snapshots
    // aligned without affecting in-game phase transitions.
    this.gameState.currentPhase = 'ring_placement';

    // Clear all timers
    this.clockManager.stopAllTimers();

    // Calculate final scores
    const finalScore: { [playerNumber: number]: number } = {};
    for (const player of this.gameState.players) {
      const playerStacks = this.boardManager.getPlayerStacks(
        this.gameState.board,
        player.playerNumber
      );
      const stackCount = playerStacks.reduce((sum, stack) => sum + stack.stackHeight, 0);

      const territories = this.boardManager.findPlayerTerritories(
        this.gameState.board,
        player.playerNumber
      );
      const territorySize = territories.reduce(
        (sum, territory) => sum + territory.spaces.length,
        0
      );

      finalScore[player.playerNumber] = stackCount + territorySize;
    }

    const gameResult: GameResult = {
      ...(winner !== undefined && { winner }),
      reason: (reason as GameResult['reason']) || 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: finalScore,
      },
    };

    // Update player ratings if this is a rated game
    if (this.gameState.isRated) {
      this.updatePlayerRatings(gameResult);
    }

    return {
      success: true,
      gameResult,
    };
  }

  private updatePlayerRatings(gameResult: GameResult): void {
    // Rating calculation logic would go here
    const winnerPlayer = this.gameState.players.find((p) => p.playerNumber === gameResult.winner);
    const loserPlayers = this.gameState.players.filter((p) => p.playerNumber !== gameResult.winner);

    // For now, just log the rating update
    debugLog(flagEnabled('RINGRIFT_TRACE_DEBUG'), 'Rating update needed for:', {
      winner: winnerPlayer?.username,
      losers: loserPlayers.map((p) => p.username),
    });
  }

  addSpectator(userId: string): boolean {
    if (!this.gameState.spectators.includes(userId)) {
      this.gameState.spectators.push(userId);
      return true;
    }
    return false;
  }

  removeSpectator(userId: string): boolean {
    const index = this.gameState.spectators.indexOf(userId);
    if (index !== -1) {
      this.gameState.spectators.splice(index, 1);
      return true;
    }
    return false;
  }

  pauseGame(): boolean {
    if (this.gameState.gameStatus === 'active') {
      this.gameState.gameStatus = 'paused';

      // Stop current player's timer
      this.stopPlayerTimer(this.gameState.currentPlayer);

      return true;
    }
    return false;
  }

  resumeGame(): boolean {
    if (this.gameState.gameStatus === 'paused') {
      this.gameState.gameStatus = 'active';

      // Restart current player's timer
      this.startPlayerTimer(this.gameState.currentPlayer);

      return true;
    }
    return false;
  }

  forfeitGame(playerNumber: string): {
    success: boolean;
    gameResult?: GameResult;
  } {
    const winner = this.gameState.players.find(
      (p) => p.playerNumber !== parseInt(playerNumber)
    )?.playerNumber;

    // Move-clock expiry is treated as a timeout, distinct from an explicit
    // resignation. This keeps GameResult.reason semantics aligned with
    // P18.3-1 decision/timeout spec while allowing rating logic to treat
    // timeouts and resignations equivalently if desired.
    return this.endGame(winner, 'timeout');
  }

  /**
   * Clean resignation initiated by a specific player seat. This is used for
   * explicit resign flows (for example, HTTP /games/:id/leave when active).
   */
  public resignPlayer(playerNumber: number): {
    success: boolean;
    gameResult: GameResult;
  } {
    const winner = this.gameState.players.find(
      (p) => p.playerNumber !== playerNumber
    )?.playerNumber;

    return this.endGame(winner, 'resignation');
  }

  /**
   * Abandonment induced by a specific player's disconnect / expired reconnect
   * window in a rated game. This credits a remaining opponent as the winner.
   */
  public abandonPlayer(playerNumber: number): {
    success: boolean;
    gameResult: GameResult;
  } {
    const winner = this.gameState.players.find(
      (p) => p.playerNumber !== playerNumber
    )?.playerNumber;

    return this.endGame(winner, 'abandonment');
  }

  /**
   * Abandonment without a specific winner (for example, unrated games or
   * scenarios where all human players have disconnected/expired).
   */
  public abandonGameAsDraw(): {
    success: boolean;
    gameResult: GameResult;
  } {
    return this.endGame(undefined, 'abandonment');
  }

  getValidMoves(_playerNumber: number): Move[] {
    const playerNumber = _playerNumber;

    // Only generate moves for the active player to keep server/UI
    // expectations clear.
    if (playerNumber !== this.gameState.currentPlayer) {
      return [];
    }

    // During an active chain_capture phase, valid moves are the explicit
    // continuation segments from the current chain position. Rather than
    // relying on any cached list, we always re-enumerate from the board
    // using the shared CaptureAggregate helper so that the options
    // exposed here stay in lockstep with the core rules and the targeted
    // triangle/zig-zag tests.
    if (this.gameState.currentPhase === 'chain_capture') {
      const state = this.chainCaptureState;

      // If for some reason the internal chain state has been cleared while
      // the the phase is still marked as chain_capture, treat this as "no legal
      // actions" rather than attempting to guess a continuation.
      if (!state) {
        return [];
      }

      // The capturing player for the entire chain is recorded on the
      // chainCaptureState. We rely on this rather than the caller's
      // playerNumber argument so that even if a test/UI accidentally
      // passes the wrong player, the engine still exposes the correct
      // follow-up segments for the active chain.
      const capturingPlayer = state.playerNumber;

      const followUpMoves = this.getCaptureOptionsFromPosition(
        state.currentPosition,
        capturingPlayer
      );

      // Keep availableMoves updated for any future PlayerChoice-based
      // integrations, but do not rely on it for correctness.
      state.availableMoves = followUpMoves;

      if (flagEnabled('RINGRIFT_TRACE_DEBUG')) {
        logger.debug('Chain capture debug', {
          component: 'GameEngine.getValidMoves',
          requestedPlayer: playerNumber,
          capturingPlayer,
          currentPhase: this.gameState.currentPhase,
          currentPosition: state.currentPosition,
          followUpCount: followUpMoves.length,
        });
      }

      if (followUpMoves.length === 0) {
        // No legal continuations remain; clear chain state and treat the
        // chain as resolved so callers do not see an interactive phase
        // with no legal actions.
        this.chainCaptureState = undefined;
        return [];
      }

      // Filter to typed capture moves for type-safe field access
      const typedCaptures = followUpMoves.filter(isCaptureMove);
      return typedCaptures.map((m) => ({
        ...m,
        // Re-label the shared overtaking_capture candidates as dedicated
        // continue_capture_segment moves for the unified Move model.
        type: 'continue_capture_segment' as const,
        // Ensure the move is attributed to the capturing player recorded
        // in the chain state, even if the caller passed a different
        // playerNumber by mistake.
        player: capturingPlayer,
        id:
          m.id && m.id.length > 0
            ? m.id.startsWith('capture-')
              ? m.id.replace('capture-', 'continue-')
              : m.id
            : `continue-${positionToString(m.from)}-${positionToString(
                m.captureTarget
              )}-${positionToString(m.to)}`,
      }));
    }

    // For automatic bookkeeping phases, expose the canonical decision moves
    // derived from the same helpers that drive line and territory
    // processing. This keeps the unified Move/GamePhase model complete even
    // though these phases are still usually resolved internally.
    if (this.gameState.currentPhase === 'line_processing') {
      const lineMoves = this.getValidLineProcessingMoves(playerNumber);

      // When a line reward elimination is pending (exact-length line or
      // Option 1 for overlength), surface explicit eliminate_rings_from_stack
      // Moves instead of further process_line / choose_line_option decisions.
      // We enumerate these directly from the current board rather than
      // borrowing the territory_processing phase so that the decision surface
      // matches the sandbox engine and shared move model.
      if (this.pendingLineRewardElimination) {
        const stacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);
        const eliminationMoves: Move[] = [];

        for (const stack of stacks) {
          if (stack.capHeight <= 0) {
            continue;
          }

          const posKey = positionToString(stack.position);

          eliminationMoves.push({
            id: `eliminate-${posKey}`,
            type: 'eliminate_rings_from_stack',
            player: playerNumber,
            to: stack.position,
            // RR-CANON-R022/R122: Line-reward eliminations remove exactly ONE ring
            // from the top of the chosen stack (not the full cap).
            eliminatedRings: [{ player: playerNumber, count: 1 }],
            eliminationContext: 'line',
            eliminationFromStack: {
              position: stack.position,
              capHeight: stack.capHeight,
              totalHeight: stack.stackHeight,
            },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: this.gameState.history.length + 1,
          } as Move);
        }

        if (eliminationMoves.length > 0) {
          return eliminationMoves;
        }

        // Defensive: if no eliminations can be generated (e.g. degenerate
        // positions with no caps), fall back to the base line decisions so
        // callers are never left without any legal actions.
        return lineMoves;
      }

      return lineMoves;
    }

    if (this.gameState.currentPhase === 'territory_processing') {
      const regionMoves = this.getValidTerritoryProcessingMoves(playerNumber);

      if (regionMoves.length > 0) {
        // When one or more regions are processable for this player and no
        // self-elimination decision is currently outstanding in this
        // territory_processing cycle, expose an explicit
        // 'skip_territory_processing' meta-move alongside the region
        // processing options. This lets humans and AI decline to process
        // any further regions this turn while still recording that choice
        // as a canonical Move.
        if (!this.pendingTerritorySelfElimination) {
          const moveNumber = this.gameState.history.length + 1;
          const skipMove: Move = {
            id: `skip-territory-${moveNumber}`,
            type: 'skip_territory_processing',
            player: playerNumber,
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move;
          return [...regionMoves, skipMove];
        }

        return regionMoves;
      }

      // When no region has been processed in the current
      // territory_processing cycle, do not surface elimination moves.
      // This prevents spurious "free" self-elimination from trivial
      // territory_processing entries that arise after every move in
      // move-driven mode.
      if (!this.pendingTerritorySelfElimination) {
        return [];
      }

      // In move-driven decision phases, once no eligible regions remain
      // for the moving player *and* a region has actually been processed
      // this cycle, surface explicit self-elimination decisions as
      // eliminate_rings_from_stack Moves. We delegate to
      // RuleEngine.getValidMoves so enumeration stays consistent with
      // the rules-level view.
      const tempTerritoryState: GameState = {
        ...this.gameState,
        currentPlayer: playerNumber,
        currentPhase: 'territory_processing',
      };

      const eliminationMoves = this.ruleEngine
        .getValidMoves(tempTerritoryState)
        .filter((m) => m.type === 'eliminate_rings_from_stack');

      return eliminationMoves;
    }

    // Base move generation comes from RuleEngine, which is responsible
    // for phase-specific legality (placement vs movement vs capture).
    let moves = this.ruleEngine.getValidMoves(this.gameState);

    // When the game is active and we're in an interactive phase but the
    // rules-level move generator returns no legal actions, fall back to the
    // shared resolveBlockedStateForCurrentPlayerForTesting safety net so
    // backend orchestrator-backed hosts never expose ACTIVE_NO_MOVES states.
    if (
      this.gameState.gameStatus === 'active' &&
      (this.gameState.currentPhase === 'ring_placement' ||
        this.gameState.currentPhase === 'movement' ||
        this.gameState.currentPhase === 'capture') &&
      moves.length === 0
    ) {
      // Record an orchestrator invariant-violation signal before we attempt
      // to resolve the blocked state. This should be extremely rare in
      // orchestrator-backed hosts and is used to back SLOs for
      // "ACTIVE_NO_MOVES" incidents in staging/production.
      getMetricsService().recordOrchestratorInvariantViolation('ACTIVE_NO_MOVES');

      const TRACE_DEBUG_ENABLED = flagEnabled('RINGRIFT_TRACE_DEBUG');

      const beforeState = this.getGameState();

      debugLog(
        TRACE_DEBUG_ENABLED,
        '[GameEngine.getValidMoves] resolving blocked interactive state',
        {
          currentPlayer: beforeState.currentPlayer,
          currentPhase: beforeState.currentPhase,
          gameStatus: beforeState.gameStatus,
          moveCount: moves.length,
          totalRingsEliminated: beforeState.totalRingsEliminated,
        }
      );

      this.resolveBlockedStateForCurrentPlayerForTesting();

      // If the resolver ended the game, there are no further moves to surface.
      if (this.gameState.gameStatus !== 'active') {
        return [];
      }

      const afterState = this.getGameState();

      debugLog(TRACE_DEBUG_ENABLED, '[GameEngine.getValidMoves] blocked state resolved', {
        previousPlayer: beforeState.currentPlayer,
        previousPhase: beforeState.currentPhase,
        currentPlayer: afterState.currentPlayer,
        currentPhase: afterState.currentPhase,
        gameStatus: afterState.gameStatus,
        totalRingsEliminated: afterState.totalRingsEliminated,
      });

      // Re-enter getValidMoves for the new active-player/phase selection. The
      // resolver guarantees either:
      //   - the game is now terminal, or
      //   - at least one player has a legal placement/movement/capture, in
      //     which case this recursive call will observe a non-empty move list.
      return this.getValidMoves(this.gameState.currentPlayer);
    }

    // Layer in the swap_sides meta-move (pie rule) for Player 2 when enabled.
    // This is treated as an additional choice alongside the underlying
    // placement/movement/capture actions, never as a replacement. We only
    // add it after the ACTIVE_NO_MOVES safeguard above so that parity and
    // termination invariants remain anchored on the core rules-level moves.
    if (this.shouldOfferSwapSidesMetaMove()) {
      const alreadyHasSwap = moves.some((m) => m.type === 'swap_sides');

      if (!alreadyHasSwap) {
        const moveNumber = this.gameState.moveHistory.length + 1;

        moves = [
          ...moves,
          {
            id: `swap_sides-${moveNumber}`,
            type: 'swap_sides',
            player: 2,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber,
          } as Move,
        ];
      }
    }

    // When a placement has occurred this turn, restrict movement/capture
    // options so that only the placed/updated stack may move.
    if (
      this.mustMoveFromStackKey &&
      (this.gameState.currentPhase === 'movement' || this.gameState.currentPhase === 'capture')
    ) {
      moves = moves.filter((m) => {
        const isMovementOrCaptureType =
          m.type === 'move_stack' ||
          m.type === 'overtaking_capture' ||
          m.type === 'continue_capture_segment';

        if (!isMovementOrCaptureType) {
          // Non-movement moves (e.g. future skip_placement) are not
          // constrained here.
          return true;
        }

        if (!m.from) {
          return false;
        }

        const fromKey = positionToString(m.from);
        return fromKey === this.mustMoveFromStackKey;
      });
    }

    return moves;
  }

  /**
   * Determine whether the current state should expose a swap_sides
   * meta-move (pie rule) for Player 2.
   *
   * Delegates to the shared shouldOfferSwapSides() helper from swapSidesHelpers.ts.
   */
  private shouldOfferSwapSidesMetaMove(): boolean {
    return shouldOfferSwapSides(this.gameState);
  }

  /**
   * Apply a canonical Move for the current phase selected by its stable
   * identifier.
   *
   * This helper is the primary bridge between PlayerChoice-based
   * transports (WebSocket, AI service) and the unified Move model: given
   * a moveId that was previously attached to a PlayerChoice option (for
   * example, line_order/region_order/ring_elimination), it will:
   *
   *   1. Re-enumerate getValidMoves(playerNumber) for the current phase,
   *   2. Locate the matching Move.id,
   *   3. Forward the Move payload through makeMove(), so all validation,
   *      history, S-invariant checks, and phase transitions apply
   *      exactly as for a direct Move submission.
   *
   * It is intentionally conservative and never mutates state when no
   * matching Move is found.
   */
  public async makeMoveById(
    playerNumber: number,
    moveId: string
  ): Promise<{
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  }> {
    // Only the active player may act.
    if (playerNumber !== this.gameState.currentPlayer) {
      return {
        success: false,
        error: `Player ${playerNumber} is not the active player`,
        gameState: this.getGameState(),
      };
    }

    const candidates = this.getValidMoves(playerNumber);

    if (candidates.length === 0) {
      return {
        success: false,
        error: `No valid moves available for player ${playerNumber} in phase ${this.gameState.currentPhase}`,
        gameState: this.getGameState(),
      };
    }

    const selected = candidates.find((m) => m.id === moveId);

    if (!selected) {
      return {
        success: false,
        error: `No valid move with id ${moveId} for player ${playerNumber} in phase ${this.gameState.currentPhase}`,
        gameState: this.getGameState(),
      };
    }

    if (selected.player !== playerNumber) {
      return {
        success: false,
        error: `Move ${moveId} belongs to player ${selected.player}, not ${playerNumber}`,
        gameState: this.getGameState(),
      };
    }

    // Delegate to the primary makeMove() entry point so all validation,
    // history, S-invariant checks, and phase/turn transitions are applied
    // in one place. The candidate already includes id/timestamp/moveNumber
    // fields, but makeMove() will generate fresh ones; we therefore strip
    // them from the payload.
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { id, timestamp, moveNumber, ...payload } = selected as Move;

    return this.makeMove(payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
  }

  /**
   * Determine whether the specified player currently has any "real"
   * actions available in the sense of R172: at least one legal ring
   * placement, non-capture movement, or overtaking capture. This helper
   * deliberately ignores bookkeeping-only moves such as skip_placement,
   * forced-elimination decisions, and line/territory processing moves.
   *
   * Delegates to the shared hasAnyRealAction helper.
   */
  private hasAnyRealActionForPlayer(playerNumber: number): boolean {
    return hasAnyRealAction(this.gameState, playerNumber, {
      hasPlacement: (pn) => enumeratePlacementPositions(this.gameState, pn).length > 0,
      hasMovement: (pn) => enumerateSimpleMovesForPlayer(this.gameState, pn).length > 0,
      hasCapture: (pn) => enumerateAllCaptureMovesAggregate(this.gameState, pn).length > 0,
    });
  }

  /**
   * Test-only helper: resolve a late-detected "blocked with no moves"
   * situation for the current player.
   *
   * In normal play, TurnEngine is responsible for ensuring that we
   * never enter an interactive phase (ring_placement, movement,
   * capture) for a player who has no legal placement, movement, or
   * capture available. If a test harness nonetheless observes
   * `gameStatus === 'active'` and `getValidMoves(currentPlayer).length
   * === 0` in an interactive phase, it may call this safety net to:
   *
   *   1. Apply forced elimination for any player who controls stacks
   *      but has no legal actions (Section 4.4 / compact rules 2.3),
   *   2. Skip over players who have no material at all,
   *   3. Stop once we either reach a player with at least one legal
   *      action (placement or movement/capture) or the game ends.
   *
   * This helper is intentionally conservative and only used from tests;
   * it does not create any new kinds of actions, it just applies the
   * same forced-elimination / skip semantics the TurnEngine would have
   * applied earlier if the blocked state had been detected on time.
   *
   * @deprecated Phase 4 legacy path — use orchestrator-backed TurnEngineAdapter
   * and shared turnLogic/advanceTurnAndPhase flows instead. This resolver will
   * be removed once all tests migrate to orchestrator-backed decision lifecycles.
   * See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * Ownership / deprecation:
   * - Still invoked from getValidMoves() as a last-resort production safety
   *   net for ACTIVE_NO_MOVES states.
   * - Used by a number of diagnostic/parity suites
   *   (for example ForcedEliminationAndStalemate, FullGameFlow.*, AI
   *   simulation debug tests, and RuleEngine.placement.shared tests) to
   *   exercise forced-elimination behaviour.
   * - New hosts should rely on TurnEngine/advanceTurnAndPhase to avoid
   *   entering blocked interactive phases in the first place, and only use
   *   this helper from tests.
   */
  public resolveBlockedStateForCurrentPlayerForTesting(): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const phase = this.gameState.currentPhase;
    if (phase !== 'ring_placement' && phase !== 'movement' && phase !== 'capture') {
      // Only resolve for interactive phases. Automatic bookkeeping
      // phases should be handled via stepAutomaticPhasesForTesting.
      return;
    }

    const players = this.gameState.players;
    const playerCount = players.length;

    // Compute an upper bound on how many forced-elimination steps we
    // might ever need: the total number of rings still in play (on the
    // board or in hand). Every forced elimination reduces this by at
    // least one, and S is globally non-decreasing, so this bound keeps
    // the resolver from looping forever even in badly broken states.
    const totalRingsRemaining = (() => {
      let total = 0;
      for (const stack of this.gameState.board.stacks.values()) {
        total += stack.stackHeight;
      }
      for (const p of players) {
        total += p.ringsInHand;
      }
      return total;
    })();

    const maxIterations = totalRingsRemaining + playerCount * 4;
    let iterations = 0;

    while (this.gameState.gameStatus === 'active' && iterations < maxIterations) {
      iterations++;

      // 1. First, look for ANY player who has at least one legal
      // placement, movement, or capture under the real rules. If we
      // find such a player, hand the turn to them and reseed the phase
      // so normal play can resume from that point.
      for (const player of players) {
        const playerNumber = player.playerNumber;

        const hasAnyPlacement = (() => {
          if (player.ringsInHand <= 0) {
            return false;
          }

          const tempPlacementState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'ring_placement',
          };

          const placementMoves = this.ruleEngine.getValidMoves(tempPlacementState);
          return placementMoves.some((m) => m.type === 'place_ring');
        })();

        const { hasMovement, hasCapture } = (() => {
          const tempMovementState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'movement',
          };

          const movementMoves = this.ruleEngine.getValidMoves(tempMovementState);
          const hasMovementLocal = movementMoves.some((m) => m.type === 'move_stack');

          const tempCaptureState: GameState = {
            ...this.gameState,
            currentPlayer: playerNumber,
            currentPhase: 'capture',
          };

          const captureMoves = enumerateAllCaptureMovesAggregate(tempCaptureState, playerNumber);
          const hasCaptureLocal = captureMoves.length > 0;

          return { hasMovement: hasMovementLocal, hasCapture: hasCaptureLocal };
        })();

        if (hasAnyPlacement || hasMovement || hasCapture) {
          // Hand the turn to this player and reseed the phase according
          // to whether they still have rings in hand and which kinds of
          // actions are actually available, mirroring the TurnEngine
          // phase flow:
          //   - Prefer ring_placement when placements are legal;
          //   - Otherwise prefer movement when non-capture moves exist;
          //   - Otherwise enter capture phase when only captures remain.
          this.gameState.currentPlayer = playerNumber;

          if (hasAnyPlacement && player.ringsInHand > 0) {
            this.gameState.currentPhase = 'ring_placement';
          } else if (hasMovement) {
            this.gameState.currentPhase = 'movement';
          } else {
            this.gameState.currentPhase = 'capture';
          }

          // Clear per-turn bookkeeping so the next explicit move is
          // treated as the start of a fresh turn.
          this.hasPlacedThisTurn = false;
          this.mustMoveFromStackKey = undefined;
          return;
        }
      }

      // 2. No player has any legal placement/movement/capture. If there
      // are no stacks left on the board, structural terminality has
      // been reached. At this point the compact rules define a
      // stalemate ladder where any rings remaining in hand are treated
      // as eliminated for tie-break purposes (hand → E).
      if (this.gameState.board.stacks.size === 0) {
        let handEliminations = 0;

        for (const player of players) {
          if (player.ringsInHand > 0) {
            const delta = player.ringsInHand;
            player.ringsInHand = 0;
            player.eliminatedRings += delta;
            handEliminations += delta;

            if (!this.gameState.board.eliminatedRings[player.playerNumber]) {
              this.gameState.board.eliminatedRings[player.playerNumber] = 0;
            }
            this.gameState.board.eliminatedRings[player.playerNumber] += delta;
          }
        }

        if (handEliminations > 0) {
          this.gameState.totalRingsEliminated += handEliminations;
        }

        const endCheck = this.ruleEngine.checkGameEnd(this.gameState);
        if (endCheck.isGameOver) {
          this.endGame(endCheck.winner, endCheck.reason || 'structural_stalemate');
        }
        return;
      }

      // 3. Global forced elimination pass: walk players in turn order
      // starting from the current player and eliminate a cap from the
      // first player who still controls stacks. This mirrors the rules
      // text: when everyone is globally blocked but stacks remain,
      // successive forced eliminations must eventually resolve the
      // stalemate until no stacks are left. We delegate the actual
      // elimination target selection and bookkeeping to the shared
      // applyForcedEliminationForPlayer helper so this resolver stays
      // aligned with canonical forced-elimination semantics.
      const currentIndex = players.findIndex(
        (p) => p.playerNumber === this.gameState.currentPlayer
      );
      let eliminatedThisIteration = false;

      for (let offset = 0; offset < playerCount; offset++) {
        const idx = (currentIndex + offset) % playerCount;
        const playerNumber = players[idx].playerNumber;
        const stacksForPlayer = this.boardManager.getPlayerStacks(
          this.gameState.board,
          playerNumber
        );

        if (stacksForPlayer.length === 0) {
          continue;
        }

        const outcome = applyForcedEliminationForPlayer(this.gameState, playerNumber);
        if (!outcome) {
          // No forced-elimination action available for this player under
          // the formal RR-CANON preconditions; continue scanning other
          // players rather than attempting an ad-hoc elimination.
          continue;
        }

        this.gameState = outcome.nextState;
        eliminatedThisIteration = true;

        const endCheck = this.ruleEngine.checkGameEnd(this.gameState);
        if (endCheck.isGameOver) {
          this.endGame(endCheck.winner, endCheck.reason || 'forced_elimination');
          return;
        }

        // After a forced elimination, make that player the current
        // player in movement phase with fresh per-turn state. The next
        // loop iteration will re-check for available actions for all
        // players from this new board state.
        this.gameState.currentPlayer = playerNumber;
        this.gameState.currentPhase = 'movement';
        this.hasPlacedThisTurn = false;
        this.mustMoveFromStackKey = undefined;
        break;
      }

      if (!eliminatedThisIteration) {
        // Safety: if we somehow failed to eliminate any rings even
        // though stacks remain, bail out rather than spin forever. The
        // caller will continue to treat this as a diagnostic failure.
        return;
      }
    }

    // If we exit because maxIterations was reached while the game is
    // still active, leave the state as-is. The AI simulation harness
    // will continue to treat this as a non-terminating diagnostic
    // failure, but we have avoided an infinite loop inside the engine.
  }

  /**
   * Test-only helper: advance through automatic phases (line_processing
   * and territory_processing) without requiring an explicit player move.
   *
   * This is used by AI simulation/diagnostic harnesses so they do not
   * treat these internal bookkeeping phases as "no legal move" stalls.
   *
   * @deprecated Phase 4 legacy path — use orchestrator-backed TurnEngineAdapter
   * and shared decision lifecycle (see docs/archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md)
   * instead. This method will be removed once all tests migrate to
   * orchestrator-backed flows. See Wave 5.4 in TODO.md for deprecation timeline.
   *
   * DO NOT REMOVE - still called internally by makeMove() legacy path and in parity tests.
   * See P20.3-2 for deprecation timeline.
   */
  public async stepAutomaticPhasesForTesting(): Promise<void> {
    // First, handle line_processing and territory_processing automatic phases
    while (
      this.gameState.gameStatus === 'active' &&
      (this.gameState.currentPhase === 'line_processing' ||
        this.gameState.currentPhase === 'territory_processing')
    ) {
      const moves = this.getValidMoves(this.gameState.currentPlayer);
      if (moves.length === 0) {
        // No moves available in an automatic phase. This implies there is
        // no work to do (no lines, no disconnected regions). We must
        // manually advance the phase/game to prevent getting stuck.
        this.advanceGame();
        continue;
      }

      // Decision moves (process_line, choose_territory_option, etc.) should
      // be submitted explicitly by the client/AI and recorded in the trace.
      // Do NOT auto-apply them here, as that would cause the backend to
      // advance past phases that should be represented as discrete moves.
      return;
    }
  }
}
