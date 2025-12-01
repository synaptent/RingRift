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
  getPathPositions,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
  countRingsInPlayForPlayer,
  filterProcessableTerritoryRegions,
  applyTerritoryRegion,
  canProcessTerritoryRegion,
  enumerateProcessTerritoryRegionMoves,
  applyForcedEliminationForPlayer,
  applyProcessTerritoryRegionDecision,
  applyEliminateRingsFromStackDecision,
  enumerateProcessLineMoves,
  enumerateChooseLineRewardMoves,
  applyProcessLineDecision,
  applyChooseLineRewardDecision,
  findLinesForPlayer,
  // Canonical non-capture movement mutation (TS SSOT)
  applySimpleMovement as applySimpleMovementAggregate,
  // Capture aggregate helpers (canonical capture mutation + global enumeration)
  enumerateAllCaptureMoves as enumerateAllCaptureMovesAggregate,
  applyCapture as applyCaptureAggregate,
  // Canonical placement mutation (TS SSOT)
  applyPlacementMove as applyPlacementMoveAggregate,
} from '../../shared/engine';
import type {
  GameResult,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  PlayerChoiceResponseFor,
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import {
  getChainCaptureContinuationInfo,
  ChainCaptureState,
  updateChainCaptureStateAfterCapture,
} from '../../shared/engine/aggregates/CaptureAggregate';
import { config } from '../config';
import {
  PerTurnState,
  advanceGameForCurrentPlayer,
  updatePerTurnStateAfterMove as updatePerTurnStateAfterMoveTurn,
  TurnEngineDeps,
  TurnEngineHooks,
} from './turn/TurnEngine';
import { orchestratorRollout } from '../services/OrchestratorRolloutService';
import { getMetricsService } from '../services/MetricsService';

// ═══════════════════════════════════════════════════════════════════════════
// Orchestrator Adapter Feature Flag
// ═══════════════════════════════════════════════════════════════════════════
// Check config for feature flag to enable orchestrator adapter by default
const ORCHESTRATOR_ADAPTER_ENABLED_BY_CONFIG = config.featureFlags.orchestrator.adapterEnabled;
import {
  TurnEngineAdapter,
  StateAccessor,
  DecisionHandler,
  EventEmitter as AdapterEventEmitter,
  AdapterMoveResult,
} from './turn/TurnEngineAdapter';

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

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;

declare const clearTimeout: (timer: any) => void;

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
  private moveTimers: Map<number, any> = new Map();
  private interactionManager: PlayerInteractionManager | undefined;
  private debugCheckpointHook: ((label: string, state: GameState) => void) | undefined;
  /**
   * When true, line and territory processing are expressed as explicit
   * canonical decision moves (process_line, choose_line_reward,
   * process_territory_region, eliminate_rings_from_stack) rather than
   * being resolved purely via PlayerChoice flows in
   * processAutomaticConsequences. This flag is opt-in so legacy tests
   * and tools can continue to rely on the older automatic behaviour
   * while WebSocket/AI integrations migrate to the unified Move model.
   */
  private useMoveDrivenDecisionPhases: boolean = false;
  /**
   * When true, makeMove() delegates to TurnEngineAdapter which wraps the
   * canonical shared orchestrator (processTurnAsync). This flag enables
   * gradual migration of rules logic from GameEngine to the shared engine.
   *
   * Phase 3 Rules Engine Consolidation - see:
   * - docs/drafts/RULES_ENGINE_CONSOLIDATION_DESIGN.md
   * - docs/drafts/PHASE3_ADAPTER_MIGRATION_REPORT.md
   *
   * Phase 5: Now reads default from config.featureFlags.orchestratorAdapterEnabled
   */
  private useOrchestratorAdapter: boolean = ORCHESTRATOR_ADAPTER_ENABLED_BY_CONFIG;
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
   * Host-internal metadata for last-player-standing (R172) detection.
   * These fields track, per round, which players had any real actions
   * available (placement, non-capture movement, or overtaking capture)
   * at the start of their most recent interactive turn. They are kept
   * off of GameState so snapshots and wire formats remain unchanged.
   */
  private lpsRoundIndex: number = 0;
  private lpsCurrentRoundActorMask: Map<number, boolean> = new Map();
  private lpsCurrentRoundFirstPlayer: number | null = null;
  private lpsExclusivePlayerForCompletedRound: number | null = null;

  /**
   * Internal helper flag for the 2-player pie rule (swap_sides).
   * We rely primarily on moveHistory shape for gating, but this flag
   * can be used for additional future diagnostics if needed.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private swapSidesApplied: boolean = false;

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true,
    interactionManager?: PlayerInteractionManager,
    rngSeed?: number,
    rulesOptions?: GameState['rulesOptions']
  ) {
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    this.interactionManager = interactionManager;
 
    const config = BOARD_CONFIGS[boardType];
 
    this.gameState = {
      id: gameId,
      boardType,
      ...(typeof rngSeed === 'number' ? { rngSeed } : {}),
      board: this.boardManager.createBoard(),
      players: players.map((p, index) => ({
        ...p,
        playerNumber: index + 1,
        timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
        isReady: p.type === 'ai', // AI players are always ready
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
      totalRingsInPlay: config.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor((config.ringsPerPlayer * players.length) / 2) + 1,
      territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1,
    };
 
    // Internal no-op hook to keep selected helpers referenced so that
    // ts-node/TypeScript with noUnusedLocals can compile the server in
    // dev without stripping them. This has no behavioural effect.
    this._debugUseInternalHelpers();
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
    const state = this.gameState;
 
    // Config gating: swap_sides must be explicitly enabled for this game.
    // When rulesOptions is absent or swapRuleEnabled === false, treat the
    // pie rule as disabled and reject the meta-move.
    if (!state.rulesOptions?.swapRuleEnabled) {
      return {
        success: false,
        error: 'swap_sides is disabled for this game',
        gameState: this.getGameState(),
      };
    }
 
    // Basic gating: only active 2-player games, Player 2, their turn.
    if (state.gameStatus !== 'active') {
      return {
        success: false,
        error: 'swap_sides is only available in active games',
        gameState: this.getGameState(),
      };
    }
 
    if (state.players.length !== 2) {
      return {
        success: false,
        error: 'swap_sides is only defined for 2-player games',
        gameState: this.getGameState(),
      };
    }

    if (state.currentPlayer !== playerNumber) {
      return {
        success: false,
        error: 'Only the active player may request swap_sides',
        gameState: this.getGameState(),
      };
    }

    if (playerNumber !== 2) {
      return {
        success: false,
        error: 'Only Player 2 may request swap_sides',
        gameState: this.getGameState(),
      };
    }

    // Must occur at the start of Player 2's first interactive turn:
    // - At least one move from Player 1 exists.
    // - No prior moves from Player 2.
    // - No prior swap_sides move.
    const hasP1Move = state.moveHistory.some((m) => m.player === 1);
    const hasP2Move = state.moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides');
    const hasSwapMove = state.moveHistory.some((m) => m.type === 'swap_sides');

    if (!hasP1Move || hasP2Move || hasSwapMove) {
      return {
        success: false,
        error: 'swap_sides is only available immediately after Player 1’s first turn',
        gameState: this.getGameState(),
      };
    }

    // Restrict to interactive phases (no swapping mid-line/territory processing).
    if (
      state.currentPhase === 'line_processing' ||
      state.currentPhase === 'territory_processing'
    ) {
      return {
        success: false,
        error: 'swap_sides is only available at the start of an interactive turn',
        gameState: this.getGameState(),
      };
    }

    // Capture a full snapshot for history before we mutate players.
    const beforeStateForHistory = this.getGameState();

    // Swap the identities of players in seats 1 and 2 while keeping the
    // numeric playerNumber identifiers stable for board geometry. This
    // effectively reassigns which user controls each colour/seat; board
    // RingStack.controllingPlayer values and other colour indices remain
    // unchanged. Time budgets and on-seat statistics (ringsInHand,
    // eliminatedRings, territorySpaces, etc.) move with the seat so that
    // the player who takes over the opening also inherits its remaining
    // clock, matching pie-rule fairness semantics.
    const currentPlayers = this.gameState.players;
    const p1 = currentPlayers.find((p) => p.playerNumber === 1);
    const p2 = currentPlayers.find((p) => p.playerNumber === 2);

    if (!p1 || !p2) {
      return {
        success: false,
        error: 'swap_sides failed: missing players for seats 1 or 2',
        gameState: this.getGameState(),
      };
    }

    // NOTE: We assert Player[] here because we are only reassigning identity
    // fields on existing Player objects. All required engine fields remain
    // intact, but exactOptionalPropertyTypes makes it difficult for the
    // compiler to infer that when swapping optional metadata like rating
    // and AI configuration between seats.
    const playersCopy = currentPlayers.map((p) => {
      if (p.playerNumber === 1) {
        return {
          ...p,
          id: p2.id,
          username: p2.username,
          type: p2.type,
          rating: p2.rating,
          aiDifficulty: p2.aiDifficulty,
          aiProfile: p2.aiProfile,
        };
      }
      if (p.playerNumber === 2) {
        return {
          ...p,
          id: p1.id,
          username: p1.username,
          type: p1.type,
          rating: p1.rating,
          aiDifficulty: p1.aiDifficulty,
          aiProfile: p1.aiProfile,
        };
      }
      return p;
    }) as Player[];

    this.gameState = {
      ...this.gameState,
      players: playersCopy,
    };

    this.swapSidesApplied = true;

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

  private debugCheckpoint(label: string): void {
    if (this.debugCheckpointHook) {
      this.debugCheckpointHook(label, this.getGameState());
    }
  }

  /**
   * Enable Move-driven decision phases (line_processing and
   * territory_processing) so that line/territory decisions are
   * expressed as explicit canonical Moves chosen via getValidMoves
   * rather than being resolved internally via PlayerChoice flows in
   * processAutomaticConsequences. This is primarily used by the
   * WebSocket/AI integration layer.
   */
  public enableMoveDrivenDecisionPhases(): void {
    this.useMoveDrivenDecisionPhases = true;
  }

  /**
   * Enable delegation to the shared orchestrator via TurnEngineAdapter.
   * When enabled, makeMove() delegates rules processing to the canonical
   * shared engine, keeping GameEngine as a thin adapter for backend concerns.
   *
   * This is part of Phase 3 Rules Engine Consolidation.
   */
  public enableOrchestratorAdapter(): void {
    this.useOrchestratorAdapter = true;
    // Orchestrator adapter requires move-driven decision phases
    this.useMoveDrivenDecisionPhases = true;
  }

  /**
   * Disable delegation to the shared orchestrator adapter and use the
   * legacy GameEngine turn processing pipeline for this instance.
   *
   * This is primarily used by orchestrator rollout logic to keep some
   * sessions on the legacy path even when the global feature flag is on.
   */
  public disableOrchestratorAdapter(): void {
    this.useOrchestratorAdapter = false;
  }

  /**
   * Check if orchestrator adapter is enabled.
   */
  public isOrchestratorAdapterEnabled(): boolean {
    return this.useOrchestratorAdapter;
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

        // Update stacks Map in-place using the orchestrator's board as source.
        existingBoard.stacks.clear();
        for (const [key, stack] of incomingBoard.stacks) {
          existingBoard.stacks.set(key, stack);
        }

        // Update markers Map in-place
        existingBoard.markers.clear();
        for (const [key, marker] of incomingBoard.markers) {
          existingBoard.markers.set(key, marker);
        }

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

        // Update territories Map in-place
        existingBoard.territories.clear();
        for (const [key, territory] of incomingBoard.territories) {
          existingBoard.territories.set(key, territory);
        }

        // Update formedLines array in-place
        existingBoard.formedLines.length = 0;
        existingBoard.formedLines.push(...incomingBoard.formedLines);

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
        const TRACE_DEBUG_ENABLED =
          typeof process !== 'undefined' &&
          !!(process as any).env &&
          ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

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
            const choice: RingEliminationChoice = {
              id: generateUUID(),
              gameId: this.gameState.id,
              playerNumber: decision.player,
              type: 'ring_elimination',
              prompt: 'Choose which stack to eliminate from',
              options: eliminationMoves.map((move) => {
                const pos = move.to as Position;
                const stack = this.boardManager.getStack(pos, this.gameState.board);
                const capHeight =
                  (move.eliminationFromStack && move.eliminationFromStack.capHeight) ||
                  (stack ? stack.capHeight : 1);
                const totalHeight =
                  (move.eliminationFromStack && move.eliminationFromStack.totalHeight) ||
                  (stack ? stack.stackHeight : capHeight || 1);

                return {
                  stackPosition: pos,
                  capHeight,
                  totalHeight,
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

            if (TRACE_DEBUG_ENABLED) {
              // eslint-disable-next-line no-console
              console.log('[GameEngine.DecisionHandler] resolved elimination_target via choice', {
                player: decision.player,
                optionCount: eliminationMoves.length,
                selectedMoveId,
              });
            }

            return chosen;
          }
        }

        // For core move-driven decision types where the backend should behave
        // like an AI host in the absence of a real PlayerInteractionManager,
        // auto-select the first available option. This mirrors the
        // TurnEngineAdapter.autoSelectForAI behaviour and preserves the
        // semantics of the shared orchestrator while avoiding HOST_REJECTED_MOVE
        // errors in soak/diagnostic runs.
        const autoResolvableTypes: string[] = ['elimination_target', 'line_order', 'region_order'];

        if (autoResolvableTypes.includes(decision.type)) {
          // Rare defensive case: an elimination_target / ordering decision with
          // zero options indicates a structurally inconsistent state from the
          // orchestrator's perspective (for example, a pending self-elimination
          // but no eligible stacks). For soak/diagnostic hosts we treat this as
          // a no-op decision rather than a hard HOST_REJECTED_MOVE error.
          if (decision.options.length === 0) {
            if (decision.type === 'elimination_target') {
              if (TRACE_DEBUG_ENABLED) {
                // eslint-disable-next-line no-console
                console.log(
                  '[GameEngine.DecisionHandler] elimination_target decision with no options; returning no-op elimination move',
                  {
                    type: decision.type,
                    player: decision.player,
                  }
                );
              }

              const noopMove: Move = {
                id: `noop-eliminate-${Date.now()}`,
                type: 'eliminate_rings_from_stack',
                player: decision.player,
                // Use a harmless sentinel coordinate that is extremely unlikely
                // to host a real stack; applyEliminateRingsFromStackDecision will
                // treat this as a no-op when no stack exists at this position.
                to: { x: 0, y: 0 },
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

          if (TRACE_DEBUG_ENABLED) {
            // eslint-disable-next-line no-console
            console.log('[GameEngine.DecisionHandler] auto-resolving decision', {
              type: decision.type,
              player: decision.player,
              optionCount: decision.options.length,
            });
          }

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
        console.error('[GameEngine.processMoveViaAdapter] Orchestrator rejected move:', {
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

    void this.performOvertakingCapture;
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
    const TRACE_DEBUG_ENABLED =
      typeof process !== 'undefined' &&
      !!(process as any).env &&
      ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

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
        // eslint-disable-next-line no-console
        console.log('[GameEngine.appendHistoryEntry] STRICT_S_INVARIANT_DECREASE', {
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
        // This is emitted only when TRACE_DEBUG is enabled and is intended as a
        // rare production/staging signal for invariant SLOs.
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
        // eslint-disable-next-line no-console
        console.log(
          '[GameEngine.appendHistoryEntry] TOTAL_RINGS_ELIMINATED_DECREASED',
          {
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
          }
        );
        getMetricsService().recordOrchestratorInvariantViolation(
          'TOTAL_RINGS_ELIMINATED_DECREASED'
        );
      }

      // Log when board-level and player-level elimination accounting diverge
      // even if S itself does not decrease. This helps diagnose cases where
      // one view of S is monotone while the other is not.
      if (eliminatedFromBoardAfter !== eliminatedFromPlayersAfter) {
        // eslint-disable-next-line no-console
        console.log('[GameEngine.appendHistoryEntry] S-elimination bookkeeping mismatch', {
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
        // eslint-disable-next-line no-console
        console.log('[GameEngine.appendHistoryEntry] S-invariant debug mismatch', {
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
        move.type === 'move_ring' ||
        move.type === 'build_stack' ||
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
    // ORCHESTRATOR ADAPTER DELEGATION (Phase A – Backend orchestrator-only)
    // ═══════════════════════════════════════════════════════════════════════
    // In Phase A the shared orchestrator, wrapped by TurnEngineAdapter, is
    // the only production backend turn-processing path. The legacy
    // RuleEngine-based pipeline is reserved for test harnesses and
    // diagnostics only.
    //
    // Behaviour:
    // - In all non-test environments (config.isTest === false), always
    //   delegate to the adapter regardless of the internal flag.
    // - In tests (config.isTest === true), allow explicit opt-out via
    //   disableOrchestratorAdapter() so legacy helpers remain available for
    //   diagnostics and archived scenarios.
    const isTestEnv = config.isTest === true;
    if (!isTestEnv || this.useOrchestratorAdapter) {
      return this.processMoveViaAdapter(move);
    }

    // Legacy backend path (RuleEngine-based). This is now restricted to
    // test/diagnostic usage and is not used by production GameSession /
    // WebSocket entrypoints. We still record usage for rollout diagnostics.
    let legacyMoveRecorded = false;
    const recordLegacyMove = (outcome: 'success' | 'error') => {
      if (!legacyMoveRecorded) {
        getMetricsService().recordOrchestratorMove('legacy', outcome);
        legacyMoveRecorded = true;
      }
    };

    // Capture a pre-move snapshot for history/event-sourcing. This uses the
    // public getGameState() so callers and history entries share the same
    // cloned view of board/maps.
    const beforeStateForHistory = this.getGameState();

    // Validate the move at the rules level
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1,
    };

    const isDecisionPhaseMove =
      fullMove.type === 'process_line' ||
      fullMove.type === 'choose_line_reward' ||
      fullMove.type === 'process_territory_region' ||
      fullMove.type === 'eliminate_rings_from_stack';

    // For decision-phase moves in move-driven mode, validation is performed
    // against a phase-normalised view of the state so that
    // eliminate_rings_from_stack moves generated from line rewards are
    // accepted using the same rules-level contract as territory-origin
    // eliminations. This keeps RuleEngine as the single source of truth for
    // elimination legality while allowing GameEngine.getValidMoves to surface
    // line-reward eliminations in the line_processing phase.
    const validationState: GameState =
      isDecisionPhaseMove &&
      this.useMoveDrivenDecisionPhases &&
      this.gameState.currentPhase === 'line_processing' &&
      fullMove.type === 'eliminate_rings_from_stack'
        ? {
            ...this.gameState,
            currentPhase: 'territory_processing',
          }
        : this.gameState;

    const validation = this.ruleEngine.validateMove(fullMove, validationState);
    if (!validation) {
      recordLegacyMove('error');
      return {
        success: false,
        error: 'Invalid move',
      };
    }

    // Decision-phase moves (line_processing / territory_processing) are
    // applied directly via internal helpers and do not go through the
    // normal placement/movement/capture pipeline.
    if (
      fullMove.type === 'process_line' ||
      fullMove.type === 'choose_line_reward' ||
      fullMove.type === 'process_territory_region' ||
      fullMove.type === 'eliminate_rings_from_stack'
    ) {
      await this.applyDecisionMove(fullMove);

      // After a decision move, if we find ourselves in an automatic
      // bookkeeping phase (line_processing / territory_processing) with
      // **no** canonical decision moves available for the active player,
      // advance the turn via the shared TurnEngine and drain any
      // remaining automatic phases. This mirrors the sandbox turn
      // engine, which never exposes a decision phase with zero legal
      // actions; callers always see either an interactive phase with at
      // least one move or a terminal game state.
      if (
        this.gameState.gameStatus === 'active' &&
        (this.gameState.currentPhase === 'line_processing' ||
          this.gameState.currentPhase === 'territory_processing')
      ) {
        const autoPhaseMoves = this.getValidMoves(this.gameState.currentPlayer);
        if (autoPhaseMoves.length === 0) {
          this.advanceGame();
          await this.stepAutomaticPhasesForTesting();
        }
      }

      // After applying a decision move (and draining any empty automatic
      // phases), re-check victory conditions. This mirrors the
      // post-move pipeline used for normal placement/movement/capture
      // actions so that eliminations caused by line processing or territory
      // disconnection can immediately end the game (FAQ Q6 / §13 Ring
      // Elimination Victory), even when those effects are expressed as explicit
      // decision moves rather than automatic post-move processing.
      //
      // We intentionally reuse the same RuleEngine.checkGameEnd entry point
      // used by the main move pipeline; endGame itself is idempotent for a
      // given terminal outcome, so calling it multiple times for the same
      // winner/reason is safe.
      let decisionGameResult: GameResult | undefined;
      const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
      if (gameEndCheck.isGameOver) {
        const endResult = this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');
        decisionGameResult = endResult.gameResult;
      }

      // After decision moves that advance through automatic phases (e.g.
      // territory processing completing via eliminate_rings_from_stack),
      // we must also check for last-player-standing victory (R172). This
      // mirrors the LPS check performed after regular moves and ensures
      // that LPS victory fires at the same point as the sandbox engine.
      if (!decisionGameResult && this.gameState.gameStatus === 'active') {
        const phase = this.gameState.currentPhase;
        if (
          phase === 'ring_placement' ||
          phase === 'movement' ||
          phase === 'capture' ||
          phase === 'chain_capture'
        ) {
          const lpsResult = this.runLpsCheckForCurrentInteractiveTurn();
          if (lpsResult) {
            decisionGameResult = lpsResult;
          }
        }
      }

      // Record a structured history entry for the decision so parity/debug
      // tooling sees the same trace format as for other canonical moves.
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      const response: {
        success: boolean;
        gameState: GameState;
        gameResult?: GameResult;
      } = {
        success: true,
        gameState: this.getGameState(),
      };

      if (decisionGameResult) {
        response.gameResult = decisionGameResult;
      }

      return response;
    }

    // Defensive runtime check: for movement and capture moves, verify that
    // the source stack actually exists and is controlled by this player.
    // This catches stale moves where validation passed on an earlier state
    // but the stack is now missing (e.g., due to intervening auto phases or
    // concurrent state changes).
    const isMovementOrCaptureType =
      fullMove.type === 'move_stack' ||
      fullMove.type === 'move_ring' ||
      fullMove.type === 'overtaking_capture' ||
      fullMove.type === 'continue_capture_segment';

    if (isMovementOrCaptureType && fullMove.from) {
      const sourceStack = this.boardManager.getStack(fullMove.from, this.gameState.board);
      if (!sourceStack) {
        console.error(
          '[GameEngine.makeMove] S-invariant violation: move validated but source stack missing',
          {
            moveType: fullMove.type,
            player: fullMove.player,
            from: positionToString(fullMove.from),
            to: fullMove.to ? positionToString(fullMove.to) : undefined,
            currentPhase: this.gameState.currentPhase,
            currentPlayer: this.gameState.currentPlayer,
          }
        );
        recordLegacyMove('error');
        return {
          success: false,
          error: 'Source stack no longer exists',
        };
      }
      if (sourceStack.controllingPlayer !== fullMove.player) {
        console.error(
          '[GameEngine.makeMove] S-invariant violation: move validated but source stack not controlled by player',
          {
            moveType: fullMove.type,
            player: fullMove.player,
            sourceControllingPlayer: sourceStack.controllingPlayer,
            from: positionToString(fullMove.from),
            to: fullMove.to ? positionToString(fullMove.to) : undefined,
            currentPhase: this.gameState.currentPhase,
            currentPlayer: this.gameState.currentPlayer,
          }
        );
        recordLegacyMove('error');
        return {
          success: false,
          error: 'Source stack is not controlled by this player',
        };
      }
    }

    // Capture context needed for chain state bookkeeping (cap height, etc.)
    let capturedCapHeight = 0;
    if (
      (fullMove.type === 'overtaking_capture' || fullMove.type === 'continue_capture_segment') &&
      fullMove.captureTarget
    ) {
      const targetStack = this.boardManager.getStack(fullMove.captureTarget, this.gameState.board);
      capturedCapHeight = targetStack ? targetStack.capHeight : 0;
    }

    // Stop current player's timer while we process the move
    this.stopPlayerTimer(this.gameState.currentPlayer);

    // Apply the move to the board state. We intentionally ignore the
    // granular result here; post-move consequences (lines, territory,
    // etc.) are processed separately based on the updated gameState.
    this.applyMove(fullMove);
    this.debugCheckpoint('after-applyMove');

    // Add move to history
    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = new Date();

    // Update per-turn placement/movement bookkeeping so that subsequent
    // phases (movement/capture) can enforce must-move constraints.
    this.updatePerTurnStateAfterMove(fullMove);

    // If this move is a capture segment (either the initial overtaking_capture
    // or a follow-up continue_capture_segment), update or start the chain
    // capture state and determine whether additional capture segments are
    // available from the new landing position. Chain continuation is now
    // driven by explicit continue_capture_segment moves chosen by the
    // player/AI during the dedicated 'chain_capture' phase rather than via
    // internal PlayerChoice callbacks.
    let chainContinuationAvailable = false;

    if (fullMove.type === 'overtaking_capture' || fullMove.type === 'continue_capture_segment') {
      this.updateChainCaptureStateAfterCaptureInternal(fullMove, capturedCapHeight);

      const state = this.chainCaptureState;
      const currentPlayer = this.gameState.currentPlayer;

      if (state && state.playerNumber === currentPlayer) {
        const followUpMoves = this.getCaptureOptionsFromPosition(
          state.currentPosition,
          currentPlayer
        );
        state.availableMoves = followUpMoves;

        if (followUpMoves.length > 0) {
          // At least one additional capture segment is available. Enter
          // the interactive chain_capture phase so the same player can
          // choose among the available follow-up segments via
          // continue_capture_segment moves.
          this.gameState.currentPhase = 'chain_capture';
          chainContinuationAvailable = true;
        } else {
          // Chain is exhausted; clear state and reset phase to 'capture'
          // so that advanceGame() can properly advance the turn. Without
          // this reset, the phase would remain 'chain_capture' but with
          // no chainCaptureState, causing getValidMoves() to return empty
          // and leaving the game stuck.
          this.chainCaptureState = undefined;
          this.gameState.currentPhase = 'capture';
        }
      } else {
        // Defensive: if we somehow lack a chain state after a capture
        // segment, clear it, reset phase, and treat this as a standalone capture.
        this.chainCaptureState = undefined;
        if (this.gameState.currentPhase === 'chain_capture') {
          this.gameState.currentPhase = 'capture';
        }
      }
    } else {
      // Any non-capture move clears any stale chain state (defensive safety).
      this.chainCaptureState = undefined;
    }

    // When a capture chain is still in progress after this move, skip
    // automatic consequences and phase advancement. The active player
    // remains the same and must now choose a continue_capture_segment
    // move from getValidMoves().
    if (chainContinuationAvailable) {
      // Restart the active player's timer for the next interactive decision.
      this.startPlayerTimer(this.gameState.currentPlayer);

      // Record a structured history entry for this capture segment.
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      recordLegacyMove('success');

      return {
        success: true,
        gameState: this.getGameState(),
      };
    }

    // When Move-driven decision phases are enabled, expose pending
    // line/territory decisions as explicit canonical Moves instead of
    // resolving them internally. This keeps backend WebSocket/AI flows
    // aligned with the sandbox engine and ensures every decision is
    // visible in the move history.
    if (this.useMoveDrivenDecisionPhases) {
      const currentPlayer = this.gameState.currentPlayer;

      // 1) Line-processing decisions: if any lines exist for the
      // current player, enter the dedicated line_processing phase so
      // clients/AI can submit process_line / choose_line_reward Moves
      // via getValidMoves.
      const lineDecisionMoves = this.getValidLineProcessingMoves(currentPlayer);
      if (lineDecisionMoves.length > 0) {
        this.gameState.currentPhase = 'line_processing';

        // Restart the active player's timer for the upcoming
        // interactive decision.
        this.startPlayerTimer(this.gameState.currentPlayer);

        // Record a structured history entry for the move that created
        // this decision state.
        this.appendHistoryEntry(beforeStateForHistory, fullMove);

        recordLegacyMove('success');

        return {
          success: true,
          gameState: this.getGameState(),
        };
      }

      // 2) Territory-processing decisions: if no lines remain but
      // disconnected regions exist for the current player, enter
      // territory_processing so process_territory_region Moves can be
      // chosen explicitly. Explicit self-elimination decisions
      // (eliminate_rings_from_stack) are still surfaced later, once a
      // region has been processed and the engine is already in the
      // territory_processing phase.
      const territoryRegionMoves = this.getValidTerritoryProcessingMoves(currentPlayer);

      if (territoryRegionMoves.length > 0) {
        this.gameState.currentPhase = 'territory_processing';

        this.startPlayerTimer(this.gameState.currentPlayer);
        this.appendHistoryEntry(beforeStateForHistory, fullMove);

        recordLegacyMove('success');

        return {
          success: true,
          gameState: this.getGameState(),
        };
      }
    }

    // Fallback: legacy automatic-consequence flow. This path remains
    // the default when Move-driven decision phases are disabled and is
    // also used when no line/territory decisions are available after
    // the move.

    // Process automatic consequences (line formations, territory, etc.) only
    // after the full move (including any mandatory chain) has resolved.
    const TRACE_DEBUG_ENABLED_LOCAL =
      typeof process !== 'undefined' &&
      !!(process as any).env &&
      ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

    if (TRACE_DEBUG_ENABLED_LOCAL) {
      const allBoardStackKeys = Array.from(this.gameState.board.stacks.keys());
      const stacksByPlayer: { [player: number]: string[] } = {};
      for (const [key, stack] of this.gameState.board.stacks.entries()) {
        const owner = stack.controllingPlayer;
        if (!stacksByPlayer[owner]) {
          stacksByPlayer[owner] = [];
        }
        stacksByPlayer[owner].push(key);
      }

      // eslint-disable-next-line no-console
      console.log('[GameEngine.makeMove.beforeAutomaticConsequences]', {
        gameId: this.gameState.id,
        moveType: fullMove.type,
        movePlayer: fullMove.player,
        boardStackCount: allBoardStackKeys.length,
        boardStackKeysSample: allBoardStackKeys.slice(0, 16),
        stacksByPlayer,
      });
    }

    // Legacy automatic consequence processing removed - orchestrator handles this
    this.debugCheckpoint('legacy-path-deprecated');

    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      const endResult = this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');

      // Even when the game ends, record a structured history entry for the
      // canonical move that produced the terminal state so parity/debug
      // tooling sees a complete move-by-move trace.
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      recordLegacyMove(endResult.success ? 'success' : 'error');

      return {
        success: endResult.success,
        gameResult: endResult.gameResult,
        gameState: this.getGameState(),
      };
    }

    // Advance to next phase/player and hand control to the shared
    // turn engine. In move-driven decision mode we mirror the sandbox
    // ClientSandboxEngine.advanceAfterMovement behaviour by treating
    // post-movement/capture consequences as having completed the
    // current player's automatic line/territory bookkeeping before we
    // consult the turn sequencer:
    //
    //   - When the move was a movement or capture segment and no
    //     explicit decision-phase moves were exposed, we normalise the
    //     phase to 'territory_processing' and let the shared TurnEngine
    //     advance from that boundary. This matches the sandbox, which
    //     always calls advanceTurnAndPhaseForCurrentPlayerSandbox from
    //     territory_processing after advanceAfterMovement.
    //   - Legacy / non-move-driven flows retain the original path:
    //     advance from the current interactive phase and then drain any
    //     empty decision phases via stepAutomaticPhasesForTesting.
    if (
      this.useMoveDrivenDecisionPhases &&
      (fullMove.type === 'move_stack' ||
        fullMove.type === 'move_ring' ||
        fullMove.type === 'overtaking_capture' ||
        fullMove.type === 'continue_capture_segment' ||
        fullMove.type === 'build_stack')
    ) {
      // Normalise to the end-of-turn bookkeeping boundary before
      // consulting the shared turn engine so that any forced
      // elimination / skipping is computed from the same phase as in
      // the sandbox trace engine.
      this.gameState = {
        ...this.gameState,
        currentPhase: 'territory_processing',
      };

      this.advanceGame();
      this.debugCheckpoint('after-advanceGame');
    } else {
      // Legacy path: advance from current phase, then drain automatic phases.
      // This is the path used by tests that do not enable move-driven decisions.
      this.advanceGame();
      this.debugCheckpoint('after-advanceGame');

      // Step through automatic bookkeeping phases (line_processing and
      // territory_processing) so the post-move snapshot and history
      // entry reflect the same next-player interactive phase that the
      // sandbox engine records in its traces.
      await this.stepAutomaticPhasesForTesting();
      this.debugCheckpoint('after-stepAutomaticPhasesForTesting');
    }

    // After all automatic phases have completed and the shared turn
    // engine has selected the next active player/phase, evaluate the
    // last-player-standing condition (R172) before exposing the new
    // interactive turn to callers.
    if (this.gameState.gameStatus === 'active') {
      const phase = this.gameState.currentPhase;
      if (
        phase === 'ring_placement' ||
        phase === 'movement' ||
        phase === 'capture' ||
        phase === 'chain_capture'
      ) {
        // Record start-of-turn availability for the new active player.
        this.updateLpsTrackingForCurrentTurn();
        const lpsResult = this.maybeEndGameByLastPlayerStanding();
        if (lpsResult) {
          this.debugCheckpoint('after-lps-check');

          // Record a structured history entry for the canonical move
          // that produced the terminal LPS state so parity/debug
          // tooling sees a complete trace.
          this.appendHistoryEntry(beforeStateForHistory, fullMove);

          recordLegacyMove('success');

          return {
            success: true,
            gameResult: lpsResult,
            gameState: this.getGameState(),
          };
        }
      }
    }

    this.debugCheckpoint('end-of-move');

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    // Record a structured history entry for this canonical move.
    this.appendHistoryEntry(beforeStateForHistory, fullMove);

    recordLegacyMove('success');

    return {
      success: true,
      gameState: this.getGameState(),
    };
  }

  private applyMove(move: Move): {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  } {
    const result = {
      captures: [] as Position[],
      territoryChanges: [] as Territory[],
      lineCollapses: [] as LineInfo[],
    };

    switch (move.type) {
      case 'place_ring':
        if (!move.to) {
          // DIAGNOSTIC: a validated place_ring move must always carry a destination.
          // Log and treat as a no-op rather than attempting a partial/manual mutation.
          console.error('[GameEngine.applyMove] BUG: place_ring without destination', {
            moveType: move.type,
            player: move.player,
            currentPhase: this.gameState.currentPhase,
            currentPlayer: this.gameState.currentPlayer,
          });
          break;
        }

        // Delegate placement mutation (stack creation/merge, marker exclusivity,
        // ringsInHand bookkeeping) to the shared PlacementAggregate so backend
        // GameEngine, RuleEngine, sandbox, and orchestrator all share a single
        // source of truth for placement effects.
        {
          const outcome = applyPlacementMoveAggregate(this.gameState, move);
          const newStateAfterPlacement = outcome.nextState;

          // Preserve the existing BoardState object reference while applying
          // the aggregate's updated board contents so callers that cache
          // engineAny.gameState.board continue to observe updates. This mirrors
          // the movement/capture branches and the TurnEngineAdapter wiring.
          const existingBoard = this.gameState.board;

          existingBoard.stacks.clear();
          for (const [key, stack] of newStateAfterPlacement.board.stacks) {
            existingBoard.stacks.set(key, stack);
          }

          existingBoard.markers.clear();
          for (const [key, marker] of newStateAfterPlacement.board.markers) {
            existingBoard.markers.set(key, marker);
          }

          existingBoard.collapsedSpaces.clear();
          for (const [key, owner] of newStateAfterPlacement.board.collapsedSpaces) {
            existingBoard.collapsedSpaces.set(key, owner);
          }

          existingBoard.territories.clear();
          for (const [key, territory] of newStateAfterPlacement.board.territories) {
            existingBoard.territories.set(key, territory);
          }

          existingBoard.formedLines.length = 0;
          existingBoard.formedLines.push(...newStateAfterPlacement.board.formedLines);

          for (const key of Object.keys(existingBoard.eliminatedRings)) {
            delete existingBoard.eliminatedRings[key as unknown as number];
          }
          for (const [key, value] of Object.entries(newStateAfterPlacement.board.eliminatedRings)) {
            existingBoard.eliminatedRings[key as unknown as number] = value;
          }

          existingBoard.size = newStateAfterPlacement.board.size;

          this.gameState = {
            ...newStateAfterPlacement,
            board: existingBoard,
          };
        }
        break;

      case 'skip_placement':
        // No-op at the board level. The RuleEngine has already verified
        // that skipping is only allowed when placement is optional, so
        // we simply advance to movement via advanceGame() without
        // modifying board or per-player ring counts.
        break;

      case 'move_ring':
      case 'move_stack':
        if (move.from && move.to) {
          const sourceStack = this.boardManager.getStack(move.from, this.gameState.board);
          if (!sourceStack) {
            // DIAGNOSTIC: This should never happen if validation and defensive
            // checks are working correctly. Log and bail to prevent silent no-ops.
            console.error('[GameEngine.applyMove] BUG: move_stack/move_ring but no source stack', {
              moveType: move.type,
              player: move.player,
              from: positionToString(move.from),
              to: positionToString(move.to),
              availableStacks: Array.from(this.gameState.board.stacks.keys()),
            });
            break; // Early exit from switch - no state change
          }

          // Delegate non-capturing movement mutation (marker-path effects,
          // stack merge, landing-on-own-marker elimination, and elimination
          // accounting) to the shared MovementAggregate so backend, sandbox,
          // and orchestrator all share a single source of truth.
          const outcome = applySimpleMovementAggregate(this.gameState, {
            from: move.from,
            to: move.to,
            player: move.player,
          });

          const newStateAfterMove = outcome.nextState;

          // Preserve the existing BoardState object reference while
          // applying the aggregate's updated board contents so callers
          // holding a cached board reference continue to observe updates.
          const existingBoard = this.gameState.board;

          existingBoard.stacks.clear();
          for (const [key, stack] of newStateAfterMove.board.stacks) {
            existingBoard.stacks.set(key, stack);
          }

          existingBoard.markers.clear();
          for (const [key, marker] of newStateAfterMove.board.markers) {
            existingBoard.markers.set(key, marker);
          }

          existingBoard.collapsedSpaces.clear();
          for (const [key, owner] of newStateAfterMove.board.collapsedSpaces) {
            existingBoard.collapsedSpaces.set(key, owner);
          }

          existingBoard.territories.clear();
          for (const [key, territory] of newStateAfterMove.board.territories) {
            existingBoard.territories.set(key, territory);
          }

          existingBoard.formedLines.length = 0;
          existingBoard.formedLines.push(...newStateAfterMove.board.formedLines);

          for (const key of Object.keys(existingBoard.eliminatedRings)) {
            delete existingBoard.eliminatedRings[key as unknown as number];
          }
          for (const [key, value] of Object.entries(newStateAfterMove.board.eliminatedRings)) {
            existingBoard.eliminatedRings[key as unknown as number] = value;
          }

          existingBoard.size = newStateAfterMove.board.size;

          this.gameState = {
            ...newStateAfterMove,
            board: existingBoard,
          };
        }
        break;

      case 'overtaking_capture':
      case 'continue_capture_segment':
        if (move.from && move.to && move.captureTarget) {
          // Delegate capture mutation to the shared CaptureAggregate so that
          // backend GameEngine, sandbox, and shared core all share a single
          // source of truth for marker-path effects, stack updates, and
          // landing-on-own-marker elimination.
          const captureResult = applyCaptureAggregate(this.gameState, move);

          if (!captureResult.success) {
            // Defensive diagnostic: this should never happen if RuleEngine
            // validation is in sync with the shared validator. Log and treat
            // as a no-op rather than attempting a partial/manual mutation.

            console.error('[GameEngine.applyMove] CaptureAggregate.applyCapture failed', {
              reason: captureResult.reason,
              moveType: move.type,
              player: move.player,
              from: move.from && positionToString(move.from),
              captureTarget: move.captureTarget && positionToString(move.captureTarget),
              to: move.to && positionToString(move.to),
            });
            break;
          }

          const newStateAfterCapture = captureResult.newState;

          // Preserve references to the existing BoardState Maps when
          // applying the shared capture mutation so that tests which cache
          // engineAny.gameState.board continue to observe updates. This
          // mirrors the TurnEngineAdapter stateAccessor.updateGameState
          // wiring used for orchestrator delegation.
          const existingBoard = this.gameState.board;

          existingBoard.stacks.clear();
          for (const [key, stack] of newStateAfterCapture.board.stacks) {
            existingBoard.stacks.set(key, stack);
          }

          existingBoard.markers.clear();
          for (const [key, marker] of newStateAfterCapture.board.markers) {
            existingBoard.markers.set(key, marker);
          }

          existingBoard.collapsedSpaces.clear();
          for (const [key, collapsed] of newStateAfterCapture.board.collapsedSpaces) {
            existingBoard.collapsedSpaces.set(key, collapsed);
          }

          existingBoard.territories.clear();
          for (const [key, territory] of newStateAfterCapture.board.territories) {
            existingBoard.territories.set(key, territory);
          }

          existingBoard.formedLines.length = 0;
          existingBoard.formedLines.push(...newStateAfterCapture.board.formedLines);

          for (const key of Object.keys(existingBoard.eliminatedRings)) {
            delete existingBoard.eliminatedRings[key as unknown as number];
          }
          for (const [key, value] of Object.entries(newStateAfterCapture.board.eliminatedRings)) {
            existingBoard.eliminatedRings[key as unknown as number] = value;
          }

          existingBoard.size = newStateAfterCapture.board.size;

          // Update the gameState while preserving the board reference
          this.gameState = {
            ...newStateAfterCapture,
            board: existingBoard,
          };

          // For diagnostics we continue to record the primary capture target
          // for this segment, matching the legacy applyMove behaviour.
          result.captures.push(move.captureTarget);
        }
        break;

      case 'build_stack':
        if (move.from && move.to && move.buildAmount) {
          const sourceStack = this.boardManager.getStack(move.from, this.gameState.board);
          const targetStack = this.boardManager.getStack(move.to, this.gameState.board);

          if (sourceStack && targetStack && move.buildAmount) {
            // Transfer rings from source to target
            const transferRings = sourceStack.rings.slice(0, move.buildAmount);
            const remainingRings = sourceStack.rings.slice(move.buildAmount);

            const newSourceStack: RingStack = {
              ...sourceStack,
              stackHeight: sourceStack.stackHeight - move.buildAmount,
              rings: remainingRings,
            };

            const newTargetStack: RingStack = {
              ...targetStack,
              stackHeight: targetStack.stackHeight + move.buildAmount,
              capHeight: Math.max(targetStack.capHeight, move.buildAmount),
              rings: [...targetStack.rings, ...transferRings],
            };

            // Update stacks
            if (newSourceStack.stackHeight > 0) {
              this.boardManager.setStack(move.from, newSourceStack, this.gameState.board);
            } else {
              this.boardManager.removeStack(move.from, this.gameState.board);
            }
            this.boardManager.setStack(move.to, newTargetStack, this.gameState.board);
          }
        }
        break;
    }

    // Line formation is processed separately in line_processing phase
    // This is just for tracking lines that were formed during the move.
    //
    // Territory disconnection and region collapse are now handled
    // exclusively via the shared territoryProcessing helper in
    // processAutomaticConsequences(), which applies the Q23
    // self-elimination prerequisite and S-invariant accounting. Any
    // legacy direct stack removal here would bypass those rules and
    // drift out of parity with the sandbox engine.

    return result;
  }

  /**
   * Update or initialize the internal chain capture state after an
   * overtaking capture has been successfully applied to the board.
   *
   * This mirrors the Rust engine's ChainCaptureState at a high level:
   * we track the start position, current position, and the sequence of
   * capture segments taken so far.
   */
  private updateChainCaptureStateAfterCaptureInternal(move: Move, capturedCapHeight: number): void {
    this.chainCaptureState = updateChainCaptureStateAfterCapture(
      this.chainCaptureState,
      move,
      capturedCapHeight
    );
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
   * Core overtaking capture operation used for both user-initiated
   * captures (via applyMove) and engine-driven chain captures.
   * Rule Reference: Section 10.2 - Overtaking Capture
   */

  private performOvertakingCapture(
    from: Position,
    captureTarget: Position,
    landing: Position,
    player: number
  ): void {
    const stack = this.boardManager.getStack(from, this.gameState.board);
    const targetStack = this.boardManager.getStack(captureTarget, this.gameState.board);

    if (!stack || !targetStack) {
      return;
    }

    // Leave a marker on the true departure space and immediately remove the
    // capturing stack from that cell so that board invariants never observe a
    // stack+marker overlap at the same key. This mirrors the sandbox capture
    // semantics, where the departure cell becomes a pure marker space before any
    // marker collapses occur along the path.
    this.boardManager.setMarker(from, player, this.gameState.board);
    this.boardManager.removeStack(from, this.gameState.board);

    // Process markers along the path from the original departure space to the
    // capture target, collapsing/flipping intermediate markers exactly as for a
    // normal movement leg.
    this.processMarkersAlongPath(from, captureTarget, player);

    // Then process markers along the path from the capture target to the landing
    // cell. As in the sandbox capture helper, we do not place an additional
    // departure marker on the capture target; only intermediate markers are
    // affected.
    this.processMarkersAlongPath(captureTarget, landing, player);

    // Check if landing on a marker before resolving the capture. Any marker
    // present at the landing cell must be removed prior to placing the
    // capturing stack so that stacks and markers never coexist on the same
    // space. If the marker belongs to the capturing player, we also apply
    // the self-elimination rule after landing.
    const landingMarkerPlayer = this.boardManager.getMarker(landing, this.gameState.board);
    const landedOnOwnMarker = landingMarkerPlayer === player;
    if (landingMarkerPlayer !== undefined) {
      // Remove the marker prior to landing; the self-elimination rule, when
      // applicable, will be applied after the capturing stack is placed.
      this.boardManager.removeMarker(landing, this.gameState.board);
    }

    // Capture top ring from target stack and add to bottom of capturing stack
    // Note: rings array is [top, ..., bottom] (based on calculateCapHeight and place_ring).
    // So top ring is at index 0.
    const capturedRing = targetStack.rings[0];

    // Add captured ring to the BOTTOM of the capturing stack (end of array)
    const newRings = [...stack.rings, capturedRing];

    // Update target stack (remove top ring)
    const remainingTargetRings = targetStack.rings.slice(1);
    if (remainingTargetRings.length > 0) {
      const newTargetStack: RingStack = {
        ...targetStack,
        rings: remainingTargetRings,
        stackHeight: remainingTargetRings.length,
        capHeight: calculateCapHeight(remainingTargetRings),
        controllingPlayer: remainingTargetRings[0],
      };
      this.boardManager.setStack(captureTarget, newTargetStack, this.gameState.board);
    } else {
      // Target stack is now empty, remove it
      this.boardManager.removeStack(captureTarget, this.gameState.board);
    }

    // Place capturing stack at landing position with captured ring
    const newStack: RingStack = {
      position: landing,
      rings: newRings,
      stackHeight: newRings.length,
      capHeight: calculateCapHeight(newRings),
      controllingPlayer: newRings[0], // Top ring is at index 0
    };
    this.boardManager.setStack(landing, newStack, this.gameState.board);

    if (landedOnOwnMarker) {
      // New rule: landing on your own marker during an overtaking capture
      // removes that marker and immediately eliminates your top ring,
      // credited toward ring-elimination victory conditions.
      this.eliminateTopRingAt(landing, player);
    }
  }

  /**
   * Enumerate canonical line-processing decision moves for the given player.
   *
   * This is now a thin adapter over the shared line-decision helpers so that
   * backend GameEngine, shared GameEngine, RuleEngine, and sandbox all share
   * a single source of truth for which `process_line` / `choose_line_reward`
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
    * - Move ID / payload conventions for process_territory_region.
    */
  private getValidTerritoryProcessingMoves(playerNumber: number): Move[] {
    return enumerateProcessTerritoryRegionMoves(this.gameState, playerNumber);
  }
  /**
   * Process all line formations with graduated rewards
   * Rule Reference: Section 11.2, 11.3
   *
   * For exact required length (4 for 8x8, 5 for 19x19/hex):
   *   - Collapse all markers
   *   - Eliminate one ring or cap from controlled stack
   *
   * For longer lines (5+ for 8x8, 6+ for 19x19/hex):
   *   - Option 1: Collapse all + eliminate ring/cap
   *   - Option 2: Collapse required markers only, no elimination
   */
  /**
   * Apply a single line-, territory-, or explicit elimination decision
   * expressed as a canonical Move. This is the entry point for the
   * unified Move model for the 'line_processing' and
   * 'territory_processing' phases; it mirrors the semantics of
   * processOneLine/processOneDisconnectedRegion but processes exactly one
   * line, region, or elimination choice selected by the Move and then
   * updates phases so callers can chain further decision Moves or
   * resume normal turn flow.
   */
  private async applyDecisionMove(move: Move): Promise<void> {
    if (move.type === 'process_line' || move.type === 'choose_line_reward') {
      const requiredLength = getEffectiveLineLengthThreshold(
        this.gameState.boardType,
        this.gameState.players.length,
        this.gameState.rulesOptions
      );

      const allLines = this.boardManager.findAllLines(this.gameState.board);
      const playerLines = allLines.filter(
        (line) => line.player === move.player && line.positions.length >= requiredLength
      );

      if (playerLines.length === 0) {
        return;
      }

      let targetLine: LineInfo | undefined;

      if (move.formedLines && move.formedLines.length > 0) {
        const target = move.formedLines[0];
        const targetKey = target.positions.map((p) => positionToString(p)).join('|');

        targetLine = playerLines.find((line) => {
          const lineKey = line.positions.map((p) => positionToString(p)).join('|');
          return lineKey === targetKey;
        });
      }

      if (!targetLine) {
        // Fallback: when no formedLines metadata is present or matching
        // fails, default to the first eligible line for this player,
        // preserving previous "first line wins" behaviour while also
        // respecting the effective threshold (e.g. 4-in-a-row for 2p 8x8).
        targetLine = playerLines[0];
      }

      // In move-driven mode with explicit decision Moves, delegate the
      // geometric and bookkeeping effects to the shared lineDecisionHelpers
      // so backend GameEngine, shared GameEngine, and sandbox all share a
      // single source of truth for line collapse + reward semantics.
      if (this.useMoveDrivenDecisionPhases) {
        const outcome =
          move.type === 'process_line'
            ? applyProcessLineDecision(this.gameState, move)
            : applyChooseLineRewardDecision(this.gameState, move);

        this.gameState = outcome.nextState;
        this.pendingLineRewardElimination = outcome.pendingLineRewardElimination;

        // When a mandatory elimination reward is pending, remain in
        // line_processing so getValidMoves can surface explicit
        // eliminate_rings_from_stack decisions for this player.
        if (this.pendingLineRewardElimination) {
          this.gameState.currentPhase = 'line_processing';
          return;
        }

        // No elimination owed; determine whether any further lines remain
        // for this player in the updated board state. We use the shared
        // enumerateProcessLineMoves helper so detection stays aligned with
        // RuleEngine and sandbox.
        const remainingProcessMoves = enumerateProcessLineMoves(this.gameState, move.player, {
          detectionMode: 'use_board_cache',
        });

        if (remainingProcessMoves.length > 0) {
          this.gameState.currentPhase = 'line_processing';
        } else {
          this.gameState.currentPhase = 'territory_processing';
        }
        return;
      }

      // Legacy / non-move-driven mode: delegate to processOneLine which
      // handles PlayerChoice flows internally for backward compatibility.
      await this.processOneLine(targetLine, requiredLength);

      // After processing one line, re-check whether any further eligible
      // lines exist for the same player. If so, stay in line_processing so
      // the client/AI can submit another decision Move. Otherwise, advance
      // to territory_processing to handle any disconnections created by the
      // collapse.
      const remainingLines = this.boardManager
        .findAllLines(this.gameState.board)
        .filter(
          (line) => line.player === move.player && line.positions.length >= requiredLength
        );

      if (remainingLines.length > 0) {
        this.gameState.currentPhase = 'line_processing';
      } else {
        this.gameState.currentPhase = 'territory_processing';
      }
    } else if (move.type === 'process_territory_region') {
      const movingPlayer = move.player;

      // Legacy / non-move-driven behaviour: process the region and
      // immediately perform mandatory self-elimination via the existing
      // PlayerChoice helper. This keeps all scenario/parity tests that do
      // not enable move-driven decision phases aligned with the original
      // semantics.
      if (!this.useMoveDrivenDecisionPhases) {
        const disconnectedRegions = this.boardManager.findDisconnectedRegions(
          this.gameState.board,
          movingPlayer
        );

        if (!disconnectedRegions || disconnectedRegions.length === 0) {
          return;
        }

        let targetRegion: Territory | undefined;

        if (move.disconnectedRegions && move.disconnectedRegions.length > 0) {
          const target = move.disconnectedRegions[0];
          const targetKeys = new Set(target.spaces.map((pos) => positionToString(pos)));

          targetRegion = disconnectedRegions.find((region) => {
            if (region.spaces.length !== target.spaces.length) {
              return false;
            }

            const regionKeys = new Set(region.spaces.map((pos) => positionToString(pos)));

            if (regionKeys.size !== targetKeys.size) {
              return false;
            }

            for (const key of targetKeys) {
              if (!regionKeys.has(key)) {
                return false;
              }
            }

            return true;
          });
        }

        if (!targetRegion) {
          // Fallback: choose the first region that satisfies the
          // self-elimination prerequisite; if none do, leave the state
          // unchanged.
          targetRegion =
            disconnectedRegions.find((region) =>
              this.canProcessDisconnectedRegion(region, movingPlayer)
            ) ?? undefined;
        }

        if (!targetRegion) {
          return;
        }

        // Respect the self-elimination prerequisite defensively even if the
        // caller attempted to target an ineligible region.
        if (!this.canProcessDisconnectedRegion(targetRegion, movingPlayer)) {
          return;
        }

        await this.processOneDisconnectedRegion(targetRegion, movingPlayer);

        // After processing this region (including mandatory self-elimination
        // and any internal eliminations), re-check for additional eligible
        // disconnected regions for the same player. If any remain, stay in
        // territory_processing so another process_territory_region Move can
        // be applied. Otherwise, hand control to the normal turn engine so
        // the next player's turn/phase is computed exactly as after
        // automatic territory processing.
        const remainingDisconnected = this.boardManager.findDisconnectedRegions(
          this.gameState.board,
          movingPlayer
        );

        const remainingEligible = remainingDisconnected.filter((region) =>
          this.canProcessDisconnectedRegion(region, movingPlayer)
        );

        if (remainingEligible.length > 0) {
          this.gameState.currentPhase = 'territory_processing';
        } else {
          this.advanceGame();
          this.stepAutomaticPhasesForTesting();
        }

        return;
      }

      // Move-driven decision phases: delegate the region-processing
      // consequences to the shared applyProcessTerritoryRegionDecision
      // helper so that backend GameEngine, RuleEngine, and sandbox all
      // share identical geometry and S-invariant accounting.
      const outcome = applyProcessTerritoryRegionDecision(this.gameState, move);

      // If no valid region can be resolved for this decision (for example,
      // due to a stale or tampered Move), treat it as a no-op.
      if (!outcome.pendingSelfElimination || outcome.processedRegion.spaces.length === 0) {
        return;
      }

      this.gameState = outcome.nextState;

      // Record that this player now owes a mandatory self-elimination
      // decision before their territory_processing cycle can end. We stay
      // in territory_processing here; phase advancement happens only after
      // an explicit eliminate_rings_from_stack decision is applied.
      this.pendingTerritorySelfElimination = true;
      this.gameState.currentPlayer = movingPlayer;
      this.gameState.currentPhase = 'territory_processing';
    } else if (move.type === 'eliminate_rings_from_stack') {
      const playerNumber = move.player;

      // This explicit elimination satisfies any pending self-elimination
      // requirement from either territory processing (after processing a
      // disconnected region) or line processing (after choosing Option 1
      // for an exact-length or overlength line).
      const wasTerritorySelfElimination = this.pendingTerritorySelfElimination;
      const wasLineRewardElimination = this.pendingLineRewardElimination;

      this.pendingTerritorySelfElimination = false;
      this.pendingLineRewardElimination = false;

      if (!move.to) {
        return;
      }

      // Delegate the actual elimination accounting to the shared
      // applyEliminateRingsFromStackDecision helper so that backend
      // GameEngine, RuleEngine, and sandbox all share identical S-invariant
      // behaviour. Any structural invalidity (e.g. missing/foreign stack)
      // results in a no-op nextState.
      const { nextState } = applyEliminateRingsFromStackDecision(this.gameState, move);
      this.gameState = nextState;

      // After an explicit elimination decision, determine which phase to
      // return to based on the origin of the elimination requirement.
      if (wasLineRewardElimination) {
        // Line reward elimination complete; check if more lines remain
        // for this player before leaving line_processing.
        const remainingLines = this.boardManager
          .findAllLines(this.gameState.board)
          .filter((line) => line.player === playerNumber);

        if (remainingLines.length > 0) {
          this.gameState.currentPhase = 'line_processing';
        } else {
          this.gameState.currentPhase = 'territory_processing';
        }
      } else if (wasTerritorySelfElimination) {
        // Territory self-elimination complete for one disconnected region.
        // Recompute whether any further eligible disconnected regions remain
        // for this player under the self-elimination prerequisite. If so, stay
        // in territory_processing for another explicit
        // process_territory_region decision; otherwise, hand control back to
        // the normal turn engine so the next player's phase is computed.
        const remainingDisconnected = this.boardManager.findDisconnectedRegions(
          this.gameState.board,
          playerNumber
        );
        const remainingEligible = remainingDisconnected.filter((region) =>
          this.canProcessDisconnectedRegion(region, playerNumber)
        );

        if (remainingEligible.length > 0) {
          // Stay in the interactive territory_processing phase so the client/AI
          // can submit another process_territory_region Move for this player.
          this.gameState.currentPhase = 'territory_processing';
          this.gameState.currentPlayer = playerNumber;
        } else {
          // No further territory decisions remain; advance the turn using the
          // normal turn engine so the next player's phase is computed.
          this.advanceGame();
          this.stepAutomaticPhasesForTesting();

          // In some terminal self-elimination cases the TurnEngine may leave
          // the phase as 'territory_processing' even though the game has
          // ended. Normalise such terminal states so callers never observe a
          // completed game still "in" the dedicated territory decision phase.
          if (
            this.gameState.gameStatus !== 'active' &&
            this.gameState.currentPhase === 'territory_processing'
          ) {
            this.gameState.currentPhase = 'ring_placement';
          }
        }
      } else {
        // Defensive fallback: if no pending flag was set, treat this as
        // a standalone elimination and advance the game normally.
        this.advanceGame();
        this.stepAutomaticPhasesForTesting();

        if (
          this.gameState.gameStatus !== 'active' &&
          this.gameState.currentPhase === 'territory_processing'
        ) {
          this.gameState.currentPhase = 'ring_placement';
        }
      }
    }
  }

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
          line.player === this.gameState.currentPlayer &&
          line.positions.length >= requiredLength
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
          m.type === 'choose_line_reward' &&
          m.id.includes(lineKey) &&
          (!m.collapsedMarkers || m.collapsedMarkers.length === line.positions.length)
      );

      const option2Move = validMoves.find(
        (m) =>
          m.type === 'choose_line_reward' &&
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
   * Eliminate one ring or cap from player's controlled stacks
   * Rule Reference: Section 11.2 - Moving player chooses which ring/stack cap to eliminate
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
   * Eliminate exactly the top ring from the stack at the given position,
   * crediting the elimination to the specified player. This is used for
   * the "landing on your own marker eliminates your top ring" rule,
   * which applies to both non-capture moves and overtaking capture
   * segments.
   */
  private eliminateTopRingAt(position: Position, creditedPlayer: number): void {
    const stack = this.boardManager.getStack(position, this.gameState.board);
    if (!stack || stack.stackHeight === 0) {
      return;
    }

    // Remove the single top ring from the stack.
    const [, ...remainingRings] = stack.rings;

    // Update global elimination counters (one ring credited to the mover).
    this.gameState.totalRingsEliminated += 1;
    if (!this.gameState.board.eliminatedRings[creditedPlayer]) {
      this.gameState.board.eliminatedRings[creditedPlayer] = 0;
    }
    this.gameState.board.eliminatedRings[creditedPlayer] += 1;

    // Update per-player elimination stats.
    this.updatePlayerEliminatedRings(creditedPlayer, 1);

    if (remainingRings.length > 0) {
      const newStack: RingStack = {
        ...stack,
        rings: remainingRings,
        stackHeight: remainingRings.length,
        capHeight: calculateCapHeight(remainingRings),
        controllingPlayer: remainingRings[0],
      };
      this.boardManager.setStack(position, newStack, this.gameState.board);
    } else {
      // If no rings remain, remove the stack entirely.
      this.boardManager.removeStack(position, this.gameState.board);
    }
  }

  /**
   * Eliminate one ring or cap using the player choice system when available.
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

    const choice: RingEliminationChoice = {
      id: generateUUID(),
      gameId: this.gameState.id,
      playerNumber: player,
      type: 'ring_elimination',
      prompt: 'Choose which stack to eliminate from',
      options: playerStacks.map((stack) => {
        const stackKey = positionToString(stack.position);
        return {
          stackPosition: stack.position,
          capHeight: stack.capHeight,
          totalHeight: stack.stackHeight,
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
   * self-elimination is layered on top differently for legacy vs
   * move-driven modes:
   *
   * - Legacy / non-move-driven: processOneDisconnectedRegion calls this
   *   helper and then immediately performs
   *   eliminatePlayerRingOrCapWithChoice.
   * - Move-driven decision phases: applyDecisionMove('process_territory_region')
   *   calls this helper directly and then surfaces explicit
   *   eliminate_rings_from_stack decision Moves via RuleEngine so the
   *   self-elimination is represented as a canonical Move.
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
   */
  private async processOneDisconnectedRegion(
    region: Territory,
    movingPlayer: number
  ): Promise<void> {
    // Always apply the geometric/core consequences first.
    this.processDisconnectedRegionCore(region, movingPlayer);

    // In legacy / non-move-driven mode, immediately perform the
    // mandatory self-elimination using the existing PlayerChoice
    // helper so behaviour remains identical for scenario/parity tests
    // that do not opt into move-driven decision phases.
    if (!this.useMoveDrivenDecisionPhases) {
      await this.eliminatePlayerRingOrCapWithChoice(movingPlayer);
    }
  }
  /**
   * Test-only helper: process all eligible disconnected regions for the
   * current player using the legacy territory-processing pipeline
   * (processOneDisconnectedRegion).
   *
   * This mirrors the behaviour exercised by
   * GameEngine.territoryDisconnection.test.ts, but is implemented purely
   * in terms of {@link processDisconnectedRegionCore} and the shared
   * territory helpers so it stays aligned with RR‑CANON‑R140–R145 and
   * the TS/Python engines.
   *
   * Production hosts do not call this method; it exists solely for
   * parity/scenario suites and diagnostic harnesses.
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
   * Process markers along the movement path
   * Rule Reference: Section 8.3 - Marker Interaction
   */
  private processMarkersAlongPath(from: Position, to: Position, player: number): void {
    // Get all positions along the straight line path
    const path = getPathPositions(from, to);

    // Process each position in the path (excluding start and end)
    for (let i = 1; i < path.length - 1; i++) {
      const pos = path[i];
      const marker = this.boardManager.getMarker(pos, this.gameState.board);

      if (marker !== undefined) {
        if (marker === player) {
          // Own marker: collapse to territory (Section 8.3)
          this.boardManager.collapseMarker(pos, player, this.gameState.board);
        } else {
          // Opponent marker: flip to your color (Section 8.3)
          this.boardManager.flipMarker(pos, player, this.gameState.board);
        }
      }
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
      console.log(`[GameEngine] advanceGame: mustMove is ${this.mustMoveFromStackKey}`);
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

  /**
   * Update internal per-turn placement/movement bookkeeping after a move
   * has been applied. This keeps the must-move origin in sync with the
   * stack that was placed or moved, mirroring the sandbox engine’s
   * behaviour while keeping these details off of GameState.
   */
  private updatePerTurnStateAfterMove(move: Move): void {
    const before: PerTurnState = {
      hasPlacedThisTurn: this.hasPlacedThisTurn,
      mustMoveFromStackKey: this.mustMoveFromStackKey,
    };

    const after = updatePerTurnStateAfterMoveTurn(before, move);

    this.hasPlacedThisTurn = after.hasPlacedThisTurn;
    this.mustMoveFromStackKey = after.mustMoveFromStackKey;

    if (move.type === 'place_ring') {
      console.log(
        `[GameEngine] updatePerTurnStateAfterMove: place_ring at ${positionToString(move.to!)}. mustMove set to ${this.mustMoveFromStackKey}`
      );
    }
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.type === 'ai') return;

    // In Jest test runs we avoid creating long-lived OS-level timers so that
    // game clocks (which may be minutes long) do not keep the Node event loop
    // alive after tests complete, which would trigger Jest's
    // "asynchronous operations that weren't stopped" warning. Tests do not
    // currently assert on real-time forfeits, so it is safe to no-op timers
    // when NODE_ENV === 'test'.
    if (config.isTest) {
      return;
    }

    const timer = setTimeout(() => {
      // Time expired, forfeit the game
      this.forfeitGame(playerNumber.toString());
    }, player.timeRemaining);

    this.moveTimers.set(playerNumber, timer);
  }

  private stopPlayerTimer(playerNumber: number): void {
    const timer = this.moveTimers.get(playerNumber);
    if (timer) {
      clearTimeout(timer);
      this.moveTimers.delete(playerNumber);
    }
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
    for (const timer of this.moveTimers.values()) {
      clearTimeout(timer);
    }
    this.moveTimers.clear();

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
      reason: (reason as any) || 'game_completed',
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
    console.log('Rating update needed for:', {
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

    return this.endGame(winner, 'resignation');
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

      // eslint-disable-next-line no-console
      console.log('[GameEngine.getValidMoves] chain_capture debug', {
        requestedPlayer: playerNumber,
        capturingPlayer,
        currentPhase: this.gameState.currentPhase,
        currentPosition: state.currentPosition,
        followUpCount: followUpMoves.length,
      });

      if (followUpMoves.length === 0) {
        // No legal continuations remain; clear chain state and treat the
        // chain as resolved so callers do not see an interactive phase
        // with no legal actions.
        this.chainCaptureState = undefined;
        return [];
      }

      return followUpMoves.map((m) => ({
        ...m,
        // Re-label the shared overtaking_capture candidates as dedicated
        // continue_capture_segment moves for the unified Move model.
        type: 'continue_capture_segment',
        // Ensure the move is attributed to the capturing player recorded
        // in the chain state, even if the caller passed a different
        // playerNumber by mistake.
        player: capturingPlayer,
        id:
          m.id && m.id.length > 0
            ? m.id.startsWith('capture-')
              ? m.id.replace('capture-', 'continue-')
              : m.id
            : `continue-${positionToString(m.from!)}-${positionToString(
                m.captureTarget!
              )}-${positionToString(m.to!)}`,
      }));
    }

    // For automatic bookkeeping phases, expose the canonical decision moves
    // derived from the same helpers that drive line and territory
    // processing. This keeps the unified Move/GamePhase model complete even
    // though these phases are still usually resolved internally.
    if (this.gameState.currentPhase === 'line_processing') {
      const lineMoves = this.getValidLineProcessingMoves(playerNumber);

      // In move-driven mode, when a line reward elimination is pending
      // (exact-length line or Option 1 for overlength), surface explicit
      // eliminate_rings_from_stack Moves instead of further process_line
      // / choose_line_reward decisions. We enumerate these directly from the
      // current board rather than borrowing the territory_processing phase so
      // that the decision surface matches the sandbox engine and shared
      // move model.
      if (this.useMoveDrivenDecisionPhases && this.pendingLineRewardElimination) {
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
            // For line-reward eliminations the full cap is always eliminated
            // from the chosen stack, mirroring the sandbox elimination
            // decision helper and the shared-engine fixtures.
            eliminatedRings: [{ player: playerNumber, count: stack.capHeight }],
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

      // In legacy / non-move-driven mode, territory-processing decisions
      // remain region-first and self-elimination is handled internally via
      // PlayerChoice flows. Only expose process_territory_region Moves
      // here to preserve existing semantics for scenario/parity tests.
      if (!this.useMoveDrivenDecisionPhases) {
        return regionMoves;
      }

      if (regionMoves.length > 0) {
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
      // RuleEngine.getValidMoves so enumeration stays consistent with the
      // rules-level view used by applyDecisionMove.
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

      const TRACE_DEBUG_ENABLED =
        typeof process !== 'undefined' &&
        !!(process as any).env &&
        ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_TRACE_DEBUG ?? '');

      const beforeState = this.getGameState();

      if (TRACE_DEBUG_ENABLED) {
        // eslint-disable-next-line no-console
        console.log('[GameEngine.getValidMoves] resolving blocked interactive state', {
          currentPlayer: beforeState.currentPlayer,
          currentPhase: beforeState.currentPhase,
          gameStatus: beforeState.gameStatus,
          moveCount: moves.length,
          totalRingsEliminated: beforeState.totalRingsEliminated,
        });
      }

      this.resolveBlockedStateForCurrentPlayerForTesting();

      // If the resolver ended the game, there are no further moves to surface.
      if (this.gameState.gameStatus !== 'active') {
        return [];
      }

      const afterState = this.getGameState();

      if (TRACE_DEBUG_ENABLED) {
        // eslint-disable-next-line no-console
        console.log('[GameEngine.getValidMoves] blocked state resolved', {
          previousPlayer: beforeState.currentPlayer,
          previousPhase: beforeState.currentPhase,
          currentPlayer: afterState.currentPlayer,
          currentPhase: afterState.currentPhase,
          gameStatus: afterState.gameStatus,
          totalRingsEliminated: afterState.totalRingsEliminated,
        });
      }

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
          m.type === 'move_ring' ||
          m.type === 'build_stack' ||
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
   */
  private shouldOfferSwapSidesMetaMove(): boolean {
    const state = this.gameState;
 
    // Config gating: swap_sides is only offered when explicitly enabled
    // for this game via state.rulesOptions.swapRuleEnabled. When the
    // flag is absent or false, the pie rule is considered disabled.
    if (!state.rulesOptions?.swapRuleEnabled) return false;
 
    if (state.gameStatus !== 'active') return false;
    if (state.players.length !== 2) return false;
    if (state.currentPlayer !== 2) return false;

    // Only in interactive phases.
    if (
      state.currentPhase !== 'ring_placement' &&
      state.currentPhase !== 'movement' &&
      state.currentPhase !== 'capture' &&
      state.currentPhase !== 'chain_capture'
    ) {
      return false;
    }

    if (state.moveHistory.length === 0) return false;

    const hasSwapMove = state.moveHistory.some((m) => m.type === 'swap_sides');
    if (hasSwapMove) return false;

    const hasP1Move = state.moveHistory.some((m) => m.player === 1);
    const hasP2Move = state.moveHistory.some((m) => m.player === 2 && m.type !== 'swap_sides');

    // Exactly: at least one move from P1, none from P2 yet.
    return hasP1Move && !hasP2Move;
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
    const { id, timestamp, moveNumber, ...payload } = selected as any;

    return this.makeMove(payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>);
  }

  /**
   * Determine whether the specified player currently has any "real"
   * actions available in the sense of R172: at least one legal ring
   * placement, non-capture movement, or overtaking capture. This helper
   * deliberately ignores bookkeeping-only moves such as skip_placement,
   * forced-elimination decisions, and line/territory processing moves.
   */
  private hasAnyRealActionForPlayer(state: GameState, playerNumber: number): boolean {
    if (state.gameStatus !== 'active') {
      return false;
    }

    const playerState = state.players.find((p) => p.playerNumber === playerNumber);
    if (!playerState) {
      return false;
    }

    // 1) Ring placement (place_ring).
    if (playerState.ringsInHand > 0) {
      const tempPlacementState: GameState = {
        ...state,
        currentPlayer: playerNumber,
        currentPhase: 'ring_placement',
      };

      const placementMoves = this.ruleEngine.getValidMoves(tempPlacementState);
      if (placementMoves.some((m) => m.type === 'place_ring')) {
        return true;
      }
    }

    // 2) Non-capture movement.
    const tempMovementState: GameState = {
      ...state,
      currentPlayer: playerNumber,
      currentPhase: 'movement',
    };
    const movementMoves = this.ruleEngine.getValidMoves(tempMovementState);
    if (
      movementMoves.some(
        (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
      )
    ) {
      return true;
    }

    // 3) Overtaking capture.
    //
    // Use the shared CaptureAggregate global enumerator so that "real action"
    // detection for LPS tracking is based on the same capture surface as the
    // turn engine and sandbox, independent of any backend RuleEngine helpers.
    const tempCaptureState: GameState = {
      ...state,
      currentPlayer: playerNumber,
      currentPhase: 'capture',
    };
    const captureMoves = enumerateAllCaptureMovesAggregate(tempCaptureState, playerNumber);
    return captureMoves.length > 0;
  }

  /**
   * Internal helper for LPS tracking: true when the given player still
   * has any rings of their own colour on the board or in hand.
   */
  private playerHasMaterial(playerNumber: number): boolean {
    const totalInPlay = countRingsInPlayForPlayer(this.gameState, playerNumber);
    return totalInPlay > 0;
  }

  /**
   * Update per-round last-player-standing tracking at the start of the
   * current player's interactive turn. This should be called after the
   * shared turn engine has selected the next active player and phase,
   * and after any forced eliminations for the previous player have been
   * applied.
   */
  private updateLpsTrackingForCurrentTurn(): void {
    const state = this.gameState;
    if (state.gameStatus !== 'active') {
      return;
    }

    const phase = state.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return;
    }

    const currentPlayer = state.currentPlayer;

    // Mirror sandbox LPS semantics by filtering to the active-player
    // set (players who still have material) and tracking rounds over
    // this set only. This keeps lpsRoundIndex, round leaders, and
    // actor masks aligned between hosts even as players drop out.
    const activePlayers = state.players
      .filter((p) => this.playerHasMaterial(p.playerNumber))
      .map((p) => p.playerNumber);

    if (activePlayers.length === 0) {
      return;
    }

    const activeSet = new Set(activePlayers);
    if (!activeSet.has(currentPlayer)) {
      // Current player has no material; they are not part of the LPS
      // round and should be ignored for tracking purposes.
      return;
    }

    const first = this.lpsCurrentRoundFirstPlayer;
    const startingNewCycle = first === null || !activeSet.has(first);

    if (startingNewCycle) {
      // Either this is the very first LPS round or the previous round
      // leader is no longer in the active set (e.g. eliminated). Start
      // a fresh round anchored at the current player.
      this.lpsRoundIndex += 1;
      this.lpsCurrentRoundFirstPlayer = currentPlayer;
      this.lpsCurrentRoundActorMask.clear();
      this.lpsExclusivePlayerForCompletedRound = null;
    } else if (currentPlayer === first && this.lpsCurrentRoundActorMask.size > 0) {
      // We have looped back to the first player seen in this cycle;
      // finalise the previous round summary over the active-player
      // set before starting a new one.
      this.finalizeCompletedLpsRound(activePlayers);
      this.lpsRoundIndex += 1;
      this.lpsCurrentRoundActorMask.clear();
      this.lpsCurrentRoundFirstPlayer = currentPlayer;
    }

    const hasRealAction = this.hasAnyRealActionForPlayer(state, currentPlayer);
    this.lpsCurrentRoundActorMask.set(currentPlayer, hasRealAction);
  }

  /**
   * Finalise the summary for the just-completed round of turns and record
   * whether there was a unique player who had real actions available on
   * every turn in that round while all other players were inactive.
   */
  private finalizeCompletedLpsRound(activePlayers: number[]): void {
    if (activePlayers.length === 0) {
      this.lpsExclusivePlayerForCompletedRound = null;
      return;
    }

    const truePlayers: number[] = [];
    for (const pid of activePlayers) {
      if (this.lpsCurrentRoundActorMask.get(pid)) {
        truePlayers.push(pid);
      }
    }

    if (truePlayers.length === 1) {
      this.lpsExclusivePlayerForCompletedRound = truePlayers[0];
    } else {
      this.lpsExclusivePlayerForCompletedRound = null;
    }
  }

  /**
   * Evaluate the last-player-standing victory condition (R172) at the
   * start of the current player's interactive turn, given any candidate
   * derived from the most recently completed round. When satisfied, this
   * ends the game with reason 'last_player_standing'.
   */
  private maybeEndGameByLastPlayerStanding(): GameResult | undefined {
    const state = this.gameState;
    if (state.gameStatus !== 'active') {
      return undefined;
    }

    const phase = state.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return undefined;
    }

    const candidate = this.lpsExclusivePlayerForCompletedRound;
    if (candidate === null || candidate === undefined) {
      return undefined;
    }

    if (state.currentPlayer !== candidate) {
      return undefined;
    }

    // Candidate must still have at least one real action.
    if (!this.hasAnyRealActionForPlayer(state, candidate)) {
      this.lpsExclusivePlayerForCompletedRound = null;
      return undefined;
    }

    // All other players who still have material must have no real actions.
    const othersHaveRealActions = state.players.some((p) => {
      if (p.playerNumber === candidate) {
        return false;
      }
      if (!this.playerHasMaterial(p.playerNumber)) {
        return false;
      }
      return this.hasAnyRealActionForPlayer(state, p.playerNumber);
    });

    if (othersHaveRealActions) {
      this.lpsExclusivePlayerForCompletedRound = null;
      return undefined;
    }

    const endResult = this.endGame(candidate, 'last_player_standing');
    return endResult.gameResult;
  }

  /**
   * Shared helper to run LPS tracking + victory check for the current
   * state when we have just selected a new interactive turn. This
   * mirrors the sandbox handleStartOfInteractiveTurn wiring and is safe
   * to call multiple times; when the game is not active or the phase is
   * not interactive it is a no-op.
   *
   * NOTE: When invoked from test-only helpers such as
   * stepAutomaticPhasesForTesting, we intentionally do not append an
   * additional history entry; the canonical move that led to this state
   * has already been recorded.
   */
  private runLpsCheckForCurrentInteractiveTurn(): GameResult | undefined {
    if (this.gameState.gameStatus !== 'active') {
      return undefined;
    }

    const phase = this.gameState.currentPhase;
    if (
      phase !== 'ring_placement' &&
      phase !== 'movement' &&
      phase !== 'capture' &&
      phase !== 'chain_capture'
    ) {
      return undefined;
    }

    this.updateLpsTrackingForCurrentTurn();
    return this.maybeEndGameByLastPlayerStanding();
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
          const hasMovementLocal = movementMoves.some(
            (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
          );

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

      // In move-driven decision phases, decision moves (process_line,
      // process_territory_region, etc.) should be submitted explicitly by
      // the client/AI and recorded in the trace. Do NOT auto-apply them
      // here, as that would cause the backend to advance past phases that
      // should be represented as discrete moves in parity traces.
      //
      // Only advance when there are no decisions to make (handled above).
      if (this.useMoveDrivenDecisionPhases) {
        return;
      }

      // Legacy mode: automatically apply the first available decision move.
      // This mimics the default behaviour of the legacy automatic pipeline
      // (first line, first region, default elimination) and drives it via
      // the unified Move model.
      const move = moves[0];
      await this.applyDecisionMove(move);
    }

    // In move-driven decision-phase mode, interactive forced-elimination and
    // skip semantics are handled centrally by the shared TurnEngine and
    // turnLogic at turn start, just as in the sandbox engine. The
    // interactive loop below is therefore only used for legacy/non-move-driven
    // flows.
    if (this.useMoveDrivenDecisionPhases) {
      return;
    }

    // Additionally, handle the case where we're in an interactive phase
    // but the current player has no valid moves (e.g., they're blocked or eliminated).
    // The sandbox's advanceTurnAndPhase logic applies forced elimination at turn start;
    // we replicate that here by checking ALL types of moves (placement, movement,
    // capture) regardless of current phase, not just phase-specific moves.

    // Diagnostic: log entry state for end-of-game debugging
    const STEP_DEBUG = this.gameState.totalRingsEliminated >= 28;
    if (STEP_DEBUG) {
      // eslint-disable-next-line no-console
      console.log('[GameEngine.stepAutomaticPhasesForTesting] entry', {
        currentPlayer: this.gameState.currentPlayer,
        currentPhase: this.gameState.currentPhase,
        gameStatus: this.gameState.gameStatus,
        totalRingsEliminated: this.gameState.totalRingsEliminated,
        stackCount: this.gameState.board.stacks.size,
      });
    }

    while (
      this.gameState.gameStatus === 'active' &&
      (this.gameState.currentPhase === 'movement' ||
        this.gameState.currentPhase === 'ring_placement' ||
        this.gameState.currentPhase === 'capture')
    ) {
      const currentPlayer = this.gameState.currentPlayer;
      const playerState = this.gameState.players.find((p) => p.playerNumber === currentPlayer);
      const stacksForCurrent = this.boardManager.getPlayerStacks(
        this.gameState.board,
        currentPlayer
      );
      const hasStacks = stacksForCurrent.length > 0;
      const hasRingsInHand = playerState && playerState.ringsInHand > 0;

      // Check for "real" actions across ALL types (not just current phase moves)
      // to match sandbox's maybeProcessForcedEliminationForCurrentPlayerSandbox:
      // 1) Placement: ringsInHand > 0 AND getValidMoves in ring_placement has place_ring
      // 2) Movement: stacks exist AND getValidMoves in movement has real moves
      // 3) Capture: stacks exist AND getValidMoves in capture has overtaking_capture

      let hasPlacement = false;
      if (hasRingsInHand) {
        const tempPlacement: GameState = {
          ...this.gameState,
          currentPlayer,
          currentPhase: 'ring_placement',
        };
        const placementMoves = this.ruleEngine.getValidMoves(tempPlacement);
        hasPlacement = placementMoves.some((m) => m.type === 'place_ring');
      }

      let hasMovement = false;
      if (hasStacks) {
        const tempMovement: GameState = {
          ...this.gameState,
          currentPlayer,
          currentPhase: 'movement',
        };
        const movementMoves = this.ruleEngine.getValidMoves(tempMovement);
        hasMovement = movementMoves.some(
          (m) => m.type === 'move_stack' || m.type === 'move_ring' || m.type === 'build_stack'
        );
      }

      let hasCapture = false;
      if (hasStacks) {
        const tempCapture: GameState = {
          ...this.gameState,
          currentPlayer,
          currentPhase: 'capture',
        };
        const captureMoves = enumerateAllCaptureMovesAggregate(tempCapture, currentPlayer);
        hasCapture = captureMoves.length > 0;
      }

      const hasRealAction = hasPlacement || hasMovement || hasCapture;

      if (hasRealAction) {
        // Player has at least one real action available; leave them in
        // the current interactive phase. LPS tracking and victory checks
        // are handled centrally at the end of makeMove, after a single
        // call to stepAutomaticPhasesForTesting.
        return;
      }

      // Player has no real actions. Attempt canonical forced elimination
      // when the RR-CANON preconditions hold (blocked with stacks and no
      // legal placements/movements/captures). We rely on the shared
      // helper so test-only flows stay aligned with the core engine.
      const outcome = applyForcedEliminationForPlayer(this.gameState, currentPlayer);
      if (outcome) {
        this.gameState = outcome.nextState;

        // Check victory conditions after the forced elimination
        const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
        if (gameEndCheck.isGameOver) {
          this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'forced_elimination');
          return;
        }
      }

      // No real actions available - advance to skip this player.
      this.advanceGame();
    }
  }
}
