import {
  GameState,
  Move,
  Player,
  BoardType,
  TimeControl,
  GameResult,
  BOARD_CONFIGS,
  Position,
  RingStack,
  Territory,
  LineInfo,
  positionToString,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  PlayerChoiceResponseFor,
  GameHistoryEntry,
} from '../../shared/types/game';
import {
  calculateCapHeight,
  getPathPositions,
  getMovementDirectionsForBoardType,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../shared/engine/core';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';
import { PlayerInteractionManager } from './PlayerInteractionManager';
import { processLinesForCurrentPlayer } from './rules/lineProcessing';
import { processDisconnectedRegionsForCurrentPlayer } from './rules/territoryProcessing';
import {
  ChainCaptureState,
  updateChainCaptureStateAfterCapture as updateChainCaptureStateAfterCaptureShared,
  getCaptureOptionsFromPosition as getCaptureOptionsFromPositionShared,
} from './rules/captureChainEngine';
import {
  PerTurnState,
  advanceGameForCurrentPlayer,
  updatePerTurnStateAfterMove as updatePerTurnStateAfterMoveTurn,
  TurnEngineDeps,
  TurnEngineHooks,
} from './turn/TurnEngine';

/**
 * Internal state for enforcing mandatory chain captures during the capture phase.
 *
 * This is intentionally kept out of the wire-level GameState so we can evolve
 * the representation without breaking clients. It is roughly modeled after the
 * Rust engine's `ChainCaptureState` and is used only inside GameEngine.
 *
 * The concrete shape is shared with the rules/captureChainEngine module; we
 * keep the Ts* aliases here to preserve existing semantics and comments while
 * centralising the implementation.
 */
type TsChainCaptureState = ChainCaptureState;

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;

declare const clearTimeout: (timer: any) => void;

// Using a simple UUID generator for now
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

export class GameEngine {
  private gameState: GameState;
  private boardManager: BoardManager;
  private ruleEngine: RuleEngine;
  private moveTimers: Map<number, any> = new Map();
  private interactionManager: PlayerInteractionManager | undefined;
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
   * Per-turn placement state: when a ring placement occurs, we track that
   * fact and remember which stack must be moved this turn. This mirrors
   * the sandbox engine’s per-turn fields but remains internal to the
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

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true,
    interactionManager?: PlayerInteractionManager
  ) {
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    this.interactionManager = interactionManager;

    const config = BOARD_CONFIGS[boardType];

    this.gameState = {
      id: gameId,
      boardType,
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

    return {
      ...state,
      board: clonedBoard,
      moveHistory: [...state.moveHistory],
      history: [...state.history],
      players: state.players.map((p) => ({ ...p })),
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
    void this.processDisconnectedRegions;
    void this.hasValidActions;
    void this.processForcedElimination;
    void this.getAdjacentPositions;
    void this.nextPlayer;
    void this.getValidLineProcessingMoves;
    void this.getValidTerritoryProcessingMoves;
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

    const validation = this.ruleEngine.validateMove(fullMove, this.gameState);
    if (!validation) {
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

      // Record a structured history entry for the decision so parity/debug
      // tooling sees the same trace format as for other canonical moves.
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      return {
        success: true,
        gameState: this.getGameState(),
      };
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
      this.updateChainCaptureStateAfterCapture(fullMove, capturedCapHeight);

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
          // Chain is exhausted; clear state and fall through to normal
          // post-move processing.
          this.chainCaptureState = undefined;
        }
      } else {
        // Defensive: if we somehow lack a chain state after a capture
        // segment, clear it and treat this as a standalone capture.
        this.chainCaptureState = undefined;
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

    await this.processAutomaticConsequences();

    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      const endResult = this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');

      // Even when the game ends, record a structured history entry for the
      // canonical move that produced the terminal state so parity/debug
      // tooling sees a complete move-by-move trace.
      this.appendHistoryEntry(beforeStateForHistory, fullMove);

      return {
        success: endResult.success,
        gameResult: endResult.gameResult,
        gameState: this.getGameState(),
      };
    }

    // Advance to next phase/player
    this.advanceGame();

    // Step through automatic bookkeeping phases (line_processing and
    // territory_processing) so the post-move snapshot and history entry
    // reflect the same next-player interactive phase that the sandbox
    // engine records in its traces.
    await this.stepAutomaticPhasesForTesting();

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    // Record a structured history entry for this canonical move.
    this.appendHistoryEntry(beforeStateForHistory, fullMove);

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
        if (move.to) {
          const board = this.gameState.board;
          const existingStack = this.boardManager.getStack(move.to, board);
          const placementCount = Math.max(1, move.placementCount ?? 1);

          const placementRings = new Array(placementCount).fill(move.player);

          let newRings: number[];
          if (existingStack && existingStack.rings.length > 0) {
            // Placing on an existing stack: new rings sit on top
            newRings = [...placementRings, ...existingStack.rings];
          } else {
            // Placing on an empty space
            newRings = placementRings;
          }

          const newStack: RingStack = {
            position: move.to,
            rings: newRings,
            stackHeight: newRings.length,
            capHeight: calculateCapHeight(newRings),
            controllingPlayer: newRings[0],
          };

          this.boardManager.setStack(move.to, newStack, board);

          // Update player state: decrement rings in hand by placementCount,
          // clamped defensively to avoid going below zero.
          const player = this.gameState.players.find((p) => p.playerNumber === move.player);
          if (player && player.ringsInHand > 0) {
            const toSpend = Math.min(placementCount, player.ringsInHand);
            player.ringsInHand -= toSpend;
          }
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
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          if (!stack) {
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

          // Rule Reference: Section 4.2.1 - Leave marker on departure space
          this.boardManager.setMarker(move.from, move.player, this.gameState.board);

          // Process markers along movement path (Section 8.3)
          this.processMarkersAlongPath(move.from, move.to, move.player);

          // Check if landing on same-color marker (Section 8.2 / 8.3.1)
          const landingMarker = this.boardManager.getMarker(move.to, this.gameState.board);
          const landedOnOwnMarker = landingMarker === move.player;
          if (landedOnOwnMarker) {
            // Stacks cannot coexist with markers; remove the marker prior
            // to landing, then apply the self-elimination rule below.
            this.boardManager.removeMarker(move.to, this.gameState.board);
          }

          // If there is an existing stack at the landing position, merge the
          // moving stack into it rather than overwriting it. This mirrors the
          // sandbox movement engine and preserves ring conservation for simple
          // moves that land on occupied spaces.
          const existingDest = this.boardManager.getStack(move.to, this.gameState.board);

          // Remove stack from source
          this.boardManager.removeStack(move.from, this.gameState.board);

          let movedStack: RingStack;
          if (existingDest && existingDest.rings.length > 0) {
            const mergedRings = [...existingDest.rings, ...stack.rings];
            movedStack = {
              position: move.to,
              rings: mergedRings,
              stackHeight: mergedRings.length,
              capHeight: calculateCapHeight(mergedRings),
              controllingPlayer: mergedRings[0],
            };
          } else {
            // Normal movement (no existing stack at landing position)
            movedStack = {
              ...stack,
              position: move.to,
            };
          }

          this.boardManager.setStack(move.to, movedStack, this.gameState.board);

          if (landedOnOwnMarker) {
            // New rule: landing on your own marker with a non-capture move
            // removes that marker and immediately eliminates your top ring,
            // credited toward ring-elimination victory conditions.
            this.eliminateTopRingAt(move.to, move.player);
          }
        }
        break;

      case 'overtaking_capture':
      case 'continue_capture_segment':
        if (move.from && move.to && move.captureTarget) {
          this.performOvertakingCapture(move.from, move.captureTarget, move.to, move.player);
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
  private updateChainCaptureStateAfterCapture(move: Move, capturedCapHeight: number): void {
    this.chainCaptureState = updateChainCaptureStateAfterCaptureShared(
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
   * This helper delegates to the shared captureChainEngine and mirrors
   * the Rust CaptureProcessor::get_available_capture_details logic. It
   * remains the canonical source for chain-continuation options; the
   * unified Move model simply re-labels these as
   * 'continue_capture_segment' during the dedicated 'chain_capture'
   * phase.
   */
  private getCaptureOptionsFromPosition(position: Position, playerNumber: number): Move[] {
    return getCaptureOptionsFromPositionShared(position, playerNumber, this.gameState, {
      boardManager: this.boardManager,
      ruleEngine: this.ruleEngine,
      interactionManager: this.interactionManager,
    });
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

    // Leave marker on departure space
    this.boardManager.setMarker(from, player, this.gameState.board);

    // Process markers along path to target
    this.processMarkersAlongPath(from, captureTarget, player);

    // Process markers along path from target to landing
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

    // Remove capturing stack from source
    this.boardManager.removeStack(from, this.gameState.board);

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
   * Process automatic consequences after a move
   * Rule Reference: Section 4.5 - Post-Movement Processing
   */
  private async processAutomaticConsequences(): Promise<void> {
    // Captures are already processed in applyMove

    // Process territory disconnections (Section 12.2) via the shared
    // canonical helper so that the automatic post-move pipeline uses
    // exactly the same semantics as the standalone
    // territoryProcessing.rules tests and the Python/TS parity layer.
    this.gameState = await processDisconnectedRegionsForCurrentPlayer(this.gameState, {
      boardManager: this.boardManager,
      interactionManager: this.interactionManager,
    });

    // Process line formations (Section 11.2, 11.3) using the shared
    // functional helper so that line semantics remain aligned between
    // the backend GameEngine and the rules-layer tests.
    this.gameState = await processLinesForCurrentPlayer(this.gameState, {
      boardManager: this.boardManager,
      interactionManager: this.interactionManager,
    });
  }

  /**
   * Enumerate canonical line-processing decision moves for the current
   * player. This is a phase-aware helper that mirrors the behaviour of
   * processLineFormations / processOneLine but expressed as Move objects:
   *
   * - process_line: select which line to process first when multiple
   *   candidate lines exist for the moving player.
   * - choose_line_reward: select Option 1 vs Option 2 when an overlength
   *   line admits both outcomes and an interaction manager is present.
   *
   * NOTE: At present, line processing is still driven via PlayerChoice
   * flows inside processAutomaticConsequences. This helper exists to
   * support the unified Move/GamePhase model and future migration of
   * those flows; it is intentionally not wired into makeMove yet.
   */
  private getValidLineProcessingMoves(playerNumber: number): Move[] {
    const moves: Move[] = [];

    const config = BOARD_CONFIGS[this.gameState.boardType];
    const requiredLength = config.lineLength;

    const allLines = this.boardManager.findAllLines(this.gameState.board);
    const playerLines = allLines.filter((line) => line.player === playerNumber);

    if (playerLines.length === 0) {
      return moves;
    }

    // One process_line move per player-owned line, uniquely identified by
    // its index and marker positions. In addition to a stable id, we
    // attach the concrete LineInfo in formedLines[0] so that the Move
    // object by itself fully describes which line is being processed.
    playerLines.forEach((line, index) => {
      const lineKey = line.positions.map((p) => positionToString(p)).join('|');
      moves.push({
        id: `process-line-${index}-${lineKey}`,
        type: 'process_line',
        player: playerNumber,
        formedLines: [line],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.moveHistory.length + 1,
      } as Move);
    });

    // For overlength lines, also surface a choose_line_reward decision so
    // that the unified Move model can express Option 1 vs Option 2 even
    // before the PlayerChoice-based flow is fully migrated. As with
    // process_line, we attach the concrete LineInfo so that tests and
    // parity tooling can identify the target line without re-deriving it
    // from ids.
    const overlengthLines = playerLines.filter((line) => line.positions.length > requiredLength);

    overlengthLines.forEach((line, index) => {
      const lineKey = line.positions.map((p) => positionToString(p)).join('|');

      // Option 1: Collapse All (default/implicit)
      moves.push({
        id: `choose-line-reward-${index}-${lineKey}-all`,
        type: 'choose_line_reward',
        player: playerNumber,
        formedLines: [line],
        // No collapsedMarkers means collapse all
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.moveHistory.length + 1,
      } as Move);

      // Option 2: Minimum Collapse
      // Note: In a full implementation, the player might choose WHICH subset
      // of markers to collapse. For now, we default to the first N markers
      // to match the legacy behaviour.
      const minMarkers = line.positions.slice(0, requiredLength);
      moves.push({
        id: `choose-line-reward-${index}-${lineKey}-min`,
        type: 'choose_line_reward',
        player: playerNumber,
        formedLines: [line],
        collapsedMarkers: minMarkers,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.moveHistory.length + 1,
      } as Move);
    });

    return moves;
  }

  /**
   * Enumerate canonical territory-processing decision moves for the
   * current player. This is a phase-aware helper that mirrors the
   * behaviour of processDisconnectedRegions / processOneDisconnectedRegion
   * but expressed as Move objects:
   *
   * - process_territory_region: choose which eligible disconnected region
   *   to process first when multiple exist.
   *
   * Self-elimination stack choices remain driven by the existing
   * RingEliminationChoice flow for now; a future migration can introduce
   * eliminate_rings_from_stack moves once the corresponding GamePhase
   * contract is fully wired through makeMove().
   */
  private getValidTerritoryProcessingMoves(playerNumber: number): Move[] {
    const moves: Move[] = [];

    const disconnectedRegions = this.boardManager.findDisconnectedRegions(
      this.gameState.board,
      playerNumber
    );

    if (disconnectedRegions.length === 0) {
      return moves;
    }

    const eligibleRegions = disconnectedRegions.filter((region) =>
      this.canProcessDisconnectedRegion(region, playerNumber)
    );

    if (eligibleRegions.length === 0) {
      return moves;
    }

    // One process_territory_region move per eligible disconnected region,
    // with the concrete Territory attached in disconnectedRegions[0] so
    // that the Move fully identifies the region to be processed.
    eligibleRegions.forEach((region, index) => {
      const representative = region.spaces[0];
      const regionKey = representative ? positionToString(representative) : `region-${index}`;
      moves.push({
        id: `process-region-${index}-${regionKey}`,
        type: 'process_territory_region',
        player: playerNumber,
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.moveHistory.length + 1,
      } as Move);
    });

    return moves;
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
      const config = BOARD_CONFIGS[this.gameState.boardType];

      const allLines = this.boardManager.findAllLines(this.gameState.board);
      const playerLines = allLines.filter((line) => line.player === move.player);

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
        // fails, default to the first line for this player, preserving
        // previous "first line wins" behaviour.
        targetLine = playerLines[0];
      }

      // For both process_line and choose_line_reward we currently delegate
      // to processOneLine, which will in turn use the interaction manager
      // (when present) to choose Option 1 vs Option 2 for overlength lines.
      await this.processOneLine(targetLine, config.lineLength);

      // After processing one line, re-check whether any further lines
      // exist for the same player. If so, stay in line_processing so the
      // client/AI can submit another decision Move. Otherwise, advance to
      // territory_processing to handle any disconnections created by the
      // collapse.
      const remainingLines = this.boardManager
        .findAllLines(this.gameState.board)
        .filter((line) => line.player === move.player);

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

      // Move-driven decision phases: apply the core region-processing
      // consequences but leave mandatory self-elimination to an explicit
      // eliminate_rings_from_stack Move surfaced via RuleEngine.
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

      // Apply geometric territory consequences only; no self-elimination
      // occurs here in move-driven mode.
      this.processDisconnectedRegionCore(targetRegion, movingPlayer);

      // Record that this player now owes a mandatory self-elimination
      // decision before their territory_processing cycle can end.
      this.pendingTerritorySelfElimination = true;

      // After processing this region, recompute decision moves in a
      // synthetic territory_processing view so we can determine whether
      // additional region decisions or explicit elimination decisions
      // remain for this player.
      const tempState: GameState = {
        ...this.gameState,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
      };

      const nextDecisionMoves = this.ruleEngine.getValidMoves(tempState);
      const remainingRegionMoves = nextDecisionMoves.filter(
        (m) => m.type === 'process_territory_region'
      );
      const eliminationMoves = nextDecisionMoves.filter(
        (m) => m.type === 'eliminate_rings_from_stack'
      );

      if (remainingRegionMoves.length > 0 || eliminationMoves.length > 0) {
        // Stay in the interactive territory_processing phase so the
        // client/AI can submit another decision Move.
        this.gameState.currentPhase = 'territory_processing';
      } else {
        // No further territory decisions remain; hand control back to the
        // normal turn engine.
        this.advanceGame();
        this.stepAutomaticPhasesForTesting();
      }
    } else if (move.type === 'eliminate_rings_from_stack') {
      const playerNumber = move.player;

      // This explicit elimination satisfies any pending
      // self-elimination requirement for the current
      // territory_processing cycle.
      this.pendingTerritorySelfElimination = false;

      if (!move.to) {
        return;
      }

      const stack = this.boardManager.getStack(move.to, this.gameState.board);
      if (!stack || stack.controllingPlayer !== playerNumber) {
        return;
      }

      // Delegate to the same core helper used by choice-driven
      // elimination flows so that elimination Moves share identical
      // ring accounting and S-invariant behaviour.
      this.eliminateFromStack(stack, playerNumber);

      // After an explicit elimination decision, treat this as the final
      // step of the current territory_processing phase and advance the
      // turn using the normal turn engine.
      this.advanceGame();
      this.stepAutomaticPhasesForTesting();
    }
  }

  private async processLineFormations(): Promise<void> {
    const config = BOARD_CONFIGS[this.gameState.boardType];

    // Keep processing until no more lines exist
    while (true) {
      const allLines = this.boardManager.findAllLines(this.gameState.board);
      if (allLines.length === 0) break;

      // Only consider lines for the moving player
      const playerLines = allLines.filter((line) => line.player === this.gameState.currentPlayer);
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

      await this.processOneLine(lineToProcess, config.lineLength);
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
  private eliminatePlayerRingOrCap(player: number): void {
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
   * Process disconnected regions with chain reactions
   * Rule Reference: Section 12.2, 12.3 - Territory Disconnection and Chain Reactions
   */
  private async processDisconnectedRegions(): Promise<void> {
    // Legacy helper retained for direct test access and parity with
    // existing GameEngine.territoryDisconnection tests. Internally this
    // now delegates to the shared canonical helper so that both the
    // explicit processDisconnectedRegions call and the automatic
    // post-move pipeline share identical semantics.
    this.gameState = await processDisconnectedRegionsForCurrentPlayer(this.gameState, {
      boardManager: this.boardManager,
      interactionManager: this.interactionManager,
    });
  }

  /**
   * Check if player can process a disconnected region
   * Rule Reference: Section 12.2 - Self-Elimination Prerequisite
   *
   * Player must have at least one ring/cap outside the region before processing
   */
  private canProcessDisconnectedRegion(region: Territory, player: number): boolean {
    const regionPositionSet = new Set(region.spaces.map((pos) => positionToString(pos)));
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player);

    // Check if player has at least one ring/cap outside this region
    for (const stack of playerStacks) {
      const stackPosKey = positionToString(stack.position);
      if (!regionPositionSet.has(stackPosKey)) {
        // Found a stack outside the region
        return true;
      }
    }

    // No stacks outside the region - cannot process
    return false;
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
    // 1. Get border markers to collapse
    const borderMarkers = this.boardManager.getBorderMarkerPositions(
      region.spaces,
      this.gameState.board
    );

    // 2. Eliminate all rings within the region (all colors) BEFORE
    //    collapsing spaces. This mirrors the Rust engine's
    //    core_apply_disconnect_region behaviour, where internal
    //    eliminations are computed from the pre-collapse stacks.
    let totalRingsEliminated = 0;
    for (const pos of region.spaces) {
      const stack = this.boardManager.getStack(pos, this.gameState.board);
      if (stack) {
        totalRingsEliminated += stack.stackHeight;
        this.boardManager.removeStack(pos, this.gameState.board);
      }
    }

    // 3. Collapse all spaces in the region to the moving player's color
    for (const pos of region.spaces) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }

    // 4. Collapse all border markers to the moving player's color
    for (const pos of borderMarkers) {
      this.boardManager.setCollapsedSpace(pos, movingPlayer, this.gameState.board);
    }

    // Update player's territory count (region spaces + border markers)
    const totalTerritoryGained = region.spaces.length + borderMarkers.length;
    this.updatePlayerTerritorySpaces(movingPlayer, totalTerritoryGained);

    // 5. Update elimination counts - ALL eliminated rings count toward moving player
    if (totalRingsEliminated > 0) {
      this.gameState.totalRingsEliminated += totalRingsEliminated;
      if (!this.gameState.board.eliminatedRings[movingPlayer]) {
        this.gameState.board.eliminatedRings[movingPlayer] = 0;
      }
      this.gameState.board.eliminatedRings[movingPlayer] += totalRingsEliminated;

      // Update player state
      this.updatePlayerEliminatedRings(movingPlayer, totalRingsEliminated);
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
      eliminatePlayerRingOrCap: (playerNumber: number) => {
        this.eliminatePlayerRingOrCap(playerNumber);
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

    // Whenever we leave the territory_processing phase for the current
    // player, clear any pending self-elimination flag so the next
    // interactive turn starts with a clean slate.
    if (
      previousPhase === 'territory_processing' &&
      this.gameState.currentPhase !== 'territory_processing'
    ) {
      this.pendingTerritorySelfElimination = false;
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

  /**
   * Check if player has any valid capture moves available
   * Rule Reference: Section 10.1
   */
  private hasValidCaptures(playerNumber: number): boolean {
    // Delegate to RuleEngine for capture generation so that the
    // decision to enter the capture phase stays in sync with the
    // actual overtaking_capture semantics. We construct a lightweight
    // view of the current state with phase forced to 'capture' for the
    // specified player and ask RuleEngine for valid moves.
    const tempState: GameState = {
      ...this.gameState,
      currentPlayer: playerNumber,
      currentPhase: 'capture',
    };

    const moves = this.ruleEngine.getValidMoves(tempState);
    return moves.some((m) => m.type === 'overtaking_capture');
  }

  /**
   * Check if player has any valid actions available
   * Rule Reference: Section 4.4
   */
  private hasValidActions(playerNumber: number): boolean {
    return (
      this.hasValidPlacements(playerNumber) ||
      this.hasValidMovements(playerNumber) ||
      this.hasValidCaptures(playerNumber)
    );
  }

  /**
   * Check if player has any valid placement moves
   * Rule Reference: Section 4.1, 6.1-6.3
   */
  private hasValidPlacements(playerNumber: number): boolean {
    const player = this.gameState.players.find((p) => p.playerNumber === playerNumber);
    if (!player || player.ringsInHand === 0) {
      return false; // No rings in hand to place
    }

    // Check for any empty, non-collapsed spaces
    // For now, we'll do a simple check - in full implementation would check all positions
    // A player can place if they have rings in hand (placement restrictions like movement validation would be checked in the actual move)
    return true; // Simplified - assumes there's usually space to place
  }

  /**
   * Check if player has any valid movement moves
   * Rule Reference: Section 8.1, 8.2
   */
  private hasValidMovements(playerNumber: number): boolean {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);

    if (playerStacks.length === 0) {
      return false; // No stacks to move
    }

    // For each player stack, check if it has any valid moves
    for (const stack of playerStacks) {
      const stackHeight = stack.stackHeight;

      // Check all 8 directions (or 6 for hexagonal)
      const directions = this.getAllDirections();

      for (const direction of directions) {
        // Check if we can move at least stack height in this direction
        let distance = 0;

        for (let step = 1; step <= stackHeight + 5; step++) {
          const nextPos: Position = {
            x: stack.position.x + direction.x * step,
            y: stack.position.y + direction.y * step,
            ...(direction.z !== undefined && { z: (stack.position.z || 0) + direction.z * step }),
          };

          if (!this.boardManager.isValidPosition(nextPos)) {
            break; // Out of bounds
          }

          // Check if this position is blocked (collapsed space or stack)
          if (this.boardManager.isCollapsedSpace(nextPos, this.gameState.board)) {
            break; // Blocked by collapsed space
          }

          const stackAtPos = this.boardManager.getStack(nextPos, this.gameState.board);
          if (stackAtPos) {
            break; // Blocked by another stack
          }

          // This position is reachable
          distance = step;

          // If we've met the minimum distance requirement, we have a valid move
          if (distance >= stackHeight) {
            return true;
          }
        }
      }
    }

    return false; // No valid movements found
  }

  /**
   * Force player to eliminate a cap when blocked with no valid moves
   * Rule Reference: Section 4.4 - Forced Elimination When Blocked
   */
  private processForcedElimination(playerNumber: number): void {
    const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, playerNumber);

    if (playerStacks.length === 0) {
      // No stacks to eliminate from - player forfeits turn
      return;
    }

    // TODO: In full implementation, player should choose which stack
    // For now, eliminate from first stack with a valid cap
    for (const stack of playerStacks) {
      if (stack.capHeight > 0) {
        // Found a stack with a cap, eliminate it
        this.eliminatePlayerRingOrCap(playerNumber);
        return;
      }
    }
  }

  /**
   * Get all movement directions based on board type
   */
  private getAllDirections(): { x: number; y: number; z?: number }[] {
    return getMovementDirectionsForBoardType(this.gameState.boardType);
  }

  /**
   * Get adjacent positions for a given position
   * Uses Moore adjacency (8-direction) for square boards, hexagonal for hex
   */
  private getAdjacentPositions(pos: Position): Position[] {
    const adjacent: Position[] = [];
    const config = BOARD_CONFIGS[this.gameState.boardType];

    if (config.type === 'hexagonal') {
      // Hexagonal adjacency (6 directions)
      const directions = [
        { x: 1, y: 0, z: -1 },
        { x: 0, y: 1, z: -1 },
        { x: -1, y: 1, z: 0 },
        { x: -1, y: 0, z: 1 },
        { x: 0, y: -1, z: 1 },
        { x: 1, y: -1, z: 0 },
      ];

      for (const dir of directions) {
        const newPos: Position = {
          x: pos.x + dir.x,
          y: pos.y + dir.y,
          z: (pos.z || 0) + dir.z,
        };
        if (this.boardManager.isValidPosition(newPos)) {
          adjacent.push(newPos);
        }
      }
    } else {
      // Moore adjacency for square boards (8 directions)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;

          const newPos: Position = {
            x: pos.x + dx,
            y: pos.y + dy,
          };
          if (this.boardManager.isValidPosition(newPos)) {
            adjacent.push(newPos);
          }
        }
      }
    }

    return adjacent;
  }

  private nextPlayer(): void {
    const currentIndex = this.gameState.players.findIndex(
      (p) => p.playerNumber === this.gameState.currentPlayer
    );
    const nextIndex = (currentIndex + 1) % this.gameState.players.length;
    this.gameState.currentPlayer = this.gameState.players[nextIndex].playerNumber;
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
    if (process.env.NODE_ENV === 'test') {
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
    // using the shared captureChainEngine helper so that the options
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
      return this.getValidLineProcessingMoves(playerNumber);
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

          const captureMoves = this.ruleEngine.getValidMoves(tempCaptureState);
          const hasCaptureLocal = captureMoves.some((m) => m.type === 'overtaking_capture');

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
      // stalemate until no stacks are left.
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

        this.eliminatePlayerRingOrCap(playerNumber);
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

      // Automatically apply the first available decision move. This mimics
      // the default behaviour of the legacy automatic pipeline (first line,
      // first region, default elimination) but drives it via the unified
      // Move model.
      const move = moves[0];
      await this.applyDecisionMove(move);
    }
  }
}
