import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  GamePhase,
  GameState,
  GameResult,
  Move,
  Player,
  PlayerType,
  Position,
  RingStack,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
  LineInfo,
  RegionOrderChoice,
  GameHistoryEntry,
  Territory,
} from '../../shared/types/game';
import { isSandboxAiTraceModeEnabled } from '../../shared/utils/envFlags';
import {
  calculateCapHeight,
  calculateDistance,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../shared/engine/core';
import { canProcessTerritoryRegion } from '../../shared/engine/territoryProcessing';
import { LocalAIRng } from '../../shared/engine/localAIMoveSelection';
import { SeededRNG, generateGameSeed } from '../../shared/utils/rng';
import { findAllLinesOnBoard } from './sandboxLines';
import {
  getValidLineProcessingMoves,
  applyLineDecisionMove,
  collapseLineMarkersOnBoard,
} from './sandboxLinesEngine';
import {
  findDisconnectedRegionsOnBoard,
  processDisconnectedRegionOnBoard,
} from './sandboxTerritory';
import {
  enumerateSimpleMovementLandings,
  applyMarkerEffectsAlongPathOnBoard,
} from './sandboxMovement';
import type { MarkerPathHelpers } from './sandboxMovement';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from './sandboxCaptures';
import {
  SandboxMovementEngineHooks,
  handleMovementClickSandbox,
  performCaptureChainSandbox,
  enumerateCaptureSegmentsFromSandbox,
} from './sandboxMovementEngine';
import { forceEliminateCapOnBoard } from './sandboxElimination';
import {
  processDisconnectedRegionsForCurrentPlayerEngine,
  TerritoryInteractionHandler,
  getValidTerritoryProcessingMoves,
  applyTerritoryDecisionMove,
} from './sandboxTerritoryEngine';
import {
  SandboxGameEndHooks,
  checkAndApplyVictorySandbox,
  resolveGlobalStalemateIfNeededSandbox,
} from './sandboxGameEnd';
import {
  SandboxTurnState,
  SandboxTurnHooks,
  startTurnForCurrentPlayerSandbox,
  maybeProcessForcedEliminationForCurrentPlayerSandbox,
  advanceTurnAndPhaseForCurrentPlayerSandbox,
} from './sandboxTurnEngine';
import {
  createHypotheticalBoardWithPlacement,
  enumerateLegalRingPlacements,
  hasAnyLegalMoveOrCaptureFrom,
  PlacementBoardView,
} from './sandboxPlacement';
import { maybeRunAITurnSandbox, SandboxAIHooks } from './sandboxAI';

/**
 * Client-local engine harness for the /sandbox route.
 *
 * Scope (current):
 * - Ring placement on non-collapsed, empty cells with no-dead-placement.
 * - Non-capturing movement with distance \u001e stack height and path/marker rules.
 * - Overtaking captures with mandatory chain continuation, using
 *   validateCaptureSegmentOnBoard for legality and a SandboxInteractionHandler
 *   for capture_direction choices when multiple options exist.
 * - Marker behaviour along movement/capture paths aligned with backend TS and Rust.
 * - Line detection & rewards (collapse markers + cap elimination) mirroring
 *   backend behaviour when no interaction manager is wired.
 * - Forced elimination when a player is fully blocked with no rings in hand.
 * - Territory disconnection chain reactions and victory checks (ring
 *   elimination + territory control) per the compact rules.
 */

export type SandboxPlayerKind = PlayerType; // 'human' | 'ai'

export interface SandboxConfig {
  boardType: BoardType;
  numPlayers: number;
  playerKinds: SandboxPlayerKind[]; // indexed 0..3 for players 1..4
}

/**
 * Interaction handler abstraction for sandbox mode. This mirrors the server
 * PlayerInteractionHandler + PlayerInteractionManager model, but runs entirely
 * on the client. For now we only need PlayerChoice support; as the sandbox
 * grows, this can be expanded.
 */
export interface SandboxInteractionHandler {
  requestChoice<TChoice extends PlayerChoice>(
    choice: TChoice
  ): Promise<PlayerChoiceResponseFor<TChoice>>;
}

export interface ClientSandboxEngineOptions {
  config: SandboxConfig;
  interactionHandler: SandboxInteractionHandler;
  /** Optional test-only flag: when true, the engine may enable additional
   *  trace/replay behaviours for parity harnesses without affecting normal
   *  sandbox UX.
   */
  traceMode?: boolean;
}

export class ClientSandboxEngine {
  private gameState: GameState;
  private interactionHandler: SandboxInteractionHandler;
  // When true, the engine is running under a trace/replay harness. This
  // is currently reserved for future parity-specific behaviour and does
  // not alter normal sandbox rules or AI policy.
  private readonly traceMode: boolean;

  // Per-game RNG for deterministic AI behavior
  private rng: SeededRNG;

  // When non-null, the sandbox game has ended with this result.
  private victoryResult: GameResult | null = null;

  // Internal turn-level state for sandbox per-turn flow.
  private _hasPlacedThisTurn: boolean = false;
  private _mustMoveFromStackKey: string | undefined;

  // Internal selection state for movement. This is intentionally kept off of
  // GameState to avoid diverging the shared type.
  private _selectedStackKey: string | undefined;

  // Internal flag used to distinguish between human-initiated movement
  // (click-driven) and canonical replay via applyCanonicalMove. This lets
  // us reuse the same movement engine while avoiding double history
  // entries for canonical moves.
  private _movementInvocationContext: 'human' | 'canonical' | null = null;

  // Test-only: last logical AI move chosen by maybeRunAITurn. This is used
  // by backend-vs-sandbox debug harnesses to map sandbox actions into a
  // canonical Move shape for comparison against backend getValidMoves.
  private _lastAIMove: Move | null = null;

  // Internal flag used in move-driven line decision phases to indicate that
  // the current player has collapsed a line in a way that requires a
  // mandatory ring elimination via an explicit eliminate_rings_from_stack
  // Move. This mirrors the backend GameEngine.pendingLineRewardElimination
  // flag but remains local to the sandbox engine.
  private _pendingLineRewardElimination: boolean = false;

  // Internal flag used in move-driven territory decision phases to indicate
  // that the current player has processed at least one disconnected region
  // and therefore owes a mandatory self-elimination via an explicit
  // eliminate_rings_from_stack decision. This mirrors the backend
  // GameEngine.pendingTerritorySelfElimination flag but remains local to
  // the sandbox engine.
  private _pendingTerritorySelfElimination: boolean = false;

  // Test-only checkpoint hook used by parity/diagnostic harnesses to capture
  // GameState snapshots at key points inside canonical move application and
  // post-movement processing. When unset, all debugCheckpoint calls are no-ops.
  private _debugCheckpointHook?: ((label: string, state: GameState) => void) | undefined;

  /**
   * Internal helper to record a single capture segment as a canonical
   * Move + GameHistoryEntry, mirroring backend GameEngine semantics.
   * The initial segment in a chain is represented as an
   * 'overtaking_capture' move; all follow-up segments use
   * 'continue_capture_segment'. For the final segment in a chain, the
   * "after" snapshot observed by history includes post-movement
   * automatic consequences (lines, territory, victory, next-player),
   * just like the backend's structured history.
   */
  private async handleCaptureSegmentApplied(info: {
    before: GameState;
    after: GameState;
    from: Position;
    target: Position;
    landing: Position;
    playerNumber: number;
    segmentIndex: number;
    isFinal: boolean;
  }): Promise<void> {
    const moveType: Move['type'] =
      info.segmentIndex === 0 ? 'overtaking_capture' : 'continue_capture_segment';

    const moveNumber = this.gameState.history.length + 1;

    const move: Move = {
      id: '',
      type: moveType,
      player: info.playerNumber,
      from: info.from,
      to: info.landing,
      captureTarget: info.target,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    // For intermediate segments, the GameState snapshot observed by
    // appendHistoryEntry will reflect board state immediately after the
    // segment is applied. For the final segment, performCaptureChainSandbox
    // ensures that onMovementComplete has already run before this callback
    // is invoked, so getGameState() includes post-movement consequences
    // (lines, territory, victory, next-player), matching backend history.
    this.appendHistoryEntry(info.before, move);
  }

  /**
   * Internal helper to record a simple non-capturing movement as a canonical
   * move + history entry, mirroring backend GameEngine semantics for
   * move_stack. This is currently used for human-driven movement clicks;
   * canonical replays continue to record history via applyCanonicalMove.
   */
  private async handleSimpleMoveApplied(info: {
    before: GameState;
    after: GameState;
    from: Position;
    landing: Position;
    playerNumber: number;
  }): Promise<void> {
    const moveNumber = this.gameState.history.length + 1;

    const move: Move = {
      id: '',
      type: 'move_stack',
      player: info.playerNumber,
      from: info.from,
      to: info.landing,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber,
    } as Move;

    this.appendHistoryEntry(info.before, move);
  }

  /**
   * Append a structured history entry for a canonical move applied to the
   * sandbox game state. This mirrors the backend GameEngine
   * appendHistoryEntry but runs entirely client-side and is primarily used
   * by parity/debug tooling.
   */
  private appendHistoryEntry(before: GameState, action: Move): void {
    const after = this.getGameState();
    const progressBefore = computeProgressSnapshot(before);
    const progressAfter = computeProgressSnapshot(after);

    // Normalise moveNumber so that sandbox history always uses a contiguous
    // 1..N sequence regardless of how callers populated Move.moveNumber.
    const nextMoveNumber = this.gameState.history.length + 1;

    const normalizedAction: Move = {
      ...action,
      moveNumber: nextMoveNumber,
    };

    const entry: GameHistoryEntry = {
      moveNumber: nextMoveNumber,
      action: normalizedAction,
      actor: normalizedAction.player,
      phaseBefore: before.currentPhase,
      phaseAfter: after.currentPhase,
      statusBefore: before.gameStatus,
      statusAfter: after.gameStatus,
      progressBefore,
      progressAfter,
      stateHashBefore: hashGameState(before),
      stateHashAfter: hashGameState(after),
      boardBeforeSummary: summarizeBoard(before.board),
      boardAfterSummary: summarizeBoard(after.board),
    };

    this.gameState = {
      ...this.gameState,
      history: [...this.gameState.history, entry],
    };
  }

  constructor(opts: ClientSandboxEngineOptions) {
    const { config, interactionHandler, traceMode } = opts;
    this.interactionHandler = interactionHandler;
    this.traceMode = traceMode ?? isSandboxAiTraceModeEnabled();

    // Initialize with temporary seed; will be set from gameState if provided
    this.rng = new SeededRNG(generateGameSeed());

    const board = this.createEmptyBoard(config.boardType);
    const players: Player[] = Array.from({ length: config.numPlayers }, (_, idx) => {
      const playerNumber = idx + 1;
      const kind = config.playerKinds[idx] ?? 'human';
      return {
        id: `sandbox-${playerNumber}`,
        username: `Player ${playerNumber}`,
        type: kind,
        playerNumber,
        isReady: true,
        timeRemaining: 0,
        aiDifficulty: kind === 'ai' ? 5 : undefined,
        ringsInHand: BOARD_CONFIGS[config.boardType].ringsPerPlayer,
        eliminatedRings: 0,
        territorySpaces: 0,
      };
    });

    const now = new Date();
    const boardConfig = BOARD_CONFIGS[config.boardType];

    // Generate seed for this sandbox game
    const gameSeed = generateGameSeed();
    this.rng = new SeededRNG(gameSeed);

    this.gameState = {
      id: 'sandbox-local',
      boardType: config.boardType,
      rngSeed: gameSeed,
      board,
      players,
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      history: [],
      timeControl: {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      },
      spectators: [],
      gameStatus: 'active',
      createdAt: now,
      lastMoveAt: now,
      isRated: false,
      maxPlayers: config.numPlayers,
      totalRingsInPlay: boardConfig.ringsPerPlayer * config.numPlayers,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor((boardConfig.ringsPerPlayer * config.numPlayers) / 2) + 1,
      territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1,
    };
  }

  /**
   * Test-only helper: register a debug checkpoint hook so parity/diagnostic
   * harnesses can capture GameState snapshots at key points inside canonical
   * move application and post-movement processing.
   */
  public setDebugCheckpointHook(
    hook: ((label: string, state: GameState) => void) | undefined
  ): void {
    this._debugCheckpointHook = hook;
  }

  private debugCheckpoint(label: string): void {
    if (this._debugCheckpointHook) {
      this._debugCheckpointHook(label, this.getGameState());
    }
  }

  /**
   * Return a defensive snapshot of the current GameState.
   *
   * Unlike the earliest sandbox version, this now deep-clones the board's
   * Map/array fields so that parity/debug tooling (and any callers holding
   * onto past snapshots) see stable pre/post views rather than aliases that
   * are mutated by subsequent moves. This mirrors the backend
   * GameEngine.getGameState semantics.
   */
  public getGameState(): GameState {
    const state = this.gameState;
    const board = state.board;

    const clonedBoard: BoardState = {
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
   * When non-null, contains the terminal GameResult for this sandbox game.
   * This mirrors the backend GameContext.victoryState shape so the
   * VictoryModal component can be reused for local games.
   */
  public getVictoryResult(): GameResult | null {
    return this.victoryResult;
  }

  /**
   * Test-only helper: expose the last logical AI move chosen by
   * maybeRunAITurn in a canonical Move shape. This is used by
   * backend-vs-sandbox debug harnesses to validate sandbox AI
   * decisions against backend getValidMoves.
   */
  public getLastAIMoveForTesting(): Move | null {
    return this._lastAIMove ? { ...this._lastAIMove } : null;
  }

  /**
   * Clear any internal movement selection state. This is used by the sandbox
   * UI when it wants to discard a previous selection and treat the next click
   * as a fresh source-selection, keeping BoardView highlights and engine
   * semantics aligned.
   */
  public clearSelection(): void {
    this._selectedStackKey = undefined;
  }

  /**
   * Handle a human click on a board cell in sandbox mode. This is the main
   * entry point for the /sandbox UI, analogous to the backend click-to-move
   * flow in GamePage, but targeting the local GameState instead of the
   * WebSocket server.
   *
   * Ring placement is now routed through the same canonical Move-applier used
   * by parity harnesses so that human and AI actions share a single semantic
   * path. Movement clicks continue to delegate to handleMovementClick.
   */
  public async handleHumanCellClick(pos: Position): Promise<void> {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    if (this.gameState.currentPhase === 'ring_placement') {
      const beforeState = this.getGameState();
      const playerNumber = beforeState.currentPlayer;

      // Preserve placed-on-stack metadata for history, mirroring the backend
      // place_ring representation.
      const key = positionToString(pos);
      const existingBefore = beforeState.board.stacks.get(key);
      const placedOnStack = !!existingBefore && existingBefore.rings.length > 0;

      const moveNumber = beforeState.history.length + 1;

      const move: Move = {
        id: '',
        type: 'place_ring',
        player: playerNumber,
        to: pos,
        placementCount: 1,
        placedOnStack,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;

      // Apply via the canonical Move-applier with no-dead-placement enforced,
      // so that human sandbox placements share the same semantics as AI
      // placements and backend RuleEngine validation.
      const changed = await this.applyCanonicalMoveInternal(move, {
        bypassNoDeadPlacement: false,
      });

      if (!changed) {
        return;
      }

      this.appendHistoryEntry(beforeState, move);
    } else if (this.gameState.currentPhase === 'movement') {
      // Human-driven movement click. Record canonical history via the
      // movement engine hooks without interfering with canonical replays.
      this._movementInvocationContext = 'human';
      try {
        await this.handleMovementClick(pos);
      } finally {
        this._movementInvocationContext = null;
      }
    }
  }

  /**
   * Run a single AI turn in sandbox mode.
   *
   * Behaviour:
   * - In ring_placement: chooses a random legal placement that satisfies
   *   no-dead-placement (if it still has rings in hand).
   * - In movement:
   *   - Prefer an overtaking capture chain when at least one capture
   *     segment exists for any of the current player's stacks.
   *   - Otherwise choose a random simple non-capturing move.
   *
   * This keeps local games progressing while remaining aligned with the
   * backend RuleEngine semantics (movement reachability + capture chains).
   */
  public async maybeRunAITurn(rng?: LocalAIRng): Promise<void> {
    // Use provided RNG if given (for testing), otherwise use instance RNG
    const effectiveRng = rng ?? (() => this.rng.next());

    const hooks: SandboxAIHooks = {
      getPlayerStacks: (playerNumber: number, board: BoardState) =>
        this.getPlayerStacks(playerNumber, board),
      hasAnyLegalMoveOrCaptureFrom: (from: Position, playerNumber: number, board: BoardState) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      enumerateLegalRingPlacements: (playerNumber: number) =>
        this.enumerateLegalRingPlacements(playerNumber),
      createHypotheticalBoardWithPlacement: (
        board: BoardState,
        position: Position,
        playerNumber: number,
        count: number = 1
      ): BoardState =>
        this.createHypotheticalBoardWithPlacement(board, position, playerNumber, count),
      tryPlaceRings: async (position: Position, count: number) =>
        await this.tryPlaceRings(position, count),
      enumerateCaptureSegmentsFrom: (from: Position, playerNumber: number) =>
        this.enumerateCaptureSegmentsFrom(from, playerNumber),
      enumerateSimpleMovementLandings: (playerNumber: number) =>
        this.enumerateSimpleMovementLandings(playerNumber),
      maybeProcessForcedEliminationForCurrentPlayer: () =>
        this.maybeProcessForcedEliminationForCurrentPlayer(),
      handleMovementClick: (position: Position) => this.handleMovementClick(position),
      appendHistoryEntry: (before: GameState, action: Move) =>
        this.appendHistoryEntry(before, action),
      getGameState: () => this.getGameState(),
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      setLastAIMove: (move: Move | null) => {
        this._lastAIMove = move;
      },
      setSelectedStackKey: (key: string | undefined) => {
        this._selectedStackKey = key;
      },
      getMustMoveFromStackKey: () => this._mustMoveFromStackKey,
      applyCanonicalMove: (move: Move) => this.applyCanonicalMove(move),
      hasPendingTerritorySelfElimination: () => this._pendingTerritorySelfElimination,
      hasPendingLineRewardElimination: () => this._pendingLineRewardElimination,
    };

    await maybeRunAITurnSandbox(hooks, effectiveRng);
  }
  /**
   * Get all valid landing positions for the current player from the given
   * source position. This is used by the UI to highlight valid targets.
   */
  public getValidLandingPositionsForCurrentPlayer(from: Position): Position[] {
    const playerNumber = this.gameState.currentPlayer;
    const fromKey = positionToString(from);

    // 1. Check for capture segments
    const captureSegments = this.enumerateCaptureSegmentsFrom(from, playerNumber);
    if (captureSegments.length > 0) {
      return captureSegments.map((seg) => seg.landing);
    }

    // 2. Check for simple movement
    const simpleMoves = this.enumerateSimpleMovementLandings(playerNumber);
    return simpleMoves.filter((m) => m.fromKey === fromKey).map((m) => m.to);
  }

  /**
   * Enumerate legal ring placement positions for the given player, enforcing
   * the same no-dead-placement rule used for human placement clicks.
   *
   * Unlike the earliest sandbox version, placement is now allowed on both
   * empty spaces and existing stacks (non-collapsed only). Stacking uses the
   * same semantics as createHypotheticalBoardWithPlacement.
   */
  private enumerateLegalRingPlacements(playerNumber: number): Position[] {
    const view: PlacementBoardView = {
      isValidPosition: (pos) => this.isValidPosition(pos),
      isCollapsedSpace: (pos, board) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos, board) => this.getMarkerOwner(pos, board),
    };

    return enumerateLegalRingPlacements(
      this.gameState.boardType,
      this.gameState.board,
      playerNumber,
      view
    );
  }

  /**
   * Enumerate simple, non-capturing movement options for the given player.
   * This mirrors the path/occupancy checks in handleMovementClick but treats
   * all legal landing positions as candidates for AI selection.
   */
  private enumerateSimpleMovementLandings(playerNumber: number): {
    fromKey: string;
    to: Position;
  }[] {
    return enumerateSimpleMovementLandings(
      this.gameState.boardType,
      this.gameState.board,
      playerNumber,
      (pos: Position) => this.isValidPosition(pos)
    );
  }

  // === Internal helpers ===

  private createEmptyBoard(boardType: BoardType): BoardState {
    const config = BOARD_CONFIGS[boardType];
    return {
      stacks: new Map<string, RingStack>(),
      markers: new Map(),
      collapsedSpaces: new Map(),
      territories: new Map(),
      formedLines: [],
      eliminatedRings: {},
      size: config.size,
      type: boardType,
    };
  }
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    playerNumber: number,
    count: number = 1
  ): BoardState {
    return createHypotheticalBoardWithPlacement(board, position, playerNumber, count);
  }

  private hasAnyLegalMoveOrCaptureFrom(
    from: Position,
    playerNumber: number,
    board: BoardState
  ): boolean {
    const view: PlacementBoardView = {
      isValidPosition: (pos) => this.isValidPosition(pos),
      isCollapsedSpace: (pos, b) => this.isCollapsedSpace(pos, b),
      getMarkerOwner: (pos, b) => this.getMarkerOwner(pos, b),
    };

    return hasAnyLegalMoveOrCaptureFrom(this.gameState.boardType, board, from, playerNumber, view);
  }

  private getPlayerStacks(
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): RingStack[] {
    const stacks: RingStack[] = [];
    for (const stack of board.stacks.values()) {
      if (stack.controllingPlayer === playerNumber) {
        stacks.push(stack);
      }
    }
    return stacks;
  }

  private enumerateCaptureSegmentsFrom(
    from: Position,
    playerNumber: number
  ): Array<{ from: Position; target: Position; landing: Position }> {
    const board = this.gameState.board;

    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, b: BoardState) => this.isCollapsedSpace(pos, b),
      getMarkerOwner: (pos: Position, b: BoardState) => this.getMarkerOwner(pos, b),
    };

    return enumerateCaptureSegmentsFromBoard(
      this.gameState.boardType,
      board,
      from,
      playerNumber,
      adapters
    );
  }

  private applyCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): void {
    const board = this.gameState.board;
    const landingKey = positionToString(landing);

    const landingMarkerOwner = this.getMarkerOwner(landing, board);
    const landedOnOwnMarker = landingMarkerOwner === playerNumber;

    const adapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (f, t, player, options) =>
        this.applyMarkerEffectsAlongPath(f, t, player, options),
    };

    applyCaptureSegmentOnBoard(board, from, target, landing, playerNumber, adapters);

    const stacksAfterCapture: Map<string, RingStack> = new Map(board.stacks);
    let eliminatedRingsMap = board.eliminatedRings;
    const state = this.gameState;
    let playersAfterCapture = state.players;
    let totalRingsEliminatedDelta = 0;

    if (landedOnOwnMarker) {
      const stackAtLanding = stacksAfterCapture.get(landingKey);
      if (stackAtLanding && stackAtLanding.stackHeight > 0) {
        const [, ...remainingRings] = stackAtLanding.rings;

        if (remainingRings.length > 0) {
          const newStack: RingStack = {
            ...stackAtLanding,
            rings: remainingRings,
            stackHeight: remainingRings.length,
            capHeight: calculateCapHeight(remainingRings),
            controllingPlayer: remainingRings[0],
          };
          stacksAfterCapture.set(landingKey, newStack);
        } else {
          stacksAfterCapture.delete(landingKey);
        }

        const creditedPlayer = playerNumber;
        eliminatedRingsMap = {
          ...eliminatedRingsMap,
          [creditedPlayer]: (eliminatedRingsMap[creditedPlayer] || 0) + 1,
        };
        playersAfterCapture = playersAfterCapture.map((p) =>
          p.playerNumber === creditedPlayer ? { ...p, eliminatedRings: p.eliminatedRings + 1 } : p
        );
        totalRingsEliminatedDelta = 1;
      }
    }

    const nextState: GameState = {
      ...state,
      board: {
        ...board,
        stacks: stacksAfterCapture,
        markers: new Map(board.markers),
        collapsedSpaces: new Map(board.collapsedSpaces),
        eliminatedRings: eliminatedRingsMap,
      },
      players: playersAfterCapture,
      totalRingsEliminated: state.totalRingsEliminated + totalRingsEliminatedDelta,
    };

    this.gameState = nextState;
  }

  /**
   * Test-only helper: perform a concrete capture chain starting from a
   * specified initial segment using the shared sandbox movement engine.
   * This is used by landing-on-own-marker tests so they can exercise the
   * same overtaking semantics without going through click selection.
   */
  private async performCaptureChain(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): Promise<void> {
    const hooks: SandboxMovementEngineHooks = {
      getGameState: () => this.gameState,
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board?: BoardState) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos: Position, board?: BoardState) => this.getMarkerOwner(pos, board),
      getPlayerStacks: (p: number, board: BoardState) => this.getPlayerStacks(p, board),
      setMarker: (pos: Position, p: number, board: BoardState) => this.setMarker(pos, p, board),
      collapseMarker: (pos: Position, p: number, board: BoardState) =>
        this.collapseMarker(pos, p, board),
      flipMarker: (pos: Position, p: number, board: BoardState) => this.flipMarker(pos, p, board),
      onMovementComplete: async () => {
        await this.advanceAfterMovement();
      },
    };

    await performCaptureChainSandbox(hooks, from, target, landing, playerNumber);
  }

  private createSandboxTurnHooks(): SandboxTurnHooks {
    return {
      enumerateLegalRingPlacements: (state, playerNumber) =>
        this.enumerateLegalRingPlacements(playerNumber),
      hasAnyLegalMoveOrCaptureFrom: (state, from, playerNumber, board) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      getPlayerStacks: (state, playerNumber, board) => this.getPlayerStacks(playerNumber, board),
      forceEliminateCap: (state, playerNumber) => {
        // forceEliminateCap mutates this.gameState; adapt to functional
        // style by operating on the provided state so callers can treat
        // the hook as a pure function of its arguments.
        this.gameState = state;
        this.forceEliminateCap(playerNumber);
        return this.gameState;
      },
      checkAndApplyVictory: (state) => {
        this.gameState = state;
        this.checkAndApplyVictory();
        return this.gameState;
      },
    };
  }

  private startTurnForCurrentPlayer(): void {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const { state, turnState } = startTurnForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = state;
    this._hasPlacedThisTurn = turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = turnState.mustMoveFromStackKey;
  }

  private maybeProcessForcedEliminationForCurrentPlayer(): boolean {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const result = maybeProcessForcedEliminationForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = result.state;
    this._hasPlacedThisTurn = result.turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = result.turnState.mustMoveFromStackKey;

    return result.eliminated;
  }

  private advanceTurnAndPhaseForCurrentPlayer(): void {
    const hooks = this.createSandboxTurnHooks();

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey,
    };

    const { state, turnState } = advanceTurnAndPhaseForCurrentPlayerSandbox(
      this.gameState,
      turnStateBefore,
      hooks
    );

    this.gameState = state;
    this._hasPlacedThisTurn = turnState.hasPlacedThisTurn;
    this._mustMoveFromStackKey = turnState.mustMoveFromStackKey;
  }

  private forceEliminateCap(playerNumber: number): void {
    const { board, players } = this.gameState;
    const stacks = this.getPlayerStacks(playerNumber, board);

    const result = forceEliminateCapOnBoard(board, players, playerNumber, stacks);
    if (result.totalRingsEliminatedDelta <= 0) {
      return;
    }

    this.gameState = {
      ...this.gameState,
      board: result.board,
      players: result.players,
      totalRingsEliminated: this.gameState.totalRingsEliminated + result.totalRingsEliminatedDelta,
    };
  }

  private getNextPlayerNumber(current: number): number {
    const players = this.gameState.players;
    const idx = players.findIndex((p) => p.playerNumber === current);
    const nextIdx = (idx + 1) % players.length;
    return players[nextIdx].playerNumber;
  }

  /**
   * Local helper to parse a position string produced by positionToString
   * back into a Position object. This mirrors the backend stringToPosition
   * but is kept local to avoid pulling in additional shared helpers.
   */
  private stringToPositionLocal(posStr: string): Position {
    const parts = posStr.split(',').map(Number);
    if (parts.length === 2) {
      const [x, y] = parts;
      return { x, y };
    }
    if (parts.length === 3) {
      const [x, y, z] = parts;
      return { x, y, z };
    }
    // Defensive fallback; should not occur if positionToString format is
    // consistent.
    return { x: 0, y: 0 };
  }

  /**
   * Local position validity check mirroring BoardManager semantics so we can
   * safely use shared capture helpers on the client.
   */
  private isValidPosition(pos: Position): boolean {
    const config = BOARD_CONFIGS[this.gameState.boardType];
    if (this.gameState.boardType === 'hexagonal') {
      const radius = config.size - 1;
      const x = pos.x;
      const y = pos.y;
      const z = pos.z !== undefined ? pos.z : -x - y;
      const distance = Math.max(Math.abs(x), Math.abs(y), Math.abs(z));
      return distance <= radius;
    }
    // Square boards: 0..size-1 grid
    return pos.x >= 0 && pos.x < config.size && pos.y >= 0 && pos.y < config.size;
  }

  private isCollapsedSpace(position: Position, board: BoardState = this.gameState.board): boolean {
    const key = positionToString(position);
    return board.collapsedSpaces.has(key);
  }

  private getMarkerOwner(
    position: Position,
    board: BoardState = this.gameState.board
  ): number | undefined {
    const key = positionToString(position);
    const marker = board.markers.get(key);
    return marker?.player;
  }

  private setMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);

    // Mirror backend BoardManager.setMarker semantics:
    // - Do not place markers on collapsed territory.
    // - Ensure stack+marker exclusivity by removing any stack at this key
    //   before writing the marker.
    if (board.collapsedSpaces.has(key)) {
      return;
    }

    if (board.stacks.has(key)) {
      board.stacks.delete(key);
    }

    board.markers.set(key, {
      player: playerNumber,
      position,
      type: 'regular',
    });
  }

  private flipMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);
    const existing = board.markers.get(key);
    if (existing && existing.player !== playerNumber) {
      board.markers.set(key, {
        player: playerNumber,
        position,
        type: 'regular',
      });
    }
  }

  private collapseMarker(
    position: Position,
    playerNumber: number,
    board: BoardState = this.gameState.board
  ): void {
    const key = positionToString(position);
    // When a marker collapses to territory, the cell becomes
    // exclusive territory: no stacks or markers may remain.
    board.markers.delete(key);
    board.stacks.delete(key);
    board.collapsedSpaces.set(key, playerNumber);
  }

  /**
   * Find all marker lines on the board for all players. Mirrors
   * BoardManager.findAllLines; only returns lines of at least the
   * configured minimum length.
   *
   * This is primarily used by test-only harnesses (e.g.
   * ClientSandboxEngine.lines.test.ts) via an `any`-cast of the engine
   * instance, so it intentionally remains a private helper rather than
   * part of the public API surface.
   */
  private findAllLines(board: BoardState): LineInfo[] {
    return findAllLinesOnBoard(
      this.gameState.boardType,
      board,
      (pos: Position) => this.isValidPosition(pos),
      (posStr: string) => this.stringToPositionLocal(posStr)
    );
  }

  /**
   * Collapse all markers in `positions` to the given player's territory,
   * removing any stacks there and updating the player's territorySpaces
   * counter. Analogue of GameEngine.collapseLineMarkers.
   */
  private collapseLineMarkers(positions: Position[], playerNumber: number): void {
    const board = this.gameState.board;
    const collapsedKeys = new Set<string>();

    for (const pos of positions) {
      const key = positionToString(pos);
      collapsedKeys.add(key);
      board.markers.delete(key);
      board.stacks.delete(key);
      board.collapsedSpaces.set(key, playerNumber);
    }

    const territoryGain = collapsedKeys.size;
    const updatedPlayers = this.gameState.players.map((p) =>
      p.playerNumber === playerNumber
        ? { ...p, territorySpaces: p.territorySpaces + territoryGain }
        : p
    );

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        markers: new Map(board.markers),
        stacks: new Map(board.stacks),
        collapsedSpaces: new Map(board.collapsedSpaces),
      },
      players: updatedPlayers,
    };
  }

  /**
   * Apply marker effects for a move or capture segment from `from` to `to`:
   * - Leave a marker on the departure space.
   * - For intermediate spaces:
   *   - Opponent markers flip to the mover's color.
   *   - Own markers collapse into territory.
   * - On landing, remove same-color marker (cannot coexist with a stack).
   *
   * The optional `options` parameter is forwarded to the shared
   * applyMarkerEffectsAlongPathOnBoard helper so that callers (notably
   * overtaking captures) can opt out of placing a departure marker on
   * intermediate stacks such as the capture target. This keeps sandbox
   * marker-path semantics aligned with the backend GameEngine for both
   * movement and capture legs.
   */
  private applyMarkerEffectsAlongPath(
    from: Position,
    to: Position,
    playerNumber: number,
    options?: { leaveDepartureMarker?: boolean }
  ): void {
    const board = this.gameState.board;

    const helpers: MarkerPathHelpers = {
      setMarker: (pos, player, b) => this.setMarker(pos, player, b),
      collapseMarker: (pos, player, b) => this.collapseMarker(pos, player, b),
      flipMarker: (pos, player, b) => this.flipMarker(pos, player, b),
    };

    applyMarkerEffectsAlongPathOnBoard(board, from, to, playerNumber, helpers, options);
  }

  // Removed unused handleRingPlacementClick helper to fix TS6133

  private async handleMovementClick(position: Position): Promise<void> {
    const board = this.gameState.board;
    const key = positionToString(position);
    const stackAtPos = board.stacks.get(key);

    // Synchronous selection / deselection logic to preserve existing
    // click-to-select semantics used by tests and the UI.
    if (!this.isValidPosition(position)) {
      this._selectedStackKey = undefined;
      return;
    }

    if (!this._selectedStackKey) {
      // If clicking on a stack belonging to the current player, select it.
      if (stackAtPos && stackAtPos.controllingPlayer === this.gameState.currentPlayer) {
        this._selectedStackKey = key;
      }
      return;
    }

    // Clicking the same cell clears selection.
    if (key === this._selectedStackKey) {
      this._selectedStackKey = undefined;
      return;
    }

    const isCanonicalReplay = this._movementInvocationContext === 'canonical';

    // Delegate actual movement / capture application to the shared
    // sandboxMovementEngine, wiring hooks back into this engine.
    const hooks: SandboxMovementEngineHooks = {
      getGameState: () => this.gameState,
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board?: BoardState) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos: Position, board?: BoardState) => this.getMarkerOwner(pos, board),
      getPlayerStacks: (playerNumber: number, board: BoardState) =>
        this.getPlayerStacks(playerNumber, board),
      setMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.setMarker(pos, playerNumber, board),
      collapseMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.collapseMarker(pos, playerNumber, board),
      flipMarker: (pos: Position, playerNumber: number, board: BoardState) =>
        this.flipMarker(pos, playerNumber, board),
      chooseCaptureSegment: async (
        options: Array<{ from: Position; target: Position; landing: Position }>
      ) => {
        if (options.length <= 1) {
          return options[0];
        }

        const playerNumber = this.gameState.currentPlayer;
        const choice = {
          id: `sandbox-capture-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          gameId: this.gameState.id,
          playerNumber,
          type: 'capture_direction' as const,
          prompt: 'Select capture direction',
          options: options.map((opt) => ({
            targetPosition: opt.target,
            landingPosition: opt.landing,
            capturedCapHeight:
              this.gameState.board.stacks.get(positionToString(opt.target))?.capHeight ?? 0,
          })),
        };

        const response = await this.interactionHandler.requestChoice(choice as any);
        const selected = options.find((opt) => {
          const o = (response as any).selectedOption;
          return (
            o &&
            positionToString(o.targetPosition) === positionToString(opt.target) &&
            positionToString(o.landingPosition) === positionToString(opt.landing)
          );
        });

        return selected ?? options[0];
      },
      // For human-driven movement, record canonical history for both
      // capture segments and simple moves via the movement engine hooks.
      ...(isCanonicalReplay
        ? {}
        : {
            onCaptureSegmentApplied: (info: any) => this.handleCaptureSegmentApplied(info),
            onSimpleMoveApplied: (info: any) => this.handleSimpleMoveApplied(info),
          }),
      onMovementComplete: async () => {
        await this.advanceAfterMovement();
      },
    };

    const result = await handleMovementClickSandbox(hooks, this._selectedStackKey, position);
    this._selectedStackKey = result.nextSelectedFromKey;
  }

  // Removed unused advanceAfterPlacement helper to fix TS6133

  private async advanceAfterMovement(): Promise<void> {
    this.debugCheckpoint('before-advanceAfterMovement');

    // Post-movement consequences for the player who just moved: lines,
    // territory disconnections, and victory checks.
    await this.processLinesForCurrentPlayer();
    this.debugCheckpoint('after-processLinesForCurrentPlayer');

    // When in traceMode and a decision phase was set, stop here so the
    // current player remains active to make the decision move on their
    // next AI turn. This keeps sandbox traces aligned with backend
    // move-driven decision phases.
    if (this.gameState.currentPhase === 'line_processing') {
      return;
    }

    await this.processDisconnectedRegionsForCurrentPlayer();
    this.debugCheckpoint('after-processDisconnectedRegionsForCurrentPlayer');

    // Same traceMode handling for territory_processing
    if (this.gameState.currentPhase === 'territory_processing') {
      return;
    }

    this.checkAndApplyVictory();
    this.debugCheckpoint('after-checkAndApplyVictory');

    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    // Hand off to the shared turn/phase sequencer so that sandbox turn
    // rotation and forced elimination mirror backend semantics. By
    // normalising the phase to territory_processing here we are telling
    // the shared helper that all automatic bookkeeping for this player
    // (lines and territory) has completed.
    this.gameState = {
      ...this.gameState,
      currentPhase: 'territory_processing',
    };

    this.advanceTurnAndPhaseForCurrentPlayer();
    this.debugCheckpoint('after-advanceTurnAndPhaseForCurrentPlayer');
  }

  /**
   * Process all disconnected regions for the current player using the
   * sandboxTerritory engine helper. This mirrors the backend GameEngine
   * behaviour, including RegionOrderChoice handling when multiple eligible
   * regions exist, while keeping the implementation purely functional.
   */
  private async processDisconnectedRegionsForCurrentPlayer(): Promise<void> {
    // Guard: when exactly one player has stacks on the board, there is no
    // meaningful notion of a "disconnected" region for self-elimination
    // purposes. The backend territory processor is only exercised in
    // practice once multiple players have on-board presence; without this
    // guard the sandbox can incorrectly treat an early sparse position
    // (e.g. after Player 1's very first move in a mixed human/AI game) as
    // a fully disconnected region and immediately collapse the entire
    // board to territory, triggering an early victory.
    const activePlayers = new Set<number>();
    for (const stack of this.gameState.board.stacks.values()) {
      activePlayers.add(stack.controllingPlayer);
    }
    // Mirror backend territory-processing semantics: only short-circuit when
    // the *moving player* is the sole player with stacks on the board.
    // When exactly one player has stacks but that player is not the mover,
    // we must still consult findDisconnectedRegions so that scenarios like
    // FAQ Q23 (control without on-board presence) are handled consistently
    // across engines.
    if (activePlayers.size === 1 && activePlayers.has(this.gameState.currentPlayer)) {
      return;
    }

    if (this.traceMode) {
      const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
      const eligible = disconnected.filter((region) =>
        this.canProcessDisconnectedRegion(
          region.spaces,
          this.gameState.currentPlayer,
          this.gameState.board
        )
      );

      if (eligible.length > 0) {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'territory_processing',
        };
        return;
      }
    }

    const nextState = await processDisconnectedRegionsForCurrentPlayerEngine(
      this.gameState,
      this.interactionHandler as unknown as TerritoryInteractionHandler,
      (regionSpaces: Position[], playerNumber: number, state: GameState) =>
        this.canProcessDisconnectedRegion(regionSpaces, playerNumber, state.board)
    );

    this.gameState = nextState;
  }

  /**
   * Self-elimination prerequisite: the current player must have at least
   * one stack outside the disconnected region before it can be processed.
   */
  private canProcessDisconnectedRegion(
    regionSpaces: Position[],
    playerNumber: number,
    board: BoardState
  ): boolean {
    // Thin wrapper around the shared self-elimination prerequisite helper so
    // sandbox territory gating stays aligned with backend GameEngine /
    // RuleEngine semantics. We construct a transient Territory wrapper here
    // purely for gating; controllingPlayer is not inspected by the helper.
    const region: Territory = {
      spaces: regionSpaces,
      controllingPlayer: playerNumber,
      isDisconnected: true,
    };

    return canProcessTerritoryRegion(board, region, { player: playerNumber });
  }

  /**
   * Enumerate canonical territory-processing decision Moves for the
   * current player in the sandbox. This mirrors the backend
   * GameEngine.getValidTerritoryProcessingMoves helper and is primarily
   * used by parity/debug tooling and future sandbox AI/decision flows:
   *
   * - process_territory_region: choose which eligible disconnected
   *   region to process first, subject to the self-elimination
   *   prerequisite from §12.2 / FAQ Q23.
   *
   * For normal human-driven sandbox games, territory collapse continues
   * to be driven automatically from advanceAfterMovement; this helper is
   * intended for canonical replay and advanced parity harnesses.
   */
  private getValidTerritoryProcessingMovesForCurrentPlayer(): Move[] {
    const moves: Move[] = [];
    const movingPlayer = this.gameState.currentPlayer;
    const board = this.gameState.board;

    const disconnected = findDisconnectedRegionsOnBoard(board);
    if (!disconnected || disconnected.length === 0) {
      return moves;
    }

    const eligible = disconnected.filter((region) =>
      this.canProcessDisconnectedRegion(region.spaces, movingPlayer, board)
    );

    eligible.forEach((region, index) => {
      if (!region.spaces || region.spaces.length === 0) {
        return;
      }

      const representative = region.spaces[0];
      const regionKey = representative ? positionToString(representative) : `region-${index}`;

      moves.push({
        id: `process-region-${index}-${regionKey}`,
        type: 'process_territory_region',
        player: movingPlayer,
        disconnectedRegions: [region],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.history.length + 1,
      } as Move);
    });

    return moves;
  }

  /**
   * Enumerate explicit self-elimination decision Moves for the current
   * player in the sandbox. This mirrors the backend
   * RuleEngine.getValidEliminationDecisionMoves helper and is primarily
   * used by parity/debug tooling:
   *
   * - eliminate_rings_from_stack: choose which controlled stack/cap to
   *   self-eliminate from when an elimination is required.
   *
   * For now, this helper does not alter sandbox turn flow; it simply
   * exposes the canonical decision surface so tests and future UI/AI
   * can treat elimination as a Move selection problem.
   */
  private getValidEliminationDecisionMovesForCurrentPlayer(): Move[] {
    // console.log('DEBUG: getValidEliminationDecisionMovesForCurrentPlayer this:', this, 'getPlayerStacks:', this.getPlayerStacks);
    const moves: Move[] = [];

    const pendingTerritory = this._pendingTerritorySelfElimination;
    const pendingLineReward = this._pendingLineRewardElimination;

    // Explicit elimination decisions are only legal when a self-elimination
    // debt is outstanding from either a prior territory decision or a
    // line-reward decision, mirroring the backend GameEngine flags.
    if (!pendingTerritory && !pendingLineReward) {
      return moves;
    }

    const movingPlayer = this.gameState.currentPlayer;
    const board = this.gameState.board;

    // Territory-origin self-elimination still obeys the region eligibility
    // prerequisite; line-reward eliminations do not depend on disconnected
    // regions.
    if (pendingTerritory) {
      const disconnected = findDisconnectedRegionsOnBoard(board);
      if (
        disconnected &&
        disconnected.some((region) =>
          this.canProcessDisconnectedRegion(region.spaces, movingPlayer, board)
        )
      ) {
        return moves;
      }
    }

    const stacks = this.getPlayerStacks(movingPlayer, board);
    if (stacks.length === 0) {
      return moves;
    }

    stacks.forEach((stack) => {
      const key = positionToString(stack.position);
      const capHeight = calculateCapHeight(stack.rings);
      if (capHeight <= 0) {
        return;
      }

      moves.push({
        id: `eliminate-${key}`,
        type: 'eliminate_rings_from_stack',
        player: movingPlayer,
        to: stack.position,
        eliminatedRings: [{ player: movingPlayer, count: capHeight }],
        eliminationFromStack: {
          position: stack.position,
          capHeight,
          totalHeight: stack.stackHeight,
        },
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.history.length + 1,
      } as Move);
    });

    return moves;
  }

  /**
   * Apply ring-elimination and territory-control victory checks after
   * post-movement processing. When a winner is found, the sandbox game
   * is marked as completed and subsequent moves are ignored.
   */
  private checkAndApplyVictory(): void {
    const hooks: SandboxGameEndHooks = {
      enumerateLegalRingPlacements: (playerNumber: number) =>
        this.enumerateLegalRingPlacements(playerNumber),
    };

    // Delegate stalemate resolution + victory detection to the shared
    // sandbox game-end helpers so that semantics stay aligned across
    // hosts. This mirrors the backend GameEngine.checkGameEnd flow.
    const before = this.gameState;
    const { state, result } = checkAndApplyVictorySandbox(this.gameState, hooks);

    // Test-only diagnostic logging: when a victory is detected, emit a
    // compact snapshot so we can understand why gameStatus flipped to
    // 'completed' in early-turn scenarios (e.g. mixedPlayers tests).
    if (
      result &&
      typeof process !== 'undefined' &&
      !!(process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      // eslint-disable-next-line no-console
      console.log('[ClientSandboxEngine Victory Debug]', {
        reason: result.reason,
        currentPlayerBefore: before.currentPlayer,
        currentPhaseBefore: before.currentPhase,
        gameStatusBefore: before.gameStatus,
        currentPlayerAfter: state.currentPlayer,
        currentPhaseAfter: state.currentPhase,
        gameStatusAfter: state.gameStatus,
        players: before.players.map((p) => ({
          playerNumber: p.playerNumber,
          type: p.type,
          ringsInHand: p.ringsInHand,
          eliminatedRings: p.eliminatedRings,
          territorySpaces: p.territorySpaces,
        })),
        stacks: Array.from(before.board.stacks.entries()).map(([key, stack]) => ({
          key,
          height: stack.stackHeight,
          cap: stack.capHeight,
          rings: stack.rings,
        })),
      });
    }

    this.gameState = state;

    // Normalise terminal states so that completed games are never left in
    // the dedicated line/territory decision phases. This mirrors the backend
    // GameEngine.applyDecisionMove elimination handling and keeps both
    // parity traces and UI-facing games in a stable 'ring_placement' phase
    // after victory.
    if (
      this.gameState.gameStatus !== 'active' &&
      (this.gameState.currentPhase === 'territory_processing' ||
        this.gameState.currentPhase === 'line_processing')
    ) {
      this.gameState = {
        ...this.gameState,
        currentPhase: 'ring_placement',
      };
    }

    if (!result) {
      return;
    }

    this.victoryResult = result;
  }

  /**
   * Detect and process marker lines for the current player.
   *
   * For human-driven games, this now routes line resolution through the
   * same canonical Move helpers used for AI and backend trace replays:
   *
   * - Exact-length lines: apply a `process_line` Move that collapses all
   *   markers in the line and eliminates a cap (via
   *   applyCanonicalProcessLine).
   * - Overlength lines: apply a `process_line` Move that collapses only
   *   the minimum required markers with no elimination, matching the
   *   default sandbox behaviour (no line_reward_option choice yet).
   *
   * This ensures that all line-processing effects are both driven by and
   * recorded as canonical Moves, keeping sandbox history aligned with
   * backend semantics while preserving the existing automatic behaviour
   * when no explicit line-order/reward decisions are exposed in the UI.
   */
  /**
   * Enumerate canonical line-processing decision Moves for the current
   * player in the sandbox. This mirrors the backend
   * GameEngine.getValidLineProcessingMoves helper and is primarily used
   * by parity/debug tooling and future sandbox PlayerChoice flows:
   *
   * - process_line: select which detected line to process next.
   * - choose_line_reward: for overlength lines, select Option 1 vs
   *   Option 2 style rewards (currently only Option 1 is modelled
   *   explicitly via applyCanonicalChooseLineReward).
   *
   * For now, ClientSandboxEngine continues to auto-select the first
   * process_line Move when resolving lines internally; this helper
   * exists so tests and future UI can see the same decision surface
   * that the backend exposes via GameEngine.getValidMoves during the
   * line_processing phase.
   */
  private getValidLineProcessingMovesForCurrentPlayer(): Move[] {
    const moves: Move[] = [];
    const boardType = this.gameState.boardType;
    const requiredLength = BOARD_CONFIGS[boardType].lineLength;
    const currentPlayer = this.gameState.currentPlayer;

    const board = this.gameState.board;
    const allLines = this.findAllLines(board);
    const playerLines = allLines.filter((line) => line.player === currentPlayer);

    playerLines.forEach((line, index) => {
      const lineKey = line.positions.map((p) => positionToString(p)).join('|');

      // Base decision: which line to process.
      moves.push({
        id: `process-line-${index}-${lineKey}`,
        type: 'process_line',
        player: currentPlayer,
        formedLines: [line],
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.history.length + 1,
      } as Move);

      // Overlength lines admit a reward choice on the backend. We surface
      // the same decision Move shape here even though the current sandbox
      // UI still defaults to "minimum collapse, no elimination" when
      // resolving lines automatically.
      if (line.length > requiredLength) {
        moves.push({
          id: `choose-line-reward-${index}-${lineKey}`,
          type: 'choose_line_reward',
          player: currentPlayer,
          formedLines: [line],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: this.gameState.history.length + 1,
        } as Move);
      }
    });

    return moves;
  }

  private async processLinesForCurrentPlayer(): Promise<void> {
    // Keep applying lines for the current player until none remain.
    // We use the shared getValidLineProcessingMoves helper to identify
    // candidates and applyLineDecisionMove to execute them.
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const moves = getValidLineProcessingMoves(this.gameState);
      const processLineMoves = moves.filter((m) => m.type === 'process_line');

      if (processLineMoves.length === 0) {
        break;
      }

      // Default behavior: pick the first line.
      // TODO: Surface choice via interactionHandler if multiple lines exist.
      const move = processLineMoves[0];

      if (this.traceMode) {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'line_processing',
        };
        return;
      }

      // Capture state before applying the line so history snapshots have
      // correct before/after semantics.
      const beforeState = this.getGameState();

      // Apply the move using the shared helper.
      const nextState = applyLineDecisionMove(this.gameState, move);

      // Check if state actually changed
      if (hashGameState(nextState) === hashGameState(this.gameState)) {
        break;
      }

      this.gameState = nextState;

      // Record the canonical decision in history so that parity harnesses
      // can replay the exact same sequence into both engines.
      this.appendHistoryEntry(beforeState, move);
    }
  }

  /**
   * Attempt to place one or more rings for the current player at the given
   * position during the ring_placement phase.
   *
   * Canonical rules enforced:
   * - Never place on collapsed spaces.
   * - Multi-ring placement is only allowed on empty cells.
   * - When placing onto an existing stack, at most one ring is added per
   *   placement action (additional requested rings are ignored).
   * - Resulting stack must have at least one legal move/capture
   *   (no-dead-placement).
   *
   * Returns true if the placement was applied, false otherwise.
   */
  public async tryPlaceRings(position: Position, requestedCount: number): Promise<boolean> {
    if (this.gameState.gameStatus !== 'active') {
      return false;
    }

    if (this.gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    if (!this.isValidPosition(position)) {
      return false;
    }

    const board = this.gameState.board;
    const key = positionToString(position);

    // Do not allow placement on collapsed territory.
    if (board.collapsedSpaces.has(key)) {
      return false;
    }

    // Do not allow placement on markers; stacks and markers must never
    // coexist on the same space. This mirrors backend RuleEngine
    // semantics and keeps the S-invariant aligned between engines.
    if (board.markers.has(key)) {
      return false;
    }

    const player = this.gameState.players.find(
      (p) => p.playerNumber === this.gameState.currentPlayer
    );
    if (!player || player.ringsInHand <= 0) {
      return false;
    }

    const existingStack = board.stacks.get(key);
    const isOccupied = !!existingStack && existingStack.rings.length > 0;
    const maxFromHand = player.ringsInHand;

    // Canonical rule: only 1 ring per placement onto an existing stack.
    const maxPerPlacement = isOccupied ? 1 : maxFromHand;
    const effectiveCount = Math.min(Math.max(requestedCount, 1), maxPerPlacement);

    if (effectiveCount <= 0) {
      return false;
    }

    // No-dead-placement: after placing effectiveCount rings here, the resulting
    // stack must have at least one legal move/capture.
    const hypotheticalBoard = this.createHypotheticalBoardWithPlacement(
      board,
      position,
      this.gameState.currentPlayer,
      effectiveCount
    );

    if (
      !this.hasAnyLegalMoveOrCaptureFrom(position, this.gameState.currentPlayer, hypotheticalBoard)
    ) {
      return false;
    }

    const nextStacks = new Map(board.stacks);

    if (isOccupied && existingStack) {
      const addedRings = Array(effectiveCount).fill(this.gameState.currentPlayer);
      const rings = [...addedRings, ...existingStack.rings];
      const newStack: RingStack = {
        ...existingStack,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: this.gameState.currentPlayer,
      };
      nextStacks.set(key, newStack);
    } else {
      const rings = Array(effectiveCount).fill(this.gameState.currentPlayer);
      const newStack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: this.gameState.currentPlayer,
      };
      nextStacks.set(key, newStack);
    }

    const updatedPlayers = this.gameState.players.map((p) =>
      p.playerNumber === this.gameState.currentPlayer
        ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - effectiveCount) }
        : p
    );

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: nextStacks,
      },
      players: updatedPlayers,
      currentPhase: 'movement',
    };

    this._hasPlacedThisTurn = true;
    this._mustMoveFromStackKey = key;
    this._selectedStackKey = key;

    // Process lines immediately after placement, mirroring backend GameEngine
    // behaviour where processAutomaticConsequences runs after every move.
    // This ensures that if a placement completes a line, it is collapsed
    // (and potentially eliminates the placed stack) before the movement phase.
    await this.processLinesForCurrentPlayer();
    if (this.gameState.currentPhase === 'line_processing') {
      return true;
    }

    await this.processDisconnectedRegionsForCurrentPlayer();
    if (this.gameState.currentPhase === 'territory_processing') {
      return true;
    }

    this.checkAndApplyVictory();

    return true;
  }

  /**
   * Internal canonical move-applier used by both AI turns and
   * applyCanonicalMove. It mutates this.gameState according to the given
   * Move and returns true when the move was applied and changed state.
   */
  private async applyCanonicalMoveInternal(
    move: Move,
    opts: { bypassNoDeadPlacement?: boolean } = {}
  ): Promise<boolean> {
    this.debugCheckpoint(`before-applyCanonicalMoveInternal-${move.type}`);
    const beforeHash = hashGameState(this.getGameState());

    // Ensure currentPlayer matches the move's player for the purposes of
    // canonical application.
    if (move.player !== this.gameState.currentPlayer) {
      this.gameState = {
        ...this.gameState,
        currentPlayer: move.player,
      };
    }

    let applied = false;

    switch (move.type) {
      case 'place_ring': {
        if (!move.to) {
          break;
        }
        if (opts.bypassNoDeadPlacement) {
          // Backend-style placement: bypass no-dead-placement gating and
          // directly add rings at the destination, clearing any marker.
          const board = this.gameState.board;
          const key = positionToString(move.to);
          const placementCount = Math.max(1, move.placementCount ?? 1);

          board.markers.delete(key);

          const existingStack = board.stacks.get(key);
          const placementRings = new Array(placementCount).fill(move.player);

          let newRings: number[];
          if (existingStack && existingStack.rings.length > 0) {
            newRings = [...placementRings, ...existingStack.rings];
          } else {
            newRings = placementRings;
          }

          const newStack: RingStack = {
            position: move.to,
            rings: newRings,
            stackHeight: newRings.length,
            capHeight: calculateCapHeight(newRings),
            controllingPlayer: newRings[0],
          };

          const nextStacks = new Map(board.stacks);
          nextStacks.set(key, newStack);

          const updatedPlayers = this.gameState.players.map((p) =>
            p.playerNumber === move.player
              ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - placementCount) }
              : p
          );

          this.gameState = {
            ...this.gameState,
            board: {
              ...board,
              stacks: nextStacks,
            },
            players: updatedPlayers,
            currentPhase: 'movement',
          };

          // When applying canonical placement moves (either from the
          // sandbox AI or backend trace replays), enforce the same
          // must-move semantics as the interactive sandbox: the next
          // movement action for this player must originate from the
          // just-updated stack. This keeps movement reachability and
          // valid-move enumeration aligned with the backend
          // RuleEngine/GameEngine, which only exposes moves from the
          // placed stack immediately after placement.
          this._hasPlacedThisTurn = true;
          this._mustMoveFromStackKey = key;
          this._selectedStackKey = key;

          applied = true;
        } else {
          // Sandbox-style placement: enforce no-dead-placement via
          // tryPlaceRings so AI turns share the same gating as human
          // sandbox interaction.
          const count = Math.max(1, move.placementCount ?? 1);
          const placed = await this.tryPlaceRings(move.to, count);
          applied = placed;
        }
        break;
      }

      case 'skip_placement': {
        this.gameState = {
          ...this.gameState,
          currentPhase: 'movement',
        };
        applied = true;
        break;
      }

      case 'move_ring':
      case 'move_stack': {
        if (!move.from || !move.to) {
          break;
        }
        // Reuse existing movement handler by simulating a source
        // selection followed by a destination click. Mark this invocation
        // as canonical so movement hooks do not emit duplicate history.
        this._movementInvocationContext = 'canonical';
        try {
          this._selectedStackKey = positionToString(move.from);
          await this.handleMovementClick(move.to);
        } finally {
          this._movementInvocationContext = null;
        }
        // handleMovementClick calls advanceAfterMovement internally via hooks,
        // so we don't need to call it again here.
        applied = true;
        break;
      }

      case 'overtaking_capture':
      case 'continue_capture_segment': {
        if (!move.from || !move.to || !move.captureTarget) {
          break;
        }

        // Apply the segment using the same helper used by sandbox-driven chains.
        this.applyCaptureSegment(move.from, move.captureTarget, move.to, move.player);

        // After applying the segment, determine whether the chain can continue
        // from the new landing position under the current board state.
        const continuationOptions = this.enumerateCaptureSegmentsFrom(move.to, move.player);

        if (continuationOptions.length > 0) {
          // At least one follow-up capture segment is available. Mirror backend
          // behaviour by entering the interactive 'chain_capture' phase while
          // deferring automatic post-movement consequences until the chain
          // eventually terminates.
          if (this.gameState.currentPhase !== 'chain_capture') {
            this.gameState = {
              ...this.gameState,
              currentPhase: 'chain_capture',
            };
          }
        } else {
          // No legal continuations remain; the capture chain is complete. Run
          // the same post-movement processing used for normal movement so
          // history snapshots observe the next-player interactive phase.
          await this.advanceAfterMovement();
        }

        applied = true;
        break;
      }

      case 'process_line':
      case 'choose_line_reward': {
        if (this.traceMode) {
          const boardType = this.gameState.boardType;
          const requiredLength = BOARD_CONFIGS[boardType].lineLength;
          const board = this.gameState.board;
          const players = this.gameState.players;

          let targetLine: LineInfo | undefined;

          if (move.formedLines && move.formedLines.length > 0) {
            targetLine = move.formedLines[0];
          } else {
            const allLines = this.findAllLines(board);
            const playerLines = allLines.filter((line) => line.player === move.player);
            if (playerLines.length === 0) {
              break;
            }
            targetLine = playerLines[0];
          }

          if (!targetLine) {
            break;
          }

          const lineLength = targetLine.positions.length;
          let nextBoard = board;
          let nextPlayers = players;
          let nextPendingLineReward = this._pendingLineRewardElimination;

          if (move.type === 'process_line') {
            if (lineLength === requiredLength) {
              // Exact-length line: collapse all markers and defer elimination to a
              // separate eliminate_rings_from_stack Move.
              const collapsed = collapseLineMarkersOnBoard(
                board,
                players,
                targetLine.positions,
                move.player
              );
              nextBoard = collapsed.board;
              nextPlayers = collapsed.players;
              nextPendingLineReward = true;
            } else {
              // Defensive: treat overlength process_line as minimum-collapse, no elimination.
              const markersToCollapse = targetLine.positions.slice(0, requiredLength);
              const collapsed = collapseLineMarkersOnBoard(
                board,
                players,
                markersToCollapse,
                move.player
              );
              nextBoard = collapsed.board;
              nextPlayers = collapsed.players;
            }
          } else {
            // choose_line_reward for overlength lines: apply the selected
            // reward option based on collapsedMarkers metadata.
            const isOption1 =
              !move.collapsedMarkers ||
              move.collapsedMarkers.length === targetLine.positions.length;

            if (lineLength === requiredLength) {
              // Exact-length choose_line_reward: treat as Option 1 – collapse all and
              // defer elimination to an explicit eliminate_rings_from_stack Move.
              const collapsed = collapseLineMarkersOnBoard(
                board,
                players,
                targetLine.positions,
                move.player
              );
              nextBoard = collapsed.board;
              nextPlayers = collapsed.players;
              nextPendingLineReward = true;
            } else if (isOption1) {
              // Option 1: collapse all markers; defer elimination to a separate Move.
              const collapsed = collapseLineMarkersOnBoard(
                board,
                players,
                targetLine.positions,
                move.player
              );
              nextBoard = collapsed.board;
              nextPlayers = collapsed.players;
              nextPendingLineReward = true;
            } else {
              // Option 2: collapse only the specified markers (or default to minimum).
              const markersToCollapse =
                move.collapsedMarkers ?? targetLine.positions.slice(0, requiredLength);
              const collapsed = collapseLineMarkersOnBoard(
                board,
                players,
                markersToCollapse,
                move.player
              );
              nextBoard = collapsed.board;
              nextPlayers = collapsed.players;
            }
          }

          const nextState: GameState = {
            ...this.gameState,
            board: nextBoard,
            players: nextPlayers,
          };

          if (hashGameState(nextState) !== hashGameState(this.gameState)) {
            // In trace/parity mode, remain in line_processing and record that a
            // line-reward elimination may now be required. The sandbox AI hook
            // hasPendingLineRewardElimination() will surface explicit
            // eliminate_rings_from_stack decisions on subsequent turns, mirroring
            // the backend getValidMoves behaviour.
            this.gameState = {
              ...nextState,
              currentPhase: 'line_processing',
            };
            this._pendingLineRewardElimination = nextPendingLineReward;
            applied = true;
          }
        } else {
          const nextState = applyLineDecisionMove(this.gameState, move);
          if (hashGameState(nextState) !== hashGameState(this.gameState)) {
            this.gameState = nextState;
            applied = true;

            // After applying a line decision in non-trace mode, keep the previous
            // behaviour: continue processing remaining lines, then hand off to
            // territory processing or the next player's turn.
            const remainingLines = this.findAllLines(this.gameState.board).filter(
              (line) => line.player === move.player
            );

            if (remainingLines.length > 0) {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'line_processing',
              };
            } else {
              const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
              const eligible = disconnected.filter((region) =>
                this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
              );

              if (eligible.length > 0) {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'territory_processing',
                };
              } else {
                this.checkAndApplyVictory();
                if (this.gameState.gameStatus === 'active') {
                  this._hasPlacedThisTurn = false;
                  this._mustMoveFromStackKey = undefined;
                  const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                  this.gameState = {
                    ...this.gameState,
                    currentPlayer: nextPlayer,
                  };
                  this.startTurnForCurrentPlayer();
                }
              }
            }
          }
        }
        break;
      }

      case 'process_territory_region': {
        const nextState = applyTerritoryDecisionMove(
          this.gameState,
          move,
          (regionSpaces: Position[], playerNumber: number, state: GameState) =>
            this.canProcessDisconnectedRegion(regionSpaces, playerNumber, state.board)
        );
        if (hashGameState(nextState) !== hashGameState(this.gameState)) {
          this.gameState = nextState;
          applied = true;
          this._pendingTerritorySelfElimination = true;

          if (this.traceMode) {
            // In trace/parity mode, mirror the backend GameEngine.makeMove
            // decision-flow by running victory checks immediately after the
            // geometric territory consequences are applied. This ensures
            // ring-elimination/territory victories surface at the same move
            // index in both engines while still keeping the game in the
            // territory_processing phase when the game remains active so that
            // explicit eliminate_rings_from_stack decisions can follow.
            this.checkAndApplyVictory();
            if (this.gameState.gameStatus === 'active') {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'territory_processing',
              };
            }
          } else {
            // In non-trace mode, check if more regions remain or advance
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length === 0) {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          }
        }
        break;
      }

      case 'eliminate_rings_from_stack': {
        const wasTerritorySelfElimination = this._pendingTerritorySelfElimination;
        const wasLineRewardElimination = this._pendingLineRewardElimination;

        const nextState = applyTerritoryDecisionMove(
          this.gameState,
          move,
          (regionSpaces: Position[], playerNumber: number, state: GameState) =>
            this.canProcessDisconnectedRegion(regionSpaces, playerNumber, state.board)
        );
        if (hashGameState(nextState) !== hashGameState(this.gameState)) {
          this.gameState = nextState;
          applied = true;

          // This explicit elimination satisfies any pending self-elimination
          // requirement from either territory processing or line processing.
          this._pendingTerritorySelfElimination = false;
          this._pendingLineRewardElimination = false;

          if (wasLineRewardElimination) {
            // Line-reward elimination complete; mirror backend behaviour:
            // - Run victory checks immediately after the elimination.
            // - If game remains active, stay in line_processing while further
            //   lines exist for this player, otherwise transition to
            //   territory_processing so any disconnections can be resolved.
            this.checkAndApplyVictory();

            if (this.gameState.gameStatus === 'active') {
              const remainingLines = this.findAllLines(this.gameState.board).filter(
                (line) => line.player === move.player
              );

              if (remainingLines.length > 0) {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'line_processing',
                };
              } else {
                this.gameState = {
                  ...this.gameState,
                  currentPhase: 'territory_processing',
                };
              }
            }
          } else if (wasTerritorySelfElimination) {
            // Territory-origin self-elimination: preserve the existing semantics
            // where we remain in territory_processing while any eligible
            // disconnected regions remain, otherwise advance to the next player.
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length > 0) {
              this.gameState = {
                ...this.gameState,
                currentPhase: 'territory_processing',
              };
            } else {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          } else {
            // Fallback: elimination without a recorded pending debt; treat as a
            // generic elimination at the end of a territory cycle.
            const disconnected = findDisconnectedRegionsOnBoard(this.gameState.board);
            const eligible = disconnected.filter((region) =>
              this.canProcessDisconnectedRegion(region.spaces, move.player, this.gameState.board)
            );

            if (eligible.length === 0) {
              this.checkAndApplyVictory();
              if (this.gameState.gameStatus === 'active') {
                this._hasPlacedThisTurn = false;
                this._mustMoveFromStackKey = undefined;
                const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
                this.gameState = {
                  ...this.gameState,
                  currentPlayer: nextPlayer,
                };
                this.startTurnForCurrentPlayer();
              }
            }
          }
        }
        break;
      }

      default: {
        // Unsupported move types are treated as no-ops here; callers that
        // care about strictness (e.g. applyCanonicalMove) can enforce
        // additional checks around this helper.
        break;
      }
    }

    if (!applied) {
      return false;
    }

    this.debugCheckpoint(`after-applyCanonicalMoveInternal-${move.type}`);
    const afterHash = hashGameState(this.getGameState());
    const changed = beforeHash !== afterHash;

    // Test-only board invariant enforcement: when running under Jest, assert
    // that the sandbox never commits a board state with overlapping stacks,
    // markers, or collapsed spaces. This mirrors the backend
    // BoardManager.assertBoardInvariants helper but is intentionally wired
    // only for tests so production builds avoid the extra scan cost.
    if (
      changed &&
      typeof process !== 'undefined' &&
      (process as any).env &&
      (process as any).env.NODE_ENV === 'test'
    ) {
      const selfAny = this as any;
      if (typeof selfAny.assertBoardInvariants === 'function') {
        selfAny.assertBoardInvariants(`applyCanonicalMoveInternal:${move.type}`);
      }
    }

    return changed;
  }

  /**
   * Test-only helper: apply a single process_territory_region Move using the
   * same canonical pipeline as applyCanonicalMove, returning a boolean that
   * indicates whether the move changed state. This exists so RulesMatrix
   * territory scenarios can exercise Q23 preconditions against the sandbox
   * without going through the full turn/phase machinery.
   */
  private async applyCanonicalProcessTerritoryRegion(move: Move): Promise<boolean> {
    if (move.type !== 'process_territory_region') {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalProcessTerritoryRegion: expected process_territory_region, got ${
          (move as any).type
        }`
      );
    }

    if (this.gameState.gameStatus !== 'active') {
      return false;
    }

    return this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true,
    });
  }

  /**
   * Test-only helper: apply a backend-style Move into the sandbox engine.
   *
   * This is used by parallel debug harnesses to replay the same canonical
   * move sequence into both engines. It intentionally bypasses the sandbox
   * AI heuristics and, for placement, the no-dead-placement gating so that
   * we can mirror backend GameEngine behaviour as closely as possible.
   */
  public async applyCanonicalMove(move: Move): Promise<void> {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const beforeStateForHistory = this.getGameState();

    const supportedTypes: Move['type'][] = [
      'place_ring',
      'skip_placement',
      'move_ring',
      'move_stack',
      'overtaking_capture',
      'continue_capture_segment',
      'process_line',
      'choose_line_reward',
      'process_territory_region',
      'eliminate_rings_from_stack',
    ];

    if (!supportedTypes.includes(move.type)) {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalMove: unsupported move type ${(move as any).type}`
      );
    }

    const changed = await this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true,
    });

    if (changed) {
      this.appendHistoryEntry(beforeStateForHistory, move);
    }
  }
}

// Test-only: attach a lightweight board-invariant helper to the prototype so
// invariant tests can exercise internal board sanity checks without expanding
// the public class surface for production code.
(ClientSandboxEngine.prototype as any).assertBoardInvariants = function (
  this: ClientSandboxEngine,
  context: string
): void {
  const isTestEnv =
    typeof process !== 'undefined' &&
    !!(process as any).env &&
    (process as any).env.NODE_ENV === 'test';

  const board: BoardState = (this as any).gameState.board as BoardState;
  const errors: string[] = [];

  // Invariant 1: no stacks may exist on collapsed territory.
  for (const key of board.stacks.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`stack present on collapsed space at ${key}`);
    }
  }

  // Invariant 2: a cell may not host both a stack and a marker.
  for (const key of board.stacks.keys()) {
    if (board.markers.has(key)) {
      errors.push(`stack and marker coexist at ${key}`);
    }
  }

  // Invariant 3: a cell may not host both a marker and collapsed territory.
  for (const key of board.markers.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`marker present on collapsed space at ${key}`);
    }
  }

  if (errors.length === 0) {
    return;
  }

  const message =
    `ClientSandboxEngine invariant violation (${context}):` + '\n' + errors.join('\n');

  // eslint-disable-next-line no-console
  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
};
