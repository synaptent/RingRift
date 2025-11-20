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
import { LocalAIRng } from '../../shared/engine/localAIMoveSelection';
import { findAllLinesOnBoard } from './sandboxLines';
import { processLinesForCurrentPlayer as processLinesForCurrentPlayerHelper } from './sandboxLinesEngine';
import { processDisconnectedRegionOnBoard } from './sandboxTerritory';
import {
  enumerateSimpleMovementLandings,
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers,
} from './sandboxMovement';
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

    this.gameState = {
      id: 'sandbox-local',
      boardType: config.boardType,
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
   * Ring placement is routed through tryPlaceRings so that both human and AI
   * turns share the same no-dead-placement and per-turn semantics. Movement
   * clicks delegate to handleMovementClick.
   */
  public async handleHumanCellClick(pos: Position): Promise<void> {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    if (this.gameState.currentPhase === 'ring_placement') {
      // Single-ring placement for the current player at the clicked position.
      // The sandbox UI is responsible for following up with a movement click
      // when required; after a successful placement, tryPlaceRings will
      // transition the turn into the movement phase for this player.
      const beforeState = this.getGameState();
      const playerNumber = beforeState.currentPlayer;
      const beforePlayer = beforeState.players.find((p) => p.playerNumber === playerNumber);
      const beforeRingsInHand = beforePlayer?.ringsInHand ?? 0;
      const key = positionToString(pos);
      const existingBefore = beforeState.board.stacks.get(key);
      const placedOnStack = !!existingBefore && existingBefore.rings.length > 0;

      const placed = this.tryPlaceRings(pos, 1);

      if (!placed) {
        return;
      }

      const afterState = this.getGameState();
      const afterPlayer = afterState.players.find((p) => p.playerNumber === playerNumber);
      const afterRingsInHand = afterPlayer?.ringsInHand ?? beforeRingsInHand;

      const delta = Math.max(0, beforeRingsInHand - afterRingsInHand);
      const placementCount = delta > 0 ? delta : 1;

      const moveNumber = this.gameState.history.length + 1;

      const move: Move = {
        id: '',
        type: 'place_ring',
        player: playerNumber,
        to: pos,
        placementCount,
        placedOnStack,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber,
      } as Move;

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
    const hooks: SandboxAIHooks = {
      getPlayerStacks: (playerNumber: number, board: BoardState) =>
        this.getPlayerStacks(playerNumber, board),
      hasAnyLegalMoveOrCaptureFrom: (from: Position, playerNumber: number, board: BoardState) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      enumerateLegalRingPlacements: (playerNumber: number) =>
        this.enumerateLegalRingPlacements(playerNumber),
      tryPlaceRings: (position: Position, count: number) =>
        this.tryPlaceRings(position, count),
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
    };

    await maybeRunAITurnSandbox(hooks, rng ?? Math.random);
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
    board.markers.delete(key);
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
    // Post-movement consequences for the player who just moved: lines,
    // territory disconnections, and victory checks.
    this.processLinesForCurrentPlayer();
    await this.processDisconnectedRegionsForCurrentPlayer();
    this.checkAndApplyVictory();

    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    // Clear per-turn placement state now that the movement step is complete.
    this._hasPlacedThisTurn = false;
    this._mustMoveFromStackKey = undefined;

    // Advance to the next player and start their turn (placement or movement).
    const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
    this.gameState = {
      ...this.gameState,
      currentPlayer: nextPlayer,
    };

    this.startTurnForCurrentPlayer();
  }

  /**
   * Process all disconnected regions for the current player using the
   * sandboxTerritory helpers. This mirrors the backend GameEngine
   * behaviour but runs entirely client-side, including optional
   * RegionOrderChoice handling when multiple eligible regions exist.
   */
  private async processDisconnectedRegionsForCurrentPlayer(): Promise<void> {
    const handler: TerritoryInteractionHandler | null = this.interactionHandler
      ? {
          requestChoice: (choice: any) => this.interactionHandler.requestChoice(choice),
        }
      : null;

    this.gameState = await processDisconnectedRegionsForCurrentPlayerEngine(
      this.gameState,
      handler,
      (regionSpaces: Position[], playerNumber: number, state: GameState) =>
        this.canProcessDisconnectedRegion(regionSpaces, playerNumber)
    );
  }

  /**
   * Self-elimination prerequisite: the current player must have at least
   * one stack outside the disconnected region before it can be processed.
   */
  private canProcessDisconnectedRegion(regionSpaces: Position[], playerNumber: number): boolean {
    const regionSet = new Set(regionSpaces.map(positionToString));
    const stacks = this.getPlayerStacks(playerNumber, this.gameState.board);

    for (const stack of stacks) {
      const key = positionToString(stack.position);
      if (!regionSet.has(key)) {
        return true;
      }
    }

    return false;
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
    const { state, result } = checkAndApplyVictorySandbox(this.gameState, hooks);

    this.gameState = state;

    if (!result) {
      return;
    }

    this.victoryResult = result;
  }

  /**
   * Detect and process marker lines for the current player, mirroring
   * backend behaviour when no interaction manager is wired. Delegates
   * to the pure sandboxLinesEngine helper for modularity.
   */
  private processLinesForCurrentPlayer(): void {
    this.gameState = processLinesForCurrentPlayerHelper(this.gameState);
  }

  /**
   * Create a hypothetical board with one or more rings placed at the given
   * position for the specified player. Used for no-dead-placement validation.
   *
   * The optional `count` parameter defaults to 1 and allows callers to model
   * multi-ring placements on empty cells while preserving the original
   * single-ring behaviour when omitted.
   */
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    playerNumber: number,
    count: number = 1
  ): BoardState {
    return createHypotheticalBoardWithPlacement(board, position, playerNumber, count);
  }

  /**
   * Check whether a stack at `from` would have at least one legal move or
   * capture on the provided board. Analogue of RuleEngine.hasAnyLegalMoveOrCaptureFrom.
   */
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

    return hasAnyLegalMoveOrCaptureFrom(
      this.gameState.boardType,
      board,
      from,
      playerNumber,
      view
    );
  }

  /**
   * Get all stacks controlled by the specified player on the given board.
   */
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

  /**
   * Initialize the start of the current player's turn, deciding whether they
   * begin in ring_placement or movement and applying forced elimination when
   * they are completely blocked with no rings in hand.
   */
  private startTurnForCurrentPlayer(): void {
    const hooks: SandboxTurnHooks = {
      enumerateLegalRingPlacements: (state, playerNumber) =>
        this.enumerateLegalRingPlacements(playerNumber),
      hasAnyLegalMoveOrCaptureFrom: (state, from, playerNumber, board) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      getPlayerStacks: (state, playerNumber, board) => this.getPlayerStacks(playerNumber, board),
      forceEliminateCap: (state, playerNumber) => {
        // forceEliminateCap mutates this.gameState; adapt to functional
        // style by operating on a local copy when needed.
        this.forceEliminateCap(playerNumber);
        return this.gameState;
      },
      checkAndApplyVictory: (state) => {
        this.gameState = state;
        this.checkAndApplyVictory();
        return this.gameState;
      },
    };

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

  /**
   * If the current player has stacks on the board, no rings in hand, and no
   * legal moves or captures from any of their stacks, perform a forced
   * elimination and advance to the next player.
   */
  private maybeProcessForcedEliminationForCurrentPlayer(): boolean {
    const hooks: SandboxTurnHooks = {
      enumerateLegalRingPlacements: (state, playerNumber) =>
        this.enumerateLegalRingPlacements(playerNumber),
      hasAnyLegalMoveOrCaptureFrom: (state, from, playerNumber, board) =>
        this.hasAnyLegalMoveOrCaptureFrom(from, playerNumber, board),
      getPlayerStacks: (state, playerNumber, board) => this.getPlayerStacks(playerNumber, board),
      forceEliminateCap: (state, playerNumber) => {
        this.forceEliminateCap(playerNumber);
        return this.gameState;
      },
      checkAndApplyVictory: (state) => {
        this.gameState = state;
        this.checkAndApplyVictory();
        return this.gameState;
      },
    };

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

  /**
   * Eliminate one cap from the specified player's stacks, updating board and
   * player elimination counters. Simplified analogue of backend elimination.
   */
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
   * Perform an overtaking capture chain starting from an initial segment.
   * Subsequent segments are mandatory: if further captures are available,
   * the engine will either auto-continue (single option) or request a
   * CaptureDirectionChoice via the sandbox interaction handler.
   */
  private async performCaptureChain(
    initialFrom: Position,
    initialTarget: Position,
    initialLanding: Position,
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
      getPlayerStacks: (playerNumberInner: number, board: BoardState) =>
        this.getPlayerStacks(playerNumberInner, board),
      setMarker: (pos: Position, playerNumberInner: number, board: BoardState) =>
        this.setMarker(pos, playerNumberInner, board),
      collapseMarker: (pos: Position, playerNumberInner: number, board: BoardState) =>
        this.collapseMarker(pos, playerNumberInner, board),
      flipMarker: (pos: Position, playerNumberInner: number, board: BoardState) =>
        this.flipMarker(pos, playerNumberInner, board),
      onCaptureSegmentApplied: (info) => this.handleCaptureSegmentApplied(info),
      chooseCaptureSegment: async (
        options: Array<{ from: Position; target: Position; landing: Position }>
      ) => {

        if (options.length <= 1) {
          return options[0];
        }

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
      onMovementComplete: async () => {
        await this.advanceAfterMovement();
      },
    };

    await performCaptureChainSandbox(
      hooks,
      initialFrom,
      initialTarget,
      initialLanding,
      playerNumber
    );
  }

  /**
   * Enumerate all legal overtaking capture segments for the given player from
   * the specified stack position.
   */
  private enumerateCaptureSegmentsFrom(
    from: Position,
    playerNumber: number
  ): Array<{ from: Position; target: Position; landing: Position }> {
    const board = this.gameState.board;

    const hooks: SandboxMovementEngineHooks = {
      getGameState: () => this.gameState,
      setGameState: (state: GameState) => {
        this.gameState = state;
      },
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, board?: BoardState) => this.isCollapsedSpace(pos, board),
      getMarkerOwner: (pos: Position, board?: BoardState) => this.getMarkerOwner(pos, board),
      getPlayerStacks: (playerNumberInner: number, boardInner: BoardState) =>
        this.getPlayerStacks(playerNumberInner, boardInner),
      setMarker: (pos: Position, playerNumberInner: number, boardInner: BoardState) =>
        this.setMarker(pos, playerNumberInner, boardInner),
      collapseMarker: (pos: Position, playerNumberInner: number, boardInner: BoardState) =>
        this.collapseMarker(pos, playerNumberInner, boardInner),
      flipMarker: (pos: Position, playerNumberInner: number, boardInner: BoardState) =>
        this.flipMarker(pos, playerNumberInner, boardInner),
      onMovementComplete: async () => {
        // No-op for pure enumeration.
      },
    };

    return enumerateCaptureSegmentsFromSandbox(hooks, from, playerNumber, board);
  }

  /**
   * Apply a single overtaking capture segment, including marker processing
   * and top-ring-only overtaking semantics.
   */
  private applyCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): void {
    const board = this.gameState.board;

    // Detect whether we are about to land on an existing same-color marker
    // before marker processing removes it. This mirrors the backend
    // GameEngine behaviour where landing on your own marker during an
    // overtaking capture immediately eliminates your top ring
    // (Section 8.2 / 8.3.1).
    const landingMarkerOwner = this.getMarkerOwner(landing, board);
    const landedOnOwnMarker = landingMarkerOwner === playerNumber;

    const adapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (f, t, player, options) =>
        this.applyMarkerEffectsAlongPath(f, t, player, options),
    };

    applyCaptureSegmentOnBoard(board, from, target, landing, playerNumber, adapters);

    const landingKey = positionToString(landing);
    const stacksAfterCapture: Map<string, RingStack> = new Map(board.stacks);
    let eliminatedRingsMap = this.gameState.board.eliminatedRings;
    let playersAfterCapture = this.gameState.players;
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

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: stacksAfterCapture,
        markers: new Map(board.markers),
        collapsedSpaces: new Map(board.collapsedSpaces),
        eliminatedRings: eliminatedRingsMap,
      },
      players: playersAfterCapture,
      totalRingsEliminated: this.gameState.totalRingsEliminated + totalRingsEliminatedDelta,
    };
  }

  /**
   * Canonical replay helper for a single overtaking capture segment.
   *
   * This mirrors backend GameEngine semantics for both the initial
   * 'overtaking_capture' segment and any 'continue_capture_segment'
   * follow-ups:
   *   - Apply exactly one capture segment on the board.
   *   - If additional segments are available from the landing position
   *     for the same player, enter the dedicated 'chain_capture' phase
   *     and defer post-movement consequences.
   *   - If no continuations remain, run advanceAfterMovement() once so
   *     automatic consequences and turn advancement are processed just
   *     as they were after the original chain completed.
   */
  private async applyCanonicalCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    playerNumber: number
  ): Promise<void> {
    // Apply the segment using the same helper used by sandbox-driven chains.
    this.applyCaptureSegment(from, target, landing, playerNumber);

    // After applying the segment, determine whether the chain can continue
    // from the new landing position under the current board state.
    const continuationOptions = this.enumerateCaptureSegmentsFrom(landing, playerNumber);

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
      return;
    }

    // No legal continuations remain; the capture chain is complete. Run
    // the same post-movement processing used for normal movement so
    // history snapshots observe the next-player interactive phase.
    await this.advanceAfterMovement();
  }

  /**
   * Canonical replay helper for a single line-processing decision,
   * expressed as a 'process_line' Move. This mirrors the sandbox
   * line engine semantics:
   *   - Exact-length lines: collapse all markers and eliminate a cap.
   *   - Overlength lines: collapse only the minimum required markers.
   *
   * Unlike processLinesForCurrentPlayer, this helper processes exactly
   * one line chosen by the Move (via formedLines[0]) so that backend
   * decision Moves can be replayed one-for-one into the sandbox.
   */
  private async applyCanonicalProcessLine(move: Move): Promise<boolean> {
    const playerNumber = move.player;
    const board = this.gameState.board;
    const allLines = this.findAllLines(board);
    const playerLines = allLines.filter((line) => line.player === playerNumber);

    if (playerLines.length === 0) {
      return false;
    }

    let targetLine: LineInfo | undefined;

    if (move.formedLines && move.formedLines.length > 0) {
      const target = move.formedLines[0];
      const targetKey = target.positions.map((p) => positionToString(p)).join('|');

      targetLine = playerLines.find((line) => {
        const lineKey = line.positions.map((pos) => positionToString(pos)).join('|');
        return lineKey === targetKey;
      });
    }

    if (!targetLine) {
      // Fallback: preserve historic sandbox behaviour by defaulting to
      // the first line for this player when metadata is missing or the
      // exact line cannot be found.
      targetLine = playerLines[0];
    }

    const requiredLength = BOARD_CONFIGS[this.gameState.boardType].lineLength;
    const lineLength = targetLine.positions.length;

    if (lineLength === requiredLength) {
      // Exact required length: collapse all markers and eliminate a cap,
      // mirroring sandboxLinesEngine.processLinesForCurrentPlayer.
      this.collapseLineMarkers(targetLine.positions, playerNumber);
      this.forceEliminateCap(playerNumber);
      return true;
    }

    if (lineLength > requiredLength) {
      // Overlength line: collapse only the minimum required markers; no
      // elimination. This matches the current sandbox line engine, which
      // does not yet surface line_reward_option choices.
      const markersToCollapse = targetLine.positions.slice(0, requiredLength);
      this.collapseLineMarkers(markersToCollapse, playerNumber);
      return true;
    }

    // Defensive: ignore undersized lines.
    return false;
  }

  /**
   * Canonical replay helper for a single territory-processing decision,
   * expressed as a 'process_territory_region' Move.
   *
   * This mirrors the core behaviour of processDisconnectedRegionOnBoard
   * and the sandboxTerritoryEngine, but applies exactly one region's
   * processing per Move, chosen via disconnectedRegions[0].spaces.
   */
  private async applyCanonicalProcessTerritoryRegion(move: Move): Promise<boolean> {
    const playerNumber = move.player;

    if (!move.disconnectedRegions || move.disconnectedRegions.length === 0) {
      return false;
    }

    const region = move.disconnectedRegions[0];
    const regionSpaces = region.spaces;

    if (!regionSpaces || regionSpaces.length === 0) {
      return false;
    }

    // Respect the same self-elimination prerequisite used by the normal
    // sandbox territory engine before processing a region.
    if (!this.canProcessDisconnectedRegion(regionSpaces, playerNumber)) {
      return false;
    }

    const beforeCollapsed = this.gameState.board.collapsedSpaces.size;
    const beforeTotalElim = this.gameState.totalRingsEliminated;

    const result = processDisconnectedRegionOnBoard(
      this.gameState.board,
      this.gameState.players,
      playerNumber,
      regionSpaces
    );

    this.gameState = {
      ...this.gameState,
      board: result.board,
      players: result.players,
      totalRingsEliminated:
        this.gameState.totalRingsEliminated + result.totalRingsEliminatedDelta,
    };

    const afterCollapsed = this.gameState.board.collapsedSpaces.size;
    const afterTotalElim = this.gameState.totalRingsEliminated;

    // Treat the move as applied when the S-invariant components moved
    // forward in a non-decreasing fashion, mirroring the monotonicity
    // checks used in the dedicated territory engine.
    return afterCollapsed >= beforeCollapsed && afterTotalElim >= beforeTotalElim;
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
  public tryPlaceRings(position: Position, requestedCount: number): boolean {
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

    // REMOVED: this.advanceAfterPlacement();
    // Instead, we rely on currentPhase === 'movement' and the normal
    // movement/turn-advancement flow (advanceAfterMovement + startTurn)
    // to drive subsequent actions.
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

          applied = true;
        } else {
          // Sandbox-style placement: enforce no-dead-placement via
          // tryPlaceRings so AI turns share the same gating as human
          // sandbox interaction.
          const count = Math.max(1, move.placementCount ?? 1);
          const placed = this.tryPlaceRings(move.to, count);
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
        applied = true;
        break;
      }

      case 'overtaking_capture': {
        if (!move.from || !move.to || !move.captureTarget) {
          break;
        }
        // Canonical capture replay applies exactly one segment and then
        // either enters 'chain_capture' (when continuations exist) or
        // runs full post-movement processing when the chain is complete.
        await this.applyCanonicalCaptureSegment(
          move.from,
          move.captureTarget,
          move.to,
          move.player
        );
        applied = true;
        break;
      }

      case 'continue_capture_segment': {
        if (!move.from || !move.to || !move.captureTarget) {
          break;
        }
        // Continuation segments share the same replay semantics as the
        // initial overtaking_capture segment: apply one step, then either
        // remain in 'chain_capture' or process automatic consequences
        // when the chain terminates.
        await this.applyCanonicalCaptureSegment(
          move.from,
          move.captureTarget,
          move.to,
          move.player
        );
        applied = true;
        break;
      }

      case 'process_line': {
        const didApply = await this.applyCanonicalProcessLine(move);
        applied = didApply;
        break;
      }

      case 'process_territory_region': {
        const didApply = await this.applyCanonicalProcessTerritoryRegion(move);
        applied = didApply;
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

    const afterHash = hashGameState(this.getGameState());
    return beforeHash !== afterHash;
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
      'process_territory_region',
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

  for (const key of board.stacks.keys()) {
    if (board.collapsedSpaces.has(key)) {
      errors.push(`stack present on collapsed space at ${key}`);
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
