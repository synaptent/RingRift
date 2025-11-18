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
  GameHistoryEntry
} from '../../shared/types/game';
import {
  calculateCapHeight,
  calculateDistance,
  getMovementDirectionsForBoardType,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
  MovementBoardView,
  hasAnyLegalMoveOrCaptureFromOnBoard,
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState
} from '../../shared/engine/core';
import { findAllLinesOnBoard } from './sandboxLines';
import { processLinesForCurrentPlayer as processLinesForCurrentPlayerHelper } from './sandboxLinesEngine';
import {
  enumerateSimpleMovementLandings,
  applyMarkerEffectsAlongPathOnBoard,
  MarkerPathHelpers
} from './sandboxMovement';
import {
  enumerateCaptureSegmentsFromBoard,
  applyCaptureSegmentOnBoard,
  CaptureBoardAdapters,
  CaptureApplyAdapters
} from './sandboxCaptures';
import { forceEliminateCapOnBoard } from './sandboxElimination';
import {
  findDisconnectedRegionsOnBoard,
  processDisconnectedRegionOnBoard
} from './sandboxTerritory';
import { processDisconnectedRegionsForCurrentPlayerEngine, TerritoryInteractionHandler } from './sandboxTerritoryEngine';
import { checkSandboxVictory } from './sandboxVictory';
import {
  SandboxTurnState,
  SandboxTurnHooks,
  startTurnForCurrentPlayerSandbox,
  maybeProcessForcedEliminationForCurrentPlayerSandbox
} from './sandboxTurnEngine';

/**
 * Client-local engine harness for the /sandbox route.
 *
 * Scope (current):
 * - Ring placement on non-collapsed, empty cells with no-dead-placement.
 * - Non-capturing movement with distance ≥ stack height and path/marker rules.
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
}

export class ClientSandboxEngine {
  private gameState: GameState;
  private interactionHandler: SandboxInteractionHandler;

  // When non-null, the sandbox game has ended with this result.
  private victoryResult: GameResult | null = null;

  // Internal turn-level state for sandbox per-turn flow.
  private _hasPlacedThisTurn: boolean = false;
  private _mustMoveFromStackKey: string | undefined;

  // Internal selection state for movement. This is intentionally kept off of
  // GameState to avoid diverging the shared type.
  private _selectedStackKey: string | undefined;

  // Test-only: last logical AI move chosen by maybeRunAITurn. This is used
  // by backend-vs-sandbox debug harnesses to map sandbox actions into a
  // canonical Move shape for comparison against backend getValidMoves.
  private _lastAIMove: Move | null = null;

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
      boardAfterSummary: summarizeBoard(after.board)
    };

    this.gameState = {
      ...this.gameState,
      history: [...this.gameState.history, entry]
    };
  }

  constructor(opts: ClientSandboxEngineOptions) {
    const { config, interactionHandler } = opts;
    this.interactionHandler = interactionHandler;

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
        territorySpaces: 0
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
        increment: 0
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
      territoryVictoryThreshold: Math.floor(boardConfig.totalSpaces / 2) + 1
    };
  }

  /**
   * Returns a shallow clone of the current GameState for use in React
   * components. BoardState retains its Map instances; callers should not
   * mutate them directly.
   */
  public getGameState(): GameState {
    return { ...this.gameState, history: [...this.gameState.history] };
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
   */
  public handleHumanCellClick(pos: Position): void {
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    if (this.gameState.currentPhase === 'ring_placement') {
      this.handleRingPlacementClick(pos);
    } else if (this.gameState.currentPhase === 'movement') {
      this.handleMovementClick(pos);
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
  public async maybeRunAITurn(): Promise<void> {
    // Capture a pre-turn snapshot for history/event-sourcing. We rely on
    // the shared hashGameState helper so that backend and sandbox traces
    // are directly comparable.
    const beforeStateForHistory = this.getGameState();
    const beforeHashForHistory = hashGameState(beforeStateForHistory);

    // Reset last-move tracker at the start of each AI turn.
    this._lastAIMove = null;

    try {
      const current = this.gameState.players.find(
        p => p.playerNumber === this.gameState.currentPlayer
      );
      if (!current || current.type !== 'ai' || this.gameState.gameStatus !== 'active') {
        return;
      }

      // Ring placement phase: try to place a ring if any legal placements exist.
      if (this.gameState.currentPhase === 'ring_placement') {
        if (current.ringsInHand <= 0) {
          return;
        }

        const placementCandidates = this.enumerateLegalRingPlacements(current.playerNumber);

        if (placementCandidates.length === 0) {
          // No legal placement that satisfies the sandbox no-dead-placement
          // check. Mirror backend skip_placement semantics by transitioning
          // this player directly into movement; a subsequent AI tick for this
          // player will attempt movement or forced elimination instead of
          // stalling in placement.
          this.gameState = {
            ...this.gameState,
            currentPhase: 'movement'
          };

          // Record an explicit skip_placement action so GameTrace entries
          // capture this canonical phase transition.
          this._lastAIMove = {
            id: '',
            type: 'skip_placement',
            player: current.playerNumber,
            from: undefined,
            // Backend uses a sentinel { x: 0, y: 0 } position for
            // skip_placement; the position is never inspected by
            // skip-specific logic, but keeping it stable simplifies
            // parity tooling.
            to: { x: 0, y: 0 },
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: this.gameState.history.length + 1
          } as Move;

          return;
        }

        const choice = placementCandidates[Math.floor(Math.random() * placementCandidates.length)];

        // Simple heuristic: on empty cells, occasionally attempt a
        // multi-ring placement; on existing stacks, always place one.
        const board = this.gameState.board;
        const key = positionToString(choice);
        const existing = board.stacks.get(key);
        const isOccupied = !!existing && existing.rings.length > 0;
        const maxFromHand = current.ringsInHand;
        const maxPerPlacement = isOccupied ? 1 : maxFromHand;

        let requestedCount = 1;
        if (!isOccupied && maxPerPlacement > 1) {
          // Randomly choose a count in [1, maxPerPlacement], biased
          // slightly toward smaller stacks for mobility.
          requestedCount = 1 + Math.floor(Math.random() * Math.min(3, maxPerPlacement));
        }

        const placed = this.tryPlaceRings(choice, requestedCount);

        // Record a canonical placement summary for debug harnesses.
        if (placed) {
          this._lastAIMove = {
            id: '',
            type: 'place_ring',
            player: current.playerNumber,
            from: undefined,
            to: choice,
            placementCount: requestedCount,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: this.gameState.history.length + 1
          } as Move;
        }

        return;
      }

      // Movement phase: captures first, then simple non-capturing moves.
      if (this.gameState.currentPhase !== 'movement') {
        return;
      }

      const playerNumber = current.playerNumber;

      // Prefer overtaking captures whenever any are available for this player.
      const captureSegments: Array<{
        from: Position;
        target: Position;
        landing: Position;
      }> = [];

      const stacks = this.getPlayerStacks(playerNumber, this.gameState.board);

      for (const stack of stacks) {
        const segmentsFromStack = this.enumerateCaptureSegmentsFrom(stack.position, playerNumber);
        for (const seg of segmentsFromStack) {
          captureSegments.push(seg);
        }
      }

      if (captureSegments.length > 0) {
        const seg = captureSegments[Math.floor(Math.random() * captureSegments.length)];

        await this.performCaptureChain(seg.from, seg.target, seg.landing, playerNumber);

        // Record the initial capture segment for debug harnesses.
        this._lastAIMove = {
          id: '',
          type: 'overtaking_capture',
          player: playerNumber,
          from: seg.from,
          captureTarget: seg.target,
          to: seg.landing,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: this.gameState.history.length + 1
        } as Move;

        return;
      }

      // No captures available – fall back to simple non-capturing movement.
      const landingCandidates = this.enumerateSimpleMovementLandings(playerNumber);

      if (landingCandidates.length === 0) {
        // When no moves or captures are available, attempt forced elimination
        // if the rules require it (no legal actions of any kind remain).
        this.maybeProcessForcedEliminationForCurrentPlayer();
        return;
      }

      const choice = landingCandidates[Math.floor(Math.random() * landingCandidates.length)];
      const fromPos = this.stringToPositionLocal(choice.fromKey);

      this.handleMovementClick(choice.to);

      // Record a canonical movement summary for debug harnesses.
      this._lastAIMove = {
        id: '',
        type: 'move_stack',
        player: playerNumber,
        from: fromPos,
        to: choice.to,
        timestamp: new Date(),
        thinkTime: 0,
        moveNumber: this.gameState.history.length + 1
      } as Move;
    } finally {
      const afterStateForHistory = this.getGameState();
      const afterHashForHistory = hashGameState(afterStateForHistory);

      // Only record a history entry when the AI actually produced a
      // canonical action and the sandbox state changed. This keeps
      // parity traces aligned with backend makeMove semantics and
      // avoids clutter from pure pass/diagnostic ticks.
      if (this._lastAIMove && beforeHashForHistory !== afterHashForHistory) {
        this.appendHistoryEntry(beforeStateForHistory, this._lastAIMove);
      }
    }
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
    const boardType = this.gameState.boardType;
    const config = BOARD_CONFIGS[boardType];
    const board = this.gameState.board;
    const results: Position[] = [];

    if (boardType === 'hexagonal') {
      const radius = config.size - 1;
      for (let x = -radius; x <= radius; x++) {
        for (let y = -radius; y <= radius; y++) {
          const z = -x - y;
          const pos: Position = { x, y, z };

          if (!this.isValidPosition(pos)) continue;

          const key = positionToString(pos);

          // Do not allow placement on collapsed territory, but allow
          // placement on existing stacks.
          if (board.collapsedSpaces.has(key)) continue;

          const hypothetical = this.createHypotheticalBoardWithPlacement(
            board,
            pos,
            playerNumber
          );

          if (this.hasAnyLegalMoveOrCaptureFrom(pos, playerNumber, hypothetical)) {
            results.push(pos);
          }
        }
      }
    } else {
      // square boards: 0..size-1 grid
      for (let x = 0; x < config.size; x++) {
        for (let y = 0; y < config.size; y++) {
          const pos: Position = { x, y };

          if (!this.isValidPosition(pos)) continue;

          const key = positionToString(pos);

          if (board.collapsedSpaces.has(key)) continue;

          const hypothetical = this.createHypotheticalBoardWithPlacement(
            board,
            pos,
            playerNumber
          );

          if (this.hasAnyLegalMoveOrCaptureFrom(pos, playerNumber, hypothetical)) {
            results.push(pos);
          }
        }
      }
    }

    return results;
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
      type: boardType
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

  private getMarkerOwner(position: Position, board: BoardState = this.gameState.board): number | undefined {
    const key = positionToString(position);
    const marker = board.markers.get(key);
    return marker?.player;
  }

  private setMarker(position: Position, playerNumber: number, board: BoardState = this.gameState.board): void {
    const key = positionToString(position);
    board.markers.set(key, {
      player: playerNumber,
      position,
      type: 'regular'
    });
  }

  private flipMarker(position: Position, playerNumber: number, board: BoardState = this.gameState.board): void {
    const key = positionToString(position);
    const existing = board.markers.get(key);
    if (existing && existing.player !== playerNumber) {
      board.markers.set(key, {
        player: playerNumber,
        position,
        type: 'regular'
      });
    }
  }

  private collapseMarker(position: Position, playerNumber: number, board: BoardState = this.gameState.board): void {
    const key = positionToString(position);
    board.markers.delete(key);
    board.collapsedSpaces.set(key, playerNumber);
  }

  /**
   * Find all marker lines on the board for all players. Mirrors
   * BoardManager.findAllLines; only returns lines of at least the
   * configured minimum length.
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
    const updatedPlayers = this.gameState.players.map(p =>
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
        collapsedSpaces: new Map(board.collapsedSpaces)
      },
      players: updatedPlayers
    };
  }

  /**
   * Apply marker effects for a move or capture segment from `from` to `to`:
   * - Leave a marker on the departure space.
   * - For intermediate spaces:
   *   - Opponent markers flip to the mover's color.
   *   - Own markers collapse into territory.
   * - On landing, remove same-color marker (cannot coexist with a stack).
   */
  private applyMarkerEffectsAlongPath(from: Position, to: Position, playerNumber: number): void {
    const board = this.gameState.board;

    const helpers: MarkerPathHelpers = {
      setMarker: (pos, player, b) => this.setMarker(pos, player, b),
      collapseMarker: (pos, player, b) => this.collapseMarker(pos, player, b),
      flipMarker: (pos, player, b) => this.flipMarker(pos, player, b)
    };

    applyMarkerEffectsAlongPathOnBoard(board, from, to, playerNumber, helpers);
  }

  private handleRingPlacementClick(position: Position): void {
    if (!this.isValidPosition(position)) {
      return;
    }

    const key = positionToString(position);
    const board = this.gameState.board;

    // Do not allow placement on collapsed territory.
    if (board.collapsedSpaces.has(key)) {
      return;
    }

    const existingStack = board.stacks.get(key);
    if (existingStack) {
      return;
    }

    const player = this.gameState.players.find(
      p => p.playerNumber === this.gameState.currentPlayer
    );
    if (!player || player.ringsInHand <= 0) {
      return;
    }

    // No-dead-placement constraint: placing must leave a legal move/capture.
    const hypotheticalBoard = this.createHypotheticalBoardWithPlacement(
      board,
      position,
      this.gameState.currentPlayer
    );
    if (
      !this.hasAnyLegalMoveOrCaptureFrom(
        position,
        this.gameState.currentPlayer,
        hypotheticalBoard
      )
    ) {
      return;
    }

    const rings = [this.gameState.currentPlayer];
    const newStack: RingStack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: calculateCapHeight(rings),
      controllingPlayer: this.gameState.currentPlayer
    };

    const nextStacks = new Map(board.stacks);
    nextStacks.set(key, newStack);

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: nextStacks
      },
      players: this.gameState.players.map(p =>
        p.playerNumber === this.gameState.currentPlayer
          ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - 1) }
          : p
      )
    };

    this.advanceAfterPlacement();
  }

  private handleMovementClick(position: Position): void {
    const board = this.gameState.board;
    const key = positionToString(position);
    const stackAtPos = board.stacks.get(key);

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

    const fromKey = this._selectedStackKey;
    const movingStack = board.stacks.get(fromKey);
    if (!movingStack || movingStack.controllingPlayer !== this.gameState.currentPlayer) {
      this._selectedStackKey = undefined;
      return;
    }

    const fromPos = movingStack.position;

    // Disallow landing on collapsed spaces.
    if (this.isCollapsedSpace(position, board)) {
      this._selectedStackKey = undefined;
      return;
    }

    // Determine whether this click represents a capture or a simple move.
    const fullPath = getPathPositions(fromPos, position);
    if (fullPath.length <= 1) {
      this._selectedStackKey = undefined;
      return;
    }

    const intermediate = fullPath.slice(1, -1);
    let targetPos: Position | undefined;
    for (const pos of intermediate) {
      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      if (stack) {
        targetPos = pos;
        break;
      }
    }

    // If there is an intermediate stack, attempt an overtaking capture.
    if (targetPos) {
      const view: CaptureSegmentBoardView = {
        isValidPosition: (pos: Position) => this.isValidPosition(pos),
        isCollapsedSpace: (pos: Position) => this.isCollapsedSpace(pos, board),
        getStackAt: (pos: Position) => {
          const sKey = positionToString(pos);
          const stack = board.stacks.get(sKey);
          if (!stack) return undefined;
          return {
            controllingPlayer: stack.controllingPlayer,
            capHeight: stack.capHeight,
            stackHeight: stack.stackHeight
          };
        },
        getMarkerOwner: (pos: Position) => this.getMarkerOwner(pos, board)
      };

      const isValid = validateCaptureSegmentOnBoard(
        this.gameState.boardType,
        fromPos,
        targetPos,
        position,
        movingStack.controllingPlayer,
        view
      );

      if (!isValid) {
        this._selectedStackKey = undefined;
        return;
      }

      // Start a mandatory capture chain beginning with this segment.
      void this.performCaptureChain(fromPos, targetPos, position, movingStack.controllingPlayer);
      this._selectedStackKey = undefined;
      return;
    }

    // No intermediate stack: treat as a simple non-capturing move.
    for (const pos of intermediate) {
      const pathKey = positionToString(pos);
      if (this.isCollapsedSpace(pos, board) || board.stacks.has(pathKey)) {
        this._selectedStackKey = undefined;
        return;
      }
    }

    const distance = calculateDistance(this.gameState.boardType, fromPos, position);
    if (distance < movingStack.stackHeight) {
      this._selectedStackKey = undefined;
      return;
    }

    const nextStacks = new Map(board.stacks);
    nextStacks.delete(fromKey);
    nextStacks.set(key, {
      ...movingStack,
      position
    });

    this.applyMarkerEffectsAlongPath(fromPos, position, movingStack.controllingPlayer);

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: nextStacks
      }
    };

    this._selectedStackKey = undefined;
    this.advanceAfterMovement();
  }

  /**
   * Advance to the next player or phase after a ring placement. For now this
   * mirrors the LocalSandboxState behaviour: cycle players until all rings in
   * hand are exhausted, then move to the movement phase.
   */
  private advanceAfterPlacement(): void {
    const allRingsExhausted = this.gameState.players.every(p => p.ringsInHand <= 0);
    const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);

    this.gameState = {
      ...this.gameState,
      currentPlayer: nextPlayer,
      currentPhase: allRingsExhausted ? 'movement' : 'ring_placement'
    };

    // At the start of the next player's turn, if they are completely blocked,
    // apply forced elimination.
    this.maybeProcessForcedEliminationForCurrentPlayer();
  }

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
      currentPlayer: nextPlayer
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
          requestChoice: (choice: any) => this.interactionHandler.requestChoice(choice)
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
    if (this.gameState.gameStatus !== 'active') {
      return;
    }

    const result = checkSandboxVictory(this.gameState);
    if (!result) {
      return;
    }

    this.victoryResult = result;
    this.gameState = {
      ...this.gameState,
      gameStatus: 'completed',
      winner: result.winner
    };
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
    const hypothetical: BoardState = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings }
    };

    const key = positionToString(position);

    // Placement semantics: a stack cannot coexist with a marker. When we
    // model a placement on a cell that currently has a marker, clear the
    // marker so the hypothetical board matches real placement behaviour.
    hypothetical.markers.delete(key);

    const existing = hypothetical.stacks.get(key);
    const effectiveCount = Math.max(1, count);

    if (existing && existing.rings.length > 0) {
      const addedRings = Array(effectiveCount).fill(playerNumber);
      const rings = [...addedRings, ...existing.rings];
      const newStack: RingStack = {
        ...existing,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: playerNumber
      };
      hypothetical.stacks.set(key, newStack);
    } else {
      const rings = Array(effectiveCount).fill(playerNumber);
      const newStack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: playerNumber
      };
      hypothetical.stacks.set(key, newStack);
    }

    return hypothetical;
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
    const view: MovementBoardView = {
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight
        };
      },
      getMarkerOwner: (pos: Position) => this.getMarkerOwner(pos, board)
    };

    return hasAnyLegalMoveOrCaptureFromOnBoard(
      this.gameState.boardType,
      from,
      playerNumber,
      view
    );
  }

  /**
   * Get all stacks controlled by the specified player on the given board.
   */
  private getPlayerStacks(playerNumber: number, board: BoardState = this.gameState.board): RingStack[] {
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
      getPlayerStacks: (state, playerNumber, board) =>
        this.getPlayerStacks(playerNumber, board),
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
      }
    };

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey
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
      getPlayerStacks: (state, playerNumber, board) =>
        this.getPlayerStacks(playerNumber, board),
      forceEliminateCap: (state, playerNumber) => {
        this.forceEliminateCap(playerNumber);
        return this.gameState;
      },
      checkAndApplyVictory: (state) => {
        this.gameState = state;
        this.checkAndApplyVictory();
        return this.gameState;
      }
    };

    const turnStateBefore: SandboxTurnState = {
      hasPlacedThisTurn: this._hasPlacedThisTurn,
      mustMoveFromStackKey: this._mustMoveFromStackKey
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
      totalRingsEliminated:
        this.gameState.totalRingsEliminated + result.totalRingsEliminatedDelta
    };
  }

  private getNextPlayerNumber(current: number): number {
    const players = this.gameState.players;
    const idx = players.findIndex(p => p.playerNumber === current);
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
    let currentPosition = initialLanding;
    let from = initialFrom;
    let target = initialTarget;
    let landing = initialLanding;

    this.applyCaptureSegment(from, target, landing, playerNumber);

    // eslint-disable-next-line no-constant-condition
    while (true) {
      const options = this.enumerateCaptureSegmentsFrom(currentPosition, playerNumber);
      if (options.length === 0) {
        break;
      }

      let nextSegment:
        | {
            from: Position;
            target: Position;
            landing: Position;
          }
        | undefined;

      if (options.length === 1) {
        nextSegment = options[0];
      } else {
        const choice = {
          id: `sandbox-capture-${Date.now()}-${Math.random().toString(36).slice(2)}`,
          gameId: this.gameState.id,
          playerNumber,
          type: 'capture_direction' as const,
          prompt: 'Select capture direction',
          options: options.map(opt => ({
            targetPosition: opt.target,
            landingPosition: opt.landing,
            capturedCapHeight:
              this.gameState.board.stacks.get(positionToString(opt.target))?.capHeight ?? 0
          }))
        };

        const response = await this.interactionHandler.requestChoice(choice);
        const selected = options.find(opt => {
          const o = (response as any).selectedOption;
          return (
            o &&
            positionToString(o.targetPosition) === positionToString(opt.target) &&
            positionToString(o.landingPosition) === positionToString(opt.landing)
          );
        });

        nextSegment = selected ?? options[0];
      }

      if (!nextSegment) {
        break;
      }

      from = currentPosition;
      target = nextSegment.target;
      landing = nextSegment.landing;
      this.applyCaptureSegment(from, target, landing, playerNumber);
      currentPosition = landing;
    }

    this.advanceAfterMovement();
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

    const adapters: CaptureBoardAdapters = {
      isValidPosition: (pos: Position) => this.isValidPosition(pos),
      isCollapsedSpace: (pos: Position, b: BoardState) => this.isCollapsedSpace(pos, b),
      getMarkerOwner: (pos: Position, b: BoardState) => this.getMarkerOwner(pos, b)
    };

    return enumerateCaptureSegmentsFromBoard(
      this.gameState.boardType,
      board,
      from,
      playerNumber,
      adapters
    );
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

    const adapters: CaptureApplyAdapters = {
      applyMarkerEffectsAlongPath: (f, t, player) => this.applyMarkerEffectsAlongPath(f, t, player)
    };

    applyCaptureSegmentOnBoard(board, from, target, landing, playerNumber, adapters);

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: new Map(board.stacks),
        markers: new Map(board.markers),
        collapsedSpaces: new Map(board.collapsedSpaces)
      }
    };
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
      p => p.playerNumber === this.gameState.currentPlayer
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
      !this.hasAnyLegalMoveOrCaptureFrom(
        position,
        this.gameState.currentPlayer,
        hypotheticalBoard
      )
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
        controllingPlayer: this.gameState.currentPlayer
      };
      nextStacks.set(key, newStack);
    } else {
      const rings = Array(effectiveCount).fill(this.gameState.currentPlayer);
      const newStack: RingStack = {
        position,
        rings,
        stackHeight: rings.length,
        capHeight: calculateCapHeight(rings),
        controllingPlayer: this.gameState.currentPlayer
      };
      nextStacks.set(key, newStack);
    }

    const updatedPlayers = this.gameState.players.map(p =>
      p.playerNumber === this.gameState.currentPlayer
        ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - effectiveCount) }
        : p
    );

    // Placement semantics: a stack cannot coexist with a marker. If there is
    // a marker at this key, clear it so the real board matches the invariant
    // and movement/capture semantics (landing on a marker removes it).
    board.markers.delete(key);

    // For human players in the sandbox, stay in ring_placement after a
    // successful placement so the user can either add more rings to the
    // same stack or decide to move it. For AI players, advance directly
    // to movement so their turn can complete automatically.
    const nextPhase: GamePhase = player.type === 'ai' ? 'movement' : this.gameState.currentPhase;

    this.gameState = {
      ...this.gameState,
      board: {
        ...board,
        stacks: nextStacks
      },
      players: updatedPlayers,
      currentPhase: nextPhase
    };

    // Record per-turn placement state so movement (when it happens) is
    // forced from this stack, and treat the newly placed/updated stack as
    // selected so that the next click on a valid landing square can perform
    // the move.
    this._hasPlacedThisTurn = true;
    this._mustMoveFromStackKey = key;
    this._selectedStackKey = key;

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
        currentPlayer: move.player
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
            controllingPlayer: newRings[0]
          };

          const nextStacks = new Map(board.stacks);
          nextStacks.set(key, newStack);

          const updatedPlayers = this.gameState.players.map(p =>
            p.playerNumber === move.player
              ? { ...p, ringsInHand: Math.max(0, p.ringsInHand - placementCount) }
              : p
          );

          this.gameState = {
            ...this.gameState,
            board: {
              ...board,
              stacks: nextStacks
            },
            players: updatedPlayers,
            currentPhase: 'movement'
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
          currentPhase: 'movement'
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
        // selection followed by a destination click.
        this._selectedStackKey = positionToString(move.from);
        this.handleMovementClick(move.to);
        applied = true;
        break;
      }

      case 'overtaking_capture': {
        if (!move.from || !move.to || !move.captureTarget) {
          break;
        }
        // Drive a single-segment capture chain corresponding to the
        // canonical move. Subsequent segments (if any) will be handled by
        // performCaptureChain's own continuation logic.
        await this.performCaptureChain(
          move.from,
          move.captureTarget,
          move.to,
          move.player
        );
        applied = true;
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
      'overtaking_capture'
    ];

    if (!supportedTypes.includes(move.type)) {
      throw new Error(
        `ClientSandboxEngine.applyCanonicalMove: unsupported move type ${(move as any).type}`
      );
    }

    const changed = await this.applyCanonicalMoveInternal(move, {
      bypassNoDeadPlacement: true
    });

    if (changed) {
      this.appendHistoryEntry(beforeStateForHistory, move);
    }
  }
}
