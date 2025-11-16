import {
  BoardState,
  BoardType,
  BOARD_CONFIGS,
  GamePhase,
  GameState,
  GameResult,
  Player,
  PlayerType,
  Position,
  RingStack,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
  LineInfo,
  RegionOrderChoice
} from '../../shared/types/game';
import {
  calculateCapHeight,
  calculateDistance,
  getMovementDirectionsForBoardType,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView
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

  // Internal selection state for movement. This is intentionally kept off of
  // GameState to avoid diverging the shared type.
  private _selectedStackKey: string | undefined;

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
    return { ...this.gameState };
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
   * Placeholder hook for running an AI turn in sandbox mode.
   * For now we only support simple non-capturing moves for AI.
   */
  public async maybeRunAITurn(): Promise<void> {
    const current = this.gameState.players.find(
      p => p.playerNumber === this.gameState.currentPlayer
    );
    if (!current || current.type !== 'ai' || this.gameState.gameStatus !== 'active') {
      return;
    }

    if (this.gameState.currentPhase !== 'movement') {
      return;
    }

    const landingCandidates = this.enumerateSimpleMovementLandings(current.playerNumber);
    if (landingCandidates.length === 0) {
      return;
    }

    const choice = landingCandidates[Math.floor(Math.random() * landingCandidates.length)];
    this.handleMovementClick(choice.to);
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

    // Advance to the next player and apply forced elimination if they are
    // completely blocked with no rings in hand.
    const nextPlayer = this.getNextPlayerNumber(this.gameState.currentPlayer);
    this.gameState = {
      ...this.gameState,
      currentPlayer: nextPlayer
      // Phase remains 'movement' in the sandbox harness; captures and
      // lines are treated as part of the move.
    };

    this.maybeProcessForcedEliminationForCurrentPlayer();
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
   * Create a hypothetical board with a ring placed at the given position for
   * the specified player. Used for no-dead-placement validation.
   */
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    playerNumber: number
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
    const existing = hypothetical.stacks.get(key);

    if (existing && existing.rings.length > 0) {
      const newStack: RingStack = {
        ...existing,
        rings: [playerNumber, ...existing.rings],
        stackHeight: existing.stackHeight + 1,
        capHeight: calculateCapHeight([playerNumber, ...existing.rings]),
        controllingPlayer: playerNumber
      };
      hypothetical.stacks.set(key, newStack);
    } else {
      const rings = [playerNumber];
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
    const fromKey = positionToString(from);
    const stack = board.stacks.get(fromKey);
    if (!stack || stack.controllingPlayer !== playerNumber) {
      return false;
    }

    const directions = getMovementDirectionsForBoardType(this.gameState.boardType);

    // Non-capture movement
    for (const dir of directions) {
      for (let distance = stack.stackHeight; distance <= stack.stackHeight + 5; distance++) {
        const target: Position = {
          x: from.x + dir.x * distance,
          y: from.y + dir.y * distance,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * distance })
        };

        if (!this.isValidPosition(target)) {
          break;
        }

        const targetKey = positionToString(target);
        if (board.collapsedSpaces.has(targetKey)) {
          break;
        }

        const path = getPathPositions(from, target).slice(1, -1);
        let blocked = false;
        for (const pos of path) {
          const pk = positionToString(pos);
          if (board.collapsedSpaces.has(pk) || board.stacks.has(pk)) {
            blocked = true;
            break;
          }
        }
        if (blocked) {
          break;
        }

        const landingStack = board.stacks.get(targetKey);
        const markerOwner = this.getMarkerOwner(target, board);

        if (!landingStack || landingStack.rings.length === 0) {
          if (markerOwner === undefined || markerOwner === playerNumber) {
            return true;
          }
        } else {
          return true;
        }
      }
    }

    // Capture
    for (const dir of directions) {
      let step = 1;
      let targetPos: Position | undefined;

      while (true) {
        const pos: Position = {
          x: from.x + dir.x * step,
          y: from.y + dir.y * step,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step })
        };

        if (!this.isValidPosition(pos)) {
          break;
        }

        const posKey = positionToString(pos);
        if (board.collapsedSpaces.has(posKey)) {
          break;
        }

        const stackAtPos = board.stacks.get(posKey);
        if (stackAtPos && stackAtPos.rings.length > 0) {
          if (stack.capHeight >= stackAtPos.capHeight) {
            targetPos = pos;
          }
          break;
        }

        step++;
      }

      if (!targetPos) continue;

      let landingStep = 1;
      while (landingStep <= stack.stackHeight + 5) {
        const landing: Position = {
          x: targetPos.x + dir.x * landingStep,
          y: targetPos.y + dir.y * landingStep,
          ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
        };

        if (!this.isValidPosition(landing)) {
          break;
        }

        const landingKey = positionToString(landing);
        if (board.collapsedSpaces.has(landingKey)) {
          break;
        }

        const landingStack = board.stacks.get(landingKey);
        if (landingStack && landingStack.rings.length > 0) {
          break;
        }

        const view: CaptureSegmentBoardView = {
          isValidPosition: (pos: Position) => this.isValidPosition(pos),
          isCollapsedSpace: (pos: Position) => {
            const k = positionToString(pos);
            return board.collapsedSpaces.has(k);
          },
          getStackAt: (pos: Position) => {
            const k = positionToString(pos);
            const s = board.stacks.get(k);
            if (!s) return undefined;
            return {
              controllingPlayer: s.controllingPlayer,
              capHeight: s.capHeight,
              stackHeight: s.stackHeight
            };
          },
          getMarkerOwner: (pos: Position) => this.getMarkerOwner(pos, board)
        };

        if (
          validateCaptureSegmentOnBoard(
            this.gameState.boardType,
            from,
            targetPos,
            landing,
            playerNumber,
            view
          )
        ) {
          return true;
        }

        landingStep++;
      }
    }

    return false;
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
   * If the current player has stacks on the board, no rings in hand, and no
   * legal moves or captures from any of their stacks, perform a forced
   * elimination and advance to the next player.
   */
  private maybeProcessForcedEliminationForCurrentPlayer(): void {
    const current = this.gameState.currentPlayer;
    const player = this.gameState.players.find(p => p.playerNumber === current);
    if (!player) {
      return;
    }

    const board = this.gameState.board;
    const stacks = this.getPlayerStacks(current, board);
    if (stacks.length === 0) {
      return;
    }

    if (player.ringsInHand > 0) {
      return;
    }

    const hasAnyAction = stacks.some(stack =>
      this.hasAnyLegalMoveOrCaptureFrom(stack.position, current, board)
    );
    if (hasAnyAction) {
      return;
    }

    this.forceEliminateCap(current);

    const nextPlayer = this.getNextPlayerNumber(current);
    this.gameState = {
      ...this.gameState,
      currentPlayer: nextPlayer
    };
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
}
