import {
  GameState,
  Move,
  Position,
  BoardState,
  RingStack,
  positionToString,
  positionsEqual,
  BOARD_CONFIGS,
  Territory,
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import {
  calculateCapHeight,
  calculateDistance,
  getMovementDirectionsForBoardType,
  getPathPositions,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView,
  MovementBoardView,
  hasAnyLegalMoveOrCaptureFromOnBoard,
} from '../../shared/engine/core';
import { createHypotheticalBoardWithPlacement as createHypotheticalBoardWithPlacementHelper } from './rules/placementHelpers';

export class RuleEngine {
  private boardManager: BoardManager;
  private boardConfig: (typeof BOARD_CONFIGS)[keyof typeof BOARD_CONFIGS];
  private boardType: keyof typeof BOARD_CONFIGS;

  constructor(boardManager: BoardManager, boardType: keyof typeof BOARD_CONFIGS) {
    this.boardManager = boardManager;
    this.boardConfig = BOARD_CONFIGS[boardType];
    this.boardType = boardType;

    // Keep selected internal helpers referenced so ts-node/TypeScript with
    // noUnusedLocals enabled does not treat them as dead code. This has
    // no runtime effect; it only preserves helpers for diagnostics and
    // future rule-engine extensions.
    this._debugUseInternalHelpers();
  }

  /**
   * Validates a move according to RingRift rules
   */
  validateMove(move: Move, gameState: GameState): boolean {
    // Basic validation
    if (!this.isValidPlayer(move.player, gameState)) {
      return false;
    }

    if (!this.isPlayerTurn(move.player, gameState)) {
      return false;
    }

    // Validate based on move type and game phase
    switch (move.type) {
      case 'place_ring':
        return this.validateRingPlacement(move, gameState);
      case 'move_ring': // legacy alias for simple stack movement
      case 'move_stack':
        return this.validateStackMovement(move, gameState);
      case 'overtaking_capture':
        return this.validateCapture(move, gameState);
      case 'continue_capture_segment':
        return this.validateChainCaptureContinuation(move, gameState);
      case 'process_line':
      case 'choose_line_reward':
        return this.validateLineProcessingMove(move, gameState);
      case 'process_territory_region':
        return this.validateTerritoryProcessingMove(move, gameState);
      case 'skip_placement':
        // Trivial validation: skipping placement is only allowed during the
        // ring_placement phase, and only when placement is optional under
        // the rules (i.e. the player has rings in hand *and* at least one
        // controlled stack with a legal move or capture available).
        return this.validateSkipPlacement(move, gameState);
      default:
        return false;
    }
  }

  /**
   * Validates a skip_placement move. This is a no-op move that is only
   * legal during the ring_placement phase when placement is *optional*:
   * the player has rings in hand, controls at least one stack on the
   * board, and has at least one legal move or capture from some
   * controlled stack. It is *not* allowed to skip when placement is
   * mandatory (no stacks on board, or stacks with no legal moves).
   *
   * Rule Reference: Section 4.1 / 2.1 – optional placement when
   * movement/capture is available.
   */
  private validateSkipPlacement(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    const playerState = gameState.players.find((p) => p.playerNumber === move.player);
    if (!playerState || playerState.ringsInHand <= 0) {
      // No rings to optionally place – skipping is meaningless here.
      return false;
    }

    const board = gameState.board;
    const playerStacks = this.getPlayerStacks(move.player, board);

    // If the player has no stacks at all, placement is mandatory.
    if (playerStacks.length === 0) {
      return false;
    }

    // Check if at least one controlled stack has a legal move or capture
    // in the *current* board state. If none do, placement is mandatory
    // (the player is effectively blocked without placing).
    const hasAnyAction = playerStacks.some((pos) =>
      this.hasAnyLegalMoveOrCaptureFrom(pos, move.player, board)
    );

    return hasAnyAction;
  }

  /**
   * Validates ring placement according to RingRift rules
   * Rule Reference: Section 7.1 - Placement must leave at least one legal move or capture
   */
  private validateRingPlacement(move: Move, gameState: GameState): boolean {
    // Ring placement is only allowed during ring placement phase
    if (gameState.currentPhase !== 'ring_placement') {
      return false;
    }

    // Basic position validity
    if (!this.boardManager.isValidPosition(move.to)) {
      return false;
    }

    // Cannot place on collapsed spaces
    if (this.boardManager.isCollapsedSpace(move.to, gameState.board)) {
      return false;
    }

    // Cannot place on a marker. Markers represent movement history and
    // must not coexist with stacks; allowing placement onto a marker
    // would create stack+marker coexistence and break the S-invariant.
    {
      const posKey = positionToString(move.to);
      if (gameState.board.markers.has(posKey)) {
        return false;
      }
    }

    const playerState = gameState.players.find((p) => p.playerNumber === move.player);
    if (!playerState) {
      return false;
    }

    const ringsInHand = playerState.ringsInHand;
    if (ringsInHand <= 0) {
      return false;
    }

    // Compute how many rings this player already has on the board
    const playerStacks = this.getPlayerStacks(move.player, gameState.board);
    const ringsOnBoard = playerStacks.reduce((sum, pos) => {
      const stackKey = positionToString(pos);
      const stackAtPos = gameState.board.stacks.get(stackKey);
      return sum + (stackAtPos ? stackAtPos.rings.length : 0);
    }, 0);

    const perPlayerCap = this.boardConfig.ringsPerPlayer;
    const remainingByCap = perPlayerCap - ringsOnBoard;
    const remainingBySupply = ringsInHand;
    const maxAvailable = Math.min(remainingByCap, remainingBySupply);

    if (maxAvailable <= 0) {
      return false;
    }

    const posKey = positionToString(move.to);
    const existingStack = gameState.board.stacks.get(posKey);
    const isOccupied = !!(existingStack && existingStack.rings.length > 0);

    let maxPerPlacement: number;
    if (isOccupied) {
      // On existing stacks we only ever place a single ring
      if (maxAvailable < 1) {
        return false;
      }
      maxPerPlacement = 1;
    } else {
      // On empty spaces we allow multi-ring placements (typically up to 3)
      maxPerPlacement = Math.min(3, maxAvailable);
    }

    const requestedCount = move.placementCount ?? 1;

    if (requestedCount < 1 || requestedCount > maxPerPlacement) {
      return false;
    }

    // No-dead-placement: after placing, the resulting stack must have at least
    // one legal move or capture.
    const hypotheticalBoard = this.createHypotheticalBoardWithPlacement(
      gameState.board,
      move.to,
      move.player,
      requestedCount
    );

    if (!this.hasAnyLegalMoveOrCaptureFrom(move.to, move.player, hypotheticalBoard)) {
      return false;
    }

    return true;
  }

  /**
   * Validates stack movement according to RingRift rules
   * Rule Reference: Section 8.2, FAQ Q2
   */
  private validateStackMovement(move: Move, gameState: GameState): boolean {
    // Stack movement is allowed during movement or capture phases
    if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
      return false;
    }

    if (!move.from) {
      return false;
    }

    // Check if source position has player's stack
    const fromKey = positionToString(move.from);
    const sourceStack = gameState.board.stacks.get(fromKey);
    if (!sourceStack || sourceStack.controllingPlayer !== move.player) {
      return false;
    }

    // Movement must follow a straight ray consistent with the board's
    // movement directions (Moore for square, cube axes for hex), matching
    // sandboxMovement and hasAnyLegalMoveOrCaptureFromOnBoard.
    if (!this.isStraightLineMovement(move.from, move.to)) {
      return false;
    }

    // Check if destination is valid
    if (!this.boardManager.isValidPosition(move.to)) {
      return false;
    }

    // Rule Reference: Section 8.2 - Cannot land on collapsed space
    if (this.boardManager.isCollapsedSpace(move.to, gameState.board)) {
      return false;
    }

    // Check movement distance based on stack height
    // Rule Reference: Section 8.2 - Must move at least stack height
    const distance = this.calculateDistance(move.from, move.to);
    const minDistance = sourceStack.stackHeight;

    if (distance < minDistance) {
      return false;
    }

    // Validate landing position
    // Rule Reference: Section 8.2, FAQ Q2 - Can land on empty or same-color marker
    const toKey = positionToString(move.to);
    const destinationStack = gameState.board.stacks.get(toKey);
    const destinationMarker = this.boardManager.getMarker(move.to, gameState.board);

    // Landing space must be either empty or have same-color marker
    if (destinationStack && destinationStack.rings.length > 0) {
      // Can land on stacks for merging
      // This is valid in normal movement
    } else if (destinationMarker !== undefined && destinationMarker !== move.player) {
      // Cannot land on opponent's marker
      return false;
    }

    // Check if path is clear
    // Rule Reference: Section 8.1 - Cannot pass through collapsed spaces or other rings
    if (!this.isPathClear(move.from, move.to, gameState.board)) {
      return false;
    }

    return true;
  }

  /**
   * Checks if the path between two positions is clear
   * Rule Reference: Section 8.1, Section 8.2
   */
  private isPathClear(from: Position, to: Position, board: BoardState): boolean {
    // Get path positions (excluding start and end)
    const pathPositions = getPathPositions(from, to).slice(1, -1);

    // Check each position along the path
    for (const pos of pathPositions) {
      const posKey = positionToString(pos);

      // Cannot pass through collapsed spaces
      if (this.boardManager.isCollapsedSpace(pos, board)) {
        return false;
      }

      // Cannot pass through other rings/stacks
      const stack = board.stacks.get(posKey);
      if (stack && stack.rings.length > 0) {
        return false;
      }

      // Markers are OK to pass through - they get flipped/collapsed
    }

    return true;
  }

  /**
   * Validates overtaking capture move according to RingRift rules
   * Rule Reference: Section 10.1, Section 10.2
   */

  /**
   * Validates overtaking capture move according to RingRift rules
   * Rule Reference: Section 10.1, Section 10.2
   */
  private validateCapture(move: Move, gameState: GameState): boolean {
    // Captures are only allowed during interactive phases (movement/capture)
    // and during the dedicated chain_capture phase used for explicit capture
    // continuation segments. This allows:
    //   - an initial overtaking capture to be chosen as the first action
    //     of the turn from either movement or capture phase, and
    //   - internal enumeration of follow-up segments during chain_capture
    //     while still disallowing captures during placement and
    //     post-processing phases.
    if (
      gameState.currentPhase !== 'capture' &&
      gameState.currentPhase !== 'movement' &&
      gameState.currentPhase !== 'chain_capture'
    ) {
      return false;
    }

    if (!move.from || !move.captureTarget) {
      return false;
    }

    return this.validateCaptureSegment(
      move.from,
      move.captureTarget,
      move.to,
      move.player,
      gameState.board
    );
  }

  /**
   * Validates a follow-up capture segment during the dedicated chain_capture
   * phase. The geometric/path semantics are identical to an overtaking_capture
   * segment; the only difference is that this move is only legal while an
   * existing chain is in progress.
   *
   * The GameEngine is responsible for enforcing that the move's `from`
   * position matches the current chain origin and that the player matches
   * the chain owner; here we only enforce phase and segment-level legality.
   */
  private validateChainCaptureContinuation(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'chain_capture') {
      return false;
    }

    if (!move.from || !move.captureTarget) {
      return false;
    }

    return this.validateCaptureSegment(
      move.from,
      move.captureTarget,
      move.to,
      move.player,
      gameState.board
    );
  }

  /**
   * Core validation for a single overtaking capture segment from `from`
   * over `target` to `landing`. This mirrors the Rust engine's
   * `validate_capture_segment` at a high level and is used both by
   * `validateCapture` and by capture move generation.
   */
  private validateCaptureSegment(
    from: Position,
    target: Position,
    landing: Position,
    player: number,
    board: BoardState
  ): boolean {
    const view: CaptureSegmentBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    return validateCaptureSegmentOnBoard(
      this.boardType as any,
      from,
      target,
      landing,
      player,
      view
    );
  }

  /**
   * Processes a move and returns the new game state
   */
  processMove(move: Move, gameState: GameState): GameState {
    const newState = this.cloneGameState(gameState);

    switch (move.type) {
      case 'place_ring':
        this.processRingPlacement(move, newState);
        break;
      case 'move_stack':
        this.processStackMovement(move, newState);
        break;
      case 'overtaking_capture':
        this.processCapture(move, newState);
        break;
    }

    // Process automatic consequences
    this.processLineFormation(newState);
    this.processTerritoryDisconnection(newState);

    return newState;
  }

  /**
   * Processes ring placement
   */
  private processRingPlacement(move: Move, gameState: GameState): void {
    const newStack: RingStack = {
      position: move.to,
      rings: [move.player],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: move.player,
    };

    const posKey = positionToString(move.to);
    gameState.board.stacks.set(posKey, newStack);
  }

  /**
   * Processes stack movement
   */
  private processStackMovement(move: Move, gameState: GameState): void {
    if (!move.from) return;

    const fromKey = positionToString(move.from);
    const toKey = positionToString(move.to);

    const sourceStack = gameState.board.stacks.get(fromKey);
    if (!sourceStack) return;

    const destinationStack = gameState.board.stacks.get(toKey);

    if (destinationStack && destinationStack.rings.length > 0) {
      // Merge stacks
      const mergedStack: RingStack = {
        position: move.to,
        rings: [...destinationStack.rings, ...sourceStack.rings],
        stackHeight: destinationStack.stackHeight + sourceStack.stackHeight,
        capHeight: sourceStack.capHeight, // Moving stack's cap becomes new cap
        controllingPlayer: sourceStack.controllingPlayer,
      };
      gameState.board.stacks.set(toKey, mergedStack);
    } else {
      // Move to empty position
      const movedStack: RingStack = {
        ...sourceStack,
        position: move.to,
      };
      gameState.board.stacks.set(toKey, movedStack);
    }

    // Remove from source
    gameState.board.stacks.delete(fromKey);
  }

  /**
   * Processes capture with chain reactions
   */
  private processCapture(move: Move, gameState: GameState): void {
    if (!move.from || !move.capturedStacks) return;

    const fromKey = positionToString(move.from);
    const attackerStack = gameState.board.stacks.get(fromKey);
    if (!attackerStack) return;

    const capturedStacks: RingStack[] = move.capturedStacks;

    // Apply overtaking one ring at a time, mirroring the GameEngine and
    // Rust behaviour: each capture segment takes only the top ring of the
    // target stack and adds it to the bottom of the attacker. Target
    // stacks that become empty are removed.
    let updatedAttacker = attackerStack;

    for (const capturedStack of capturedStacks) {
      const capturedKey = positionToString(capturedStack.position);
      const currentTarget = gameState.board.stacks.get(capturedKey);
      if (!currentTarget || currentTarget.rings.length === 0) {
        continue;
      }

      const [capturedRing, ...remaining] = currentTarget.rings;

      if (remaining.length > 0) {
        const newTarget: RingStack = {
          ...currentTarget,
          rings: remaining,
          stackHeight: remaining.length,
          capHeight: calculateCapHeight(remaining),
          controllingPlayer: remaining[0],
        };
        gameState.board.stacks.set(capturedKey, newTarget);
      } else {
        gameState.board.stacks.delete(capturedKey);
      }

      const newRings = [...updatedAttacker.rings, capturedRing];
      updatedAttacker = {
        ...updatedAttacker,
        rings: newRings,
        stackHeight: newRings.length,
        capHeight: calculateCapHeight(newRings),
        controllingPlayer: newRings[0],
      };
    }

    gameState.board.stacks.set(fromKey, updatedAttacker);

    // Check for chain reactions (legacy behaviour; GameEngine now drives
    // chain captures via chainCaptureState and CaptureDirectionChoice).
    this.processChainReactions(move.from, gameState);
  }

  /**
   * Processes chain reactions from captures
   */
  private processChainReactions(triggerPos: Position, gameState: GameState): void {
    const triggerKey = positionToString(triggerPos);
    const triggerStack = gameState.board.stacks.get(triggerKey);
    if (!triggerStack) return;

    const adjacentPositions = this.getAdjacentPositions(triggerPos);

    for (const adjPos of adjacentPositions) {
      const adjKey = positionToString(adjPos);
      const adjStack = gameState.board.stacks.get(adjKey);
      if (
        adjStack &&
        adjStack.controllingPlayer !== triggerStack.controllingPlayer &&
        triggerStack.capHeight >= adjStack.capHeight
      ) {
        // Trigger another capture
        const captureMove: Move = {
          id: `chain-${Date.now()}`,
          type: 'overtaking_capture',
          player: triggerStack.controllingPlayer,
          from: triggerPos,
          to: adjPos,
          capturedStacks: [adjStack],
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1,
        };

        this.processCapture(captureMove, gameState);
      }
    }
  }

  /**
   * Processes line formation and marker collapse
   */
  private processLineFormation(gameState: GameState): void {
    const lines = this.boardManager.findAllLines(gameState.board);

    for (const line of lines) {
      if (line.positions.length >= this.boardConfig.lineLength) {
        // Collapse line - remove all stacks in the line
        for (const pos of line.positions) {
          const posKey = positionToString(pos);
          gameState.board.stacks.delete(posKey);
        }
      }
    }
  }

  /**
   * Processes territory disconnection
   */
  private processTerritoryDisconnection(gameState: GameState): void {
    // Check territories for each player
    for (const player of gameState.players) {
      const territories = this.boardManager.findAllTerritories(
        player.playerNumber,
        gameState.board
      );

      for (const territory of territories) {
        if (territory.isDisconnected) {
          // Remove all stacks in disconnected territory
          for (const pos of territory.spaces) {
            const posKey = positionToString(pos);
            gameState.board.stacks.delete(posKey);
          }
        }
      }
    }
  }

  /**
   * Checks for game end conditions.
   *
   * Backend victory semantics are aligned with the sandbox victory helper
   * (src/client/sandbox/sandboxVictory.ts) and the compact rules:
   *
   * - Ring-elimination victory: a player wins when their eliminatedRings
   *   count reaches or exceeds GameState.victoryThreshold (>50% of total
   *   rings in play).
   * - Territory-control victory: a player wins when their territorySpaces
   *   count reaches or exceeds GameState.territoryVictoryThreshold (>50%
   *   of board spaces).
   * - Fallback structural terminality: when there are no stacks on the
   *   board and no player has ringsInHand, the game cannot progress.
   *   In that case we apply territory, then eliminated-rings tie-breakers;
   *   if still tied, we mark the game as completed with no winner.
   */
  checkGameEnd(gameState: GameState): { isGameOver: boolean; winner?: number; reason?: string } {
    const players = gameState.players;

    // 1) Ring-elimination victory: strictly more than 50% of total rings
    // in play have been eliminated for a single player.
    const ringWinner = players.find((p) => p.eliminatedRings >= gameState.victoryThreshold);
    if (ringWinner) {
      return {
        isGameOver: true,
        winner: ringWinner.playerNumber,
        reason: 'ring_elimination',
      };
    }

    // 2) Territory-control victory: strictly more than 50% of the board's
    // spaces are controlled as territory by a single player.
    const territoryWinner = players.find(
      (p) => p.territorySpaces >= gameState.territoryVictoryThreshold
    );
    if (territoryWinner) {
      return {
        isGameOver: true,
        winner: territoryWinner.playerNumber,
        reason: 'territory_control',
      };
    }

    // 3) Fallback structural terminality: no stacks on the board and no
    // rings in hand for any player. In this situation the game cannot
    // progress further, even if nobody has reached the strict thresholds.
    const noStacksLeft = gameState.board.stacks.size === 0;
    const anyRingsInHand = players.some((p) => p.ringsInHand > 0);

    // Only trigger fallback termination if NO stacks are left AND NO rings are in hand.
    // The previous logic was correct, but we want to be absolutely sure we aren't
    // triggering this prematurely.
    if (noStacksLeft && !anyRingsInHand) {
      // First tie-breaker: territory spaces.
      const maxTerritory = Math.max(...players.map((p) => p.territorySpaces));
      const territoryLeaders = players.filter((p) => p.territorySpaces === maxTerritory);

      if (territoryLeaders.length === 1 && maxTerritory > 0) {
        return {
          isGameOver: true,
          winner: territoryLeaders[0].playerNumber,
          reason: 'territory_control',
        };
      }

      // Second tie-breaker: eliminated rings.
      const maxEliminated = Math.max(...players.map((p) => p.eliminatedRings));
      const eliminationLeaders = players.filter((p) => p.eliminatedRings === maxEliminated);

      if (eliminationLeaders.length === 1 && maxEliminated > 0) {
        return {
          isGameOver: true,
          winner: eliminationLeaders[0].playerNumber,
          reason: 'ring_elimination',
        };
      }

      // Third tie-breaker: remaining markers on the board. This mirrors
      // the complete rules' S-invariant ladder (markers, collapsed,
      // eliminated) and ensures structural terminality still yields a
      // definitive winner when possible.
      const markerCountsByPlayer: { [player: number]: number } = {};
      for (const p of players) {
        markerCountsByPlayer[p.playerNumber] = 0;
      }
      for (const marker of gameState.board.markers.values()) {
        const owner = marker.player;
        if (markerCountsByPlayer[owner] !== undefined) {
          markerCountsByPlayer[owner] += 1;
        }
      }

      const markerCounts = players.map((p) => markerCountsByPlayer[p.playerNumber] ?? 0);
      const maxMarkers = Math.max(...markerCounts);
      const markerLeaders = players.filter(
        (p) => (markerCountsByPlayer[p.playerNumber] ?? 0) === maxMarkers
      );

      if (markerLeaders.length === 1 && maxMarkers > 0) {
        return {
          isGameOver: true,
          winner: markerLeaders[0].playerNumber,
          reason: 'last_player_standing',
        };
      }

      // Final tie-breaker: last player to complete a valid turn action.
      const lastActor = this.getLastActor(gameState);
      if (lastActor !== undefined) {
        return {
          isGameOver: true,
          winner: lastActor,
          reason: 'last_player_standing',
        };
      }

      // Safety fallback: in degenerate cases where no last actor can be
      // determined (e.g. malformed game state), mark the game as
      // completed without a specific winner.
      return {
        isGameOver: true,
        reason: 'game_completed',
      };
    }

    return { isGameOver: false };
  }

  /**
   * Gets valid moves for the current game state
   */
  getValidMoves(gameState: GameState): Move[] {
    const moves: Move[] = [];
    const currentPlayer = gameState.currentPlayer;

    switch (gameState.currentPhase) {
      case 'ring_placement': {
        // Generate all legal ring placements for the active player.
        moves.push(...this.getValidRingPlacements(currentPlayer, gameState));

        // Also expose the skip_placement no-op when (and only when) placement
        // is optional under the rules. This mirrors TurnEngine.hasValidPlacements
        // and prevents "active game with no legal moves" states in
        // ring_placement when the player has rings in hand *and* at least one
        // legal move/capture from a controlled stack.
        const skipMoveCandidate: Move = {
          id: 'skip_placement',
          type: 'skip_placement',
          player: currentPlayer,
          // Skip placement is a pure phase transition with no board
          // coordinates. However, the shared Move type requires a `to`
          // position, so we supply a harmless sentinel value that is
          // never inspected by skip_placement-specific logic.
          to: { x: 0, y: 0 },
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1,
        };

        if (this.validateSkipPlacement(skipMoveCandidate, gameState)) {
          moves.push(skipMoveCandidate);
        }
        break;
      }
      case 'movement':
        // During movement phase, expose both simple movements and
        // overtaking capture options so that an initial capture can be
        // chosen directly when legal.
        moves.push(...this.getValidStackMovements(currentPlayer, gameState));
        moves.push(...this.getValidCaptures(currentPlayer, gameState));
        break;
      case 'capture':
        moves.push(...this.getValidCaptures(currentPlayer, gameState));
        break;
    }

    return moves;
  }

  /**
   * Gets valid ring placement moves
   */
  private getValidRingPlacements(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const playerState = gameState.players.find((p) => p.playerNumber === player);
    if (!playerState || playerState.ringsInHand <= 0) {
      return moves;
    }

    const board = gameState.board;
    const allPositions = this.boardManager.getAllPositions();

    // Compute rings on board for capacity checks
    const playerStacks = this.getPlayerStacks(player, board);
    const ringsOnBoard = playerStacks.reduce((sum, pos) => {
      const stackKey = positionToString(pos);
      const stackAtPos = board.stacks.get(stackKey);
      return sum + (stackAtPos ? stackAtPos.rings.length : 0);
    }, 0);

    const perPlayerCap = this.boardConfig.ringsPerPlayer;
    const remainingByCap = perPlayerCap - ringsOnBoard;
    const remainingBySupply = playerState.ringsInHand;
    const maxAvailableGlobal = Math.min(remainingByCap, remainingBySupply);

    if (maxAvailableGlobal <= 0) {
      return moves;
    }

    for (const pos of allPositions) {
      if (!this.boardManager.isValidPosition(pos)) {
        continue;
      }

      if (this.boardManager.isCollapsedSpace(pos, board)) {
        continue;
      }

      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      const isOccupied = !!(stack && stack.rings.length > 0);

      if (isOccupied) {
        if (maxAvailableGlobal < 1) {
          continue;
        }

        const candidate: Move = {
          id: `place-${positionToString(pos)}-stack`,
          type: 'place_ring',
          player,
          to: pos,
          placedOnStack: true,
          placementCount: 1,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1,
        };

        if (this.validateRingPlacement(candidate, gameState)) {
          moves.push(candidate);
        }
      } else {
        const maxPerPlacement = Math.min(3, maxAvailableGlobal);
        for (let count = 1; count <= maxPerPlacement; count++) {
          const candidate: Move = {
            id: `place-${positionToString(pos)}-x${count}`,
            type: 'place_ring',
            player,
            to: pos,
            placedOnStack: false,
            placementCount: count,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: gameState.moveHistory.length + 1,
          };

          if (this.validateRingPlacement(candidate, gameState)) {
            moves.push(candidate);
          }
        }
      }
    }

    return moves;
  }

  /**
   * Gets valid stack movement moves
   */
  private getValidStackMovements(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const playerStacks = this.getPlayerStacks(player, gameState.board);

    for (const stackPos of playerStacks) {
      const allPositions = this.boardManager.getAllPositions();

      for (const targetPos of allPositions) {
        if (positionsEqual(stackPos, targetPos)) continue;

        const testMove: Move = {
          id: '',
          type: 'move_stack',
          player,
          from: stackPos,
          to: targetPos,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 0,
        };

        if (this.validateStackMovement(testMove, gameState)) {
          moves.push({
            ...testMove,
            id: `move-${positionToString(stackPos)}-${positionToString(targetPos)}`,
            moveNumber: gameState.moveHistory.length + 1,
          });
        }
      }
    }

    return moves;
  }

  /**
   * Gets valid capture moves using a directional enumeration similar to the
   * Rust CaptureProcessor: from each stack, walk along rays in all movement
   * directions, find the first capturable target on each ray, then enumerate
   * valid landing positions beyond that target.
   *
   * Rule Reference: Section 10.1, 10.2 - Overtaking capture requirements
   */
  private getValidCaptures(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const board = gameState.board;
    const playerStacks = this.getPlayerStacks(player, board);

    const directions = this.getCaptureDirections();

    for (const stackPos of playerStacks) {
      const fromKey = positionToString(stackPos);
      const attackerStack = board.stacks.get(fromKey);
      if (!attackerStack) continue;

      for (const dir of directions) {
        // Step outward from the attacker to find the first potential target
        let step = 1;
        let targetPos: Position | undefined;

        while (true) {
          const pos: Position = {
            x: stackPos.x + dir.x * step,
            y: stackPos.y + dir.y * step,
            ...(dir.z !== undefined && { z: (stackPos.z || 0) + dir.z * step }),
          };

          if (!this.boardManager.isValidPosition(pos)) {
            break; // Off-board
          }

          // Collapsed spaces block both target search and landing beyond
          if (this.boardManager.isCollapsedSpace(pos, board)) {
            break;
          }

          const posKey = positionToString(pos);
          const stackAtPos = board.stacks.get(posKey);

          if (stackAtPos && stackAtPos.rings.length > 0) {
            // First stack encountered along this ray is the only possible
            // capture target in this direction.
            // Rule fix: Can overtake own stacks (no same-player restriction)
            if (attackerStack.capHeight >= stackAtPos.capHeight) {
              targetPos = pos;
            }
            break;
          }

          step++;
        }

        if (!targetPos) continue;

        // From the target, walk further along the same ray to find candidate
        // landing positions. Each candidate is validated via
        // validateCaptureSegment to ensure consistency with validateMove.
        let landingStep = 1;
        while (true) {
          const landingPos: Position = {
            x: targetPos.x + dir.x * landingStep,
            y: targetPos.y + dir.y * landingStep,
            ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep }),
          };

          if (!this.boardManager.isValidPosition(landingPos)) {
            break;
          }

          // Collapsed spaces and stacks at the landing position block further
          // landings along this ray (Section 10.2).
          if (this.boardManager.isCollapsedSpace(landingPos, board)) {
            break;
          }

          const landingKey = positionToString(landingPos);
          const landingStack = board.stacks.get(landingKey);
          if (landingStack && landingStack.rings.length > 0) {
            break;
          }

          const testMove: Move = {
            id: '',
            type: 'overtaking_capture',
            player,
            from: stackPos,
            captureTarget: targetPos,
            to: landingPos,
            timestamp: new Date(),
            thinkTime: 0,
            moveNumber: 0,
          };

          if (this.validateCaptureSegment(stackPos, targetPos, landingPos, player, board)) {
            moves.push({
              ...testMove,
              id: `capture-${positionToString(stackPos)}-${positionToString(targetPos)}-${positionToString(landingPos)}`,
              moveNumber: gameState.moveHistory.length + 1,
            });
          }

          // Continue stepping to allow landing further along the ray, as the
          // rules permit landing on any valid space beyond the target
          // provided distance and path constraints are satisfied.
          landingStep++;
        }
      }
    }

    return moves;
  }

  /**
   * Directions along which captures may occur, based on board type.
   * For square boards we use the 8 Moore directions; for hex we use
   * the 6 standard cube-coordinate directions.
   */
  private getCaptureDirections(): { x: number; y: number; z?: number }[] {
    // Capture directions mirror movement directions: 8-direction Moore
    // adjacency for square boards and the 6 standard cube directions for hex.
    return getMovementDirectionsForBoardType(this.boardConfig.type as any);
  }

  /**
   * Helper methods
   */
  private isValidPlayer(player: number, gameState: GameState): boolean {
    return gameState.players.some((p) => p.playerNumber === player);
  }

  private isPlayerTurn(player: number, gameState: GameState): boolean {
    return gameState.currentPlayer === player;
  }

  /**
   * Determine the last player to complete a valid turn action, used as the
   * final rung of the stalemate tie-break ladder. Preference order:
   *
   * 1. The actor of the last structured history entry, when available.
   * 2. The player of the last legacy moveHistory entry.
   * 3. The player immediately preceding currentPlayer in turn order.
   *
   * This mirrors the intent of the complete rules that "the last player to
   * complete a valid turn action" wins when all other tiebreakers are
   * exhausted, while remaining robust for synthetic test states that may not
   * have full history recorded.
   */
  private getLastActor(gameState: GameState): number | undefined {
    // 1) Prefer the canonical structured history when present.
    if (gameState.history && gameState.history.length > 0) {
      const lastEntry = gameState.history[gameState.history.length - 1];
      if (lastEntry && typeof lastEntry.actor === 'number') {
        return lastEntry.actor;
      }
    }

    // 2) Fall back to the legacy moveHistory when available.
    if (gameState.moveHistory && gameState.moveHistory.length > 0) {
      const lastMove = gameState.moveHistory[gameState.moveHistory.length - 1];
      if (lastMove && typeof lastMove.player === 'number') {
        return lastMove.player;
      }
    }

    // 3) As a defensive fallback (primarily for unit tests that construct
    // minimal states), treat the previous player in turn order as the last
    // actor. This preserves the "no perfect tie" guarantee even when no
    // explicit history is recorded.
    const players = gameState.players;
    if (!players || players.length === 0) {
      return undefined;
    }

    const currentIdx = players.findIndex((p) => p.playerNumber === gameState.currentPlayer);
    if (currentIdx === -1) {
      return players[0].playerNumber;
    }

    const lastIdx = (currentIdx - 1 + players.length) % players.length;
    return players[lastIdx].playerNumber;
  }

  private getPlayerStacks(player: number, board: BoardState): Position[] {
    const positions: Position[] = [];

    for (const [, stack] of board.stacks) {
      if (stack.controllingPlayer === player) {
        positions.push(stack.position);
      }
    }

    return positions;
  }

  private getPlayerStats(gameState: GameState): {
    [player: number]: { totalRings: number; controlledPositions: number };
  } {
    const stats: { [player: number]: { totalRings: number; controlledPositions: number } } = {};

    // Initialize stats
    for (const player of gameState.players) {
      stats[player.playerNumber] = { totalRings: 0, controlledPositions: 0 };
    }

    // Count rings and controlled positions
    for (const [, stack] of gameState.board.stacks) {
      if (stack.rings.length > 0) {
        const player = stack.controllingPlayer;
        stats[player].totalRings += stack.rings.length;
        stats[player].controlledPositions += 1;
      }
    }

    return stats;
  }

  private calculateDistance(from: Position, to: Position): number {
    // Delegate to the shared core helper so distance semantics match
    // ClientSandboxEngine and capture validation (Chebyshev for
    // square boards, cube distance for hex).
    return calculateDistance(this.boardConfig.type as any, from, to);
  }

  private areAdjacent(pos1: Position, pos2: Position): boolean {
    const distance = this.calculateDistance(pos1, pos2);
    return distance === 1;
  }

  private getAdjacentPositions(pos: Position): Position[] {
    const adjacent: Position[] = [];

    if (this.boardConfig.type === 'hexagonal') {
      // Hexagonal adjacency
      const directions = [
        { x: 1, y: 0, z: -1 }, // East
        { x: 0, y: 1, z: -1 }, // Southeast
        { x: -1, y: 1, z: 0 }, // Southwest
        { x: -1, y: 0, z: 1 }, // West
        { x: 0, y: -1, z: 1 }, // Northwest
        { x: 1, y: -1, z: 0 }, // Northeast
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

  /**
   * True if the move from `from` to `to` is along a straight ray consistent
   * with the board's movement directions (8-direction Moore for square,
   * 6 cube-coordinate axes for hex). This mirrors the directional checks
   * used by the shared capture validator and sandbox movement.
   */
  private isStraightLineMovement(from: Position, to: Position): boolean {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);

    if (this.boardConfig.type === 'hexagonal') {
      // In cube coordinates, an axis-aligned ray changes exactly two
      // coordinates (the third is implied by x + y + z = 0).
      const coordChanges = [dx !== 0, dy !== 0, dz !== 0].filter(Boolean).length;
      return coordChanges === 2;
    }

    // Square boards: orthogonal or diagonal only.
    if (dx === 0 && dy === 0) {
      return false;
    }
    if (dx !== 0 && dy !== 0 && Math.abs(dx) !== Math.abs(dy)) {
      return false;
    }
    return true;
  }

  private cloneGameState(gameState: GameState): GameState {
    return {
      ...gameState,
      board: {
        ...gameState.board,
        stacks: new Map(gameState.board.stacks),
        markers: new Map(gameState.board.markers),
        territories: new Map(gameState.board.territories),
        formedLines: [...gameState.board.formedLines],
        eliminatedRings: { ...gameState.board.eliminatedRings },
      },
      moveHistory: [...gameState.moveHistory],
      players: [...gameState.players],
      spectators: [...gameState.spectators],
    };
  }

  /**
   * Creates a hypothetical board state with a ring placed at the specified position.
   * Used for placement validation to check if the placement leaves legal moves.
   */
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    player: number,
    count: number = 1
  ): BoardState {
    return createHypotheticalBoardWithPlacementHelper(board, position, player, count);
  }

  /**
   * Checks if a stack at the given position has any legal moves or captures.
   * Used for placement validation to ensure rings aren't placed in positions
   * with no legal moves.
   *
   * Rule Reference: Section 7.1 - Must have at least one legal move or capture
   */
  private hasAnyLegalMoveOrCaptureFrom(from: Position, player: number, board: BoardState): boolean {
    const view: MovementBoardView = {
      isValidPosition: (pos: Position) => this.boardManager.isValidPosition(pos),
      isCollapsedSpace: (pos: Position) => this.boardManager.isCollapsedSpace(pos, board),
      getStackAt: (pos: Position) => {
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        if (!stack) return undefined;
        return {
          controllingPlayer: stack.controllingPlayer,
          capHeight: stack.capHeight,
          stackHeight: stack.stackHeight,
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board),
    };

    return hasAnyLegalMoveOrCaptureFromOnBoard(this.boardConfig.type as any, from, player, view);
  }

  /**
   * Helper for hypothetical move checking - simpler path validation
   * that works with a board state rather than game state.
   */
  private isPathClearForHypothetical(from: Position, to: Position, board: BoardState): boolean {
    const pathPositions = getPathPositions(from, to).slice(1, -1);

    for (const pos of pathPositions) {
      if (this.boardManager.isCollapsedSpace(pos, board)) {
        return false;
      }

      const posKey = positionToString(pos);
      const stack = board.stacks.get(posKey);
      if (stack && stack.rings.length > 0) {
        return false;
      }
    }

    return true;
  }

  /**
   * Internal no-op hook to keep selected helper methods referenced so that
   * ts-node/TypeScript with noUnusedLocals can compile the server in dev
   * without treating them as dead code. This has no runtime impact; it
   * simply preserves helpers for parity/debug tooling and future rules
   * extensions.
   */
  /**
   * Validate line-processing decision moves (process_line / choose_line_reward)
   * during the dedicated 'line_processing' phase.
   *
   * For now we treat these as structurally valid when:
   * - The game is in line_processing phase,
   * - The move is made by the current player, and
   * - When formedLines[0] is provided, it matches one of the currently
   *   detected lines for that player according to BoardManager.findAllLines.
   *
   * This keeps decision moves aligned with the same line view used for
   * enumeration in GameEngine.getValidLineProcessingMoves.
   */
  private validateLineProcessingMove(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'line_processing') {
      return false;
    }

    if (move.player !== gameState.currentPlayer) {
      return false;
    }

    const allLines = this.boardManager.findAllLines(gameState.board);
    const playerLines = allLines.filter((line) => line.player === move.player);

    if (playerLines.length === 0) {
      return false;
    }

    // When the Move carries a concrete line in formedLines[0], ensure it
    // corresponds to one of the currently-detected lines by comparing
    // ordered marker positions.
    if (move.formedLines && move.formedLines.length > 0) {
      const target = move.formedLines[0];

      const matchesExisting = playerLines.some((line) => {
        if (line.positions.length !== target.positions.length) {
          return false;
        }
        return line.positions.every((pos, idx) =>
          positionsEqual(pos, target.positions[idx])
        );
      });

      if (!matchesExisting) {
        return false;
      }
    }

    return true;
  }

  /**
   * Validate territory-processing decision moves (process_territory_region)
   * during the dedicated 'territory_processing' phase.
   *
   * A move is considered valid when:
   * - The game is in territory_processing phase,
   * - The move is made by the current player, and
   * - When disconnectedRegions[0] is provided, its space set matches one of
   *   the regions currently reported by BoardManager.findDisconnectedRegions
   *   for the moving player.
   *
   * The stricter self-elimination prerequisite (having an outside stack) is
   * enforced by GameEngine.canProcessDisconnectedRegion when enumerating
   * these moves; here we focus on structural identity.
   */
  private validateTerritoryProcessingMove(move: Move, gameState: GameState): boolean {
    if (gameState.currentPhase !== 'territory_processing') {
      return false;
    }

    if (move.player !== gameState.currentPlayer) {
      return false;
    }

    const disconnectedRegions = this.boardManager.findDisconnectedRegions(
      gameState.board,
      move.player
    );

    if (!disconnectedRegions || disconnectedRegions.length === 0) {
      return false;
    }

    if (move.disconnectedRegions && move.disconnectedRegions.length > 0) {
      const target = move.disconnectedRegions[0];
      const targetKeySet = new Set(
        target.spaces.map((pos: Position) => positionToString(pos))
      );

      const matchesExisting = disconnectedRegions.some((region: Territory) => {
        if (region.spaces.length !== target.spaces.length) {
          return false;
        }

        const regionKeySet = new Set(
          region.spaces.map((pos: Position) => positionToString(pos))
        );

        if (regionKeySet.size !== targetKeySet.size) {
          return false;
        }

        for (const key of targetKeySet) {
          if (!regionKeySet.has(key)) {
            return false;
          }
        }

        return true;
      });

      if (!matchesExisting) {
        return false;
      }
    }

    return true;
  }

  private _debugUseInternalHelpers(): void {
    void this.getPlayerStats;
    void this.areAdjacent;
    void this.isPathClearForHypothetical;
  }
}
