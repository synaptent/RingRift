import {
  GameState,
  Move,
  Position,
  BoardState,
  RingStack,
  positionToString,
  positionsEqual,
  BOARD_CONFIGS
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import {
  calculateCapHeight,
  getMovementDirectionsForBoardType,
  validateCaptureSegmentOnBoard,
  CaptureSegmentBoardView
} from '../../shared/engine/core';

export class RuleEngine {
  private boardManager: BoardManager;
  private boardConfig: typeof BOARD_CONFIGS[keyof typeof BOARD_CONFIGS];
  private boardType: keyof typeof BOARD_CONFIGS;

  constructor(boardManager: BoardManager, boardType: keyof typeof BOARD_CONFIGS) {
    this.boardManager = boardManager;
    this.boardConfig = BOARD_CONFIGS[boardType];
    this.boardType = boardType;
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
      case 'move_stack':
        return this.validateStackMovement(move, gameState);
      case 'overtaking_capture':
        return this.validateCapture(move, gameState);
      default:
        return false;
    }
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

    // Check if position is valid and empty
    if (!this.boardManager.isValidPosition(move.to)) {
      return false;
    }

    const posKey = positionToString(move.to);
    const stack = gameState.board.stacks.get(posKey);
    if (stack && stack.rings.length > 0) {
      return false; // Position already occupied
    }

    // Check if player has rings available to place
    const playerStacks = this.getPlayerStacks(move.player, gameState.board);
    const totalRingsPlaced = playerStacks.reduce((sum, pos) => {
      const stackKey = positionToString(pos);
      const stack = gameState.board.stacks.get(stackKey);
      return sum + (stack ? stack.rings.length : 0);
    }, 0);
    
    if (totalRingsPlaced >= this.boardConfig.ringsPerPlayer) {
      return false;
    }

    // Rule fix: Placement must leave at least one legal move or capture
    // Create a hypothetical board state with the ring placed
    const hypotheticalBoard = this.createHypotheticalBoardWithPlacement(
      gameState.board,
      move.to,
      move.player
    );

    // Check if the resulting stack has any legal moves or captures
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
    const pathPositions = this.getPathPositions(from, to);
    
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
   * Gets all positions along a straight path (excluding start and end)
   * Helper for path validation and marker processing
   */
  private getPathPositions(from: Position, to: Position): Position[] {
    const positions: Position[] = [];
    
    // Calculate direction
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = (to.z || 0) - (from.z || 0);
    
    // Determine step count
    const steps = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
    
    if (steps === 0) {
      return positions; // Same position
    }
    
    // Calculate unit step
    const stepX = dx / steps;
    const stepY = dy / steps;
    const stepZ = dz / steps;
    
    // Generate intermediate positions
    for (let i = 1; i < steps; i++) {
      const x = Math.round(from.x + stepX * i);
      const y = Math.round(from.y + stepY * i);
      
      // Only include z if the positions have z coordinates (hexagonal board)
      if (from.z !== undefined || to.z !== undefined) {
        const z = Math.round((from.z || 0) + stepZ * i);
        positions.push({ x, y, z });
      } else {
        positions.push({ x, y });
      }
    }
    
    return positions;
  }

  /**
   * Validates overtaking capture move according to RingRift rules
   * Rule Reference: Section 10.1, Section 10.2
   */
  private validateCapture(move: Move, gameState: GameState): boolean {
    // Captures are only allowed during capture phase
    if (gameState.currentPhase !== 'capture') {
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
          stackHeight: stack.stackHeight
        };
      },
      getMarkerOwner: (pos: Position) => this.boardManager.getMarker(pos, board)
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
      controllingPlayer: move.player
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
        controllingPlayer: sourceStack.controllingPlayer
      };
      gameState.board.stacks.set(toKey, mergedStack);
    } else {
      // Move to empty position
      const movedStack: RingStack = {
        ...sourceStack,
        position: move.to
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
          controllingPlayer: remaining[0]
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
        controllingPlayer: newRings[0]
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
      if (adjStack && 
          adjStack.controllingPlayer !== triggerStack.controllingPlayer &&
          triggerStack.capHeight >= adjStack.capHeight) {
        
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
          moveNumber: gameState.moveHistory.length + 1
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
      const territories = this.boardManager.findAllTerritories(player.playerNumber, gameState.board);
      
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
   * Checks for game end conditions
   */
  checkGameEnd(gameState: GameState): { isGameOver: boolean; winner?: number; reason?: string } {
    const playerStats = this.getPlayerStats(gameState);
    const totalRings = Object.values(playerStats).reduce((sum, stats) => sum + stats.totalRings, 0);
    const totalPositions = this.boardConfig.totalSpaces;

    // Check ring elimination victory (>50% of total rings eliminated)
    for (const [playerNumber, stats] of Object.entries(playerStats)) {
      const eliminationPercentage = (totalRings - stats.totalRings) / totalRings;
      if (eliminationPercentage > 0.5) {
        return {
          isGameOver: true,
          winner: parseInt(playerNumber),
          reason: 'ring_elimination'
        };
      }
    }

    // Check territory control victory (>50% of board controlled)
    for (const [playerNumber, stats] of Object.entries(playerStats)) {
      const territoryPercentage = stats.controlledPositions / totalPositions;
      if (territoryPercentage > 0.5) {
        return {
          isGameOver: true,
          winner: parseInt(playerNumber),
          reason: 'territory_control'
        };
      }
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
      case 'ring_placement':
        moves.push(...this.getValidRingPlacements(currentPlayer, gameState));
        break;
      case 'movement':
        moves.push(...this.getValidStackMovements(currentPlayer, gameState));
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
    const allPositions = this.boardManager.getAllPositions();
    
    for (const pos of allPositions) {
      const posKey = positionToString(pos);
      const stack = gameState.board.stacks.get(posKey);
      if (!stack || stack.rings.length === 0) {
        moves.push({
          id: `place-${positionToString(pos)}`,
          type: 'place_ring',
          player,
          to: pos,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1
        });
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
          moveNumber: 0
        };

        if (this.validateStackMovement(testMove, gameState)) {
          moves.push({
            ...testMove,
            id: `move-${positionToString(stackPos)}-${positionToString(targetPos)}`,
            moveNumber: gameState.moveHistory.length + 1
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
            ...(dir.z !== undefined && { z: (stackPos.z || 0) + dir.z * step })
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
            ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
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
            moveNumber: 0
          };

          if (this.validateCaptureSegment(stackPos, targetPos, landingPos, player, board)) {
            moves.push({
              ...testMove,
              id: `capture-${positionToString(stackPos)}-${positionToString(targetPos)}-${positionToString(landingPos)}`,
              moveNumber: gameState.moveHistory.length + 1
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
    return gameState.players.some(p => p.playerNumber === player);
  }

  private isPlayerTurn(player: number, gameState: GameState): boolean {
    return gameState.currentPlayer === player;
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

  private getPlayerStats(gameState: GameState): { [player: number]: { totalRings: number; controlledPositions: number } } {
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
    if (this.boardConfig.type === 'hexagonal') {
      // Hexagonal distance
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const dz = (to.z || 0) - (from.z || 0);
      return (Math.abs(dx) + Math.abs(dy) + Math.abs(dz)) / 2;
    } else {
      // Manhattan distance for square boards
      return Math.abs(to.x - from.x) + Math.abs(to.y - from.y);
    }
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
        { x: 1, y: 0, z: -1 },   // East
        { x: 0, y: 1, z: -1 },   // Southeast
        { x: -1, y: 1, z: 0 },   // Southwest
        { x: -1, y: 0, z: 1 },   // West
        { x: 0, y: -1, z: 1 },   // Northwest
        { x: 1, y: -1, z: 0 }    // Northeast
      ];
      
      for (const dir of directions) {
        const newPos: Position = {
          x: pos.x + dir.x,
          y: pos.y + dir.y,
          z: (pos.z || 0) + dir.z
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
            y: pos.y + dy
          };
          if (this.boardManager.isValidPosition(newPos)) {
            adjacent.push(newPos);
          }
        }
      }
    }

    return adjacent;
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
        eliminatedRings: { ...gameState.board.eliminatedRings }
      },
      moveHistory: [...gameState.moveHistory],
      players: [...gameState.players],
      spectators: [...gameState.spectators]
    };
  }

  /**
   * Creates a hypothetical board state with a ring placed at the specified position.
   * Used for placement validation to check if the placement leaves legal moves.
   */
  private createHypotheticalBoardWithPlacement(
    board: BoardState,
    position: Position,
    player: number
  ): BoardState {
    const hypotheticalBoard: BoardState = {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings }
    };

    const posKey = positionToString(position);
    const existingStack = hypotheticalBoard.stacks.get(posKey);

    if (existingStack && existingStack.rings.length > 0) {
      // Placing on existing stack (shouldn't happen in normal play but handle safely)
      const newStack: RingStack = {
        ...existingStack,
        rings: [player, ...existingStack.rings],
        stackHeight: existingStack.stackHeight + 1,
        capHeight: 1, // New ring on top forms cap height 1
        controllingPlayer: player
      };
      hypotheticalBoard.stacks.set(posKey, newStack);
    } else {
      // Placing on empty space
      const newStack: RingStack = {
        position,
        rings: [player],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: player
      };
      hypotheticalBoard.stacks.set(posKey, newStack);
    }

    return hypotheticalBoard;
  }

  /**
   * Checks if a stack at the given position has any legal moves or captures.
   * Used for placement validation to ensure rings aren't placed in positions
   * with no legal moves.
   *
   * Rule Reference: Section 7.1 - Must have at least one legal move or capture
   */
  private hasAnyLegalMoveOrCaptureFrom(
    from: Position,
    player: number,
    board: BoardState
  ): boolean {
    const fromKey = positionToString(from);
    const stack = board.stacks.get(fromKey);
    if (!stack || stack.controllingPlayer !== player) {
      return false;
    }

    const directions = getMovementDirectionsForBoardType(this.boardConfig.type as any);

    // Check for any legal non-capture movement
    // Stack height is 1 for a newly placed ring, so we need distance >= 1
    for (const dir of directions) {
      for (let distance = stack.stackHeight; distance <= 8; distance++) {
        const targetPos: Position = {
          x: from.x + dir.x * distance,
          y: from.y + dir.y * distance,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * distance })
        };

        if (!this.boardManager.isValidPosition(targetPos)) {
          break; // Off board in this direction
        }

        if (this.boardManager.isCollapsedSpace(targetPos, board)) {
          break; // Can't move through collapsed space
        }

        // Check if path is clear
        const pathClear = this.isPathClearForHypothetical(from, targetPos, board);
        if (!pathClear) {
          break; // Blocked by stacks/collapsed spaces
        }

        // Check landing position
        const targetKey = positionToString(targetPos);
        const targetStack = board.stacks.get(targetKey);
        const targetMarker = this.boardManager.getMarker(targetPos, board);

        // Can land on empty space, same-color marker, or own/opponent stacks (for merging)
        if (!targetStack || targetStack.rings.length === 0) {
          // Empty space or marker
          if (targetMarker === undefined || targetMarker === player) {
            return true; // Found a legal move
          }
        } else {
          // Landing on a stack - allowed for merging
          return true;
        }
      }
    }

    // Check for any legal capture
    for (const dir of directions) {
      let step = 1;
      let targetPos: Position | undefined;

      // Find first stack along this ray
      while (true) {
        const pos: Position = {
          x: from.x + dir.x * step,
          y: from.y + dir.y * step,
          ...(dir.z !== undefined && { z: (from.z || 0) + dir.z * step })
        };

        if (!this.boardManager.isValidPosition(pos)) {
          break;
        }

        if (this.boardManager.isCollapsedSpace(pos, board)) {
          break;
        }

        const posKey = positionToString(pos);
        const stackAtPos = board.stacks.get(posKey);

        if (stackAtPos && stackAtPos.rings.length > 0) {
          // Check if this stack is capturable
          if (stack.capHeight >= stackAtPos.capHeight) {
            targetPos = pos;
          }
          break;
        }

        step++;
      }

      if (!targetPos) continue;

      // Try to find at least one valid landing position beyond the target
      let landingStep = 1;
      while (landingStep <= 5) {
        // Check a few positions
        const landingPos: Position = {
          x: targetPos.x + dir.x * landingStep,
          y: targetPos.y + dir.y * landingStep,
          ...(dir.z !== undefined && { z: (targetPos.z || 0) + dir.z * landingStep })
        };

        if (!this.boardManager.isValidPosition(landingPos)) {
          break;
        }

        if (this.boardManager.isCollapsedSpace(landingPos, board)) {
          break;
        }

        const landingKey = positionToString(landingPos);
        const landingStack = board.stacks.get(landingKey);
        if (landingStack && landingStack.rings.length > 0) {
          break;
        }

        // Validate this capture segment
        if (this.validateCaptureSegment(from, targetPos, landingPos, player, board)) {
          return true; // Found a legal capture
        }

        landingStep++;
      }
    }

    return false; // No legal moves or captures found
  }

  /**
   * Helper for hypothetical move checking - simpler path validation
   * that works with a board state rather than game state.
   */
  private isPathClearForHypothetical(from: Position, to: Position, board: BoardState): boolean {
    const pathPositions = this.getPathPositions(from, to);

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
}
