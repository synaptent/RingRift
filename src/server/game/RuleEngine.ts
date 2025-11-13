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

export class RuleEngine {
  private boardManager: BoardManager;
  private boardConfig: typeof BOARD_CONFIGS[keyof typeof BOARD_CONFIGS];

  constructor(boardManager: BoardManager, boardType: keyof typeof BOARD_CONFIGS) {
    this.boardManager = boardManager;
    this.boardConfig = BOARD_CONFIGS[boardType];
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
    
    return totalRingsPlaced < this.boardConfig.ringsPerPlayer;
  }

  /**
   * Validates stack movement according to RingRift rules
   */
  private validateStackMovement(move: Move, gameState: GameState): boolean {
    // Stack movement is only allowed during main game phase
    if (gameState.currentPhase !== 'main_game') {
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

    // Check movement distance based on stack height
    const distance = this.calculateDistance(move.from, move.to);
    const minDistance = Math.max(1, sourceStack.stackHeight);
    const maxDistance = this.boardConfig.size; // Can move across entire board

    if (distance < minDistance || distance > maxDistance) {
      return false;
    }

    // Check if path is clear (using movement adjacency)
    const blockedPositions = new Set<string>();
    for (const [posKey, stack] of gameState.board.stacks) {
      if (stack.rings.length > 0 && posKey !== positionToString(move.to)) {
        blockedPositions.add(posKey);
      }
    }

    const path = this.boardManager.findPath(move.from, move.to, blockedPositions);
    if (!path || path.length === 0) {
      return false; // No valid path
    }

    return true;
  }

  /**
   * Validates capture move according to RingRift rules
   */
  private validateCapture(move: Move, gameState: GameState): boolean {
    // Captures are only allowed during main game phase
    if (gameState.currentPhase !== 'main_game') {
      return false;
    }

    if (!move.from || !move.capturedStacks || move.capturedStacks.length === 0) {
      return false;
    }

    const fromKey = positionToString(move.from);
    const attackerStack = gameState.board.stacks.get(fromKey);
    if (!attackerStack || attackerStack.controllingPlayer !== move.player) {
      return false;
    }

    // Validate each captured stack
    for (const capturedStack of move.capturedStacks) {
      if (!this.isValidCapture(move.from, capturedStack.position, gameState.board)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Checks if a capture is valid (overtaking rules)
   */
  private isValidCapture(attackerPos: Position, targetPos: Position, board: BoardState): boolean {
    const attackerKey = positionToString(attackerPos);
    const targetKey = positionToString(targetPos);
    
    const attackerStack = board.stacks.get(attackerKey);
    const targetStack = board.stacks.get(targetKey);

    if (!attackerStack || !targetStack) {
      return false;
    }

    // Cannot capture own stacks
    if (attackerStack.controllingPlayer === targetStack.controllingPlayer) {
      return false;
    }

    // Must be adjacent (using movement adjacency)
    if (!this.areAdjacent(attackerPos, targetPos)) {
      return false;
    }

    // Overtaking rules: attacker's cap height must be >= target's cap height
    return attackerStack.capHeight >= targetStack.capHeight;
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

    // Process each capture
    for (const capturedStack of capturedStacks) {
      const capturedKey = positionToString(capturedStack.position);
      gameState.board.stacks.delete(capturedKey);
    }

    // Add captured rings to attacker's stack
    if (capturedStacks.length > 0) {
      const totalCapturedRings = capturedStacks.reduce((sum, stack) => sum + stack.rings.length, 0);
      const newAttackerStack: RingStack = {
        ...attackerStack,
        rings: [...attackerStack.rings, ...capturedStacks.flatMap(s => s.rings)],
        stackHeight: attackerStack.stackHeight + totalCapturedRings,
        capHeight: attackerStack.capHeight,
        controllingPlayer: attackerStack.controllingPlayer
      };
      gameState.board.stacks.set(fromKey, newAttackerStack);
    }

    // Check for chain reactions
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
      case 'main_game':
        moves.push(...this.getValidStackMovements(currentPlayer, gameState));
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
   * Gets valid capture moves
   */
  private getValidCaptures(player: number, gameState: GameState): Move[] {
    const moves: Move[] = [];
    const playerStacks = this.getPlayerStacks(player, gameState.board);
    
    for (const stackPos of playerStacks) {
      const adjacentPositions = this.getAdjacentPositions(stackPos);
      const captureTargets: RingStack[] = [];

      for (const adjPos of adjacentPositions) {
        if (this.isValidCapture(stackPos, adjPos, gameState.board)) {
          const adjKey = positionToString(adjPos);
          const adjStack = gameState.board.stacks.get(adjKey);
          if (adjStack) {
            captureTargets.push(adjStack);
          }
        }
      }

      if (captureTargets.length > 0) {
        moves.push({
          id: `capture-${positionToString(stackPos)}`,
          type: 'overtaking_capture',
          player,
          from: stackPos,
          to: captureTargets[0].position, // Primary target
          capturedStacks: captureTargets,
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: gameState.moveHistory.length + 1
        });
      }
    }

    return moves;
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
}