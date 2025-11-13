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
  LineInfo
} from '../../shared/types/game';
import { BoardManager } from './BoardManager';
import { RuleEngine } from './RuleEngine';

// Timer functions for Node.js environment
declare const setTimeout: (callback: () => void, ms: number) => any;
declare const clearTimeout: (timer: any) => void;

// Using a simple UUID generator for now
function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

export class GameEngine {
  private gameState: GameState;
  private boardManager: BoardManager;
  private ruleEngine: RuleEngine;
  private moveTimers: Map<number, any> = new Map();

  constructor(
    gameId: string,
    boardType: BoardType,
    players: Player[],
    timeControl: TimeControl,
    isRated: boolean = true
  ) {
    this.boardManager = new BoardManager(boardType);
    this.ruleEngine = new RuleEngine(this.boardManager, boardType);
    
    const config = BOARD_CONFIGS[boardType];
    
    this.gameState = {
      id: gameId,
      boardType,
      board: this.boardManager.createBoard(),
      players: players.map((p, index) => ({
        ...p,
        playerNumber: index + 1,
        timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
        isReady: p.type === 'ai' // AI players are always ready
      })),
      currentPhase: 'ring_placement',
      currentPlayer: 1,
      moveHistory: [],
      timeControl,
      spectators: [],
      gameStatus: 'waiting',
      createdAt: new Date(),
      lastMoveAt: new Date(),
      isRated,
      maxPlayers: players.length,
      totalRingsInPlay: config.ringsPerPlayer * players.length,
      totalRingsEliminated: 0,
      victoryThreshold: Math.floor(config.ringsPerPlayer * players.length / 2) + 1,
      territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1
    };
  }

  getGameState(): GameState {
    return { ...this.gameState };
  }

  startGame(): boolean {
    // Check if all players are ready
    const allReady = this.gameState.players.every(p => p.isReady);
    if (!allReady) {
      return false;
    }

    this.gameState.gameStatus = 'active';
    this.gameState.lastMoveAt = new Date();
    
    // Start the first player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);
    
    return true;
  }

  makeMove(move: Omit<Move, 'id' | 'timestamp' | 'moveNumber'>): {
    success: boolean;
    error?: string;
    gameState?: GameState;
    gameResult?: GameResult;
  } {
    // Validate the move
    const fullMove: Move = {
      ...move,
      id: generateUUID(),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: this.gameState.moveHistory.length + 1
    };

    const validation = this.ruleEngine.validateMove(fullMove, this.gameState);
    if (!validation) {
      return {
        success: false,
        error: 'Invalid move'
      };
    }

    // Stop current player's timer
    this.stopPlayerTimer(this.gameState.currentPlayer);

    // Apply the move
    const moveResult = this.applyMove(fullMove);
    
    // Add move to history
    this.gameState.moveHistory.push(fullMove);
    this.gameState.lastMoveAt = new Date();

    // Process automatic consequences
    this.processAutomaticConsequences(moveResult);

    // Check for game end conditions
    const gameEndCheck = this.ruleEngine.checkGameEnd(this.gameState);
    if (gameEndCheck.isGameOver) {
      return this.endGame(gameEndCheck.winner, gameEndCheck.reason || 'unknown');
    }

    // Advance to next phase/player
    this.advanceGame();

    // Start next player's timer
    this.startPlayerTimer(this.gameState.currentPlayer);

    return {
      success: true,
      gameState: this.getGameState()
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
      lineCollapses: [] as LineInfo[]
    };

    switch (move.type) {
      case 'place_ring':
        if (move.to) {
          const newStack: RingStack = {
            position: move.to,
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: move.player,
            rings: [move.player]
          };
          this.boardManager.setStack(move.to, newStack, this.gameState.board);
        }
        break;

      case 'move_ring':
        if (move.from && move.to) {
          const stack = this.boardManager.getStack(move.from, this.gameState.board);
          if (stack) {
            this.boardManager.removeStack(move.from, this.gameState.board);
            
            const destinationStack = this.boardManager.getStack(move.to, this.gameState.board);
            if (destinationStack && destinationStack.controllingPlayer !== move.player) {
              // Overtaking capture
              const newStack: RingStack = {
                position: move.to,
                stackHeight: stack.stackHeight + destinationStack.stackHeight,
                capHeight: stack.stackHeight,
                controllingPlayer: move.player,
                rings: [...stack.rings, ...destinationStack.rings]
              };
              this.boardManager.setStack(move.to, newStack, this.gameState.board);
              
              // Check for chain reactions
              // Chain captures are handled internally by the rule engine
              // Additional captures would be processed here if needed
            } else {
              // Normal movement
              this.boardManager.setStack(move.to, stack, this.gameState.board);
            }
          }
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
              rings: remainingRings
            };
            
            const newTargetStack: RingStack = {
              ...targetStack,
              stackHeight: targetStack.stackHeight + move.buildAmount,
              capHeight: Math.max(targetStack.capHeight, move.buildAmount),
              rings: [...targetStack.rings, ...transferRings]
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

    // Check for line formation and collapse
    const lines = this.boardManager.findAllLines(this.gameState.board);
    const config = BOARD_CONFIGS[this.gameState.boardType];
    
    for (const line of lines) {
      if (line.positions.length >= config.lineLength) {
        // Collapse the line
        for (const pos of line.positions) {
          this.boardManager.removeStack(pos, this.gameState.board);
        }
        result.lineCollapses.push(line);
      }
    }

    // Check for territory disconnection
    const territories = this.boardManager.findAllTerritoriesForAllPlayers(this.gameState.board);
    for (const territory of territories) {
      if (territory.isDisconnected) {
        // Remove disconnected territory
        for (const pos of territory.spaces) {
          this.boardManager.removeStack(pos, this.gameState.board);
        }
        result.territoryChanges.push(territory);
      }
    }

    return result;
  }

  private processAutomaticConsequences(moveResult: {
    captures: Position[];
    territoryChanges: Territory[];
    lineCollapses: LineInfo[];
  }): void {
    // Process captures
    for (const capturePos of moveResult.captures) {
      this.boardManager.removeStack(capturePos, this.gameState.board);
    }

    // Process territory changes
    for (const territory of moveResult.territoryChanges) {
      for (const pos of territory.spaces) {
        this.boardManager.removeStack(pos, this.gameState.board);
      }
    }

    // Line collapses are already processed in applyMove
  }

  private advanceGame(): void {
    
    switch (this.gameState.currentPhase) {
      case 'ring_placement':
        // Check if player has any stacks on the board
        const playerHasStacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer).length > 0;
        if (playerHasStacks) {
          this.gameState.currentPhase = 'movement';
        } else {
          // Move to next player for ring placement
          this.nextPlayer();
        }
        break;

      case 'movement':
        // Check if player has valid moves
        const stacks = this.boardManager.getPlayerStacks(this.gameState.board, this.gameState.currentPlayer);
        if (stacks.length === 0) {
          // Player eliminated, move to next player
          this.nextPlayer();
        } else {
          // Move to capture phase
          this.gameState.currentPhase = 'capture';
        }
        break;

      case 'capture':
        this.gameState.currentPhase = 'territory_processing';
        break;

      case 'territory_processing':
        // Complete turn, move to next player
        this.nextPlayer();
        this.gameState.currentPhase = 'movement';
        break;
    }
  }

  private nextPlayer(): void {
    const currentIndex = this.gameState.players.findIndex(p => p.playerNumber === this.gameState.currentPlayer);
    const nextIndex = (currentIndex + 1) % this.gameState.players.length;
    this.gameState.currentPlayer = this.gameState.players[nextIndex].playerNumber;
  }

  private startPlayerTimer(playerNumber: number): void {
    const player = this.gameState.players.find(p => p.playerNumber === playerNumber);
    if (!player || player.type === 'ai') return;

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

  private endGame(winner?: number, reason?: string): {
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
      const playerStacks = this.boardManager.getPlayerStacks(this.gameState.board, player.playerNumber);
      const stackCount = playerStacks.reduce((sum, stack) => sum + stack.stackHeight, 0);
      
      const territories = this.boardManager.findPlayerTerritories(this.gameState.board, player.playerNumber);
      const territorySize = territories.reduce((sum, territory) => sum + territory.spaces.length, 0);
      
      finalScore[player.playerNumber] = stackCount + territorySize;
    }

    const gameResult: GameResult = {
      ...(winner !== undefined && { winner }),
      reason: (reason as any) || 'game_completed',
      finalScore: {
        ringsEliminated: {},
        territorySpaces: {},
        ringsRemaining: finalScore
      },
    };

    // Update player ratings if this is a rated game
    if (this.gameState.isRated) {
      this.updatePlayerRatings(gameResult);
    }

    return {
      success: true,
      gameResult
    };
  }

  private updatePlayerRatings(gameResult: GameResult): void {
    // Rating calculation logic would go here
    const winnerPlayer = this.gameState.players.find(p => p.playerNumber === gameResult.winner);
    const loserPlayers = this.gameState.players.filter(p => p.playerNumber !== gameResult.winner);

    // For now, just log the rating update
    console.log('Rating update needed for:', {
      winner: winnerPlayer?.username,
      losers: loserPlayers.map(p => p.username)
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
    const winner = this.gameState.players.find(p => p.playerNumber !== parseInt(playerNumber))?.playerNumber;
    
    return this.endGame(winner, 'resignation');
  }

  getValidMoves(_playerNumber: number): Move[] {
    // This would return all valid moves for the current player
    // For now, return empty array
    return [];
  }

}