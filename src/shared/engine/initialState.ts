import { GameState } from './types';
import { BoardType, TimeControl, Player, BOARD_CONFIGS } from '../types/game';

/**
 * Creates a pristine initial GameState for a new game.
 *
 * @param gameId - Unique identifier for the game
 * @param boardType - The type of board to use (square8, square19, hexagonal)
 * @param players - Array of players participating in the game
 * @param timeControl - Time control settings
 * @param isRated - Whether the game is rated (default: true)
 * @returns A new immutable GameState object
 */
export function createInitialGameState(
  gameId: string,
  boardType: BoardType,
  players: Player[],
  timeControl: TimeControl,
  isRated: boolean = true
): GameState {
  const config = BOARD_CONFIGS[boardType];

  // Initialize players with starting values
  const initializedPlayers = players.map((p, index) => ({
    ...p,
    playerNumber: index + 1,
    timeRemaining: timeControl.initialTime * 1000, // Convert to milliseconds
    isReady: p.type === 'ai', // AI players are always ready
    ringsInHand: config.ringsPerPlayer,
    eliminatedRings: 0,
    territorySpaces: 0,
  }));

  // Create empty board state
  const board = {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {} as { [player: number]: number },
    size: config.size,
    type: boardType,
  };

  // Initialize eliminated rings count for each player
  initializedPlayers.forEach((p) => {
    board.eliminatedRings[p.playerNumber] = 0;
  });

  return {
    id: gameId,
    board,
    players: initializedPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    gameStatus: 'waiting',
    timeControl,
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated,
    maxPlayers: players.length,
    totalRingsInPlay: 0, // Starts at 0, increments as rings are placed
    totalRingsEliminated: 0,
    victoryThreshold: Math.floor((config.ringsPerPlayer * players.length) / 2) + 1,
    territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1,
  };
}
