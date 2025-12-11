import {
  GameState,
  BoardType,
  TimeControl,
  Player,
  BOARD_CONFIGS,
  RulesOptions,
} from '../types/game';
import { generateGameSeed } from '../utils/rng';

/**
 * Creates a pristine initial GameState for a new game.
 *
 * @param gameId - Unique identifier for the game
 * @param boardType - The type of board to use (square8, square19, hexagonal)
 * @param players - Array of players participating in the game
 * @param timeControl - Time control settings
 * @param isRated - Whether the game is rated (default: true)
 * @param rngSeed - Optional RNG seed for deterministic games; auto-generated if not provided
 * @param rulesOptions - Optional per-game rules configuration (e.g., swap rule)
 * @returns A new immutable GameState object
 */
export function createInitialGameState(
  gameId: string,
  boardType: BoardType,
  players: Player[],
  timeControl: TimeControl,
  isRated: boolean = true,
  rngSeed?: number,
  rulesOptions?: RulesOptions
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
    boardType,
    rngSeed: rngSeed ?? generateGameSeed(),
    board,
    players: initializedPlayers,
    currentPhase: 'ring_placement',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    gameStatus: 'waiting',
    spectators: [],
    timeControl,
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated,
    // Optional per-game rules configuration (e.g., swap rule, future variants).
    // Callers may omit this to use host-level defaults.
    ...(rulesOptions ? { rulesOptions } : {}),
    maxPlayers: players.length,
    totalRingsInPlay: 0, // Starts at 0, increments as rings are placed
    totalRingsEliminated: 0,
    // Per RR-CANON-R061: victoryThreshold = round((1/3) × ownStartingRings + (2/3) × opponentsCombinedStartingRings)
    // Simplified: round(ringsPerPlayer × (1/3 + 2/3 × (numPlayers - 1)))
    // Note: Using Math.round() to handle floating-point precision (e.g., 18 * (1/3 + 2/3*2) = 29.999... → 30)
    // This makes elimination harder in 2p (need to eliminate more of opponent's rings)
    // and scales appropriately for 3p/4p games
    victoryThreshold: Math.round(config.ringsPerPlayer * (1 / 3 + (2 / 3) * (players.length - 1))),
    territoryVictoryThreshold: Math.floor(config.totalSpaces / 2) + 1,
  };
}
