/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Game View Model Adapters
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module provides view model types and adapter functions that transform
 * shared engine types (GameState, BoardState, etc.) into presentation-ready
 * data structures.
 *
 * Goals:
 * - Decouple UI components from direct GameState knowledge
 * - Make components easier to test in isolation
 * - Allow UI iteration without touching core game state
 * - Provide clear input contracts for each component
 *
 * Pattern:
 * - Define view model interfaces (what the UI needs)
 * - Create transformer functions that extract from GameState
 * - Components receive view models as props, not raw domain types
 */

import type {
  GameState,
  GamePhase,
  GameResult,
  Player,
  Position,
  BoardState,
  RingStack,
  MarkerInfo,
  GameHistoryEntry,
  Move,
  BoardType,
  BOARD_CONFIGS,
} from '../../shared/types/game';
import { positionToString } from '../../shared/types/game';
import type { ConnectionStatus } from '../domain/GameAPI';

// ═══════════════════════════════════════════════════════════════════════════
// HUD View Model
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Phase information for display in the HUD
 */
export interface PhaseViewModel {
  /** Internal phase key for styling/logic */
  phaseKey: GamePhase;
  /** User-friendly label */
  label: string;
  /** Brief description of current phase actions */
  description: string;
  /** Tailwind color class for phase indicator */
  colorClass: string;
  /** Emoji icon for the phase */
  icon: string;
}

/**
 * Player ring statistics for display
 */
export interface PlayerRingStatsViewModel {
  inHand: number;
  onBoard: number;
  eliminated: number;
  total: number;
}

/**
 * AI configuration display information
 */
export interface AIInfoViewModel {
  isAI: boolean;
  difficulty?: number;
  difficultyLabel?: string;
  difficultyColor?: string;
  difficultyBgColor?: string;
  aiTypeLabel?: string;
}

/**
 * Per-player display information
 */
export interface PlayerViewModel {
  id: string;
  playerNumber: number;
  username: string;
  isCurrentPlayer: boolean;
  isUserPlayer: boolean;
  colorClass: string;
  ringStats: PlayerRingStatsViewModel;
  territorySpaces: number;
  timeRemaining?: number;
  aiInfo: AIInfoViewModel;
}

/**
 * Complete HUD view model
 */
export interface HUDViewModel {
  phase: PhaseViewModel;
  players: PlayerViewModel[];
  turnNumber: number;
  moveNumber: number;
  instruction?: string;
  connectionStatus: ConnectionStatus;
  isConnectionStale: boolean;
  isSpectator: boolean;
  spectatorCount: number;
  /**
   * Sub-phase details (e.g., "Processing 3 lines")
   */
  subPhaseDetail?: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Log View Model
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Single event log entry for display
 */
export interface EventLogItemViewModel {
  /** Unique key for React list rendering */
  key: string;
  /** Display text for the entry */
  text: string;
  /** Entry type for styling */
  type: 'move' | 'system' | 'victory';
  /** Move number if applicable */
  moveNumber?: number;
}

/**
 * Complete event log view model
 */
export interface EventLogViewModel {
  entries: EventLogItemViewModel[];
  victoryMessage?: string;
  hasContent: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Board View Model
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Single ring in a stack for display
 */
export interface RingViewModel {
  playerNumber: number;
  colorClass: string;
  borderClass: string;
  isTop: boolean;
  isInCap: boolean;
}

/**
 * Stack display model
 */
export interface StackViewModel {
  position: Position;
  positionKey: string;
  rings: RingViewModel[];
  stackHeight: number;
  capHeight: number;
  controllingPlayer: number;
}

/**
 * Marker display model
 */
export interface MarkerViewModel {
  position: Position;
  positionKey: string;
  playerNumber: number;
  colorClass: string;
}

/**
 * Collapsed space (territory) display model
 */
export interface CollapsedSpaceViewModel {
  position: Position;
  positionKey: string;
  ownerPlayerNumber: number;
  territoryColorClass: string;
  borderClass: string;
}

/**
 * Cell display state
 */
export interface CellViewModel {
  position: Position;
  positionKey: string;
  stack?: StackViewModel;
  marker?: MarkerViewModel;
  collapsedSpace?: CollapsedSpaceViewModel;
  isSelected: boolean;
  isValidTarget: boolean;
  isDarkSquare: boolean;
}

/**
 * Complete board view model
 */
export interface BoardViewModel {
  boardType: BoardType;
  size: number;
  cells: CellViewModel[];
  /**
   * For square boards: cells organized by row for grid rendering
   */
  rows?: CellViewModel[][];
}

// ═══════════════════════════════════════════════════════════════════════════
// Victory Modal View Model
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Player final stats for victory display
 */
export interface PlayerFinalStatsViewModel {
  player: PlayerViewModel;
  ringsOnBoard: number;
  ringsEliminated: number;
  territorySpaces: number;
  totalMoves: number;
  isWinner: boolean;
}

/**
 * Victory modal view model
 */
export interface VictoryViewModel {
  isVisible: boolean;
  title: string;
  description: string;
  titleColorClass: string;
  winner?: PlayerViewModel;
  finalStats: PlayerFinalStatsViewModel[];
  gameSummary: {
    boardType: BoardType;
    totalTurns: number;
    playerCount: number;
    isRated: boolean;
  };
  userWon: boolean;
  userLost: boolean;
  isDraw: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Color Constants
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Player color palette for consistent theming
 */
export const PLAYER_COLORS = {
  1: {
    ring: 'bg-emerald-400',
    ringBorder: 'border-emerald-200',
    marker: 'border-emerald-400',
    territory: 'bg-emerald-700/85',
    card: 'bg-emerald-500',
    hex: '#10b981',
  },
  2: {
    ring: 'bg-sky-600',
    ringBorder: 'border-sky-300',
    marker: 'border-sky-500',
    territory: 'bg-sky-700/85',
    card: 'bg-sky-500',
    hex: '#3b82f6',
  },
  3: {
    ring: 'bg-amber-400',
    ringBorder: 'border-amber-200',
    marker: 'border-amber-400',
    territory: 'bg-amber-600/85',
    card: 'bg-amber-500',
    hex: '#f59e0b',
  },
  4: {
    ring: 'bg-fuchsia-400',
    ringBorder: 'border-fuchsia-200',
    marker: 'border-fuchsia-400',
    territory: 'bg-fuchsia-700/85',
    card: 'bg-fuchsia-500',
    hex: '#d946ef',
  },
} as const;

const DEFAULT_PLAYER_COLORS = {
  ring: 'bg-slate-300',
  ringBorder: 'border-slate-100',
  marker: 'border-slate-300',
  territory: 'bg-slate-800/70',
  card: 'bg-slate-500',
  hex: '#64748b',
};

export function getPlayerColors(playerNumber?: number) {
  if (!playerNumber) return DEFAULT_PLAYER_COLORS;
  return PLAYER_COLORS[playerNumber as keyof typeof PLAYER_COLORS] || DEFAULT_PLAYER_COLORS;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase Helpers
// ═══════════════════════════════════════════════════════════════════════════

const PHASE_INFO: Record<GamePhase, Omit<PhaseViewModel, 'phaseKey'>> = {
  ring_placement: {
    label: 'Ring Placement',
    description: 'Place your rings on the board',
    colorClass: 'bg-blue-500',
    icon: '🎯',
  },
  movement: {
    label: 'Movement Phase',
    description: 'Move a stack or capture opponent pieces',
    colorClass: 'bg-green-500',
    icon: '⚡',
  },
  capture: {
    label: 'Capture Phase',
    description: 'Execute a capture move',
    colorClass: 'bg-orange-500',
    icon: '⚔️',
  },
  chain_capture: {
    label: 'Chain Capture',
    description: 'Continue capturing or end your turn',
    colorClass: 'bg-orange-500',
    icon: '🔗',
  },
  line_processing: {
    label: 'Line Reward',
    description: 'Choose how to process your line',
    colorClass: 'bg-purple-500',
    icon: '📏',
  },
  territory_processing: {
    label: 'Territory Claim',
    description: 'Choose regions to collapse',
    colorClass: 'bg-pink-500',
    icon: '🏰',
  },
};

function toPhaseViewModel(phase: GamePhase): PhaseViewModel {
  const info = PHASE_INFO[phase] || {
    label: 'Unknown Phase',
    description: '',
    colorClass: 'bg-gray-400',
    icon: '❓',
  };
  return { phaseKey: phase, ...info };
}

// ═══════════════════════════════════════════════════════════════════════════
// AI Helpers
// ═══════════════════════════════════════════════════════════════════════════

function getAIDifficultyInfo(difficulty: number): {
  label: string;
  color: string;
  bgColor: string;
} {
  if (difficulty === 1) {
    return { label: 'Beginner · Random', color: 'text-green-300', bgColor: 'bg-green-900/40' };
  }
  if (difficulty === 2) {
    return { label: 'Easy · Heuristic', color: 'text-emerald-300', bgColor: 'bg-emerald-900/40' };
  }
  if (difficulty >= 3 && difficulty <= 6) {
    return { label: 'Advanced · Minimax', color: 'text-blue-300', bgColor: 'bg-blue-900/40' };
  }
  if (difficulty === 7 || difficulty === 8) {
    return { label: 'Expert · MCTS', color: 'text-purple-300', bgColor: 'bg-purple-900/40' };
  }
  return { label: 'Grandmaster · Descent', color: 'text-red-300', bgColor: 'bg-red-900/40' };
}

function getAITypeLabel(aiType?: string): string {
  switch (aiType) {
    case 'random':
      return 'Random';
    case 'heuristic':
      return 'Heuristic';
    case 'minimax':
      return 'Minimax';
    case 'mcts':
      return 'MCTS';
    case 'descent':
      return 'Descent';
    default:
      return 'AI';
  }
}

function toAIInfoViewModel(player: Player): AIInfoViewModel {
  if (player.type !== 'ai') {
    return { isAI: false };
  }

  const difficulty = player.aiProfile?.difficulty ?? player.aiDifficulty ?? 5;
  const aiType = player.aiProfile?.aiType ?? 'heuristic';
  const diffInfo = getAIDifficultyInfo(difficulty);

  return {
    isAI: true,
    difficulty,
    difficultyLabel: diffInfo.label,
    difficultyColor: diffInfo.color,
    difficultyBgColor: diffInfo.bgColor,
    aiTypeLabel: getAITypeLabel(aiType),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Ring Statistics
// ═══════════════════════════════════════════════════════════════════════════

const BOARD_CONFIGS_LOCAL = {
  square8: { ringsPerPlayer: 18 },
  square19: { ringsPerPlayer: 36 },
  hexagonal: { ringsPerPlayer: 36 },
} as const;

function calculateRingStats(player: Player, gameState: GameState): PlayerRingStatsViewModel {
  const boardConfig = BOARD_CONFIGS_LOCAL[gameState.boardType];
  const total = boardConfig?.ringsPerPlayer ?? 18;

  let onBoard = 0;
  for (const stack of gameState.board.stacks.values()) {
    onBoard += stack.rings.filter((r) => r === player.playerNumber).length;
  }

  return {
    inHand: player.ringsInHand ?? 0,
    onBoard,
    eliminated: player.eliminatedRings ?? 0,
    total,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Player View Model
// ═══════════════════════════════════════════════════════════════════════════

function toPlayerViewModel(
  player: Player,
  gameState: GameState,
  currentUserId?: string
): PlayerViewModel {
  const colors = getPlayerColors(player.playerNumber);
  return {
    id: player.id,
    playerNumber: player.playerNumber,
    username: player.username || `Player ${player.playerNumber}`,
    isCurrentPlayer: player.playerNumber === gameState.currentPlayer,
    isUserPlayer: player.id === currentUserId,
    colorClass: colors.card,
    ringStats: calculateRingStats(player, gameState),
    territorySpaces: player.territorySpaces ?? 0,
    timeRemaining: player.timeRemaining,
    aiInfo: toAIInfoViewModel(player),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// HUD Transformer
// ═══════════════════════════════════════════════════════════════════════════

export interface ToHUDViewModelOptions {
  instruction?: string;
  connectionStatus: ConnectionStatus;
  lastHeartbeatAt: number | null;
  isSpectator: boolean;
  currentUserId?: string;
}

/**
 * Transform GameState into HUDViewModel
 */
export function toHUDViewModel(gameState: GameState, options: ToHUDViewModelOptions): HUDViewModel {
  const { instruction, connectionStatus, lastHeartbeatAt, isSpectator, currentUserId } = options;

  const HEARTBEAT_STALE_THRESHOLD_MS = 8000;
  const heartbeatAge = lastHeartbeatAt ? Date.now() - lastHeartbeatAt : null;
  const isConnectionStale =
    heartbeatAge !== null &&
    heartbeatAge > HEARTBEAT_STALE_THRESHOLD_MS &&
    connectionStatus === 'connected';

  const phase = toPhaseViewModel(gameState.currentPhase);
  const players = gameState.players.map((p) => toPlayerViewModel(p, gameState, currentUserId));

  const turnNumber = gameState.moveHistory.length;
  const moveNumber =
    gameState.history.length > 0 ? gameState.history[gameState.history.length - 1]?.moveNumber : 0;

  // Sub-phase detail
  let subPhaseDetail: string | undefined;
  if (gameState.currentPhase === 'line_processing') {
    const lineCount = gameState.board.formedLines?.length ?? 0;
    if (lineCount > 0) {
      subPhaseDetail = `Processing ${lineCount} line${lineCount !== 1 ? 's' : ''}`;
    }
  } else if (gameState.currentPhase === 'territory_processing') {
    subPhaseDetail = 'Processing disconnected regions';
  }

  return {
    phase,
    players,
    turnNumber,
    moveNumber: moveNumber ?? 0,
    instruction,
    connectionStatus,
    isConnectionStale,
    isSpectator,
    spectatorCount: gameState.spectators.length,
    subPhaseDetail,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Event Log Transformer
// ═══════════════════════════════════════════════════════════════════════════

function formatPosition(pos?: Position): string {
  if (!pos) return '';
  if (typeof pos.z === 'number') {
    return `(${pos.x}, ${pos.y}, ${pos.z})`;
  }
  return `(${pos.x}, ${pos.y})`;
}

function describeHistoryEntry(entry: GameHistoryEntry): string {
  const { action } = entry;
  const moveLabel = `#${entry.moveNumber}`;
  const playerLabel = `P${action.player}`;

  switch (action.type) {
    case 'place_ring': {
      const count = action.placementCount ?? 1;
      return `${moveLabel} — ${playerLabel} placed ${count} ring${count === 1 ? '' : 's'} at ${formatPosition(action.to)}`;
    }
    case 'move_ring':
    case 'move_stack': {
      return `${moveLabel} — ${playerLabel} moved from ${formatPosition(action.from)} to ${formatPosition(action.to)}`;
    }
    case 'build_stack': {
      return `${moveLabel} — ${playerLabel} built stack at ${formatPosition(action.to)} (Δ=${action.buildAmount ?? 1})`;
    }
    case 'overtaking_capture': {
      const capturedCount = action.overtakenRings?.length ?? 0;
      const captureSuffix = capturedCount > 0 ? ` x${capturedCount}` : '';
      return `${moveLabel} — ${playerLabel} capture from ${formatPosition(action.from)} over ${formatPosition(action.captureTarget)} to ${formatPosition(action.to)}${captureSuffix}`;
    }
    case 'continue_capture_segment': {
      const capturedCount = action.overtakenRings?.length ?? 0;
      const captureSuffix = capturedCount > 0 ? ` x${capturedCount}` : '';
      return `${moveLabel} — ${playerLabel} continued capture over ${formatPosition(action.captureTarget)} to ${formatPosition(action.to)}${captureSuffix}`;
    }
    case 'process_line':
    case 'choose_line_reward': {
      const lineCount = action.formedLines?.length ?? 0;
      if (lineCount > 0) {
        return `${moveLabel} — ${playerLabel} processed ${lineCount} line${lineCount === 1 ? '' : 's'}`;
      }
      return `${moveLabel} — Line processing by ${playerLabel}`;
    }
    case 'process_territory_region':
    case 'eliminate_rings_from_stack': {
      const regionCount =
        action.claimedTerritory?.length ?? action.disconnectedRegions?.length ?? 0;
      const eliminatedTotal = (action.eliminatedRings ?? []).reduce(
        (sum, entry) => sum + entry.count,
        0
      );
      const parts: string[] = [];
      if (regionCount > 0) {
        parts.push(`${regionCount} region${regionCount === 1 ? '' : 's'}`);
      }
      if (eliminatedTotal > 0) {
        parts.push(`${eliminatedTotal} ring${eliminatedTotal === 1 ? '' : 's'} eliminated`);
      }
      const detail = parts.length > 0 ? ` (${parts.join(', ')})` : '';
      return `${moveLabel} — Territory / elimination processing by ${playerLabel}${detail}`;
    }
    case 'skip_placement': {
      return `${moveLabel} — ${playerLabel} skipped placement`;
    }
    default: {
      return `${moveLabel} — ${playerLabel} performed ${action.type}`;
    }
  }
}

function formatVictoryReason(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Ring Elimination';
    case 'territory_control':
      return 'Territory Control';
    case 'last_player_standing':
      return 'Last Player Standing';
    case 'timeout':
      return 'Timeout';
    case 'resignation':
      return 'Resignation';
    case 'abandonment':
      return 'Abandonment';
    case 'draw':
      return 'Draw';
    default:
      return reason.replace(/_/g, ' ');
  }
}

function describeVictory(victory?: GameResult | null): string | null {
  if (!victory) return null;

  const reasonLabel = formatVictoryReason(victory.reason);

  if (victory.winner === undefined) {
    if (victory.reason === 'draw') {
      return 'Game ended in a draw.';
    }
    return `Result: ${reasonLabel}`;
  }

  return `Player P${victory.winner} wins by ${reasonLabel}`;
}

export interface ToEventLogViewModelOptions {
  maxEntries?: number;
}

/**
 * Transform game history into EventLogViewModel
 */
export function toEventLogViewModel(
  history: GameHistoryEntry[],
  systemEvents: string[],
  victoryState: GameResult | null | undefined,
  options: ToEventLogViewModelOptions = {}
): EventLogViewModel {
  const { maxEntries = 40 } = options;

  const entries: EventLogItemViewModel[] = [];

  // Victory entry first (if any)
  const victoryMessage = describeVictory(victoryState);
  if (victoryMessage) {
    entries.push({
      key: 'victory',
      text: victoryMessage,
      type: 'victory',
    });
  }

  // Recent moves (most recent first)
  const recentMoves = (history || []).slice(-maxEntries).reverse();
  for (const entry of recentMoves) {
    entries.push({
      key: `move-${entry.moveNumber}`,
      text: describeHistoryEntry(entry),
      type: 'move',
      moveNumber: entry.moveNumber,
    });
  }

  // System events
  for (let i = 0; i < systemEvents.length; i++) {
    entries.push({
      key: `system-${i}`,
      text: systemEvents[i],
      type: 'system',
    });
  }

  return {
    entries,
    victoryMessage: victoryMessage ?? undefined,
    hasContent: entries.length > 0,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Board View Model Transformer
// ═══════════════════════════════════════════════════════════════════════════

export interface ToBoardViewModelOptions {
  selectedPosition?: Position;
  validTargets?: Position[];
}

/**
 * Transform BoardState into BoardViewModel
 */
export function toBoardViewModel(
  board: BoardState,
  options: ToBoardViewModelOptions = {}
): BoardViewModel {
  const { selectedPosition, validTargets = [] } = options;

  const cells: CellViewModel[] = [];

  // Generate all positions based on board type
  if (board.type === 'square8' || board.type === 'square19') {
    const size = board.size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pos: Position = { x, y };
        const cell = createCellViewModel(pos, board, selectedPosition, validTargets);
        cells.push(cell);
      }
    }
  } else if (board.type === 'hexagonal') {
    // Hex board: iterate over all positions in stacks, markers, collapsedSpaces
    const allKeys = new Set<string>();
    for (const key of board.stacks.keys()) allKeys.add(key);
    for (const key of board.markers.keys()) allKeys.add(key);
    for (const key of board.collapsedSpaces.keys()) allKeys.add(key);

    for (const key of allKeys) {
      const parts = key.split(',').map(Number);
      const pos: Position =
        parts.length === 3
          ? { x: parts[0], y: parts[1], z: parts[2] }
          : { x: parts[0], y: parts[1] };
      const cell = createCellViewModel(pos, board, selectedPosition, validTargets);
      cells.push(cell);
    }
  }

  // For square boards, organize into rows
  let rows: CellViewModel[][] | undefined;
  if (board.type === 'square8' || board.type === 'square19') {
    rows = [];
    const size = board.size;
    for (let y = 0; y < size; y++) {
      const row: CellViewModel[] = [];
      for (let x = 0; x < size; x++) {
        const cell = cells.find((c) => c.position.x === x && c.position.y === y);
        if (cell) row.push(cell);
      }
      rows.push(row);
    }
  }

  return {
    boardType: board.type,
    size: board.size,
    cells,
    rows,
  };
}

function createCellViewModel(
  pos: Position,
  board: BoardState,
  selectedPosition?: Position,
  validTargets: Position[] = []
): CellViewModel {
  const key = positionToString(pos);
  const stack = board.stacks.get(key);
  const marker = board.markers.get(key);
  const collapsedOwner = board.collapsedSpaces.get(key);

  const isSelected = selectedPosition
    ? pos.x === selectedPosition.x &&
      pos.y === selectedPosition.y &&
      (pos.z || 0) === (selectedPosition.z || 0)
    : false;

  const isValidTarget = validTargets.some(
    (t) => t.x === pos.x && t.y === pos.y && (t.z || 0) === (pos.z || 0)
  );

  const isDarkSquare = (pos.x + pos.y) % 2 === 0;

  let stackViewModel: StackViewModel | undefined;
  if (stack) {
    stackViewModel = toStackViewModel(stack, pos);
  }

  let markerViewModel: MarkerViewModel | undefined;
  if (marker && marker.type === 'regular') {
    const colors = getPlayerColors(marker.player);
    markerViewModel = {
      position: pos,
      positionKey: key,
      playerNumber: marker.player,
      colorClass: colors.marker,
    };
  }

  let collapsedSpaceViewModel: CollapsedSpaceViewModel | undefined;
  if (collapsedOwner !== undefined) {
    const colors = getPlayerColors(collapsedOwner);
    collapsedSpaceViewModel = {
      position: pos,
      positionKey: key,
      ownerPlayerNumber: collapsedOwner,
      territoryColorClass: colors.territory,
      borderClass: colors.marker,
    };
  }

  return {
    position: pos,
    positionKey: key,
    stack: stackViewModel,
    marker: markerViewModel,
    collapsedSpace: collapsedSpaceViewModel,
    isSelected,
    isValidTarget,
    isDarkSquare,
  };
}

function toStackViewModel(stack: RingStack, pos: Position): StackViewModel {
  const key = positionToString(pos);
  const rings: RingViewModel[] = [];

  const topIndex = 0;
  const capEndIndex = Math.min(stack.capHeight - 1, stack.rings.length - 1);

  for (let i = 0; i < stack.rings.length; i++) {
    const playerNumber = stack.rings[i];
    const colors = getPlayerColors(playerNumber);
    rings.push({
      playerNumber,
      colorClass: colors.ring,
      borderClass: colors.ringBorder,
      isTop: i === topIndex,
      isInCap: i <= capEndIndex,
    });
  }

  return {
    position: pos,
    positionKey: key,
    rings,
    stackHeight: stack.stackHeight,
    capHeight: stack.capHeight,
    controllingPlayer: stack.controllingPlayer,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Victory Modal Transformer
// ═══════════════════════════════════════════════════════════════════════════

function countRingsOnBoard(playerNumber: number, gameState: GameState): number {
  let count = 0;
  for (const stack of gameState.board.stacks.values()) {
    count += stack.rings.filter((r) => r === playerNumber).length;
  }
  return count;
}

function countPlayerMoves(playerNumber: number, gameState: GameState): number {
  if (gameState.history && gameState.history.length > 0) {
    return gameState.history.filter((entry) => entry.actor === playerNumber).length;
  }
  return gameState.moveHistory.filter((move) => move.player === playerNumber).length;
}

export interface ToVictoryViewModelOptions {
  currentUserId?: string;
  isDismissed?: boolean;
}

/**
 * Transform GameResult into VictoryViewModel
 * Returns null if no victory state or if dismissed
 */
export function toVictoryViewModel(
  gameResult: GameResult | null | undefined,
  players: Player[],
  gameState: GameState | undefined,
  options: ToVictoryViewModelOptions = {}
): VictoryViewModel | null {
  const { currentUserId, isDismissed = false } = options;

  if (!gameResult || isDismissed) {
    return null;
  }

  const winner =
    gameResult.winner !== undefined
      ? players.find((p) => p.playerNumber === gameResult.winner)
      : undefined;

  const winnerViewModel =
    winner && gameState ? toPlayerViewModel(winner, gameState, currentUserId) : undefined;

  const userWon = !!(currentUserId && winner && winner.id === currentUserId);
  const userLost = !!(
    currentUserId &&
    gameResult.winner !== undefined &&
    winner &&
    winner.id !== currentUserId
  );
  const isDraw = gameResult.reason === 'draw';

  // Generate title and description
  const { title, description, titleColorClass } = getVictoryMessage(
    gameResult,
    winner,
    userWon,
    userLost,
    isDraw
  );

  // Build final stats
  const finalStats: PlayerFinalStatsViewModel[] = players.map((player) => {
    const playerVM = gameState
      ? toPlayerViewModel(player, gameState, currentUserId)
      : {
          id: player.id,
          playerNumber: player.playerNumber,
          username: player.username || `Player ${player.playerNumber}`,
          isCurrentPlayer: false,
          isUserPlayer: player.id === currentUserId,
          colorClass: getPlayerColors(player.playerNumber).card,
          ringStats: { inHand: 0, onBoard: 0, eliminated: 0, total: 0 },
          territorySpaces: 0,
          aiInfo: toAIInfoViewModel(player),
        };

    const ringsOnBoard = gameState
      ? countRingsOnBoard(player.playerNumber, gameState)
      : (gameResult.finalScore.ringsRemaining[player.playerNumber] ?? 0);

    const totalMoves = gameState ? countPlayerMoves(player.playerNumber, gameState) : 0;

    return {
      player: playerVM,
      ringsOnBoard,
      ringsEliminated: gameResult.finalScore.ringsEliminated[player.playerNumber] ?? 0,
      territorySpaces: gameResult.finalScore.territorySpaces[player.playerNumber] ?? 0,
      totalMoves,
      isWinner: winner?.playerNumber === player.playerNumber,
    };
  });

  // Sort by winner first, then by rings eliminated
  finalStats.sort((a, b) => {
    if (a.isWinner) return -1;
    if (b.isWinner) return 1;
    return b.ringsEliminated - a.ringsEliminated;
  });

  const gameSummary = {
    boardType: gameState?.boardType ?? 'square8',
    totalTurns: gameState?.history?.length || gameState?.moveHistory?.length || 0,
    playerCount: players.length,
    isRated: gameState?.isRated ?? false,
  };

  return {
    isVisible: true,
    title,
    description,
    titleColorClass,
    winner: winnerViewModel,
    finalStats,
    gameSummary,
    userWon,
    userLost,
    isDraw,
  };
}

function getVictoryMessage(
  gameResult: GameResult,
  winner: Player | undefined,
  userWon: boolean,
  userLost: boolean,
  isDraw: boolean
): { title: string; description: string; titleColorClass: string } {
  const titleColorClass = userWon ? 'text-green-400' : userLost ? 'text-red-400' : 'text-slate-100';

  if (isDraw) {
    return {
      title: '🤝 Draw!',
      description: 'The game ended in a stalemate with equal positions',
      titleColorClass,
    };
  }

  const winnerName = winner?.username || `Player ${winner?.playerNumber || '?'}`;

  switch (gameResult.reason) {
    case 'ring_elimination':
      return {
        title: `🏆 ${winnerName} Wins!`,
        description: 'Victory by eliminating all opponent rings',
        titleColorClass,
      };
    case 'territory_control':
      return {
        title: `🏰 ${winnerName} Wins!`,
        description: 'Victory by controlling majority of the board',
        titleColorClass,
      };
    case 'last_player_standing': {
      const subject = userWon ? 'You' : winnerName;
      const verb = userWon ? 'were' : 'was';
      return {
        title: '👑 Last Player Standing',
        description: `${subject} ${verb} the only player able to make real moves (placements, movements, or captures) for a full round of turns.`,
        titleColorClass,
      };
    }
    case 'timeout':
      return {
        title: `⏰ ${winnerName} Wins!`,
        description: 'Victory by opponent timeout',
        titleColorClass,
      };
    case 'resignation':
      return {
        title: `${winnerName} Wins!`,
        description: 'Victory by opponent resignation',
        titleColorClass,
      };
    case 'abandonment':
      return {
        title: 'Game Abandoned',
        description: 'The game was left in an unresolved state',
        titleColorClass,
      };
    default:
      return {
        title: `${winnerName} Wins!`,
        description: 'Game over',
        titleColorClass,
      };
  }
}
