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
  GameHistoryEntry,
  Move,
  BoardType,
  PlayerChoice,
  PlayerChoiceType,
} from '../../shared/types/game';
import type { FSMOrchestrationResult } from '../../shared/engine/fsm';
import { positionToString, positionsEqual, BOARD_CONFIGS } from '../../shared/types/game';
import type { ConnectionStatus } from '../domain/GameAPI';
import { getChoiceViewModel, getChoiceViewModelForType } from './choiceViewModels';
import type { ChoiceKind } from './choiceViewModels';
import { getWeirdStateBanner, type WeirdStateBanner } from '../utils/gameStateWeirdness';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import { DEFAULT_PLAYER_THEME, getPlayerTheme, type ColorVisionMode } from '../utils/playerTheme';

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
  /**
   * Short, action-oriented hint for the active player.
   * E.g., "Click an empty cell to place" or "Select your stack, then destination".
   */
  actionHint: string;
  /**
   * Spectator-oriented description of what's happening.
   * E.g., "Watching ring placement" or "Player is selecting a move".
   */
  spectatorHint: string;
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
 * UX-facing banner metadata for weird rules states (ANM / forced
 * elimination / structural stalemate). Derived from getWeirdStateBanner(...)
 * and used purely for HUD copy.
 */
export interface HUDWeirdStateViewModel {
  type: WeirdStateBanner['type'];
  title: string;
  body: string;
  /** Visual tone hint for the banner styling. */
  tone: 'info' | 'warning' | 'critical';
}

/**
 * Complete HUD view model
 */
export interface HUDDecisionPhaseViewModel {
  /** Whether a decision is currently active for any player. */
  isActive: boolean;
  /** Acting player number and display name. */
  actingPlayerNumber: number;
  actingPlayerName: string;
  /** True when the local user is the acting player. */
  isLocalActor: boolean;
  /**
   * Primary status line for the HUD, e.g. "Your decision: Choose Line Reward" or
   * "Waiting for Alice to choose a line reward option".
   */
  label: string;
  /** Optional longer description derived from the underlying ChoiceViewModel. */
  description?: string | undefined;
  /** Short label suitable for compact chips/badges. */
  shortLabel: string;
  /**
   * Remaining time in milliseconds according to the client-side clock, or null
   * when no explicit timeout exists or countdown UI should be suppressed.
   */
  timeRemainingMs: number | null;
  /** Whether the countdown should be rendered in the HUD. */
  showCountdown: boolean;
  /** Optional soft warning threshold for low-time styling. */
  warningThresholdMs?: number | undefined;
  /**
   * True when the underlying decision countdown has been shortened/capped by
   * authoritative server timeout metadata. Used purely for UI emphasis.
   */
  isServerCapped?: boolean | undefined;
  /** Spectator-oriented status text derived from ChoiceViewModel.copy.spectatorLabel. */
  spectatorLabel: string;
  /**
   * Optional compact status chip used by HUDs to surface high-attention prompts
   * for specific decision types (e.g., ring elimination).
   */
  statusChip?:
    | {
        text: string;
        /** Visual tone hint for the chip styling. */
        tone: 'info' | 'attention';
      }
    | undefined;
  /**
   * Optional flag indicating that a skip action is available in this decision
   * phase (for example, skip_territory_processing when disconnected regions
   * exist). Used by HUDs to render an explicit "Skip" control instead of
   * relying solely on board highlights.
   */
  canSkip?: boolean | undefined;
}
export interface HUDViewModel {
  phase: PhaseViewModel;
  players: PlayerViewModel[];
  turnNumber: number;
  moveNumber: number;
  /** Optional short summary when the pie rule has been used recently. */
  pieRuleSummary?: string | undefined;
  instruction?: string | undefined;
  connectionStatus: ConnectionStatus;
  isConnectionStale: boolean;
  isSpectator: boolean;
  spectatorCount: number;
  /**
   * Sub-phase details (e.g., "Processing 3 lines")
   */
  subPhaseDetail?: string | undefined;
  /**
   * When present, describes the currently-active decision phase (line reward,
   * territory region order, ring elimination, capture direction, etc.). This is
   * derived from PlayerChoice + choiceViewModels and is intended to be the
   * single source of truth for HUD decision copy.
   */
  decisionPhase?: HUDDecisionPhaseViewModel | undefined;
  /**
   * Optional high-level banner describing unusual rules states such as
   * active-no-moves or forced elimination. When present, GameHUD renders
   * a prominent explanation panel above the phase indicator.
   */
  weirdState?: HUDWeirdStateViewModel | undefined;
  /**
   * LPS (Last-Player-Standing) tracking state for UI display.
   * Shows round counter and progress toward LPS victory.
   * Per RR-CANON-R172, LPS requires 3 consecutive rounds where only 1 player has real actions.
   */
  lpsTracking?:
    | {
        roundIndex: number;
        consecutiveExclusiveRounds: number;
        consecutiveExclusivePlayer: number | null;
      }
    | undefined;
  /**
   * Victory progress tracking for ring-elimination and territory-control.
   * Per RR-CANON-R061: victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
   * Per RR-CANON-R062: territoryThreshold = floor(totalSpaces/2)+1
   */
  victoryProgress?: {
    ringElimination: {
      threshold: number;
      leader: { playerNumber: number; eliminated: number; percentage: number } | null;
    };
    territory: {
      threshold: number;
      leader: { playerNumber: number; spaces: number; percentage: number } | null;
    };
  };
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
  victoryMessage?: string | undefined;
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
 * Decision-phase highlight semantics for the board. This is intentionally
 * lightweight and board-centric: it describes which cells should be
 * emphasised while a PlayerChoice is pending, without embedding any
 * transport- or engine-specific types.
 */
export type DecisionHighlightIntensity = 'primary' | 'secondary';

export interface DecisionHighlight {
  /** Position key as produced by positionToString. */
  positionKey: string;
  /** Visual intensity category for UI styling. */
  intensity: DecisionHighlightIntensity;
  /**
   * Optional grouping identifier used by some decision kinds (e.g., territory
   * region order) to associate multiple highlighted cells with the same
   * semantic region. UI layers may ignore this when they do not need
   * per-region styling.
   */
  groupId?: string;
}

export interface BoardDecisionHighlightsViewModel {
  /** High-level semantic grouping of the underlying choice. */
  choiceKind: ChoiceKind;
  /** Flat collection of highlighted board cells. */
  highlights: DecisionHighlight[];
  /**
   * Optional territory-region metadata used by the board to apply
   * per-region styling. Only populated for territory_region_order
   * decisions and kept intentionally lightweight so view logic
   * remains UI-framework agnostic.
   */
  territoryRegions?: {
    /** Stable region identifiers in the order presented to the player. */
    regionIdsInDisplayOrder: string[];
  };
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
  stack?: StackViewModel | undefined;
  marker?: MarkerViewModel | undefined;
  collapsedSpace?: CollapsedSpaceViewModel | undefined;
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
  rows?: CellViewModel[][] | undefined;
  /**
   * Optional decision-phase highlights derived from a pending PlayerChoice.
   * When present, BoardView may render lightweight overlays to guide the
   * acting player, non-acting players, and spectators toward the relevant
   * geometry (lines, regions, stacks, capture directions, etc.).
   */
  decisionHighlights?: BoardDecisionHighlightsViewModel | undefined;
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
  winner?: PlayerViewModel | undefined;
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
  1: getPlayerTheme(1, 'normal'),
  2: getPlayerTheme(2, 'normal'),
  3: getPlayerTheme(3, 'normal'),
  4: getPlayerTheme(4, 'normal'),
} as const;

export function getPlayerColors(
  playerNumber?: number,
  colorVisionMode: ColorVisionMode = 'normal'
) {
  if (!playerNumber) return DEFAULT_PLAYER_THEME;
  return getPlayerTheme(playerNumber, colorVisionMode);
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase Helpers
// ═══════════════════════════════════════════════════════════════════════════

export const PHASE_INFO: Record<GamePhase, Omit<PhaseViewModel, 'phaseKey'>> = {
  ring_placement: {
    label: 'Place Rings',
    description: 'Add rings to the board to build your stacks',
    colorClass: 'bg-blue-500',
    icon: '🎯',
    actionHint: 'Tap an empty space or one of your stacks to place a ring',
    spectatorHint: 'Placing rings',
  },
  movement: {
    label: 'Your Move',
    description: 'Move one of your stacks or jump to capture',
    colorClass: 'bg-green-500',
    icon: '⚡',
    actionHint: 'Tap your stack, then tap where to move it',
    spectatorHint: 'Choosing a move',
  },
  capture: {
    label: 'Capture!',
    description: 'Jump over an opponent to capture their rings',
    colorClass: 'bg-orange-500',
    icon: '⚔️',
    actionHint: 'Tap your stack, then tap beyond an opponent to jump over them',
    spectatorHint: 'Capturing',
  },
  chain_capture: {
    label: 'Keep Capturing!',
    description: 'You can make another capture—keep jumping!',
    colorClass: 'bg-orange-500',
    icon: '🔗',
    actionHint: 'Tap the next opponent to jump over, or skip if none available',
    spectatorHint: 'Chain capturing',
  },
  line_processing: {
    label: 'Line Scored!',
    description: 'You made a line of 5+ markers—choose your reward',
    colorClass: 'bg-purple-500',
    icon: '📏',
    actionHint: 'Pick your line reward option',
    spectatorHint: 'Choosing line reward',
  },
  territory_processing: {
    label: 'Territory!',
    description: 'You isolated a region—claim it as your territory',
    colorClass: 'bg-pink-500',
    icon: '🏰',
    actionHint: 'Choose which region to claim',
    spectatorHint: 'Claiming territory',
  },
  forced_elimination: {
    label: 'Blocked!',
    description: 'No moves available—you must remove a ring from one of your stacks',
    colorClass: 'bg-red-600',
    icon: '💥',
    actionHint: 'Choose which stack to remove a ring from',
    spectatorHint: 'Forced elimination',
  },
  game_over: {
    label: 'Game Over',
    description: 'The game has ended',
    colorClass: 'bg-slate-600',
    icon: '🏁',
    actionHint: '',
    spectatorHint: 'Game finished',
  },
};

function toPhaseViewModel(phase: GamePhase): PhaseViewModel {
  const info = PHASE_INFO[phase] || {
    label: 'Unknown Phase',
    description: '',
    colorClass: 'bg-gray-400',
    icon: '❓',
    actionHint: 'Make your move',
    spectatorHint: 'Waiting for player action',
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

function calculateRingStats(player: Player, gameState: GameState): PlayerRingStatsViewModel {
  const total = BOARD_CONFIGS[gameState.boardType]?.ringsPerPlayer ?? 18;

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
  currentUserId?: string,
  colorVisionMode: ColorVisionMode = 'normal'
): PlayerViewModel {
  const colors = getPlayerColors(player.playerNumber, colorVisionMode);
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
  instruction?: string | undefined;
  connectionStatus: ConnectionStatus;
  lastHeartbeatAt: number | null;
  isSpectator: boolean;
  currentUserId?: string | undefined;
  colorVisionMode?: ColorVisionMode | undefined;
  /** Optional pending choice used to derive decision-phase HUD messaging. */
  pendingChoice?: PlayerChoice | null | undefined;
  /** Optional absolute deadline timestamp in ms for the pending choice. */
  choiceDeadline?: number | null | undefined;
  /**
   * Optional precomputed client-side countdown for the pending choice. When
   * provided (e.g. from a dedicated timer hook), this will be preferred over
   * deriving the remaining time from choiceDeadline.
   */
  choiceTimeRemainingMs?: number | null | undefined;
  /**
   * Optional flag indicating that the effective decision countdown has been
   * capped/shortened by authoritative server timeout metadata. This allows the
   * HUD to surface a subtle "server deadline" indicator without needing to
   * understand reconciliation semantics itself.
   */
  decisionIsServerCapped?: boolean | undefined;
  /**
   * Optional terminal victory state for the current game, when known. Used
   * purely for UX surfaces such as structural-stalemate banners; does not
   * affect rules semantics or engine behaviour.
   */
  victoryState?: GameResult | null | undefined;
  /**
   * Optional canonical GameEndExplanation derived from the shared engine view.
   * Used to surface concept-aligned banners (e.g. "Structural Stalemate") in
   * the HUD when the game has ended.
   */
  gameEndExplanation?: GameEndExplanation | null | undefined;
  /**
   * Optional LPS tracking state from sandbox engine. Per RR-CANON-R172, LPS
   * requires 3 consecutive rounds where only 1 player has real actions.
   */
  lpsTracking?:
    | {
        roundIndex: number;
        consecutiveExclusiveRounds: number;
        consecutiveExclusivePlayer: number | null;
      }
    | undefined;
}

/**
 * Transform GameState into HUDViewModel
 */
export function toHUDViewModel(gameState: GameState, options: ToHUDViewModelOptions): HUDViewModel {
  const {
    instruction,
    connectionStatus,
    lastHeartbeatAt,
    isSpectator,
    currentUserId,
    colorVisionMode = 'normal',
    pendingChoice,
    choiceDeadline,
    choiceTimeRemainingMs,
    decisionIsServerCapped,
    victoryState,
    gameEndExplanation,
    lpsTracking,
  } = options;

  const HEARTBEAT_STALE_THRESHOLD_MS = 8000;
  const heartbeatAge = lastHeartbeatAt ? Date.now() - lastHeartbeatAt : null;
  const isConnectionStale =
    heartbeatAge !== null &&
    heartbeatAge > HEARTBEAT_STALE_THRESHOLD_MS &&
    connectionStatus === 'connected';

  let phase = toPhaseViewModel(gameState.currentPhase);
  const players = gameState.players.map((p) =>
    toPlayerViewModel(p, gameState, currentUserId, colorVisionMode)
  );

  // Precompute choice view model (if any) so we can reuse it for both HUD phase
  // styling and decision-phase metadata without duplicating mapping logic.
  const pendingChoiceVm = pendingChoice ? getChoiceViewModel(pendingChoice) : null;

  const turnNumber = gameState.moveHistory.length;
  const moveNumber =
    gameState.history.length > 0 ? gameState.history[gameState.history.length - 1]?.moveNumber : 0;

  // Surface a short "pie rule used" summary for a few moves after a
  // swap_sides action so spectators understand any seat/colour changes.
  let pieRuleSummary: string | undefined;
  if (gameState.history && gameState.history.length > 0) {
    const lastSwapEntry = [...gameState.history]
      .slice()
      .reverse()
      .find((entry) => entry.action.type === 'swap_sides');
    if (lastSwapEntry) {
      const movesSinceSwap = gameState.history.length - lastSwapEntry.moveNumber;
      if (movesSinceSwap <= 3) {
        const actor = lastSwapEntry.action.player;
        const otherSeat = actor === 1 ? 2 : 1;
        pieRuleSummary = `P${actor} swapped colours with P${otherSeat}`;
      }
    }
  }

  // Sub-phase detail
  let subPhaseDetail: string | undefined;
  if (gameState.currentPhase === 'line_processing') {
    const lineCount = gameState.board.formedLines?.length ?? 0;
    if (lineCount > 0) {
      subPhaseDetail = `Processing ${lineCount} line${lineCount !== 1 ? 's' : ''}`;
    }
  } else if (gameState.currentPhase === 'territory_processing') {
    subPhaseDetail =
      'Processing disconnected regions; you must eliminate one outside ring per region.';
  }

  // When a line-related decision is active during line/territory processing,
  // present a celebratory "Line Formation" phase label and styling in the HUD
  // while keeping the underlying phaseKey aligned with engine semantics.
  if (
    pendingChoiceVm &&
    gameState.gameStatus === 'active' &&
    (gameState.currentPhase === 'line_processing' ||
      gameState.currentPhase === 'territory_processing')
  ) {
    const isLineDecisionKind =
      pendingChoiceVm.kind === 'line_order' ||
      pendingChoiceVm.kind === 'line_reward' ||
      pendingChoiceVm.kind === 'ring_elimination';

    if (isLineDecisionKind) {
      phase = {
        ...phase,
        label: 'Line Formation',
        description: 'Resolve completed lines and select any elimination rewards.',
        colorClass: 'bg-amber-500',
        icon: '✨',
      };
    }
  }

  // Decision-phase detail derived from pending PlayerChoice + choiceViewModels.
  let decisionPhase: HUDDecisionPhaseViewModel | undefined;
  if (pendingChoice && pendingChoiceVm && gameState.gameStatus === 'active') {
    const actingPlayer = gameState.players.find(
      (p) => p.playerNumber === pendingChoice.playerNumber
    );
    const actingPlayerName = actingPlayer?.username || `Player ${pendingChoice.playerNumber}`;
    const isLocalActor = !!(currentUserId && actingPlayer && actingPlayer.id === currentUserId);

    const vm = pendingChoiceVm;

    // Prefer a precomputed countdown when provided (e.g. from a dedicated
    // DecisionUI hook); otherwise derive remaining time from the deadline.
    let timeRemainingMs: number | null = null;
    if (typeof choiceTimeRemainingMs === 'number') {
      timeRemainingMs = choiceTimeRemainingMs >= 0 ? choiceTimeRemainingMs : 0;
    } else if (choiceDeadline) {
      const remaining = choiceDeadline - Date.now();
      timeRemainingMs = remaining > 0 ? remaining : 0;
    }

    // Compose a primary status label. For the acting player we emphasise their
    // own decision; for everyone else we use the spectator-oriented copy.
    const label = isLocalActor
      ? `Your decision: ${vm.copy.title}`
      : vm.copy.spectatorLabel({ actingPlayerName });

    const spectatorLabel = vm.copy.spectatorLabel({ actingPlayerName });

    // Optional compact status chip for specific decision kinds.
    let statusChip: HUDDecisionPhaseViewModel['statusChip'];
    if (vm.kind === 'ring_elimination') {
      statusChip = {
        text: 'Select stack cap to eliminate',
        tone: 'attention',
      };
    } else if (vm.kind === 'territory_region_order') {
      statusChip = {
        text: 'Territory claimed – choose region to process or skip',
        tone: 'attention',
      };
    }

    decisionPhase = {
      isActive: true,
      actingPlayerNumber: pendingChoice.playerNumber,
      actingPlayerName,
      isLocalActor,
      label,
      description: vm.copy.description,
      shortLabel: vm.copy.shortLabel,
      timeRemainingMs: vm.timeout.showCountdown ? timeRemainingMs : null,
      showCountdown: vm.timeout.showCountdown,
      warningThresholdMs: vm.timeout.warningThresholdMs,
      isServerCapped: vm.timeout.showCountdown ? decisionIsServerCapped : undefined,
      spectatorLabel,
      statusChip,
      // Expose a generic skip hint when the underlying pending choice is a
      // region_order that includes an explicit "skip" option. In both backend
      // and sandbox flows this is represented by a RegionOrderChoice option
      // whose regionId is 'skip' or whose size is <= 0; the canonical
      // skip_territory_processing Move is selected via moveId.
      canSkip:
        vm.kind === 'territory_region_order' &&
        pendingChoice.type === 'region_order' &&
        pendingChoice.options.some(
          (opt) =>
            !!opt && (opt.regionId === 'skip' || (typeof opt.size === 'number' && opt.size <= 0))
        ),
    };
  }

  // Weird-state banner: interpret ANM / forced-elimination / structural-stalemate
  // states for HUD copy. Victory state is passed purely for UX; rules semantics
  // remain owned by the shared engine.
  const weird = getWeirdStateBanner(gameState, { victoryState: victoryState ?? null });
  let weirdState: HUDWeirdStateViewModel | undefined;

  const resolvePlayerLabel = (
    playerNumber: number
  ): {
    label: string;
    isUser: boolean;
  } => {
    const playerVm = players.find((p) => p.playerNumber === playerNumber);
    if (!playerVm) {
      return { label: `P${playerNumber}`, isUser: false };
    }
    return {
      label: playerVm.isUserPlayer ? 'You' : playerVm.username,
      isUser: playerVm.isUserPlayer,
    };
  };

  // Prefer canonical GameEndExplanation when provided so HUD copy stays aligned
  // across backend/sandbox hosts and VictoryModal.
  if (gameEndExplanation) {
    if (gameEndExplanation.outcomeType === 'structural_stalemate') {
      weirdState = {
        type: 'structural-stalemate',
        title: 'Game Ended: Stalemate',
        body: 'Nobody can make any more moves. The winner is decided by who has more territory and eliminated rings.',
        tone: 'critical',
      };
    } else if (gameEndExplanation.outcomeType === 'last_player_standing') {
      const winnerVm = players.find((p) => p.id === gameEndExplanation.winnerPlayerId);
      const winnerLabel = winnerVm
        ? winnerVm.isUserPlayer
          ? 'You'
          : winnerVm.username
        : 'A player';
      const subject = winnerLabel === 'You' ? 'You were' : `${winnerLabel} was`;
      weirdState = {
        type: 'last-player-standing',
        title: 'Last Player Standing Wins!',
        body: `${subject} the only player able to move for three rounds in a row. When one player dominates the board this completely, they win!`,
        tone: 'warning',
      };
    }
  }

  // Fall back to legacy weird-state detection if no explanation-driven banner was set.
  if (!weirdState) {
    switch (weird.type) {
      case 'active-no-moves-movement': {
        const { label, isUser } = resolvePlayerLabel(weird.playerNumber);
        weirdState = {
          type: weird.type,
          title: isUser ? 'No Moves Available!' : `${label} is blocked`,
          body: isUser
            ? 'You have stacks on the board but nowhere to move them. Rings will be removed from your stacks until you can move again.'
            : `${label} has stacks but nowhere to move. Rings will be removed until they can move.`,
          tone: 'warning',
        };
        break;
      }
      case 'active-no-moves-line': {
        const { label, isUser } = resolvePlayerLabel(weird.playerNumber);
        weirdState = {
          type: weird.type,
          title: isUser ? 'Line phase skipped' : `${label}'s line phase skipped`,
          body: 'No line actions needed right now. Moving to the next phase automatically.',
          tone: 'info',
        };
        break;
      }
      case 'active-no-moves-territory': {
        const { label, isUser } = resolvePlayerLabel(weird.playerNumber);
        weirdState = {
          type: weird.type,
          title: isUser ? 'Territory phase skipped' : `${label}'s territory phase skipped`,
          body: 'No territory actions needed right now. Moving to the next phase automatically.',
          tone: 'info',
        };
        break;
      }
      case 'forced-elimination': {
        const { label, isUser } = resolvePlayerLabel(weird.playerNumber);
        weirdState = {
          type: weird.type,
          title: isUser ? 'Your stacks are being reduced' : `${label}'s stacks are shrinking`,
          body: isUser
            ? 'You have stacks but no moves. Rings are being removed—each one counts toward Ring Elimination victory!'
            : `${label} has stacks but no moves. Rings are being removed—each counts toward Ring Elimination.`,
          tone: 'warning',
        };
        break;
      }
      case 'last-player-standing': {
        const winnerNumber = weird.winner;
        const winnerLabel =
          typeof winnerNumber === 'number' ? resolvePlayerLabel(winnerNumber).label : 'A player';
        const subject = winnerLabel === 'You' ? 'You were' : `${winnerLabel} was`;
        weirdState = {
          type: weird.type,
          title: 'Last Player Standing Wins!',
          body: `${subject} the only player able to move for three rounds in a row. When one player dominates the board this completely, they win!`,
          tone: 'warning',
        };
        break;
      }
      case 'structural-stalemate': {
        weirdState = {
          type: weird.type,
          title: 'Game Ended: Stalemate',
          body: 'Nobody can make any more moves. The winner is decided by who has more territory and eliminated rings.',
          tone: 'critical',
        };
        break;
      }
      default:
        break;
    }
  }

  // Compute victory progress per RR-CANON-R061 (ring elimination) and RR-CANON-R062-v2 (territory)
  const ringThreshold = gameState.victoryThreshold;

  // Territory victory: dual-condition rule (RR-CANON-R062-v2)
  // Use new field if available, fall back to computation
  const boardConfig = BOARD_CONFIGS[gameState.boardType];
  const totalSpaces = boardConfig?.totalSpaces ?? 64;
  const territoryMinimum =
    gameState.territoryVictoryMinimum ?? Math.floor(totalSpaces / gameState.players.length) + 1;

  // Find leading player for ring elimination
  const ringLeader = [...gameState.players]
    .filter((p) => p.eliminatedRings > 0)
    .sort((a, b) => b.eliminatedRings - a.eliminatedRings)[0];

  // Find leading player for territory
  const territoryLeader = [...gameState.players]
    .filter((p) => p.territorySpaces > 0)
    .sort((a, b) => b.territorySpaces - a.territorySpaces)[0];

  // Calculate opponent territory for dominance check
  const opponentTerritorySum = territoryLeader
    ? gameState.players
        .filter((p) => p.playerNumber !== territoryLeader.playerNumber)
        .reduce((sum, p) => sum + p.territorySpaces, 0)
    : 0;

  // Territory progress: worst of the two conditions determines overall progress
  const territoryProgressInfo = territoryLeader
    ? (() => {
        const minimumProgress = (territoryLeader.territorySpaces / territoryMinimum) * 100;
        // For dominance: need > opponents, so target is (opponentSum + 1)
        const dominanceTarget = opponentTerritorySum + 1;
        const dominanceProgress =
          dominanceTarget > 0 ? (territoryLeader.territorySpaces / dominanceTarget) * 100 : 100;
        // Overall progress is the worse of the two conditions
        const overallPercentage = Math.round(Math.min(minimumProgress, dominanceProgress));
        return {
          playerNumber: territoryLeader.playerNumber,
          spaces: territoryLeader.territorySpaces,
          percentage: Math.min(overallPercentage, 100),
          meetsMinimum: territoryLeader.territorySpaces >= territoryMinimum,
          dominatesOpponents: territoryLeader.territorySpaces > opponentTerritorySum,
        };
      })()
    : null;

  const victoryProgress = {
    ringElimination: {
      threshold: ringThreshold,
      leader: ringLeader
        ? {
            playerNumber: ringLeader.playerNumber,
            eliminated: ringLeader.eliminatedRings,
            percentage: Math.round((ringLeader.eliminatedRings / ringThreshold) * 100),
          }
        : null,
    },
    territory: {
      threshold: territoryMinimum, // Use new minimum, not legacy threshold
      leader: territoryProgressInfo,
    },
  };

  return {
    phase,
    players,
    turnNumber,
    moveNumber: moveNumber ?? 0,
    pieRuleSummary,
    instruction,
    connectionStatus,
    isConnectionStale,
    isSpectator,
    spectatorCount: gameState.spectators.length,
    subPhaseDetail,
    decisionPhase,
    weirdState,
    lpsTracking,
    victoryProgress,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Board Decision Highlights Adapter
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Derive a lightweight board-centric highlight model for a pending
 * PlayerChoice. This keeps all semantics in terms of GameState +
 * PlayerChoice while remaining UI-framework agnostic.
 */
export function deriveBoardDecisionHighlights(
  gameState: GameState,
  pendingChoice: PlayerChoice | null | undefined
): BoardDecisionHighlightsViewModel | undefined {
  if (!pendingChoice) return undefined;

  const vm = getChoiceViewModel(pendingChoice);
  const highlights: DecisionHighlight[] = [];

  const pushPosition = (
    position: Position | undefined,
    intensity: DecisionHighlightIntensity,
    groupId?: string
  ) => {
    if (!position) return;
    const positionKey = positionToString(position);
    highlights.push(groupId ? { positionKey, intensity, groupId } : { positionKey, intensity });
  };

  switch (pendingChoice.type) {
    case 'line_order': {
      // Highlight markers for all candidate lines. This makes the decision
      // surface visible to both the acting player and spectators.
      for (const option of pendingChoice.options) {
        for (const pos of option.markerPositions) {
          pushPosition(pos, 'primary');
        }
      }
      break;
    }
    case 'line_reward_option': {
      // Reward choices operate over the currently-detected lines for the
      // acting player. We highlight all formedLines owned by that player so
      // the geometry remains visible while the reward UI is active.
      const linesForPlayer = (gameState.board.formedLines || []).filter(
        (line) => line.player === pendingChoice.playerNumber
      );
      for (const line of linesForPlayer) {
        for (const pos of line.positions) {
          pushPosition(pos, 'primary');
        }
      }
      break;
    }
    case 'ring_elimination': {
      // Each option exposes a concrete stackPosition; highlight all such
      // stacks as primary candidates.
      for (const option of pendingChoice.options) {
        pushPosition(option.stackPosition, 'primary');
      }
      break;
    }
    case 'region_order': {
      // Territory decisions expose a representativePosition for each region.
      // Prefer to highlight the full region geometry when the board's
      // territories map is populated; otherwise fall back to the
      // representative positions so the decision surface remains visible.
      const territories = gameState.board.territories;
      const regionIdsInDisplayOrder: string[] = [];

      for (const option of pendingChoice.options) {
        // Skip options represent the meta-move "skip_territory_processing"
        // and do not correspond to a concrete region on the board. Avoid
        // highlighting their representative sentinel position.
        if (option.regionId === 'skip' || option.size <= 0) {
          continue;
        }

        if (!regionIdsInDisplayOrder.includes(option.regionId)) {
          regionIdsInDisplayOrder.push(option.regionId);
        }

        const rep = option.representativePosition;
        if (!rep) continue;

        let highlighted = false;

        if (territories && territories.size > 0) {
          // Primary mapping: look up the concrete Territory by regionId
          // when available so geometry/choice stay in lockstep even if
          // representativePosition drifts.
          const territoryById = territories.get(option.regionId);
          if (territoryById) {
            for (const pos of territoryById.spaces) {
              pushPosition(pos, 'primary', option.regionId);
            }
            highlighted = true;
          } else {
            // Fallback: scan for the territory whose spaces contain the
            // representative position, mirroring earlier behaviour.
            territories.forEach((territory) => {
              if (highlighted) return;
              const spaces = territory.spaces ?? [];
              const containsRep = spaces.some((pos) => positionsEqual(pos, rep));
              if (!containsRep) return;

              for (const pos of spaces) {
                pushPosition(pos, 'primary', option.regionId);
              }
              highlighted = true;
            });
          }
        }

        // Fallback: if we could not map the representative position to
        // a concrete territory region, still highlight the representative
        // cell so players and spectators see where the choice lives.
        if (!highlighted) {
          pushPosition(rep, 'primary', option.regionId);
        }
      }

      if (highlights.length === 0) {
        return undefined;
      }

      return {
        choiceKind: vm.kind,
        highlights,
        territoryRegions: {
          regionIdsInDisplayOrder,
        },
      };
    }
    case 'capture_direction': {
      // For capture-direction decisions, we highlight both the capture target
      // and the prospective landing positions, using a slightly lower
      // intensity for the targets.
      for (const option of pendingChoice.options) {
        pushPosition(option.landingPosition, 'primary');
        pushPosition(option.targetPosition, 'secondary');
      }
      break;
    }
    default:
      break;
  }

  if (highlights.length === 0) return undefined;

  return {
    choiceKind: vm.kind,
    highlights,
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

function mapMoveTypeToChoiceType(actionType: Move['type']): PlayerChoiceType | undefined {
  switch (actionType) {
    case 'process_line':
      return 'line_order';
    case 'choose_line_option':
    case 'choose_line_reward':
      return 'line_reward_option';
    case 'process_territory_region':
    case 'choose_territory_option':
    case 'skip_territory_processing':
      return 'region_order';
    case 'eliminate_rings_from_stack':
      return 'ring_elimination';
    default:
      return undefined;
  }
}

function getDecisionTagForMove(action: Move): string | null {
  const choiceType = mapMoveTypeToChoiceType(action.type);
  if (!choiceType) return null;

  const vm = getChoiceViewModelForType(choiceType);
  return `[${vm.copy.shortLabel}] `;
}

function describeHistoryEntry(entry: GameHistoryEntry): string {
  const { action } = entry;
  const moveLabel = `#${entry.moveNumber}`;
  const playerLabel = `P${action.player}`;

  switch (action.type) {
    case 'swap_sides': {
      const otherSeat = action.player === 1 ? 2 : 1;
      return `${moveLabel} — ${playerLabel} invoked the pie rule and swapped colours with P${otherSeat}`;
    }
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
    case 'choose_line_option':
    case 'choose_line_reward': {
      const decisionTag = getDecisionTagForMove(action) ?? '';
      const lineCount = action.formedLines?.length ?? 0;
      if (lineCount > 0) {
        return `${moveLabel} — ${playerLabel} ${decisionTag}processed ${lineCount} line${lineCount === 1 ? '' : 's'}`;
      }
      return `${moveLabel} — ${decisionTag}Line processing by ${playerLabel}`;
    }
    case 'process_territory_region':
    case 'choose_territory_option':
    case 'eliminate_rings_from_stack': {
      const decisionTag = getDecisionTagForMove(action) ?? '';
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
      return `${moveLabel} — ${decisionTag}Territory / elimination processing by ${playerLabel}${detail}`;
    }
    case 'skip_placement': {
      return `${moveLabel} — ${playerLabel} skipped placement`;
    }
    case 'skip_territory_processing': {
      const decisionTag = getDecisionTagForMove(action) ?? '';
      return `${moveLabel} — ${decisionTag}${playerLabel} skipped further territory processing this turn`;
    }
    default: {
      return `${moveLabel} — ${playerLabel} performed ${action.type}`;
    }
  }
}

export function formatVictoryReason(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Ring Elimination';
    case 'territory_control':
      return 'Territory Control';
    case 'last_player_standing':
      return 'Last Player Standing';
    case 'game_completed':
      return 'Structural Stalemate';
    case 'timeout':
      return 'Timeout';
    case 'resignation':
      return 'Resignation';
    case 'abandonment':
      return 'Abandonment';
    case 'draw':
      return 'Draw';
    default: {
      // Fallback for any future or engine-specific reasons that are not
      // explicitly mapped above. At this point `reason` is narrowed to
      // `never` in the type system, so cast defensively for formatting.
      const label = String(reason);
      return label.replace(/_/g, ' ');
    }
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
  selectedPosition?: Position | undefined;
  validTargets?: Position[] | undefined;
  colorVisionMode?: ColorVisionMode | undefined;
  /** Optional decision-phase highlights derived from a pending PlayerChoice. */
  decisionHighlights?: BoardDecisionHighlightsViewModel | undefined;
}

/**
 * Transform BoardState into BoardViewModel
 */
export function toBoardViewModel(
  board: BoardState,
  options: ToBoardViewModelOptions = {}
): BoardViewModel {
  const { selectedPosition, validTargets = [], colorVisionMode = 'normal' } = options;

  const cells: CellViewModel[] = [];

  // Generate all positions based on board type
  if (board.type === 'square8' || board.type === 'square19') {
    const size = board.size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const pos: Position = { x, y };
        const cell = createCellViewModel(
          pos,
          board,
          selectedPosition,
          validTargets,
          colorVisionMode
        );
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
      const cell = createCellViewModel(pos, board, selectedPosition, validTargets, colorVisionMode);
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
    decisionHighlights: options.decisionHighlights,
  };
}

function createCellViewModel(
  pos: Position,
  board: BoardState,
  selectedPosition?: Position,
  validTargets: Position[] = [],
  colorVisionMode: ColorVisionMode = 'normal'
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
    stackViewModel = toStackViewModel(stack, pos, colorVisionMode);
  }

  let markerViewModel: MarkerViewModel | undefined;
  if (!stack && marker && marker.type === 'regular') {
    const colors = getPlayerColors(marker.player, colorVisionMode);
    markerViewModel = {
      position: pos,
      positionKey: key,
      playerNumber: marker.player,
      colorClass: colors.marker,
    };
  }

  let collapsedSpaceViewModel: CollapsedSpaceViewModel | undefined;
  if (collapsedOwner !== undefined) {
    const colors = getPlayerColors(collapsedOwner, colorVisionMode);
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

function toStackViewModel(
  stack: RingStack,
  pos: Position,
  colorVisionMode: ColorVisionMode = 'normal'
): StackViewModel {
  const key = positionToString(pos);
  const rings: RingViewModel[] = [];

  const topIndex = 0;
  const capEndIndex = Math.min(stack.capHeight - 1, stack.rings.length - 1);

  for (let i = 0; i < stack.rings.length; i++) {
    const playerNumber = stack.rings[i];
    const colors = getPlayerColors(playerNumber, colorVisionMode);
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
  currentUserId?: string | undefined;
  isDismissed?: boolean | undefined;
  colorVisionMode?: ColorVisionMode | undefined;
  /**
   * Optional canonical GameEndExplanation used to refine the title/description
   * copy. When provided, the adapter may select more specific wording based on
   * uxCopy keys and concept ids while preserving existing behaviour as the
   * fallback.
   */
  gameEndExplanation?: GameEndExplanation | null | undefined;
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
  const {
    currentUserId,
    isDismissed = false,
    gameEndExplanation,
    colorVisionMode = 'normal',
  } = options;

  if (!gameResult || isDismissed) {
    return null;
  }

  const winner =
    gameResult.winner !== undefined
      ? players.find((p) => p.playerNumber === gameResult.winner)
      : undefined;

  const winnerViewModel =
    winner && gameState
      ? toPlayerViewModel(winner, gameState, currentUserId, colorVisionMode)
      : undefined;

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
    isDraw,
    gameEndExplanation ?? null
  );

  // Build final stats
  const finalStats: PlayerFinalStatsViewModel[] = players.map((player) => {
    const playerVM = gameState
      ? toPlayerViewModel(player, gameState, currentUserId, colorVisionMode)
      : {
          id: player.id,
          playerNumber: player.playerNumber,
          username: player.username || `Player ${player.playerNumber}`,
          isCurrentPlayer: false,
          isUserPlayer: player.id === currentUserId,
          colorClass: getPlayerColors(player.playerNumber, colorVisionMode).card,
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
  isDraw: boolean,
  explanation: GameEndExplanation | null
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

  const victoryReasonCode = explanation?.victoryReasonCode;
  const outcomeType = explanation?.outcomeType;

  // Prefer explanation-driven copy when a canonical GameEndExplanation is present.
  if (
    victoryReasonCode === 'victory_last_player_standing' ||
    outcomeType === 'last_player_standing'
  ) {
    const subject = userWon ? 'You' : winnerName;
    const verb = userWon ? 'were' : 'was';
    return {
      title: '👑 Last Player Standing',
      description: `${subject} ${verb} the only player able to make real moves (placements, movements, or captures) for three consecutive full rounds. Other players either had no real moves or could only perform forced eliminations, which do not count as real moves for Last Player Standing even though they still remove caps and permanently eliminate rings.`,
      titleColorClass,
    };
  }

  if (
    victoryReasonCode === 'victory_structural_stalemate_tiebreak' ||
    outcomeType === 'structural_stalemate'
  ) {
    return {
      title: '🧱 Structural Stalemate',
      description:
        'The game reached a structural stalemate: no player had any legal placements, movements, captures, or forced eliminations left. The winner was chosen by the tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action.',
      titleColorClass,
    };
  }

  const key = explanation?.uxCopy?.detailedSummaryKey ?? explanation?.uxCopy?.shortSummaryKey ?? '';
  if (
    key.startsWith('game_end.territory_mini_region') ||
    explanation?.primaryConceptId === 'territory_mini_region'
  ) {
    return {
      title: `🏰 ${winnerName} Wins!`,
      description:
        'Victory by Territory Control after resolving the final disconnected mini-region. Processing that region eliminated all interior rings, converted its spaces into the winner’s Territory, and pushed them over the Territory victory threshold.',
      titleColorClass,
    };
  }

  // Fallback to legacy reason-based copy when no specific uxCopy key is recognised.
  switch (gameResult.reason) {
    case 'ring_elimination':
      return {
        title: `🏆 ${winnerName} Wins!`,
        description: 'Victory by Ring Elimination: eliminated rings reached the victory threshold.',
        titleColorClass,
      };
    case 'territory_control':
      return {
        title: `🏰 ${winnerName} Wins!`,
        description: 'Victory by controlling the majority of territory.',
        titleColorClass,
      };
    case 'last_player_standing': {
      const subject = userWon ? 'You' : winnerName;
      const verb = userWon ? 'were' : 'was';
      return {
        title: '👑 Last Player Standing',
        description: `${subject} ${verb} the only player able to make real moves (placements, movements, or captures) for three consecutive full rounds. Other players either had no real moves or could only perform forced eliminations, which do not count as real moves for Last Player Standing even though they still remove caps and permanently eliminate rings.`,
        titleColorClass,
      };
    }
    case 'game_completed':
      return {
        title: '🧱 Structural Stalemate',
        description:
          'The game reached a structural stalemate: no player had any legal placements, movements, captures, or forced eliminations left. The winner was chosen by the tiebreak ladder: 1) Territory spaces, 2) Eliminated rings (including rings in hand), 3) Markers, 4) Who made the last real action.',
        titleColorClass,
      };
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

// ═══════════════════════════════════════════════════════════════════════════
// FSM Decision Surface View Model (Phase 5: UI/Telemetry Integration)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * View model for FSM decision surface data.
 *
 * This interface bridges the FSMDecisionSurface (from the FSM module) with
 * UI components and telemetry, providing a presentation-ready summary of
 * pending decisions.
 */
export interface FSMDecisionSurfaceViewModel {
  /** Whether a decision surface is currently active */
  isActive: boolean;

  /** FSM phase that generated this decision surface */
  phase: GamePhase;

  /** Type of pending decision */
  decisionType?: FSMOrchestrationResult['pendingDecisionType'];

  /** Number of pending lines (line_processing phase) */
  pendingLineCount: number;

  /** Number of pending regions (territory_processing phase) */
  pendingRegionCount: number;

  /** Number of chain capture continuations available */
  chainContinuationCount: number;

  /** Number of forced eliminations required */
  forcedEliminationCount: number;

  /**
   * Human-readable summary of the decision surface.
   * E.g., "2 lines to process", "Choose continuation target", "3 regions pending"
   */
  summary: string;

  /**
   * Action hint for the current decision type.
   * Used by UI components to guide the player.
   */
  actionHint: string;
}

/**
 * Extract FSM decision surface view model from game state and optional
 * FSM orchestration result.
 *
 * This adapter transforms FSM-internal decision surface data into a
 * UI-friendly format for display in GameHUD, telemetry emission, and
 * teaching overlays.
 *
 * @param gameState Current game state
 * @param fsmResult Optional FSM orchestration result containing decision surface
 * @returns FSMDecisionSurfaceViewModel for UI consumption
 */
export function toFSMDecisionSurfaceViewModel(
  gameState: GameState,
  fsmResult?: FSMOrchestrationResult | null
): FSMDecisionSurfaceViewModel {
  const phase = gameState.currentPhase;
  const decisionSurface = fsmResult?.decisionSurface;
  const decisionType = fsmResult?.pendingDecisionType;

  const pendingLineCount = decisionSurface?.pendingLines?.length ?? 0;
  const pendingRegionCount = decisionSurface?.pendingRegions?.length ?? 0;
  const chainContinuationCount = decisionSurface?.chainContinuations?.length ?? 0;
  const forcedEliminationCount = decisionSurface?.forcedEliminationCount ?? 0;

  const isActive = !!(
    decisionType ||
    pendingLineCount > 0 ||
    pendingRegionCount > 0 ||
    chainContinuationCount > 0 ||
    forcedEliminationCount > 0
  );

  // Generate human-readable summary based on decision type
  let summary = '';
  let actionHint = '';

  switch (decisionType) {
    case 'line_order_required':
      summary =
        pendingLineCount === 1 ? '1 line to process' : `${pendingLineCount} lines to process`;
      actionHint = 'Choose which line to process first, then select your reward';
      break;
    case 'no_line_action_required':
      summary = 'No lines to process';
      actionHint = 'Phase will advance automatically';
      break;
    case 'region_order_required':
      summary =
        pendingRegionCount === 1
          ? '1 region to resolve'
          : `${pendingRegionCount} regions to resolve`;
      actionHint = 'Choose which disconnected region to process first';
      break;
    case 'no_territory_action_required':
      summary = 'No territory regions to process';
      actionHint = 'Phase will advance automatically';
      break;
    case 'chain_capture':
      summary =
        chainContinuationCount === 1
          ? '1 capture continuation available'
          : `${chainContinuationCount} capture continuations available`;
      actionHint = 'Choose your next capture target to continue the chain';
      break;
    case 'forced_elimination':
      summary =
        forcedEliminationCount === 1
          ? '1 ring must be eliminated'
          : `${forcedEliminationCount} rings must be eliminated`;
      actionHint = 'Select a stack to eliminate rings from';
      break;
    default:
      if (phase === 'line_processing' && pendingLineCount > 0) {
        summary = `${pendingLineCount} line${pendingLineCount !== 1 ? 's' : ''} detected`;
        actionHint = 'Process detected lines for rewards';
      } else if (phase === 'territory_processing' && pendingRegionCount > 0) {
        summary = `${pendingRegionCount} region${pendingRegionCount !== 1 ? 's' : ''} detected`;
        actionHint = 'Resolve disconnected territory regions';
      } else if (phase === 'chain_capture' && chainContinuationCount > 0) {
        summary = 'Chain capture in progress';
        actionHint = 'Continue capturing or end chain if no targets';
      } else if (phase === 'forced_elimination') {
        summary = 'Forced elimination required';
        actionHint = 'You have stacks but no legal moves - must eliminate';
      } else {
        summary = '';
        actionHint = '';
      }
      break;
  }

  return {
    isActive,
    phase,
    decisionType,
    pendingLineCount,
    pendingRegionCount,
    chainContinuationCount,
    forcedEliminationCount,
    summary,
    actionHint,
  };
}

/**
 * Extract telemetry-ready FSM decision surface metrics from a view model.
 *
 * This helper extracts the low-cardinality counts and types suitable for
 * inclusion in RulesUxEventPayload.
 *
 * @param vm FSM decision surface view model
 * @returns Object with FSM telemetry fields
 */
export function extractFSMTelemetryFields(vm: FSMDecisionSurfaceViewModel): {
  fsmPhase: string;
  fsmDecisionType?: FSMOrchestrationResult['pendingDecisionType'];
  fsmPendingLineCount?: number;
  fsmPendingRegionCount?: number;
  fsmChainContinuationCount?: number;
  fsmForcedEliminationCount?: number;
} {
  return {
    fsmPhase: vm.phase,
    fsmDecisionType: vm.decisionType,
    fsmPendingLineCount: vm.pendingLineCount > 0 ? vm.pendingLineCount : undefined,
    fsmPendingRegionCount: vm.pendingRegionCount > 0 ? vm.pendingRegionCount : undefined,
    fsmChainContinuationCount:
      vm.chainContinuationCount > 0 ? vm.chainContinuationCount : undefined,
    fsmForcedEliminationCount:
      vm.forcedEliminationCount > 0 ? vm.forcedEliminationCount : undefined,
  };
}
