import React, { useEffect, useState } from 'react';
import { GameState, Player, GamePhase, BOARD_CONFIGS, TimeControl } from '../../shared/types/game';
import { ConnectionStatus } from '../contexts/GameContext';
import { Button } from './ui/Button';
import { getCountdownSeverity } from '../utils/countdown';
import type {
  HUDViewModel,
  PhaseViewModel,
  PlayerViewModel,
  PlayerRingStatsViewModel,
  HUDDecisionPhaseViewModel,
} from '../adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// Props Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * View model-based Sub-components
 * ═══════════════════════════════════════════════════════════════════════════
 */

function DecisionPhaseBanner({ vm }: { vm: HUDDecisionPhaseViewModel }) {
  const { label, description, timeRemainingMs, showCountdown, isServerCapped } = vm;

  let countdownLabel: string | null = null;
  if (showCountdown && timeRemainingMs !== null) {
    const totalSeconds = Math.max(0, Math.floor(timeRemainingMs / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    countdownLabel = `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  const severity =
    showCountdown && timeRemainingMs !== null ? getCountdownSeverity(timeRemainingMs) : null;

  const pillBgClass =
    severity === 'critical'
      ? 'bg-red-900/70 border-red-400/80'
      : severity === 'warning'
        ? 'bg-amber-900/70 border-amber-400/80'
        : severity === 'normal'
          ? 'bg-emerald-900/70 border-emerald-400/80'
          : 'bg-slate-900/60 border-slate-500/70';

  const pillTextClass =
    severity === 'critical'
      ? 'text-red-50'
      : severity === 'warning'
        ? 'text-amber-50'
        : severity === 'normal'
          ? 'text-emerald-50'
          : 'text-slate-50';

  const pillTimerClass =
    severity === 'critical'
      ? 'font-mono text-xs font-semibold text-red-100'
      : severity === 'warning'
        ? 'font-mono text-xs font-semibold text-amber-100'
        : severity === 'normal'
          ? 'font-mono text-xs text-emerald-100'
          : 'font-mono text-xs text-slate-100';

  const pillRingClass = isServerCapped ? 'ring-1 ring-amber-300/80' : '';
  const pillLabel = isServerCapped ? 'Server deadline' : 'Time';

  return (
    <div
      className="mt-3 px-4 py-2 bg-indigo-900/60 border border-indigo-400/70 rounded-lg text-[11px] sm:text-xs flex items-center justify-between gap-3"
      data-testid="decision-phase-banner"
    >
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-slate-50 truncate">{label}</div>
        {description && (
          <div className="text-[11px] text-slate-200/85 mt-0.5 line-clamp-2">{description}</div>
        )}
      </div>
      {countdownLabel && (
        <div
          className={`shrink-0 inline-flex items-center gap-2 px-2 py-1 rounded-full border ${pillBgClass} ${pillRingClass} ${
            severity === 'critical' ? 'animate-pulse' : ''
          }`}
          aria-label="Decision timer"
          data-testid="decision-phase-countdown"
          data-severity={severity ?? undefined}
          data-server-capped={isServerCapped ? 'true' : undefined}
        >
          <span className={`text-[10px] uppercase tracking-wide ${pillTextClass}`}>{pillLabel}</span>
          <span className={pillTimerClass}>{countdownLabel}</span>
        </div>
      )}
    </div>
  );
}
export interface GameHUDLegacyProps {
  gameState: GameState;
  currentPlayer: Player | undefined;
  instruction?: string;
  connectionStatus?: ConnectionStatus;
  isSpectator?: boolean;
  lastHeartbeatAt?: number | null;
  currentUserId?: string;
  /**
   * Optional callback used by hosts to surface a contextual
   * "Board controls & shortcuts" overlay entry point.
   */
  onShowBoardControls?: () => void;
}

/**
 * New view model props interface.
 * Components pass pre-transformed view model for maximum decoupling.
 */
export interface GameHUDViewModelProps {
  /** Pre-transformed view model from useHUDViewModel or toHUDViewModel */
  viewModel: HUDViewModel;
  /** Additional GameState needed for time control display */
  timeControl?: TimeControl;
  /**
   * Optional callback used by hosts to surface a contextual
   * "Board controls & shortcuts" overlay entry point.
   */
  onShowBoardControls?: () => void;
}

/**
 * Combined props type supporting both legacy and view model interfaces.
 * When viewModel is provided, legacy props are ignored.
 */
export type GameHUDProps = GameHUDLegacyProps | GameHUDViewModelProps;

// ═══════════════════════════════════════════════════════════════════════════
// Type Guards
// ═══════════════════════════════════════════════════════════════════════════

function isViewModelProps(props: GameHUDProps): props is GameHUDViewModelProps {
  return 'viewModel' in props && props.viewModel !== undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase Helpers (kept for legacy compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Phase information with user-friendly labels, descriptions, colors, and icons
 */
interface PhaseInfo {
  label: string;
  description: string;
  color: string;
  icon: string;
}

function getPhaseInfo(phase: GamePhase): PhaseInfo {
  switch (phase) {
    case 'ring_placement':
      return {
        label: 'Ring Placement',
        description: 'Place your rings on the board',
        color: 'bg-blue-500',
        icon: '🎯',
      };
    case 'movement':
      return {
        label: 'Movement Phase',
        description: 'Move a stack or capture opponent pieces',
        color: 'bg-green-500',
        icon: '⚡',
      };
    case 'capture':
      return {
        label: 'Capture Phase',
        description: 'Execute a capture move',
        color: 'bg-orange-500',
        icon: '⚔️',
      };
    case 'chain_capture':
      return {
        label: 'Chain Capture',
        description: 'Continue capturing or end your turn',
        color: 'bg-orange-500',
        icon: '🔗',
      };
    case 'line_processing':
      return {
        label: 'Line Reward',
        description: 'Choose how to process your line',
        color: 'bg-purple-500',
        icon: '📏',
      };
    case 'territory_processing':
      return {
        label: 'Territory Claim',
        description: 'Choose regions to collapse',
        color: 'bg-pink-500',
        icon: '🏰',
      };
    default:
      return {
        label: 'Unknown Phase',
        description: '',
        color: 'bg-gray-400',
        icon: '❓',
      };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// View Model-based Sub-components
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Phase indicator showing current game phase with icon and description
 * Supports both legacy GameState and PhaseViewModel
 */
function PhaseIndicator({ phase }: { phase: PhaseViewModel }) {
  return (
    <div className={`${phase.colorClass} text-white px-4 py-2 rounded-lg shadow-lg`}>
      <div className="flex items-center gap-2">
        {phase.icon && <span className="text-2xl">{phase.icon}</span>}
        <div>
          <div className="font-bold">{phase.label}</div>
          <div className="text-sm opacity-90">{phase.description}</div>
        </div>
      </div>
    </div>
  );
}

/**
 * Legacy phase indicator for backward compatibility
 */
function LegacyPhaseIndicator({ gameState }: { gameState: GameState }) {
  const phaseInfo = getPhaseInfo(gameState.currentPhase);

  return (
    <div className={`${phaseInfo.color} text-white px-4 py-2 rounded-lg shadow-lg`}>
      <div className="flex items-center gap-2">
        {phaseInfo.icon && <span className="text-2xl">{phaseInfo.icon}</span>}
        <div>
          <div className="font-bold">{phaseInfo.label}</div>
          <div className="text-sm opacity-90">{phaseInfo.description}</div>
        </div>
      </div>
    </div>
  );
}

/**
 * Sub-phase details display
 */
function SubPhaseDetails({ detail }: { detail?: string }) {
  if (!detail) return null;
  return <div className="text-sm text-gray-600 mt-1">{detail}</div>;
}

/**
 * Legacy sub-phase details for backward compatibility
 */
function LegacySubPhaseDetails({ gameState }: { gameState: GameState }) {
  // For line processing, check if we have pending line decisions
  if (gameState.currentPhase === 'line_processing') {
    const formedLines = gameState.board.formedLines || [];
    if (formedLines.length > 0) {
      return (
        <div className="text-sm text-gray-600 mt-1">
          Processing {formedLines.length} line{formedLines.length !== 1 ? 's' : ''}
        </div>
      );
    }
  }

  // For territory processing, we don't have a direct count in GameState,
  // but we can show a generic message
  if (gameState.currentPhase === 'territory_processing') {
    return <div className="text-sm text-gray-600 mt-1">Processing disconnected regions</div>;
  }

  return null;
}

/**
 * Player timer with countdown display
 */
interface PlayerTimerProps {
  player: Player;
  isActive: boolean;
  timeControl?: TimeControl;
}

function PlayerTimer({ player, isActive, timeControl }: PlayerTimerProps) {
  const [timeRemaining, setTimeRemaining] = useState(player.timeRemaining ?? 0);

  useEffect(() => {
    if (!isActive || !timeControl) return;

    const interval = setInterval(() => {
      setTimeRemaining((prev) => Math.max(0, prev - 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive, timeControl]);

  // Sync with player's actual time when it updates
  useEffect(() => {
    setTimeRemaining(player.timeRemaining ?? 0);
  }, [player.timeRemaining]);

  if (!timeControl) return null;

  const minutes = Math.floor(timeRemaining / 60000);
  const seconds = Math.floor((timeRemaining % 60000) / 1000);
  const isLowTime = timeRemaining < 60000; // Less than 1 minute

  return (
    <div className={`font-mono ${isLowTime ? 'text-red-600 font-bold' : 'text-gray-700'}`}>
      {minutes}:{seconds.toString().padStart(2, '0')}
    </div>
  );
}

/**
 * Calculate ring statistics for a player
 */
function calculateRingStats(player: Player, gameState: GameState) {
  const boardConfig = BOARD_CONFIGS[gameState.boardType];
  const total = boardConfig.ringsPerPlayer;

  // Count rings on board
  const onBoard = Array.from(gameState.board.stacks.values()).reduce((count, stack) => {
    return count + stack.rings.filter((r) => r === player.playerNumber).length;
  }, 0);

  const eliminated = player.eliminatedRings ?? 0;
  const inHand = player.ringsInHand ?? 0;

  return { inHand, onBoard, eliminated, total };
}

/**
 * Detailed ring statistics display
 */
interface RingStatsProps {
  player: Player;
  gameState: GameState;
}

function RingStats({ player, gameState }: RingStatsProps) {
  const ringStats = calculateRingStats(player, gameState);

  return (
    <div className="grid grid-cols-3 gap-2 text-xs mt-2">
      <div className="text-center">
        <div className="font-bold">{ringStats.inHand}</div>
        <div className="text-gray-500">In Hand</div>
      </div>
      <div className="text-center">
        <div className="font-bold">{ringStats.onBoard}</div>
        <div className="text-gray-500">On Board</div>
      </div>
      <div className="text-center">
        <div className="font-bold text-red-600">{ringStats.eliminated}</div>
        <div className="text-gray-500">Lost</div>
      </div>
    </div>
  );
}

/**
 * Territory statistics display
 */
function TerritoryStats({ player }: { player: Player }) {
  const territoryCount = player.territorySpaces ?? 0;

  if (territoryCount === 0) return null;

  return (
    <div className="text-sm mt-1 text-center">
      <span className="font-semibold">{territoryCount}</span> territory space
      {territoryCount !== 1 ? 's' : ''}
    </div>
  );
}

/**
 * Game progress display (turn/move counter)
 */
function GameProgress({ gameState }: { gameState: GameState }) {
  const turnNumber = gameState.moveHistory.length;
  const moveNumber =
    gameState.history.length > 0 ? gameState.history[gameState.history.length - 1]?.moveNumber : 0;

  return (
    <div className="text-center py-2 bg-gray-100 rounded">
      <div className="text-2xl font-bold">{turnNumber}</div>
      <div className="text-xs text-gray-600">Turn</div>
      {moveNumber > 0 && <div className="text-xs text-gray-500">Move #{moveNumber}</div>}
    </div>
  );
}

/**
 * Get friendly display label and color for AI difficulty
 */
function getAIDifficultyInfo(difficulty: number): {
  label: string;
  color: string;
  bgColor: string;
} {
  // Keep this categorisation broadly aligned with the canonical ladder:
  // 1 → Random, 2 → Heuristic, 3–6 → Minimax, 7–8 → MCTS, 9–10 → Descent.
  if (difficulty === 1) {
    return {
      label: 'Beginner · Random',
      color: 'text-green-300',
      bgColor: 'bg-green-900/40',
    };
  }
  if (difficulty === 2) {
    return {
      label: 'Easy · Heuristic',
      color: 'text-emerald-300',
      bgColor: 'bg-emerald-900/40',
    };
  }
  if (difficulty >= 3 && difficulty <= 6) {
    return {
      label: 'Advanced · Minimax',
      color: 'text-blue-300',
      bgColor: 'bg-blue-900/40',
    };
  }
  if (difficulty === 7 || difficulty === 8) {
    return {
      label: 'Expert · MCTS',
      color: 'text-purple-300',
      bgColor: 'bg-purple-900/40',
    };
  }
  // 9–10
  return {
    label: 'Grandmaster · Descent',
    color: 'text-red-300',
    bgColor: 'bg-red-900/40',
  };
}

/**
 * Get display name for AI type
 */
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

/**
 * Badge component for player indicators
 */
function Badge({
  variant = 'default',
  children,
}: {
  variant?: 'default' | 'primary';
  children: React.ReactNode;
}) {
  const classes =
    variant === 'primary'
      ? 'px-2 py-0.5 rounded-full bg-blue-500 text-white text-xs font-semibold'
      : 'px-2 py-0.5 rounded-full bg-gray-600 text-gray-200 text-xs font-semibold';

  return <span className={classes}>{children}</span>;
}

const PLAYER_COLOR_CLASSES = ['bg-emerald-500', 'bg-sky-500', 'bg-amber-500', 'bg-fuchsia-500'];

// ═══════════════════════════════════════════════════════════════════════════
// View Model-based Player Card
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Ring stats display using view model
 */
function RingStatsFromVM({ stats }: { stats: PlayerRingStatsViewModel }) {
  return (
    <div className="grid grid-cols-3 gap-2 text-xs mt-2">
      <div className="text-center">
        <div className="font-bold">{stats.inHand}</div>
        <div className="text-gray-500">In Hand</div>
      </div>
      <div className="text-center">
        <div className="font-bold">{stats.onBoard}</div>
        <div className="text-gray-500">On Board</div>
      </div>
      <div className="text-center">
        <div className="font-bold text-red-600">{stats.eliminated}</div>
        <div className="text-gray-500">Lost</div>
      </div>
    </div>
  );
}

/**
 * Player card using view model
 */
function PlayerCardFromVM({
  player,
  timeControl,
}: {
  player: PlayerViewModel;
  timeControl?: TimeControl;
}) {
  return (
    <div
      className={`
      p-3 rounded-lg border-2 transition-all
      ${player.isCurrentPlayer ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}
      ${player.isUserPlayer ? 'ring-2 ring-green-400' : ''}
    `}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded-full ${player.colorClass}`} />
          <span className="font-semibold">{player.username}</span>
          {player.aiInfo.isAI && <Badge>🤖 AI</Badge>}
          {player.isCurrentPlayer && <Badge variant="primary">Current Turn</Badge>}
        </div>

        {timeControl && player.timeRemaining !== undefined && (
          <PlayerTimerFromVM
            timeRemaining={player.timeRemaining}
            isActive={player.isCurrentPlayer}
          />
        )}
      </div>

      {player.aiInfo.isAI && (
        <div className="flex flex-col gap-1 mb-2">
          <span
            className={`text-[10px] px-1.5 py-0.5 rounded ${player.aiInfo.difficultyBgColor} ${player.aiInfo.difficultyColor} font-semibold`}
          >
            {player.aiInfo.difficultyLabel} Lv{player.aiInfo.difficulty}
          </span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-300">
            {player.aiInfo.aiTypeLabel}
          </span>
        </div>
      )}

      <RingStatsFromVM stats={player.ringStats} />
      {player.territorySpaces > 0 && (
        <div className="text-sm mt-1 text-center">
          <span className="font-semibold">{player.territorySpaces}</span> territory space
          {player.territorySpaces !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

/**
 * Simplified timer for view model usage
 */
function PlayerTimerFromVM({
  timeRemaining,
  isActive,
}: {
  timeRemaining: number;
  isActive: boolean;
}) {
  const [displayTime, setDisplayTime] = useState(timeRemaining);

  useEffect(() => {
    if (!isActive) {
      setDisplayTime(timeRemaining);
      return;
    }

    const interval = setInterval(() => {
      setDisplayTime((prev) => Math.max(0, prev - 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive, timeRemaining]);

  useEffect(() => {
    setDisplayTime(timeRemaining);
  }, [timeRemaining]);

  const minutes = Math.floor(displayTime / 60000);
  const seconds = Math.floor((displayTime % 60000) / 1000);
  const isLowTime = displayTime < 60000;

  return (
    <div className={`font-mono ${isLowTime ? 'text-red-600 font-bold' : 'text-gray-700'}`}>
      {minutes}:{seconds.toString().padStart(2, '0')}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy Player Card (for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Legacy player card component
 */
interface LegacyPlayerCardProps {
  player: Player;
  gameState: GameState;
  isCurrentPlayer: boolean;
  isUserPlayer: boolean;
}

function LegacyPlayerCard({
  player,
  gameState,
  isCurrentPlayer,
  isUserPlayer,
}: LegacyPlayerCardProps) {
  const colorClass = PLAYER_COLOR_CLASSES[player.playerNumber - 1] ?? 'bg-gray-500';

  return (
    <div
      className={`
      p-3 rounded-lg border-2 transition-all
      ${isCurrentPlayer ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}
      ${isUserPlayer ? 'ring-2 ring-green-400' : ''}
    `}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={`w-4 h-4 rounded-full ${colorClass}`} />
          <span className="font-semibold">{player.username}</span>
          {player.type === 'ai' && <Badge>🤖 AI</Badge>}
          {isCurrentPlayer && <Badge variant="primary">Current Turn</Badge>}
        </div>

        {gameState.timeControl && (
          <PlayerTimer
            player={player}
            isActive={isCurrentPlayer}
            timeControl={gameState.timeControl}
          />
        )}
      </div>

      {player.type === 'ai' && (
        <div className="flex flex-col gap-1 mb-2">
          {(() => {
            const difficulty = player.aiProfile?.difficulty ?? player.aiDifficulty ?? 5;
            const aiType = player.aiProfile?.aiType ?? 'heuristic';
            const diffInfo = getAIDifficultyInfo(difficulty);
            return (
              <>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded ${diffInfo.bgColor} ${diffInfo.color} font-semibold`}
                >
                  {diffInfo.label} Lv{difficulty}
                </span>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700/60 text-slate-300">
                  {getAITypeLabel(aiType)}
                </span>
              </>
            );
          })()}
        </div>
      )}

      <RingStats player={player} gameState={gameState} />
      <TerritoryStats player={player} />
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Game HUD Component
 *
 * Displays game status, phase, players, and connection information.
 *
 * Supports two usage patterns:
 *
 * 1. Legacy (backward compatible):
 * ```tsx
 * <GameHUD
 *   gameState={gameState}
 *   currentPlayer={currentPlayer}
 *   instruction="Place a ring"
 *   connectionStatus={connectionStatus}
 * />
 * ```
 *
 * 2. View Model (recommended for new code):
 * ```tsx
 * const viewModel = useHUDViewModel({ instruction, currentUserId });
 * <GameHUD viewModel={viewModel} timeControl={gameState?.timeControl} />
 * ```
 */
export function GameHUD(props: GameHUDProps) {
  // Use view model props if available
  if (isViewModelProps(props)) {
    return (
      <GameHUDFromViewModel
        viewModel={props.viewModel}
        timeControl={props.timeControl}
        onShowBoardControls={props.onShowBoardControls}
      />
    );
  }

  // Legacy rendering path
  return <GameHUDLegacy {...props} />;
}

/**
 * View model-based HUD implementation
 */
function GameHUDFromViewModel({
  viewModel,
  timeControl,
  onShowBoardControls,
}: {
  viewModel: HUDViewModel;
  timeControl?: TimeControl;
  onShowBoardControls?: () => void;
}) {
  const {
    phase,
    players,
    turnNumber,
    moveNumber,
    pieRuleSummary,
    instruction,
    connectionStatus,
    isConnectionStale,
    isSpectator,
    spectatorCount,
    subPhaseDetail,
    decisionPhase,
  } = viewModel;

  const connectionLabel = (() => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting…';
      case 'reconnecting':
        return 'Reconnecting…';
      case 'disconnected':
      default:
        return 'Disconnected';
    }
  })();

  const connectionColor =
    connectionStatus === 'connected' && !isConnectionStale
      ? 'text-emerald-300'
      : connectionStatus === 'reconnecting'
        ? 'text-amber-300'
        : 'text-rose-300';

  return (
    <div className="w-full max-w-4xl mx-auto mb-4" data-testid="game-hud">
      {/* Connection Status */}
      <div className="flex items-center justify-between text-xs text-slate-300 mb-3">
        <div className={`font-semibold ${connectionColor}`}>
          Connection: {connectionLabel}
          {isConnectionStale && (
            <span className="ml-1 text-[11px] text-amber-200">(awaiting update…)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {spectatorCount > 0 && (
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="w-3 h-3"
              >
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path
                  fillRule="evenodd"
                  d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              {spectatorCount}
            </span>
          )}
          {isSpectator && (
            <span className="px-2 py-0.5 rounded-full bg-purple-900/40 border border-purple-500/40 text-purple-100 uppercase tracking-wide font-semibold">
              Spectator
            </span>
          )}
          {onShowBoardControls && (
            <Button
              variant="ghost"
              size="sm"
              aria-label="Show board controls"
              onClick={onShowBoardControls}
              data-testid="board-controls-button"
              className="h-7 w-7 rounded-full border border-slate-600 text-[11px] leading-none px-0"
            >
              ?
            </Button>
          )}
        </div>
      </div>

      {/* Phase Indicator */}
      <PhaseIndicator phase={phase} />
      <SubPhaseDetails detail={subPhaseDetail} />

      {pieRuleSummary && (
        <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-900/60 border border-amber-500/60 text-[11px] text-amber-100">
          <span className="font-semibold uppercase tracking-wide">Pie rule</span>
          <span className="text-amber-100/90">{pieRuleSummary}</span>
        </div>
      )}

      {/* Game Progress */}
      <div className="mt-3">
        <div className="text-center py-2 bg-gray-100 rounded">
          <div className="text-2xl font-bold">{turnNumber}</div>
          <div className="text-xs text-gray-600">Turn</div>
          {moveNumber > 0 && <div className="text-xs text-gray-500">Move #{moveNumber}</div>}
        </div>
      </div>

      {/* Instruction Banner */}
      {instruction && (
        <div className="mt-3 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-center">
          <span className="text-slate-200 font-medium">{instruction}</span>
        </div>
      )}

      {/* Decision Phase Banner */}
      {decisionPhase && decisionPhase.isActive && <DecisionPhaseBanner vm={decisionPhase} />}

      {/* Victory Conditions Helper */}
      <div
        className="mt-3 px-4 py-2 bg-slate-800/60 border border-slate-700 rounded-lg text-[11px] text-slate-300 leading-snug"
        data-testid="victory-conditions-help"
      >
        <div className="font-semibold text-slate-100 mb-1">Victory</div>
        <div>• Elimination – eliminate {'>'}50% of all rings.</div>
        <div>• Territory – control {'>'}50% of all board spaces.</div>
        <div>
          • Last Player Standing – after a full round you are the only player able to make real
          moves (placements, movements, or captures).
        </div>
      </div>

      {/* Player Cards */}
      <div className="mt-4 space-y-3">
        {players.map((player) => (
          <PlayerCardFromVM key={player.id} player={player} timeControl={timeControl} />
        ))}
      </div>
    </div>
  );
}

/**
 * Legacy HUD implementation for backward compatibility
 */
function GameHUDLegacy({
  gameState,
  currentPlayer,
  instruction,
  connectionStatus = 'connected',
  isSpectator = false,
  lastHeartbeatAt,
  currentUserId,
  onShowBoardControls,
}: GameHUDLegacyProps) {
  if (!currentPlayer) return null;

  const spectatorCount = gameState.spectators.length;

  const connectionLabel = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting…';
      case 'reconnecting':
        return 'Reconnecting…';
      case 'disconnected':
      default:
        return 'Disconnected';
    }
  };

  const HEARTBEAT_STALE_THRESHOLD_MS = 8000;
  const heartbeatAge = lastHeartbeatAt ? Date.now() - lastHeartbeatAt : null;
  const isHeartbeatStale =
    heartbeatAge !== null &&
    heartbeatAge > HEARTBEAT_STALE_THRESHOLD_MS &&
    connectionStatus === 'connected';

  const connectionColor =
    connectionStatus === 'connected' && !isHeartbeatStale
      ? 'text-emerald-300'
      : connectionStatus === 'reconnecting'
        ? 'text-amber-300'
        : 'text-rose-300';

  return (
    <div className="w-full max-w-4xl mx-auto mb-4" data-testid="game-hud">
      {/* Connection Status */}
      <div className="flex items-center justify-between text-xs text-slate-300 mb-3">
        <div className={`font-semibold ${connectionColor}`}>
          Connection: {connectionLabel()}
          {isHeartbeatStale && (
            <span className="ml-1 text-[11px] text-amber-200">(awaiting update…)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {spectatorCount > 0 && (
            <span className="text-[11px] text-slate-400 flex items-center gap-1">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="w-3 h-3"
              >
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path
                  fillRule="evenodd"
                  d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              {spectatorCount}
            </span>
          )}
          {isSpectator && (
            <span className="px-2 py-0.5 rounded-full bg-purple-900/40 border border-purple-500/40 text-purple-100 uppercase tracking-wide font-semibold">
              Spectator
            </span>
          )}
          {onShowBoardControls && (
            <Button
              variant="ghost"
              size="sm"
              aria-label="Show board controls"
              onClick={onShowBoardControls}
              data-testid="board-controls-button"
              className="h-7 w-7 rounded-full border border-slate-600 text-[11px] leading-none px-0"
            >
              ?
            </Button>
          )}
        </div>
      </div>

      {/* Phase Indicator */}
      <LegacyPhaseIndicator gameState={gameState} />
      <LegacySubPhaseDetails gameState={gameState} />

      {/* Game Progress */}
      <div className="mt-3">
        <GameProgress gameState={gameState} />
      </div>

      {/* Instruction Banner */}
      {instruction && (
        <div className="mt-3 px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-center">
          <span className="text-slate-200 font-medium">{instruction}</span>
        </div>
      )}

      {/* Victory Conditions Helper */}
      <div
        className="mt-3 px-4 py-2 bg-slate-800/60 border border-slate-700 rounded-lg text-[11px] text-slate-300 leading-snug"
        data-testid="victory-conditions-help"
      >
        <div className="font-semibold text-slate-100 mb-1">Victory</div>
        <div>• Elimination – eliminate {'>'}50% of all rings.</div>
        <div>• Territory – control {'>'}50% of all board spaces.</div>
        <div>
          • Last Player Standing – after a full round you are the only player able to make real
          moves (placements, movements, or captures).
        </div>
      </div>

      {/* Player Cards */}
      <div className="mt-4 space-y-3">
        {gameState.players.map((player) => (
          <LegacyPlayerCard
            key={player.id}
            player={player}
            gameState={gameState}
            isCurrentPlayer={player.playerNumber === gameState.currentPlayer}
            isUserPlayer={player.id === currentUserId}
          />
        ))}
      </div>
    </div>
  );
}
