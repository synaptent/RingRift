import React, { useEffect, useRef, useState } from 'react';
import {
  GameState,
  Player,
  GamePhase,
  BOARD_CONFIGS,
  TimeControl,
  BoardType,
} from '../../shared/types/game';
import { ConnectionStatus } from '../contexts/GameContext';
import { Button } from './ui/Button';
import { Tooltip } from './ui/Tooltip';
import { getCountdownSeverity } from '../utils/countdown';
import type {
  HUDViewModel,
  PhaseViewModel,
  PlayerViewModel,
  PlayerRingStatsViewModel,
  HUDDecisionPhaseViewModel,
  HUDWeirdStateViewModel,
} from '../adapters/gameViewModels';
import {
  TeachingOverlay,
  useTeachingOverlay,
  type TeachingTopic,
  type WeirdStateOverlayContext,
} from './TeachingOverlay';
import type { RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import {
  sendRulesUxEvent,
  logRulesUxEvent,
  logHelpOpenEvent,
  newHelpSessionId,
  newOverlaySessionId,
} from '../utils/rulesUxTelemetry';
import {
  getWeirdStateReasonForType,
  isSurfaceableWeirdStateType,
} from '../../shared/engine/weirdStateReasons';

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

  // Build comprehensive aria-label for the decision phase banner
  const bannerAriaLabelParts = [`Decision phase: ${label}`];
  if (description) {
    bannerAriaLabelParts.push(description);
  }
  if (countdownLabel) {
    bannerAriaLabelParts.push(`Time remaining: ${countdownLabel}`);
    if (severity === 'critical') {
      bannerAriaLabelParts.push('Time is running out!');
    }
  }
  const bannerAriaLabel = bannerAriaLabelParts.join('. ');

  return (
    <div
      className="mt-3 px-4 py-2 bg-indigo-900/60 border border-indigo-400/70 rounded-lg text-[11px] sm:text-xs flex items-center justify-between gap-3"
      role="status"
      aria-live={severity === 'critical' ? 'assertive' : 'polite'}
      aria-label={bannerAriaLabel}
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
          <span className={`text-[10px] uppercase tracking-wide ${pillTextClass}`}>
            {pillLabel}
          </span>
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
  timeControl?: TimeControl | undefined;
  /**
   * Optional flag used by sandbox hosts to indicate that this game is
   * running entirely as a local sandbox (no backend game session).
   * When true, the HUD surfaces a small non-intrusive banner so users
   * understand why features like matchmaking or persistence are absent.
   */
  isLocalSandboxOnly?: boolean;
  /**
   * Optional callback used by hosts to surface a contextual
   * "Board controls & shortcuts" overlay entry point.
   */
  onShowBoardControls?: (() => void) | undefined;
  /** Hide the victory conditions panel (when rendering it elsewhere) */
  hideVictoryConditions?: boolean;
  /**
   * Optional context used for lightweight rules-UX telemetry. When provided,
   * the HUD will emit privacy-aware telemetry for teaching/help usage and
   * weird-state interactions. When omitted, telemetry is disabled for this HUD.
   */
  rulesUxContext?: {
    boardType: BoardType;
    numPlayers: number;
    aiDifficulty?: number;
    rulesConcept?: string;
    scenarioId?: string;
  };
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
        description: 'Place a ring on an empty space or on top of one of your own stacks.',
        color: 'bg-blue-500',
        icon: '🎯',
      };
    case 'movement':
      return {
        label: 'Movement Phase',
        description: 'Select one of your stacks and move it to a legal destination.',
        color: 'bg-green-500',
        icon: '⚡',
      };
    case 'capture':
      return {
        label: 'Capture Phase',
        description: 'Overtake an adjacent stack by jumping over it to a legal landing space.',
        color: 'bg-orange-500',
        icon: '⚔️',
      };
    case 'chain_capture':
      return {
        label: 'Chain Capture',
        description: 'Continue the capture by choosing the next target and landing space.',
        color: 'bg-orange-500',
        icon: '🔗',
      };
    case 'line_processing':
      return {
        label: 'Line Processing',
        description: 'Choose which completed line to resolve and what reward to take.',
        color: 'bg-purple-500',
        icon: '📏',
      };
    case 'territory_processing':
      return {
        label: 'Territory Processing',
        description: 'Choose a disconnected region to resolve; some choices may cost you rings.',
        color: 'bg-pink-500',
        icon: '🏰',
      };
    case 'forced_elimination':
      return {
        label: 'Forced Elimination',
        description: 'No legal moves available – you must eliminate from one of your stacks.',
        color: 'bg-red-600',
        icon: '💥',
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
 * Phase indicator showing current game phase with icon, description, and action hint
 * Supports both legacy GameState and PhaseViewModel
 */
function PhaseIndicator({
  phase,
  isMyTurn,
  isSpectator,
}: {
  phase: PhaseViewModel;
  isMyTurn?: boolean;
  isSpectator?: boolean;
}) {
  // Determine which hint to display based on the viewer's role
  const contextualHint = isSpectator
    ? phase.spectatorHint
    : isMyTurn
      ? phase.actionHint
      : undefined;

  const tooltipLines: string[] = [];
  if (phase.description) {
    tooltipLines.push(phase.description);
  }
  if (isSpectator) {
    tooltipLines.push('', `Spectators: ${phase.spectatorHint}`);
  } else if (phase.actionHint) {
    tooltipLines.push('', `On your turn: ${phase.actionHint}`);
  }
  const tooltipContent = tooltipLines.join('\n');

  // Build comprehensive aria-label for the phase indicator
  const phaseAriaLabel = [
    `Current phase: ${phase.label}`,
    phase.description,
    contextualHint ? `Action: ${contextualHint}` : undefined,
  ]
    .filter(Boolean)
    .join('. ');

  return (
    <div
      className={`${phase.colorClass} text-white px-4 py-3 rounded-xl shadow-lg ring-1 ring-white/10 transition-all duration-300 ease-out`}
      role="status"
      aria-live="polite"
      aria-label={phaseAriaLabel}
      data-testid="phase-indicator"
    >
      <div className="flex items-center gap-3">
        {phase.icon && (
          <span className="text-2xl flex-shrink-0 bg-white/15 rounded-lg p-2 shadow-inner">
            {phase.icon}
          </span>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div className="font-bold text-base tracking-tight">{phase.label}</div>
            <Tooltip content={tooltipContent}>
              <span
                className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/40 bg-white/15 text-[10px] text-white font-medium hover:bg-white/25 hover:border-white/60 transition-colors cursor-help"
                aria-label="Phase details"
                data-testid="phase-tooltip-trigger"
              >
                ?
              </span>
            </Tooltip>
          </div>
          <div className="text-sm text-white/85 mt-0.5">{phase.description}</div>
          {contextualHint && (
            <div
              className="mt-2 text-xs font-semibold bg-white/25 backdrop-blur-sm rounded-md px-2.5 py-1.5 inline-flex items-center gap-1.5 shadow-sm border border-white/20"
              data-testid="phase-action-hint"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
              {contextualHint}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * High-level banner for unusual rules states such as Active-No-Moves,
 * Forced Elimination, or structural stalemate. This is driven entirely
 * by HUDWeirdStateViewModel and does not affect rules semantics.
 */
function WeirdStateBanner({
  weirdState,
  onShowHelp,
}: {
  weirdState: HUDWeirdStateViewModel;
  onShowHelp?: () => void;
}) {
  const toneClasses =
    weirdState.tone === 'critical'
      ? 'border-red-400/60 bg-gradient-to-r from-red-950/80 to-red-900/60 text-red-50 shadow-lg shadow-red-900/30'
      : weirdState.tone === 'warning'
        ? 'border-amber-400/60 bg-gradient-to-r from-amber-950/80 to-amber-900/60 text-amber-50 shadow-lg shadow-amber-900/20'
        : 'border-sky-400/60 bg-gradient-to-r from-sky-950/80 to-sky-900/60 text-sky-50 shadow-lg shadow-sky-900/20';

  const badgeLabel =
    weirdState.type === 'last-player-standing'
      ? 'Last Player Standing'
      : weirdState.type === 'forced-elimination'
        ? 'Forced Elimination'
        : weirdState.type === 'structural-stalemate'
          ? 'Structural stalemate'
          : weirdState.type.startsWith('active-no-moves')
            ? 'No Legal Moves'
            : 'Rules notice';

  const icon = weirdState.tone === 'critical' ? '⚠️' : weirdState.tone === 'warning' ? '⚠️' : 'ℹ️';

  const iconBgClass =
    weirdState.tone === 'critical'
      ? 'bg-red-900/50'
      : weirdState.tone === 'warning'
        ? 'bg-amber-900/50'
        : 'bg-sky-900/50';

  return (
    <div
      className={`mb-3 px-4 py-3 rounded-xl border text-xs sm:text-[13px] flex items-start gap-3 transition-all duration-300 ${toneClasses}`}
      role="status"
      aria-live="polite"
      data-testid="hud-weird-state-banner"
    >
      <div className={`mt-0.5 text-lg p-1.5 rounded-lg ${iconBgClass}`} aria-hidden="true">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] uppercase tracking-wider font-bold opacity-90 px-2 py-0.5 rounded-full bg-white/10 border border-white/10">
            {badgeLabel}
          </span>
        </div>
        <div className="font-semibold text-sm leading-snug">{weirdState.title}</div>
        <div className="mt-1.5 text-[11px] leading-snug opacity-85">{weirdState.body}</div>
      </div>
      {onShowHelp && (
        <button
          type="button"
          onClick={onShowHelp}
          className="ml-1 mt-0.5 inline-flex h-8 w-8 items-center justify-center rounded-full border-2 border-current/50 text-sm font-semibold bg-white/10 hover:bg-white/20 hover:border-current/70 transition-colors"
          aria-label="Learn more about this situation"
          data-testid="hud-weird-state-help"
        >
          ?
        </button>
      )}
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
function SubPhaseDetails({ detail }: { detail?: string | undefined }) {
  if (!detail) return null;
  return <div className="text-sm text-gray-600 mt-1">{detail}</div>;
}

/**
 * LPS (Last-Player-Standing) tracking indicator.
 * Shows round counter and progress toward LPS victory when a player has consecutive exclusive rounds.
 * Per RR-CANON-R172, LPS requires 3 consecutive rounds where only 1 player has real actions.
 */
function LpsTrackingIndicator({
  lpsTracking,
  players,
}: {
  lpsTracking?: {
    roundIndex: number;
    consecutiveExclusiveRounds: number;
    consecutiveExclusivePlayer: number | null;
  };
  players: PlayerViewModel[];
}) {
  // Only show when there's progress toward LPS (at least 1 consecutive exclusive round)
  if (!lpsTracking || lpsTracking.consecutiveExclusiveRounds < 1) {
    return null;
  }

  const { consecutiveExclusiveRounds, consecutiveExclusivePlayer } = lpsTracking;
  const exclusivePlayer = players.find((p) => p.playerNumber === consecutiveExclusivePlayer);
  const playerName = exclusivePlayer?.username ?? `Player ${consecutiveExclusivePlayer}`;

  // Color progression: amber (1-2), red (3 = victory imminent)
  const isVictoryImminent = consecutiveExclusiveRounds >= 3;
  const colorClass = isVictoryImminent
    ? 'border-red-400/60 bg-gradient-to-r from-red-950/80 to-red-900/60 text-red-50 shadow-lg shadow-red-900/30'
    : 'border-amber-400/60 bg-gradient-to-r from-amber-950/80 to-amber-900/60 text-amber-50 shadow-md shadow-amber-900/20';

  const roundsLeft = Math.max(0, 3 - consecutiveExclusiveRounds);
  const statusText = isVictoryImminent
    ? 'LPS Victory!'
    : `${roundsLeft} round${roundsLeft !== 1 ? 's' : ''} until LPS`;

  return (
    <div
      className={`mb-3 px-4 py-3 rounded-xl border text-xs flex items-center gap-3 transition-all duration-300 ${colorClass} ${
        isVictoryImminent ? 'animate-pulse' : ''
      }`}
      role="status"
      aria-live="polite"
      data-testid="hud-lps-indicator"
    >
      <span
        className={`text-xl p-1.5 rounded-lg ${isVictoryImminent ? 'bg-red-900/50' : 'bg-amber-900/50'}`}
        aria-hidden="true"
      >
        🏆
      </span>
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-sm">{playerName} has exclusive actions</div>
        <div className="text-[11px] opacity-90 mt-0.5 flex items-center gap-2">
          <span>{statusText}</span>
          <span className="text-[10px] opacity-70">•</span>
          <span className="font-mono">Round {consecutiveExclusiveRounds}/3</span>
        </div>
      </div>
      {/* Progress dots */}
      <div className="flex gap-1.5" aria-label={`${consecutiveExclusiveRounds} of 3 rounds`}>
        {[1, 2, 3].map((n) => (
          <span
            key={n}
            className={`w-3 h-3 rounded-full transition-all duration-300 ${
              n <= consecutiveExclusiveRounds
                ? 'bg-current shadow-sm'
                : 'bg-current/25 border border-current/30'
            }`}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Victory progress indicator showing ring elimination and territory control progress.
 * Per RR-CANON-R061: victoryThreshold = round(ringsPerPlayer × (2/3 + 1/3 × (numPlayers - 1)))
 * Per RR-CANON-R062-v2: Territory victory requires BOTH:
 *   - territoryMinimum = floor(totalSpaces/numPlayers)+1 AND
 *   - territory > sum of all opponents' territory
 */
function VictoryProgressIndicator({
  victoryProgress,
  players,
}: {
  victoryProgress?: HUDViewModel['victoryProgress'];
  players: PlayerViewModel[];
}) {
  // Only show when there's meaningful progress (at least 20% toward any goal)
  if (!victoryProgress) return null;

  const { ringElimination, territory } = victoryProgress;
  const ringLeader = ringElimination.leader;
  const territoryLeader = territory.leader;

  // Only show if someone has made progress
  const showRings = ringLeader && ringLeader.percentage >= 20;
  const showTerritory = territoryLeader && territoryLeader.percentage >= 20;

  if (!showRings && !showTerritory) return null;

  const getPlayerName = (playerNumber: number) => {
    const player = players.find((p) => p.playerNumber === playerNumber);
    return player?.username ?? `P${playerNumber}`;
  };

  return (
    <div
      className="mb-2 px-3 py-2 rounded-lg border border-slate-600/50 bg-slate-900/60 text-xs"
      data-testid="hud-victory-progress"
    >
      <div className="text-[10px] uppercase tracking-wide text-slate-400 mb-1">
        Victory Progress
      </div>
      <div className="space-y-1.5">
        {showRings && ringLeader && (
          <div className="flex items-center gap-2">
            <span className="text-slate-400 w-16 shrink-0">Rings:</span>
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-rose-500 transition-all duration-300"
                style={{ width: `${Math.min(ringLeader.percentage, 100)}%` }}
              />
            </div>
            <span className="text-slate-300 text-[10px] w-20 text-right">
              {getPlayerName(ringLeader.playerNumber)}: {ringLeader.eliminated}/
              {ringElimination.threshold}
            </span>
          </div>
        )}
        {showTerritory && territoryLeader && (
          <div className="flex items-center gap-2">
            <span className="text-slate-400 w-16 shrink-0">Territory:</span>
            <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-300"
                style={{ width: `${Math.min(territoryLeader.percentage, 100)}%` }}
              />
            </div>
            <span className="text-slate-300 text-[10px] w-20 text-right">
              {getPlayerName(territoryLeader.playerNumber)}: {territoryLeader.spaces}/
              {territory.threshold}
            </span>
          </div>
        )}
      </div>
    </div>
  );
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
  // RR-CANON-R022/R145: Territory processing requires entire cap from any controlled stack
  // (including height-1 standalone rings)
  if (gameState.currentPhase === 'territory_processing') {
    return (
      <div className="text-sm text-gray-600 mt-1">
        Processing disconnected regions; you must eliminate the entire cap from any controlled stack
        outside each region (including height-1 standalone rings).
      </div>
    );
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
        <div className="text-gray-500">Rings Eliminated</div>
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
    <div className="text-center py-2 bg-slate-800/60 border border-slate-700 rounded">
      <div className="text-2xl font-bold text-slate-100">{turnNumber}</div>
      <div className="text-xs text-slate-400">Turn</div>
      {moveNumber > 0 && <div className="text-xs text-slate-500">Move #{moveNumber}</div>}
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
  // 1 → Random, 2 → Heuristic, 3–4 → Minimax, 5–6 → Descent, 7–8 → MCTS, 9–10 → Gumbel MCTS.
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
  if (difficulty >= 3 && difficulty <= 4) {
    return {
      label: 'Advanced · Minimax',
      color: 'text-blue-300',
      bgColor: 'bg-blue-900/40',
    };
  }
  if (difficulty >= 5 && difficulty <= 6) {
    return {
      label: 'Tough · Descent',
      color: 'text-yellow-300',
      bgColor: 'bg-yellow-900/40',
    };
  }
  if (difficulty >= 7 && difficulty <= 8) {
    return {
      label: 'Expert · MCTS',
      color: 'text-purple-300',
      bgColor: 'bg-purple-900/40',
    };
  }
  if (difficulty === 9) {
    return {
      label: 'Master · Gumbel MCTS',
      color: 'text-red-300',
      bgColor: 'bg-red-900/40',
    };
  }
  // 10
  return {
    label: 'Grandmaster · Gumbel MCTS',
    color: 'text-red-400',
    bgColor: 'bg-red-900/50',
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
    case 'gumbel_mcts':
      return 'Gumbel MCTS';
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
    <div className="grid grid-cols-3 gap-2 text-xs mt-3 bg-slate-900/50 rounded-lg p-2 border border-slate-700/50">
      <div className="text-center">
        <div className="font-bold text-lg text-slate-200">{stats.inHand}</div>
        <Tooltip content="Rings not yet placed on the board. Place these during the ring placement phase to build your stacks.">
          <span className="text-[10px] text-slate-400 cursor-help border-b border-dotted border-slate-600 hover:text-slate-300 transition-colors">
            In Hand
          </span>
        </Tooltip>
      </div>
      <div className="text-center border-x border-slate-700/50">
        <div className="font-bold text-lg text-slate-200">{stats.onBoard}</div>
        <Tooltip content="Rings currently in your stacks on the board. These can be moved, captured, or eliminated.">
          <span className="text-[10px] text-slate-400 cursor-help border-b border-dotted border-slate-600 hover:text-slate-300 transition-colors">
            On Board
          </span>
        </Tooltip>
      </div>
      <div className="text-center">
        <div className="font-bold text-lg text-red-400">{stats.eliminated}</div>
        <Tooltip content="Rings permanently removed from play. Each elimination counts toward Ring Elimination victory!">
          <span className="text-[10px] text-slate-400 cursor-help border-b border-dotted border-slate-600 hover:text-slate-300 transition-colors">
            Eliminated
          </span>
        </Tooltip>
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
  timeControl?: TimeControl | undefined;
}) {
  // Build comprehensive aria-label for the player card
  const playerStatusParts: string[] = [player.username];
  if (player.isUserPlayer) {
    playerStatusParts.push('(you)');
  }
  if (player.aiInfo.isAI) {
    playerStatusParts.push(`AI ${player.aiInfo.difficultyLabel}`);
  }
  if (player.isCurrentPlayer) {
    playerStatusParts.push('- current turn');
  }
  playerStatusParts.push(`- ${player.ringStats.inHand} rings in hand`);
  playerStatusParts.push(`${player.ringStats.onBoard} on board`);
  playerStatusParts.push(`${player.ringStats.eliminated} eliminated`);
  if (player.territorySpaces > 0) {
    playerStatusParts.push(`${player.territorySpaces} territory spaces`);
  }

  const playerAriaLabel = playerStatusParts.join(' ');

  // Current player gets a more prominent visual treatment
  const isActivePlayer = player.isCurrentPlayer;
  const cardBaseClasses = isActivePlayer
    ? 'bg-gradient-to-br from-blue-900/80 to-slate-900/80 border-blue-400/70 shadow-lg shadow-blue-500/20'
    : 'bg-slate-800/60 border-slate-700 hover:border-slate-600';

  const userRingClasses = player.isUserPlayer
    ? 'ring-2 ring-green-400/70 ring-offset-1 ring-offset-slate-900'
    : '';

  return (
    <article
      className={`
      p-3 rounded-xl border-2 transition-all duration-300 ease-out
      ${cardBaseClasses}
      ${userRingClasses}
    `}
      aria-label={playerAriaLabel}
      data-testid={`player-card-${player.id}`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className={`w-5 h-5 rounded-full ${player.colorClass} shadow-sm ${
              isActivePlayer ? 'ring-2 ring-white/40 animate-pulse' : ''
            }`}
          />
          <span className="font-semibold text-slate-100">{player.username}</span>
          {player.aiInfo.isAI && <Badge>🤖 AI</Badge>}
          {player.isCurrentPlayer && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/90 text-white text-xs font-semibold shadow-sm animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-white" />
              Current Turn
            </span>
          )}
        </div>

        {timeControl && player.timeRemaining !== undefined && (
          <PlayerTimerFromVM
            timeRemaining={player.timeRemaining}
            isActive={player.isCurrentPlayer}
            playerName={player.username}
          />
        )}
      </div>

      {player.aiInfo.isAI && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          <span
            className={`text-[10px] px-2 py-1 rounded-md ${player.aiInfo.difficultyBgColor} ${player.aiInfo.difficultyColor} font-semibold shadow-sm`}
          >
            {player.aiInfo.difficultyLabel} Lv{player.aiInfo.difficulty}
          </span>
          <span className="text-[10px] px-2 py-1 rounded-md bg-slate-700/80 text-slate-300 font-medium">
            {player.aiInfo.aiTypeLabel}
          </span>
        </div>
      )}

      <RingStatsFromVM stats={player.ringStats} />
      {player.territorySpaces > 0 && (
        <div className="text-sm mt-2 text-center text-slate-300 bg-emerald-900/30 rounded-md py-1 border border-emerald-700/40">
          <span className="font-semibold text-emerald-300">{player.territorySpaces}</span>{' '}
          <span className="text-emerald-200/80">
            territory space{player.territorySpaces !== 1 ? 's' : ''}
          </span>
        </div>
      )}
    </article>
  );
}

/**
 * Simplified timer for view model usage
 */
function PlayerTimerFromVM({
  timeRemaining,
  isActive,
  playerName,
}: {
  timeRemaining: number;
  isActive: boolean;
  playerName: string;
}) {
  // Use a ref to track when the countdown started and the initial value
  const countdownStartRef = useRef<{ startTime: number; startValue: number } | null>(null);
  const [displayTime, setDisplayTime] = useState(timeRemaining);

  useEffect(() => {
    if (!isActive) {
      // Not this player's turn - sync to server value and clear countdown tracking
      countdownStartRef.current = null;
      setDisplayTime(timeRemaining);
      return;
    }

    // Start or continue countdown
    if (!countdownStartRef.current) {
      // Initialize countdown tracking when turn starts
      countdownStartRef.current = {
        startTime: Date.now(),
        startValue: timeRemaining,
      };
    }

    // Update display immediately based on elapsed time
    const updateDisplay = () => {
      if (!countdownStartRef.current) return;
      const elapsed = Date.now() - countdownStartRef.current.startTime;
      const newValue = Math.max(0, countdownStartRef.current.startValue - elapsed);
      setDisplayTime(newValue);
    };

    // Initial update
    updateDisplay();

    // Use a shorter interval for smoother visual countdown
    const interval = setInterval(updateDisplay, 200);

    return () => clearInterval(interval);
  }, [isActive, timeRemaining]);

  // Sync with server when timeRemaining changes (e.g., after move with increment)
  useEffect(() => {
    if (isActive && countdownStartRef.current) {
      // Server sent a new timeRemaining (likely after a move with increment)
      // Reset our countdown tracking to use the new server value
      countdownStartRef.current = {
        startTime: Date.now(),
        startValue: timeRemaining,
      };
      setDisplayTime(timeRemaining);
    }
  }, [timeRemaining, isActive]);

  const minutes = Math.floor(displayTime / 60000);
  const seconds = Math.floor((displayTime % 60000) / 1000);

  const severity = getCountdownSeverity(displayTime);

  // High contrast for visibility on dark HUD background
  let timerClass = 'font-mono text-white font-semibold text-sm';
  if (severity === 'warning') {
    timerClass = 'font-mono text-amber-300 font-bold text-sm';
  } else if (severity === 'critical') {
    timerClass = 'font-mono text-red-400 font-bold animate-pulse text-sm';
  }

  const clockLabel = `${minutes}:${seconds.toString().padStart(2, '0')}`;
  const ariaLabel = `${playerName} time remaining: ${clockLabel}`;

  return (
    <div className={timerClass} aria-label={ariaLabel} data-severity={severity ?? undefined}>
      {clockLabel}
    </div>
  );
}

/**
 * Compact timer display for the header row (P1: 9:58, P2: 9:34)
 * Uses countdown logic when it's this player's turn.
 */
function CompactPlayerTimer({
  playerNumber,
  timeRemaining,
  isActive,
}: {
  playerNumber: number;
  timeRemaining: number;
  isActive: boolean;
}) {
  const countdownStartRef = useRef<{ startTime: number; startValue: number } | null>(null);
  const [displayTime, setDisplayTime] = useState(timeRemaining);

  useEffect(() => {
    if (!isActive) {
      countdownStartRef.current = null;
      setDisplayTime(timeRemaining);
      return;
    }

    if (!countdownStartRef.current) {
      countdownStartRef.current = {
        startTime: Date.now(),
        startValue: timeRemaining,
      };
    }

    const updateDisplay = () => {
      if (!countdownStartRef.current) return;
      const elapsed = Date.now() - countdownStartRef.current.startTime;
      const newValue = Math.max(0, countdownStartRef.current.startValue - elapsed);
      setDisplayTime(newValue);
    };

    updateDisplay();
    const interval = setInterval(updateDisplay, 200);
    return () => clearInterval(interval);
  }, [isActive, timeRemaining]);

  // Sync with server when timeRemaining changes
  useEffect(() => {
    if (isActive && countdownStartRef.current) {
      countdownStartRef.current = {
        startTime: Date.now(),
        startValue: timeRemaining,
      };
      setDisplayTime(timeRemaining);
    }
  }, [timeRemaining, isActive]);

  const minutes = Math.floor(displayTime / 60000);
  const seconds = Math.floor((displayTime % 60000) / 1000);
  const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
  const isLow = displayTime < 60000;
  const isCritical = displayTime < 30000;

  return (
    <span
      className={`font-mono ${
        isActive
          ? isCritical
            ? 'text-red-400 font-bold animate-pulse'
            : isLow
              ? 'text-amber-400 font-bold'
              : 'text-slate-100 font-semibold'
          : 'text-slate-500 text-[10px]'
      }`}
      title={`Player ${playerNumber}${isActive ? ' (current turn)' : ''}`}
    >
      P{playerNumber}: {timeStr}
      {isActive && <span className="ml-0.5 text-green-400">⏱</span>}
    </span>
  );
}

/**
 * Format a compact time control summary for display in the HUD.
 * Examples: "Rapid • 5+0", "Blitz • 3+2".
 */
function formatTimeControlSummary(timeControl?: TimeControl | undefined): string | null {
  if (!timeControl) return null;

  const { initialTime, increment } = timeControl;

  if (typeof initialTime !== 'number' || initialTime <= 0) {
    return null;
  }

  const minutes = Math.round(initialTime / 60);
  const incrementSeconds = typeof increment === 'number' && increment >= 0 ? increment : 0;

  // TimeControl.type is typically a friendly enum, but we treat it as optional
  // so the HUD does not depend on any particular backend schema.
  const rawType = (timeControl as { type?: string }).type;
  let typeLabel: string;
  switch (rawType) {
    case 'blitz':
      typeLabel = 'Blitz';
      break;
    case 'rapid':
      typeLabel = 'Rapid';
      break;
    case 'classical':
      typeLabel = 'Classical';
      break;
    default:
      typeLabel = 'Time';
      break;
  }

  return `${typeLabel} • ${minutes}+${incrementSeconds}`;
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
    const {
      viewModel,
      timeControl,
      onShowBoardControls,
      hideVictoryConditions,
      isLocalSandboxOnly,
      rulesUxContext,
    } = props;

    return (
      <GameHUDFromViewModel
        viewModel={viewModel}
        {...(timeControl !== undefined ? { timeControl } : {})}
        {...(onShowBoardControls ? { onShowBoardControls } : {})}
        {...(hideVictoryConditions !== undefined ? { hideVictoryConditions } : {})}
        {...(isLocalSandboxOnly !== undefined ? { isLocalSandboxOnly } : {})}
        {...(rulesUxContext ? { rulesUxContext } : {})}
      />
    );
  }

  // Legacy rendering path
  return <GameHUDLegacy {...props} />;
}

/**
  /**
   * Standalone Victory conditions panel - can be rendered outside the HUD
   */
export function VictoryConditionsPanel({ className = '' }: { className?: string }) {
  return (
    <div
      className={`px-4 py-3 bg-gradient-to-br from-slate-800/80 to-slate-900/80 border border-slate-700/80 rounded-xl text-[11px] text-slate-300 leading-snug shadow-sm ${className}`}
      data-testid="victory-conditions-help"
    >
      <div className="font-semibold text-slate-100 mb-2 text-xs uppercase tracking-wider flex items-center gap-2">
        <span className="text-amber-400">🏆</span>
        Victory Conditions
      </div>

      <div className="space-y-2">
        <div className="flex items-start gap-2 p-2 rounded-lg bg-rose-950/30 border border-rose-900/40 hover:border-rose-800/50 transition-colors">
          <span className="text-rose-400 text-sm mt-0.5">⚔️</span>
          <div className="flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-rose-200">Ring Elimination</span>
              <Tooltip
                content={
                  'How to eliminate rings:\n• Move onto a marker to remove it as a ring\n• Choose "full bonus" when you score a line\n• Claim territory (costs the cap from an eligible stack)\n• Forced elimination when you have no moves\n\nThe number of rings you need depends on player count. Check your progress in the Score Summary above!'
                }
              >
                <span
                  className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-rose-600/50 bg-rose-900/30 text-[9px] text-rose-300 hover:bg-rose-900/50 transition-colors cursor-help"
                  aria-label="Elimination victory details"
                  data-testid="victory-tooltip-elimination-trigger"
                >
                  ?
                </span>
              </Tooltip>
            </div>
            <div className="text-[10px] text-rose-300/70 mt-0.5">
              Eliminate enough opponent rings
            </div>
          </div>
        </div>

        <div className="flex items-start gap-2 p-2 rounded-lg bg-emerald-950/30 border border-emerald-900/40 hover:border-emerald-800/50 transition-colors">
          <span className="text-emerald-400 text-sm mt-0.5">🏰</span>
          <div className="flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-emerald-200">Territory Control</span>
              <Tooltip
                content={
                  'Territory = spaces you permanently control.\n\nTo claim a region:\n1. Isolate a group of spaces by surrounding them\n2. Spend the cap from an eligible stack outside the region\n   (multicolor stacks or stacks taller than 1 ring)\n\nWin by controlling at least your fair share (1/N of board) AND more territory than all opponents combined!'
                }
              >
                <span
                  className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-emerald-600/50 bg-emerald-900/30 text-[9px] text-emerald-300 hover:bg-emerald-900/50 transition-colors cursor-help"
                  aria-label="Territory victory details"
                  data-testid="victory-tooltip-territory-trigger"
                >
                  ?
                </span>
              </Tooltip>
            </div>
            <div className="text-[10px] text-emerald-300/70 mt-0.5">
              Dominate the board with your territory
            </div>
          </div>
        </div>

        <div className="flex items-start gap-2 p-2 rounded-lg bg-purple-950/30 border border-purple-900/40 hover:border-purple-800/50 transition-colors">
          <span className="text-purple-400 text-sm mt-0.5">👑</span>
          <div className="flex-1">
            <div className="flex items-center gap-1.5">
              <span className="font-medium text-purple-200">Last Player Standing</span>
              <Tooltip
                content={
                  'If you are the only player who can actually move for 3 rounds in a row, you win!\n\nWhat counts as a "real move":\n• Placing rings\n• Moving stacks\n• Capturing opponents\n\nIf opponents are stuck and can only do forced eliminations, that doesn\'t count as a real move for them.'
                }
              >
                <span
                  className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-purple-600/50 bg-purple-900/30 text-[9px] text-purple-300 hover:bg-purple-900/50 transition-colors cursor-help"
                  aria-label="Last Player Standing victory details"
                  data-testid="victory-tooltip-last-player-standing-trigger"
                >
                  ?
                </span>
              </Tooltip>
            </div>
            <div className="text-[10px] text-purple-300/70 mt-0.5">
              Be the only active player for 3 rounds
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Compact per-player score summary highlighting rings captured and territory.
 * Uses the shared PlayerViewModel so backend and sandbox hosts share the same
 * representation.
 */
function CompactScoreSummary({ players }: { players: PlayerViewModel[] }) {
  if (!players || players.length === 0) return null;

  // Build aria-label summarizing all player scores
  const scoreSummaryParts = players.map((player) => {
    const name = player.isUserPlayer ? 'You' : player.username;
    return `${name}: ${player.ringStats.eliminated} rings eliminated, ${player.territorySpaces} territory`;
  });
  const scoreSummaryAriaLabel = `Score summary. ${scoreSummaryParts.join('. ')}`;

  return (
    <div
      className="mt-3 px-3 py-3 bg-gradient-to-br from-slate-900/90 to-slate-800/70 border border-slate-700/80 rounded-xl text-[11px] text-slate-200 shadow-sm"
      role="region"
      aria-label={scoreSummaryAriaLabel}
      data-testid="hud-score-summary"
    >
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
        <span className="text-slate-500">📊</span>
        Score Summary
      </div>
      <div className="grid grid-cols-1 gap-2">
        {players.map((player) => (
          <div
            key={player.id}
            className={`flex items-center justify-between gap-2 p-2 rounded-lg transition-colors ${
              player.isCurrentPlayer
                ? 'bg-blue-900/30 border border-blue-700/40'
                : 'bg-slate-800/40 border border-slate-700/30 hover:bg-slate-800/60'
            }`}
          >
            <div className="flex items-center gap-2 min-w-0">
              <span
                className={`inline-block w-3 h-3 rounded-full ${player.colorClass} shadow-sm ${
                  player.isCurrentPlayer ? 'ring-2 ring-white/30' : ''
                }`}
                aria-hidden="true"
              />
              <span className="truncate">
                <span className="font-semibold text-slate-100">
                  {player.isUserPlayer ? 'You' : player.username}
                </span>
                {player.isCurrentPlayer && (
                  <span className="ml-1.5 inline-flex items-center gap-0.5 rounded-full bg-blue-500/30 px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-blue-200 border border-blue-400/30">
                    <span className="w-1 h-1 rounded-full bg-blue-300 animate-pulse" />
                    Turn
                  </span>
                )}
              </span>
            </div>
            <div className="flex items-center gap-3 text-[10px]">
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-rose-950/40 border border-rose-800/30">
                <span className="text-rose-400">⚔</span>
                <span className="font-mono font-medium text-rose-200">
                  {player.ringStats.eliminated}/{player.ringStats.total}
                </span>
              </span>
              <span className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald-950/40 border border-emerald-800/30">
                <span className="text-emerald-400">🏰</span>
                <span className="font-mono font-medium text-emerald-200">
                  {player.territorySpaces}
                </span>
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
/**
 * View model-based HUD implementation
 */
function GameHUDFromViewModel({
  viewModel,
  timeControl,
  onShowBoardControls,
  hideVictoryConditions = false,
  isLocalSandboxOnly = false,
  rulesUxContext,
}: {
  viewModel: HUDViewModel;
  timeControl?: TimeControl | undefined;
  onShowBoardControls?: (() => void) | undefined;
  hideVictoryConditions?: boolean;
  isLocalSandboxOnly?: boolean;
  rulesUxContext?: GameHUDViewModelProps['rulesUxContext'];
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
    weirdState,
    lpsTracking,
    victoryProgress,
  } = viewModel;

  const rulesUxBoardType = rulesUxContext?.boardType;
  const rulesUxNumPlayers = rulesUxContext?.numPlayers ?? players.length;
  const rulesUxAiDifficulty = rulesUxContext?.aiDifficulty;
  const rulesUxRulesConcept = rulesUxContext?.rulesConcept;
  const rulesUxScenarioId = rulesUxContext?.scenarioId;
  const surfaceableWeirdState =
    weirdState && isSurfaceableWeirdStateType(weirdState.type as RulesUxWeirdStateType)
      ? weirdState
      : undefined;

  const timeControlSummary = formatTimeControlSummary(timeControl);

  // Derive isMyTurn: true if the current user is both the active player and it's their turn
  const isMyTurn = players.some((p) => p.isUserPlayer && p.isCurrentPlayer);

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

  const decisionSeverity =
    decisionPhase && decisionPhase.showCountdown && decisionPhase.timeRemainingMs !== null
      ? getCountdownSeverity(decisionPhase.timeRemainingMs)
      : null;

  const { currentTopic, isOpen, showTopic, hideTopic } = useTeachingOverlay();

  const helpOpenCountsRef = React.useRef<Record<string, number>>({});
  const helpSessionIdRef = React.useRef<string | null>(null);
  const prevIsOpenRef = React.useRef(false);
  const prevTopicRef = React.useRef<TeachingTopic | null>(null);
  const autoScenarioHelpShownRef = React.useRef<Record<string, boolean>>({});
  const weirdStateOverlaySessionRef = React.useRef<string | null>(null);
  const lastWeirdStateTypeRef = React.useRef<RulesUxWeirdStateType | null>(null);
  const teachingWeirdStateContextRef = React.useRef<WeirdStateOverlayContext | null>(null);
  const phaseHelpTopic: TeachingTopic | null = React.useMemo(() => {
    switch (phase.phaseKey) {
      case 'movement':
        return 'stack_movement';
      case 'capture':
        return 'capturing';
      case 'chain_capture':
        return 'chain_capture';
      case 'line_processing':
        return 'line_bonus';
      default:
        return null;
    }
  }, [phase.phaseKey]);

  // Map curated scenario rulesConcepts to TeachingOverlay topics and the
  // phases where it makes sense to auto-open contextual help.
  const scenarioHelpConfig = React.useMemo(() => {
    if (!rulesUxRulesConcept) return null;

    type PhaseKey = typeof phase.phaseKey;
    type Config = { topic: TeachingTopic; phaseKeys: PhaseKey[] };

    const concept = rulesUxRulesConcept;

    if (concept === 'capture_basic') {
      const cfg: Config = { topic: 'capturing', phaseKeys: ['movement', 'capture'] as PhaseKey[] };
      return cfg;
    }

    if (concept === 'chain_capture_mandatory') {
      const cfg: Config = { topic: 'chain_capture', phaseKeys: ['chain_capture'] as PhaseKey[] };
      return cfg;
    }

    if (concept === 'lines_basic' || concept === 'lines_overlength_option2') {
      const cfg: Config = { topic: 'line_bonus', phaseKeys: ['line_processing'] as PhaseKey[] };
      return cfg;
    }

    if (
      concept === 'territory_basic' ||
      concept === 'territory_near_victory' ||
      concept === 'territory_mini_region_q23'
    ) {
      const cfg: Config = {
        topic: 'territory',
        phaseKeys: ['territory_processing'] as PhaseKey[],
      };
      return cfg;
    }

    if (concept === 'movement_basic' || concept === 'stack_height_mobility') {
      const cfg: Config = { topic: 'stack_movement', phaseKeys: ['movement'] as PhaseKey[] };
      return cfg;
    }

    return null;
  }, [phase.phaseKey, rulesUxRulesConcept]);

  // Emit telemetry when the contextual TeachingOverlay opens for a topic.
  React.useEffect(() => {
    const prevIsOpen = prevIsOpenRef.current;
    const prevTopic = prevTopicRef.current;

    prevIsOpenRef.current = isOpen;
    prevTopicRef.current = currentTopic;

    if (!isOpen || !currentTopic) {
      return;
    }

    // Detect overlay opening for a topic (either from closed or when the topic changes).
    const isNewOpen = !prevIsOpen || prevTopic !== currentTopic;
    if (!isNewOpen) {
      return;
    }

    if (!rulesUxBoardType) {
      // Without board context we skip emitting telemetry entirely for this HUD.
      return;
    }

    const key = currentTopic;
    const counts = helpOpenCountsRef.current;
    const newCount = (counts[key] ?? 0) + 1;
    counts[key] = newCount;

    const legacyBaseEvent = {
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(currentTopic ? { topic: currentTopic } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
    } as const;

    if (!helpSessionIdRef.current) {
      helpSessionIdRef.current = newHelpSessionId();
    }
    const helpSessionId = helpSessionIdRef.current;

    // Spec-aligned help_open event for rules telemetry dashboards.
    void logHelpOpenEvent({
      ...legacyBaseEvent,
      source: 'hud',
      entrypoint: 'hud_help_chip',
      helpSessionId,
    });

    // Legacy metrics events kept for backwards compatibility.
    void sendRulesUxEvent({
      type: 'rules_help_open',
      ...legacyBaseEvent,
    });

    if (newCount >= 2) {
      void sendRulesUxEvent({
        type: 'rules_help_repeat',
        ...legacyBaseEvent,
        repeatCount: newCount,
      });
    }
  }, [
    isOpen,
    currentTopic,
    rulesUxBoardType,
    rulesUxNumPlayers,
    rulesUxAiDifficulty,
    rulesUxRulesConcept,
    rulesUxScenarioId,
  ]);
  // Auto-open curated TeachingOverlay topics for selected onboarding / rules
  // scenarios the first time their key phase becomes active. This uses the
  // same TeachingOverlay surface as the explicit help chips but avoids
  // re-opening on every render or phase change.
  React.useEffect(() => {
    if (!scenarioHelpConfig || !rulesUxScenarioId) {
      return;
    }

    if (!scenarioHelpConfig.phaseKeys.includes(phase.phaseKey)) {
      return;
    }

    const topic = scenarioHelpConfig.topic;
    const key = `${rulesUxScenarioId}:${topic}`;

    if (autoScenarioHelpShownRef.current[key]) {
      return;
    }

    autoScenarioHelpShownRef.current[key] = true;
    showTopic(topic);
  }, [phase.phaseKey, rulesUxScenarioId, scenarioHelpConfig, showTopic]);

  // Emit a spec-aligned weird_state_banner_impression when a new weird-state
  // banner becomes active in the HUD. This uses the coarse weird-state type
  // from the view model and maps it to a stable reason code and rules_context.
  React.useEffect(() => {
    if (!surfaceableWeirdState || !rulesUxBoardType) {
      lastWeirdStateTypeRef.current = null;
      weirdStateOverlaySessionRef.current = null;
      return;
    }

    const currentType = surfaceableWeirdState.type as RulesUxWeirdStateType;

    if (lastWeirdStateTypeRef.current === currentType && weirdStateOverlaySessionRef.current) {
      return;
    }

    lastWeirdStateTypeRef.current = currentType;

    const { reasonCode, rulesContext, weirdStateType } = getWeirdStateReasonForType(currentType);
    const overlaySessionId = newOverlaySessionId();
    weirdStateOverlaySessionRef.current = overlaySessionId;

    void logRulesUxEvent({
      type: 'weird_state_banner_impression',
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      rulesContext,
      source: 'hud',
      weirdStateType,
      reasonCode,
      overlaySessionId,
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
    });
  }, [
    surfaceableWeirdState,
    rulesUxAiDifficulty,
    rulesUxBoardType,
    rulesUxNumPlayers,
    rulesUxRulesConcept,
    rulesUxScenarioId,
  ]);

  const handleWeirdStateHelp = React.useCallback(() => {
    if (!surfaceableWeirdState) return;

    let topic: TeachingTopic | null = null;

    switch (surfaceableWeirdState.type) {
      case 'active-no-moves-movement':
        topic = 'active_no_moves';
        break;
      case 'forced-elimination':
        topic = 'forced_elimination';
        break;
      case 'structural-stalemate':
        topic = 'victory_stalemate';
        break;
      default:
        topic = null;
        break;
    }

    if (topic) {
      showTopic(topic);
    }

    if (!rulesUxBoardType || !topic) {
      return;
    }

    const weirdStateType = surfaceableWeirdState.type as RulesUxWeirdStateType;
    const { reasonCode, rulesContext } = getWeirdStateReasonForType(weirdStateType);

    let overlaySessionId = weirdStateOverlaySessionRef.current;
    if (!overlaySessionId) {
      overlaySessionId = newOverlaySessionId();
      weirdStateOverlaySessionRef.current = overlaySessionId;
    }

    // Spec-aligned event for weird-state detail opens.
    void logRulesUxEvent({
      type: 'weird_state_details_open',
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      rulesContext,
      source: 'hud',
      weirdStateType,
      reasonCode,
      overlaySessionId,
      topic,
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
    });

    // Legacy event kept for backwards-compatible metrics.
    void sendRulesUxEvent({
      type: 'rules_weird_state_help',
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      weirdStateType,
      topic,
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
    });

    // Provide a narrow weird-state context for TeachingOverlay so it can emit
    // overlay lifecycle telemetry only when opened from this help entrypoint.
    teachingWeirdStateContextRef.current = {
      reasonCode,
      rulesContext,
      weirdStateType,
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      ...(isLocalSandboxOnly ? { isSandbox: true } : {}),
      overlaySessionId,
    };
  }, [
    isLocalSandboxOnly,
    rulesUxAiDifficulty,
    rulesUxBoardType,
    rulesUxNumPlayers,
    rulesUxRulesConcept,
    rulesUxScenarioId,
    showTopic,
    surfaceableWeirdState,
  ]);

  const handleTerritoryHelp = React.useCallback(() => {
    // Territory-processing decisions often confuse new players; surface the
    // dedicated TeachingOverlay topic for a concise rules recap.
    showTopic('territory');
  }, [showTopic]);

  // Clear any stale weird-state overlay context once the TeachingOverlay
  // has fully closed so future non-weird-state help uses do not emit
  // overlay lifecycle telemetry.
  React.useEffect(() => {
    if (!isOpen) {
      teachingWeirdStateContextRef.current = null;
    }
  }, [isOpen]);

  function formatMsAsClock(ms: number): string {
    const totalSeconds = Math.max(0, Math.floor(ms / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  return (
    <div className="w-full max-w-4xl mx-auto mb-4" data-testid="game-hud">
      {/* Local sandbox-only banner: surfaced by SandboxGameHost when backend
          game creation is skipped (e.g. unauthenticated users). */}
      {isLocalSandboxOnly && (
        <div
          className="mb-3 px-3 py-2 rounded-lg bg-slate-900/70 border border-slate-600 flex items-center justify-between text-xs text-slate-200"
          role="status"
          aria-live="polite"
          data-testid="sandbox-local-only-banner"
        >
          <div className="flex items-center gap-2">
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-[11px] text-slate-100 border border-slate-500">
              !
            </span>
            <div className="flex flex-col">
              <span className="font-semibold">
                You&apos;re not logged in; this game runs as a local sandbox only.
              </span>
              <span className="text-[11px] text-slate-300/80">
                Moves, AI, and analysis stay in this browser session and are not saved to your
                account.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Spectator mode banner - prominent indicator when spectating */}
      {isSpectator && (
        <div
          className="mb-3 px-3 py-2 rounded-lg bg-purple-900/30 border border-purple-500/40 flex items-center justify-between"
          role="status"
          aria-live="polite"
          aria-label="Spectator Mode - You are watching this game"
        >
          <div className="flex items-center gap-2 text-purple-100">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-4 h-4"
              aria-hidden="true"
            >
              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
              <path
                fillRule="evenodd"
                d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm font-semibold">Spectator Mode</span>
            <span className="text-xs text-purple-200/80">You are watching this game</span>
          </div>
          {spectatorCount > 0 && (
            <span className="text-xs text-purple-200/70">
              {spectatorCount} {spectatorCount === 1 ? 'viewer' : 'viewers'} total
            </span>
          )}
        </div>
      )}

      {/* Connection Status - hidden in local sandbox since always connected */}
      {!isLocalSandboxOnly && (
        <div className="flex items-center justify-between text-xs text-slate-300 mb-3">
          <div className={`font-semibold ${connectionColor}`}>
            Connection: {connectionLabel}
            {isConnectionStale && (
              <span className="ml-1 text-[11px] text-amber-200">
                (no recent updates from server)
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {spectatorCount > 0 && !isSpectator && (
              <span
                className="text-[11px] text-slate-400 flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800/60 border border-slate-700"
                title={`${spectatorCount} ${spectatorCount === 1 ? 'person' : 'people'} watching`}
                role="status"
                aria-label={`${spectatorCount} spectator${spectatorCount === 1 ? '' : 's'}`}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                  className="w-3 h-3"
                  aria-hidden="true"
                >
                  <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                  <path
                    fillRule="evenodd"
                    d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                    clipRule="evenodd"
                  />
                </svg>
                <span>{spectatorCount} watching</span>
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
      )}

      {/* Time control summary and player times (when clocks are enabled) - min-height prevents layout shift */}
      <div className="min-h-[40px] mb-2">
        {timeControlSummary && (
          <>
            <div className="flex items-center justify-between text-[11px] text-slate-300">
              <span className="text-slate-400">
                Clock:{' '}
                <span className="font-mono text-slate-100" data-testid="hud-time-control-summary">
                  {timeControlSummary}
                </span>
              </span>
            </div>
            {/* Player times row - uses countdown for active player */}
            {players.some((p) => p.timeRemaining !== undefined) && (
              <div className="mt-1 flex items-center gap-2 text-[11px]">
                {players.map((p) => {
                  if (p.timeRemaining === undefined) return null;
                  return (
                    <CompactPlayerTimer
                      key={p.playerNumber}
                      playerNumber={p.playerNumber}
                      timeRemaining={p.timeRemaining}
                      isActive={p.isCurrentPlayer}
                    />
                  );
                })}
              </div>
            )}
          </>
        )}
        {!timeControlSummary && isLocalSandboxOnly && (
          <div className="text-[11px] text-slate-500">Clock: No clock (local sandbox)</div>
        )}
      </div>

      {/* Phase Indicator & weird-state banner */}
      {surfaceableWeirdState && (
        <WeirdStateBanner weirdState={surfaceableWeirdState} onShowHelp={handleWeirdStateHelp} />
      )}
      {/* Dynamic indicators zone - reserve space to prevent layout shifts when indicators appear/disappear */}
      <div className="min-h-[32px]">
        {/* LPS tracking indicator - shows when a player has consecutive exclusive rounds */}
        <LpsTrackingIndicator lpsTracking={lpsTracking} players={players} />
        {/* Victory progress indicator - shows ring elimination and territory progress */}
        <VictoryProgressIndicator victoryProgress={victoryProgress} players={players} />
      </div>
      <PhaseIndicator phase={phase} isMyTurn={isMyTurn} isSpectator={isSpectator} />
      {/* Phase help button zone - min-height prevents layout shift */}
      <div className="min-h-[24px] flex items-end">
        {phaseHelpTopic && (
          <div className="mt-1 flex items-center text-[11px] text-slate-300">
            <button
              type="button"
              onClick={() => showTopic(phaseHelpTopic)}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-slate-600 bg-slate-900/80 hover:border-slate-300 hover:text-slate-50 transition-colors"
              aria-label="Learn more about this phase"
              data-testid={`hud-phase-help-${phase.phaseKey}`}
            >
              <span className="text-xs">?</span>
              <span>Phase rules</span>
            </button>
          </div>
        )}
      </div>
      <SubPhaseDetails detail={subPhaseDetail} />

      {/* Dynamic alerts zone - min-height and transitions prevent jarring layout shift when chips appear/disappear */}
      <div className="min-h-[96px] flex flex-col justify-end transition-all duration-150 ease-in-out">
        {/* High-level decision time-pressure cue for the current phase */}
        {decisionPhase && decisionSeverity && (
          <div className="mt-1 flex items-center text-[11px] text-slate-200">
            <span
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border ${
                decisionSeverity === 'critical'
                  ? 'bg-red-900/80 border-red-400/80 text-red-50 animate-pulse'
                  : decisionSeverity === 'warning'
                    ? 'bg-amber-900/80 border-amber-400/80 text-amber-50'
                    : 'bg-emerald-900/70 border-emerald-400/80 text-emerald-50'
              }`}
              data-testid="hud-decision-time-pressure"
              data-severity={decisionSeverity}
            >
              <span aria-hidden="true" className="text-xs">
                ⏱
              </span>
              <span className="truncate">
                {decisionPhase.isLocalActor
                  ? `Your decision timer: ${formatMsAsClock(decisionPhase.timeRemainingMs ?? 0)}`
                  : `Time left for ${decisionPhase.actingPlayerName}'s decision: ${formatMsAsClock(
                      decisionPhase.timeRemainingMs ?? 0
                    )}`}
              </span>
            </span>
          </div>
        )}

        {/* Decision-specific status chip (e.g. ring elimination prompt) */}
        {decisionPhase?.statusChip && (
          <div className="mt-2 inline-flex items-center gap-2">
            <span
              className={
                decisionPhase.statusChip.tone === 'attention'
                  ? 'px-3 py-1 rounded-full bg-amber-500 text-slate-950 text-[11px] font-semibold shadow-sm shadow-amber-500/40 border border-amber-300'
                  : 'px-3 py-1 rounded-full bg-sky-900/60 text-sky-100 text-[11px] font-medium border border-sky-500/60'
              }
              data-testid="hud-decision-status-chip"
            >
              {decisionPhase.statusChip.text}
            </span>
            {decisionPhase.canSkip && (
              <span
                className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 text-[10px] uppercase tracking-wide text-slate-200"
                data-testid="hud-decision-skip-hint"
              >
                Skip available
              </span>
            )}
            {phase.phaseKey === 'territory_processing' && (
              <button
                type="button"
                onClick={handleTerritoryHelp}
                className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-slate-600 bg-slate-900/80 text-[11px] text-slate-200 hover:border-slate-300 hover:text-slate-50 transition-colors"
                aria-label="Learn more about territory processing"
                data-testid="hud-territory-help"
              >
                ?
              </button>
            )}
          </div>
        )}

        {pieRuleSummary && (
          <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-900/60 border border-amber-500/60 text-[11px] text-amber-100">
            <span className="font-semibold uppercase tracking-wide">Pie rule</span>
            <span className="text-amber-100/90">{pieRuleSummary}</span>
          </div>
        )}
      </div>

      {/* Game Progress */}
      <div className="mt-3">
        <div className="text-center py-3 rounded-xl border border-slate-700/80 bg-gradient-to-b from-slate-800/80 to-slate-900/90 shadow-inner">
          <div className="text-3xl font-bold text-slate-100 tracking-tight">{turnNumber}</div>
          <Tooltip
            content={
              'Turn counts completed full cycles around the table.\nMove # counts individual actions within those turns.'
            }
          >
            <span className="inline-flex items-center justify-center gap-1.5 text-xs text-slate-400 mt-0.5">
              <span className="uppercase tracking-wider font-medium">Turn</span>
              <span
                aria-hidden="true"
                className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-600 bg-slate-800 text-[10px] text-slate-300 hover:border-slate-500 hover:text-slate-200 transition-colors cursor-help"
              >
                ?
              </span>
            </span>
          </Tooltip>
          {moveNumber > 0 && (
            <div className="text-[11px] text-slate-500 mt-0.5 font-mono">Move #{moveNumber}</div>
          )}
        </div>
      </div>

      {/* Score summary */}
      <CompactScoreSummary players={players} />

      {/* Instruction Banner */}
      {instruction && (
        <div className="mt-3 px-4 py-3 bg-gradient-to-r from-amber-900/40 via-amber-900/30 to-amber-900/40 border border-amber-600/40 rounded-xl text-center shadow-sm">
          <span className="text-amber-100 font-medium text-sm">{instruction}</span>
        </div>
      )}

      {/* Decision Phase Banner */}
      {decisionPhase && decisionPhase.isActive && <DecisionPhaseBanner vm={decisionPhase} />}

      {/* Victory Conditions Helper - can be hidden when rendered elsewhere */}
      {!hideVictoryConditions && <VictoryConditionsPanel className="mt-3" />}

      {/* Player Cards */}
      <div className="mt-4 space-y-3">
        {players.map((player) => (
          <PlayerCardFromVM key={player.id} player={player} timeControl={timeControl} />
        ))}
      </div>

      {/* Contextual rules help for weird states and other mechanics */}
      {currentTopic && (
        <TeachingOverlay
          topic={currentTopic}
          isOpen={isOpen}
          onClose={hideTopic}
          position="bottom-right"
          weirdStateOverlayContext={teachingWeirdStateContextRef.current}
        />
      )}
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
      {/* Spectator mode banner - prominent indicator when spectating */}
      {isSpectator && (
        <div
          className="mb-3 px-3 py-2 rounded-lg bg-purple-900/30 border border-purple-500/40 flex items-center justify-between"
          role="status"
          aria-live="polite"
        >
          <div className="flex items-center gap-2 text-purple-100">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="w-4 h-4"
              aria-hidden="true"
            >
              <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
              <path
                fillRule="evenodd"
                d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm font-semibold">Spectator Mode</span>
            <span className="text-xs text-purple-200/80">You are watching this game</span>
          </div>
          {spectatorCount > 0 && (
            <span className="text-xs text-purple-200/70">
              {spectatorCount} {spectatorCount === 1 ? 'viewer' : 'viewers'} total
            </span>
          )}
        </div>
      )}

      {/* Connection Status */}
      <div className="flex items-center justify-between text-xs text-slate-300 mb-3">
        <div className={`font-semibold ${connectionColor}`}>
          Connection: {connectionLabel()}
          {isHeartbeatStale && (
            <span className="ml-1 text-[11px] text-amber-200">(no recent updates from server)</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {spectatorCount > 0 && !isSpectator && (
            <span
              className="text-[11px] text-slate-400 flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800/60 border border-slate-700"
              title={`${spectatorCount} ${spectatorCount === 1 ? 'person' : 'people'} watching`}
              role="status"
              aria-label={`${spectatorCount} spectator${spectatorCount === 1 ? '' : 's'}`}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 20 20"
                fill="currentColor"
                className="w-3 h-3"
                aria-hidden="true"
              >
                <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
                <path
                  fillRule="evenodd"
                  d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                  clipRule="evenodd"
                />
              </svg>
              <span>{spectatorCount} watching</span>
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
      <VictoryConditionsPanel className="mt-3" />

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
