/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MobileGameHUD Component
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A compact, mobile-optimized HUD that surfaces essential game information
 * without overwhelming small screens. Designed to work alongside the touch
 * controls panel and responsive board.
 *
 * Key differences from full GameHUD:
 * - Single-row phase indicator instead of full-width card
 * - Collapsed player summary (expandable on tap)
 * - Timer and turn info always visible in a sticky header bar
 * - Victory conditions moved to a help button
 */

import React, { useState } from 'react';
import type {
  HUDViewModel,
  PlayerViewModel,
  HUDWeirdStateViewModel,
} from '../adapters/gameViewModels';
import type { TimeControl, BoardType } from '../../shared/types/game';
import { getCountdownSeverity } from '../utils/countdown';
import { Tooltip } from './ui/Tooltip';
import { TeachingOverlay, useTeachingOverlay, type TeachingTopic } from './TeachingOverlay';
import type { RulesUxWeirdStateType } from '../../shared/telemetry/rulesUxEvents';
import { sendRulesUxEvent } from '../utils/rulesUxTelemetry';
import { isSurfaceableWeirdStateType } from '../../shared/engine/weirdStateReasons';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface MobileGameHUDProps {
  viewModel: HUDViewModel;
  timeControl?: TimeControl;
  isLocalSandboxOnly?: boolean;
  onShowBoardControls?: () => void;
  /** Handler for tapping a player card (e.g. to show details) */
  onPlayerTap?: (player: PlayerViewModel) => void;
  /**
   * Optional context used for lightweight rules-UX telemetry. When provided,
   * the mobile HUD will emit privacy-aware telemetry for TeachingOverlay
   * opens and weird-state help interactions.
   */
  rulesUxContext?: {
    boardType: BoardType;
    numPlayers: number;
    aiDifficulty?: number;
    rulesConcept?: string;
    scenarioId?: string;
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Components
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compact banner for unusual rules states (Active–No–Moves, Forced Elimination,
 * structural stalemate) in the mobile HUD.
 */
function MobileWeirdStateBanner({
  weirdState,
  onShowHelp,
}: {
  weirdState: HUDWeirdStateViewModel;
  onShowHelp?: () => void;
}) {
  const toneClasses =
    weirdState.tone === 'critical'
      ? 'border-red-400/80 bg-red-950/80 text-red-50'
      : weirdState.tone === 'warning'
        ? 'border-amber-400/80 bg-amber-950/80 text-amber-50'
        : 'border-sky-400/80 bg-sky-950/80 text-sky-50';

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

  return (
    <div
      className={`px-2 py-1.5 rounded-lg border text-[10px] flex items-start gap-2 ${toneClasses}`}
      role="status"
      aria-live="polite"
      data-testid="mobile-weird-state-banner"
    >
      <span className="mt-0.5 text-base" aria-hidden="true">
        {icon}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1 mb-0.5">
          <span className="text-[9px] uppercase tracking-wide font-semibold opacity-80">
            {badgeLabel}
          </span>
        </div>
        <div className="font-semibold text-xs leading-snug">{weirdState.title}</div>
        <div className="mt-0.5 text-[10px] leading-snug opacity-90">{weirdState.body}</div>
      </div>
      {onShowHelp && (
        <button
          type="button"
          onClick={onShowHelp}
          className="ml-1 mt-0.5 inline-flex h-11 w-11 items-center justify-center rounded-full border border-current/60 text-sm font-semibold touch-manipulation"
          aria-label="Learn more about this situation"
          data-testid="mobile-weird-state-help"
        >
          ?
        </button>
      )}
    </div>
  );
}

/**
 * Mobile LPS (Last-Player-Standing) tracking indicator.
 * Compact version showing progress toward LPS victory.
 * Per RR-CANON-R172, LPS requires 2 consecutive rounds where only 1 player has real actions.
 */
function MobileLpsIndicator({
  lpsTracking,
  players,
}: {
  lpsTracking?: HUDViewModel['lpsTracking'];
  players: PlayerViewModel[];
}) {
  // Only show when there's progress toward LPS (at least 1 consecutive exclusive round)
  if (!lpsTracking || lpsTracking.consecutiveExclusiveRounds < 1) {
    return null;
  }

  const { consecutiveExclusiveRounds, consecutiveExclusivePlayer } = lpsTracking;
  const exclusivePlayer = players.find((p) => p.playerNumber === consecutiveExclusivePlayer);
  const playerName = exclusivePlayer?.username ?? `P${consecutiveExclusivePlayer}`;

  // Color progression: amber (1), red (2 = victory imminent)
  const colorClass =
    consecutiveExclusiveRounds >= 2
      ? 'border-red-400/80 bg-red-950/80 text-red-50'
      : 'border-amber-400/80 bg-amber-950/80 text-amber-50';

  return (
    <div
      className={`mb-2 px-2 py-1.5 rounded-lg border text-[11px] flex items-center gap-2 ${colorClass}`}
      role="status"
      aria-live="polite"
      data-testid="mobile-lps-indicator"
    >
      <span aria-hidden="true">🏆</span>
      <span className="font-semibold truncate flex-1">{playerName} exclusive</span>
      {/* Progress dots */}
      <div className="flex gap-0.5" aria-label={`${consecutiveExclusiveRounds} of 2 rounds`}>
        {[1, 2].map((n) => (
          <span
            key={n}
            className={`w-1.5 h-1.5 rounded-full ${
              n <= consecutiveExclusiveRounds ? 'bg-current' : 'bg-current/30'
            }`}
          />
        ))}
      </div>
    </div>
  );
}

/**
 * Compact mobile victory progress indicator.
 * Shows ring elimination and territory progress when meaningful.
 * Per RR-CANON-R061: victoryThreshold = round(ringsPerPlayer × (1/3 + 2/3 × (numPlayers - 1)))
 * Per RR-CANON-R062: territoryThreshold = floor(totalSpaces/2)+1
 */
function MobileVictoryProgress({
  victoryProgress,
  players,
}: {
  victoryProgress?: HUDViewModel['victoryProgress'];
  players: PlayerViewModel[];
}) {
  if (!victoryProgress) return null;

  const { ringElimination, territory } = victoryProgress;
  const ringLeader = ringElimination.leader;
  const territoryLeader = territory.leader;

  // Only show if someone has meaningful progress (>=25% toward any goal)
  const showRings = ringLeader && ringLeader.percentage >= 25;
  const showTerritory = territoryLeader && territoryLeader.percentage >= 25;

  if (!showRings && !showTerritory) return null;

  const getPlayerName = (playerNumber: number) => {
    const player = players.find((p) => p.playerNumber === playerNumber);
    return player?.username ?? `P${playerNumber}`;
  };

  return (
    <div
      className="mb-2 px-2 py-1.5 rounded-lg border border-slate-600/50 bg-slate-900/60 text-[10px]"
      data-testid="mobile-victory-progress"
    >
      <div className="flex items-center gap-3">
        {showRings && ringLeader && (
          <div className="flex items-center gap-1.5 flex-1">
            <span className="text-rose-400">⚔</span>
            <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-rose-500"
                style={{ width: `${Math.min(ringLeader.percentage, 100)}%` }}
              />
            </div>
            <span className="text-slate-300 whitespace-nowrap">
              {getPlayerName(ringLeader.playerNumber).slice(0, 6)}: {ringLeader.eliminated}/
              {ringElimination.threshold}
            </span>
          </div>
        )}
        {showTerritory && territoryLeader && (
          <div className="flex items-center gap-1.5 flex-1">
            <span className="text-emerald-400">🏰</span>
            <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{ width: `${Math.min(territoryLeader.percentage, 100)}%` }}
              />
            </div>
            <span className="text-slate-300 whitespace-nowrap">
              {getPlayerName(territoryLeader.playerNumber).slice(0, 6)}: {territoryLeader.spaces}/
              {territory.threshold}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Compact turn/phase indicator bar for mobile
 */
function MobilePhaseBar({
  phase,
  turnNumber,
  isMyTurn,
}: {
  phase: HUDViewModel['phase'];
  turnNumber: number;
  isMyTurn: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between px-3 py-2 rounded-lg ${phase.colorClass}`}
      data-testid="mobile-phase-bar"
    >
      <div className="flex items-center gap-2 min-w-0">
        {phase.icon && <span className="text-lg">{phase.icon}</span>}
        <span className="font-semibold text-sm text-white truncate">{phase.label}</span>
      </div>
      <div className="flex items-center gap-2 text-xs text-white/90">
        <span className="font-mono">Turn {turnNumber}</span>
        {isMyTurn && (
          <span className="px-1.5 py-0.5 rounded-full bg-white/20 text-[10px] font-semibold uppercase">
            Your turn
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Compact player summary row for mobile
 */
function MobilePlayerRow({ player, isExpanded }: { player: PlayerViewModel; isExpanded: boolean }) {
  const { ringStats, territorySpaces, aiInfo } = player;

  return (
    <div
      className={`flex items-center justify-between gap-2 px-2 py-1.5 rounded-lg transition-colors ${
        player.isCurrentPlayer ? 'bg-blue-900/40 border border-blue-500/50' : 'bg-slate-800/50'
      } ${player.isUserPlayer ? 'ring-1 ring-green-400/50' : ''}`}
    >
      {/* Player identity */}
      <div className="flex items-center gap-2 min-w-0 flex-1">
        <div className={`w-3 h-3 rounded-full shrink-0 ${player.colorClass}`} />
        <span className="text-xs font-medium text-slate-100 truncate">
          {player.isUserPlayer ? 'You' : player.username}
        </span>
        {aiInfo.isAI && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-slate-700 text-slate-300">AI</span>
        )}
        {player.isCurrentPlayer && (
          <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
        )}
      </div>

      {/* Quick stats */}
      <div className="flex items-center gap-3 text-[10px] text-slate-300">
        <span className="flex items-center gap-1">
          <span className="text-red-300">🔴</span>
          <span className="font-mono">
            {ringStats.eliminated}/{ringStats.total}
          </span>
        </span>
        <span className="flex items-center gap-1">
          <span className="text-emerald-300">🏰</span>
          <span className="font-mono">{territorySpaces}</span>
        </span>
      </div>

      {/* Expanded details */}
      {isExpanded && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] text-slate-300 pt-2 border-t border-slate-700">
          <div className="text-center">
            <div className="font-bold text-slate-100">{ringStats.inHand}</div>
            <div>In Hand</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-slate-100">{ringStats.onBoard}</div>
            <div>On Board</div>
          </div>
          <div className="text-center">
            <div className="font-bold text-red-400">{ringStats.eliminated}</div>
            <div>Captured</div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Decision timer pill for mobile
 */
function MobileDecisionTimer({
  timeRemainingMs,
  isServerCapped,
}: {
  timeRemainingMs: number;
  isServerCapped?: boolean;
}) {
  const totalSeconds = Math.max(0, Math.floor(timeRemainingMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const timeLabel = `${minutes}:${seconds.toString().padStart(2, '0')}`;

  const severity = getCountdownSeverity(timeRemainingMs);

  const bgClass =
    severity === 'critical'
      ? 'bg-red-900/80 border-red-400/60'
      : severity === 'warning'
        ? 'bg-amber-900/80 border-amber-400/60'
        : 'bg-slate-800/80 border-slate-600';

  const textClass =
    severity === 'critical'
      ? 'text-red-100'
      : severity === 'warning'
        ? 'text-amber-100'
        : 'text-slate-100';

  return (
    <div
      className={`inline-flex items-center gap-1 px-2 py-1 rounded-full border text-xs ${bgClass} ${
        severity === 'critical' ? 'animate-pulse' : ''
      }`}
      data-testid="mobile-decision-timer"
      data-severity={severity ?? undefined}
    >
      <span className="text-[10px]">⏱</span>
      <span className={`font-mono ${textClass}`}>{timeLabel}</span>
      {isServerCapped && <span className="text-[9px] text-amber-200">*</span>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * MobileGameHUD - Compact HUD optimized for mobile viewports
 *
 * Usage:
 * ```tsx
 * const isMobile = useIsMobile();
 * {isMobile ? (
 *   <MobileGameHUD viewModel={hudViewModel} timeControl={timeControl} />
 * ) : (
 *   <GameHUD viewModel={hudViewModel} timeControl={timeControl} />
 * )}
 * ```
 */
export function MobileGameHUD({
  viewModel,
  timeControl: _timeControl,
  isLocalSandboxOnly = false,
  onShowBoardControls,
  onPlayerTap,
  rulesUxContext,
}: MobileGameHUDProps) {
  const [expandedPlayerId, setExpandedPlayerId] = useState<string | null>(null);
  const { currentTopic, isOpen, showTopic, hideTopic } = useTeachingOverlay();

  const {
    phase,
    players,
    turnNumber,
    instruction,
    connectionStatus,
    isSpectator,
    spectatorCount,
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

  const isMyTurn = players.some((p) => p.isUserPlayer && p.isCurrentPlayer);

  const connectionColor =
    connectionStatus === 'connected'
      ? 'text-emerald-400'
      : connectionStatus === 'reconnecting'
        ? 'text-amber-400'
        : 'text-rose-400';

  const helpOpenCountsRef = React.useRef<Record<string, number>>({});
  const prevIsOpenRef = React.useRef(false);
  const prevTopicRef = React.useRef<TeachingTopic | null>(null);
  const autoScenarioHelpShownRef = React.useRef<Record<string, boolean>>({});

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
  // phases where it makes sense to auto-open contextual help on mobile.
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

  // Emit telemetry when the mobile TeachingOverlay opens for a topic.
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

    const baseEvent = {
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      topic: currentTopic,
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
    } as const;

    void sendRulesUxEvent({
      type: 'rules_help_open',
      ...baseEvent,
    });

    if (newCount >= 2) {
      void sendRulesUxEvent({
        type: 'rules_help_repeat',
        ...baseEvent,
        repeatCount: newCount,
      });
    }
  }, [
    currentTopic,
    isOpen,
    rulesUxAiDifficulty,
    rulesUxBoardType,
    rulesUxNumPlayers,
    rulesUxRulesConcept,
    rulesUxScenarioId,
  ]);

  // Auto-open curated TeachingOverlay topics for selected onboarding / rules
  // scenarios on mobile the first time their key phase becomes active.
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
    void sendRulesUxEvent({
      type: 'rules_weird_state_help',
      boardType: rulesUxBoardType,
      numPlayers: rulesUxNumPlayers,
      weirdStateType,
      topic,
      ...(rulesUxAiDifficulty !== undefined ? { aiDifficulty: rulesUxAiDifficulty } : {}),
      ...(rulesUxRulesConcept ? { rulesConcept: rulesUxRulesConcept } : {}),
      ...(rulesUxScenarioId ? { scenarioId: rulesUxScenarioId } : {}),
    });
  }, [
    rulesUxAiDifficulty,
    rulesUxBoardType,
    rulesUxNumPlayers,
    rulesUxRulesConcept,
    rulesUxScenarioId,
    showTopic,
    surfaceableWeirdState,
  ]);

  const togglePlayerExpand = (playerId: string) => {
    setExpandedPlayerId((prev) => (prev === playerId ? null : playerId));
  };

  const handlePlayerTap = (player: PlayerViewModel) => {
    togglePlayerExpand(player.id);
    if (onPlayerTap) {
      onPlayerTap(player);
    }
  };

  return (
    <div className="space-y-2" data-testid="mobile-game-hud">
      {/* Local sandbox banner (compact) */}
      {isLocalSandboxOnly && (
        <div className="px-2 py-1 rounded bg-slate-900/70 border border-slate-600 text-[10px] text-slate-300">
          <span className="font-semibold">Local sandbox</span> – not logged in
        </div>
      )}

      {/* Spectator badge */}
      {isSpectator && (
        <div
          className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-purple-900/40 border border-purple-500/40 text-xs text-purple-100"
          data-testid="mobile-spectator-badge"
        >
          <svg className="w-3 h-3" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path d="M10 12.5a2.5 2.5 0 100-5 2.5 2.5 0 000 5z" />
            <path
              fillRule="evenodd"
              d="M.664 10.59a1.651 1.651 0 010-1.186A10.004 10.004 0 0110 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0110 17c-4.257 0-7.893-2.66-9.336-6.41zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
              clipRule="evenodd"
            />
          </svg>
          <span>Spectating</span>
          {typeof spectatorCount === 'number' && spectatorCount > 0 && (
            <span className="text-[10px] text-purple-200/80">
              • {spectatorCount} {spectatorCount === 1 ? 'viewer' : 'viewers'}
            </span>
          )}
        </div>
      )}

      {/* Weird-state banner (compact) */}
      {surfaceableWeirdState && (
        <MobileWeirdStateBanner
          weirdState={surfaceableWeirdState}
          onShowHelp={handleWeirdStateHelp}
        />
      )}

      {/* LPS tracking indicator (compact) */}
      <MobileLpsIndicator lpsTracking={lpsTracking} players={players} />

      {/* Victory progress indicator (compact) */}
      <MobileVictoryProgress victoryProgress={victoryProgress} players={players} />

      {/* Phase + turn bar */}
      <MobilePhaseBar phase={phase} turnNumber={turnNumber} isMyTurn={isMyTurn} />

      {/* Phase-level rules help */}
      {phaseHelpTopic && (
        <div className="flex justify-end mt-1">
          <button
            type="button"
            onClick={() => showTopic(phaseHelpTopic)}
            className="inline-flex items-center gap-1.5 px-3 py-2.5 min-h-[44px] rounded-full border border-slate-600 bg-slate-900/80 text-xs text-slate-100 touch-manipulation active:bg-slate-800"
            aria-label="Phase rules help"
            data-testid={`mobile-phase-help-${phase.phaseKey}`}
          >
            <span className="text-sm">?</span>
            <span>Phase rules</span>
          </button>
        </div>
      )}

      {/* Decision timer (when active) */}
      {decisionPhase &&
        decisionPhase.isActive &&
        decisionPhase.showCountdown &&
        decisionPhase.timeRemainingMs !== null && (
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-300 truncate flex-1">
              {decisionPhase.label}
            </span>
            <MobileDecisionTimer
              timeRemainingMs={decisionPhase.timeRemainingMs}
              {...(decisionPhase.isServerCapped ? { isServerCapped: true } : {})}
            />
          </div>
        )}

      {/* Decision-specific status chip (e.g., ring elimination prompt) */}
      {decisionPhase?.statusChip && (
        <div className="mt-1 flex items-center justify-between">
          <span
            className={
              decisionPhase.statusChip.tone === 'attention'
                ? 'px-2 py-0.5 rounded-full bg-amber-500 text-slate-950 text-[10px] font-semibold border border-amber-300'
                : 'px-2 py-0.5 rounded-full bg-sky-900/60 text-sky-100 text-[10px] font-medium border border-sky-500/60'
            }
            data-testid="mobile-decision-status-chip"
          >
            {decisionPhase.statusChip.text}
          </span>
          {decisionPhase.canSkip && (
            <span
              className="ml-2 px-1.5 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 text-[9px] uppercase tracking-wide text-slate-200"
              data-testid="mobile-decision-skip-hint"
            >
              Skip available
            </span>
          )}
        </div>
      )}

      {/* Territory-processing help: surface a concise rules recap when players
          are choosing regions or eliminations during the territory phase. */}
      {phase.phaseKey === 'territory_processing' && (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => showTopic('territory')}
            className="mt-1 inline-flex items-center gap-1.5 px-3 py-2.5 min-h-[44px] rounded-full border border-slate-600 bg-slate-900/80 text-xs text-slate-100 touch-manipulation active:bg-slate-800"
            aria-label="Territory rules help"
            data-testid="mobile-territory-help"
          >
            <span className="text-sm">?</span>
            <span>Territory rules</span>
          </button>
        </div>
      )}

      {/* Instruction banner (compact) */}
      {instruction && (
        <div className="px-2 py-1.5 rounded bg-slate-700/50 border border-slate-600 text-xs text-slate-200 text-center">
          {instruction}
        </div>
      )}

      {/* Player list (compact) */}
      <div className="space-y-1">
        {players.map((player) => (
          <button
            key={player.id}
            className="w-full text-left"
            onClick={() => handlePlayerTap(player)}
            aria-expanded={expandedPlayerId === player.id}
          >
            <MobilePlayerRow player={player} isExpanded={expandedPlayerId === player.id} />
          </button>
        ))}
      </div>

      {/* Footer bar: connection + help */}
      <div className="flex items-center justify-between text-[10px] text-slate-400">
        <span className={connectionColor}>● {connectionStatus}</span>
        <div className="flex items-center gap-2">
          {onShowBoardControls && (
            <button
              onClick={onShowBoardControls}
              className="px-3 py-2.5 min-h-[44px] rounded-lg border border-slate-600 bg-slate-900/70 hover:border-slate-400 transition-colors touch-manipulation active:bg-slate-800 text-xs"
              aria-label="Board controls"
              data-testid="mobile-controls-button"
            >
              <Tooltip content="Keyboard shortcuts and controls">
                <span>? Help</span>
              </Tooltip>
            </button>
          )}
        </div>
      </div>

      {currentTopic && (
        <TeachingOverlay
          topic={currentTopic}
          isOpen={isOpen}
          onClose={hideTopic}
          position="bottom-right"
        />
      )}
    </div>
  );
}

export default MobileGameHUD;
