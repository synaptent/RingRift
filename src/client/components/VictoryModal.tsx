import React, { useEffect, useRef, useState } from 'react';
import { GameResult, Player, GameState } from '../../shared/types/game';
import {
  toVictoryViewModel,
  type VictoryViewModel,
  type PlayerFinalStatsViewModel,
  type PlayerViewModel,
} from '../adapters/gameViewModels';
import {
  TeachingOverlay,
  useTeachingOverlay,
  type TeachingTopic,
  type WeirdStateOverlayContext,
} from './TeachingOverlay';
import {
  getWeirdStateReasonForGameResult,
  getTeachingTopicForReason,
} from '../../shared/engine/weirdStateReasons';
import { logRulesUxEvent, newOverlaySessionId } from '../utils/rulesUxTelemetry';

/**
 * Status of a pending rematch request.
 */
export interface RematchStatus {
  /** Whether a rematch request is currently pending */
  isPending: boolean;
  /** ID of the pending rematch request */
  requestId?: string;
  /** Username of the player who requested the rematch */
  requesterUsername?: string;
  /** Whether the current user initiated the request */
  isRequester?: boolean;
  /** Expiration timestamp (ISO-8601) */
  expiresAt?: string;
  /** Current status */
  status?: 'pending' | 'accepted' | 'declined' | 'expired';
}

interface VictoryModalProps {
  isOpen: boolean;
  gameResult?: GameResult | null;
  players?: Player[];
  gameState?: GameState;
  /**
   * Optional pre-transformed view model. When provided, this is used as the
   * primary source of truth; legacy props (gameResult/players/gameState) are
   * only used as a fallback to construct a VictoryViewModel.
   */
  viewModel?: VictoryViewModel | null;
  onClose: () => void;
  onReturnToLobby: () => void;
  /** Called when user requests a rematch (for local/sandbox games) */
  onRematch?: () => void;
  /** Called when user requests a rematch in backend games */
  onRequestRematch?: () => void;
  /** Called when user accepts a rematch request */
  onAcceptRematch?: (requestId: string) => void;
  /** Called when user declines a rematch request */
  onDeclineRematch?: (requestId: string) => void;
  /** Current rematch status for backend games */
  rematchStatus?: RematchStatus;
  currentUserId?: string;
  /**
   * Optional flag indicating that this VictoryModal is being shown from a
   * sandbox / local game context. Used for low-cardinality rules-UX telemetry
   * labels (is_sandbox).
   */
  isSandbox?: boolean;
}

/**
 * Confetti particle component for celebration effect
 */
function ConfettiParticles() {
  const confettiEmojis = ['🎉', '🎊', '✨', '⭐', '🌟', '💫', '🏆', '👑'];

  return (
    <>
      {confettiEmojis.map((emoji, index) => (
        <span key={index} className={`confetti-particle confetti-${index + 1}`} aria-hidden="true">
          {emoji}
        </span>
      ))}
    </>
  );
}

/**
 * Animated trophy display for victory celebration
 */
function AnimatedTrophy({ victoryCondition }: { victoryCondition: GameResult['reason'] }) {
  const getTrophyEmoji = () => {
    switch (victoryCondition) {
      case 'ring_elimination':
        return '🏆';
      case 'territory_control':
        return '🏰';
      case 'last_player_standing':
        return '👑';
      case 'timeout':
        return '⏰';
      case 'resignation':
        return '🏳️';
      case 'abandonment':
        return '🚪';
      case 'draw':
        return '🤝';
      default:
        return '🏆';
    }
  };

  return (
    <div className="flex justify-center mb-2">
      <span className="trophy-animate text-6xl celebrating" role="img" aria-label="trophy">
        {getTrophyEmoji()}
      </span>
    </div>
  );
}

/**
 * Final statistics table component
 */
function FinalStatsTable({
  stats,
  winner,
}: {
  stats: PlayerFinalStatsViewModel[];
  winner?: PlayerViewModel | undefined;
}) {
  // Sort by winner first, then by rings eliminated (descending)
  const sortedStats = [...stats].sort((a, b) => {
    if (a.isWinner && !b.isWinner) return -1;
    if (!a.isWinner && b.isWinner) return 1;
    return b.ringsEliminated - a.ringsEliminated;
  });

  return (
    <div className="overflow-x-auto">
      <table className="w-full">
        <thead>
          <tr className="bg-slate-700 text-slate-100">
            <th className="px-4 py-2 text-left">Player</th>
            <th className="px-4 py-2 text-center">Rings on Board</th>
            <th className="px-4 py-2 text-center">Rings Eliminated</th>
            <th className="px-4 py-2 text-center">Territory</th>
            <th className="px-4 py-2 text-center">Moves</th>
          </tr>
        </thead>
        <tbody>
          {sortedStats.map((stat) => (
            <tr
              key={stat.player.playerNumber}
              className={`border-b border-slate-600 ${
                winner?.playerNumber === stat.player.playerNumber
                  ? 'bg-yellow-900/30 font-semibold'
                  : 'bg-slate-800/50'
              }`}
            >
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded-full border border-slate-400"
                    style={{
                      backgroundColor: getPlayerColor(stat.player.playerNumber),
                    }}
                  />
                  <span className="text-slate-100">
                    {stat.player.username || `Player ${stat.player.playerNumber}`}
                  </span>
                  {stat.isWinner && <span className="text-yellow-500">👑</span>}
                </div>
              </td>
              <td className="px-4 py-3 text-center text-slate-200">{stat.ringsOnBoard}</td>
              <td className="px-4 py-3 text-center text-slate-200">{stat.ringsEliminated}</td>
              <td className="px-4 py-3 text-center text-slate-200">{stat.territorySpaces}</td>
              <td className="px-4 py-3 text-center text-slate-200">{stat.totalMoves}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Game summary component
 */
function GameSummary({ summary }: { summary: VictoryViewModel['gameSummary'] }) {
  if (!summary) return null;

  return (
    <div className="bg-slate-800/50 rounded-lg p-4 space-y-2 text-sm text-slate-200">
      <div className="flex justify-between">
        <span className="text-slate-400">Board Type:</span>
        <span className="font-semibold">{summary.boardType}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Total Turns:</span>
        <span className="font-semibold">{summary.totalTurns}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Players:</span>
        <span className="font-semibold">{summary.playerCount}</span>
      </div>
      {summary.isRated && (
        <div className="flex justify-between">
          <span className="text-slate-400">Game Type:</span>
          <span className="font-semibold text-purple-400">Rated</span>
        </div>
      )}
    </div>
  );
}

/**
 * Get player color based on player number
 */
function getPlayerColor(playerNumber: number): string {
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b']; // blue, red, green, amber
  return colors[(playerNumber - 1) % colors.length];
}

/**
 * Hook to calculate countdown time remaining
 */
function useCountdown(expiresAt?: string): number {
  const [remaining, setRemaining] = useState(() => {
    if (!expiresAt) return 0;
    return Math.max(0, Math.floor((new Date(expiresAt).getTime() - Date.now()) / 1000));
  });

  useEffect(() => {
    if (!expiresAt) return;

    const update = () => {
      const ms = new Date(expiresAt).getTime() - Date.now();
      setRemaining(Math.max(0, Math.floor(ms / 1000)));
    };

    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, [expiresAt]);

  return remaining;
}

/**
 * Rematch section component
 */
function RematchSection({
  rematchStatus,
  onRematch,
  onRequestRematch,
  onAcceptRematch,
  onDeclineRematch,
}: {
  rematchStatus?: RematchStatus | undefined;
  onRematch?: (() => void) | undefined;
  onRequestRematch?: (() => void) | undefined;
  onAcceptRematch?: ((requestId: string) => void) | undefined;
  onDeclineRematch?: ((requestId: string) => void) | undefined;
}) {
  const countdown = useCountdown(rematchStatus?.expiresAt);

  // Local/sandbox game: simple rematch button
  if (onRematch && !onRequestRematch) {
    return (
      <button
        onClick={onRematch}
        className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
      >
        Play Again
      </button>
    );
  }

  // No rematch functionality available
  if (!onRequestRematch) {
    return null;
  }

  // Rematch request was accepted - show redirect message
  if (rematchStatus?.status === 'accepted') {
    return (
      <div className="px-6 py-3 bg-green-900/50 border border-green-600 rounded-lg text-green-200 text-center">
        Rematch accepted! Joining new game...
      </div>
    );
  }

  // Rematch request was declined
  if (rematchStatus?.status === 'declined') {
    return (
      <div className="px-6 py-3 bg-red-900/50 border border-red-600 rounded-lg text-red-200 text-center">
        Rematch declined
      </div>
    );
  }

  // Rematch request expired
  if (rematchStatus?.status === 'expired' || (rematchStatus?.isPending && countdown <= 0)) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-slate-300 text-sm">
          Rematch request expired
        </div>
        <button
          onClick={onRequestRematch}
          className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Request Rematch
        </button>
      </div>
    );
  }

  // Pending rematch request from another player
  if (rematchStatus?.isPending && !rematchStatus.isRequester && rematchStatus.requestId) {
    const requestId = rematchStatus.requestId;
    return (
      <div className="flex flex-col items-center gap-3">
        <div className="text-center">
          <p className="text-slate-200 font-medium">
            {rematchStatus.requesterUsername || 'Your opponent'} wants a rematch!
          </p>
          <p className="text-sm text-slate-400">
            {countdown > 0 ? `${countdown}s remaining` : 'Expired'}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => onAcceptRematch?.(requestId)}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Accept
          </button>
          <button
            onClick={() => onDeclineRematch?.(requestId)}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Decline
          </button>
        </div>
      </div>
    );
  }

  // Current user has a pending rematch request
  if (rematchStatus?.isPending && rematchStatus.isRequester) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="px-6 py-3 bg-slate-700/50 border border-slate-600 rounded-lg text-slate-200 text-center">
          <p className="font-medium">Waiting for opponent...</p>
          <p className="text-sm text-slate-400">
            {countdown > 0 ? `${countdown}s remaining` : 'Expired'}
          </p>
        </div>
      </div>
    );
  }

  // No pending request - show request button
  return (
    <button
      onClick={onRequestRematch}
      className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
    >
      Request Rematch
    </button>
  );
}

/**
 * Victory Modal Component
 */
const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

export function VictoryModal({
  isOpen,
  gameResult,
  players,
  gameState,
  viewModel,
  onClose,
  onReturnToLobby,
  onRematch,
  onRequestRematch,
  onAcceptRematch,
  onDeclineRematch,
  rematchStatus,
  currentUserId,
  isSandbox,
}: VictoryModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);
  const overlaySessionIdRef = useRef<string | null>(null);
  const { currentTopic, isOpen: isTeachingOpen, showTopic, hideTopic } = useTeachingOverlay();

  // Keyboard navigation (Escape to close)
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Focus trap within the modal and focus restoration
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    previouslyFocusedElementRef.current = (document.activeElement as HTMLElement | null) ?? null;

    if (!dialogEl) return;

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (first) {
      first.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || focusable.length === 0) return;

      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      const isShift = event.shiftKey;

      if (isShift && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!isShift && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    dialogEl.addEventListener('keydown', handleKeyDown);

    return () => {
      dialogEl.removeEventListener('keydown', handleKeyDown);
      if (previouslyFocusedElementRef.current) {
        previouslyFocusedElementRef.current.focus();
      }
    };
  }, [isOpen]);

  // Rules-UX telemetry: emit weird_state_banner_impression for weird-state victories
  // when the VictoryModal first becomes visible for a given result.
  useEffect(() => {
    if (!isOpen) return;

    const effectivePlayersLocal = players ?? [];
    const effectiveGameResultLocal = gameResult ?? null;

    const vm =
      viewModel ??
      toVictoryViewModel(effectiveGameResultLocal, effectivePlayersLocal, gameState, {
        currentUserId,
        isDismissed: false,
      });

    if (!vm || !vm.isVisible) {
      return;
    }

    const weirdInfo = getWeirdStateReasonForGameResult(effectiveGameResultLocal);
    if (!weirdInfo) {
      return;
    }

    let overlaySessionId = overlaySessionIdRef.current;
    if (!overlaySessionId) {
      overlaySessionId = newOverlaySessionId();
      overlaySessionIdRef.current = overlaySessionId;
    }

    void logRulesUxEvent({
      type: 'weird_state_banner_impression',
      boardType: vm.gameSummary.boardType,
      numPlayers: vm.gameSummary.playerCount,
      rulesContext: weirdInfo.rulesContext,
      source: 'victory_modal',
      weirdStateType: weirdInfo.weirdStateType,
      reasonCode: weirdInfo.reasonCode,
      isRanked: gameState?.isRated,
      isSandbox: isSandbox ?? false,
      overlaySessionId,
    });
  }, [isOpen, gameResult, players, gameState, viewModel, currentUserId, isSandbox]);

  if (!isOpen) return null;

  const effectivePlayers = players ?? [];
  const effectiveGameResult = gameResult ?? null;

  const effectiveViewModel: VictoryViewModel | null =
    viewModel ??
    toVictoryViewModel(effectiveGameResult, effectivePlayers, gameState, {
      currentUserId,
      isDismissed: false,
    });

  if (!effectiveViewModel || !effectiveViewModel.isVisible) {
    return null;
  }

  const weirdStateInfo = getWeirdStateReasonForGameResult(effectiveGameResult);

  let weirdStateTeachingTopic: TeachingTopic | null = null;
  if (weirdStateInfo) {
    const teachingId = getTeachingTopicForReason(weirdStateInfo.reasonCode);
    switch (teachingId) {
      case 'teaching.active_no_moves':
        weirdStateTeachingTopic = 'active_no_moves';
        break;
      case 'teaching.forced_elimination':
        weirdStateTeachingTopic = 'forced_elimination';
        break;
      case 'teaching.line_bonus':
        weirdStateTeachingTopic = 'line_bonus';
        break;
      case 'teaching.territory':
        weirdStateTeachingTopic = 'territory';
        break;
      case 'teaching.victory_stalemate':
        weirdStateTeachingTopic = 'victory_stalemate';
        break;
      default:
        weirdStateTeachingTopic = null;
        break;
    }
  }

  const teachingWeirdStateContext: WeirdStateOverlayContext | null =
    weirdStateInfo && overlaySessionIdRef.current
      ? {
          reasonCode: weirdStateInfo.reasonCode,
          rulesContext: weirdStateInfo.rulesContext,
          weirdStateType: weirdStateInfo.weirdStateType,
          boardType: gameSummary.boardType,
          numPlayers: gameSummary.playerCount,
          isRanked: gameState?.isRated,
          isSandbox: isSandbox ?? false,
          overlaySessionId: overlaySessionIdRef.current,
        }
      : null;

  const { title, description, finalStats, winner, gameSummary, userWon, userLost, isDraw } =
    effectiveViewModel;

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  const handleWeirdStateHelpClick = () => {
    if (!weirdStateInfo || !weirdStateTeachingTopic) {
      return;
    }

    showTopic(weirdStateTeachingTopic);

    let overlaySessionId = overlaySessionIdRef.current;
    if (!overlaySessionId) {
      overlaySessionId = newOverlaySessionId();
      overlaySessionIdRef.current = overlaySessionId;
    }

    void logRulesUxEvent({
      type: 'weird_state_details_open',
      boardType: gameSummary.boardType,
      numPlayers: gameSummary.playerCount,
      rulesContext: weirdStateInfo.rulesContext,
      source: 'victory_modal',
      weirdStateType: weirdStateInfo.weirdStateType,
      reasonCode: weirdStateInfo.reasonCode,
      isRanked: gameState?.isRated,
      isSandbox: isSandbox ?? false,
      overlaySessionId,
      topic: weirdStateTeachingTopic,
    });
  };

  const showConfetti = !isDraw && effectiveGameResult?.reason !== 'abandonment';

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 backdrop-blur-sm modal-backdrop-animate"
      role="dialog"
      aria-modal="true"
      aria-labelledby="victory-title"
      aria-describedby="victory-description"
      onClick={handleBackdropClick}
    >
      {/* Confetti particles for celebration */}
      {showConfetti && <ConfettiParticles />}

      <div className="victory-modal modal-content-animate bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-3xl w-full mx-4 p-6 space-y-6 relative overflow-hidden">
        {/* Shimmer effect overlay for victories */}
        {showConfetti && (
          <div
            className="absolute inset-0 victory-shimmer pointer-events-none"
            aria-hidden="true"
          />
        )}

        {/* Header with celebration animation */}
        <div className="text-center space-y-2 relative z-10">
          {/* Animated Trophy */}
          <AnimatedTrophy victoryCondition={effectiveGameResult?.reason ?? 'ring_elimination'} />

          <h1
            id="victory-title"
            className={`winner-text-animate text-4xl font-bold ${
              userWon ? 'text-green-400 winner-glow' : userLost ? 'text-red-400' : 'text-slate-100'
            }`}
          >
            {title}
          </h1>
          <p
            id="victory-description"
            className="winner-text-animate text-slate-300 text-lg"
            style={{ animationDelay: '300ms' }}
          >
            {description}
          </p>
          {weirdStateInfo && weirdStateTeachingTopic && (
            <div className="mt-2">
              <button
                type="button"
                onClick={handleWeirdStateHelpClick}
                className="text-sm text-sky-300 hover:text-sky-200 underline decoration-dotted"
              >
                What happened?
              </button>
            </div>
          )}
        </div>

        {/* Statistics Table with staggered animation */}
        <div className="stats-animate relative z-10">
          <FinalStatsTable stats={finalStats} winner={winner} />
        </div>

        {/* Game Details with staggered animation */}
        <div className="summary-animate relative z-10">
          <GameSummary summary={gameSummary} />
        </div>

        {/* Rematch section */}
        {(onRematch || onRequestRematch) && (
          <div className="buttons-animate flex justify-center relative z-10">
            <RematchSection
              rematchStatus={rematchStatus}
              onRematch={onRematch}
              onRequestRematch={onRequestRematch}
              onAcceptRematch={onAcceptRematch}
              onDeclineRematch={onDeclineRematch}
            />
          </div>
        )}

        {/* Action Buttons with staggered animation */}
        <div className="buttons-animate flex gap-3 justify-center flex-wrap relative z-10">
          <button
            onClick={onReturnToLobby}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Return to Lobby
          </button>

          <button
            onClick={onClose}
            className="px-6 py-3 bg-slate-700 text-slate-100 rounded-lg hover:bg-slate-600 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Close
          </button>
        </div>
      </div>
      {currentTopic && (
        <TeachingOverlay
          topic={currentTopic}
          isOpen={isTeachingOpen}
          onClose={hideTopic}
          position="center"
          weirdStateOverlayContext={teachingWeirdStateContext}
        />
      )}
    </div>
  );
}
