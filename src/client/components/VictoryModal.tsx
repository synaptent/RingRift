import React, { useEffect, useRef } from 'react';
import { GameResult, Player, GameState } from '../../shared/types/game';
import {
  toVictoryViewModel,
  type VictoryViewModel,
  type PlayerFinalStatsViewModel,
  type PlayerViewModel,
} from '../adapters/gameViewModels';

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
  onRematch?: () => void;
  currentUserId?: string;
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
  winner?: PlayerViewModel;
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
            <th className="px-4 py-2 text-center">Rings Lost</th>
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
  currentUserId,
}: VictoryModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);

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

  const { title, description, finalStats, winner, gameSummary, userWon, userLost, isDraw } =
    effectiveViewModel;

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
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
        </div>

        {/* Statistics Table with staggered animation */}
        <div className="stats-animate relative z-10">
          <FinalStatsTable stats={finalStats} winner={winner} />
        </div>

        {/* Game Details with staggered animation */}
        <div className="summary-animate relative z-10">
          <GameSummary summary={gameSummary} />
        </div>

        {/* Action Buttons with staggered animation */}
        <div className="buttons-animate flex gap-3 justify-center flex-wrap relative z-10">
          <button
            onClick={onReturnToLobby}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Return to Lobby
          </button>

          {onRematch && (
            <button
              onClick={onRematch}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              Request Rematch
            </button>
          )}

          <button
            onClick={onClose}
            className="px-6 py-3 bg-slate-700 text-slate-100 rounded-lg hover:bg-slate-600 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
