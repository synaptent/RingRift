import { useCallback, useEffect, useRef, useState } from 'react';
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
  getRulesContextForReason,
  getWeirdStateTypeForReason,
  getTeachingTopicForReason,
  isSurfaceableWeirdStateReason,
  isSurfaceableWeirdStateType,
} from '../../shared/engine/weirdStateReasons';
import { logRulesUxEvent, newOverlaySessionId } from '../utils/rulesUxTelemetry';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { Dialog } from './ui/Dialog';
import { Button } from './ui/Button';
import { getPlayerIndicatorPatternClass, getPlayerTheme } from '../utils/playerTheme';

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

/**
 * State for the training submission feature.
 */
export interface TrainingSubmissionState {
  /** Whether training submission is available for this game */
  isAvailable: boolean;
  /** Whether submission is in progress */
  isSubmitting: boolean;
  /** Whether submission was successful */
  wasSubmitted: boolean;
  /** Error message if submission failed */
  error?: string | null;
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
  /**
   * Optional canonical GameEndExplanation derived from the shared engine view.
   * This is currently advisory; VictoryModal still derives its copy from
   * GameResult, but callers may progressively migrate rules-UX and teaching
   * behaviour to rely on this explanation structure.
   */
  gameEndExplanation?: GameEndExplanation | null;
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
  /**
   * Called when user wants to submit their winning game for AI training.
   * Only shown when human wins against AI in sandbox mode.
   * January 2026 - Human game training enhancement.
   */
  onSubmitForTraining?: () => Promise<void>;
  /**
   * Current state of training submission (managed by parent component).
   */
  trainingSubmission?: TrainingSubmissionState;
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
function AnimatedTrophy({
  victoryCondition,
  animate = true,
}: {
  victoryCondition: GameResult['reason'];
  animate?: boolean;
}) {
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
      <span
        className={animate ? 'trophy-animate text-6xl celebrating' : 'text-6xl'}
        role="img"
        aria-label="trophy"
      >
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
  const { colorVisionMode } = useAccessibility();
  const patternClassesForPlayer = (playerNumber: number) =>
    colorVisionMode === 'normal'
      ? ''
      : `${getPlayerIndicatorPatternClass(playerNumber)} text-black/30`;
  const playerHex = (playerNumber: number) => getPlayerTheme(playerNumber, colorVisionMode).hex;

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
                    className={`w-4 h-4 rounded-full border border-slate-400 ${patternClassesForPlayer(
                      stat.player.playerNumber
                    )}`}
                    style={{
                      backgroundColor: playerHex(stat.player.playerNumber),
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
 * Victory breakdown component explaining why the winner won
 */
function VictoryBreakdown({
  gameResult,
  winner,
  playerCount,
}: {
  gameResult: GameResult | null;
  winner?: PlayerFinalStatsViewModel | undefined;
  playerCount: number;
}) {
  if (!gameResult) return null;

  const { reason, finalScore } = gameResult;
  const winnerNum = winner?.player.playerNumber;

  const getBreakdownContent = () => {
    switch (reason) {
      case 'ring_elimination': {
        if (!winnerNum) return null;
        const winnerEliminated = finalScore.ringsEliminated[winnerNum] ?? 0;
        const totalEliminated = Object.values(finalScore.ringsEliminated).reduce(
          (a, b) => a + b,
          0
        );
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-green-400">
              <span className="text-xl">💀</span>
              <span className="font-medium">Ring Elimination Victory</span>
            </div>
            <p className="text-slate-300 text-sm">
              {winner?.player.username || `Player ${winnerNum}`} eliminated all opponents&apos;
              rings.
            </p>
            <div className="flex gap-4 text-sm">
              <div className="bg-slate-700/50 px-3 py-1 rounded">
                <span className="text-slate-400">Rings eliminated: </span>
                <span className="text-green-400 font-semibold">{winnerEliminated}</span>
              </div>
              {totalEliminated > winnerEliminated && (
                <div className="bg-slate-700/50 px-3 py-1 rounded">
                  <span className="text-slate-400">Total eliminated: </span>
                  <span className="text-slate-200">{totalEliminated}</span>
                </div>
              )}
            </div>
          </div>
        );
      }

      case 'territory_control': {
        if (!winnerNum) return null;
        const allTerritories = Object.entries(finalScore.territorySpaces)
          .map(([num, spaces]) => ({ player: parseInt(num), spaces }))
          .sort((a, b) => b.spaces - a.spaces);

        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-blue-400">
              <span className="text-xl">🏰</span>
              <span className="font-medium">Territory Control Victory</span>
            </div>
            <p className="text-slate-300 text-sm">
              {winner?.player.username || `Player ${winnerNum}`} controlled the most territory when
              the board was full.
            </p>
            <div className="flex flex-wrap gap-2 text-sm">
              {allTerritories.map(({ player, spaces }, idx) => (
                <div
                  key={player}
                  className={`px-3 py-1 rounded ${
                    player === winnerNum
                      ? 'bg-blue-900/50 border border-blue-600'
                      : 'bg-slate-700/50'
                  }`}
                >
                  <span className="text-slate-400">P{player}: </span>
                  <span
                    className={
                      player === winnerNum ? 'text-blue-400 font-semibold' : 'text-slate-200'
                    }
                  >
                    {spaces} cells
                  </span>
                  {idx === 0 && player === winnerNum && (
                    <span className="ml-1 text-yellow-400">👑</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      }

      case 'last_player_standing': {
        const activePlayers = Object.entries(finalScore.ringsRemaining)
          .filter(([_, rings]) => rings > 0)
          .map(([num]) => parseInt(num));

        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-purple-400">
              <span className="text-xl">👑</span>
              <span className="font-medium">Last Player Standing</span>
            </div>
            <p className="text-slate-300 text-sm">
              {winner?.player.username || `Player ${winnerNum}`} was the only player with rings
              remaining.
            </p>
            <div className="bg-slate-700/50 px-3 py-1 rounded text-sm inline-block">
              <span className="text-slate-400">Opponents eliminated: </span>
              <span className="text-purple-400 font-semibold">
                {playerCount - activePlayers.length}
              </span>
            </div>
          </div>
        );
      }

      case 'timeout':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-orange-400">
              <span className="text-xl">⏰</span>
              <span className="font-medium">Victory by Timeout</span>
            </div>
            <p className="text-slate-300 text-sm">An opponent ran out of time on their clock.</p>
          </div>
        );

      case 'resignation':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-slate-400">
              <span className="text-xl">🏳️</span>
              <span className="font-medium">Victory by Resignation</span>
            </div>
            <p className="text-slate-300 text-sm">An opponent resigned from the game.</p>
          </div>
        );

      case 'abandonment':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-red-400">
              <span className="text-xl">🚪</span>
              <span className="font-medium">Victory by Abandonment</span>
            </div>
            <p className="text-slate-300 text-sm">An opponent left the game without resigning.</p>
          </div>
        );

      case 'draw':
        return (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-slate-300">
              <span className="text-xl">🤝</span>
              <span className="font-medium">Game Drawn</span>
            </div>
            <p className="text-slate-300 text-sm">The game ended in a draw - no clear winner.</p>
          </div>
        );

      default:
        return null;
    }
  };

  const content = getBreakdownContent();
  if (!content) return null;

  return <div className="bg-slate-800/70 rounded-lg p-4 border border-slate-600/50">{content}</div>;
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
      <Button type="button" size="lg" onClick={onRematch}>
        Play Again
      </Button>
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
        Rematch on! Joining new game...
      </div>
    );
  }

  // Rematch request was declined
  if (rematchStatus?.status === 'declined') {
    return (
      <div className="px-6 py-3 bg-red-900/50 border border-red-600 rounded-lg text-red-200 text-center">
        Opponent declined the rematch
      </div>
    );
  }

  // Rematch request expired
  if (rematchStatus?.status === 'expired' || (rematchStatus?.isPending && countdown <= 0)) {
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-slate-300 text-sm">
          Rematch offer expired
        </div>
        <Button type="button" size="lg" onClick={onRequestRematch}>
          Play Again?
        </Button>
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
          <Button type="button" size="lg" onClick={() => onAcceptRematch?.(requestId)}>
            Accept
          </Button>
          <Button
            type="button"
            size="lg"
            variant="danger"
            onClick={() => onDeclineRematch?.(requestId)}
          >
            Decline
          </Button>
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
    <Button type="button" size="lg" onClick={onRequestRematch}>
      Play Again?
    </Button>
  );
}

/**
 * Training submission section component (January 2026).
 * Allows users to submit their winning games against AI for training.
 */
function TrainingSubmissionSection({
  onSubmit,
  state,
}: {
  onSubmit: () => Promise<void>;
  state: TrainingSubmissionState;
}) {
  // Already submitted successfully
  if (state.wasSubmitted) {
    return (
      <div className="bg-green-900/30 border border-green-600/50 rounded-lg px-4 py-3 text-center">
        <div className="flex items-center justify-center gap-2 text-green-400">
          <span className="text-lg">&#10003;</span>
          <span className="font-medium">Thanks for contributing!</span>
        </div>
        <p className="text-sm text-green-300/80 mt-1">Your game is helping train the AI. Thanks!</p>
      </div>
    );
  }

  // Submission error
  if (state.error) {
    return (
      <div className="bg-red-900/30 border border-red-600/50 rounded-lg px-4 py-3 text-center">
        <div className="flex items-center justify-center gap-2 text-red-400">
          <span className="text-lg">&#x2717;</span>
          <span className="font-medium">Submission failed</span>
        </div>
        <p className="text-sm text-red-300/80 mt-1">{state.error}</p>
        <button
          type="button"
          onClick={onSubmit}
          disabled={state.isSubmitting}
          className="mt-2 px-3 py-1 text-sm bg-red-800/50 hover:bg-red-800/70 rounded text-red-200"
        >
          Try Again
        </button>
      </div>
    );
  }

  // Show submission button
  return (
    <div className="bg-blue-900/20 border border-blue-600/30 rounded-lg px-4 py-3 text-center">
      <p className="text-sm text-blue-200/80 mb-2">
        Your win can help improve the AI. Submit this game for training?
      </p>
      <button
        type="button"
        onClick={onSubmit}
        disabled={state.isSubmitting}
        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
          state.isSubmitting
            ? 'bg-blue-800/50 text-blue-300/50 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-500 text-white'
        }`}
      >
        {state.isSubmitting ? (
          <span className="flex items-center gap-2">
            <span className="animate-spin">&#8987;</span>
            Submitting...
          </span>
        ) : (
          'Submit for Training'
        )}
      </button>
    </div>
  );
}

/**
 * Share button that copies a challenge link to clipboard
 */
function ShareChallengeButton() {
  const [copied, setCopied] = useState(false);

  const handleShare = useCallback(async () => {
    const url = `${window.location.origin}/sandbox?preset=sq8-1h-1ai`;

    // Try Web Share API first (mobile)
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Play RingRift',
          text: 'Challenge me to a game of RingRift!',
          url,
        });
        return;
      } catch {
        // User cancelled or not supported, fall through to clipboard
      }
    }

    // Clipboard fallback
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = url;
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, []);

  return (
    <Button type="button" variant="secondary" size="lg" onClick={handleShare}>
      {copied ? 'Link Copied!' : 'Challenge a Friend'}
    </Button>
  );
}

/**
 * Share result to social platforms
 */
function ShareResultButtons({ title }: { title: string }) {
  const text = `${title} on RingRift — a strategy game with self-improving AI`;
  const url = 'https://ringrift.ai';

  return (
    <div className="flex gap-2 justify-center">
      <a
        href={`https://x.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-sm transition-colors"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
        </svg>
        Post
      </a>
      <a
        href={`https://www.reddit.com/submit?title=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white text-sm transition-colors"
      >
        <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z" />
        </svg>
        Share
      </a>
    </div>
  );
}

/**
 * Victory Modal Component
 */
export function VictoryModal({
  isOpen,
  gameResult,
  players,
  gameState,
  viewModel,
  gameEndExplanation,
  onClose,
  onReturnToLobby,
  onRematch,
  onRequestRematch,
  onAcceptRematch,
  onDeclineRematch,
  rematchStatus,
  currentUserId,
  isSandbox,
  onSubmitForTraining,
  trainingSubmission,
}: VictoryModalProps) {
  const overlaySessionIdRef = useRef<string | null>(null);
  const weirdStateImpressionLoggedRef = useRef<string | null>(null);
  const { currentTopic, isOpen: isTeachingOpen, showTopic, hideTopic } = useTeachingOverlay();
  const { colorVisionMode, effectiveReducedMotion } = useAccessibility();

  // Reset per-session impression tracking whenever the modal closes.
  useEffect(() => {
    if (!isOpen) {
      weirdStateImpressionLoggedRef.current = null;
    }
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
        colorVisionMode,
        gameEndExplanation,
      });

    if (!vm || !vm.isVisible) {
      return;
    }

    const effectiveExplanation = gameEndExplanation ?? null;
    const fallbackWeirdInfo = getWeirdStateReasonForGameResult(effectiveGameResultLocal);

    const reasonCode =
      effectiveExplanation?.weirdStateContext?.primaryReasonCode ??
      effectiveExplanation?.weirdStateContext?.reasonCodes?.[0] ??
      fallbackWeirdInfo?.reasonCode;

    if (!reasonCode) {
      return;
    }

    const rulesContext =
      (effectiveExplanation?.weirdStateContext?.rulesContextTags &&
      effectiveExplanation.weirdStateContext.rulesContextTags.length > 0
        ? effectiveExplanation.weirdStateContext.rulesContextTags[0]
        : undefined) ?? getRulesContextForReason(reasonCode);

    const weirdStateType = getWeirdStateTypeForReason(reasonCode);

    if (
      !isSurfaceableWeirdStateReason(reasonCode) ||
      !isSurfaceableWeirdStateType(weirdStateType)
    ) {
      return;
    }

    let overlaySessionId = overlaySessionIdRef.current;
    if (!overlaySessionId) {
      overlaySessionId = newOverlaySessionId();
      overlaySessionIdRef.current = overlaySessionId;
    }

    if (weirdStateImpressionLoggedRef.current === overlaySessionId) {
      return;
    }

    void logRulesUxEvent({
      type: 'weird_state_banner_impression',
      boardType: vm.gameSummary.boardType,
      numPlayers: vm.gameSummary.playerCount,
      rulesContext,
      source: 'victory_modal',
      weirdStateType,
      reasonCode,
      isRanked: gameState?.isRated ?? false,
      isSandbox: isSandbox ?? false,
      overlaySessionId,
    });
    weirdStateImpressionLoggedRef.current = overlaySessionId;
  }, [
    isOpen,
    gameResult,
    players,
    gameState,
    viewModel,
    gameEndExplanation,
    currentUserId,
    colorVisionMode,
    isSandbox,
  ]);

  if (!isOpen) return null;

  const effectivePlayers = players ?? [];
  const effectiveGameResult = gameResult ?? null;

  const effectiveViewModel: VictoryViewModel | null =
    viewModel ??
    toVictoryViewModel(effectiveGameResult, effectivePlayers, gameState, {
      currentUserId,
      isDismissed: false,
      colorVisionMode,
      gameEndExplanation,
    });

  if (!effectiveViewModel || !effectiveViewModel.isVisible) {
    return null;
  }

  const effectiveExplanation = gameEndExplanation ?? null;

  // Prefer weird-state reason codes / rules contexts from GameEndExplanation when
  // available; otherwise fall back to GameResult-based mapping.
  const fallbackWeirdInfo = getWeirdStateReasonForGameResult(effectiveGameResult);
  const reasonCode =
    effectiveExplanation?.weirdStateContext?.primaryReasonCode ??
    effectiveExplanation?.weirdStateContext?.reasonCodes?.[0] ??
    fallbackWeirdInfo?.reasonCode;

  const weirdStateInfo = reasonCode
    ? {
        reasonCode,
        rulesContext:
          (effectiveExplanation?.weirdStateContext?.rulesContextTags &&
          effectiveExplanation.weirdStateContext.rulesContextTags.length > 0
            ? effectiveExplanation.weirdStateContext.rulesContextTags[0]
            : undefined) ?? getRulesContextForReason(reasonCode),
        weirdStateType: getWeirdStateTypeForReason(reasonCode),
      }
    : null;

  const surfaceableWeirdStateInfo =
    weirdStateInfo &&
    isSurfaceableWeirdStateReason(weirdStateInfo.reasonCode) &&
    isSurfaceableWeirdStateType(weirdStateInfo.weirdStateType)
      ? weirdStateInfo
      : null;

  let weirdStateTeachingTopic: TeachingTopic | null = null;
  if (surfaceableWeirdStateInfo) {
    const teachingId = getTeachingTopicForReason(surfaceableWeirdStateInfo.reasonCode);
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

  const { title, description, finalStats, winner, gameSummary, userWon, userLost, isDraw } =
    effectiveViewModel;

  const teachingWeirdStateContext: WeirdStateOverlayContext | null =
    surfaceableWeirdStateInfo &&
    overlaySessionIdRef.current &&
    surfaceableWeirdStateInfo.rulesContext
      ? {
          reasonCode: surfaceableWeirdStateInfo.reasonCode,
          rulesContext: surfaceableWeirdStateInfo.rulesContext,
          weirdStateType: surfaceableWeirdStateInfo.weirdStateType,
          boardType: gameSummary.boardType,
          numPlayers: gameSummary.playerCount,
          isRanked: gameState?.isRated ?? false,
          isSandbox: isSandbox ?? false,
          overlaySessionId: overlaySessionIdRef.current,
        }
      : null;

  const handleWeirdStateHelpClick = () => {
    if (
      !surfaceableWeirdStateInfo ||
      !weirdStateTeachingTopic ||
      !surfaceableWeirdStateInfo.rulesContext
    ) {
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
      rulesContext: surfaceableWeirdStateInfo.rulesContext,
      source: 'victory_modal',
      weirdStateType: surfaceableWeirdStateInfo.weirdStateType,
      reasonCode: surfaceableWeirdStateInfo.reasonCode,
      isRanked: gameState?.isRated ?? false,
      isSandbox: isSandbox ?? false,
      overlaySessionId,
      topic: weirdStateTeachingTopic,
    });
  };

  const showConfetti =
    !effectiveReducedMotion && !isDraw && effectiveGameResult?.reason !== 'abandonment';

  return (
    <>
      <Dialog
        isOpen={isOpen}
        onClose={onClose}
        labelledBy="victory-title"
        describedBy="victory-description"
        closeOnBackdropClick={false}
        backdropClassName={`bg-black/70 backdrop-blur-sm ${
          effectiveReducedMotion ? '' : 'modal-backdrop-animate'
        }`.trim()}
        className="w-full max-w-3xl mx-4"
      >
        {/* Confetti particles for celebration */}
        {showConfetti && <ConfettiParticles />}

        <div
          className={`victory-modal bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-full p-6 space-y-6 relative overflow-hidden ${
            effectiveReducedMotion ? '' : 'modal-content-animate'
          }`.trim()}
          data-testid="victory-modal"
        >
          {/* Shimmer effect overlay for victories */}
          {showConfetti && (
            <div
              className="absolute top-0 left-0 right-0 bottom-0 w-full h-full victory-shimmer pointer-events-none rounded-xl"
              aria-hidden="true"
            />
          )}

          {/* Header with celebration animation */}
          <div className="text-center space-y-2 relative z-10">
            {/* Animated Trophy */}
            <AnimatedTrophy
              victoryCondition={effectiveGameResult?.reason ?? 'ring_elimination'}
              animate={!effectiveReducedMotion}
            />

            <h1
              id="victory-title"
              className={`text-4xl font-bold ${!effectiveReducedMotion ? 'winner-text-animate' : ''} ${
                userWon
                  ? 'text-green-400 winner-glow'
                  : userLost
                    ? 'text-red-400'
                    : 'text-slate-100'
              }`}
            >
              {title}
            </h1>
            <p
              id="victory-description"
              className={`text-slate-300 text-lg ${!effectiveReducedMotion ? 'winner-text-animate' : ''}`}
              style={!effectiveReducedMotion ? { animationDelay: '300ms' } : undefined}
            >
              {description}
            </p>
            {surfaceableWeirdStateInfo && weirdStateTeachingTopic && (
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

          {/* Victory Breakdown - explains why the winner won */}
          <div className={`${!effectiveReducedMotion ? 'stats-animate' : ''} relative z-10`.trim()}>
            <VictoryBreakdown
              gameResult={effectiveGameResult}
              winner={finalStats.find((s) => s.isWinner)}
              playerCount={gameSummary.playerCount}
            />
          </div>

          {/* Statistics Table with staggered animation */}
          <div className={`${!effectiveReducedMotion ? 'stats-animate' : ''} relative z-10`.trim()}>
            <FinalStatsTable stats={finalStats} winner={winner} />
          </div>

          {/* Game Details with staggered animation */}
          <div
            className={`${!effectiveReducedMotion ? 'summary-animate' : ''} relative z-10`.trim()}
          >
            <GameSummary summary={gameSummary} />
          </div>

          {/* Rematch section */}
          {(onRematch || onRequestRematch) && (
            <div
              className={`${!effectiveReducedMotion ? 'buttons-animate' : ''} flex justify-center relative z-10`.trim()}
            >
              <RematchSection
                rematchStatus={rematchStatus}
                onRematch={onRematch}
                onRequestRematch={onRequestRematch}
                onAcceptRematch={onAcceptRematch}
                onDeclineRematch={onDeclineRematch}
              />
            </div>
          )}

          {/* Training submission section (January 2026) - for human wins against AI in sandbox */}
          {onSubmitForTraining && trainingSubmission?.isAvailable && (
            <div
              className={`${!effectiveReducedMotion ? 'buttons-animate' : ''} relative z-10`.trim()}
            >
              <TrainingSubmissionSection
                onSubmit={onSubmitForTraining}
                state={trainingSubmission}
              />
            </div>
          )}

          {/* Action Buttons with staggered animation */}
          <div
            className={`${!effectiveReducedMotion ? 'buttons-animate' : ''} flex flex-col items-center gap-3 relative z-10`.trim()}
          >
            <div className="flex gap-3 justify-center flex-wrap">
              <Button type="button" size="lg" onClick={onReturnToLobby}>
                Return to Lobby
              </Button>

              <ShareChallengeButton />

              <Button type="button" variant="secondary" size="lg" onClick={onClose}>
                Close
              </Button>
            </div>
            <ShareResultButtons title={title} />
          </div>
        </div>
      </Dialog>

      {currentTopic && (
        <TeachingOverlay
          topic={currentTopic}
          isOpen={isTeachingOpen}
          onClose={hideTopic}
          position="center"
          weirdStateOverlayContext={teachingWeirdStateContext}
        />
      )}
    </>
  );
}
