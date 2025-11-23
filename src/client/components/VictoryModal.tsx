import React, { useEffect, useRef } from 'react';
import { GameResult, Player, GameState, positionToString } from '../../shared/types/game';

interface VictoryModalProps {
  isOpen: boolean;
  gameResult: GameResult | null;
  players: Player[];
  gameState?: GameState;
  onClose: () => void;
  onReturnToLobby: () => void;
  onRematch?: () => void;
  currentUserId?: string;
}

interface VictoryInfo {
  winner: Player | null;
  victoryCondition: GameResult['reason'];
  finalStats: PlayerStats[];
  isDraw: boolean;
  userWon: boolean;
  userLost: boolean;
}

interface PlayerStats {
  player: Player;
  ringsOnBoard: number;
  ringsEliminated: number;
  territorySpaces: number;
  totalMoves: number;
}

/**
 * Extract victory information from game state
 */
function extractVictoryInfo(
  gameResult: GameResult,
  players: Player[],
  gameState: GameState | undefined,
  currentUserId: string | undefined
): VictoryInfo {
  const winner =
    gameResult.winner !== undefined
      ? (players.find((p) => p.playerNumber === gameResult.winner) ?? null)
      : null;

  const finalStats = players.map((player) => {
    const ringsOnBoard = gameState
      ? countRingsOnBoard(player.playerNumber, gameState)
      : (gameResult.finalScore.ringsRemaining[player.playerNumber] ?? 0);

    const totalMoves = gameState ? countPlayerMoves(player.playerNumber, gameState) : 0;

    return {
      player,
      ringsOnBoard,
      ringsEliminated: gameResult.finalScore.ringsEliminated[player.playerNumber] ?? 0,
      territorySpaces: gameResult.finalScore.territorySpaces[player.playerNumber] ?? 0,
      totalMoves,
    };
  });

  return {
    winner,
    victoryCondition: gameResult.reason,
    finalStats,
    // Treat only explicit 'draw' results as draws so that abandonment
    // (and other no-winner outcomes) can use their dedicated messaging.
    isDraw: gameResult.reason === 'draw',
    userWon: !!(currentUserId && winner && winner.id === currentUserId),
    userLost: !!(
      currentUserId &&
      gameResult.winner !== undefined &&
      winner &&
      winner.id !== currentUserId
    ),
  };
}

/**
 * Count rings on board for a player
 */
function countRingsOnBoard(playerNumber: number, gameState: GameState): number {
  let count = 0;
  for (const stack of gameState.board.stacks.values()) {
    count += stack.rings.filter((r) => r === playerNumber).length;
  }
  return count;
}

/**
 * Count total moves made by a player
 */
function countPlayerMoves(playerNumber: number, gameState: GameState): number {
  // Use structured history if available, otherwise fall back to moveHistory
  if (gameState.history && gameState.history.length > 0) {
    return gameState.history.filter((entry) => entry.actor === playerNumber).length;
  }
  return gameState.moveHistory.filter((move) => move.player === playerNumber).length;
}

/**
 * Generate victory message based on condition
 */
function getVictoryMessage(info: VictoryInfo): { title: string; description: string } {
  if (info.isDraw) {
    return {
      title: '🤝 Draw!',
      description: 'The game ended in a stalemate with equal positions',
    };
  }

  const winnerName = info.winner?.username || `Player ${info.winner?.playerNumber || '?'}`;

  switch (info.victoryCondition) {
    case 'ring_elimination':
      return {
        title: `🏆 ${winnerName} Wins!`,
        description: 'Victory by eliminating all opponent rings',
      };
    case 'territory_control':
      return {
        title: `🏰 ${winnerName} Wins!`,
        description: 'Victory by controlling majority of the board',
      };
    case 'last_player_standing':
      return {
        title: `👑 ${winnerName} Wins!`,
        description: 'Victory as the last player remaining',
      };
    case 'timeout':
      return {
        title: `⏰ ${winnerName} Wins!`,
        description: 'Victory by opponent timeout',
      };
    case 'resignation':
      return {
        title: `${winnerName} Wins!`,
        description: 'Victory by opponent resignation',
      };
    case 'abandonment':
      return {
        title: 'Game Abandoned',
        description: 'The game was left in an unresolved state',
      };
    default:
      return {
        title: `${winnerName} Wins!`,
        description: 'Game over',
      };
  }
}

/**
 * Final statistics table component
 */
function FinalStatsTable({ stats, winner }: { stats: PlayerStats[]; winner: Player | null }) {
  // Sort by winner first, then by rings eliminated (descending)
  const sortedStats = [...stats].sort((a, b) => {
    if (winner) {
      if (a.player.playerNumber === winner.playerNumber) return -1;
      if (b.player.playerNumber === winner.playerNumber) return 1;
    }
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
                  {winner?.playerNumber === stat.player.playerNumber && (
                    <span className="text-yellow-500">👑</span>
                  )}
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
function GameSummary({ gameState }: { gameState: GameState | undefined }) {
  if (!gameState) return null;

  return (
    <div className="bg-slate-800/50 rounded-lg p-4 space-y-2 text-sm text-slate-200">
      <div className="flex justify-between">
        <span className="text-slate-400">Board Type:</span>
        <span className="font-semibold">{gameState.boardType}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Total Turns:</span>
        <span className="font-semibold">
          {gameState.history?.length || gameState.moveHistory?.length || 0}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-slate-400">Players:</span>
        <span className="font-semibold">{gameState.players.length}</span>
      </div>
      {gameState.isRated && (
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

  if (!isOpen || !gameResult) return null;

  const victoryInfo = extractVictoryInfo(gameResult, players, gameState, currentUserId);
  const { title, description } = getVictoryMessage(victoryInfo);

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="victory-title"
      aria-describedby="victory-description"
      onClick={handleBackdropClick}
    >
      <div className="victory-modal bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-3xl w-full mx-4 p-6 space-y-6">
        {/* Header with animation */}
        <div className="text-center space-y-2">
          <h1
            id="victory-title"
            className={`text-4xl font-bold ${
              victoryInfo.userWon
                ? 'text-green-400'
                : victoryInfo.userLost
                  ? 'text-red-400'
                  : 'text-slate-100'
            }`}
          >
            {title}
          </h1>
          <p id="victory-description" className="text-slate-300 text-lg">
            {description}
          </p>
        </div>

        {/* Statistics Table */}
        <FinalStatsTable stats={victoryInfo.finalStats} winner={victoryInfo.winner} />

        {/* Game Details */}
        <GameSummary gameState={gameState} />

        {/* Action Buttons */}
        <div className="flex gap-3 justify-center flex-wrap">
          <button
            onClick={onReturnToLobby}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Return to Lobby
          </button>

          {onRematch && (
            <button
              onClick={onRematch}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              Request Rematch
            </button>
          )}

          <button
            onClick={onClose}
            className="px-6 py-3 bg-slate-700 text-slate-100 rounded-lg hover:bg-slate-600 font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
