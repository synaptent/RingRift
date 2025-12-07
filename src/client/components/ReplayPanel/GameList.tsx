/**
 * GameList - Paginated table of games from the replay database.
 */

import type { ReplayGameMetadata } from '../../types/replay';

export interface GameListProps {
  games: ReplayGameMetadata[];
  selectedGameId: string | null;
  onSelectGame: (gameId: string) => void;
  isLoading?: boolean;
  error?: string | null;
  /** Total count of games matching filters */
  total: number;
  /** Current page offset */
  offset: number;
  /** Page size */
  limit: number;
  /** Whether there are more results */
  hasMore: boolean;
  onPageChange: (newOffset: number) => void;
  className?: string;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatBoardType(boardType: string): string {
  const map: Record<string, string> = {
    square8: '8×8',
    square19: '19×19',
    hexagonal: 'Hex',
  };
  return map[boardType] ?? boardType;
}

function formatTermination(reason: string | null): string {
  if (!reason) return '—';
  const map: Record<string, string> = {
    ring_elimination: 'Ring Elim.',
    territory: 'Territory',
    last_player_standing: 'LPS',
    stalemate: 'Stalemate',
  };
  return map[reason] ?? reason;
}

export function GameList({
  games,
  selectedGameId,
  onSelectGame,
  isLoading = false,
  error = null,
  total,
  offset,
  limit,
  hasMore,
  onPageChange,
  className = '',
}: GameListProps) {
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(total / limit);

  if (error) {
    return <div className={`text-xs text-red-400 p-2 ${className}`}>Error: {error}</div>;
  }

  if (isLoading && games.length === 0) {
    return <div className={`text-xs text-slate-400 p-2 ${className}`}>Loading games...</div>;
  }

  if (games.length === 0) {
    return (
      <div className={`text-xs text-slate-400 p-2 ${className}`}>
        No games found. Try adjusting your filters or run some self-play games.
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Game list */}
      <div className="max-h-48 overflow-y-auto space-y-1">
        {games.map((game) => {
          const isSelected = game.gameId === selectedGameId;
          return (
            <button
              key={game.gameId}
              type="button"
              onClick={() => onSelectGame(game.gameId)}
              className={`w-full text-left px-2 py-1.5 rounded-lg text-xs transition ${
                isSelected
                  ? 'bg-emerald-900/40 border border-emerald-500/50 text-emerald-100'
                  : 'bg-slate-800/60 border border-slate-700 text-slate-200 hover:border-slate-500'
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-mono text-[10px] text-slate-400 truncate max-w-[60px]">
                    {game.gameId.slice(0, 8)}
                  </span>
                  <span className="px-1.5 py-0.5 rounded bg-slate-700/80 text-[10px]">
                    {formatBoardType(game.boardType as string)}
                  </span>
                  <span className="text-slate-400">{game.numPlayers}P</span>
                </div>
                <div className="flex items-center gap-2 text-[10px]">
                  {game.winner !== null && (
                    <span className="text-emerald-400">P{game.winner} won</span>
                  )}
                  <span className="text-slate-500">{game.totalMoves} moves</span>
                </div>
              </div>
              <div className="flex items-center justify-between gap-2 mt-0.5 text-[10px] text-slate-500">
                <span>{formatTermination(game.terminationReason)}</span>
                <span>{formatDate(game.createdAt)}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-700">
        <span>
          {total > 0
            ? `${offset + 1}–${Math.min(offset + games.length, total)} of ${total}`
            : '0 games'}
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => onPageChange(Math.max(0, offset - limit))}
            disabled={offset === 0}
            className="px-2 py-0.5 rounded border border-slate-600 disabled:opacity-40 disabled:cursor-not-allowed hover:border-slate-400"
            aria-label="Previous page"
          >
            ‹
          </button>
          <span className="px-2">
            {currentPage} / {totalPages || 1}
          </span>
          <button
            type="button"
            onClick={() => onPageChange(offset + limit)}
            disabled={!hasMore}
            className="px-2 py-0.5 rounded border border-slate-600 disabled:opacity-40 disabled:cursor-not-allowed hover:border-slate-400"
            aria-label="Next page"
          >
            ›
          </button>
        </div>
      </div>
    </div>
  );
}
