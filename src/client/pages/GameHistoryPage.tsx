import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { gameApi, type GameSummary } from '../services/api';
import { GameHistorySkeleton } from '../components/Skeleton';
import { formatVictoryReason } from '../adapters/gameViewModels';
import type { BoardType } from '../../shared/types/game';

const PAGE_SIZE = 20;

const BOARD_LABELS: Record<string, string> = {
  square8: 'Square 8x8',
  square19: 'Square 19x19',
  hex8: 'Hex Small',
  hexagonal: 'Hex Large',
};

type ResultFilter = 'all' | 'wins' | 'losses' | 'draws';

export default function GameHistoryPage() {
  useDocumentTitle('Game History');
  const { user } = useAuth();
  const [games, setGames] = useState<GameSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [boardFilter, setBoardFilter] = useState<BoardType | 'all'>('all');
  const [resultFilter, setResultFilter] = useState<ResultFilter>('all');

  const fetchGames = useCallback(async () => {
    if (!user) return;
    setIsLoading(true);
    try {
      const data = await gameApi.getUserGames(user.id, {
        limit: PAGE_SIZE,
        offset,
        status: 'completed',
      });
      setGames(data.games);
      setTotal(data.pagination.total);
    } catch (err) {
      console.error('Failed to fetch game history:', err);
    } finally {
      setIsLoading(false);
    }
  }, [user, offset]);

  useEffect(() => {
    fetchGames();
  }, [fetchGames]);

  // Client-side filtering (board type and result) applied on top of server data.
  // For a large game library the backend should support these filters, but for
  // now this is simple and functional.
  const filtered = games.filter((g) => {
    if (boardFilter !== 'all' && g.boardType !== boardFilter) return false;
    if (resultFilter !== 'all') {
      const isWin = g.winnerId === user?.id;
      const isDraw = g.status === 'completed' && !g.winnerId;
      if (resultFilter === 'wins' && !isWin) return false;
      if (resultFilter === 'losses' && (isWin || isDraw)) return false;
      if (resultFilter === 'draws' && !isDraw) return false;
    }
    return true;
  });

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  if (isLoading) {
    return <GameHistorySkeleton />;
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Game History</h1>
        <Link
          to="/profile"
          className="text-sm text-slate-400 hover:text-slate-200 transition-colors"
        >
          Back to Profile
        </Link>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <select
          value={boardFilter}
          onChange={(e) => setBoardFilter(e.target.value as BoardType | 'all')}
          className="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
        >
          <option value="all">All Boards</option>
          <option value="square8">Square 8x8</option>
          <option value="square19">Square 19x19</option>
          <option value="hex8">Hex Small</option>
          <option value="hexagonal">Hex Large</option>
        </select>

        <select
          value={resultFilter}
          onChange={(e) => setResultFilter(e.target.value as ResultFilter)}
          className="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-emerald-500"
        >
          <option value="all">All Results</option>
          <option value="wins">Wins</option>
          <option value="losses">Losses</option>
          <option value="draws">Draws</option>
        </select>

        <span className="ml-auto text-sm text-slate-500 self-center">{total} total games</span>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-12 text-slate-500 bg-slate-800/50 rounded-xl border border-slate-700">
          {games.length === 0 ? 'No completed games yet' : 'No games match the selected filters'}
        </div>
      ) : (
        <>
          <div className="space-y-2">
            {filtered.map((game) => {
              const isWin = game.winnerId === user?.id;
              const isDraw = game.status === 'completed' && !game.winnerId;
              const reason = game.outcome || game.resultReason;
              const reasonLabel = reason ? formatVictoryReason(reason as string) : null;

              return (
                <div
                  key={game.id}
                  className="p-4 bg-slate-800 rounded-lg border border-slate-700 flex items-center justify-between hover:border-slate-600 transition-colors"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span
                        className={`text-sm font-bold ${
                          isWin ? 'text-emerald-400' : isDraw ? 'text-slate-400' : 'text-red-400'
                        }`}
                      >
                        {isWin ? 'Victory' : isDraw ? 'Draw' : 'Defeat'}
                      </span>
                      <span className="text-slate-500 text-xs">|</span>
                      <span className="text-slate-300 text-sm">
                        {BOARD_LABELS[game.boardType] ?? game.boardType}
                      </span>
                      <span className="text-slate-500 text-xs">|</span>
                      <span className="text-slate-400 text-xs">
                        {game.playerCount ?? game.maxPlayers}p
                      </span>
                      {reasonLabel && (
                        <>
                          <span className="text-slate-500 text-xs">|</span>
                          <span className="text-slate-400 text-xs">{reasonLabel}</span>
                        </>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                      <span>{new Date(game.createdAt).toLocaleDateString()}</span>
                      <span>{game.moveCount} moves</span>
                      {game.isRated && <span className="text-amber-500/70 font-medium">Rated</span>}
                    </div>
                  </div>
                  <Link
                    to={`/game/${game.id}`}
                    className="ml-3 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-white text-xs font-medium rounded transition-colors flex-shrink-0"
                  >
                    View
                  </Link>
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 mt-6">
              <button
                type="button"
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                className="px-3 py-1.5 text-sm bg-slate-800 border border-slate-700 rounded text-slate-300 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Previous
              </button>
              <span className="text-sm text-slate-400">
                Page {currentPage} of {totalPages}
              </span>
              <button
                type="button"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
                className="px-3 py-1.5 text-sm bg-slate-800 border border-slate-700 rounded text-slate-300 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
