/**
 * Browser for viewing and loading recorded self-play games.
 *
 * Displays games from SQLite databases recorded during CMA-ES training,
 * self-play soaks, and other AI training activities.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { LoadableScenario } from '../sandbox/scenarioTypes';
import type { BoardType, Move, Position } from '../../shared/types/game';

// API response types matching the backend service
interface SelfPlayGameSummary {
  gameId: string;
  boardType: string;
  numPlayers: number;
  winner: number | null;
  totalMoves: number;
  totalTurns: number;
  createdAt: string;
  completedAt: string | null;
  source: string | null;
  terminationReason: string | null;
  durationMs: number | null;
}

interface DatabaseInfo {
  path: string;
  name: string;
  gameCount: number;
  createdAt: string | null;
}

interface SelfPlayGameDetail extends SelfPlayGameSummary {
  initialState: unknown;
  moves: Array<{
    moveNumber: number;
    turnNumber: number;
    player: number;
    phase: string;
    moveType: string;
    move: Move;
    thinkTimeMs: number | null;
    engineEval: number | null;
  }>;
  players: Array<{
    playerNumber: number;
    playerType: string;
    aiType: string | null;
    aiDifficulty: number | null;
    aiProfileId: string | null;
    finalEliminatedRings: number | null;
    finalTerritorySpaces: number | null;
  }>;
}

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

export interface SelfPlayBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectGame: (scenario: LoadableScenario) => void;
}

/**
 * Normalize a recorded move from the self-play database into the canonical
 * Move surface expected by the sandbox engine.
 *
 * Responsibilities:
 * - Map legacy `forced_elimination` records to `eliminate_rings_from_stack`
 *   so ClientSandboxEngine.applyCanonicalMove can consume them.
 * - Ensure timestamp is a Date instance.
 * - Provide a sane moveNumber/thinkTime when the recorder omitted them.
 */
function normalizeRecordedMove(rawMove: Move, fallbackMoveNumber: number): Move {
  const anyMove = rawMove as any;

  const type: Move['type'] =
    anyMove.type === 'forced_elimination' ? 'eliminate_rings_from_stack' : anyMove.type;

  const timestampRaw = anyMove.timestamp;
  const timestamp: Date =
    timestampRaw instanceof Date
      ? timestampRaw
      : typeof timestampRaw === 'string'
        ? new Date(timestampRaw)
        : new Date();

  const from: Position | undefined =
    anyMove.from && typeof anyMove.from === 'object' ? anyMove.from : undefined;

  const moveNumber =
    typeof anyMove.moveNumber === 'number' && Number.isFinite(anyMove.moveNumber)
      ? anyMove.moveNumber
      : fallbackMoveNumber;

  const thinkTime =
    typeof anyMove.thinkTime === 'number'
      ? anyMove.thinkTime
      : typeof anyMove.thinkTimeMs === 'number'
        ? anyMove.thinkTimeMs
        : 0;

  return {
    ...anyMove,
    type,
    from,
    timestamp,
    thinkTime,
    moveNumber,
  } as Move;
}

export const SelfPlayBrowser: React.FC<SelfPlayBrowserProps> = ({
  isOpen,
  onClose,
  onSelectGame,
}) => {
  const [databases, setDatabases] = useState<DatabaseInfo[]>([]);
  const [selectedDb, setSelectedDb] = useState<string | null>(null);
  const [games, setGames] = useState<SelfPlayGameSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingGame, setLoadingGame] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [boardTypeFilter, setBoardTypeFilter] = useState<BoardType | 'all'>('all');
  const [playerCountFilter, setPlayerCountFilter] = useState<number | 'all'>('all');
  const [sourceFilter, setSourceFilter] = useState<string | 'all'>('all');
  const [hasWinnerFilter, setHasWinnerFilter] = useState<boolean | 'all'>('all');

  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Load available databases
  useEffect(() => {
    if (!isOpen) return;

    const loadDatabases = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch('/api/selfplay/databases');

        if (!response.ok) {
          // Attempt to parse an error payload, but tolerate empty/non‑JSON bodies.
          let errorMessage = `Failed to load databases (HTTP ${response.status})`;
          try {
            const data = await response.json();
            if (data && typeof data === 'object' && 'error' in data && data.error) {
              errorMessage = String((data as any).error);
            }
          } catch {
            // Ignore JSON parse errors for non‑JSON error bodies.
          }
          setError(errorMessage);
          return;
        }

        const data = await response.json();
        if (data.success) {
          setDatabases(data.databases);
          if (data.databases.length > 0 && !selectedDb) {
            setSelectedDb(data.databases[0].path);
          }
        } else {
          setError(data.error || 'Failed to load databases');
        }
      } catch (err) {
        setError('Failed to connect to server');
        console.error('Failed to load databases:', err);
      } finally {
        setLoading(false);
      }
    };

    loadDatabases();
  }, [isOpen, selectedDb]);

  // Load games when database selection or filters change
  useEffect(() => {
    if (!selectedDb) {
      setGames([]);
      return;
    }

    const loadGames = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({ db: selectedDb, limit: '100' });
        if (boardTypeFilter !== 'all') params.set('boardType', boardTypeFilter);
        if (playerCountFilter !== 'all') params.set('numPlayers', String(playerCountFilter));
        if (sourceFilter !== 'all') params.set('source', sourceFilter);
        if (hasWinnerFilter !== 'all') params.set('hasWinner', String(hasWinnerFilter));

        const response = await fetch(`/api/selfplay/games?${params}`);

        if (!response.ok) {
          let errorMessage = `Failed to load games (HTTP ${response.status})`;
          try {
            const data = await response.json();
            if (data && typeof data === 'object' && 'error' in data && data.error) {
              errorMessage = String((data as any).error);
            }
          } catch {
            // Ignore JSON parse errors for non‑JSON error bodies.
          }
          setError(errorMessage);
          return;
        }

        const data = await response.json();
        if (data.success) {
          setGames(data.games);
        } else {
          setError(data.error || 'Failed to load games');
        }
      } catch (err) {
        setError('Failed to load games');
        console.error('Failed to load games:', err);
      } finally {
        setLoading(false);
      }
    };

    loadGames();
  }, [selectedDb, boardTypeFilter, playerCountFilter, sourceFilter, hasWinnerFilter]);

  // Focus trap
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    if (!dialogEl) return;

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (first) {
      first.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }

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
    return () => dialogEl.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleLoadGame = useCallback(
    async (game: SelfPlayGameSummary) => {
      if (!selectedDb) return;

      setLoadingGame(game.gameId);
      setError(null);

      try {
        const response = await fetch(
          `/api/selfplay/games/${encodeURIComponent(game.gameId)}?db=${encodeURIComponent(selectedDb)}`
        );

        if (!response.ok) {
          let errorMessage = `Failed to load game (HTTP ${response.status})`;
          try {
            const maybeData = await response.json();
            if (
              maybeData &&
              typeof maybeData === 'object' &&
              'error' in maybeData &&
              (maybeData as any).error
            ) {
              errorMessage = String((maybeData as any).error);
            }
          } catch {
            // Ignore JSON parse errors when the server returned a non‑JSON body.
          }
          setError(errorMessage);
          return;
        }

        const data = await response.json();

        if (!data.success) {
          setError(data.error || 'Failed to load game');
          return;
        }

        const detail: SelfPlayGameDetail = data.game;

        // Sanitize the initial state for sandbox consumption. Some self-play
        // databases (especially CMA-ES iterative runs) record initial_state_json
        // as a mid-game snapshot that already includes a non-empty moveHistory,
        // while the game_moves table only contains the suffix of moves from
        // that point onward. For /sandbox replay we want:
        // - move index 0 to represent "this snapshot as-is", and
        // - the recorded moves list to be the full canonical sequence from
        //   that snapshot forward.
        //
        // To avoid double-counting earlier moves, we drop any pre-populated
        // moveHistory/history from the serialized state here and treat the
        // DB's moves array as the complete canonical sequence for replay.
        const rawState = detail.initialState as any;
        const sanitizedState =
          rawState && typeof rawState === 'object' ? { ...rawState } : rawState;
        if (sanitizedState && Array.isArray(sanitizedState.moveHistory)) {
          sanitizedState.moveHistory = [];
        }
        if (sanitizedState && Array.isArray(sanitizedState.history)) {
          sanitizedState.history = [];
        }

        // Convert to LoadableScenario format. We attach selfPlayMeta so hosts
        // can distinguish recorded self-play scenarios from other vectors and
        // optionally wire them into the ReplayPanel / history playback.
        const scenario: LoadableScenario = {
          id: `selfplay-${detail.gameId}`,
          name: `${detail.source || 'Self-Play'} Game (${detail.boardType}, ${detail.numPlayers}P)`,
          description: `Recorded ${formatDate(detail.createdAt)}. ${detail.totalMoves} moves, ${formatDuration(detail.durationMs)}. ${detail.winner !== null ? `Winner: P${detail.winner}` : 'Draw/Incomplete'}`,
          category: 'custom',
          tags: [detail.source || 'selfplay', detail.terminationReason || 'unknown'].filter(
            Boolean
          ),
          boardType: detail.boardType as BoardType,
          playerCount: detail.numPlayers,
          createdAt: detail.createdAt,
          source: 'custom',
          state: sanitizedState as LoadableScenario['state'],
          selfPlayMeta: {
            dbPath: selectedDb,
            gameId: detail.gameId,
            totalMoves: detail.totalMoves,
            // Include a canonicalized move sequence so the sandbox host can
            // reconstruct the full game trajectory locally via the
            // ClientSandboxEngine without needing an extra round-trip to
            // the self-play service. Legacy forced_elimination records are
            // mapped to eliminate_rings_from_stack for compatibility with
            // the canonical Move surface.
            moves: detail.moves.map((m) => normalizeRecordedMove(m.move, m.moveNumber)),
          },
        };

        onSelectGame(scenario);
        onClose();
      } catch (err) {
        setError('Failed to load game details');
        console.error('Failed to load game:', err);
      } finally {
        setLoadingGame(null);
      }
    },
    [selectedDb, onSelectGame, onClose]
  );

  if (!isOpen) return null;

  // Extract unique sources from games for filter dropdown
  const uniqueSources = Array.from(
    new Set(games.map((g) => g.source).filter((s): s is string => Boolean(s)))
  );

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="selfplay-browser-title"
    >
      <div
        ref={dialogRef}
        className="bg-slate-900 rounded-2xl border border-slate-700 w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl"
      >
        {/* Header */}
        <div className="p-4 border-b border-slate-700 flex justify-between items-center">
          <h2 id="selfplay-browser-title" className="text-xl font-bold text-white">
            Self-Play Game Browser
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-1"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Database Selector */}
        <div className="p-3 border-b border-slate-700">
          <label className="block text-sm text-slate-400 mb-1">Database</label>
          <select
            value={selectedDb || ''}
            onChange={(e) => setSelectedDb(e.target.value || null)}
            className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            disabled={databases.length === 0}
          >
            {databases.length === 0 ? (
              <option value="">No databases found</option>
            ) : (
              databases.map((db) => (
                <option key={db.path} value={db.path}>
                  {db.name} ({db.gameCount} games)
                </option>
              ))
            )}
          </select>
        </div>

        {/* Filters */}
        <div className="p-3 border-b border-slate-700 flex gap-3 flex-wrap">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Board Type</label>
            <select
              value={boardTypeFilter}
              onChange={(e) => setBoardTypeFilter(e.target.value as BoardType | 'all')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="all">All</option>
              <option value="square8">Square 8x8</option>
              <option value="square19">Square 19x19</option>
              <option value="hex">Hex</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Players</label>
            <select
              value={playerCountFilter}
              onChange={(e) =>
                setPlayerCountFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))
              }
              className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="all">All</option>
              <option value="2">2P</option>
              <option value="3">3P</option>
              <option value="4">4P</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Source</label>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="all">All</option>
              {uniqueSources.map((src) => (
                <option key={src} value={src}>
                  {src}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Outcome</label>
            <select
              value={String(hasWinnerFilter)}
              onChange={(e) =>
                setHasWinnerFilter(e.target.value === 'all' ? 'all' : e.target.value === 'true')
              }
              className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
            >
              <option value="all">All</option>
              <option value="true">Has Winner</option>
              <option value="false">Draw/Incomplete</option>
            </select>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mx-4 mt-3 p-3 rounded-lg bg-red-900/30 border border-red-700 text-red-300 text-sm">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-400 hover:text-red-200">
              Dismiss
            </button>
          </div>
        )}

        {/* Game List */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center text-slate-400 py-8">Loading games...</div>
          ) : games.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              {selectedDb ? 'No games match your filters.' : 'Select a database to browse games.'}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-sm text-slate-400 mb-3">
                Showing {games.length} game{games.length !== 1 ? 's' : ''}
              </div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-slate-700">
                    <th className="py-2 px-2">Board</th>
                    <th className="py-2 px-2">Players</th>
                    <th className="py-2 px-2">Moves</th>
                    <th className="py-2 px-2">Winner</th>
                    <th className="py-2 px-2">Source</th>
                    <th className="py-2 px-2">Date</th>
                    <th className="py-2 px-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {games.map((game) => (
                    <tr
                      key={game.gameId}
                      className="border-b border-slate-800 hover:bg-slate-800/50"
                    >
                      <td className="py-2 px-2 text-white">{game.boardType}</td>
                      <td className="py-2 px-2 text-white">{game.numPlayers}P</td>
                      <td className="py-2 px-2 text-slate-300">{game.totalMoves}</td>
                      <td className="py-2 px-2">
                        {game.winner !== null ? (
                          <span className="text-emerald-400">P{game.winner}</span>
                        ) : (
                          <span className="text-slate-500">-</span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-slate-400">{game.source || '-'}</td>
                      <td className="py-2 px-2 text-slate-400">{formatDate(game.createdAt)}</td>
                      <td className="py-2 px-2">
                        <button
                          onClick={() => handleLoadGame(game)}
                          disabled={loadingGame === game.gameId}
                          className="px-3 py-1 text-xs rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium transition-colors"
                        >
                          {loadingGame === game.gameId ? 'Loading...' : 'Load'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

function formatDate(isoDate: string): string {
  try {
    const date = new Date(isoDate);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return isoDate;
  }
}

function formatDuration(ms: number | null): string {
  if (ms === null) return 'unknown duration';
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

export default SelfPlayBrowser;
