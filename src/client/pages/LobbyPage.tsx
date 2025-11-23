import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { io, Socket } from 'socket.io-client';
import { gameApi } from '../services/api';
import { BoardType, CreateGameRequest, Game } from '../../shared/types/game';
import LoadingSpinner from '../components/LoadingSpinner';
import { Button } from '../components/ui/Button';
import { Select } from '../components/ui/Select';

interface FormState {
  boardType: BoardType;
  maxPlayers: number;
  isRated: boolean;
  isPrivate: boolean;
  timeControlType: 'blitz' | 'rapid' | 'classical';
  initialTime: number;
  increment: number;
  aiCount: number;
  aiDifficulty: number;
  aiMode: 'local_heuristic' | 'service';
  aiType: 'random' | 'heuristic' | 'minimax' | 'mcts';
}

interface LobbyFilters {
  boardType?: BoardType | 'all';
  isRated?: boolean | 'all';
  playerCount?: number | 'all';
  searchTerm?: string;
  showInProgress?: boolean;
}

type SortOption = 'created' | 'players' | 'board_type' | 'rating';

const defaultForm: FormState = {
  boardType: 'square8',
  maxPlayers: 2,
  isRated: true,
  isPrivate: false,
  timeControlType: 'blitz',
  initialTime: 600,
  increment: 0,
  aiCount: 1,
  aiDifficulty: 5,
  aiMode: 'service',
  aiType: 'heuristic',
};

function getSocketBaseUrl(): string {
  const env =
    typeof process !== 'undefined' && (process as any).env
      ? ((process as any).env as Record<string, string | undefined>)
      : {};

  const wsUrl = env.VITE_WS_URL;
  if (wsUrl) return wsUrl.replace(/\/$/, '');

  const apiUrl = env.VITE_API_URL;
  if (apiUrl) {
    const base = apiUrl.replace(/\/?api\/?$/, '');
    return base.replace(/\/$/, '');
  }

  if (typeof window !== 'undefined' && window.location?.origin) {
    const origin = window.location.origin;
    if (origin.startsWith('http://localhost:5173') || origin.startsWith('https://localhost:5173')) {
      return 'http://localhost:3000';
    }
    return origin;
  }

  return 'http://localhost:3000';
}

function GameCard({
  game,
  onJoin,
  onSpectate,
  onCancel,
  currentUserId,
}: {
  game: Game;
  onJoin: (gameId: string) => void;
  onSpectate: (gameId: string) => void;
  onCancel: (gameId: string) => void;
  currentUserId: string;
}) {
  const isCreator = game.player1?.id === currentUserId;
  const playerCount = [game.player1Id, game.player2Id, game.player3Id, game.player4Id].filter(
    Boolean
  ).length;
  const isFull = playerCount >= game.maxPlayers;
  const canJoin = !isCreator && !isFull && game.status === 'waiting';

  return (
    <div className="bg-slate-800/70 rounded-xl shadow-md p-4 hover:shadow-lg transition-all border border-slate-700 hover:border-emerald-500/50">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <h4 className="font-semibold text-white">
            {`${game.player1?.username || 'Unknown'}'s Game`}
          </h4>
          {game.isRated && (
            <span className="px-2 py-0.5 bg-purple-500/20 text-purple-300 text-xs rounded border border-purple-500/30">
              Rated
            </span>
          )}
          {game.status === 'active' && (
            <span className="px-2 py-0.5 bg-green-500/20 text-green-300 text-xs rounded border border-green-500/30">
              In Progress
            </span>
          )}
        </div>

        <div className="text-sm text-slate-400">
          {playerCount}/{game.maxPlayers} players
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm text-slate-400 mb-3">
        <div>
          <span className="font-medium text-slate-300">Board:</span>{' '}
          <span className="text-slate-400">
            {game.boardType === 'square8'
              ? '8x8'
              : game.boardType === 'square19'
                ? '19x19'
                : 'Hexagonal'}
          </span>
        </div>
        {game.timeControl && (
          <div>
            <span className="font-medium text-slate-300">Time:</span>{' '}
            <span className="text-slate-400">
              {Math.floor(game.timeControl.initialTime / 60)}m +{game.timeControl.increment}s
            </span>
          </div>
        )}
        <div>
          <span className="font-medium text-slate-300">Rating:</span>{' '}
          <span className="text-slate-400">{game.player1?.rating ?? '?'}</span>
        </div>
      </div>

      <div className="flex gap-2">
        {canJoin && (
          <Button type="button" className="flex-1" onClick={() => onJoin(game.id)}>
            Join Game
          </Button>
        )}

        {isFull && game.status === 'waiting' && (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled
            className="flex-1 cursor-not-allowed"
          >
            Game Full
          </Button>
        )}

        {isCreator && game.status === 'waiting' && (
          <Button type="button" variant="danger" size="sm" onClick={() => onCancel(game.id)}>
            Cancel
          </Button>
        )}

        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={() => onSpectate(game.id)}
          title="Watch this game"
        >
          👁️ Watch
        </Button>
      </div>
    </div>
  );
}

function LobbyFilters({
  filters,
  onFilterChange,
}: {
  filters: LobbyFilters;
  onFilterChange: (filters: LobbyFilters) => void;
}) {
  return (
    <div className="bg-slate-800/70 rounded-xl shadow-md p-4 space-y-4 border border-slate-700 sticky top-4">
      <h3 className="font-semibold text-lg text-white">Filters</h3>

      {/* Board Type Filter */}
      <div>
        <label
          htmlFor="lobby-filter-board-type"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Board Type
        </label>
        <Select
          id="lobby-filter-board-type"
          value={filters.boardType ?? 'all'}
          onChange={(e) =>
            onFilterChange({
              ...filters,
              boardType: e.target.value === 'all' ? undefined : (e.target.value as BoardType),
            })
          }
        >
          <option value="all">All</option>
          <option value="square8">Square 8x8</option>
          <option value="square19">Square 19x19</option>
          <option value="hexagonal">Hexagonal</option>
        </Select>
      </div>

      {/* Rated Filter */}
      <div>
        <label
          htmlFor="lobby-filter-game-type"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Game Type
        </label>
        <Select
          id="lobby-filter-game-type"
          value={filters.isRated === undefined ? 'all' : filters.isRated.toString()}
          onChange={(e) =>
            onFilterChange({
              ...filters,
              isRated: e.target.value === 'all' ? undefined : e.target.value === 'true',
            })
          }
        >
          <option value="all">All</option>
          <option value="true">Rated Only</option>
          <option value="false">Unrated Only</option>
        </Select>
      </div>

      {/* Player Count Filter */}
      <div>
        <label
          htmlFor="lobby-filter-player-count"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Players
        </label>
        <Select
          id="lobby-filter-player-count"
          value={filters.playerCount ?? 'all'}
          onChange={(e) =>
            onFilterChange({
              ...filters,
              playerCount: e.target.value === 'all' ? undefined : parseInt(e.target.value, 10),
            })
          }
        >
          <option value="all">All</option>
          <option value="2">2 Players</option>
          <option value="3">3 Players</option>
          <option value="4">4 Players</option>
        </Select>
      </div>

      {/* Search */}
      <div>
        <label
          htmlFor="lobby-filter-search"
          className="block text-sm font-medium text-slate-300 mb-1"
        >
          Search
        </label>
        <input
          id="lobby-filter-search"
          type="text"
          className="w-full px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
          placeholder="Creator name or game ID..."
          value={filters.searchTerm ?? ''}
          onChange={(e) =>
            onFilterChange({
              ...filters,
              searchTerm: e.target.value || undefined,
            })
          }
        />
      </div>

      {/* Show In Progress */}
      <div className="flex items-center">
        <input
          type="checkbox"
          id="show-in-progress"
          checked={filters.showInProgress ?? false}
          onChange={(e) =>
            onFilterChange({
              ...filters,
              showInProgress: e.target.checked,
            })
          }
          className="mr-2 rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
        />
        <label htmlFor="show-in-progress" className="text-sm text-slate-300">
          Show games in progress
        </label>
      </div>

      {/* Clear Filters */}
      <Button
        type="button"
        variant="secondary"
        size="sm"
        fullWidth
        onClick={() => onFilterChange({})}
      >
        Clear All Filters
      </Button>
    </div>
  );
}

function SortControls({
  sortBy,
  onSortChange,
}: {
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-slate-400">Sort by:</span>
      <Select value={sortBy} onChange={(e) => onSortChange(e.target.value as SortOption)} size="sm">
        <option value="created">Newest First</option>
        <option value="players">Most Players</option>
        <option value="board_type">Board Type</option>
        <option value="rating">Rated First</option>
      </Select>
    </div>
  );
}

function EmptyLobby({
  hasFilters,
  onClearFilters,
  onCreate,
}: {
  hasFilters: boolean;
  onClearFilters: () => void;
  onCreate: () => void;
}) {
  if (hasFilters) {
    return (
      <div className="text-center py-16 bg-slate-800/40 rounded-xl border border-slate-700">
        <div className="text-6xl mb-4">🔍</div>
        <h3 className="text-xl font-semibold text-slate-200 mb-2">No games match your filters</h3>
        <p className="text-slate-400 mb-4">Try adjusting your filters or create a new game</p>
        <Button type="button" onClick={onClearFilters}>
          Clear Filters
        </Button>
      </div>
    );
  }

  return (
    <div className="text-center py-16 bg-slate-800/40 rounded-xl border border-slate-700">
      <div className="text-6xl mb-4">🎮</div>
      <h3 className="text-xl font-semibold text-slate-200 mb-2">No games available</h3>
      <p className="text-slate-400 mb-4">Be the first to create a game!</p>
      <Button type="button" onClick={onCreate}>
        Create Game
      </Button>
    </div>
  );
}

function filterGames(games: Game[], filters: LobbyFilters): Game[] {
  return games.filter((game) => {
    // Board type filter
    if (filters.boardType && game.boardType !== filters.boardType) {
      return false;
    }

    // Rated filter
    if (filters.isRated !== undefined && game.isRated !== filters.isRated) {
      return false;
    }

    // Player count filter
    if (filters.playerCount && game.maxPlayers !== filters.playerCount) {
      return false;
    }

    // Search term
    if (filters.searchTerm) {
      const term = filters.searchTerm.toLowerCase();
      const matchesCreator = game.player1?.username?.toLowerCase().includes(term);
      const matchesId = game.id.toLowerCase().includes(term);
      if (!matchesCreator && !matchesId) {
        return false;
      }
    }

    // Show in progress
    if (!filters.showInProgress && game.status !== 'waiting') {
      return false;
    }

    return true;
  });
}

function sortGames(games: Game[], sortBy: SortOption): Game[] {
  const sorted = [...games];

  switch (sortBy) {
    case 'created':
      return sorted.sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
    case 'players': {
      const getPlayerCount = (g: Game) =>
        [g.player1Id, g.player2Id, g.player3Id, g.player4Id].filter(Boolean).length;
      return sorted.sort((a, b) => getPlayerCount(b) - getPlayerCount(a));
    }
    case 'board_type':
      return sorted.sort((a, b) => a.boardType.localeCompare(b.boardType));
    case 'rating':
      return sorted.sort((a, b) => (b.isRated ? 1 : 0) - (a.isRated ? 1 : 0));
    default:
      return sorted;
  }
}

export default function LobbyPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>(defaultForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableGames, setAvailableGames] = useState<Game[]>([]);
  const [isLoadingGames, setIsLoadingGames] = useState(true);
  const [joinError, setJoinError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [filters, setFilters] = useState<LobbyFilters>({});
  const [sortBy, setSortBy] = useState<SortOption>('created');
  const socketRef = useRef<Socket | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string>('');

  // Get current user ID
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        setCurrentUserId(payload.userId || '');
      } catch (e) {
        console.error('Failed to parse token:', e);
      }
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchAvailableGames();
  }, []);

  // WebSocket real-time updates
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) return;

    const baseUrl = getSocketBaseUrl();
    const socket = io(baseUrl, {
      transports: ['websocket', 'polling'],
      auth: { token },
    });

    socketRef.current = socket;

    socket.on('connect', () => {
      console.log('Lobby WebSocket connected');
      socket.emit('lobby:subscribe');
    });

    socket.on('lobby:game_created', (game: Game) => {
      console.log('New game created:', game);
      setAvailableGames((prev) => [game, ...prev]);
    });

    socket.on(
      'lobby:game_joined',
      ({ gameId, playerCount }: { gameId: string; playerCount: number }) => {
        console.log('Game joined:', gameId, playerCount);
        setAvailableGames((prev) =>
          prev.map((g) => {
            if (g.id === gameId) {
              // Update player count (simplified - in production would refetch full game data)
              return { ...g };
            }
            return g;
          })
        );
      }
    );

    socket.on('lobby:game_started', ({ gameId }: { gameId: string }) => {
      console.log('Game started:', gameId);
      setAvailableGames((prev) => prev.filter((g) => g.id !== gameId));
    });

    socket.on('lobby:game_cancelled', ({ gameId }: { gameId: string }) => {
      console.log('Game cancelled:', gameId);
      setAvailableGames((prev) => prev.filter((g) => g.id !== gameId));
    });

    socket.on('connect_error', (err) => {
      console.error('Lobby socket error:', err);
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.emit('lobby:unsubscribe');
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, []);

  const fetchAvailableGames = async () => {
    try {
      setIsLoadingGames(true);
      const response = await gameApi.getAvailableGames();
      setAvailableGames(response.games);
    } catch (err) {
      console.error('Failed to fetch games:', err);
    } finally {
      setIsLoadingGames(false);
    }
  };

  const handleChange = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const payload: CreateGameRequest = {
        boardType: form.boardType,
        maxPlayers: form.maxPlayers,
        isRated: form.isRated,
        isPrivate: form.isPrivate,
        timeControl: {
          type: form.timeControlType,
          initialTime: form.initialTime,
          increment: form.increment,
        },
        aiOpponents:
          form.aiCount > 0
            ? {
                count: form.aiCount,
                difficulty: Array(form.aiCount).fill(form.aiDifficulty),
                mode: form.aiMode,
                aiType: form.aiType,
              }
            : undefined,
      };

      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
    } catch (err: any) {
      const message =
        err?.response?.data?.error?.message || err?.message || 'Failed to create game';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleJoinGame = async (gameId: string) => {
    try {
      setJoinError(null);
      await gameApi.joinGame(gameId);
      navigate(`/game/${gameId}`);
    } catch (err: any) {
      const message = err?.response?.data?.error?.message || err?.message || 'Failed to join game';
      setJoinError(message);
      fetchAvailableGames();
    }
  };

  const handleSpectate = (gameId: string) => {
    navigate(`/game/${gameId}`);
  };

  const handleCancelGame = async (gameId: string) => {
    try {
      await gameApi.leaveGame(gameId);
      setAvailableGames((prev) => prev.filter((g) => g.id !== gameId));
    } catch (err: any) {
      console.error('Failed to cancel game:', err);
    }
  };

  const filteredGames = filterGames(availableGames, filters);
  const sortedGames = sortGames(filteredGames, sortBy);
  const hasFilters = Object.keys(filters).some(
    (k) => filters[k as keyof LobbyFilters] !== undefined
  );

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <header className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-50">Game Lobby</h1>
          <p className="text-sm text-slate-400 mt-1">Find and join games in real-time</p>
        </div>
        <Button type="button" size="lg" onClick={() => setShowCreateForm(!showCreateForm)}>
          {showCreateForm ? '← Back to Lobby' : '+ Create Game'}
        </Button>
      </header>

      {showCreateForm ? (
        <section className="bg-slate-800/70 rounded-2xl border border-slate-700 p-6 shadow-xl">
          <h2 className="text-xl font-semibold text-white mb-4">Create Backend Game</h2>
          <p className="text-sm text-slate-300 mb-6">
            Choose board size, time control, and optional AI opponents. This creates a server-side
            game.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="p-3 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-lg">
                {error}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-slate-100">Board type</label>
                <Select
                  value={form.boardType}
                  onChange={(e) => handleChange('boardType', e.target.value as BoardType)}
                >
                  <option value="square8">8x8 (compact)</option>
                  <option value="square19">19x19 (full)</option>
                  <option value="hexagonal">Hexagonal</option>
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1 text-slate-100">Max players</label>
                <Select
                  value={form.maxPlayers}
                  onChange={(e) => handleChange('maxPlayers', Number(e.target.value))}
                >
                  {[2, 3, 4].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </Select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1 text-slate-100">
                  Time control
                </label>
                <div className="flex gap-2">
                  <Select
                    value={form.timeControlType}
                    onChange={(e) =>
                      handleChange(
                        'timeControlType',
                        e.target.value as FormState['timeControlType']
                      )
                    }
                  >
                    <option value="blitz">Blitz</option>
                    <option value="rapid">Rapid</option>
                    <option value="classical">Classical</option>
                  </Select>
                  <input
                    type="number"
                    min={60}
                    max={7200}
                    className="w-28 px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.initialTime}
                    onChange={(e) => handleChange('initialTime', Number(e.target.value))}
                    title="Initial time in seconds"
                  />
                  <input
                    type="number"
                    min={0}
                    max={60}
                    className="w-24 px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.increment}
                    onChange={(e) => handleChange('increment', Number(e.target.value))}
                    title="Increment in seconds per move"
                  />
                </div>
              </div>

              <div className="flex flex-col justify-center gap-2">
                <label className="inline-flex items-center text-sm text-slate-100">
                  <input
                    type="checkbox"
                    className="mr-2 rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
                    checked={form.isRated}
                    onChange={(e) => handleChange('isRated', e.target.checked)}
                  />
                  Rated game
                </label>
                <label className="inline-flex items-center text-sm text-slate-100">
                  <input
                    type="checkbox"
                    className="mr-2 rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
                    checked={form.isPrivate}
                    onChange={(e) => handleChange('isPrivate', e.target.checked)}
                  />
                  Private (not listed in public lobby)
                </label>
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <Button type="submit" size="sm" disabled={isSubmitting}>
                {isSubmitting ? 'Creating game…' : 'Create Game'}
              </Button>
            </div>
          </form>
        </section>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1">
            <LobbyFilters filters={filters} onFilterChange={setFilters} />
          </div>

          {/* Games List */}
          <div className="lg:col-span-3">
            <div className="flex justify-between items-center mb-4">
              <div className="text-slate-400 text-sm">
                {filteredGames.length} game
                {filteredGames.length !== 1 ? 's' : ''} available
              </div>
              <SortControls sortBy={sortBy} onSortChange={setSortBy} />
            </div>

            {joinError && (
              <div className="p-3 mb-4 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-lg">
                {joinError}
              </div>
            )}

            {isLoadingGames ? (
              <div className="flex justify-center py-16">
                <LoadingSpinner size="lg" />
              </div>
            ) : sortedGames.length === 0 ? (
              <EmptyLobby
                hasFilters={hasFilters}
                onClearFilters={() => setFilters({})}
                onCreate={() => setShowCreateForm(true)}
              />
            ) : (
              <div className="space-y-3">
                {sortedGames.map((game) => (
                  <GameCard
                    key={game.id}
                    game={game}
                    onJoin={handleJoinGame}
                    onSpectate={handleSpectate}
                    onCancel={handleCancelGame}
                    currentUserId={currentUserId}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
