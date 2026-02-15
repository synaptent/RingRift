import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { io, Socket } from 'socket.io-client';
import { gameApi } from '../services/api';
import { BoardType, CreateGameRequest, Game } from '../../shared/types/game';
import LoadingSpinner from '../components/LoadingSpinner';
import { Button } from '../components/ui/Button';
import { Dialog } from '../components/ui/Dialog';
import { InlineAlert } from '../components/ui/InlineAlert';
import { Select } from '../components/ui/Select';
import type { ClientToServerEvents, ServerToClientEvents } from '../../shared/types/websocket';
import { readEnv } from '../../shared/utils/envFlags';
import { extractErrorMessage } from '../utils/errorReporting';
import { DIFFICULTY_DESCRIPTORS, getDifficultyDescriptor } from '../utils/difficultyUx';
import {
  sendDifficultyCalibrationEvent,
  storeDifficultyCalibrationSession,
} from '../utils/difficultyCalibrationTelemetry';
import { useMatchmaking } from '../hooks/useMatchmaking';
import { QueueStatus } from '../components/QueueStatus';
import { AIQuickPlayPanel, AIQuickPlayOption } from '../components/AIQuickPlayPanel';
import { getDifficultyAiType } from '../config/aiQuickPlay';

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
  /** When true, create a canonical Square-8 2-player calibration game vs AI. */
  isCalibrationGame: boolean;
  /** Enable pie rule (swap sides) for 2-player games. Off by default. */
  pieRuleEnabled: boolean;
}

interface LobbyFilters {
  boardType?: BoardType | 'all' | undefined;
  isRated?: boolean | 'all' | undefined;
  playerCount?: number | 'all' | undefined;
  searchTerm?: string | undefined;
  showInProgress?: boolean | undefined;
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
  isCalibrationGame: false,
  pieRuleEnabled: false,
};

function getSocketBaseUrl(): string {
  const wsUrl = readEnv('VITE_WS_URL');
  if (wsUrl) return wsUrl.replace(/\/$/, '');

  const apiUrl = readEnv('VITE_API_URL');
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

function CopyInviteLinkButton({ inviteCode }: { inviteCode: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    const url = `${window.location.origin}/join/${inviteCode}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for non-HTTPS contexts
      const textArea = document.createElement('textarea');
      textArea.value = url;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [inviteCode]);

  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={handleCopy}
      title="Copy invite link to clipboard"
    >
      {copied ? 'Copied!' : 'Copy Invite Link'}
    </Button>
  );
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
                : game.boardType === 'hex8'
                  ? 'Hex 8'
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

        {isCreator && game.status === 'waiting' && game.inviteCode && (
          <CopyInviteLinkButton inviteCode={game.inviteCode} />
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
          <option value="hex8">Hex 8 (small)</option>
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
  useDocumentTitle(
    'Lobby',
    'Create or join live multiplayer RingRift games. Play rated matches on square and hexagonal boards.'
  );
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
  const [showDifficultyInfo, setShowDifficultyInfo] = useState(false);
  const difficultyInfoCloseButtonRef = useRef<HTMLButtonElement | null>(null);
  const socketRef = useRef<Socket<ServerToClientEvents, ClientToServerEvents> | null>(null);
  const [currentUserId, setCurrentUserId] = useState<string>('');
  const [showFindMatchForm, setShowFindMatchForm] = useState(false);
  const [matchmakingBoardType, setMatchmakingBoardType] = useState<BoardType>('square8');

  // Matchmaking hook
  const {
    inQueue,
    estimatedWaitTime,
    queuePosition,
    searchCriteria,
    matchFound,
    error: matchmakingError,
    joinQueue,
    leaveQueue,
  } = useMatchmaking();

  const selectedDifficultyDescriptor =
    getDifficultyDescriptor(form.aiDifficulty) ??
    getDifficultyDescriptor(5) ??
    DIFFICULTY_DESCRIPTORS.find((d) => d.id === 5);

  const _difficultyOptions = form.isCalibrationGame
    ? DIFFICULTY_DESCRIPTORS.filter((d) => d.id === 2 || d.id === 4 || d.id === 6 || d.id === 8)
    : DIFFICULTY_DESCRIPTORS; // Reserved for future difficulty selection UI

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
    }) as Socket<ServerToClientEvents, ClientToServerEvents>;

    socketRef.current = socket;

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');
    });

    socket.on('lobby:game_created', (game: Game) => {
      setAvailableGames((prev) => [game, ...prev]);
    });

    socket.on(
      'lobby:game_joined',
      ({ gameId, playerCount: _playerCount }: { gameId: string; playerCount: number }) => {
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
      setAvailableGames((prev) => prev.filter((g) => g.id !== gameId));
    });

    socket.on('lobby:game_cancelled', ({ gameId }: { gameId: string }) => {
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

  const createGameFromForm = async (formState: FormState) => {
    const isCalibration = formState.isCalibrationGame;
    const effectiveBoardType: BoardType = isCalibration ? 'square8' : formState.boardType;
    const effectiveMaxPlayers = isCalibration ? 2 : formState.maxPlayers;
    const effectiveAiCount = isCalibration ? 1 : formState.aiCount;

    let effectiveAiDifficulty = formState.aiDifficulty;
    if (
      isCalibration &&
      effectiveAiDifficulty !== 2 &&
      effectiveAiDifficulty !== 4 &&
      effectiveAiDifficulty !== 6 &&
      effectiveAiDifficulty !== 8
    ) {
      effectiveAiDifficulty = 4;
    }

    const isAiGame = effectiveAiCount > 0;
    const isRated = isAiGame ? false : formState.isRated;

    const payload: CreateGameRequest = {
      boardType: effectiveBoardType,
      maxPlayers: effectiveMaxPlayers,
      isRated,
      isPrivate: formState.isPrivate,
      timeControl: {
        type: formState.timeControlType,
        initialTime: formState.initialTime,
        increment: formState.increment,
      },
      ...(effectiveAiCount > 0
        ? {
            aiOpponents: {
              count: effectiveAiCount,
              difficulty: Array(effectiveAiCount).fill(effectiveAiDifficulty),
              mode: formState.aiMode,
              aiType: formState.aiType,
            },
          }
        : {}),
      // Pie rule (swap sides) is opt-in for 2-player games
      ...(effectiveMaxPlayers === 2 && formState.pieRuleEnabled
        ? { rulesOptions: { swapRuleEnabled: true } }
        : {}),
      ...(isCalibration
        ? {
            isCalibrationGame: true,
            calibrationDifficulty: effectiveAiDifficulty,
          }
        : {}),
    };

    const game = await gameApi.createGame(payload);

    if (isCalibration) {
      storeDifficultyCalibrationSession(game.id, {
        boardType: effectiveBoardType,
        numPlayers: effectiveMaxPlayers,
        difficulty: effectiveAiDifficulty,
        isCalibrationOptIn: true,
      });

      void sendDifficultyCalibrationEvent({
        type: 'difficulty_calibration_game_started',
        boardType: effectiveBoardType,
        numPlayers: effectiveMaxPlayers,
        difficulty: effectiveAiDifficulty,
        isCalibrationOptIn: true,
      });
    }

    return game;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const game = await createGameFromForm(form);
      navigate(`/game/${game.id}`, { state: { inviteCode: game.inviteCode } });
    } catch (error: unknown) {
      setError(extractErrorMessage(error, 'Failed to create game'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleGuidedIntro = async () => {
    setError(null);
    setShowCreateForm(true);

    const guidedForm: FormState = {
      ...defaultForm,
      boardType: 'square8',
      maxPlayers: 2,
      isRated: false,
      isPrivate: false,
      timeControlType: 'rapid',
      initialTime: 600,
      increment: 5,
      aiCount: 1,
      aiDifficulty: 2,
      aiMode: 'service',
      aiType: 'heuristic',
      isCalibrationGame: true,
    };

    setForm(guidedForm);
    setIsSubmitting(true);

    try {
      const game = await createGameFromForm(guidedForm);
      navigate(`/game/${game.id}`);
    } catch (error: unknown) {
      setError(extractErrorMessage(error, 'Failed to create guided intro game'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleJoinGame = async (gameId: string) => {
    try {
      setJoinError(null);
      await gameApi.joinGame(gameId);
      navigate(`/game/${gameId}`);
    } catch (error: unknown) {
      setJoinError(extractErrorMessage(error, 'Failed to join game'));
      fetchAvailableGames();
    }
  };

  const handleSpectate = (gameId: string) => {
    // Spectate via the dedicated read-only spectator route so that backend
    // games treat this viewer as a spectator rather than a participant.
    navigate(`/spectate/${gameId}`);
  };

  const handleCancelGame = async (gameId: string) => {
    try {
      await gameApi.leaveGame(gameId);
      setAvailableGames((prev) => prev.filter((g) => g.id !== gameId));
    } catch (error: unknown) {
      // Best-effort cancellation; log but don't surface to user
      if (typeof window !== 'undefined' && 'console' in window) {
        console.error('Failed to cancel game:', error);
      }
    }
  };

  const handleAIQuickPlay = async (option: AIQuickPlayOption) => {
    try {
      setIsSubmitting(true);
      setError(null);
      const payload: CreateGameRequest = {
        boardType: option.boardType,
        maxPlayers: option.numPlayers,
        isRated: false,
        isPrivate: false,
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 15,
        },
        aiOpponents: {
          count: option.numPlayers - 1,
          difficulty: Array(option.numPlayers - 1).fill(option.difficulty),
          mode: 'service',
          aiType: getDifficultyAiType(option.difficulty) as
            | 'random'
            | 'heuristic'
            | 'minimax'
            | 'mcts',
        },
      };
      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
    } catch (error: unknown) {
      setError(extractErrorMessage(error, 'Failed to start AI game'));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFindMatch = () => {
    // Get user rating from localStorage or default to 1200
    const storedRating = localStorage.getItem('userRating');
    const userRating = storedRating ? parseInt(storedRating, 10) : 1200;
    const ratingRange = 200; // +/- 200 rating points

    joinQueue({
      boardType: matchmakingBoardType,
      ratingRange: {
        min: Math.max(0, userRating - ratingRange),
        max: userRating + ratingRange,
      },
      timeControl: {
        min: 300, // 5 minutes minimum
        max: 1800, // 30 minutes maximum
      },
    });
    setShowFindMatchForm(false);
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
          <h1 className="text-3xl font-bold tracking-tight text-slate-50 flex items-center gap-2">
            <img src="/ringrift-icon.png" alt="RingRift" className="w-8 h-8" />
            Game Lobby
          </h1>
          <p className="text-sm text-slate-400 mt-1">Find and join games in real-time</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <p className="text-[11px] text-slate-400 text-right max-w-xs">
            New to RingRift?{' '}
            <span className="font-semibold text-slate-200">Guided Intro vs AI</span> starts an easy
            8x8 game against a gentle AI and contributes to difficulty calibration.
          </p>
          <div className="flex items-center gap-3">
            <Button type="button" size="lg" onClick={() => setShowCreateForm(!showCreateForm)}>
              {showCreateForm ? '← Back to Lobby' : '+ Create Game'}
            </Button>
            <Button
              type="button"
              size="lg"
              variant="outline"
              onClick={() => setShowFindMatchForm(!showFindMatchForm)}
              disabled={inQueue}
            >
              {inQueue ? 'In Queue...' : 'Find Match'}
            </Button>
            <Button
              type="button"
              size="lg"
              variant="secondary"
              onClick={handleGuidedIntro}
              data-testid="guided-intro-button"
            >
              Guided Intro vs AI
            </Button>
          </div>
        </div>
      </header>

      {/* Queue Status - shown when in matchmaking queue */}
      {(inQueue || matchFound) && (
        <QueueStatus
          inQueue={inQueue}
          estimatedWaitTime={estimatedWaitTime}
          queuePosition={queuePosition}
          searchCriteria={searchCriteria}
          matchFound={matchFound}
          onLeaveQueue={leaveQueue}
        />
      )}

      {/* Matchmaking error */}
      {matchmakingError && <InlineAlert variant="error">{matchmakingError}</InlineAlert>}

      {/* Find Match Form */}
      {showFindMatchForm && !inQueue && (
        <section className="bg-slate-800/70 rounded-2xl border border-slate-700 p-6 shadow-xl">
          <h2 className="text-xl font-semibold text-white mb-4">Find a Match</h2>
          <p className="text-sm text-slate-300 mb-6">
            Join the matchmaking queue to be paired with an opponent of similar skill level.
          </p>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1 text-slate-100">Board Type</label>
              <Select
                value={matchmakingBoardType}
                onChange={(e) => setMatchmakingBoardType(e.target.value as BoardType)}
              >
                <option value="square8">8x8 (compact)</option>
                <option value="square19">19x19 (full)</option>
                <option value="hex8">Hex 8 (small)</option>
                <option value="hexagonal">Hexagonal</option>
              </Select>
            </div>

            <div className="flex gap-3">
              <Button type="button" onClick={handleFindMatch}>
                Start Searching
              </Button>
              <Button type="button" variant="secondary" onClick={() => setShowFindMatchForm(false)}>
                Cancel
              </Button>
            </div>
          </div>
        </section>
      )}

      {showCreateForm ? (
        <section className="bg-slate-800/70 rounded-2xl border border-slate-700 p-6 shadow-xl">
          <h2 className="text-xl font-semibold text-white mb-4">Create Backend Game</h2>
          <p className="text-sm text-slate-300 mb-6">
            Choose board size, time control, and optional AI opponents. This creates a server-side
            game.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && <InlineAlert variant="error">{error}</InlineAlert>}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1 text-slate-100">Board type</label>
                <Select
                  value={form.boardType}
                  onChange={(e) => handleChange('boardType', e.target.value as BoardType)}
                >
                  <option value="square8">8x8 (compact) — Recommended for new players</option>
                  <option value="square19">19x19 (full)</option>
                  <option value="hex8">Hex 8 (small)</option>
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
                <p className="mt-1 text-[11px] text-slate-400">
                  {form.maxPlayers === 2 && '2-player: Pure strategy, no politics'}
                  {form.maxPlayers === 3 && '3-player: Moderate politics, dynamic alliances'}
                  {form.maxPlayers === 4 && '4-player: High politics, kingmaking possible'}
                </p>
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
                    inputMode="numeric"
                    className="w-28 px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.initialTime}
                    onChange={(e) => handleChange('initialTime', Number(e.target.value))}
                    title="Initial time in seconds"
                    aria-label="Initial time (seconds)"
                  />
                  <input
                    type="number"
                    min={0}
                    max={60}
                    inputMode="numeric"
                    className="w-24 px-3 py-2 rounded-lg bg-slate-900 border border-slate-600 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.increment}
                    onChange={(e) => handleChange('increment', Number(e.target.value))}
                    title="Increment in seconds per move"
                    aria-label="Increment per move (seconds)"
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
                {form.maxPlayers === 2 && (
                  <label
                    className="inline-flex items-center text-sm text-slate-100"
                    title="Pie rule: After Player 1's first turn, Player 2 may swap sides"
                  >
                    <input
                      type="checkbox"
                      className="mr-2 rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
                      checked={form.pieRuleEnabled}
                      onChange={(e) => handleChange('pieRuleEnabled', e.target.checked)}
                    />
                    Pie rule (swap sides)
                  </label>
                )}
              </div>
            </div>

            <div className="mt-4 border-t border-slate-700 pt-4 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-sm font-semibold text-slate-100">AI opponent</h3>
                <button
                  type="button"
                  className="text-xs text-slate-300 hover:text-emerald-300 underline underline-offset-2"
                  onClick={() => setShowDifficultyInfo(true)}
                >
                  About difficulty levels
                </button>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 text-sm">
                <label className="inline-flex items-center gap-2 text-slate-100">
                  <input
                    type="checkbox"
                    className="rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
                    checked={form.isCalibrationGame ? true : form.aiCount > 0}
                    disabled={form.isCalibrationGame}
                    onChange={(e) => handleChange('aiCount', e.target.checked ? 1 : 0)}
                  />
                  <span>Play vs AI</span>
                </label>
                <p className="text-xs text-slate-400">
                  {form.aiCount > 0 || form.isCalibrationGame
                    ? 'You will be matched against a computer opponent at the selected difficulty.'
                    : 'Uncheck to create a human-only game.'}
                </p>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 text-sm">
                <label className="inline-flex items-center gap-2 text-slate-100">
                  <input
                    type="checkbox"
                    className="rounded border-slate-600 bg-slate-900 text-emerald-600 focus:ring-emerald-500"
                    checked={form.isCalibrationGame}
                    onChange={(e) => {
                      const enabled = e.target.checked;
                      handleChange('isCalibrationGame', enabled);
                      if (enabled) {
                        handleChange('boardType', 'square8' as BoardType);
                        handleChange('maxPlayers', 2);
                        handleChange('aiCount', 1);
                        handleChange('isRated', false);
                        if (
                          form.aiDifficulty !== 2 &&
                          form.aiDifficulty !== 4 &&
                          form.aiDifficulty !== 6 &&
                          form.aiDifficulty !== 8
                        ) {
                          handleChange('aiDifficulty', 4);
                        }
                      }
                    }}
                  />
                  <span>Contribute to AI difficulty calibration (Square-8 vs AI)</span>
                </label>
                <p className="text-xs text-slate-400">
                  Calibration games are unrated, 2-player Square-8 vs AI at a canonical tier (D2,
                  D4, D6, or D8).
                </p>
              </div>
              {(form.aiCount > 0 || form.isCalibrationGame) && (
                <div className="space-y-2">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-slate-100">
                      AI difficulty
                    </label>
                    <Select
                      value={form.aiDifficulty}
                      onChange={(e) => handleChange('aiDifficulty', Number(e.target.value))}
                    >
                      {DIFFICULTY_DESCRIPTORS.map((descriptor) => (
                        <option key={descriptor.id} value={descriptor.id}>
                          {descriptor.name} (D{descriptor.id})
                        </option>
                      ))}
                    </Select>
                  </div>
                  {selectedDifficultyDescriptor && (
                    <div className="text-xs text-slate-300 space-y-1">
                      <div className="font-semibold text-slate-100">
                        {selectedDifficultyDescriptor.name}{' '}
                        <span className="text-slate-400">(D{selectedDifficultyDescriptor.id})</span>
                      </div>
                      <p>{selectedDifficultyDescriptor.shortDescription}</p>
                      <p className="text-slate-400">
                        Recommended for: {selectedDifficultyDescriptor.recommendedAudience}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex justify-end pt-2">
              <Button type="submit" size="sm" disabled={isSubmitting}>
                {isSubmitting ? 'Creating game…' : 'Create Game'}
              </Button>
            </div>
          </form>
          <Dialog
            isOpen={showDifficultyInfo}
            onClose={() => setShowDifficultyInfo(false)}
            labelledBy="difficulty-info-title"
            describedBy="difficulty-info-description"
            initialFocusRef={difficultyInfoCloseButtonRef}
            overlayClassName="z-40 items-center justify-center"
            backdropClassName="bg-slate-950/80"
            className="max-w-xl w-full mx-4 rounded-2xl bg-slate-900 border border-slate-700 shadow-xl p-4 space-y-3"
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 id="difficulty-info-title" className="text-lg font-semibold text-white">
                  About AI difficulty levels
                </h3>
                <p id="difficulty-info-description" className="text-xs text-slate-300">
                  Difficulty levels use the same 1–10 ladder as the AI service and are calibrated on
                  compact Square‑8 2-player games. Tiers D2, D4, D6, and D8 are the main anchors for
                  casual, intermediate, advanced, and near‑expert play (see the Human Calibration
                  Guide).
                </p>
              </div>
              <button
                ref={difficultyInfoCloseButtonRef}
                type="button"
                onClick={() => setShowDifficultyInfo(false)}
                className="ml-2 text-slate-400 hover:text-slate-100"
                aria-label="Close difficulty information"
              >
                ✕
              </button>
            </div>
            <div className="max-h-80 overflow-y-auto mt-2 space-y-2">
              {DIFFICULTY_DESCRIPTORS.map((descriptor) => (
                <div
                  key={descriptor.id}
                  className="p-2 rounded-xl border border-slate-700 bg-slate-800/60 text-xs text-slate-200"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-semibold text-slate-100">
                      {descriptor.name} <span className="text-slate-400">(D{descriptor.id})</span>
                    </div>
                    <span className="text-[10px] text-slate-400">
                      {descriptor.recommendedAudience}
                    </span>
                  </div>
                  <p className="mt-1 text-slate-200">{descriptor.detailedDescription}</p>
                </div>
              ))}
              <div className="pt-2 border-t border-slate-700 text-[11px] text-slate-400">
                Experimental tiers D9–D10 are the strongest public levels and may take longer per
                move. Use them when you want a maximum-strength challenge.
              </div>
            </div>
          </Dialog>
        </section>
      ) : (
        <div className="space-y-6">
          {/* AI Quick Play Panel */}
          <AIQuickPlayPanel onStartGame={handleAIQuickPlay} isLoading={isSubmitting} />

          {/* Multiplayer Games Section */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Filters Sidebar */}
            <div className="lg:col-span-1">
              <LobbyFilters filters={filters} onFilterChange={setFilters} />
            </div>

            {/* Games List */}
            <div className="lg:col-span-3">
              <h2 className="text-lg font-semibold text-white mb-4">Open Games (Multiplayer)</h2>
              <div className="flex justify-between items-center mb-4">
                <div className="text-slate-400 text-sm">
                  {filteredGames.length} game
                  {filteredGames.length !== 1 ? 's' : ''} available
                </div>
                <SortControls sortBy={sortBy} onSortChange={setSortBy} />
              </div>

              {joinError && (
                <InlineAlert variant="error" className="mb-4">
                  {joinError}
                </InlineAlert>
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
        </div>
      )}
    </div>
  );
}
