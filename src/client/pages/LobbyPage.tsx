import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { gameApi } from '../services/api';
import { BoardType, CreateGameRequest, Game } from '../../shared/types/game';
import LoadingSpinner from '../components/LoadingSpinner';

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

const defaultForm: FormState = {
  boardType: 'square8',
  maxPlayers: 2,
  isRated: true,
  isPrivate: false,
  timeControlType: 'blitz',
  initialTime: 600, // seconds
  increment: 0,
  aiCount: 1,
  aiDifficulty: 5,
  aiMode: 'service',
  aiType: 'heuristic',
};

export default function LobbyPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>(defaultForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableGames, setAvailableGames] = useState<Game[]>([]);
  const [isLoadingGames, setIsLoadingGames] = useState(true);
  const [joinError, setJoinError] = useState<string | null>(null);

  useEffect(() => {
    fetchAvailableGames();
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
      // Refresh list in case game is no longer available
      fetchAvailableGames();
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">Game Lobby</h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          Join an existing backend game or create a new one with custom board, time control, and AI
          configuration. Games created here are persisted in the database and played via the
          backend GameEngine.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Available Games Section */}
        <section className="space-y-4">
          <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-4 text-slate-100 shadow-lg">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Lobby</p>
                <h2 className="text-lg font-semibold text-white">Available Games</h2>
                <p className="mt-1 text-xs text-slate-300">
                  Browse open backend games you can join immediately. New games appear here as they
                  are created in the lobby.
                </p>
              </div>
              <button
                onClick={fetchAvailableGames}
                className="shrink-0 inline-flex items-center justify-center px-4 py-2 rounded-xl border border-slate-500 bg-slate-900/70 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition-colors"
              >
                Refresh
              </button>
            </div>

            {joinError && (
              <div className="p-2 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded">
                {joinError}
              </div>
            )}

            {isLoadingGames ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner size="md" />
              </div>
            ) : !Array.isArray(availableGames) || availableGames.length === 0 ? (
              <div className="p-8 text-center bg-slate-900/70 rounded-xl border border-slate-800 text-slate-400">
                No open games found. Create one to get started!
              </div>
            ) : (
              <div className="space-y-3">
                {availableGames.map((game) => (
                  <button
                    key={game.id}
                    type="button"
                    onClick={() => handleJoinGame(game.id)}
                    className="w-full text-left p-4 bg-slate-900 rounded-2xl border border-slate-700 hover:border-emerald-500/80 hover:bg-slate-900/90 transition-colors flex justify-between items-center gap-4"
                  >
                    <div>
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-medium text-white">
                          {game.player1?.username || 'Unknown'}
                        </span>
                        <span className="text-xs px-1.5 py-0.5 rounded bg-slate-700 text-slate-200">
                          {game.player1?.rating ?? '?'}
                        </span>
                      </div>
                      <div className="text-xs text-slate-400 flex flex-wrap gap-x-2 gap-y-1">
                        <span>{game.boardType}</span>
                        <span>•</span>
                        <span>
                          {game.timeControl
                            ? `${game.timeControl.type} (${Math.round(
                                game.timeControl.initialTime / 60
                              )}m + ${game.timeControl.increment}s)`
                            : 'time control: n/a'}
                        </span>
                        <span>•</span>
                        <span>{game.isRated ? 'Rated' : 'Casual'}</span>
                      </div>
                    </div>
                    <div className="shrink-0 inline-flex items-center justify-center px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold">
                      Join
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Create Game Section */}
        <section className="space-y-4">
          <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-6 text-slate-100 shadow-lg">
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-wide text-slate-400">Backend Game</p>
              <h2 className="text-xl font-semibold text-white">Create Backend Game</h2>
              <p className="text-sm text-slate-300">
                Choose board size, time control, and optional AI opponents. This creates a
                server-side game that will appear in the lobby and can be resumed later.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="p-2 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded">
                  {error}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1 text-slate-100">Board type</label>
                  <select
                    className="w-full px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.boardType}
                    onChange={(e) => handleChange('boardType', e.target.value as BoardType)}
                  >
                    <option value="square8">8x8 (compact)</option>
                    <option value="square19">19x19 (full)</option>
                    <option value="hexagonal">Hexagonal</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-slate-100">Max players</label>
                  <select
                    className="w-full px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    value={form.maxPlayers}
                    onChange={(e) => handleChange('maxPlayers', Number(e.target.value))}
                  >
                    {[2, 3, 4].map((n) => (
                      <option key={n} value={n}>
                        {n}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-slate-100">
                    Time control
                  </label>
                  <div className="flex flex-wrap items-center gap-2">
                    <select
                      className="px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
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
                    </select>
                    <input
                      type="number"
                      min={60}
                      max={7200}
                      className="w-28 px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.initialTime}
                      onChange={(e) => handleChange('initialTime', Number(e.target.value))}
                      title="Initial time in seconds"
                    />
                    <input
                      type="number"
                      min={0}
                      max={60}
                      className="w-24 px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.increment}
                      onChange={(e) => handleChange('increment', Number(e.target.value))}
                      title="Increment in seconds per move"
                    />
                  </div>
                  <p className="mt-1 text-xs text-slate-500">
                    Example: <span className="font-mono">600 + 0</span> = 10 minutes each, no
                    increment.
                  </p>
                </div>

                <div className="flex flex-col justify-center gap-2 mt-2 md:mt-0">
                  <label className="inline-flex items-center text-sm text-slate-100">
                    <input
                      type="checkbox"
                      className="mr-2 rounded border-slate-600 bg-slate-900"
                      checked={form.isRated}
                      onChange={(e) => handleChange('isRated', e.target.checked)}
                    />
                    Rated game
                  </label>
                  <label className="inline-flex items-center text-sm text-slate-100">
                    <input
                      type="checkbox"
                      className="mr-2 rounded border-slate-600 bg-slate-900"
                      checked={form.isPrivate}
                      onChange={(e) => handleChange('isPrivate', e.target.checked)}
                    />
                    Private (not listed in public lobby)
                  </label>
                </div>
              </div>

              <hr className="border-slate-700" />

              <section className="space-y-3">
                <h3 className="text-sm font-semibold text-slate-100">AI Opponents</h3>
                <p className="text-xs text-slate-400">
                  Configure zero or more AI opponents. The lobby sends a unified{' '}
                  <span className="font-mono">aiOpponents</span> config (count, difficulty, mode,
                  aiType) which the server stores in <span className="font-mono">gameState</span> and
                  uses to instantiate AI players.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
                  <div>
                    <label className="block text-sm font-medium mb-1 text-slate-100">
                      Number of AI opponents
                    </label>
                    <input
                      type="number"
                      min={0}
                      max={3}
                      className="w-24 px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.aiCount}
                      onChange={(e) =>
                        handleChange(
                          'aiCount',
                          Math.max(0, Math.min(3, Number(e.target.value)))
                        )
                      }
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1 text-slate-100">
                      Difficulty (1–10)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={10}
                      className="w-24 px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.aiDifficulty}
                      onChange={(e) =>
                        handleChange(
                          'aiDifficulty',
                          Math.max(1, Math.min(10, Number(e.target.value)))
                        )
                      }
                    />
                    <p className="mt-1 text-xs text-slate-500">
                      Applied uniformly to all AI opponents for now.
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1 text-slate-100">
                      AI control mode
                    </label>
                    <select
                      className="w-full px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.aiMode}
                      onChange={(e) => handleChange('aiMode', e.target.value as FormState['aiMode'])}
                    >
                      <option value="service">Python service (default)</option>
                      <option value="local_heuristic">Local heuristic</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-1 text-slate-100">AI type</label>
                    <select
                      className="w-full px-2 py-1.5 rounded-md bg-slate-900 border border-slate-600 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      value={form.aiType}
                      onChange={(e) => handleChange('aiType', e.target.value as FormState['aiType'])}
                    >
                      <option value="random">Random (Easy)</option>
                      <option value="heuristic">Heuristic (Medium)</option>
                      <option value="minimax">Minimax (Hard)</option>
                      <option value="mcts">MCTS (Expert)</option>
                    </select>
                  </div>
                </div>
              </section>

              <div className="flex justify-end pt-2">
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="px-4 py-2 rounded-md bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-sm font-semibold text-white"
                >
                  {isSubmitting ? 'Creating game…' : 'Create Game'}
                </button>
              </div>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
}
