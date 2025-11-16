import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { gameApi } from '../services/api';
import { BoardType, CreateGameRequest } from '../../shared/types/game';

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
  aiType: 'heuristic'
};

export default function LobbyPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>(defaultForm);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm(prev => ({ ...prev, [key]: value }));
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
          increment: form.increment
        },
        aiOpponents:
          form.aiCount > 0
            ? {
                count: form.aiCount,
                difficulty: Array(form.aiCount).fill(form.aiDifficulty),
                mode: form.aiMode,
                aiType: form.aiType
              }
            : undefined
      };

      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
    } catch (err: any) {
      const message = err?.response?.data?.error?.message || err?.message || 'Failed to create game';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <header>
        <h1 className="text-3xl font-bold mb-1">Game Lobby</h1>
        <p className="text-sm text-gray-500">
          Create a new backend RingRift game with optional AI opponents. This form uses the
          shared CreateGameRequest / CreateGameSchema shape end-to-end, so server, client,
          and validation stay in sync.
        </p>
      </header>

      <form
        onSubmit={handleSubmit}
        className="max-w-2xl p-4 rounded-md bg-slate-900 border border-slate-700 space-y-4"
      >
        {error && (
          <div className="p-2 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-200">Board type</label>
            <select
              className="w-full px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
              value={form.boardType}
              onChange={e => handleChange('boardType', e.target.value as BoardType)}
            >
              <option value="square8">8x8 (compact)</option>
              <option value="square19">19x19 (full)</option>
              <option value="hexagonal">Hexagonal</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1 text-gray-200">Max players</label>
            <select
              className="w-full px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
              value={form.maxPlayers}
              onChange={e => handleChange('maxPlayers', Number(e.target.value))}
            >
              {[2, 3, 4].map(n => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1 text-gray-200">Time control</label>
            <div className="flex space-x-2">
              <select
                className="px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.timeControlType}
                onChange={e => handleChange('timeControlType', e.target.value as FormState['timeControlType'])}
              >
                <option value="blitz">Blitz</option>
                <option value="rapid">Rapid</option>
                <option value="classical">Classical</option>
              </select>
              <input
                type="number"
                min={60}
                max={7200}
                className="w-24 px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.initialTime}
                onChange={e => handleChange('initialTime', Number(e.target.value))}
                title="Initial time in seconds"
              />
              <input
                type="number"
                min={0}
                max={60}
                className="w-20 px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.increment}
                onChange={e => handleChange('increment', Number(e.target.value))}
                title="Increment in seconds per move"
              />
            </div>
          </div>

          <div className="flex items-center space-x-4 mt-6 md:mt-0">
            <label className="inline-flex items-center text-sm text-gray-200">
              <input
                type="checkbox"
                className="mr-2 rounded border-slate-600 bg-slate-800"
                checked={form.isRated}
                onChange={e => handleChange('isRated', e.target.checked)}
              />
              Rated
            </label>
            <label className="inline-flex items-center text-sm text-gray-200">
              <input
                type="checkbox"
                className="mr-2 rounded border-slate-600 bg-slate-800"
                checked={form.isPrivate}
                onChange={e => handleChange('isPrivate', e.target.checked)}
              />
              Private
            </label>
          </div>
        </div>

        <hr className="border-slate-700" />

        <section className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-200">AI Opponents</h2>
          <p className="text-xs text-gray-400">
            Configure zero or more AI opponents. The lobby sends a unified aiOpponents config
            (count, difficulty, mode, aiType) which the server persists in gameState.aiOpponents
            and uses to construct per-player aiProfile entries in GameState.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-200">Number of AI opponents</label>
              <input
                type="number"
                min={0}
                max={3}
                className="w-24 px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.aiCount}
                onChange={e => handleChange('aiCount', Math.max(0, Math.min(3, Number(e.target.value))))}
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-200">Difficulty (1–10)</label>
              <input
                type="number"
                min={1}
                max={10}
                className="w-24 px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.aiDifficulty}
                onChange={e => handleChange('aiDifficulty', Math.max(1, Math.min(10, Number(e.target.value))))}
              />
              <p className="mt-1 text-xs text-gray-500">Applied uniformly to all AI opponents for now.</p>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-200">AI control mode</label>
              <select
                className="w-full px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.aiMode}
                onChange={e => handleChange('aiMode', e.target.value as FormState['aiMode'])}
              >
                <option value="service">Python service (default)</option>
                <option value="local_heuristic">Local heuristic</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1 text-gray-200">AI type</label>
              <select
                className="w-full px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
                value={form.aiType}
                onChange={e => handleChange('aiType', e.target.value as FormState['aiType'])}
              >
                <option value="random">Random</option>
                <option value="heuristic">Heuristic</option>
                <option value="minimax">Minimax</option>
                <option value="mcts">MCTS</option>
              </select>
            </div>
          </div>
        </section>

        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 text-sm font-semibold text-white"
          >
            {isSubmitting ? 'Creating game…' : 'Create Game'}
          </button>
        </div>
      </form>
    </div>
  );
}
