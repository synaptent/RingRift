import React, { useState } from 'react';
import { BoardType } from '../../shared/types/game';
import {
  AIQuickPlayOption,
  TIER_COLORS,
  getOptionsForConfig,
  DifficultyTier,
} from '../config/aiQuickPlay';

interface AIQuickPlayPanelProps {
  onStartGame: (option: AIQuickPlayOption) => void;
  isLoading: boolean;
}

const BOARD_OPTIONS: { value: BoardType; label: string; subtitle: string }[] = [
  { value: 'square8', label: 'Sq 8×8', subtitle: 'Compact' },
  { value: 'hex8', label: 'Hex 8', subtitle: 'Small hex' },
  { value: 'square19', label: 'Sq 19×19', subtitle: 'Classic' },
  { value: 'hexagonal', label: 'Full Hex', subtitle: 'Large hex' },
];

function DifficultyCard({
  option,
  onClick,
  disabled,
}: {
  option: AIQuickPlayOption;
  onClick: () => void;
  disabled: boolean;
}) {
  const colors = TIER_COLORS[option.difficultyTier];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`
        p-4 rounded-xl border-2 ${colors.border} ${colors.bg}
        transition-all duration-200 text-left
        hover:scale-[1.02] hover:shadow-lg
        focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 focus:ring-offset-slate-900
        disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100
      `}
    >
      <div className={`font-semibold ${colors.text}`}>{option.displayName}</div>
      <div className="text-xs text-slate-400 mt-1">D{option.difficulty}</div>
      <div className="text-xs text-slate-300 mt-2">{option.description}</div>
      <div className="text-[10px] text-slate-500 mt-1">~{option.estimatedElo} Elo</div>
    </button>
  );
}

export function AIQuickPlayPanel({ onStartGame, isLoading }: AIQuickPlayPanelProps) {
  const [selectedBoard, setSelectedBoard] = useState<BoardType>('square8');
  const [selectedPlayers, setSelectedPlayers] = useState<number>(2);

  const filteredOptions = getOptionsForConfig(selectedBoard, selectedPlayers);

  // Sort options by difficulty tier order
  const tierOrder: DifficultyTier[] = ['easy', 'medium', 'hard', 'expert'];
  const sortedOptions = [...filteredOptions].sort(
    (a, b) => tierOrder.indexOf(a.difficultyTier) - tierOrder.indexOf(b.difficultyTier)
  );

  return (
    <div className="bg-slate-800/70 rounded-xl p-6 border border-emerald-500/30 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          Play vs AI
          <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-300 rounded-full border border-emerald-500/30">
            Instant
          </span>
        </h2>
        <p className="text-xs text-slate-400">Choose difficulty and start immediately</p>
      </div>

      {/* Board selector - card grid */}
      <div className="mb-4">
        <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">Board</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {BOARD_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setSelectedBoard(opt.value)}
              className={`p-3 text-left rounded-xl border transition ${
                selectedBoard === opt.value
                  ? 'border-emerald-400 bg-emerald-900/20 text-white'
                  : 'border-slate-600 bg-slate-900/60 text-slate-200 hover:border-slate-400'
              }`}
            >
              <p className="font-semibold text-sm">{opt.label}</p>
              <p className="text-[10px] text-slate-400">{opt.subtitle}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Player count selector - pill buttons */}
      <div className="mb-6">
        <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">Players</p>
        <div className="flex gap-2">
          {[2, 3, 4].map((count) => (
            <button
              key={count}
              type="button"
              onClick={() => setSelectedPlayers(count)}
              className={`px-4 py-2 rounded-full border transition text-sm ${
                selectedPlayers === count
                  ? 'border-emerald-400 text-emerald-200 bg-emerald-900/30'
                  : 'border-slate-600 text-slate-300 hover:border-slate-400'
              }`}
            >
              1v{count - 1} AI
            </button>
          ))}
        </div>
      </div>

      {/* Difficulty cards */}
      <div className="mb-2">
        <p className="text-xs uppercase tracking-wide text-slate-400 mb-2">Difficulty</p>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {sortedOptions.map((option) => (
          <DifficultyCard
            key={option.id}
            option={option}
            onClick={() => onStartGame(option)}
            disabled={isLoading}
          />
        ))}
      </div>

      {/* Helper text */}
      <p className="mt-4 text-xs text-slate-500 text-center">
        AI games are unrated. Elo estimates are approximate based on AI training data.
      </p>
    </div>
  );
}

export type { AIQuickPlayOption };
