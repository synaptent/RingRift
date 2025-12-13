import React, { useId } from 'react';
import { DIFFICULTY_DESCRIPTORS, getDifficultyDescriptor } from '../utils/difficultyUx';
import { Tooltip } from './ui/Tooltip';

export interface AIDifficultySelectorProps {
  /** Current difficulty value (1-10) */
  value: number;
  /** Callback when difficulty changes */
  onChange: (difficulty: number) => void;
  /** Player number for labeling (1-4) */
  playerNumber: number;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Compact mode for space-constrained layouts */
  compact?: boolean;
}

/**
 * Accessible AI difficulty selector for sandbox game setup.
 *
 * Features:
 * - Slider with labeled steps for all 10 difficulty levels
 * - Keyboard accessible (arrow keys to adjust, tab to focus)
 * - Screen reader friendly with ARIA labels and live regions
 * - Tooltips with difficulty descriptions
 * - Compact mode option for inline use
 */
export function AIDifficultySelector({
  value,
  onChange,
  playerNumber,
  disabled = false,
  compact = false,
}: AIDifficultySelectorProps) {
  const sliderId = useId();
  const descriptionId = useId();
  const descriptor = getDifficultyDescriptor(value);

  // Key difficulty tiers for visual markers
  const keyTiers = [2, 4, 6, 8];

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(parseInt(e.target.value, 10));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Allow fine-grained control with arrow keys
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      if (value > 1) onChange(value - 1);
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (value < 10) onChange(value + 1);
    } else if (e.key === 'Home') {
      e.preventDefault();
      onChange(1);
    } else if (e.key === 'End') {
      e.preventDefault();
      onChange(10);
    }
  };

  // Quick select buttons for key tiers
  const QuickSelectButtons = () => (
    <div className="flex gap-1" role="group" aria-label="Quick difficulty presets">
      {keyTiers.map((tier) => {
        const tierDesc = getDifficultyDescriptor(tier);
        const isSelected = value === tier;
        return (
          <Tooltip key={tier} content={tierDesc?.shortDescription || `Difficulty ${tier}`}>
            <button
              type="button"
              onClick={() => onChange(tier)}
              disabled={disabled}
              aria-pressed={isSelected}
              className={`px-2 py-0.5 text-xs rounded-md border transition-colors ${
                isSelected
                  ? 'border-sky-400 bg-sky-900/40 text-sky-200'
                  : 'border-slate-600 text-slate-400 hover:border-slate-400 hover:text-slate-200'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              D{tier}
            </button>
          </Tooltip>
        );
      })}
    </div>
  );

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <label htmlFor={sliderId} className="sr-only">
          AI Difficulty for Player {playerNumber}
        </label>
        <select
          id={sliderId}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
          disabled={disabled}
          className={`px-2 py-1 text-xs rounded-md border bg-slate-900 text-slate-100
            ${disabled ? 'opacity-50 cursor-not-allowed border-slate-700' : 'border-slate-600 focus:border-sky-400 focus:ring-1 focus:ring-sky-400'}`}
          aria-describedby={descriptionId}
        >
          {DIFFICULTY_DESCRIPTORS.map((d) => (
            <option key={d.id} value={d.id}>
              D{d.id} - {d.name.split(' ')[0]}
            </option>
          ))}
        </select>
        <span id={descriptionId} className="sr-only">
          {descriptor?.shortDescription}
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Header with label and quick select buttons */}
      <div className="flex items-center justify-between gap-2">
        <label htmlFor={sliderId} className="text-xs font-medium text-slate-300">
          AI Strength
        </label>
        <QuickSelectButtons />
      </div>

      {/* Slider */}
      <div className="relative">
        <input
          type="range"
          id={sliderId}
          min={1}
          max={10}
          step={1}
          value={value}
          onChange={handleSliderChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          aria-valuemin={1}
          aria-valuemax={10}
          aria-valuenow={value}
          aria-valuetext={`Difficulty ${value}: ${descriptor?.name || ''}`}
          aria-describedby={descriptionId}
          className={`w-full h-2 rounded-full appearance-none cursor-pointer
            bg-gradient-to-r from-emerald-900 via-amber-900 to-red-900
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-sky-400
            [&::-webkit-slider-thumb]:border-2
            [&::-webkit-slider-thumb]:border-slate-900
            [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-sky-400
            [&::-moz-range-thumb]:border-2
            [&::-moz-range-thumb]:border-slate-900
            [&::-moz-range-thumb]:shadow-md
            focus:outline-none
            focus-visible:ring-2
            focus-visible:ring-sky-400
            focus-visible:ring-offset-2
            focus-visible:ring-offset-slate-900
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        />

        {/* Tick marks for key difficulties */}
        <div className="absolute -bottom-1 left-0 right-0 flex justify-between px-[6px] pointer-events-none">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((tick) => (
            <div
              key={tick}
              className={`w-0.5 h-1.5 rounded-full ${
                keyTiers.includes(tick) ? 'bg-slate-400' : 'bg-slate-600'
              }`}
            />
          ))}
        </div>
      </div>

      {/* Current difficulty info */}
      <div
        id={descriptionId}
        className="p-2 rounded-lg bg-slate-800/50 border border-slate-700"
        aria-live="polite"
        aria-atomic="true"
      >
        <div className="flex items-center gap-2">
          <span className="px-1.5 py-0.5 text-xs font-bold rounded bg-sky-900/60 text-sky-300 border border-sky-700">
            D{value}
          </span>
          <span className="text-sm font-medium text-slate-100">
            {descriptor?.name || `Difficulty ${value}`}
          </span>
        </div>
        <p className="mt-1 text-xs text-slate-400 leading-relaxed">
          {descriptor?.shortDescription || 'Select a difficulty level'}
        </p>
      </div>
    </div>
  );
}

/**
 * Compact inline difficulty badge that shows current difficulty and allows quick changes.
 * Used in player cards where space is limited.
 */
export function AIDifficultyBadge({
  value,
  onChange,
  disabled = false,
}: {
  value: number;
  onChange: (difficulty: number) => void;
  disabled?: boolean;
}) {
  const descriptor = getDifficultyDescriptor(value);
  const badgeId = useId();

  return (
    <Tooltip content={descriptor?.shortDescription || `Difficulty ${value}`}>
      <div className="relative inline-flex items-center">
        <label htmlFor={badgeId} className="sr-only">
          AI Difficulty Level
        </label>
        <select
          id={badgeId}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value, 10))}
          disabled={disabled}
          className={`appearance-none pl-2 pr-6 py-0.5 text-xs font-medium rounded-md border
            bg-sky-900/40 text-sky-200 border-sky-600
            focus:outline-none focus:ring-1 focus:ring-sky-400
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:bg-sky-900/60'}`}
          aria-label={`Difficulty: ${descriptor?.name || value}`}
        >
          {DIFFICULTY_DESCRIPTORS.map((d) => (
            <option key={d.id} value={d.id}>
              D{d.id}
            </option>
          ))}
        </select>
        {/* Custom dropdown arrow */}
        <svg
          className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-sky-400 pointer-events-none"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </Tooltip>
  );
}
