/**
 * GameFilters - Filter dropdowns for browsing stored games.
 *
 * Provides board type, player count, outcome, and source filters.
 */

import type { ReplayGameQueryParams } from '../../types/replay';

export interface GameFiltersProps {
  filters: ReplayGameQueryParams;
  onFilterChange: (filters: ReplayGameQueryParams) => void;
  className?: string;
}

const BOARD_TYPE_OPTIONS = [
  { value: '', label: 'All Boards' },
  { value: 'square8', label: '8×8' },
  { value: 'square19', label: '19×19' },
  { value: 'hexagonal', label: 'Hex' },
];

const PLAYER_COUNT_OPTIONS = [
  { value: '', label: 'Any Players' },
  { value: '2', label: '2 Players' },
  { value: '3', label: '3 Players' },
  { value: '4', label: '4 Players' },
];

const TERMINATION_OPTIONS = [
  { value: '', label: 'Any Outcome' },
  { value: 'ring_elimination', label: 'Ring Elim.' },
  { value: 'territory', label: 'Territory' },
  { value: 'last_player_standing', label: 'Last Standing' },
  { value: 'stalemate', label: 'Stalemate' },
];

const SOURCE_OPTIONS = [
  { value: '', label: 'Any Source' },
  { value: 'self_play', label: 'Self-play' },
  { value: 'sandbox', label: 'Sandbox' },
  { value: 'tournament', label: 'Tournament' },
];

export function GameFilters({ filters, onFilterChange, className = '' }: GameFiltersProps) {
  const handleChange = (key: keyof ReplayGameQueryParams, value: string) => {
    const newFilters = { ...filters };
    if (value === '') {
      delete newFilters[key];
    } else if (key === 'num_players') {
      newFilters[key] = parseInt(value, 10);
    } else {
      (newFilters as Record<string, string | number | undefined>)[key] = value;
    }
    // Reset offset when filters change
    newFilters.offset = 0;
    onFilterChange(newFilters);
  };

  const selectClass =
    'px-2 py-1 text-xs rounded-lg border border-slate-600 bg-slate-800 text-slate-200 ' +
    'focus:outline-none focus:border-emerald-400 cursor-pointer';

  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      <select
        value={filters.board_type ?? ''}
        onChange={(e) => handleChange('board_type', e.target.value)}
        className={selectClass}
        aria-label="Filter by board type"
      >
        {BOARD_TYPE_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>

      <select
        value={filters.num_players?.toString() ?? ''}
        onChange={(e) => handleChange('num_players', e.target.value)}
        className={selectClass}
        aria-label="Filter by player count"
      >
        {PLAYER_COUNT_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>

      <select
        value={filters.termination_reason ?? ''}
        onChange={(e) => handleChange('termination_reason', e.target.value)}
        className={selectClass}
        aria-label="Filter by outcome"
      >
        {TERMINATION_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>

      <select
        value={filters.source ?? ''}
        onChange={(e) => handleChange('source', e.target.value)}
        className={selectClass}
        aria-label="Filter by source"
      >
        {SOURCE_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
