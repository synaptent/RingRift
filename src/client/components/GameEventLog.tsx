import React from 'react';
import { GameHistoryEntry, GameResult, Position } from '../../shared/types/game';
import type { EventLogViewModel, EventLogItemViewModel } from '../adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// Props Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Legacy props interface for backward compatibility.
 * Components can pass raw domain data which will be transformed internally.
 */
export interface GameEventLogLegacyProps {
  history: GameHistoryEntry[];
  /**
   * Optional stream of system-level events (phase changes, connection
   * transitions, choice prompts) recorded by the hosting page.
   */
  systemEvents?: string[];
  victoryState?: GameResult | null;
  maxEntries?: number;
}

/**
 * New view model props interface.
 * Components pass pre-transformed view model for maximum decoupling.
 */
export interface GameEventLogViewModelProps {
  /** Pre-transformed view model from useEventLogViewModel or toEventLogViewModel */
  viewModel: EventLogViewModel;
}

/**
 * Combined props type supporting both legacy and view model interfaces.
 * When viewModel is provided, legacy props are ignored.
 */
export type GameEventLogProps = GameEventLogLegacyProps | GameEventLogViewModelProps;

// ═══════════════════════════════════════════════════════════════════════════
// Type Guards
// ═══════════════════════════════════════════════════════════════════════════

function isViewModelProps(props: GameEventLogProps): props is GameEventLogViewModelProps {
  return 'viewModel' in props && props.viewModel !== undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy Transformation Functions (kept for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_MAX_ENTRIES = 40;

function formatPosition(pos?: Position): string {
  if (!pos) return '';
  if (typeof pos.z === 'number') {
    return `(${pos.x}, ${pos.y}, ${pos.z})`;
  }
  return `(${pos.x}, ${pos.y})`;
}

function describeHistoryEntry(entry: GameHistoryEntry): string {
  const { action } = entry;
  const moveLabel = `#${entry.moveNumber}`;
  const playerLabel = `P${action.player}`;

  switch (action.type) {
    case 'place_ring': {
      const count = action.placementCount ?? 1;
      return `${moveLabel} — ${playerLabel} placed ${count} ring${count === 1 ? '' : 's'} at ${formatPosition(action.to)}`;
    }
    case 'move_ring':
    case 'move_stack': {
      return `${moveLabel} — ${playerLabel} moved from ${formatPosition(action.from)} to ${formatPosition(action.to)}`;
    }
    case 'build_stack': {
      return `${moveLabel} — ${playerLabel} built stack at ${formatPosition(action.to)} (Δ=${action.buildAmount ?? 1})`;
    }
    case 'overtaking_capture': {
      const capturedCount = action.overtakenRings?.length ?? 0;
      const captureSuffix = capturedCount > 0 ? ` x${capturedCount}` : '';
      return `${moveLabel} — ${playerLabel} capture from ${formatPosition(action.from)} over ${formatPosition(action.captureTarget)} to ${formatPosition(action.to)}${captureSuffix}`;
    }
    case 'continue_capture_segment': {
      const capturedCount = action.overtakenRings?.length ?? 0;
      const captureSuffix = capturedCount > 0 ? ` x${capturedCount}` : '';
      return `${moveLabel} — ${playerLabel} continued capture over ${formatPosition(action.captureTarget)} to ${formatPosition(action.to)}${captureSuffix}`;
    }
    case 'process_line':
    case 'choose_line_reward': {
      const lineCount = action.formedLines?.length ?? 0;
      if (lineCount > 0) {
        return `${moveLabel} — ${playerLabel} processed ${lineCount} line${lineCount === 1 ? '' : 's'}`;
      }
      return `${moveLabel} — Line processing by ${playerLabel}`;
    }
    case 'process_territory_region':
    case 'eliminate_rings_from_stack': {
      const regionCount =
        action.claimedTerritory?.length ?? action.disconnectedRegions?.length ?? 0;
      const eliminatedTotal = (action.eliminatedRings ?? []).reduce(
        (sum, entry) => sum + entry.count,
        0
      );
      const parts: string[] = [];
      if (regionCount > 0) {
        parts.push(`${regionCount} region${regionCount === 1 ? '' : 's'}`);
      }
      if (eliminatedTotal > 0) {
        parts.push(`${eliminatedTotal} ring${eliminatedTotal === 1 ? '' : 's'} eliminated`);
      }
      const detail = parts.length > 0 ? ` (${parts.join(', ')})` : '';
      return `${moveLabel} — Territory / elimination processing by ${playerLabel}${detail}`;
    }
    case 'skip_placement': {
      return `${moveLabel} — ${playerLabel} skipped placement`;
    }
    default: {
      // Fallback for legacy / experimental move types.
      return `${moveLabel} — ${playerLabel} performed ${action.type}`;
    }
  }
}

function describeVictory(victory?: GameResult | null): string | null {
  if (!victory) return null;

  const formatReason = (reason: GameResult['reason']): string => {
    switch (reason) {
      case 'ring_elimination':
        return 'Ring Elimination';
      case 'territory_control':
        return 'Territory Control';
      case 'last_player_standing':
        return 'Last Player Standing';
      case 'timeout':
        return 'Timeout';
      case 'resignation':
        return 'Resignation';
      case 'abandonment':
        return 'Abandonment';
      case 'draw':
        return 'Draw';
      default:
        return reason.replace(/_/g, ' ');
    }
  };

  const reasonLabel = formatReason(victory.reason);

  if (victory.winner === undefined) {
    if (victory.reason === 'draw') {
      return 'Game ended in a draw.';
    }
    return `Result: ${reasonLabel}`;
  }

  // Winner known: surface a concise "wins by" message so LPS is clearly visible.
  return `Player P${victory.winner} wins by ${reasonLabel}`;
}

/**
 * Convert legacy props to view model for internal use
 */
function toLegacyViewModel(props: GameEventLogLegacyProps): EventLogViewModel {
  const { history, systemEvents = [], victoryState, maxEntries = DEFAULT_MAX_ENTRIES } = props;

  const entries: EventLogItemViewModel[] = [];

  // Victory entry first
  const victoryMessage = describeVictory(victoryState);
  if (victoryMessage) {
    entries.push({
      key: 'victory',
      text: victoryMessage,
      type: 'victory',
    });
  }

  // Recent moves
  const recentMoves = (history || []).slice(-maxEntries).reverse();
  for (const entry of recentMoves) {
    entries.push({
      key: `move-${entry.moveNumber}`,
      text: describeHistoryEntry(entry),
      type: 'move',
      moveNumber: entry.moveNumber,
    });
  }

  // System events
  for (let i = 0; i < systemEvents.length; i++) {
    entries.push({
      key: `system-${i}`,
      text: systemEvents[i],
      type: 'system',
    });
  }

  return {
    entries,
    victoryMessage: victoryMessage ?? undefined,
    hasContent: entries.length > 0,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Game Event Log Component
 *
 * Displays game history, system events, and victory messages in a scrollable list.
 *
 * Supports two usage patterns:
 *
 * 1. Legacy (backward compatible):
 * ```tsx
 * <GameEventLog
 *   history={gameState.history}
 *   systemEvents={eventLog}
 *   victoryState={victoryState}
 * />
 * ```
 *
 * 2. View Model (recommended for new code):
 * ```tsx
 * const viewModel = useEventLogViewModel({ systemEvents, maxEntries: 30 });
 * <GameEventLog viewModel={viewModel} />
 * ```
 */
export function GameEventLog(props: GameEventLogProps) {
  // Determine which props interface is being used
  const viewModel: EventLogViewModel = isViewModelProps(props)
    ? props.viewModel
    : toLegacyViewModel(props);

  const { entries, victoryMessage, hasContent } = viewModel;

  // Separate entries by type for structured rendering
  const victoryEntry = entries.find((e) => e.type === 'victory');
  const moveEntries = entries.filter((e) => e.type === 'move');
  const systemEntries = entries.filter((e) => e.type === 'system');

  return (
    <div
      className="p-3 border border-slate-700 rounded bg-slate-900/50 max-h-64 overflow-y-auto"
      data-testid="game-event-log"
    >
      <h2 className="font-semibold mb-2 text-sm">Game log</h2>

      {!hasContent && <div className="text-slate-300 text-xs">No events yet.</div>}

      {hasContent && (
        <div className="space-y-3 text-xs text-slate-200">
          {victoryEntry && (
            <div className="px-2 py-1 rounded bg-emerald-900/40 border border-emerald-500/40 text-emerald-100 font-semibold">
              {victoryEntry.text}
            </div>
          )}

          {moveEntries.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                Recent moves
              </div>
              <ul className="list-disc list-inside space-y-0.5">
                {moveEntries.map((entry) => (
                  <li key={entry.key}>{entry.text}</li>
                ))}
              </ul>
            </div>
          )}

          {systemEntries.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                System events
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-slate-300">
                {systemEntries.map((entry) => (
                  <li key={entry.key}>{entry.text}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
