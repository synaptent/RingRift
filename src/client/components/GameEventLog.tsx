import React from 'react';
import { GameHistoryEntry, GameResult, Position } from '../../shared/types/game';

export interface GameEventLogProps {
  history: GameHistoryEntry[];
  /**
   * Optional stream of system-level events (phase changes, connection
   * transitions, choice prompts) recorded by the hosting page.
   */
  systemEvents?: string[];
  victoryState?: GameResult | null;
  maxEntries?: number;
}

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
      const regionCount = action.claimedTerritory?.length ?? action.disconnectedRegions?.length ?? 0;
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
  const winner =
    typeof victory.winner === 'number' ? `Winner: P${victory.winner}` : 'Result: draw / no winner';
  const reason = victory.reason.replace(/_/g, ' ');
  return `${winner} — ${reason}`;
}

export function GameEventLog({
  history,
  systemEvents = [],
  victoryState,
  maxEntries = DEFAULT_MAX_ENTRIES,
}: GameEventLogProps) {
  const recentMoves = (history || []).slice(-maxEntries).reverse();
  const victoryLine = describeVictory(victoryState);

  const hasAnyContent = recentMoves.length > 0 || systemEvents.length > 0 || !!victoryLine;

  return (
    <div className="p-3 border border-slate-700 rounded bg-slate-900/50 max-h-64 overflow-y-auto">
      <h2 className="font-semibold mb-2 text-sm">Game log</h2>

      {!hasAnyContent && <div className="text-slate-300 text-xs">No events yet.</div>}

      {hasAnyContent && (
        <div className="space-y-3 text-xs text-slate-200">
          {victoryLine && (
            <div className="px-2 py-1 rounded bg-emerald-900/40 border border-emerald-500/40 text-emerald-100 font-semibold">
              {victoryLine}
            </div>
          )}

          {recentMoves.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                Recent moves
              </div>
              <ul className="list-disc list-inside space-y-0.5">
                {recentMoves.map((entry) => (
                  <li key={entry.moveNumber}>{describeHistoryEntry(entry)}</li>
                ))}
              </ul>
            </div>
          )}

          {systemEvents.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wide text-slate-400 mb-1">
                System events
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-slate-300">
                {systemEvents.map((entry, idx) => (
                  // eslint-disable-next-line react/no-array-index-key
                  <li key={idx}>{entry}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
