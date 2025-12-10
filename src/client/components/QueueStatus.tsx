/**
 * QueueStatus Component
 *
 * Displays the current matchmaking queue status including:
 * - Queue position
 * - Estimated wait time
 * - Search criteria
 * - Cancel button
 *
 * Shows a pulsing indicator while in queue and transitions smoothly
 * when a match is found.
 */

import React from 'react';
import clsx from 'clsx';
import { Button } from './ui/Button';
import type { MatchmakingPreferences } from '../../shared/types/websocket';

interface QueueStatusProps {
  /** Whether currently in the matchmaking queue */
  inQueue: boolean;
  /** Estimated wait time in milliseconds */
  estimatedWaitTime: number | null;
  /** Current position in queue */
  queuePosition: number | null;
  /** Search criteria being used */
  searchCriteria: MatchmakingPreferences | null;
  /** Whether a match has been found */
  matchFound: boolean;
  /** Callback to leave the queue */
  onLeaveQueue: () => void;
  /** Optional className for custom styling */
  className?: string;
}

/**
 * Format milliseconds as a human-readable time string
 */
function formatWaitTime(ms: number): string {
  const seconds = Math.ceil(ms / 1000);
  if (seconds < 60) {
    return `~${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (remainingSeconds === 0) {
    return `~${minutes}m`;
  }
  return `~${minutes}m ${remainingSeconds}s`;
}

/**
 * Format board type for display
 */
function formatBoardType(boardType: string): string {
  switch (boardType) {
    case 'square8':
      return 'Square 8×8';
    case 'square19':
      return 'Square 19×19';
    case 'hexagonal':
      return 'Hexagonal';
    default:
      return boardType;
  }
}

export function QueueStatus({
  inQueue,
  estimatedWaitTime,
  queuePosition,
  searchCriteria,
  matchFound,
  onLeaveQueue,
  className,
}: QueueStatusProps) {
  if (!inQueue && !matchFound) {
    return null;
  }

  return (
    <div
      className={clsx(
        'rounded-lg border p-4 transition-all duration-300',
        matchFound ? 'border-emerald-500 bg-emerald-900/30' : 'border-amber-500/50 bg-amber-900/20',
        className
      )}
      role="status"
      aria-live="polite"
    >
      {matchFound ? (
        <MatchFoundDisplay />
      ) : (
        <QueueingDisplay
          estimatedWaitTime={estimatedWaitTime}
          queuePosition={queuePosition}
          searchCriteria={searchCriteria}
          onLeaveQueue={onLeaveQueue}
        />
      )}
    </div>
  );
}

function MatchFoundDisplay() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20">
        <svg
          className="h-6 w-6 text-emerald-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <div>
        <p className="font-semibold text-emerald-300">Match Found!</p>
        <p className="text-sm text-slate-400">Joining game...</p>
      </div>
    </div>
  );
}

function QueueingDisplay({
  estimatedWaitTime,
  queuePosition,
  searchCriteria,
  onLeaveQueue,
}: {
  estimatedWaitTime: number | null;
  queuePosition: number | null;
  searchCriteria: MatchmakingPreferences | null;
  onLeaveQueue: () => void;
}) {
  return (
    <div className="space-y-3">
      {/* Header with pulsing indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative flex h-3 w-3">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-amber-400 opacity-75" />
            <span className="relative inline-flex h-3 w-3 rounded-full bg-amber-500" />
          </div>
          <span className="font-semibold text-amber-300">Finding Opponent...</span>
        </div>
        <Button variant="ghost" size="sm" onClick={onLeaveQueue}>
          Cancel
        </Button>
      </div>

      {/* Queue info */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-slate-500">Queue Position</p>
          <p className="font-medium text-slate-200">
            {queuePosition !== null ? `#${queuePosition}` : '—'}
          </p>
        </div>
        <div>
          <p className="text-slate-500">Est. Wait</p>
          <p className="font-medium text-slate-200">
            {estimatedWaitTime !== null ? formatWaitTime(estimatedWaitTime) : '—'}
          </p>
        </div>
      </div>

      {/* Search criteria */}
      {searchCriteria && (
        <div className="border-t border-slate-700/50 pt-3">
          <p className="mb-1 text-xs text-slate-500">Searching for:</p>
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
              {formatBoardType(searchCriteria.boardType)}
            </span>
            <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
              {searchCriteria.ratingRange.min}–{searchCriteria.ratingRange.max} Rating
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default QueueStatus;
