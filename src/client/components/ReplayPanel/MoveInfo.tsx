/**
 * MoveInfo - Display details about the current move in replay.
 */

import type { ReplayMoveRecord } from '../../types/replay';
import { formatMoveTypeLabel } from '../../utils/moveTypeLabels';

export interface MoveInfoProps {
  move: ReplayMoveRecord | null;
  moveNumber: number;
  className?: string;
}

function formatMoveType(moveType: string): string {
  return formatMoveTypeLabel(moveType as ReplayMoveRecord['moveType']);
}

function formatPosition(pos: { x: number; y: number; z?: number } | null | undefined): string {
  if (!pos) return '—';
  if (pos.z !== undefined && pos.z !== null) {
    return `(${pos.x},${pos.y},${pos.z})`;
  }
  return `(${pos.x},${pos.y})`;
}

function formatEval(value: number | undefined | null): string {
  if (value === undefined || value === null) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}`;
}

export function MoveInfo({ move, moveNumber, className = '' }: MoveInfoProps) {
  if (moveNumber === 0) {
    return (
      <div className={`text-xs text-slate-400 ${className}`}>
        Initial position (before any moves)
      </div>
    );
  }

  if (!move) {
    return <div className={`text-xs text-slate-400 ${className}`}>No move data available</div>;
  }

  const fromPos = move.move?.from as { x: number; y: number; z?: number } | undefined;
  const toPos = move.move?.to as { x: number; y: number; z?: number } | undefined;

  return (
    <div className={`space-y-1 ${className}`}>
      {/* Move header */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="px-1.5 py-0.5 rounded bg-slate-700 text-[10px] font-medium">
            Move {moveNumber}
          </span>
          <span className="text-xs text-slate-200">
            P{move.player}: {formatMoveType(move.moveType)}
          </span>
        </div>
        {move.thinkTimeMs !== null && (
          <span className="text-[10px] text-slate-500">{move.thinkTimeMs}ms</span>
        )}
      </div>

      {/* Position info */}
      {(fromPos || toPos) && (
        <div className="text-[10px] text-slate-400">
          {fromPos && <span>{formatPosition(fromPos)}</span>}
          {fromPos && toPos && <span className="mx-1">→</span>}
          {toPos && <span>{formatPosition(toPos)}</span>}
        </div>
      )}

      {/* Engine evaluation (if available) */}
      {move.engineEval !== undefined && move.engineEval !== null && (
        <div className="flex items-center gap-2 text-[10px]">
          <span className="text-slate-500">Eval:</span>
          <span
            className={`font-mono ${
              move.engineEval > 0
                ? 'text-emerald-400'
                : move.engineEval < 0
                  ? 'text-red-400'
                  : 'text-slate-300'
            }`}
          >
            {formatEval(move.engineEval)}
          </span>
          {move.engineEvalType && <span className="text-slate-600">({move.engineEvalType})</span>}
          {move.engineDepth !== undefined && move.engineDepth !== null && (
            <span className="text-slate-600">d{move.engineDepth}</span>
          )}
        </div>
      )}

      {/* Principal variation (if available) */}
      {move.enginePV && move.enginePV.length > 0 && (
        <div className="text-[10px] text-slate-500">
          <span className="text-slate-600">PV:</span>{' '}
          <span className="font-mono">{move.enginePV.slice(0, 5).join(' ')}</span>
          {move.enginePV.length > 5 && <span>...</span>}
        </div>
      )}

      {/* Time remaining (if available) */}
      {move.timeRemainingMs !== undefined && move.timeRemainingMs !== null && (
        <div className="text-[10px] text-slate-500">
          Clock: {Math.floor(move.timeRemainingMs / 1000)}s remaining
        </div>
      )}
    </div>
  );
}
