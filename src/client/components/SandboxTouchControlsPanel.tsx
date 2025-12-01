import React from 'react';
import type { Position } from '../../shared/types/game';

export interface SandboxTouchControlsPanelProps {
  selectedPosition?: Position;
  selectedStackDetails?: {
    height: number;
    cap: number;
    controllingPlayer: number;
  } | null;
  validTargets: Position[];
  isCaptureDirectionPending: boolean;
  captureTargets: Position[];
  canUndoSegment: boolean;
  onClearSelection: () => void;
  onUndoSegment: () => void;
  onApplyMove: () => void;
  showMovementGrid: boolean;
  onToggleMovementGrid: (next: boolean) => void;
  showValidTargets: boolean;
  onToggleValidTargets: (next: boolean) => void;
  phaseLabel: string;
}

/**
 * Presentational-only touch controls panel for the sandbox host.
 *
 * Responsibilities:
 * - Render the current selection summary (cell + stack details).
 * - Surface high-level sandbox actions as buttons:
 *   - Clear selection
 *   - Undo last segment (when enabled by host)
 *   - Finish capture / Apply move (when a multi-step interaction is ready)
 * - Offer simple toggles for visual overlays:
 *   - Show valid targets
 *   - Show movement grid
 *
 * All state and orchestration are owned by the host and hooks layer; this
 * component remains rules-agnostic and only calls the provided callbacks.
 */
export const SandboxTouchControlsPanel: React.FC<SandboxTouchControlsPanelProps> = ({
  selectedPosition,
  selectedStackDetails,
  validTargets,
  isCaptureDirectionPending,
  captureTargets,
  canUndoSegment,
  onClearSelection,
  onUndoSegment,
  onApplyMove,
  showMovementGrid,
  onToggleMovementGrid,
  showValidTargets,
  onToggleValidTargets,
  phaseLabel,
}) => {
  const hasSelection = !!selectedPosition;
  const hasTargets = validTargets.length > 0;
  const hasCaptureTargets = captureTargets.length > 0;

  const selectionLabel = selectedPosition
    ? `(${selectedPosition.x}, ${selectedPosition.y}${
        typeof selectedPosition.z === 'number' ? `, ${selectedPosition.z}` : ''
      })`
    : 'None';

  const stackSummary = selectedStackDetails
    ? `H${selectedStackDetails.height} · C${selectedStackDetails.cap} · P${selectedStackDetails.controllingPlayer}`
    : null;

  const showApplyButton = hasSelection && hasTargets && !isCaptureDirectionPending;

  return (
    <div
      className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3 text-xs text-slate-100"
      data-testid="sandbox-touch-controls"
    >
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold">Touch Controls</h2>
          <p className="text-[11px] text-slate-400">Tap, then use these controls to refine moves.</p>
        </div>
        <span className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 text-[10px] uppercase tracking-wide text-slate-300">
          Phase: {phaseLabel}
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-[11px]">Selection</span>
          <span className="text-[11px] text-slate-400">
            Targets: {validTargets.length}
          </span>
        </div>

        {hasSelection ? (
          <div className="space-y-1">
            <div className="font-mono text-sm text-white">{selectionLabel}</div>
            {stackSummary ? (
              <div className="text-[11px] text-slate-300">{stackSummary}</div>
            ) : (
              <div className="text-[11px] text-slate-300">
                Empty cell – select a highlighted target to place and move.
              </div>
            )}
            {hasTargets && !isCaptureDirectionPending && (
              <p className="text-[11px] text-slate-400">
                Tap a highlighted destination on the board, or use "Finish move" once you are done.
              </p>
            )}
            {isCaptureDirectionPending && (
              <p className="text-[11px] text-amber-300">
                Capture continuation available – tap a highlighted landing cell to choose direction.
              </p>
            )}
          </div>
        ) : (
          <p className="text-[11px] text-slate-300">
            Tap any stack or empty cell to begin. Use the buttons below to clear or adjust your selection.
          </p>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onClearSelection}
          disabled={!hasSelection || isCaptureDirectionPending}
          className={`px-3 py-1.5 rounded-lg border text-[11px] font-semibold transition ${
            !hasSelection || isCaptureDirectionPending
              ? 'border-slate-700 text-slate-500 cursor-not-allowed opacity-60'
              : 'border-slate-500 text-slate-100 hover:border-emerald-400 hover:text-emerald-200'
          }`}
        >
          Clear selection
        </button>

        <button
          type="button"
          onClick={onUndoSegment}
          disabled={!canUndoSegment}
          className={`px-3 py-1.5 rounded-lg border text-[11px] font-semibold transition ${
            !canUndoSegment
              ? 'border-slate-700 text-slate-500 cursor-not-allowed opacity-60'
              : 'border-amber-400 text-amber-100 hover:border-amber-200 hover:text-amber-50'
          }`}
        >
          Undo last segment
        </button>

        <button
          type="button"
          onClick={onApplyMove}
          disabled={!showApplyButton}
          className={`px-3 py-1.5 rounded-lg border text-[11px] font-semibold transition ${
            !showApplyButton
              ? 'border-slate-700 text-slate-500 cursor-not-allowed opacity-60'
              : 'border-emerald-500 text-emerald-100 bg-emerald-900/40 hover:border-emerald-300 hover:bg-emerald-800/70'
          }`}
        >
          Finish move
        </button>
      </div>

      <div className="border-t border-slate-700 pt-3 space-y-2">
        <span className="font-semibold text-[11px]">Visual aids</span>
        <div className="flex flex-col gap-1 text-[11px]">
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
              checked={showValidTargets}
              onChange={(e) => onToggleValidTargets(e.target.checked)}
            />
            <span className="text-slate-200">Show valid targets</span>
          </label>
          <label className="inline-flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-slate-600 bg-slate-900 text-sky-500 focus:ring-sky-500"
              checked={showMovementGrid}
              onChange={(e) => onToggleMovementGrid(e.target.checked)}
            />
            <span className="text-slate-200">Show movement grid</span>
          </label>
        </div>
      </div>

      {hasCaptureTargets && (
        <div className="border-t border-slate-700 pt-3 space-y-2">
          <span className="font-semibold text-[11px]">Capture segments</span>
          <p className="text-[11px] text-slate-400">
            Pending capture directions are highlighted on the board. Tap a landing cell to continue the chain.
          </p>
          <div className="flex flex-wrap gap-1">
            {captureTargets.map((pos, idx) => (
              <span
                // eslint-disable-next-line react/no-array-index-key
                key={`${pos.x},${pos.y},${pos.z ?? 0}-${idx}`}
                className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 font-mono text-[10px]"
              >
                ({pos.x}, {pos.y}
                {typeof pos.z === 'number' ? `, ${pos.z}` : ''})
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};