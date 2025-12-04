import React from 'react';
import type { Position } from '../../shared/types/game';

export interface SandboxTouchControlsPanelProps {
  selectedPosition?: Position | undefined;
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
  /** Toggle overlay highlighting detected lines on the board */
  showLineOverlays?: boolean;
  onToggleLineOverlays?: (next: boolean) => void;
  /** Toggle overlay highlighting territory regions on the board */
  showTerritoryOverlays?: boolean;
  onToggleTerritoryOverlays?: (next: boolean) => void;
  phaseLabel: string;
  /** Optional short hint for the current phase/decision, derived by the host. */
  phaseHint?: string;
  /**
   * Optional flag indicating that a skip_territory_processing action is
   * currently available during territory processing. When true and
   * onSkipTerritoryProcessing is provided, the panel surfaces a dedicated
   * "Skip territory processing" control.
   */
  canSkipTerritoryProcessing?: boolean;
  /** Optional handler invoked when the user taps the skip territory button. */
  onSkipTerritoryProcessing?: () => void;
  autoSaveGames?: boolean;
  onToggleAutoSave?: (next: boolean) => void;
  gameSaveStatus?: 'idle' | 'saving' | 'saved' | 'saved-local' | 'error';
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
  showLineOverlays,
  onToggleLineOverlays,
  showTerritoryOverlays,
  onToggleTerritoryOverlays,
  phaseLabel,
  phaseHint,
  canSkipTerritoryProcessing,
  onSkipTerritoryProcessing,
  autoSaveGames,
  onToggleAutoSave,
  gameSaveStatus,
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
          <p className="text-[11px] text-slate-400">
            Tap, then use these controls to refine moves.
          </p>
          {phaseHint && <p className="mt-0.5 text-[10px] text-amber-300">{phaseHint}</p>}
        </div>
        <span className="px-2 py-0.5 rounded-full bg-slate-800/80 border border-slate-600 text-[10px] uppercase tracking-wide text-slate-300">
          Phase: {phaseLabel}
        </span>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-[11px]">Selection</span>
          <span className="text-[11px] text-slate-400">Targets: {validTargets.length}</span>
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
            Tap any stack or empty cell to begin. Use the buttons below to clear or adjust your
            selection.
          </p>
        )}

        {canSkipTerritoryProcessing && onSkipTerritoryProcessing && (
          <div className="mt-2 space-y-1">
            <button
              type="button"
              onClick={onSkipTerritoryProcessing}
              className="px-3 py-1.5 rounded-lg border border-amber-400 text-[11px] font-semibold text-amber-100 bg-amber-900/40 hover:border-amber-200 hover:bg-amber-800/70 transition"
              data-testid="sandbox-skip-territory-button"
            >
              Skip territory processing
            </button>
            <p className="text-[10px] text-amber-200/80">
              Leave remaining disconnected regions unprocessed for this turn.
            </p>
          </div>
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

      {onToggleLineOverlays !== undefined && onToggleTerritoryOverlays !== undefined && (
        <div className="border-t border-slate-700 pt-3 space-y-2">
          <span className="font-semibold text-[11px]">Debug overlays</span>
          <div className="flex flex-col gap-1 text-[11px]">
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-amber-500 focus:ring-amber-500"
                checked={showLineOverlays ?? true}
                onChange={(e) => onToggleLineOverlays(e.target.checked)}
              />
              <span className="text-slate-200">Show detected lines</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-fuchsia-500 focus:ring-fuchsia-500"
                checked={showTerritoryOverlays ?? true}
                onChange={(e) => onToggleTerritoryOverlays(e.target.checked)}
              />
              <span className="text-slate-200">Show territory regions</span>
            </label>
          </div>
          <p className="text-[10px] text-slate-400">
            Highlights completed lines and territory control on the board
          </p>
        </div>
      )}

      {onToggleAutoSave !== undefined && autoSaveGames !== undefined && (
        <div className="border-t border-slate-700 pt-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-[11px]">Game storage</span>
            {gameSaveStatus && gameSaveStatus !== 'idle' && (
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                  gameSaveStatus === 'saving'
                    ? 'bg-amber-900/50 text-amber-200 border border-amber-500/50'
                    : gameSaveStatus === 'saved'
                      ? 'bg-emerald-900/50 text-emerald-200 border border-emerald-500/50'
                      : gameSaveStatus === 'saved-local'
                        ? 'bg-amber-900/50 text-amber-200 border border-amber-500/50'
                        : 'bg-red-900/50 text-red-200 border border-red-500/50'
                }`}
              >
                {gameSaveStatus === 'saving' && 'Saving...'}
                {gameSaveStatus === 'saved' && 'Saved'}
                {gameSaveStatus === 'saved-local' && 'Saved locally'}
                {gameSaveStatus === 'error' && 'Error'}
              </span>
            )}
          </div>
          <div className="flex flex-col gap-1 text-[11px]">
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-violet-500 focus:ring-violet-500"
                checked={autoSaveGames}
                onChange={(e) => onToggleAutoSave(e.target.checked)}
              />
              <span className="text-slate-200">Auto-save completed games</span>
            </label>
            <p className="text-[10px] text-slate-400 pl-5">
              Stores finished games to the replay database for analysis
            </p>
          </div>
        </div>
      )}

      {hasCaptureTargets && (
        <div className="border-t border-slate-700 pt-3 space-y-2">
          <span className="font-semibold text-[11px]">Capture segments</span>
          <p className="text-[11px] text-slate-400">
            Pending capture directions are highlighted on the board. Tap a landing cell to continue
            the chain.
          </p>
          <div className="flex flex-wrap gap-1">
            {captureTargets.map((pos, idx) => (
              <span
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
