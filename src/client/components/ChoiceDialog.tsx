import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  PlayerChoice,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
} from '../../shared/types/game';
import { getChoiceViewModel, type ChoiceViewModel } from '../adapters/choiceViewModels';
import { getCountdownSeverity } from '../utils/countdown';
import { Dialog } from './ui/Dialog';

const OPTION_BUTTON_SELECTOR = '[data-choice-option]';

export interface ChoiceDialogProps {
  choice: PlayerChoice | null;
  /**
   * Optional precomputed view model. When omitted, the dialog derives its own
   * view model from the PlayerChoice type via getChoiceViewModel.
   */
  choiceViewModel?: ChoiceViewModel | undefined;
  /** Optional absolute deadline (ms since epoch) when this choice expires. */
  deadline?: number | null;
  /** Live countdown supplied by the parent (ms). */
  timeRemainingMs?: number | null;
  /**
   * When true, indicates that the effective countdown has been shortened by a
   * server-emitted timeout warning relative to the client-local baseline. This
   * is a presentation-only hint used to adjust copy and styling; reconciliation
   * semantics remain in useDecisionCountdown.
   */
  isServerCapped?: boolean;
  onSelectOption: <TChoice extends PlayerChoice>(
    choice: TChoice,
    option: TChoice['options'][number]
  ) => void;
  onCancel?: () => void;
}

export const ChoiceDialog: React.FC<ChoiceDialogProps> = ({
  choice,
  choiceViewModel,
  deadline,
  timeRemainingMs,
  isServerCapped,
  onSelectOption,
  onCancel,
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [focusedOptionIndex, setFocusedOptionIndex] = useState(0);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  // Get all option buttons for arrow key navigation
  const getOptionButtons = useCallback((): HTMLButtonElement[] => {
    const dialogEl = dialogRef.current;
    if (!dialogEl) return [];
    return Array.from(dialogEl.querySelectorAll<HTMLButtonElement>(OPTION_BUTTON_SELECTOR));
  }, []);

  // Reset focused option when choice changes
  useEffect(() => {
    setIsSubmitting(false);
    setFocusedOptionIndex(0);
    if (!choice?.id) return;
    const optionButtons = getOptionButtons();
    optionButtons[0]?.focus();
  }, [choice?.id, getOptionButtons]);

  const handleDialogKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (event.defaultPrevented) return;

      // Arrow key navigation between option buttons
      const optionButtons = getOptionButtons();
      if (optionButtons.length > 0 && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
        event.preventDefault();

        // Calculate new index
        let newIndex = focusedOptionIndex;
        if (event.key === 'ArrowUp') {
          newIndex = focusedOptionIndex <= 0 ? optionButtons.length - 1 : focusedOptionIndex - 1;
        } else {
          newIndex = focusedOptionIndex >= optionButtons.length - 1 ? 0 : focusedOptionIndex + 1;
        }

        setFocusedOptionIndex(newIndex);
        optionButtons[newIndex]?.focus();
      }
    },
    [focusedOptionIndex, getOptionButtons]
  );

  // Update focused option index when an option button receives focus
  const handleOptionFocus = useCallback((index: number) => {
    setFocusedOptionIndex(index);
  }, []);

  if (!choice) return null;

  const resolvedChoiceViewModel: ChoiceViewModel = choiceViewModel ?? getChoiceViewModel(choice);

  const totalTimeoutMs = typeof choice.timeoutMs === 'number' ? choice.timeoutMs : null;
  const countdownMs = typeof timeRemainingMs === 'number' ? Math.max(0, timeRemainingMs) : null;
  const countdownSeconds = countdownMs !== null ? Math.ceil(countdownMs / 1000) : null;
  const progressPercent =
    totalTimeoutMs && countdownMs !== null
      ? Math.min(100, Math.max(0, (countdownMs / totalTimeoutMs) * 100))
      : null;

  const severity = countdownMs !== null ? getCountdownSeverity(countdownMs) : null;
  const countdownLabelCopy = isServerCapped ? 'Server deadline – respond within' : 'Respond within';

  const countdownTextClass =
    severity === 'critical'
      ? 'text-[11px] font-mono text-red-200 font-semibold'
      : severity === 'warning'
        ? 'text-[11px] font-mono text-amber-200 font-semibold'
        : severity === 'normal'
          ? 'text-[11px] font-mono text-emerald-200'
          : 'text-[11px] font-mono text-gray-200';

  const progressBarClass =
    severity === 'critical'
      ? 'h-full bg-red-400 rounded transition-all duration-200 animate-pulse'
      : severity === 'warning'
        ? 'h-full bg-amber-400 rounded transition-all duration-200'
        : severity === 'normal'
          ? 'h-full bg-emerald-400 rounded transition-all duration-200'
          : 'h-full bg-amber-400 rounded transition-all duration-200';

  const renderLineOrder = (c: LineOrderChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto" role="listbox" aria-label="Line options">
        {c.options.map((opt, index) => (
          <button
            key={opt.lineId}
            type="button"
            data-choice-option
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            onFocus={() => handleOptionFocus(index)}
            className="w-full text-left px-3 py-2 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 text-xs border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 touch-manipulation"
            role="option"
            aria-selected={index === focusedOptionIndex}
          >
            Line {index + 1} {opt.markerPositions.length} markers
          </button>
        ))}
      </div>
    </div>
  );

  const renderLineReward = (c: LineRewardChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-2">{c.prompt}</p>
      <div className="flex flex-col space-y-2 text-xs" role="listbox" aria-label="Reward options">
        <button
          type="button"
          data-choice-option
          disabled={isSubmitting}
          onClick={() => {
            if (isSubmitting) return;
            setIsSubmitting(true);
            onSelectOption(c, 'option_1_collapse_all_and_eliminate');
          }}
          onFocus={() => handleOptionFocus(0)}
          className="px-4 py-3 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 active:scale-[0.98] transition touch-manipulation"
          role="option"
          aria-selected={focusedOptionIndex === 0}
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-emerald-300">Full Collapse + Elimination</span>
            <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold rounded bg-red-900/60 text-red-300 border border-red-700/50">
              -1 RING
            </span>
          </div>
          <div className="text-gray-300 mt-1">
            Convert entire line to territory. Progress toward victory!
          </div>
        </button>
        <button
          type="button"
          data-choice-option
          disabled={isSubmitting}
          onClick={() => {
            if (isSubmitting) return;
            setIsSubmitting(true);
            onSelectOption(c, 'option_2_min_collapse_no_elimination');
          }}
          onFocus={() => handleOptionFocus(1)}
          className="px-4 py-3 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 active:scale-[0.98] transition touch-manipulation"
          role="option"
          aria-selected={focusedOptionIndex === 1}
        >
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sky-300">Minimum Collapse</span>
            <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold rounded bg-slate-700/60 text-slate-300 border border-slate-600/50">
              NO COST
            </span>
          </div>
          <div className="text-gray-300 mt-1">
            Convert minimum markers to territory, keep extra markers on board
          </div>
        </button>
      </div>
    </div>
  );

  const renderRingElimination = (c: RingEliminationChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div
        className="space-y-1 max-h-48 overflow-auto text-xs"
        role="listbox"
        aria-label="Ring elimination options"
      >
        {c.options.map((opt, index) => {
          // Use ringsToEliminate if available, fallback to capHeight for backwards compatibility
          const ringsToEliminate = opt.ringsToEliminate ?? opt.capHeight;
          const ringLabel = ringsToEliminate === 1 ? 'ring' : 'rings';

          return (
            <button
              key={`${opt.stackPosition.x},${opt.stackPosition.y},${index}`}
              type="button"
              data-choice-option
              disabled={isSubmitting}
              onClick={() => {
                if (isSubmitting) return;
                setIsSubmitting(true);
                onSelectOption(c, opt);
              }}
              onFocus={() => handleOptionFocus(index)}
              className="w-full text-left px-3 py-2 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 text-xs border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 touch-manipulation"
              role="option"
              aria-selected={index === focusedOptionIndex}
            >
              Stack at ({opt.stackPosition.x}, {opt.stackPosition.y}
              {opt.stackPosition.z !== undefined ? `, ${opt.stackPosition.z}` : ''}) — eliminate{' '}
              <span className="text-red-400 font-semibold">
                {ringsToEliminate} {ringLabel}
              </span>{' '}
              (cap {opt.capHeight}, total {opt.totalHeight})
            </button>
          );
        })}
      </div>
    </div>
  );

  const renderRegionOrder = (c: RegionOrderChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div
        className="space-y-2 max-h-48 overflow-auto text-xs"
        role="listbox"
        aria-label="Region options"
      >
        {c.options.map((opt, index) => (
          <button
            key={opt.regionId}
            type="button"
            data-choice-option
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            onFocus={() => handleOptionFocus(index)}
            className="w-full text-left px-3 py-2.5 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 active:scale-[0.98] transition touch-manipulation"
            role="option"
            aria-selected={index === focusedOptionIndex}
          >
            {opt.regionId === 'skip' || opt.size <= 0 ? (
              <div className="flex items-center justify-between">
                <span>Skip territory processing for this turn</span>
                <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold rounded bg-slate-700/60 text-slate-300 border border-slate-600/50">
                  NO COST
                </span>
              </div>
            ) : (
              <div className="flex items-center justify-between">
                <span>
                  Region {opt.size} spaces at ({opt.representativePosition.x},{' '}
                  {opt.representativePosition.y}
                  {opt.representativePosition.z !== undefined
                    ? `, ${opt.representativePosition.z}`
                    : ''}
                  )
                </span>
                <span className="ml-2 px-2 py-0.5 text-[10px] font-semibold rounded bg-amber-900/60 text-amber-300 border border-amber-700/50 whitespace-nowrap">
                  FULL CAP
                </span>
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );

  const renderCaptureDirection = (c: CaptureDirectionChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div
        className="space-y-2 max-h-48 overflow-auto text-xs"
        role="listbox"
        aria-label="Capture direction options"
      >
        {c.options.map((opt, index) => (
          <button
            key={`${opt.targetPosition.x},${opt.targetPosition.y},${index}`}
            type="button"
            data-choice-option
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            onFocus={() => handleOptionFocus(index)}
            className="w-full text-left px-3 py-2.5 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 active:scale-[0.98] transition touch-manipulation"
            role="option"
            aria-selected={index === focusedOptionIndex}
          >
            Direction {index + 1}: target ({opt.targetPosition.x}, {opt.targetPosition.y}
            {opt.targetPosition.z !== undefined ? `, ${opt.targetPosition.z}` : ''}) landing (
            {opt.landingPosition.x}, {opt.landingPosition.y}
            {opt.landingPosition.z !== undefined ? `, ${opt.landingPosition.z}` : ''}) cap{' '}
            {opt.capturedCapHeight}
          </button>
        ))}
      </div>
    </div>
  );

  let content: React.ReactNode = null;

  switch (choice.type) {
    case 'line_order':
      content = renderLineOrder(choice as LineOrderChoice);
      break;
    case 'line_reward_option':
      content = renderLineReward(choice as LineRewardChoice);
      break;
    case 'ring_elimination':
      content = renderRingElimination(choice as RingEliminationChoice);
      break;
    case 'region_order':
      content = renderRegionOrder(choice as RegionOrderChoice);
      break;
    case 'capture_direction':
      content = renderCaptureDirection(choice as CaptureDirectionChoice);
      break;
    default: {
      // Generic fallback for unknown/experimental choice types. This ensures
      // that new PlayerChoice variants remain at least minimally operable
      // (options can still be selected) even before specialised UI is added.
      // Options are rendered as "Option N" without inspecting their shape.

      const genericChoice = choice as PlayerChoice & { options?: unknown[] };
      const options = Array.isArray(genericChoice.options) ? genericChoice.options : [];
      content = (
        <div className="space-y-2">
          <p className="text-sm text-gray-200 mb-1">{genericChoice.prompt}</p>
          {options.length > 0 ? (
            <div
              className="space-y-2 max-h-48 overflow-auto text-xs"
              role="listbox"
              aria-label="Choice options"
            >
              {options.map((opt, index) => (
                <button
                  key={index}
                  type="button"
                  data-choice-option
                  disabled={isSubmitting}
                  onClick={() => {
                    if (isSubmitting) return;
                    setIsSubmitting(true);
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic option for extensibility
                    onSelectOption(genericChoice, opt as any);
                  }}
                  onFocus={() => handleOptionFocus(index)}
                  className="w-full text-left px-3 py-2.5 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-1 focus:ring-offset-slate-900 active:scale-[0.98] transition touch-manipulation"
                  role="option"
                  aria-selected={index === focusedOptionIndex}
                >
                  Option {index + 1}
                </button>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400">
              No options are available for this decision. Please contact support if this persists.
            </p>
          )}
        </div>
      );
      break;
    }
  }

  if (!content) return null;

  const canCancel = Boolean(onCancel) && !isSubmitting;
  const describedBy = resolvedChoiceViewModel?.copy.description
    ? 'choice-dialog-description'
    : undefined;

  return (
    <Dialog
      isOpen
      onClose={() => onCancel?.()}
      closeOnEscape={canCancel}
      closeOnBackdropClick={false}
      labelledBy="choice-dialog-title"
      describedBy={describedBy}
      overlayClassName="z-40 items-center justify-center"
      className="w-full max-w-md mx-4"
      onKeyDown={handleDialogKeyDown}
    >
      <div
        ref={dialogRef}
        className="p-4 rounded-md bg-slate-900 border border-slate-700 shadow-lg"
      >
        {resolvedChoiceViewModel && (
          <div className="mb-3">
            <div className="text-[11px] uppercase tracking-wide text-emerald-300/80">
              {resolvedChoiceViewModel.copy.shortLabel}
            </div>
            <h2 id="choice-dialog-title" className="text-sm font-semibold text-gray-100">
              {resolvedChoiceViewModel.copy.title}
            </h2>
            {resolvedChoiceViewModel.copy.description && (
              <p id="choice-dialog-description" className="mt-0.5 text-xs text-gray-400">
                {resolvedChoiceViewModel.copy.description}
              </p>
            )}
            {resolvedChoiceViewModel.copy.strategicTip && (
              <div className="mt-2 px-2 py-1.5 rounded bg-amber-950/40 border border-amber-800/30">
                <span className="text-[10px] uppercase tracking-wide text-amber-400/80 font-semibold">
                  💡 Tip:{' '}
                </span>
                <span className="text-xs text-amber-200/90">
                  {resolvedChoiceViewModel.copy.strategicTip}
                </span>
              </div>
            )}
          </div>
        )}

        {content}

        <div className="mt-4 flex flex-col space-y-2 text-xs">
          {deadline && (
            <div
              data-testid="choice-countdown"
              data-severity={severity ?? undefined}
              data-server-capped={isServerCapped ? 'true' : undefined}
              aria-live="polite"
              aria-atomic="true"
              role="timer"
            >
              {countdownSeconds !== null ? (
                <>
                  <div className="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                    <span>{countdownLabelCopy}</span>
                    <span className={countdownTextClass}>{countdownSeconds}s</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded">
                    <div
                      className={progressBarClass}
                      data-testid="choice-countdown-bar"
                      style={{ width: `${progressPercent ?? 100}%` }}
                    />
                  </div>
                </>
              ) : (
                <span className="text-[11px] text-gray-400">Choice timeout active</span>
              )}
            </div>
          )}
          {onCancel && (
            <div className="flex justify-end">
              <button
                type="button"
                disabled={isSubmitting}
                onClick={() => {
                  if (isSubmitting) return;
                  onCancel();
                }}
                className="px-4 py-2.5 min-h-[44px] rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-gray-200 disabled:opacity-60 disabled:cursor-not-allowed active:scale-[0.98] transition touch-manipulation"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
};
