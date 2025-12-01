import React, { useState } from 'react';
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

export interface ChoiceDialogProps {
  choice: PlayerChoice | null;
  /**
   * Optional precomputed view model. When omitted, the dialog derives its own
   * view model from the PlayerChoice type via getChoiceViewModel.
   */
  choiceViewModel?: ChoiceViewModel;
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
  const countdownLabelCopy = isServerCapped
    ? 'Server deadline – respond within'
    : 'Respond within';

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
      <div className="space-y-1 max-h-48 overflow-auto">
        {c.options.map((opt, index) => (
          <button
            key={opt.lineId}
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Line {index + 1}  {opt.markerPositions.length} markers
          </button>
        ))}
      </div>
    </div>
  );

  const renderLineReward = (c: LineRewardChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-2">{c.prompt}</p>
      <div className="flex flex-col space-y-2 text-xs">
        <button
          type="button"
          disabled={isSubmitting}
          onClick={() => {
            if (isSubmitting) return;
            setIsSubmitting(true);
            onSelectOption(c, 'option_1_collapse_all_and_eliminate');
          }}
          className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <div className="font-semibold text-emerald-300">Full Collapse + Elimination Bonus</div>
          <div className="text-gray-300">
            Convert entire line to territory and eliminate 1 of your rings (progress toward victory!)
          </div>
        </button>
        <button
          type="button"
          disabled={isSubmitting}
          onClick={() => {
            if (isSubmitting) return;
            setIsSubmitting(true);
            onSelectOption(c, 'option_2_min_collapse_no_elimination');
          }}
          className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <div className="font-semibold text-sky-300">Minimum Collapse</div>
          <div className="text-gray-300">
            Convert minimum markers to territory, keep extra markers on board
          </div>
        </button>
      </div>
    </div>
  );

  const renderRingElimination = (c: RingEliminationChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto text-xs">
        {c.options.map((opt, index) => (
          <button
            key={`${opt.stackPosition.x},${opt.stackPosition.y},${index}`}
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Stack at ({opt.stackPosition.x}, {opt.stackPosition.y}
            {opt.stackPosition.z !== undefined ? `, ${opt.stackPosition.z}` : ''})  cap{' '}
            {opt.capHeight}, total {opt.totalHeight}
          </button>
        ))}
      </div>
    </div>
  );

  const renderRegionOrder = (c: RegionOrderChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto text-xs">
        {c.options.map((opt) => (
          <button
            key={opt.regionId}
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Region {opt.regionId}  {opt.size} spaces, sample ({opt.representativePosition.x},{' '}
            {opt.representativePosition.y}
            {opt.representativePosition.z !== undefined
              ? `, ${opt.representativePosition.z}`
              : ''})
          </button>
        ))}
      </div>
    </div>
  );

  const renderCaptureDirection = (c: CaptureDirectionChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto text-xs">
        {c.options.map((opt, index) => (
          <button
            key={`${opt.targetPosition.x},${opt.targetPosition.y},${index}`}
            type="button"
            disabled={isSubmitting}
            onClick={() => {
              if (isSubmitting) return;
              setIsSubmitting(true);
              onSelectOption(c, opt);
            }}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            Direction {index + 1}: target ({opt.targetPosition.x}, {opt.targetPosition.y}
            {opt.targetPosition.z !== undefined ? `, ${opt.targetPosition.z}` : ''})  landing (
            {opt.landingPosition.x}, {opt.landingPosition.y}
            {opt.landingPosition.z !== undefined ? `, ${opt.landingPosition.z}` : ''})  cap{' '}
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
    default:
      // Generic fallback for unknown/experimental choice types. This ensures
      // that new PlayerChoice variants remain at least minimally operable
      // (options can still be selected) even before specialised UI is added.
      // Options are rendered as "Option N" without inspecting their shape.
      content = (
        <div className="space-y-2">
          <p className="text-sm text-gray-200 mb-1">{(choice as any).prompt}</p>
          {Array.isArray((choice as any).options) && (choice as any).options.length > 0 ? (
            <div className="space-y-1 max-h-48 overflow-auto text-xs">
              {(choice as any).options.map((opt: unknown, index: number) => (
                <button
                  // eslint-disable-next-line react/no-array-index-key
                  key={index}
                  type="button"
                  disabled={isSubmitting}
                  onClick={() => {
                    if (isSubmitting) return;
                    setIsSubmitting(true);
                    onSelectOption(choice as PlayerChoice, opt as any);
                  }}
                  className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 disabled:opacity-60 disabled:cursor-not-allowed"
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

  if (!content) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md mx-4 p-4 rounded-md bg-slate-900 border border-slate-700 shadow-lg">
        {resolvedChoiceViewModel && (
          <div className="mb-3">
            <div className="text-[11px] uppercase tracking-wide text-emerald-300/80">
              {resolvedChoiceViewModel.copy.shortLabel}
            </div>
            <h2 className="text-sm font-semibold text-gray-100">
              {resolvedChoiceViewModel.copy.title}
            </h2>
            {resolvedChoiceViewModel.copy.description && (
              <p className="mt-0.5 text-xs text-gray-400">
                {resolvedChoiceViewModel.copy.description}
              </p>
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
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-gray-200 disabled:opacity-60 disabled:cursor-not-allowed"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
