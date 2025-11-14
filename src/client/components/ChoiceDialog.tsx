import React from 'react';
import {
  PlayerChoice,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice
} from '../../shared/types/game';

export interface ChoiceDialogProps {
  choice: PlayerChoice | null;
  /** Optional absolute deadline (ms since epoch) when this choice expires. */
  deadline?: number | null;
  onSelectOption: <TChoice extends PlayerChoice>(
    choice: TChoice,
    option: TChoice['options'][number]
  ) => void;
  onCancel?: () => void;
}

export const ChoiceDialog: React.FC<ChoiceDialogProps> = ({
  choice,
  deadline,
  onSelectOption,
  onCancel
}) => {
  if (!choice) return null;

  const renderLineOrder = (c: LineOrderChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto">
        {c.options.map((opt, index) => (
          <button
            key={opt.lineId}
            type="button"
            onClick={() => onSelectOption(c, opt)}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs border border-slate-600"
          >
            Line {index + 1} – {opt.markerPositions.length} markers
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
          onClick={() =>
            onSelectOption(
              c,
              'option_1_collapse_all_and_eliminate'
            )
          }
          className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left"
        >
          <div className="font-semibold">Option 1</div>
          <div>Collapse entire line and eliminate one of your rings/caps.</div>
        </button>
        <button
          type="button"
          onClick={() =>
            onSelectOption(
              c,
              'option_2_min_collapse_no_elimination'
            )
          }
          className="px-3 py-2 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-left"
        >
          <div className="font-semibold">Option 2</div>
          <div>Collapse only the minimum required markers with no elimination.</div>
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
            onClick={() => onSelectOption(c, opt)}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600"
          >
            Stack at ({opt.stackPosition.x}, {opt.stackPosition.y}
            {opt.stackPosition.z !== undefined ? `, ${opt.stackPosition.z}` : ''}) – cap {opt.capHeight}, total {opt.totalHeight}
          </button>
        ))}
      </div>
    </div>
  );

  const renderRegionOrder = (c: RegionOrderChoice) => (
    <div className="space-y-2">
      <p className="text-sm text-gray-200 mb-1">{c.prompt}</p>
      <div className="space-y-1 max-h-48 overflow-auto text-xs">
        {c.options.map(opt => (
          <button
            key={opt.regionId}
            type="button"
            onClick={() => onSelectOption(c, opt)}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600"
          >
            Region {opt.regionId} – {opt.size} spaces, sample ({
              opt.representativePosition.x
            }, {opt.representativePosition.y}
            {opt.representativePosition.z !== undefined ? `, ${opt.representativePosition.z}` : ''})
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
            onClick={() => onSelectOption(c, opt)}
            className="w-full text-left px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600"
          >
            Direction {index + 1}: target ({opt.targetPosition.x}, {opt.targetPosition.y}
            {opt.targetPosition.z !== undefined ? `, ${opt.targetPosition.z}` : ''}) → landing ({
              opt.landingPosition.x
            }, {opt.landingPosition.y}
            {opt.landingPosition.z !== undefined ? `, ${opt.landingPosition.z}` : ''}) – cap {
              opt.capturedCapHeight
            }
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
      content = null;
  }

  if (!content) return null;

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md mx-4 p-4 rounded-md bg-slate-900 border border-slate-700 shadow-lg">
        {content}

        <div className="mt-4 flex justify-between items-center space-x-2 text-xs">
          {deadline && (
            <span className="text-[11px] text-gray-400">
              Choice timeout active
            </span>
          )}
          <div className="flex justify-end space-x-2 text-xs">
            {onCancel && (
              <button
                type="button"
                onClick={onCancel}
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-gray-200"
              >
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
