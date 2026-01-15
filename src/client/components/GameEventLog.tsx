import type { GameHistoryEntry, GameResult, BoardType } from '../../shared/types/game';
import { toEventLogViewModel, type EventLogViewModel } from '../adapters/gameViewModels';

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
  /** Board type for position formatting (defaults to square8) */
  boardType?: BoardType;
  /** When true, square board ranks are computed from the bottom (chess style) */
  squareRankFromBottom?: boolean;
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

/**
 * Convert legacy props to view model for internal use.
 * Delegates to the canonical toEventLogViewModel adapter so that both
 * backend and sandbox hosts share identical event formatting.
 */
function toLegacyViewModel(props: GameEventLogLegacyProps): EventLogViewModel {
  const {
    history,
    systemEvents = [],
    victoryState,
    maxEntries,
    boardType,
    squareRankFromBottom,
  } = props;

  // Default squareRankFromBottom based on board type if not explicitly provided
  const effectiveBoardType = boardType ?? 'square8';
  const effectiveSquareRankFromBottom =
    squareRankFromBottom ?? (effectiveBoardType === 'square8' || effectiveBoardType === 'square19');

  return toEventLogViewModel(history, systemEvents, victoryState ?? null, {
    maxEntries: maxEntries ?? DEFAULT_MAX_ENTRIES,
    boardType: effectiveBoardType,
    squareRankFromBottom: effectiveSquareRankFromBottom,
  });
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

  const { entries, victoryMessage: _victoryMessage, hasContent } = viewModel;

  // Separate entries by type for structured rendering
  const victoryEntry = entries.find((e) => e.type === 'victory');
  const moveEntries = entries.filter((e) => e.type === 'move');
  const systemEntries = entries.filter((e) => e.type === 'system');

  return (
    <div
      className="p-3 border border-slate-700 rounded-xl bg-slate-900/70 max-h-64 overflow-y-auto"
      data-testid="game-event-log"
      role="log"
      aria-live="polite"
      aria-labelledby="game-event-log-title"
    >
      <h2 id="game-event-log-title" className="font-semibold mb-2 text-sm">
        Game log
      </h2>

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
