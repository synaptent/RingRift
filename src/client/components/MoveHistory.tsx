import { useRef, useEffect } from 'react';
import { formatMove, formatPosition, MoveNotationOptions } from '../../shared/engine/notation';
import type { Move, BoardType, GameHistoryEntry } from '../../shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

export interface MoveHistoryProps {
  /** Array of moves from game state */
  moves: Move[];
  /** Board type for position notation formatting */
  boardType: BoardType;
  /** Current move index (0-based), usually moves.length - 1 for the latest */
  currentMoveIndex?: number | undefined;
  /** Optional callback when a move is clicked (for future position review) */
  onMoveClick?: ((index: number) => void) | undefined;
  /** Maximum height for the scrollable area */
  maxHeight?: string | undefined;
  /** Optional class name for additional styling */
  className?: string | undefined;
  /**
   * Optional notation options to customize coordinate display.
   * When provided, these override the default notation derived from boardType.
   * Use this for sandbox contexts where the board labels use bottom-origin ranks.
   */
  notationOptions?: MoveNotationOptions | undefined;
}

/**
 * Alternative props using GameHistoryEntry for richer data
 */
export interface MoveHistoryFromEntriesProps {
  /** Array of history entries from game state */
  entries: GameHistoryEntry[];
  /** Board type for position notation formatting */
  boardType: BoardType;
  /** Current move index (0-based), usually entries.length - 1 for the latest */
  currentMoveIndex?: number | undefined;
  /** Optional callback when a move is clicked (for future position review) */
  onMoveClick?: ((index: number) => void) | undefined;
  /** Maximum height for the scrollable area */
  maxHeight?: string | undefined;
  /** Optional class name for additional styling */
  className?: string | undefined;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Get a short notation for the move type
 */
function getMoveTypeSymbol(move: Move): string {
  switch (move.type) {
    case 'place_ring':
      return '◯'; // Ring symbol for placement
    case 'move_ring':
    case 'move_stack':
      return '→'; // Arrow for movement
    case 'overtaking_capture':
    case 'continue_capture_segment':
      return '×'; // Cross for capture
    case 'swap_sides':
      return '⇆'; // Swap symbol
    case 'skip_placement':
      return '−'; // Skip
    case 'recovery_slide':
      return '↻'; // Recovery action
    case 'process_line':
    case 'choose_line_reward':
      return '━'; // Line
    case 'process_territory_region':
    case 'eliminate_rings_from_stack':
      return '◼'; // Territory
    default:
      return '•';
  }
}

/**
 * Format a move into compact notation
 */
function formatMoveCompact(move: Move, options: MoveNotationOptions): string {
  const typeSymbol = getMoveTypeSymbol(move);

  switch (move.type) {
    case 'place_ring': {
      const count = move.placementCount && move.placementCount > 1 ? `×${move.placementCount}` : '';
      const pos = formatPosition(move.to, options);
      return `${typeSymbol}${pos}${count}`;
    }
    case 'move_ring':
    case 'move_stack': {
      if (move.from) {
        return `${formatPosition(move.from, options)}${typeSymbol}${formatPosition(move.to, options)}`;
      }
      return `${typeSymbol}${formatPosition(move.to, options)}`;
    }
    case 'overtaking_capture':
    case 'continue_capture_segment': {
      if (move.from && move.captureTarget) {
        return `${formatPosition(move.from, options)}${typeSymbol}${formatPosition(move.captureTarget, options)}→${formatPosition(move.to, options)}`;
      }
      if (move.from) {
        return `${formatPosition(move.from, options)}${typeSymbol}${formatPosition(move.to, options)}`;
      }
      return `${typeSymbol}${formatPosition(move.to, options)}`;
    }
    case 'swap_sides':
      return `${typeSymbol} Swap`;
    case 'skip_placement':
      return `${typeSymbol} Pass`;
    case 'recovery_slide': {
      // Recovery: ↻ a3→b3 or ↻ a3→b3 (min) for Option 2
      if (move.from) {
        const optionSuffix = move.recoveryOption === 2 ? ' (min)' : '';
        return `${typeSymbol}${formatPosition(move.from, options)}→${formatPosition(move.to, options)}${optionSuffix}`;
      }
      return `${typeSymbol}${formatPosition(move.to, options)}`;
    }
    case 'process_line':
    case 'choose_line_reward':
      return `${typeSymbol} Line`;
    case 'process_territory_region':
    case 'eliminate_rings_from_stack':
      return `${typeSymbol} Territory`;
    default:
      return formatMove(move, options);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Move Item Component
// ═══════════════════════════════════════════════════════════════════════════

interface MoveItemProps {
  move: Move;
  index: number;
  isCurrentMove: boolean;
  boardType: BoardType;
  onClick?: ((index: number) => void) | undefined;
  /** Optional notation options for custom coordinate display */
  notationOptions?: MoveNotationOptions | undefined;
}

function MoveItem({
  move,
  index,
  isCurrentMove,
  boardType,
  onClick,
  notationOptions: explicitOptions,
}: MoveItemProps) {
  // Merge explicit notation options with boardType fallback
  const notationOptions: MoveNotationOptions = explicitOptions ?? { boardType };
  const notation = formatMoveCompact(move, notationOptions);

  // Get player indicator color class
  const playerColorClass = (() => {
    switch (move.player) {
      case 1:
        return 'bg-emerald-500';
      case 2:
        return 'bg-sky-500';
      case 3:
        return 'bg-amber-500';
      case 4:
        return 'bg-fuchsia-500';
      default:
        return 'bg-slate-500';
    }
  })();

  return (
    <button
      type="button"
      className={`
        flex items-center gap-1.5 px-2 py-1 rounded text-xs font-mono transition-colors
        ${
          isCurrentMove
            ? 'bg-blue-900/50 border border-blue-500/60 text-blue-100'
            : 'hover:bg-slate-800/50 text-slate-300 border border-transparent'
        }
        ${onClick ? 'cursor-pointer' : 'cursor-default'}
      `}
      onClick={() => onClick?.(index)}
      disabled={!onClick}
      aria-current={isCurrentMove ? 'true' : undefined}
      title={`Move ${index + 1}: ${formatMove(move, notationOptions)}`}
    >
      {/* Move number */}
      <span className="text-slate-500 w-5 text-right">{index + 1}.</span>

      {/* Player indicator */}
      <span
        className={`w-2 h-2 rounded-full ${playerColorClass} shrink-0`}
        aria-label={`Player ${move.player}`}
      />

      {/* Move notation */}
      <span className="truncate">{notation}</span>

      {/* Current move indicator */}
      {isCurrentMove && <span className="ml-auto text-blue-400 text-[10px]">◀</span>}
    </button>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * MoveHistory Component
 *
 * Displays a compact, scrollable list of moves in chess-like notation.
 * Highlights the current move and auto-scrolls to keep it visible.
 *
 * @example
 * ```tsx
 * <MoveHistory
 *   moves={gameState.moveHistory}
 *   boardType={gameState.boardType}
 *   currentMoveIndex={gameState.moveHistory.length - 1}
 *   onMoveClick={(index) => console.log('Review move', index)}
 * />
 * ```
 */
export function MoveHistory({
  moves,
  boardType,
  currentMoveIndex,
  onMoveClick,
  maxHeight = 'max-h-48',
  className = '',
  notationOptions,
}: MoveHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentMoveRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to current move when it changes (within container only)
  useEffect(() => {
    if (currentMoveRef.current && scrollRef.current) {
      const container = scrollRef.current;
      const element = currentMoveRef.current;

      // Calculate if element is visible within the scroll container
      const containerRect = container.getBoundingClientRect();
      const elementRect = element.getBoundingClientRect();

      const isAboveViewport = elementRect.top < containerRect.top;
      const isBelowViewport = elementRect.bottom > containerRect.bottom;

      // Only scroll within the container, don't affect page scroll
      if (isAboveViewport || isBelowViewport) {
        const scrollTop = element.offsetTop - container.offsetTop;
        container.scrollTo({
          top: Math.max(0, scrollTop - container.clientHeight / 2 + element.clientHeight / 2),
          behavior: 'smooth',
        });
      }
    }
  }, [currentMoveIndex]);

  if (!moves || moves.length === 0) {
    return (
      <div
        className={`p-3 border border-slate-700 rounded bg-slate-900/50 ${className}`}
        data-testid="move-history"
      >
        <h2 className="font-semibold mb-2 text-sm text-slate-200">Moves</h2>
        <div className="text-slate-400 text-xs">No moves yet.</div>
      </div>
    );
  }

  const effectiveCurrentIndex = currentMoveIndex ?? moves.length - 1;

  return (
    <div
      className={`p-3 border border-slate-700 rounded bg-slate-900/50 ${className}`}
      data-testid="move-history"
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-sm text-slate-200">Moves</h2>
        <span className="text-[10px] text-slate-500">{moves.length} total</span>
      </div>

      <div
        ref={scrollRef}
        className={`${maxHeight} overflow-y-auto pr-1 space-y-0.5`}
        role="list"
        aria-label="Move history"
      >
        {moves.map((move, index) => (
          <div
            key={`${move.id ?? 'move'}-${index}`}
            ref={index === effectiveCurrentIndex ? currentMoveRef : undefined}
            role="listitem"
          >
            <MoveItem
              move={move}
              index={index}
              isCurrentMove={index === effectiveCurrentIndex}
              boardType={boardType}
              onClick={onMoveClick}
              notationOptions={notationOptions}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * MoveHistoryFromEntries Component
 *
 * Alternative component that works with GameHistoryEntry array
 * for richer move information.
 */
export function MoveHistoryFromEntries({
  entries,
  boardType,
  currentMoveIndex,
  onMoveClick,
  maxHeight = 'max-h-48',
  className = '',
}: MoveHistoryFromEntriesProps) {
  // Extract moves from entries
  const moves = entries.map((entry) => entry.action);

  return (
    <MoveHistory
      moves={moves}
      boardType={boardType}
      currentMoveIndex={currentMoveIndex}
      onMoveClick={onMoveClick}
      maxHeight={maxHeight}
      className={className}
    />
  );
}

export default MoveHistory;
