import React, { useState, useEffect, useRef, useCallback, useMemo, useLayoutEffect } from 'react';
import {
  BoardType,
  BoardState,
  Position,
  RingStack,
  positionToString,
  positionsEqual,
} from '../../shared/types/game';
import { formatPosition, type MoveNotationOptions } from '../../shared/engine/notation';
import { debugLog, isSandboxAnimationDebugEnabled } from '../../shared/utils/envFlags';
import { computeBoardMovementGrid } from '../utils/boardMovementGrid';
import type { MovementGrid } from '../utils/boardMovementGrid';
import type { BoardViewModel, CellViewModel, StackViewModel } from '../adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// Touch Gesture Support
// ═══════════════════════════════════════════════════════════════════════════

// Touch timing constants - per RR-CANON-R076, this is a Host/Adapter Layer UX concern
const LONG_PRESS_DELAY_MS = 500;
const DOUBLE_TAP_DELAY_MS = 300;

// Keyboard navigation helpers
interface FocusableCell {
  position: Position;
  key: string;
}

// Animation type for tracking piece animations
type AnimationType = 'place' | 'move' | 'capture' | 'chain_capture';

interface AnimationState {
  position: string; // positionToString key
  type: AnimationType;
}

/**
 * Animation data for a move, capture, or chain capture.
 * Used to animate pieces traveling from source to destination(s).
 */
export interface MoveAnimationData {
  /** Type of animation: 'move' for regular moves, 'capture' for captures, 'chain_capture' for chain captures */
  type: 'move' | 'capture' | 'chain_capture' | 'place';
  /** Source position (where the piece started) */
  from?: Position;
  /** Destination position (where the piece ended) */
  to: Position;
  /**
   * For chain captures: array of intermediate landing positions visited
   * during the chain, in order. The animation will visit each position sequentially.
   */
  intermediatePositions?: Position[];
  /** Player number who made the move (for piece color) */
  playerNumber: number;
  /** Stack height at the source (for rendering the moving piece) */
  stackHeight?: number;
  /** Cap height at the source */
  capHeight?: number;
  /** Unique ID for this animation (used for React keys and deduplication) */
  id: string;
}

export interface BoardViewProps {
  boardType: BoardType;
  board: BoardState;
  /**
   * Optional precomputed board view model. When provided, this is used
   * for selection/valid-target state and cell-level presentation data,
   * while still allowing the component to fall back to the raw BoardState
   * for geometry and animation until the full view-model migration is
   * complete.
   */
  viewModel?: BoardViewModel;
  /** Currently selected position, or undefined if nothing is selected */
  selectedPosition?: Position | undefined;
  validTargets?: Position[];
  onCellClick?: ((position: Position) => void) | undefined;
  isSpectator?: boolean;
  /**
   * Optional double-click handler, primarily used by the local sandbox
   * to distinguish between selection (single click) and stacked ring
   * placement (double click) during the ring placement phase.
   */
  onCellDoubleClick?: ((position: Position) => void) | undefined;
  /**
   * Optional context menu handler (right-click / long-press proxy),
   * used by the local sandbox to surface ring-count selection dialogs
   * for multi-ring placements.
   */
  onCellContextMenu?: ((position: Position) => void) | undefined;
  /**
   * Optional movement grid overlay toggle. When true, an SVG overlay
   * renders faint movement lines and node dots based on a board-local
   * normalized geometry. This is shared between square and hex boards
   * and is intended for future use by the sandbox harness, history
   * visualizations, and AI/debug overlays.
   */
  showMovementGrid?: boolean;
  /**
   * When true, render algebraic coordinate labels (files/ranks) around
   * square boards. Currently leveraged by the sandbox to aid manual play.
   */
  showCoordinateLabels?: boolean;
  /**
   * When true, square board ranks are labeled from the bottom (chess-style):
   * bottom row = rank 1, top row = rank size. This matches the visual
   * convention where row y=0 (rendered at top) has the highest rank number.
   *
   * When false (default), ranks match the canonical notation system:
   * top row = rank 1, bottom row = rank size.
   *
   * This prop should be coordinated with the `squareRankFromBottom` option
   * in `MoveNotationOptions` so that MoveHistory coordinates align with
   * the board edge labels.
   */
  squareRankFromBottom?: boolean;
  /**
   * Optional rules-lab overlay: when true, highlight any detected lines
   * present in board.formedLines. Primarily used by the sandbox host and
   * scenario viewers for debugging line geometry.
   */
  showLineOverlays?: boolean;
  /**
   * Optional rules-lab overlay: when true, highlight territory/disconnected
   * regions derived from board.territories. Intended for sandbox/rules-lab
   * use rather than general gameplay.
   */
  showTerritoryRegionOverlays?: boolean;
  /**
   * Optional animation data for the last move. When provided, an animated
   * piece will travel from source to destination, visiting intermediate
   * positions for chain captures.
   */
  pendingAnimation?: MoveAnimationData | undefined;
  /**
   * Callback when an animation completes. Used to clear the animation state.
   */
  onAnimationComplete?: () => void;
  /**
   * Optional callback to show keyboard shortcuts help overlay.
   * Triggered when user presses "?" key while focused on the board.
   */
  onShowKeyboardHelp?: () => void;
  /**
   * Optional chain capture path to visualize. When provided, renders
   * arrows connecting the positions in the capture sequence. Used during
   * chain_capture phase to show the path traversed so far.
   */
  chainCapturePath?: Position[];
  /**
   * Position key of a cell that should display the invalid-move shake animation.
   * Used to provide visual feedback when a player attempts an invalid move.
   * The animation is typically cleared automatically after ~400ms.
   */
  shakingCellKey?: string | null;
}

// Tailwind-friendly, fixed color classes per player number to avoid
// dynamic class name generation that PurgeCSS cannot see.
const PLAYER_COLOR_CLASSES: Record<
  number,
  { ring: string; ringBorder: string; marker: string; territory: string }
> = {
  1: {
    ring: 'bg-emerald-400',
    ringBorder: 'border-emerald-200',
    marker: 'border-emerald-400',
    territory: 'bg-emerald-700/85',
  },
  2: {
    ring: 'bg-sky-600',
    ringBorder: 'border-sky-300',
    marker: 'border-sky-500',
    territory: 'bg-sky-700/85',
  },
  3: {
    ring: 'bg-amber-400',
    ringBorder: 'border-amber-200',
    marker: 'border-amber-400',
    territory: 'bg-amber-600/85',
  },
  4: {
    ring: 'bg-fuchsia-400',
    ringBorder: 'border-fuchsia-200',
    marker: 'border-fuchsia-400',
    territory: 'bg-fuchsia-700/85',
  },
};

const getPlayerColors = (playerNumber?: number) => {
  if (!playerNumber) {
    return {
      ring: 'bg-slate-300',
      ringBorder: 'border-slate-100',
      marker: 'border-slate-300',
      territory: 'bg-slate-800/70',
    };
  }
  return (
    PLAYER_COLOR_CLASSES[playerNumber] || {
      ring: 'bg-slate-300',
      ringBorder: 'border-slate-100',
      marker: 'border-slate-300',
      territory: 'bg-slate-800/70',
    }
  );
};

// Contrasting highlight colors for just-moved rings (complements player colors)
const getJustMovedHighlight = (playerNumber?: number): string => {
  switch (playerNumber) {
    case 1: // Emerald → Rose highlight
      return 'ring-[2px] ring-rose-400 ring-inset';
    case 2: // Sky/Blue → Orange highlight
      return 'ring-[2px] ring-orange-400 ring-inset';
    case 3: // Amber/Yellow → Violet highlight
      return 'ring-[2px] ring-violet-400 ring-inset';
    case 4: // Fuchsia → Cyan highlight
      return 'ring-[2px] ring-cyan-400 ring-inset';
    default:
      return 'ring-[2px] ring-white ring-inset';
  }
};

// Derive highlight from colorClass for ViewModel-based rendering
const getJustMovedHighlightFromColorClass = (colorClass: string): string => {
  if (colorClass.includes('emerald')) return 'ring-[2px] ring-rose-400 ring-inset';
  if (colorClass.includes('sky')) return 'ring-[2px] ring-orange-400 ring-inset';
  if (colorClass.includes('amber')) return 'ring-[2px] ring-violet-400 ring-inset';
  if (colorClass.includes('fuchsia')) return 'ring-[2px] ring-cyan-400 ring-inset';
  return 'ring-[2px] ring-white ring-inset';
};

const generateFileLabels = (size: number, skipI = false): string[] => {
  const labels: string[] = [];
  let code = 'a'.charCodeAt(0);
  while (labels.length < size) {
    const char = String.fromCharCode(code);
    if (skipI && char === 'i') {
      code += 1;
      continue;
    }
    labels.push(char);
    code += 1;
  }
  return labels;
};

/**
 * Generate rank labels for square boards.
 * @param size - The board size (8 or 19)
 * @param fromBottom - If true, labels are chess-style: bottom row = rank 1.
 *                     If false (default), labels match canonical notation: top row = rank 1.
 */
const generateRankLabels = (size: number, fromBottom = false): string[] =>
  fromBottom
    ? Array.from({ length: size }, (_, idx) => (size - idx).toString()) // [size..1] - chess style
    : Array.from({ length: size }, (_, idx) => (idx + 1).toString()); // [1..size] - canonical style

const StackWidget: React.FC<{
  stack: RingStack;
  boardType: BoardType;
  animationClass?: string | undefined;
  isSelected?: boolean | undefined;
  ownerPlayerId?: number | undefined;
  isJustMoved?: boolean | undefined;
}> = ({
  stack,
  boardType,
  animationClass,
  isSelected = false,
  ownerPlayerId,
  isJustMoved = false,
}) => {
  const { rings, capHeight, stackHeight } = stack;

  // Engine semantics: rings[0] is the top ring (cap); additional rings
  // are appended toward the bottom. Reflect that here so colors and cap
  // highlighting match the actual stack state.
  const topIndex = 0;
  const capEndIndex = Math.min(capHeight - 1, rings.length - 1);

  const isSquare8 = boardType === 'square8';
  const isHex = boardType === 'hexagonal';

  // Slight vertical offset so stacks sit comfortably inside both square
  // and hex cells, leaving room for tall stacks while keeping labels legible.
  const verticalOffsetClasses = 'translate-y-[3px] md:translate-y-[4px]';

  // Mobile-responsive ring sizing for W3-12 (44px mobile cells)
  const ringSizeClasses = isSquare8
    ? 'w-5 sm:w-6 md:w-8 h-[4px] sm:h-[5px] md:h-[6px]'
    : isHex
      ? 'w-5 md:w-6 h-[3px] md:h-[4px]'
      : 'w-5 sm:w-6 md:w-7 h-[4px] sm:h-[4px] md:h-[5px]';

  const labelTextClasses = isSquare8
    ? 'text-[8px] sm:text-[9px] md:text-[10px]'
    : isHex
      ? 'text-[7px] md:text-[8px]'
      : 'text-[7px] sm:text-[8px] md:text-[9px]';

  // Selection pulse classes: when selected, apply player-specific pulse animation
  const selectionClasses =
    isSelected && ownerPlayerId
      ? `animate-selection-pulse selected-piece player-${ownerPlayerId}`
      : '';

  return (
    <div
      className={`flex flex-col items-center justify-center gap-[1px] ${verticalOffsetClasses} ${animationClass || ''} ${selectionClasses}`}
    >
      <div className="flex flex-col items-center -space-y-[1px]">
        {rings.map((playerNumber, index) => {
          const { ring, ringBorder } = getPlayerColors(playerNumber);
          const isTop = index === topIndex;
          const isInCap = index <= capEndIndex;

          // Base shape with 3px border for visibility
          const baseShape = `${ringSizeClasses} rounded-full border-[3px]`;
          // Cap outline for rings in the cap
          const capOutline = isInCap
            ? 'ring-[0.5px] ring-offset-[0.5px] ring-offset-slate-900'
            : '';
          // Just-moved highlight: contrasting color based on ring color
          const justMovedHighlight = isJustMoved ? getJustMovedHighlight(playerNumber) : '';
          const topShadow = isTop ? 'shadow-md shadow-slate-900/70' : 'shadow-sm';

          return (
            <div
              key={index}
              className={`${baseShape} ${ring} ${ringBorder} ${capOutline} ${justMovedHighlight} ${topShadow}`}
            />
          );
        })}
      </div>
      <div className={`mt-[1px] leading-tight font-semibold text-slate-900 ${labelTextClasses}`}>
        H{stackHeight} C{capHeight}
      </div>
    </div>
  );
};

const StackFromViewModel: React.FC<{
  stack: StackViewModel;
  boardType: BoardType;
  animationClass?: string | undefined;
  isSelected?: boolean | undefined;
  ownerPlayerId?: number | undefined;
  isJustMoved?: boolean | undefined;
}> = ({
  stack,
  boardType,
  animationClass,
  isSelected = false,
  ownerPlayerId,
  isJustMoved = false,
}) => {
  const { rings, stackHeight, capHeight } = stack;

  const isSquare8 = boardType === 'square8';
  const isHex = boardType === 'hexagonal';

  const verticalOffsetClasses = 'translate-y-[3px] md:translate-y-[4px]';

  // Mobile-responsive ring sizing for W3-12 (44px mobile cells)
  const ringSizeClasses = isSquare8
    ? 'w-5 sm:w-6 md:w-8 h-[4px] sm:h-[5px] md:h-[6px]'
    : isHex
      ? 'w-5 md:w-6 h-[3px] md:h-[4px]'
      : 'w-5 sm:w-6 md:w-7 h-[4px] sm:h-[4px] md:h-[5px]';

  const labelTextClasses = isSquare8
    ? 'text-[8px] sm:text-[9px] md:text-[10px]'
    : isHex
      ? 'text-[7px] md:text-[8px]'
      : 'text-[7px] sm:text-[8px] md:text-[9px]';

  const selectionClasses =
    isSelected && ownerPlayerId
      ? `animate-selection-pulse selected-piece player-${ownerPlayerId}`
      : '';

  return (
    <div
      className={`flex flex-col items-center justify-center gap-[1px] ${verticalOffsetClasses} ${animationClass || ''} ${selectionClasses}`}
    >
      <div className="flex flex-col items-center -space-y-[1px]">
        {rings.map((ringVM, index) => {
          const { colorClass, borderClass, isTop, isInCap } = ringVM;

          // Base shape with 3px border for visibility
          const baseShape = `${ringSizeClasses} rounded-full border-[3px]`;
          const capOutline = isInCap
            ? 'ring-[0.5px] ring-offset-[0.5px] ring-offset-slate-900'
            : '';
          // Just-moved highlight: contrasting color based on ring color
          const justMovedHighlight = isJustMoved
            ? getJustMovedHighlightFromColorClass(colorClass)
            : '';
          const topShadow = isTop ? 'shadow-md shadow-slate-900/70' : 'shadow-sm';

          return (
            <div
              key={index}
              className={`${baseShape} ${colorClass} ${borderClass} ${capOutline} ${justMovedHighlight} ${topShadow}`}
            />
          );
        })}
      </div>
      <div className={`mt-[1px] leading-tight font-semibold text-slate-900 ${labelTextClasses}`}>
        H{stackHeight} C{capHeight}
      </div>
    </div>
  );
};

/**
 * MoveAnimationLayer: Renders an animated piece traveling from source to destination.
 * For chain captures, visits each intermediate position sequentially.
 */
interface MoveAnimationLayerProps {
  animation: MoveAnimationData;
  cellRefs: React.MutableRefObject<Map<string, HTMLButtonElement>>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  onComplete: () => void;
}

const MoveAnimationLayer: React.FC<MoveAnimationLayerProps> = ({
  animation,
  cellRefs,
  containerRef,
  onComplete,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);

  // Build the full path of positions to visit
  const path = useMemo(() => {
    const positions: Position[] = [];
    if (animation.from) {
      positions.push(animation.from);
    }
    if (animation.intermediatePositions) {
      positions.push(...animation.intermediatePositions);
    }
    positions.push(animation.to);
    return positions;
  }, [animation]);

  // Get pixel position for a cell
  const getCellCenter = useCallback(
    (pos: Position): { x: number; y: number } | null => {
      const key = positionToString(pos);
      const cell = cellRefs.current.get(key);
      const container = containerRef.current;
      if (!cell || !container) return null;

      const cellRect = cell.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();

      return {
        x: cellRect.left - containerRect.left + cellRect.width / 2,
        y: cellRect.top - containerRect.top + cellRect.height / 2,
      };
    },
    [cellRefs, containerRef]
  );

  // Duration per segment – tuned so motion is readable without feeling sluggish.
  // Slightly faster than the previous values (~50% speed-up).
  const segmentDuration = animation.type === 'chain_capture' ? 700 : 800;

  // Animate through the path
  // NOTE: `position` is intentionally NOT in the dependency array.
  // Including it would cause infinite re-renders because setPosition() is called
  // in the animation loop, which would trigger effect cleanup and restart,
  // preventing the animation from ever completing.
  useEffect(() => {
    if (path.length < 2) {
      onComplete();
      return;
    }

    const fromPos = path[currentStep];
    const toPos = path[currentStep + 1];

    if (!fromPos || !toPos) {
      onComplete();
      return;
    }

    const fromCenter = getCellCenter(fromPos);
    const toCenter = getCellCenter(toPos);

    if (!fromCenter || !toCenter) {
      // Cell refs not ready, skip animation
      onComplete();
      return;
    }

    // Set initial position (only on first render when position is null)
    setPosition((prev) => prev ?? fromCenter);

    startTimeRef.current = null;

    const animate = (timestamp: number) => {
      if (startTimeRef.current === null) {
        startTimeRef.current = timestamp;
      }

      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / segmentDuration, 1);

      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);

      const currentX = fromCenter.x + (toCenter.x - fromCenter.x) * eased;
      const currentY = fromCenter.y + (toCenter.y - fromCenter.y) * eased;

      setPosition({ x: currentX, y: currentY });

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        // Move to next segment or complete
        if (currentStep + 2 < path.length) {
          setCurrentStep((prev) => prev + 1);
        } else {
          onComplete();
        }
      }
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [currentStep, path, getCellCenter, segmentDuration, onComplete]);
  // ^ position intentionally excluded - see comment above
  if (!position || path.length < 2) {
    return null;
  }

  const colors = getPlayerColors(animation.playerNumber);

  return (
    <div
      className="absolute pointer-events-none z-50"
      style={{
        left: position.x,
        top: position.y,
        transform: 'translate(-50%, -50%)',
      }}
    >
      {/* Animated piece representation */}
      <div className="flex flex-col items-center gap-[1px]">
        {Array.from({ length: Math.min(animation.stackHeight || 1, 3) }).map((_, i) => (
          <div
            key={i}
            className={`w-6 h-[4px] rounded-full border ${colors.ring} ${colors.ringBorder} shadow-lg`}
          />
        ))}
      </div>
      {/* Trail effect for chain captures */}
      {animation.type === 'chain_capture' && (
        <div
          className="absolute inset-0 rounded-full opacity-30"
          style={{
            background: `radial-gradient(circle, ${colors.ring.replace('bg-', '')} 0%, transparent 70%)`,
            width: '24px',
            height: '24px',
            transform: 'translate(-50%, -50%)',
          }}
        />
      )}
    </div>
  );
};

export const BoardView: React.FC<BoardViewProps> = ({
  boardType,
  board,
  viewModel,
  selectedPosition,
  validTargets = [],
  onCellClick,
  onCellDoubleClick,
  onCellContextMenu,
  showMovementGrid = false,
  showCoordinateLabels = false,
  isSpectator = false,
  showLineOverlays = false,
  showTerritoryRegionOverlays = false,
  pendingAnimation,
  onAnimationComplete,
  onShowKeyboardHelp,
  chainCapturePath,
  shakingCellKey,
  squareRankFromBottom = false,
}) => {
  // Animation state tracking
  const [animations, setAnimations] = useState<AnimationState[]>([]);
  const prevBoardRef = useRef<BoardState | null>(null);

  // Keyboard navigation state
  const [focusedPosition, setFocusedPosition] = useState<Position | null>(null);
  const boardContainerRef = useRef<HTMLDivElement>(null);
  // Geometry container for DOM-based movement grid alignment (the element
  // that the SVG overlay is absolutely positioned inside of).
  const boardGeometryRef = useRef<HTMLDivElement | null>(null);
  const cellRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

  // Touch gesture state - persists across renders for double-tap detection
  // Per RR-CANON-R076, this is a Host/Adapter Layer UX concern
  const touchStateRef = useRef<{
    lastTapKey: string | null;
    lastTapTime: number;
    longPressTimer: ReturnType<typeof setTimeout> | null;
    activeKey: string | null;
  }>({ lastTapKey: null, lastTapTime: 0, longPressTimer: null, activeKey: null });

  // Screen reader announcements
  const [announcement, setAnnouncement] = useState<string>('');

  // Precompute a lookup map for view-model cells when provided
  const cellByKey = useMemo(() => {
    if (!viewModel) return null;
    const map = new Map<string, CellViewModel>();
    for (const cell of viewModel.cells) {
      map.set(cell.positionKey, cell);
    }
    return map;
  }, [viewModel]);
  // Precompute a lookup map for decision highlights when provided. When
  // multiple highlights target the same cell, primary intensity wins.
  const highlightByKey = useMemo(() => {
    type HighlightMeta = { intensity: 'primary' | 'secondary'; groupIds?: string[] };

    const map = new Map<string, HighlightMeta>();
    if (viewModel?.decisionHighlights) {
      for (const h of viewModel.decisionHighlights.highlights) {
        const existing = map.get(h.positionKey);
        if (!existing) {
          const next: HighlightMeta = {
            intensity: h.intensity,
          };
          if (h.groupId) {
            next.groupIds = [h.groupId];
          }
          map.set(h.positionKey, next);
          continue;
        }

        const nextIntensity =
          existing.intensity === 'primary' || h.intensity === 'primary' ? 'primary' : 'secondary';

        const nextGroupIds = existing.groupIds ? [...existing.groupIds] : [];
        if (h.groupId && !nextGroupIds.includes(h.groupId)) {
          nextGroupIds.push(h.groupId);
        }

        const next: HighlightMeta = {
          intensity: nextIntensity,
        };
        if (nextGroupIds.length > 0) {
          next.groupIds = nextGroupIds;
        }

        map.set(h.positionKey, next);
      }
    }
    return map;
  }, [viewModel]);

  // Lightweight flags indicating the current decision highlight context. These
  // are used to tailor board-level animations (line bursts, pulsing targets,
  // etc.) without leaking PlayerChoice details into the view.
  const decisionChoiceKind = viewModel?.decisionHighlights?.choiceKind;
  const isLineDecisionContext =
    decisionChoiceKind === 'line_order' || decisionChoiceKind === 'line_reward';
  const isCaptureDirectionDecisionContext = decisionChoiceKind === 'capture_direction';
  const isRingEliminationDecisionContext = decisionChoiceKind === 'ring_elimination';
  const isTerritoryRegionDecisionContext = decisionChoiceKind === 'territory_region_order';
  const territoryRegionIdsInDisplayOrder =
    viewModel?.decisionHighlights?.territoryRegions?.regionIdsInDisplayOrder ?? [];

  // Chain capture accessibility helpers: track which cells are part of the
  // currently visualized chain path so we can expose richer aria-labels.
  const chainCaptureKeySet = useMemo(() => {
    if (!chainCapturePath || chainCapturePath.length === 0) {
      return null;
    }
    const set = new Set<string>();
    for (const pos of chainCapturePath) {
      set.add(positionToString(pos));
    }
    return set;
  }, [chainCapturePath]);

  const chainCaptureCurrentKey =
    chainCapturePath && chainCapturePath.length > 0
      ? positionToString(chainCapturePath[chainCapturePath.length - 1])
      : null;

  // Rules-lab overlays: lightweight lookup maps for lines and territory
  // regions so we can cheaply decorate cells without re-walking geometry.
  const lineOverlayByKey = useMemo(() => {
    const map = new Map<string, number>();
    if (!showLineOverlays || !board.formedLines || board.formedLines.length === 0) {
      return map;
    }

    for (const line of board.formedLines) {
      for (const pos of line.positions) {
        const key = positionToString(pos);
        if (!map.has(key)) {
          map.set(key, line.player);
        }
      }
    }

    return map;
  }, [showLineOverlays, board]);

  const territoryOverlayByKey = useMemo(() => {
    const map = new Map<string, { player: number; isDisconnected: boolean }>();
    if (!showTerritoryRegionOverlays || !board.territories || board.territories.size === 0) {
      return map;
    }

    board.territories.forEach((territory) => {
      const { controllingPlayer, isDisconnected, spaces } = territory;
      for (const pos of spaces) {
        const key = positionToString(pos);
        if (!map.has(key)) {
          map.set(key, { player: controllingPlayer, isDisconnected });
        }
      }
    });

    return map;
  }, [showTerritoryRegionOverlays, board]);

  const effectiveBoardType: BoardType = viewModel?.boardType ?? boardType;
  const effectiveSize = viewModel?.size ?? board.size;
  const notationOptions = useMemo<MoveNotationOptions>(
    () => ({
      boardType: effectiveBoardType,
      squareRankFromBottom: squareRankFromBottom ?? false,
    }),
    [effectiveBoardType, squareRankFromBottom]
  );

  // Build list of all valid positions for navigation
  const allPositions = useMemo((): FocusableCell[] => {
    const positions: FocusableCell[] = [];

    if (effectiveBoardType === 'square8' || effectiveBoardType === 'square19') {
      const size = effectiveBoardType === 'square8' ? 8 : 19;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const pos: Position = { x, y };
          positions.push({ position: pos, key: positionToString(pos) });
        }
      }
    } else if (effectiveBoardType === 'hexagonal') {
      const radius = effectiveSize - 1;
      for (let q = -radius; q <= radius; q++) {
        const r1 = Math.max(-radius, -q - radius);
        const r2 = Math.min(radius, -q + radius);
        for (let r = r1; r <= r2; r++) {
          const s = -q - r;
          const pos: Position = { x: q, y: r, z: s };
          positions.push({ position: pos, key: positionToString(pos) });
        }
      }
    }

    return positions;
  }, [effectiveBoardType, effectiveSize]);

  // Find neighboring position in a given direction
  const findNeighbor = useCallback(
    (current: Position, dx: number, dy: number): Position | null => {
      if (effectiveBoardType === 'square8' || effectiveBoardType === 'square19') {
        const size = effectiveBoardType === 'square8' ? 8 : 19;
        const newX = current.x + dx;
        const newY = current.y + dy;

        if (newX >= 0 && newX < size && newY >= 0 && newY < size) {
          return { x: newX, y: newY };
        }
        return null;
      }

      // Hex board navigation - map arrow keys to approximate hex directions
      // Using cube coordinates where q + r + s = 0
      if (effectiveBoardType === 'hexagonal') {
        const radius = effectiveSize - 1;
        let newQ = current.x;
        let newR = current.y;

        // Arrow key mappings for hex grid (approximate visual directions)
        if (dy === -1) {
          // Up: decrease q
          newQ = current.x - 1;
          newR = current.y;
        } else if (dy === 1) {
          // Down: increase q
          newQ = current.x + 1;
          newR = current.y;
        } else if (dx === -1) {
          // Left: decrease r (or shift diagonally)
          newR = current.y - 1;
        } else if (dx === 1) {
          // Right: increase r (or shift diagonally)
          newR = current.y + 1;
        }

        const newS = -newQ - newR;

        // Check if position is valid (within hex radius)
        if (Math.abs(newQ) <= radius && Math.abs(newR) <= radius && Math.abs(newS) <= radius) {
          return { x: newQ, y: newR, z: newS };
        }
        return null;
      }

      return null;
    },
    [effectiveBoardType, effectiveSize]
  );

  // Move focus in a direction
  const moveFocus = useCallback(
    (dx: number, dy: number) => {
      if (!focusedPosition) {
        // If no focus, start at first position
        if (allPositions.length > 0) {
          const firstPos = allPositions[0].position;
          setFocusedPosition(firstPos);
          const key = positionToString(firstPos);
          const cellRef = cellRefs.current.get(key);
          if (cellRef) {
            cellRef.focus();
          }
        }
        return;
      }

      const neighbor = findNeighbor(focusedPosition, dx, dy);
      if (neighbor) {
        setFocusedPosition(neighbor);
        const key = positionToString(neighbor);
        const cellRef = cellRefs.current.get(key);
        if (cellRef) {
          cellRef.focus();
        }
      }
    },
    [focusedPosition, findNeighbor, allPositions]
  );

  // Clear selection handler
  const clearSelection = useCallback(() => {
    setFocusedPosition(null);
    // Announce to screen readers
    setAnnouncement('Selection cleared');
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      // Only handle navigation if we're in the board area
      if (!boardContainerRef.current?.contains(e.target as Node)) {
        return;
      }

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          moveFocus(0, -1);
          break;
        case 'ArrowDown':
          e.preventDefault();
          moveFocus(0, 1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          moveFocus(-1, 0);
          break;
        case 'ArrowRight':
          e.preventDefault();
          moveFocus(1, 0);
          break;
        case 'Enter':
        case ' ':
          e.preventDefault();
          if (focusedPosition && onCellClick && !isSpectator) {
            onCellClick(focusedPosition);
            // Announce selection
            const stack = board.stacks.get(positionToString(focusedPosition));
            if (stack) {
              setAnnouncement(
                `Selected stack at position. Height ${stack.stackHeight}, controlled by player ${stack.controllingPlayer}`
              );
            } else {
              setAnnouncement('Selected empty cell');
            }
          }
          break;
        case 'Escape':
          e.preventDefault();
          clearSelection();
          break;
        case 'Home': {
          e.preventDefault();
          if (allPositions.length > 0) {
            const firstPos = allPositions[0].position;
            setFocusedPosition(firstPos);
            const key = positionToString(firstPos);
            const cellRef = cellRefs.current.get(key);
            cellRef?.focus();
          }
          break;
        }
        case 'End': {
          e.preventDefault();
          if (allPositions.length > 0) {
            const lastPos = allPositions[allPositions.length - 1].position;
            setFocusedPosition(lastPos);
            const key = positionToString(lastPos);
            const cellRef = cellRefs.current.get(key);
            cellRef?.focus();
          }
          break;
        }
        case '?':
          if (onShowKeyboardHelp) {
            e.preventDefault();
            onShowKeyboardHelp();
          }
          break;
        default:
          break;
      }
    },
    [
      focusedPosition,
      onCellClick,
      isSpectator,
      board.stacks,
      moveFocus,
      clearSelection,
      allPositions,
      onShowKeyboardHelp,
    ]
  );

  // Announce valid moves when selection changes
  useEffect(() => {
    if (selectedPosition && validTargets.length > 0) {
      setAnnouncement(
        `Piece selected. ${validTargets.length} valid move${validTargets.length === 1 ? '' : 's'} available`
      );
    }
  }, [selectedPosition, validTargets.length]);

  // Register cell ref for keyboard navigation
  const registerCellRef = useCallback((key: string, ref: HTMLButtonElement | null) => {
    if (ref) {
      cellRefs.current.set(key, ref);
    } else {
      cellRefs.current.delete(key);
    }
  }, []);

  // Helper to get animation class for a position
  const getAnimationClass = useCallback(
    (posKey: string): string | undefined => {
      const anim = animations.find((a) => a.position === posKey);
      if (!anim) return undefined;

      switch (anim.type) {
        case 'place':
          return 'animate-piece-appear';
        case 'move':
          return 'animate-piece-move';
        case 'capture':
          return 'animate-capture-bounce';
        default:
          return undefined;
      }
    },
    [animations]
  );

  // Detect board changes and trigger animations
  useEffect(() => {
    const prevBoard = prevBoardRef.current;

    if (!prevBoard) {
      // First render - no animations needed
      prevBoardRef.current = board;
      return undefined;
    }

    const newAnimations: AnimationState[] = [];

    // Detect new stacks (pieces placed or moved to new positions)
    board.stacks.forEach((currentStack, posKey) => {
      const prevStack = prevBoard.stacks.get(posKey);

      if (!prevStack) {
        // New stack at this position - could be a placement or move destination
        // Check if this stack existed elsewhere before (would indicate a move)
        let foundSource = false;
        prevBoard.stacks.forEach((_, prevPosKey) => {
          if (!board.stacks.has(prevPosKey)) {
            // A stack disappeared from another position - this is likely a move
            foundSource = true;
          }
        });

        if (foundSource) {
          // This is likely the destination of a move
          newAnimations.push({ position: posKey, type: 'move' });
        } else {
          // This is a new placement
          newAnimations.push({ position: posKey, type: 'place' });
        }
      } else if (currentStack.stackHeight !== prevStack.stackHeight) {
        // Stack height changed - could be a capture or stacking
        if (currentStack.stackHeight > prevStack.stackHeight) {
          // Stack grew - likely a capture or rings stacking
          newAnimations.push({ position: posKey, type: 'capture' });
        }
      }
    });

    // Update animations state
    if (newAnimations.length > 0) {
      debugLog(
        isSandboxAnimationDebugEnabled(),
        '[SandboxAnimationDebug] BoardView: board-diff animations',
        {
          boardType: board.type,
          positions: newAnimations.map((a) => ({ key: a.position, type: a.type })),
        }
      );

      setAnimations(newAnimations);

      // Clear animations after they complete (use the longest duration: 400ms for capture)
      const timeoutId = setTimeout(() => {
        setAnimations([]);
      }, 450);

      // Store timeout for cleanup
      return () => clearTimeout(timeoutId);
    }

    // Update ref with current board state
    prevBoardRef.current = board;
    return undefined;
  }, [board]);

  // Update the ref after animations are processed
  useEffect(() => {
    prevBoardRef.current = board;
  }, [board]);

  // Square boards: simple grid using (x, y) coordinates.
  // Hex board: rendered using the same cube/axial coordinate system
  // as BoardManager (q = x, r = y, s = z, with q + r + s = 0).

  // DOM-aware movement grid: build grid entirely from DOM cell positions.
  // This ensures perfect alignment regardless of CSS changes.
  // Store both the grid AND the dimensions used to compute it, so we can
  // render with exactly the same coordinate system.
  // Also store the scale factor for containers with CSS transforms.
  const [movementGridData, setMovementGridData] = useState<{
    grid: MovementGrid;
    width: number;
    height: number;
    scaleFactor: number;
  } | null>(null);

  useLayoutEffect(() => {
    if (!showMovementGrid) {
      setMovementGridData(null);
      return;
    }

    const container = boardGeometryRef.current;
    if (!container) {
      // Fallback to logical grid if no DOM container - use 1x1 viewBox
      const fallbackGrid = computeBoardMovementGrid(board);
      setMovementGridData({ grid: fallbackGrid, width: 1, height: 1, scaleFactor: 1 });
      return;
    }

    // Detect CSS scale transform on the container
    // getBoundingClientRect() returns scaled dimensions, but we need unscaled
    // dimensions for the SVG to render correctly inside the scaled container.
    const computedStyle = window.getComputedStyle(container);
    const transform = computedStyle.transform;
    let scaleFactor = 1;
    if (transform && transform !== 'none') {
      // Try matrix format first: matrix(a, b, c, d, tx, ty) where scaleX is 'a'
      const matrixMatch = transform.match(/matrix\(([^,]+),/);
      if (matrixMatch) {
        scaleFactor = parseFloat(matrixMatch[1]) || 1;
      } else {
        // Try scale format: scale(x) or scale(x, y)
        const scaleMatch = transform.match(/scale\(([^,)]+)/);
        if (scaleMatch) {
          scaleFactor = parseFloat(scaleMatch[1]) || 1;
        }
      }
    }

    const containerRect = container.getBoundingClientRect();
    // getBoundingClientRect returns scaled dimensions - unscale them for internal coords
    const width = containerRect.width / scaleFactor;
    const height = containerRect.height / scaleFactor;

    if (!width || !height) {
      const fallbackGrid = computeBoardMovementGrid(board);
      setMovementGridData({ grid: fallbackGrid, width: 1, height: 1, scaleFactor: 1 });
      return;
    }

    // Build centers entirely from DOM - no dependency on computeBoardMovementGrid
    const centers: MovementGrid['centers'] = [];
    const centersByKey = new Map<string, { cx: number; cy: number; position: Position }>();

    const cellElements = container.querySelectorAll<HTMLButtonElement>('button[data-x]');
    cellElements.forEach((el) => {
      const xAttr = el.getAttribute('data-x');
      const yAttr = el.getAttribute('data-y');
      if (xAttr == null || yAttr == null) return;

      const zAttr = el.getAttribute('data-z');
      const position: Position =
        zAttr != null
          ? { x: Number(xAttr), y: Number(yAttr), z: Number(zAttr) }
          : { x: Number(xAttr), y: Number(yAttr) };

      const key = positionToString(position);
      const rect = el.getBoundingClientRect();
      // Unscale the positions from screen coords to pre-transform coords
      const centerX = (rect.left + rect.width / 2 - containerRect.left) / scaleFactor;
      const centerY = (rect.top + rect.height / 2 - containerRect.top) / scaleFactor;

      const cx = centerX / width;
      const cy = centerY / height;

      centersByKey.set(key, { cx, cy, position });
      centers.push({ key, position, cx, cy });
    });

    if (centers.length === 0) {
      const fallbackGrid = computeBoardMovementGrid(board);
      setMovementGridData({ grid: fallbackGrid, width: 1, height: 1, scaleFactor: 1 });
      return;
    }

    // Build edges based on board type adjacency rules
    const edges: MovementGrid['edges'] = [];
    const addedEdges = new Set<string>();

    const isHex = effectiveBoardType === 'hexagonal';

    if (isHex) {
      // Hex adjacency: 6 neighbors using cube coordinates
      const hexDirections = [
        { dx: 1, dy: -1, dz: 0 },
        { dx: 1, dy: 0, dz: -1 },
        { dx: 0, dy: 1, dz: -1 },
        { dx: -1, dy: 1, dz: 0 },
        { dx: -1, dy: 0, dz: 1 },
        { dx: 0, dy: -1, dz: 1 },
      ];

      for (const center of centers) {
        const { position } = center;
        const q = position.x;
        const r = position.y;
        const s = position.z ?? -q - r;

        for (const { dx, dy, dz } of hexDirections) {
          const nq = q + dx;
          const nr = r + dy;
          const ns = s + dz;
          if (nq + nr + ns !== 0) continue;

          const neighborKey = positionToString({ x: nq, y: nr, z: ns });
          if (!centersByKey.has(neighborKey)) continue;

          // Only add each edge once (undirected)
          const edgeKey =
            center.key < neighborKey
              ? `${center.key}->${neighborKey}`
              : `${neighborKey}->${center.key}`;
          if (addedEdges.has(edgeKey)) continue;
          addedEdges.add(edgeKey);

          if (center.key < neighborKey) {
            edges.push({ fromKey: center.key, toKey: neighborKey });
          } else {
            edges.push({ fromKey: neighborKey, toKey: center.key });
          }
        }
      }
    } else {
      // Square board adjacency: 8 neighbors (orthogonal + diagonal)
      const squareDeltas = [
        { dx: 1, dy: 0 },
        { dx: -1, dy: 0 },
        { dx: 0, dy: 1 },
        { dx: 0, dy: -1 },
        { dx: 1, dy: 1 },
        { dx: 1, dy: -1 },
        { dx: -1, dy: 1 },
        { dx: -1, dy: -1 },
      ];

      for (const center of centers) {
        const { position } = center;
        for (const { dx, dy } of squareDeltas) {
          const nx = position.x + dx;
          const ny = position.y + dy;
          const neighborKey = positionToString({ x: nx, y: ny });
          if (!centersByKey.has(neighborKey)) continue;

          const edgeKey =
            center.key < neighborKey
              ? `${center.key}->${neighborKey}`
              : `${neighborKey}->${center.key}`;
          if (addedEdges.has(edgeKey)) continue;
          addedEdges.add(edgeKey);

          if (center.key < neighborKey) {
            edges.push({ fromKey: center.key, toKey: neighborKey });
          } else {
            edges.push({ fromKey: neighborKey, toKey: center.key });
          }
        }
      }
    }

    // Store grid with the exact dimensions used to compute normalized coords.
    // NOTE: We intentionally avoid depending on the full BoardState here
    // because hosts like the sandbox clone the board on every render; using
    // board identity as a dependency would cause this layout effect to fire
    // after each commit, leading to an infinite update loop. Geometry only
    // depends on board type and logical size.
    setMovementGridData({ grid: { centers, edges }, width, height, scaleFactor });
  }, [showMovementGrid, effectiveBoardType, board.size]);

  // Extract grid and stored dimensions for rendering
  const movementGrid = movementGridData?.grid ?? null;
  const storedWidth = movementGridData?.width ?? 0;
  const storedHeight = movementGridData?.height ?? 0;

  const centersByKey = movementGrid
    ? new Map(movementGrid.centers.map((center) => [center.key, center] as const))
    : null;

  const renderMovementOverlay = () => {
    if (!movementGrid || !centersByKey || movementGrid.edges.length === 0) {
      return null;
    }

    // Use the stored dimensions from when grid was computed
    // This ensures perfect alignment since normalized coords were computed
    // against these exact dimensions
    const width = storedWidth;
    const height = storedHeight;

    if (!width || !height) return null;

    // Scale factors for stroke/circle sizes based on container
    const scale = Math.min(width, height);
    const strokeWidth = Math.max(1, scale * 0.003);
    const dotRadius = Math.max(2, scale * 0.006);

    return (
      <svg
        className="pointer-events-none absolute inset-0"
        style={{
          width: '100%',
          height: '100%',
        }}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid slice"
      >
        {movementGrid.edges.map((edge) => {
          const from = centersByKey.get(edge.fromKey);
          const to = centersByKey.get(edge.toKey);
          if (!from || !to) return null;

          // Convert normalized coords back to pixels
          const x1 = from.cx * width;
          const y1 = from.cy * height;
          const x2 = to.cx * width;
          const y2 = to.cy * height;

          return (
            <line
              key={`${edge.fromKey}->${edge.toKey}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="rgba(148, 163, 184, 0.45)"
              strokeWidth={strokeWidth}
            />
          );
        })}

        {movementGrid.centers.map((center) => (
          <circle
            key={center.key}
            cx={center.cx * width}
            cy={center.cy * height}
            r={dotRadius}
            fill="rgba(148, 163, 184, 0.75)"
          />
        ))}
      </svg>
    );
  };

  /**
   * Renders an SVG overlay showing the chain capture path with arrows.
   * Arrows connect each position in the sequence, and the current position
   * (last in path) is highlighted with a pulsing ring.
   */
  const renderChainCapturePathOverlay = () => {
    if (!chainCapturePath || chainCapturePath.length < 2) {
      return null;
    }

    const container = boardGeometryRef.current;
    if (!container) return null;

    const containerRect = container.getBoundingClientRect();
    const width = containerRect.width;
    const height = containerRect.height;

    if (!width || !height) return null;

    // Build position to pixel center map from DOM
    const cellPositions = new Map<string, { cx: number; cy: number }>();
    const cellElements = container.querySelectorAll<HTMLButtonElement>('button[data-x]');
    cellElements.forEach((el) => {
      const xAttr = el.getAttribute('data-x');
      const yAttr = el.getAttribute('data-y');
      if (xAttr == null || yAttr == null) return;

      const zAttr = el.getAttribute('data-z');
      const position: Position =
        zAttr != null
          ? { x: Number(xAttr), y: Number(yAttr), z: Number(zAttr) }
          : { x: Number(xAttr), y: Number(yAttr) };

      const key = positionToString(position);
      const rect = el.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2 - containerRect.left;
      const centerY = rect.top + rect.height / 2 - containerRect.top;

      cellPositions.set(key, { cx: centerX, cy: centerY });
    });

    // Scale factors for visual elements
    const scale = Math.min(width, height);
    const strokeWidth = Math.max(2, scale * 0.006);
    const arrowSize = Math.max(6, scale * 0.015);
    const currentPosRadius = Math.max(8, scale * 0.02);

    // Generate path segments
    const pathSegments: Array<{
      from: { cx: number; cy: number };
      to: { cx: number; cy: number };
      key: string;
    }> = [];

    for (let i = 0; i < chainCapturePath.length - 1; i++) {
      const fromPos = chainCapturePath[i];
      const toPos = chainCapturePath[i + 1];
      const fromKey = positionToString(fromPos);
      const toKey = positionToString(toPos);
      const fromCenter = cellPositions.get(fromKey);
      const toCenter = cellPositions.get(toKey);

      if (fromCenter && toCenter) {
        pathSegments.push({
          from: fromCenter,
          to: toCenter,
          key: `${fromKey}->${toKey}`,
        });
      }
    }

    // Current position (end of chain)
    const currentPos = chainCapturePath[chainCapturePath.length - 1];
    const currentKey = positionToString(currentPos);
    const currentCenter = cellPositions.get(currentKey);

    return (
      <svg
        className="pointer-events-none absolute inset-0"
        style={{
          width: '100%',
          height: '100%',
        }}
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid slice"
      >
        {/* Arrow marker definition */}
        <defs>
          <marker
            id="chain-capture-arrow"
            markerWidth={arrowSize}
            markerHeight={arrowSize}
            refX={arrowSize - 1}
            refY={arrowSize / 2}
            orient="auto"
            markerUnits="userSpaceOnUse"
          >
            <polygon
              points={`0,0 ${arrowSize},${arrowSize / 2} 0,${arrowSize}`}
              fill="rgba(251, 146, 60, 0.9)"
            />
          </marker>
        </defs>

        {/* Path segments as arrows */}
        {pathSegments.map((segment, index) => {
          // Shorten the line slightly so arrow doesn't overlap with target
          const dx = segment.to.cx - segment.from.cx;
          const dy = segment.to.cy - segment.from.cy;
          const len = Math.sqrt(dx * dx + dy * dy);
          const shortenBy = arrowSize * 0.8;
          const adjustedTo = {
            cx: segment.to.cx - (dx / len) * shortenBy,
            cy: segment.to.cy - (dy / len) * shortenBy,
          };

          // Fade earlier segments slightly
          const opacity = 0.6 + (index / pathSegments.length) * 0.4;

          return (
            <line
              key={segment.key}
              x1={segment.from.cx}
              y1={segment.from.cy}
              x2={adjustedTo.cx}
              y2={adjustedTo.cy}
              stroke={`rgba(251, 146, 60, ${opacity})`}
              strokeWidth={strokeWidth}
              strokeLinecap="round"
              markerEnd="url(#chain-capture-arrow)"
            />
          );
        })}

        {/* Highlight circles at each visited position */}
        {chainCapturePath.slice(0, -1).map((pos, index) => {
          const key = positionToString(pos);
          const center = cellPositions.get(key);
          if (!center) return null;

          const opacity = 0.3 + (index / chainCapturePath.length) * 0.3;
          return (
            <circle
              key={`visited-${key}`}
              cx={center.cx}
              cy={center.cy}
              r={currentPosRadius * 0.7}
              fill="none"
              stroke={`rgba(251, 146, 60, ${opacity})`}
              strokeWidth={strokeWidth * 0.8}
            />
          );
        })}

        {/* Current position - pulsing highlight (inner ring pulses, outer ring pings once) */}
        {currentCenter && (
          <>
            <circle
              cx={currentCenter.cx}
              cy={currentCenter.cy}
              r={currentPosRadius}
              fill="none"
              stroke="rgba(251, 146, 60, 0.9)"
              strokeWidth={strokeWidth * 1.2}
              className="animate-pulse"
            />
            <circle
              cx={currentCenter.cx}
              cy={currentCenter.cy}
              r={currentPosRadius * 1.5}
              fill="none"
              stroke="rgba(251, 146, 60, 0.4)"
              strokeWidth={strokeWidth * 0.6}
              className="animate-ping"
              style={{ animationDuration: '1.5s', animationIterationCount: 1 }}
            />
          </>
        )}
      </svg>
    );
  };

  // Square boards: simple grid using (x, y) coordinates.
  // Hex board: rendered using the same cube/axial coordinate system
  // as BoardManager (q = x, r = y, s = z, with q + r + s = 0).

  const renderSquareCoordinateLabels = (size: number) => {
    const skipI = effectiveBoardType === 'square19';
    const files = generateFileLabels(size, skipI);
    const ranks = generateRankLabels(size, squareRankFromBottom ?? false);
    const topOffset = effectiveBoardType === 'square19' ? 28 : 22;
    const sideOffset = effectiveBoardType === 'square19' ? 28 : 24;
    const labelClass =
      'pointer-events-none absolute text-[10px] md:text-[11px] font-semibold tracking-wide text-slate-400 uppercase';

    return (
      <>
        <div className="pointer-events-none absolute left-0 right-0" style={{ top: -topOffset }}>
          {files.map((file, idx) => (
            <span
              key={`file-top-${file}`}
              className={labelClass}
              style={{
                left: `${((idx + 0.5) / size) * 100}%`,
                transform: 'translateX(-50%)',
              }}
            >
              {file}
            </span>
          ))}
        </div>
        <div className="pointer-events-none absolute left-0 right-0" style={{ bottom: -topOffset }}>
          {files.map((file, idx) => (
            <span
              key={`file-bottom-${file}`}
              className={labelClass}
              style={{
                left: `${((idx + 0.5) / size) * 100}%`,
                transform: 'translateX(-50%)',
              }}
            >
              {file}
            </span>
          ))}
        </div>
        <div className="pointer-events-none absolute top-0 bottom-0" style={{ left: -sideOffset }}>
          {ranks.map((rank, idx) => (
            <span
              key={`rank-left-${rank}`}
              className={labelClass}
              style={{
                top: `${((idx + 0.5) / size) * 100}%`,
                transform: 'translateY(-50%)',
              }}
            >
              {rank}
            </span>
          ))}
        </div>
        <div className="pointer-events-none absolute top-0 bottom-0" style={{ right: -sideOffset }}>
          {ranks.map((rank, idx) => (
            <span
              key={`rank-right-${rank}`}
              className={labelClass}
              style={{
                top: `${((idx + 0.5) / size) * 100}%`,
                transform: 'translateY(-50%)',
              }}
            >
              {rank}
            </span>
          ))}
        </div>
      </>
    );
  };

  const renderSquareBoard = (size: number) => {
    const rows: React.ReactNode[] = [];
    // Cell sizing: mobile-first responsive sizing for 375px-768px viewports.
    // Square8: 44px cells fit 8 columns in 375px (8×44 + 7×2 gaps = 366px)
    // Square19: 44px minimum for touch targets; uses horizontal scroll on mobile.
    // Touch target minimum: 44px (WCAG 2.1 AAA recommendation)
    const squareCellSizeClasses =
      boardType === 'square8'
        ? 'w-11 h-11 sm:w-14 sm:h-14 md:w-20 md:h-20' // 44px → 56px → 80px
        : 'w-11 h-11 md:w-14 md:h-14'; // 44px minimum for touch, 56px on desktop

    for (let y = 0; y < size; y++) {
      const cells: React.ReactNode[] = [];
      for (let x = 0; x < size; x++) {
        const pos: Position = { x, y };
        const key = positionToString(pos);

        const cellVM = cellByKey?.get(key);

        const stack = board.stacks.get(key);
        const marker = board.markers.get(key);
        const collapsedOwnerFromBoard = board.collapsedSpaces.get(key);

        const effectiveIsSelected =
          cellVM?.isSelected ?? !!(selectedPosition && positionsEqual(selectedPosition, pos));
        const effectiveIsValid =
          cellVM?.isValidTarget ?? validTargets.some((p) => positionsEqual(p, pos));

        // Determine owner of stack for selection pulse color
        const stackOwner = stack?.rings?.[0];

        const isDarkSquare =
          cellVM?.isDarkSquare !== undefined ? cellVM.isDarkSquare : (x + y) % 2 === 0;
        const baseSquareBg = isDarkSquare ? 'bg-slate-300' : 'bg-slate-100';

        const collapsedOwner =
          cellVM?.collapsedSpace?.ownerPlayerNumber ||
          (collapsedOwnerFromBoard !== undefined ? collapsedOwnerFromBoard : undefined);
        const territoryClasses = collapsedOwner ? getPlayerColors(collapsedOwner).territory : '';

        // Rules-lab overlays: detected lines and territory/disconnected regions.
        // These are visual debugging aids only and do not affect rules.
        const lineOverlayPlayer = showLineOverlays ? lineOverlayByKey.get(key) : undefined;
        const territoryOverlayInfo = showTerritoryRegionOverlays
          ? territoryOverlayByKey.get(key)
          : undefined;

        // Decision-phase highlight metadata for this cell, if any. Primary highlights
        // are rendered more prominently than secondary ones, but both should coexist
        // cleanly with selection and valid-move styling.
        const highlightMeta = highlightByKey.get(key);
        const decisionHighlight = highlightMeta?.intensity;
        const territoryRegionGroupIds =
          isTerritoryRegionDecisionContext && highlightMeta?.groupIds ? highlightMeta.groupIds : [];
        const hasStackForPulse = !!(cellVM?.stack || stack);
        // Destination pulse for recent moves: derive from the internal
        // board-diff animation state so we always highlight the actual
        // landing stack, independent of move-history timing.
        const isMoveDestination =
          hasStackForPulse &&
          animations.some(
            (anim) =>
              anim.position === key &&
              (anim.type === 'move' || anim.type === 'capture' || anim.type === 'chain_capture')
          );

        if (isMoveDestination) {
          debugLog(
            isSandboxAnimationDebugEnabled(),
            '[SandboxAnimationDebug] BoardView: destination pulse (square)',
            {
              key,
              boardType,
              hasStack: !!stack,
              hasCellVMStack: !!cellVM?.stack,
            }
          );
        }
        let decisionHighlightClass = '';
        if (decisionHighlight === 'primary') {
          if (isLineDecisionContext) {
            decisionHighlightClass = 'decision-highlight-primary line-formation-burst';
          } else if (!isRingEliminationDecisionContext) {
            // For elimination decisions, rely on the dedicated amber
            // pulse/halo classes so they remain visually distinct from
            // generic cyan decision highlights.
            decisionHighlightClass = 'decision-highlight-primary';
          }
        } else if (decisionHighlight === 'secondary') {
          decisionHighlightClass = 'decision-highlight-secondary';
        }

        // Invalid move shake animation: apply when this cell is the target of an invalid move attempt
        const isShaking = shakingCellKey === key;

        const shouldPulseCaptureLanding =
          decisionHighlight === 'primary' && isCaptureDirectionDecisionContext;
        const shouldPulseCaptureTarget =
          decisionHighlight === 'secondary' && isCaptureDirectionDecisionContext;
        const shouldPulseEliminationTarget =
          decisionHighlight === 'primary' && isRingEliminationDecisionContext;
        const shouldPulseTerritoryRegion =
          decisionHighlight === 'primary' && isTerritoryRegionDecisionContext;

        // Track palette index for ARIA descriptions; -1 means "not assigned".
        let territoryRegionPaletteIndex = -1;
        const territoryRegionClasses: string[] = [];
        if (shouldPulseTerritoryRegion && territoryRegionGroupIds.length > 0) {
          if (territoryRegionGroupIds.length > 1) {
            // Any multi-region overlap gets a dedicated, more intense visual.
            territoryRegionClasses.push('territory-region-overlap');
          } else {
            const regionId = territoryRegionGroupIds[0];
            const paletteIndex = Math.max(0, territoryRegionIdsInDisplayOrder.indexOf(regionId));
            territoryRegionPaletteIndex = paletteIndex;
            switch (paletteIndex) {
              case 0:
                territoryRegionClasses.push('territory-region-a');
                break;
              case 1:
                territoryRegionClasses.push('territory-region-b');
                break;
              case 2:
                territoryRegionClasses.push('territory-region-c');
                break;
              default:
                territoryRegionClasses.push('territory-region-d');
                break;
            }
          }
        }

        const cellClasses = [
          'relative border flex items-center justify-center text-[11px] md:text-xs rounded-sm',
          squareCellSizeClasses,
          'border-slate-600 text-slate-900',
          territoryClasses || baseSquareBg,
          decisionHighlightClass,
          isMoveDestination ? 'move-destination-pulse' : '',
          shouldPulseCaptureLanding ? 'decision-pulse-capture' : '',
          shouldPulseCaptureTarget ? 'capture-target-pulse' : '',
          shouldPulseEliminationTarget ? 'decision-pulse-elimination' : '',
          shouldPulseTerritoryRegion ? 'decision-pulse-territory' : '',
          ...territoryRegionClasses,
          effectiveIsSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          // Valid target highlighting on square boards: thin, bright-green inset
          // ring plus a deeper emerald tint that reads clearly even over the
          // dark board container background. Also apply noticeable pulse animation.
          effectiveIsValid
            ? 'outline outline-[2px] outline-emerald-400/95 outline-offset-[-4px] bg-emerald-100/90 valid-move-cell valid-move-cell-square'
            : '',
          // Invalid move shake animation
          isShaking ? 'invalid-move-shake' : '',
        ]
          .filter(Boolean)
          .join(' ');

        const hasMarkerVM = !!cellVM?.marker;
        // If a stack is present, suppress raw marker rendering from the
        // board state to avoid visual stack+marker overlap. In that case,
        // rely on the view model (if any) to decide what to show.
        const hasMarkerBoard = !stack && marker && marker.type === 'regular';
        const hasMarker = hasMarkerVM || hasMarkerBoard;
        const markerColorClass =
          cellVM?.marker?.colorClass ??
          (hasMarkerBoard && marker ? getPlayerColors(marker.player).marker : null);

        // Mobile-responsive marker sizing for W3-12 (44px mobile cells)
        const markerOuterSizeClasses =
          boardType === 'square8' ? 'w-5 h-5 sm:w-6 sm:h-6 md:w-7 md:h-7' : 'w-5 h-5 md:w-6 md:h-6';

        // Check if this cell is keyboard-focused (different from selected)
        const isFocused = focusedPosition && positionsEqual(focusedPosition, pos);
        const focusClasses =
          isFocused && !effectiveIsSelected
            ? 'ring-2 ring-amber-400 ring-offset-1 ring-offset-slate-950'
            : '';

        // Generate accessible label for the cell. We enrich the generic
        // "valid move" wording with more specific descriptions for capture
        // landings/targets, territory-region decisions, and chain-capture
        // path cells so screen readers surface the same intent as visuals.
        const stackInfo = cellVM?.stack
          ? `Stack height ${cellVM.stack.stackHeight}, cap ${cellVM.stack.capHeight}, player ${cellVM.stack.controllingPlayer}`
          : stack
            ? `Stack height ${stack.stackHeight}, cap ${stack.capHeight}, player ${stack.controllingPlayer}`
            : 'Empty cell';
        const accessibilityAnnotations: string[] = [];

        if (shouldPulseCaptureLanding) {
          accessibilityAnnotations.push('Capture landing, valid move target');
        } else if (shouldPulseCaptureTarget) {
          accessibilityAnnotations.push('Capture target stack');
        }

        if (shouldPulseEliminationTarget) {
          accessibilityAnnotations.push('Ring elimination candidate');
        }

        if (shouldPulseTerritoryRegion) {
          if (territoryRegionGroupIds.length > 1) {
            accessibilityAnnotations.push('Overlapping territory regions for this choice');
          } else if (territoryRegionPaletteIndex >= 0) {
            const humanIndex = territoryRegionPaletteIndex + 1;
            accessibilityAnnotations.push(`Territory region option ${humanIndex}`);
          } else {
            accessibilityAnnotations.push('Territory region option');
          }
        }

        if (chainCaptureKeySet && chainCaptureKeySet.has(key)) {
          if (chainCaptureCurrentKey && chainCaptureCurrentKey === key) {
            accessibilityAnnotations.push('Current capture position in chain');
          } else {
            accessibilityAnnotations.push('Visited position in capture chain');
          }
        }

        if (effectiveIsValid && accessibilityAnnotations.length === 0) {
          accessibilityAnnotations.push('Valid move target');
        }

        const accessibilitySuffix =
          accessibilityAnnotations.length > 0 ? `. ${accessibilityAnnotations.join('. ')}` : '';

        const coordLabel = formatPosition(pos, notationOptions);
        const cellLabel = `Cell ${coordLabel}. ${stackInfo}${accessibilitySuffix}`;

        cells.push(
          <button
            key={key}
            ref={(ref) => registerCellRef(key, ref)}
            type="button"
            data-x={x}
            data-y={y}
            data-decision-highlight={decisionHighlight || undefined}
            data-line-overlay={lineOverlayPlayer !== undefined ? 'true' : undefined}
            data-line-overlay-player={
              lineOverlayPlayer !== undefined ? String(lineOverlayPlayer) : undefined
            }
            data-region-overlay={territoryOverlayInfo ? 'true' : undefined}
            data-region-overlay-player={
              territoryOverlayInfo ? String(territoryOverlayInfo.player) : undefined
            }
            data-region-overlay-disconnected={
              territoryOverlayInfo?.isDisconnected ? 'true' : undefined
            }
            onClick={() => !isSpectator && onCellClick?.(pos)}
            onDoubleClick={() => !isSpectator && onCellDoubleClick?.(pos)}
            onContextMenu={(e) => {
              e.preventDefault();
              !isSpectator && onCellContextMenu?.(pos);
            }}
            onFocus={() => setFocusedPosition(pos)}
            onTouchStart={(e) => {
              if (isSpectator || e.touches.length !== 1) return;
              const ts = touchStateRef.current;
              if (ts.longPressTimer) clearTimeout(ts.longPressTimer);
              ts.activeKey = key;
              ts.longPressTimer = setTimeout(() => {
                if (ts.activeKey === key) {
                  ts.activeKey = null;
                  onCellContextMenu?.(pos);
                }
              }, LONG_PRESS_DELAY_MS);
            }}
            onTouchEnd={(e) => {
              if (isSpectator) return;
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              if (ts.activeKey !== key) return;
              ts.activeKey = null;
              const now = Date.now();
              if (ts.lastTapKey === key && now - ts.lastTapTime < DOUBLE_TAP_DELAY_MS) {
                e.preventDefault();
                ts.lastTapKey = null;
                ts.lastTapTime = 0;
                onCellDoubleClick?.(pos);
              } else {
                ts.lastTapKey = key;
                ts.lastTapTime = now;
              }
            }}
            onTouchMove={() => {
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              ts.activeKey = null;
            }}
            onTouchCancel={() => {
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              ts.activeKey = null;
            }}
            className={`${cellClasses} ${focusClasses} ${isSpectator ? 'cursor-default' : 'cursor-pointer'}`}
            disabled={isSpectator}
            tabIndex={isFocused ? 0 : -1}
            role="gridcell"
            aria-label={cellLabel}
            aria-selected={effectiveIsSelected || undefined}
          >
            {cellVM?.stack ? (
              <StackFromViewModel
                stack={cellVM.stack}
                boardType={boardType}
                animationClass={`${getAnimationClass(key) ?? ''} ${
                  shouldPulseEliminationTarget ? 'decision-elimination-stack-pulse' : ''
                }`.trim()}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
                isJustMoved={isMoveDestination}
              />
            ) : stack ? (
              <StackWidget
                stack={stack}
                boardType={boardType}
                animationClass={`${getAnimationClass(key) ?? ''} ${
                  shouldPulseEliminationTarget ? 'decision-elimination-stack-pulse' : ''
                }`.trim()}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
                isJustMoved={isMoveDestination}
              />
            ) : null}

            {hasMarker && markerColorClass && (
              <div
                className={`absolute inset-0 m-auto rounded-full border-[6px] ${markerColorClass} bg-transparent shadow-sm shadow-slate-900/70 z-10 flex items-center justify-center ${markerOuterSizeClasses}`}
              >
                <div className="w-2/3 h-2/3 rounded-full bg-slate-950/75 flex items-center justify-center">
                  <div className="w-1/3 h-1/3 rounded-full bg-slate-950/95 border border-slate-900/95" />
                </div>
              </div>
            )}

            {collapsedOwner && (
              <div
                className={`pointer-events-none absolute inset-[2px] rounded-sm border-2 ${getPlayerColors(collapsedOwner).marker} border-opacity-90`}
              />
            )}
          </button>
        );
      }
      rows.push(
        <div key={y} className="flex gap-[2px]">
          {cells}
        </div>
      );
    }
    const containerClasses =
      boardType === 'square8'
        ? 'relative space-y-1 bg-slate-800/60 p-2 rounded-md border border-slate-700 shadow-inner inline-block'
        : 'relative space-y-0.5 bg-slate-800/60 p-2 rounded-md border border-slate-700 shadow-inner inline-block scale-75 origin-top-left';

    return (
      <div
        ref={boardGeometryRef}
        className={containerClasses}
        role="grid"
        aria-label={`${boardType === 'square8' ? '8x8' : '19x19'} game board`}
      >
        {rows}
        {renderMovementOverlay()}
        {renderChainCapturePathOverlay()}
        {showCoordinateLabels ? renderSquareCoordinateLabels(size) : null}
        {/* Move animation layer */}
        {pendingAnimation && (
          <MoveAnimationLayer
            animation={pendingAnimation}
            cellRefs={cellRefs}
            containerRef={boardGeometryRef}
            onComplete={() => onAnimationComplete?.()}
          />
        )}
      </div>
    );
  };

  const renderHexBoard = () => {
    // True hex layout matching BoardManager.generateValidPositions.
    // For radius R = size - 1, valid cube coords (q, r, s) satisfy
    //   -R <= q <= R
    //   max(-R, -q-R) <= r <= min(R, -q+R)
    //   s = -q - r
    // We render each q as a row and then wrap all rows in a flex column
    // with slight negative spacing to reduce vertical gaps.

    const radius = board.size - 1; // e.g. size=13 => radius=12
    const rows: React.ReactNode[] = [];

    // Helper to generate algebraic labels for hex cells
    // Matches logic in notation.ts:
    //   Rank = radius - q + 1
    //   File = 'a' + (r + radius)
    const getHexLabel = (q: number, r: number) => {
      if (!showCoordinateLabels) return null;
      const rankNum = radius - q + 1;
      const fileCode = 'a'.charCodeAt(0) + (r + radius);
      const file = String.fromCharCode(fileCode);
      return `${file}${rankNum}`;
    };

    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);

      const cells: React.ReactNode[] = [];
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        const pos: Position = { x: q, y: r, z: s };
        const key = positionToString(pos);

        const cellVM = cellByKey?.get(key);

        const stack = board.stacks.get(key);
        const marker = board.markers.get(key);
        const collapsedOwnerFromBoard = board.collapsedSpaces.get(key);

        const effectiveIsSelected =
          cellVM?.isSelected ?? !!(selectedPosition && positionsEqual(selectedPosition, pos));
        const effectiveIsValid =
          cellVM?.isValidTarget ?? validTargets.some((p) => positionsEqual(p, pos));

        // Determine owner of stack for selection pulse color
        const stackOwner = stack?.rings?.[0];

        const collapsedOwner =
          cellVM?.collapsedSpace?.ownerPlayerNumber ||
          (collapsedOwnerFromBoard !== undefined ? collapsedOwnerFromBoard : undefined);

        const territoryClasses = collapsedOwner
          ? getPlayerColors(collapsedOwner).territory
          : 'bg-slate-300/80';

        const lineOverlayPlayer = showLineOverlays ? lineOverlayByKey.get(key) : undefined;
        const territoryOverlayInfo = showTerritoryRegionOverlays
          ? territoryOverlayByKey.get(key)
          : undefined;

        // Decision-phase highlight metadata for this hex cell, if any.
        const hexHighlightMeta = highlightByKey.get(key);
        const decisionHighlight = hexHighlightMeta?.intensity;
        const territoryRegionGroupIdsHex =
          isTerritoryRegionDecisionContext && hexHighlightMeta?.groupIds
            ? hexHighlightMeta.groupIds
            : [];
        const hasStackForPulse = !!(cellVM?.stack || stack);
        const isMoveDestination =
          hasStackForPulse &&
          animations.some(
            (anim) =>
              anim.position === key &&
              (anim.type === 'move' || anim.type === 'capture' || anim.type === 'chain_capture')
          );

        if (isMoveDestination) {
          debugLog(
            isSandboxAnimationDebugEnabled(),
            '[SandboxAnimationDebug] BoardView: destination pulse (hex)',
            {
              key,
              boardType,
              hasStack: !!stack,
              hasCellVMStack: !!cellVM?.stack,
            }
          );
        }
        let decisionHighlightClass = '';
        if (decisionHighlight === 'primary') {
          if (isLineDecisionContext) {
            decisionHighlightClass = 'decision-highlight-primary line-formation-burst';
          } else if (!isRingEliminationDecisionContext) {
            decisionHighlightClass = 'decision-highlight-primary';
          }
        } else if (decisionHighlight === 'secondary') {
          decisionHighlightClass = 'decision-highlight-secondary';
        }

        // Invalid move shake animation: apply when this cell is the target of an invalid move attempt
        const isShaking = shakingCellKey === key;

        const shouldPulseCaptureTargetHex =
          decisionHighlight === 'secondary' && isCaptureDirectionDecisionContext;
        const shouldPulseEliminationTargetHex =
          decisionHighlight === 'primary' && isRingEliminationDecisionContext;
        const shouldPulseTerritoryRegionHex =
          decisionHighlight === 'primary' && isTerritoryRegionDecisionContext;

        // Track palette index for ARIA descriptions; -1 means "not assigned".
        let territoryRegionPaletteIndexHex = -1;
        const territoryRegionClassesHex: string[] = [];
        if (shouldPulseTerritoryRegionHex && territoryRegionGroupIdsHex.length > 0) {
          if (territoryRegionGroupIdsHex.length > 1) {
            territoryRegionClassesHex.push('territory-region-overlap');
          } else {
            const regionId = territoryRegionGroupIdsHex[0];
            const paletteIndex = Math.max(0, territoryRegionIdsInDisplayOrder.indexOf(regionId));
            territoryRegionPaletteIndexHex = paletteIndex;
            switch (paletteIndex) {
              case 0:
                territoryRegionClassesHex.push('territory-region-a');
                break;
              case 1:
                territoryRegionClassesHex.push('territory-region-b');
                break;
              case 2:
                territoryRegionClassesHex.push('territory-region-c');
                break;
              default:
                territoryRegionClassesHex.push('territory-region-d');
                break;
            }
          }
        }

        // Mobile-responsive hex cell sizing for W3-12 (44px touch targets)
        // Hex cells: 44px minimum for touch, 48px on md (vs square8's 44px/56px/80px)
        // Using rounded-full for hex shape appearance
        const hexCellSizeClasses = 'w-11 h-11 md:w-12 md:h-12'; // 44px → 48px

        const cellClasses = [
          `relative ${hexCellSizeClasses} mx-0 flex items-center justify-center text-[11px] md:text-xs rounded-full border`,
          'border-slate-600 text-slate-100',
          territoryClasses,
          decisionHighlightClass,
          isMoveDestination ? 'move-destination-pulse' : '',
          decisionHighlight === 'primary' && isCaptureDirectionDecisionContext
            ? 'decision-pulse-capture'
            : '',
          shouldPulseCaptureTargetHex ? 'capture-target-pulse' : '',
          shouldPulseEliminationTargetHex ? 'decision-pulse-elimination' : '',
          shouldPulseTerritoryRegionHex ? 'decision-pulse-territory' : '',
          ...territoryRegionClassesHex,
          effectiveIsSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          // Valid target highlighting with subtle pulse animation
          effectiveIsValid
            ? 'outline outline-[2px] outline-emerald-300/90 outline-offset-[-4px] bg-emerald-400/[0.03] valid-move-cell'
            : '',
          // Invalid move shake animation
          isShaking ? 'invalid-move-shake' : '',
        ]
          .filter(Boolean)
          .join(' ');

        const hasMarkerVM = !!cellVM?.marker;
        // Suppress raw marker rendering when a stack is present at the same
        // position, so hex cells do not visually show both a stack and a
        // marker overlay. Any combined representation should be driven by
        // the view model instead.
        const hasMarkerBoard = !stack && marker && marker.type === 'regular';
        const hasMarker = hasMarkerVM || hasMarkerBoard;
        const markerColorClass =
          cellVM?.marker?.colorClass ??
          (hasMarkerBoard && marker ? getPlayerColors(marker.player).marker : null);

        // Mobile-responsive hex marker sizing (scales with larger 44px cells)
        const markerOuterSizeClasses = 'w-5 h-5 md:w-6 md:h-6';
        const label = getHexLabel(q, r);

        // Check if this cell is keyboard-focused (different from selected)
        const isFocused = focusedPosition && positionsEqual(focusedPosition, pos);
        const focusClasses =
          isFocused && !effectiveIsSelected
            ? 'ring-2 ring-amber-400 ring-offset-1 ring-offset-slate-950'
            : '';

        // Generate accessible label for hex cells
        const stackInfo = cellVM?.stack
          ? `Stack height ${cellVM.stack.stackHeight}, cap ${cellVM.stack.capHeight}, player ${cellVM.stack.controllingPlayer}`
          : stack
            ? `Stack height ${stack.stackHeight}, cap ${stack.capHeight}, player ${stack.controllingPlayer}`
            : 'Empty cell';
        const accessibilityAnnotations: string[] = [];

        if (decisionHighlight === 'primary' && isCaptureDirectionDecisionContext) {
          accessibilityAnnotations.push('Capture landing, valid move target');
        } else if (shouldPulseCaptureTargetHex) {
          accessibilityAnnotations.push('Capture target stack');
        }

        if (shouldPulseEliminationTargetHex) {
          accessibilityAnnotations.push('Ring elimination candidate');
        }

        if (shouldPulseTerritoryRegionHex) {
          if (territoryRegionGroupIdsHex.length > 1) {
            accessibilityAnnotations.push('Overlapping territory regions for this choice');
          } else if (territoryRegionPaletteIndexHex >= 0) {
            const humanIndex = territoryRegionPaletteIndexHex + 1;
            accessibilityAnnotations.push(`Territory region option ${humanIndex}`);
          } else {
            accessibilityAnnotations.push('Territory region option');
          }
        }

        if (chainCaptureKeySet && chainCaptureKeySet.has(key)) {
          if (chainCaptureCurrentKey && chainCaptureCurrentKey === key) {
            accessibilityAnnotations.push('Current capture position in chain');
          } else {
            accessibilityAnnotations.push('Visited position in capture chain');
          }
        }

        if (effectiveIsValid && accessibilityAnnotations.length === 0) {
          accessibilityAnnotations.push('Valid move target');
        }

        const accessibilitySuffix =
          accessibilityAnnotations.length > 0 ? `. ${accessibilityAnnotations.join('. ')}` : '';

        const coordLabel = formatPosition(pos, notationOptions);
        const cellLabel = `Cell ${coordLabel}. ${stackInfo}${accessibilitySuffix}`;

        cells.push(
          <button
            key={key}
            ref={(ref) => registerCellRef(key, ref)}
            type="button"
            data-x={pos.x}
            data-y={pos.y}
            data-z={typeof pos.z === 'number' ? pos.z : undefined}
            data-decision-highlight={decisionHighlight || undefined}
            data-line-overlay={lineOverlayPlayer !== undefined ? 'true' : undefined}
            data-line-overlay-player={
              lineOverlayPlayer !== undefined ? String(lineOverlayPlayer) : undefined
            }
            data-region-overlay={territoryOverlayInfo ? 'true' : undefined}
            data-region-overlay-player={
              territoryOverlayInfo ? String(territoryOverlayInfo.player) : undefined
            }
            data-region-overlay-disconnected={
              territoryOverlayInfo?.isDisconnected ? 'true' : undefined
            }
            onClick={() => !isSpectator && onCellClick?.(pos)}
            onDoubleClick={() => !isSpectator && onCellDoubleClick?.(pos)}
            onContextMenu={(e) => {
              e.preventDefault();
              !isSpectator && onCellContextMenu?.(pos);
            }}
            onFocus={() => setFocusedPosition(pos)}
            onTouchStart={(e) => {
              if (isSpectator || e.touches.length !== 1) return;
              const ts = touchStateRef.current;
              if (ts.longPressTimer) clearTimeout(ts.longPressTimer);
              ts.activeKey = key;
              ts.longPressTimer = setTimeout(() => {
                if (ts.activeKey === key) {
                  ts.activeKey = null;
                  onCellContextMenu?.(pos);
                }
              }, LONG_PRESS_DELAY_MS);
            }}
            onTouchEnd={(e) => {
              if (isSpectator) return;
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              if (ts.activeKey !== key) return;
              ts.activeKey = null;
              const now = Date.now();
              if (ts.lastTapKey === key && now - ts.lastTapTime < DOUBLE_TAP_DELAY_MS) {
                e.preventDefault();
                ts.lastTapKey = null;
                ts.lastTapTime = 0;
                onCellDoubleClick?.(pos);
              } else {
                ts.lastTapKey = key;
                ts.lastTapTime = now;
              }
            }}
            onTouchMove={() => {
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              ts.activeKey = null;
            }}
            onTouchCancel={() => {
              const ts = touchStateRef.current;
              if (ts.longPressTimer) {
                clearTimeout(ts.longPressTimer);
                ts.longPressTimer = null;
              }
              ts.activeKey = null;
            }}
            className={`${cellClasses} ${focusClasses} ${isSpectator ? 'cursor-default' : 'cursor-pointer'}`}
            disabled={isSpectator}
            tabIndex={isFocused ? 0 : -1}
            role="gridcell"
            aria-label={cellLabel}
            aria-selected={effectiveIsSelected || undefined}
          >
            {cellVM?.stack ? (
              <StackFromViewModel
                stack={cellVM.stack}
                boardType={boardType}
                animationClass={`${getAnimationClass(key) ?? ''} ${
                  shouldPulseEliminationTargetHex ? 'decision-elimination-stack-pulse' : ''
                }`.trim()}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
                isJustMoved={isMoveDestination}
              />
            ) : stack ? (
              <StackWidget
                stack={stack}
                boardType={boardType}
                animationClass={`${getAnimationClass(key) ?? ''} ${
                  shouldPulseEliminationTargetHex ? 'decision-elimination-stack-pulse' : ''
                }`.trim()}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
                isJustMoved={isMoveDestination}
              />
            ) : null}

            {hasMarker && markerColorClass && (
              <div
                className={`absolute inset-0 m-auto rounded-full border-[4px] ${markerColorClass} bg-transparent shadow-sm shadow-slate-900/70 z-10 flex items-center justify-center ${markerOuterSizeClasses}`}
              >
                <div className="w-2/3 h-2/3 rounded-full bg-slate-950/75 flex items-center justify-center">
                  <div className="w-1/3 h-1/3 rounded-full bg-slate-950/95 border border-slate-900/95" />
                </div>
              </div>
            )}

            {collapsedOwner && (
              <div
                className={`pointer-events-none absolute inset-[1.5px] rounded-full border-2 ${getPlayerColors(collapsedOwner).marker} border-opacity-90`}
              />
            )}

            {label && !stack && !hasMarker && (
              <span className="pointer-events-none absolute text-[8px] text-slate-500/50 font-mono">
                {label}
              </span>
            )}
          </button>
        );
      }

      rows.push(
        <div key={q} className="flex justify-center -space-x-0.5">
          {cells}
        </div>
      );
    }

    // Wrap all rows in a column with slight negative vertical spacing so
    // circles appear in a tight hexagonal packing without overlaps.
    // This inner container is the geometry reference for DOM-based grid alignment.
    return (
      <div
        ref={boardGeometryRef}
        className="relative flex flex-col -space-y-1"
        role="grid"
        aria-label="Hexagonal game board"
      >
        {rows}
        {renderMovementOverlay()}
        {renderChainCapturePathOverlay()}
        {/* Move animation layer */}
        {pendingAnimation && (
          <MoveAnimationLayer
            animation={pendingAnimation}
            cellRefs={cellRefs}
            containerRef={boardGeometryRef}
            onComplete={() => onAnimationComplete?.()}
          />
        )}
      </div>
    );
  };

  // Render the board with keyboard navigation wrapper
  const renderBoard = () => {
    if (effectiveBoardType === 'square8') {
      return renderSquareBoard(8);
    }
    if (effectiveBoardType === 'square19') {
      return renderSquareBoard(19);
    }
    if (effectiveBoardType === 'hexagonal') {
      // Hex board has boardGeometryRef and overlay inside renderHexBoard()
      // for tighter alignment with the actual cell grid (no padding interference)
      return (
        <div className="relative inline-block p-2 border border-slate-300 rounded-md bg-white text-slate-900 shadow-inner">
          {renderHexBoard()}
        </div>
      );
    }
    return null;
  };

  // Mobile viewport scroll handling (W3-12):
  // - Square8: 8×44px cells + 7×2px gaps = 366px fits in 375px viewport
  // - Square19: 19×44px + 18×2px = 872px requires horizontal scroll
  // - board-scroll-container provides touch-friendly scrolling for oversized boards
  // - board-container prevents text selection during drag
  const needsScroll = effectiveBoardType === 'square19' || effectiveBoardType === 'hexagonal';
  const boardAriaName =
    effectiveBoardType === 'square8'
      ? '8x8'
      : effectiveBoardType === 'square19'
        ? '19x19'
        : 'Hexagonal';

  return (
    <div
      ref={boardContainerRef}
      className={`inline-block board-container ${needsScroll ? 'board-scroll-container' : ''}`}
      data-testid="board-view"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      role="region"
      aria-label={`${boardAriaName} game board. Use arrow keys to navigate, Enter or Space to select, Escape to clear selection, Home or End to jump, question mark for board controls and shortcuts`}
    >
      {renderBoard()}
      {/* Screen reader announcements */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>
    </div>
  );
};
