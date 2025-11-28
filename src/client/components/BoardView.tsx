import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  BoardType,
  BoardState,
  Position,
  RingStack,
  positionToString,
  positionsEqual,
} from '../../shared/types/game';
import { computeBoardMovementGrid } from '../utils/boardMovementGrid';
import type { BoardViewModel, CellViewModel, StackViewModel } from '../adapters/gameViewModels';

// Keyboard navigation helpers
interface FocusableCell {
  position: Position;
  key: string;
}

// Animation type for tracking piece animations
type AnimationType = 'place' | 'move' | 'capture';

interface AnimationState {
  position: string; // positionToString key
  type: AnimationType;
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
  selectedPosition?: Position;
  validTargets?: Position[];
  onCellClick?: (position: Position) => void;
  isSpectator?: boolean;
  /**
   * Optional double-click handler, primarily used by the local sandbox
   * to distinguish between selection (single click) and stacked ring
   * placement (double click) during the ring placement phase.
   */
  onCellDoubleClick?: (position: Position) => void;
  /**
   * Optional context menu handler (right-click / long-press proxy),
   * used by the local sandbox to surface ring-count selection dialogs
   * for multi-ring placements.
   */
  onCellContextMenu?: (position: Position) => void;
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

const generateRankLabels = (size: number): string[] =>
  Array.from({ length: size }, (_, idx) => (size - idx).toString());

const StackWidget: React.FC<{
  stack: RingStack;
  boardType: BoardType;
  animationClass?: string | undefined;
  isSelected?: boolean | undefined;
  ownerPlayerId?: number | undefined;
}> = ({ stack, boardType, animationClass, isSelected = false, ownerPlayerId }) => {
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

  const ringSizeClasses = isSquare8
    ? 'w-7 md:w-8 h-[5px] md:h-[6px]'
    : isHex
      ? 'w-5 md:w-6 h-[3px] md:h-[4px]'
      : 'w-6 md:w-7 h-[4px] md:h-[5px]';

  const labelTextClasses = isSquare8
    ? 'text-[9px] md:text-[10px]'
    : isHex
      ? 'text-[7px] md:text-[8px]'
      : 'text-[8px] md:text-[9px]';

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

          const baseShape = `${ringSizeClasses} rounded-full border`;
          const capOutline = isInCap
            ? 'ring-[0.5px] ring-offset-[0.5px] ring-offset-slate-900'
            : '';
          const topShadow = isTop ? 'shadow-md shadow-slate-900/70' : 'shadow-sm';

          return (
            <div
              // eslint-disable-next-line react/no-array-index-key
              key={index}
              className={`${baseShape} ${ring} ${ringBorder} ${capOutline} ${topShadow}`}
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
}> = ({ stack, boardType, animationClass, isSelected = false, ownerPlayerId }) => {
  const { rings, stackHeight, capHeight } = stack;

  const isSquare8 = boardType === 'square8';
  const isHex = boardType === 'hexagonal';

  const verticalOffsetClasses = 'translate-y-[3px] md:translate-y-[4px]';

  const ringSizeClasses = isSquare8
    ? 'w-7 md:w-8 h-[5px] md:h-[6px]'
    : isHex
      ? 'w-5 md:w-6 h-[3px] md:h-[4px]'
      : 'w-6 md:w-7 h-[4px] md:h-[5px]';

  const labelTextClasses = isSquare8
    ? 'text-[9px] md:text-[10px]'
    : isHex
      ? 'text-[7px] md:text-[8px]'
      : 'text-[8px] md:text-[9px]';

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

          const baseShape = `${ringSizeClasses} rounded-full border`;
          const capOutline = isInCap
            ? 'ring-[0.5px] ring-offset-[0.5px] ring-offset-slate-900'
            : '';
          const topShadow = isTop ? 'shadow-md shadow-slate-900/70' : 'shadow-sm';

          return (
            // eslint-disable-next-line react/no-array-index-key
            <div
              key={index}
              className={`${baseShape} ${colorClass} ${borderClass} ${capOutline} ${topShadow}`}
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
}) => {
  // Animation state tracking
  const [animations, setAnimations] = useState<AnimationState[]>([]);
  const prevBoardRef = useRef<BoardState | null>(null);

  // Keyboard navigation state
  const [focusedPosition, setFocusedPosition] = useState<Position | null>(null);
  const boardContainerRef = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<Map<string, HTMLButtonElement>>(new Map());

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

  const effectiveBoardType: BoardType = viewModel?.boardType ?? boardType;
  const effectiveSize = viewModel?.size ?? board.size;

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
          setFocusedPosition(allPositions[0].position);
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
        default:
          break;
      }
    },
    [focusedPosition, onCellClick, isSpectator, board.stacks, moveFocus, clearSelection]
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

  const movementGrid = showMovementGrid ? computeBoardMovementGrid(board) : null;
  const centersByKey = movementGrid
    ? new Map(movementGrid.centers.map((center) => [center.key, center] as const))
    : null;

  const renderMovementOverlay = () => {
    if (!movementGrid || !centersByKey || movementGrid.edges.length === 0) {
      return null;
    }

    return (
      <svg
        className="pointer-events-none absolute inset-0 z-10"
        viewBox="0 0 1 1"
        preserveAspectRatio="xMidYMid meet"
      >
        {movementGrid.edges.map((edge) => {
          const from = centersByKey.get(edge.fromKey);
          const to = centersByKey.get(edge.toKey);
          if (!from || !to) return null;

          return (
            <line
              key={`${edge.fromKey}->${edge.toKey}`}
              x1={from.cx}
              y1={from.cy}
              x2={to.cx}
              y2={to.cy}
              stroke="rgba(148, 163, 184, 0.45)" // slate-400/45
              strokeWidth={0.0025}
            />
          );
        })}

        {movementGrid.centers.map((center) => (
          <circle
            key={center.key}
            cx={center.cx}
            cy={center.cy}
            r={0.004}
            fill="rgba(148, 163, 184, 0.75)" // slightly brighter node dots
          />
        ))}
      </svg>
    );
  };

  // Square boards: simple grid using (x, y) coordinates.
  // Hex board: rendered using the same cube/axial coordinate system
  // as BoardManager (q = x, r = y, s = z, with q + r + s = 0).

  const renderSquareCoordinateLabels = (size: number) => {
    const skipI = effectiveBoardType === 'square19';
    const files = generateFileLabels(size, skipI);
    const ranks = generateRankLabels(size);
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
    const rows: JSX.Element[] = [];
    // Cell sizing: make 8x8 squares roughly 2x the original size, and
    // 19x19 squares ~30% larger, so stacks up to height 10 remain legible
    // without overwhelming the viewport.
    const squareCellSizeClasses =
      boardType === 'square8' ? 'w-16 h-16 md:w-20 md:h-20' : 'w-11 h-11 md:w-14 md:h-14';

    for (let y = 0; y < size; y++) {
      const cells: JSX.Element[] = [];
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
          cellVM?.collapsedSpace?.ownerPlayerNumber ??
          (collapsedOwnerFromBoard ? collapsedOwnerFromBoard.controllingPlayer : undefined);
        const territoryClasses = collapsedOwner ? getPlayerColors(collapsedOwner).territory : '';

        const cellClasses = [
          'relative border flex items-center justify-center text-[11px] md:text-xs rounded-sm',
          squareCellSizeClasses,
          'border-slate-600 text-slate-900',
          territoryClasses || baseSquareBg,
          effectiveIsSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          // Valid target highlighting on square boards: thin, bright-green inset
          // ring plus a light near-white emerald tint that reads clearly even
          // over the dark board container background. Also apply subtle pulse animation.
          effectiveIsValid
            ? 'outline outline-[2px] outline-emerald-300/90 outline-offset-[-4px] bg-emerald-50 valid-move-cell'
            : '',
        ]
          .filter(Boolean)
          .join(' ');

        const hasMarkerVM = !!cellVM?.marker;
        const hasMarkerBoard = marker && marker.type === 'regular';
        const hasMarker = hasMarkerVM || hasMarkerBoard;
        const markerColorClass =
          cellVM?.marker?.colorClass ??
          (hasMarkerBoard ? getPlayerColors(marker!.player).marker : null);

        const markerOuterSizeClasses =
          boardType === 'square8' ? 'w-6 h-6 md:w-7 md:h-7' : 'w-5 h-5 md:w-6 md:h-6';

        // Check if this cell is keyboard-focused (different from selected)
        const isFocused = focusedPosition && positionsEqual(focusedPosition, pos);
        const focusClasses =
          isFocused && !effectiveIsSelected
            ? 'ring-2 ring-amber-400 ring-offset-1 ring-offset-slate-950'
            : '';

        // Generate accessible label for the cell
        const stackInfo = cellVM?.stack
          ? `Stack height ${cellVM.stack.stackHeight}, cap ${cellVM.stack.capHeight}, player ${cellVM.stack.controllingPlayer}`
          : stack
            ? `Stack height ${stack.stackHeight}, cap ${stack.capHeight}, player ${stack.controllingPlayer}`
            : 'Empty cell';
        const cellLabel = `Row ${y + 1}, Column ${x + 1}. ${stackInfo}${effectiveIsValid ? '. Valid move target' : ''}`;

        cells.push(
          <button
            key={key}
            ref={(ref) => registerCellRef(key, ref)}
            type="button"
            onClick={() => !isSpectator && onCellClick?.(pos)}
            onDoubleClick={() => !isSpectator && onCellDoubleClick?.(pos)}
            onContextMenu={(e) => {
              e.preventDefault();
              !isSpectator && onCellContextMenu?.(pos);
            }}
            onFocus={() => setFocusedPosition(pos)}
            className={`${cellClasses} ${focusClasses} ${isSpectator ? 'cursor-default' : 'cursor-pointer'}`}
            disabled={isSpectator}
            tabIndex={0}
            role="gridcell"
            aria-label={cellLabel}
            aria-selected={effectiveIsSelected || undefined}
          >
            {cellVM?.stack ? (
              <StackFromViewModel
                stack={cellVM.stack}
                boardType={boardType}
                animationClass={getAnimationClass(key)}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
              />
            ) : stack ? (
              <StackWidget
                stack={stack}
                boardType={boardType}
                animationClass={getAnimationClass(key)}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
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
      <div className={containerClasses}>
        {rows}
        {renderMovementOverlay()}
        {showCoordinateLabels ? renderSquareCoordinateLabels(size) : null}
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

    const radius = board.size - 1; // e.g. size=11 => radius=10
    const rows: JSX.Element[] = [];

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

      const cells: JSX.Element[] = [];
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
          cellVM?.collapsedSpace?.ownerPlayerNumber ??
          (collapsedOwnerFromBoard ? collapsedOwnerFromBoard.controllingPlayer : undefined);

        const territoryClasses = collapsedOwner
          ? getPlayerColors(collapsedOwner).territory
          : 'bg-slate-300/80';

        const cellClasses = [
          'relative w-8 h-8 md:w-9 md:h-9 mx-0 flex items-center justify-center text-[11px] md:text-xs rounded-full border',
          'border-slate-600 text-slate-100',
          territoryClasses,
          effectiveIsSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          // Valid target highlighting with subtle pulse animation
          effectiveIsValid
            ? 'outline outline-[2px] outline-emerald-300/90 outline-offset-[-4px] bg-emerald-400/[0.03] valid-move-cell'
            : '',
        ]
          .filter(Boolean)
          .join(' ');

        const hasMarkerVM = !!cellVM?.marker;
        const hasMarkerBoard = marker && marker.type === 'regular';
        const hasMarker = hasMarkerVM || hasMarkerBoard;
        const markerColorClass =
          cellVM?.marker?.colorClass ??
          (hasMarkerBoard ? getPlayerColors(marker!.player).marker : null);

        const markerOuterSizeClasses = 'w-4 h-4 md:w-5 md:h-5';
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
        const cellLabel = `Hex position q${q} r${r}. ${stackInfo}${effectiveIsValid ? '. Valid move target' : ''}`;

        cells.push(
          <button
            key={key}
            ref={(ref) => registerCellRef(key, ref)}
            type="button"
            onClick={() => !isSpectator && onCellClick?.(pos)}
            onDoubleClick={() => !isSpectator && onCellDoubleClick?.(pos)}
            onContextMenu={(e) => {
              e.preventDefault();
              !isSpectator && onCellContextMenu?.(pos);
            }}
            onFocus={() => setFocusedPosition(pos)}
            className={`${cellClasses} ${focusClasses} ${isSpectator ? 'cursor-default' : 'cursor-pointer'}`}
            disabled={isSpectator}
            tabIndex={0}
            role="gridcell"
            aria-label={cellLabel}
            aria-selected={effectiveIsSelected || undefined}
          >
            {cellVM?.stack ? (
              <StackFromViewModel
                stack={cellVM.stack}
                boardType={boardType}
                animationClass={getAnimationClass(key)}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
              />
            ) : stack ? (
              <StackWidget
                stack={stack}
                boardType={boardType}
                animationClass={getAnimationClass(key)}
                isSelected={!!effectiveIsSelected}
                ownerPlayerId={stackOwner}
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
    return <div className="flex flex-col -space-y-1">{rows}</div>;
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
      return (
        <div className="relative inline-block p-2 border border-slate-300 rounded-md bg-white text-slate-900 shadow-inner">
          {renderHexBoard()}
          {renderMovementOverlay()}
        </div>
      );
    }
    return null;
  };

  return (
    <div
      ref={boardContainerRef}
      className="inline-block"
      data-testid="board-view"
      onKeyDown={handleKeyDown}
      role="grid"
      aria-label={`${effectiveBoardType} game board. Use arrow keys to navigate, Enter or Space to select, Escape to clear selection`}
    >
      {renderBoard()}
      {/* Screen reader announcements */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>
    </div>
  );
};
