import React from 'react';
import {
  BoardType,
  BoardState,
  Position,
  RingStack,
  positionToString,
  positionsEqual
} from '../../shared/types/game';
import { computeBoardMovementGrid } from '../utils/boardMovementGrid';

export interface BoardViewProps {
  boardType: BoardType;
  board: BoardState;
  selectedPosition?: Position;
  validTargets?: Position[];
  onCellClick?: (position: Position) => void;
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
    territory: 'bg-emerald-700/85'
  },
  2: {
    ring: 'bg-sky-600',
    ringBorder: 'border-sky-300',
    marker: 'border-sky-500',
    territory: 'bg-sky-700/85'
  },
  3: {
    ring: 'bg-amber-400',
    ringBorder: 'border-amber-200',
    marker: 'border-amber-400',
    territory: 'bg-amber-600/85'
  },
  4: {
    ring: 'bg-fuchsia-400',
    ringBorder: 'border-fuchsia-200',
    marker: 'border-fuchsia-400',
    territory: 'bg-fuchsia-700/85'
  }
};

const getPlayerColors = (playerNumber?: number) => {
  if (!playerNumber) {
    return {
      ring: 'bg-slate-300',
      ringBorder: 'border-slate-100',
      marker: 'border-slate-300',
      territory: 'bg-slate-800/70'
    };
  }
  return (
    PLAYER_COLOR_CLASSES[playerNumber] || {
      ring: 'bg-slate-300',
      ringBorder: 'border-slate-100',
      marker: 'border-slate-300',
      territory: 'bg-slate-800/70'
    }
  );
};

const StackWidget: React.FC<{ stack: RingStack; boardType: BoardType }> = ({ stack, boardType }) => {
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

  return (
    <div className={`flex flex-col items-center justify-center gap-[1px] ${verticalOffsetClasses}`}>
      <div className="flex flex-col items-center -space-y-[1px]">
        {rings.map((playerNumber, index) => {
          const { ring, ringBorder } = getPlayerColors(playerNumber);
          const isTop = index === topIndex;
          const isInCap = index <= capEndIndex;

          const baseShape = `${ringSizeClasses} rounded-full border`;
          const capOutline = isInCap ? 'ring-[0.5px] ring-offset-[0.5px] ring-offset-slate-900' : '';
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

export const BoardView: React.FC<BoardViewProps> = ({
  boardType,
  board,
  selectedPosition,
  validTargets = [],
  onCellClick,
  onCellDoubleClick,
  onCellContextMenu,
  showMovementGrid = false
}) => {
  // Square boards: simple grid using (x, y) coordinates.
  // Hex board: rendered using the same cube/axial coordinate system
  // as BoardManager (q = x, r = y, s = z, with q + r + s = 0).

  const movementGrid = showMovementGrid ? computeBoardMovementGrid(board) : null;
  const centersByKey = movementGrid
    ? new Map(movementGrid.centers.map(center => [center.key, center] as const))
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
        {movementGrid.edges.map(edge => {
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

        {movementGrid.centers.map(center => (
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

  const renderSquareBoard = (size: number) => {
    const rows: JSX.Element[] = [];
    // Cell sizing: make 8x8 squares roughly 2x the original size, and
    // 19x19 squares ~30% larger, so stacks up to height 10 remain legible
    // without overwhelming the viewport.
    const squareCellSizeClasses =
      boardType === 'square8'
        ? 'w-16 h-16 md:w-20 md:h-20'
        : 'w-11 h-11 md:w-14 md:h-14';

    for (let y = 0; y < size; y++) {
      const cells: JSX.Element[] = [];
      for (let x = 0; x < size; x++) {
        const pos: Position = { x, y };
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        const marker = board.markers.get(key);
        const collapsedOwner = board.collapsedSpaces.get(key);
        const isSelected = selectedPosition && positionsEqual(selectedPosition, pos);
        const isValid = validTargets.some(p => positionsEqual(p, pos));

        const isDarkSquare = (x + y) % 2 === 0;
        const baseSquareBg = isDarkSquare ? 'bg-slate-300' : 'bg-slate-100';
        const territoryClasses = collapsedOwner
          ? getPlayerColors(collapsedOwner).territory
          : '';

        const cellClasses = [
          'relative border flex items-center justify-center text-[11px] md:text-xs rounded-sm',
          squareCellSizeClasses,
          'border-slate-600 text-slate-900',
          territoryClasses || baseSquareBg,
          isSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          // Valid target highlighting on square boards: thin, bright-green inset
          // ring plus a light near-white emerald tint that reads clearly even
          // over the dark board container background.
          isValid
            ? 'outline outline-[2px] outline-emerald-300/90 outline-offset-[-4px] bg-emerald-50'
            : ''
        ]
          .filter(Boolean)
          .join(' ');

        const hasMarker = marker && marker.type === 'regular';
        const markerColors = hasMarker ? getPlayerColors(marker.player) : null;

        const markerOuterSizeClasses =
          boardType === 'square8'
            ? 'w-6 h-6 md:w-7 md:h-7'
            : 'w-5 h-5 md:w-6 md:h-6';

        cells.push(
          <button
            key={key}
            type="button"
            onClick={() => onCellClick?.(pos)}
            onDoubleClick={() => onCellDoubleClick?.(pos)}
            onContextMenu={e => {
              e.preventDefault();
              onCellContextMenu?.(pos);
            }}
            className={cellClasses}
          >
            {stack ? <StackWidget stack={stack} boardType={boardType} /> : null}

            {hasMarker && markerColors && (
              <div
                className={`absolute inset-0 m-auto rounded-full border-[6px] ${markerColors.marker} bg-transparent shadow-sm shadow-slate-900/70 z-10 flex items-center justify-center ${markerOuterSizeClasses}`}
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
    return rows;
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

    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);

      const cells: JSX.Element[] = [];
      for (let r = r1; r <= r2; r++) {
        const s = -q - r;
        const pos: Position = { x: q, y: r, z: s };
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        const marker = board.markers.get(key);
        const collapsedOwner = board.collapsedSpaces.get(key);
        const isSelected = selectedPosition && positionsEqual(selectedPosition, pos);
        const isValid = validTargets.some(p => positionsEqual(p, pos));

        const territoryClasses = collapsedOwner
          ? getPlayerColors(collapsedOwner).territory
          : 'bg-slate-300/80';

        const cellClasses = [
          'relative w-8 h-8 md:w-9 md:h-9 mx-0 flex items-center justify-center text-[11px] md:text-xs rounded-full border',
          'border-slate-600 text-slate-100',
          territoryClasses,
          isSelected ? 'ring-2 ring-emerald-400 ring-offset-2 ring-offset-slate-950' : '',
          isValid
            ? 'outline outline-[2px] outline-emerald-300/90 outline-offset-[-4px] bg-emerald-400/[0.03]'
            : ''
        ]
          .filter(Boolean)
          .join(' ');


        const hasMarker = marker && marker.type === 'regular';
        const markerColors = hasMarker ? getPlayerColors(marker.player) : null;

        const markerOuterSizeClasses = 'w-4 h-4 md:w-5 md:h-5';

        cells.push(
          <button
            key={key}
            type="button"
            onClick={() => onCellClick?.(pos)}
            onDoubleClick={() => onCellDoubleClick?.(pos)}
            onContextMenu={e => {
              e.preventDefault();
              onCellContextMenu?.(pos);
            }}
            className={cellClasses}
          >
            {stack ? <StackWidget stack={stack} boardType={boardType} /> : null}

            {hasMarker && markerColors && (
              <div
                className={`absolute inset-0 m-auto rounded-full border-[4px] ${markerColors.marker} bg-transparent shadow-sm shadow-slate-900/70 z-10 flex items-center justify-center ${markerOuterSizeClasses}`}
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

  return (
    <div className="inline-block">
      {boardType === 'square8' && (
        <div className="relative space-y-1 bg-slate-800/60 p-2 rounded-md border border-slate-700 shadow-inner">
          {renderSquareBoard(8)}
          {renderMovementOverlay()}
        </div>
      )}
      {boardType === 'square19' && (
        <div className="relative space-y-0.5 scale-75 origin-top-left bg-slate-800/60 p-2 rounded-md border border-slate-700 shadow-inner">
          {renderSquareBoard(19)}
          {renderMovementOverlay()}
        </div>
      )}
      {boardType === 'hexagonal' && (
        <div className="relative p-2 border border-slate-300 rounded-md bg-white text-slate-900 shadow-inner">
          {renderHexBoard()}
          {renderMovementOverlay()}
        </div>
      )}
    </div>
  );
};
