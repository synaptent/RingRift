import React from 'react';
import {
  BoardType,
  BoardState,
  Position,
  positionToString,
  positionsEqual
} from '../../shared/types/game';

export interface BoardViewProps {
  boardType: BoardType;
  board: BoardState;
  selectedPosition?: Position;
  validTargets?: Position[];
  onCellClick?: (position: Position) => void;
}

export const BoardView: React.FC<BoardViewProps> = ({
  boardType,
  board,
  selectedPosition,
  validTargets = [],
  onCellClick
}) => {
  // Square boards: simple grid using (x, y) coordinates.
  // Hex board: rendered using the same cube/axial coordinate system
  // as BoardManager (q = x, r = y, s = z, with q + r + s = 0).

  const renderSquareBoard = (size: number) => {
    const rows = [];
    for (let y = 0; y < size; y++) {
      const cells = [];
      for (let x = 0; x < size; x++) {
        const pos: Position = { x, y };
        const key = positionToString(pos);
        const stack = board.stacks.get(key);
        const isSelected = selectedPosition && positionsEqual(selectedPosition, pos);
        const isValid = validTargets.some(p => positionsEqual(p, pos));

        cells.push(
          <button
            key={key}
            type="button"
            onClick={() => onCellClick?.(pos)}
            className={`w-8 h-8 border border-gray-500 flex items-center justify-center text-xs
              ${isSelected ? 'bg-blue-500 text-white' : ''}
              ${isValid ? 'ring-2 ring-green-400' : ''}
            `}
          >
            {stack ? stack.stackHeight : ''}
          </button>
        );
      }
      rows.push(
        <div key={y} className="flex">
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
    // We render each q as a row, centered via flexbox.

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
        const isSelected = selectedPosition && positionsEqual(selectedPosition, pos);
        const isValid = validTargets.some(p => positionsEqual(p, pos));

        cells.push(
          <button
            key={key}
            type="button"
            onClick={() => onCellClick?.(pos)}
            className={`w-8 h-8 mx-[2px] flex items-center justify-center text-xs rounded-full border border-gray-500
              ${isSelected ? 'bg-blue-500 text-white' : 'bg-slate-900/60'}
              ${isValid ? 'ring-2 ring-green-400' : ''}
            `}
          >
            {stack ? stack.stackHeight : ''}
          </button>
        );
      }

      rows.push(
        <div key={q} className="flex justify-center">
          {cells}
        </div>
      );
    }

    // If the board is empty (no stacks yet), we still show the empty hex grid.
    return rows;
  };

  return (
    <div className="inline-block">
      {boardType === 'square8' && (
        <div className="space-y-1">{renderSquareBoard(8)}</div>
      )}
      {boardType === 'square19' && (
        <div className="space-y-0.5 scale-75 origin-top-left">{renderSquareBoard(19)}</div>
      )}
      {boardType === 'hexagonal' && (
        <div className="p-2 border border-gray-500 rounded-md bg-slate-900/50 text-slate-100 space-y-1">
          {renderHexBoard()}
        </div>
      )}
    </div>
  );
};
