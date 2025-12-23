import { BoardState, Position, positionToString } from '../../shared/types/game';

export interface MovementGridCellCenter {
  /** Position key as produced by positionToString */
  key: string;
  /** Discrete board position (square or hex) */
  position: Position;
  /**
   * Normalized center coordinates in board-local space.
   * - (0,0) is top-left of the board area.
   * - (1,1) is bottom-right of the board area.
   * These are independent of actual DOM pixel sizes.
   */
  cx: number;
  cy: number;
}

export interface MovementGridEdge {
  fromKey: string;
  toKey: string;
}

export interface MovementGrid {
  centers: MovementGridCellCenter[];
  edges: MovementGridEdge[];
}

/**
 * Compute a normalized movement grid (cell centers + adjacency links) for the
 * given BoardState.
 *
 * - Square boards: centers at (x + 0.5, y + 0.5) normalized by board size.
 *   Edges connect orthogonal + diagonal neighbors to visualize horizontal,
 *   vertical, and diagonal movement axes.
 * - Hex boards: centers derived from axial cube coordinates using a standard
 *   "pointy-top" hex layout, then normalized to [0,1]x[0,1]. Edges connect
 *   the 6 hex neighbors.
 */
export function computeBoardMovementGrid(board: BoardState): MovementGrid {
  switch (board.type) {
    case 'square8':
    case 'square19':
      return computeSquareMovementGrid(board);
    case 'hex8':
    case 'hexagonal':
      return computeHexMovementGrid(board);
    default:
      return { centers: [], edges: [] };
  }
}

function computeSquareMovementGrid(board: BoardState): MovementGrid {
  const size = board.size;
  const centers: MovementGridCellCenter[] = [];
  const indexByKey = new Map<string, number>();

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const position: Position = { x, y };
      const key = positionToString(position);
      // Start from ideal normalized centers and then apply a small
      // visual adjustment so the overlay better matches the actual
      // DOM layout (which includes padding and gaps between cells).
      const baseCx = (x + 0.5) / size;
      const baseCy = (y + 0.5) / size;
      const { cx, cy } = adjustSquareCenter(baseCx, baseCy, size);

      indexByKey.set(key, centers.length);
      centers.push({ key, position, cx, cy });
    }
  }

  const edges: MovementGridEdge[] = [];
  const deltas = [
    // orthogonal
    { dx: 1, dy: 0 },
    { dx: -1, dy: 0 },
    { dx: 0, dy: 1 },
    { dx: 0, dy: -1 },
    // diagonal
    { dx: 1, dy: 1 },
    { dx: 1, dy: -1 },
    { dx: -1, dy: 1 },
    { dx: -1, dy: -1 },
  ];

  for (const center of centers) {
    const { position } = center;
    for (const { dx, dy } of deltas) {
      const nx = position.x + dx;
      const ny = position.y + dy;
      if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;
      const neighborKey = positionToString({ x: nx, y: ny });
      if (!indexByKey.has(neighborKey)) continue;
      // Store undirected edge; we will render edges as-is (duplicates are cheap
      // and visually harmless), but we can skip dupes by ensuring an ordering.
      if (center.key < neighborKey) {
        edges.push({ fromKey: center.key, toKey: neighborKey });
      }
    }
  }

  return { centers, edges };
}

function computeHexMovementGrid(board: BoardState): MovementGrid {
  const radius = (board.size - 1) / 2; // size is bounding box, radius = (size-1)/2

  interface RawCenter {
    key: string;
    position: Position;
    px: number;
    py: number;
  }

  const rawCenters: RawCenter[] = [];
  const indexByKey = new Map<string, number>();

  // First pass: determine per-row widths so we can center rows horizontally
  // similar to the flex layout in BoardView (justify-center with shorter rows).
  const rowWidths: number[] = [];
  let maxRowWidth = 0;

  for (let q = -radius; q <= radius; q++) {
    const r1 = Math.max(-radius, -q - radius);
    const r2 = Math.min(radius, -q + radius);
    const width = r2 - r1 + 1;
    rowWidths.push(width);
    if (width > maxRowWidth) maxRowWidth = width;
  }

  // Second pass: assign simple row/column-based positions that mirror the
  // visual layout used by BoardView (rows centered, uniform vertical spacing).
  let rowIndex = 0;
  for (let q = -radius; q <= radius; q++, rowIndex++) {
    const r1 = Math.max(-radius, -q - radius);
    const r2 = Math.min(radius, -q + radius);
    const width = r2 - r1 + 1;

    // Center this row horizontally within the widest row.
    const rowOffset = (maxRowWidth - width) / 2;

    let colIndex = 0;
    for (let r = r1; r <= r2; r++, colIndex++) {
      const s = -q - r;
      const position: Position = { x: q, y: r, z: s };
      const key = positionToString(position);

      // Treat centers as (rowOffset + colIndex + 0.5, rowIndex + 0.5) so
      // we align the SVG nodes with the visual centers of the circles in
      // BoardView's flex layout. Absolute scale is normalized below.
      const px = rowOffset + colIndex + 0.5;
      const py = rowIndex + 0.5;

      indexByKey.set(key, rawCenters.length);
      rawCenters.push({ key, position, px, py });
    }
  }

  if (rawCenters.length === 0) {
    return { centers: [], edges: [] };
  }

  // Normalize px/py into [0,1]x[0,1]
  let minX = rawCenters[0].px;
  let maxX = rawCenters[0].px;
  let minY = rawCenters[0].py;
  let maxY = rawCenters[0].py;

  for (const c of rawCenters) {
    if (c.px < minX) minX = c.px;
    if (c.px > maxX) maxX = c.px;
    if (c.py < minY) minY = c.py;
    if (c.py > maxY) maxY = c.py;
  }

  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const centers: MovementGridCellCenter[] = rawCenters.map((c) => {
    const baseCx = (c.px - minX) / spanX;
    const baseCy = (c.py - minY) / spanY;
    const { cx, cy } = adjustHexCenter(baseCx, baseCy);

    return {
      key: c.key,
      position: c.position,
      cx,
      cy,
    };
  });

  const edges: MovementGridEdge[] = [];

  const hexDirections: Array<{ dx: number; dy: number; dz: number }> = [
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
      if (!indexByKey.has(neighborKey)) continue;
      if (center.key < neighborKey) {
        edges.push({ fromKey: center.key, toKey: neighborKey });
      }
    }
  }

  return { centers, edges };
}

/**
 * Apply small, view-layer-only adjustments to square board centers to
 * compensate for padding and inter-cell gaps in the DOM layout.
 *
 * These are intentionally tiny (single‑percent scale and a slight vertical
 * offset) and can be tuned in the future if the Tailwind layout changes.
 */
function adjustSquareCenter(
  baseCx: number,
  baseCy: number,
  _size: number
): { cx: number; cy: number } {
  // The BoardView now derives final movement-grid centers directly from
  // DOM cell positions for alignment, so we keep the logical geometry
  // here as a simple identity transform. This avoids layering additional
  // magic-number tweaks on top of the DOM-based projection while still
  // providing a sensible normalized grid for non-DOM callers.
  return { cx: baseCx, cy: baseCy };
}

/**
 * Similar small adjustment for hex centers: shrink the overall lattice a
 * bit so outer lines do not overshoot the rendered circles, and keep it
 * centred within the board container.
 */
function adjustHexCenter(baseCx: number, baseCy: number): { cx: number; cy: number } {
  // As with square boards, BoardView now reprojects hex centers from the
  // actual DOM layout when rendering the overlay. Keeping this as an
  // identity transform ensures any remaining non-DOM uses see the same
  // canonical geometry without extra scaling or offsets.
  return { cx: baseCx, cy: baseCy };
}
