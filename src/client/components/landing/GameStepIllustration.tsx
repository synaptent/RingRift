/**
 * Small inline SVG illustrations for the "How to Play" landing section.
 * Each depicts a key game concept using the game's visual language.
 */

/** 3x3 board fragment with a ring being placed on a highlighted cell */
export function PlaceRingsIllustration() {
  const size = 28;
  const gap = 2;
  const step = size + gap;

  return (
    <svg viewBox="0 0 92 92" className="w-16 h-16" aria-hidden="true">
      {/* 3x3 grid */}
      {[0, 1, 2].map((r) =>
        [0, 1, 2].map((c) => {
          const x = 2 + c * step;
          const y = 2 + r * step;
          const isTarget = r === 1 && c === 1;
          return (
            <rect
              key={`${r}-${c}`}
              x={x}
              y={y}
              width={size}
              height={size}
              rx={3}
              fill={isTarget ? 'rgba(52, 211, 153, 0.25)' : 'rgba(51, 65, 85, 0.5)'}
              stroke={isTarget ? '#34d399' : 'rgb(71, 85, 105)'}
              strokeWidth="1.5"
            />
          );
        })
      )}
      {/* Ring being placed on center cell */}
      <circle
        cx={2 + 1 * step + size / 2}
        cy={2 + 1 * step + size / 2}
        r={9}
        fill="none"
        stroke="#34d399"
        strokeWidth="3.5"
      />
      {/* Arrow indicator from above */}
      <line
        x1={2 + 1 * step + size / 2}
        y1={2}
        x2={2 + 1 * step + size / 2}
        y2={2 + 1 * step - 2}
        stroke="#34d399"
        strokeWidth="1.5"
        strokeDasharray="3 2"
        opacity={0.6}
      />
    </svg>
  );
}

/** Row of connected rings with a glowing line indicator */
export function FormLinesIllustration() {
  const cellSize = 28;
  const gap = 2;
  const step = cellSize + gap;

  return (
    <svg viewBox="0 0 122 32" className="w-20 h-8" aria-hidden="true">
      {/* 4 cells in a row */}
      {[0, 1, 2, 3].map((c) => (
        <rect
          key={c}
          x={2 + c * step}
          y={2}
          width={cellSize}
          height={cellSize}
          rx={3}
          fill="rgba(51, 65, 85, 0.5)"
          stroke="rgb(71, 85, 105)"
          strokeWidth="1.5"
        />
      ))}
      {/* Connecting line glow */}
      <line
        x1={2 + cellSize / 2}
        y1={2 + cellSize / 2}
        x2={2 + 3 * step + cellSize / 2}
        y2={2 + cellSize / 2}
        stroke="#fbbf24"
        strokeWidth="2"
        opacity={0.4}
      />
      {/* Rings on each cell */}
      {[0, 1, 2, 3].map((c) => (
        <circle
          key={c}
          cx={2 + c * step + cellSize / 2}
          cy={2 + cellSize / 2}
          r={8}
          fill="none"
          stroke="#0284c7"
          strokeWidth="3"
        />
      ))}
    </svg>
  );
}

/** Grid section with filled territory cells */
export function ClaimTerritoryIllustration() {
  const size = 28;
  const gap = 2;
  const step = size + gap;
  // Territory pattern: some cells owned, some empty
  const territory = [
    [false, true, true],
    [true, true, true],
    [false, true, false],
  ];

  return (
    <svg viewBox="0 0 92 92" className="w-16 h-16" aria-hidden="true">
      {territory.map((row, r) =>
        row.map((owned, c) => (
          <rect
            key={`${r}-${c}`}
            x={2 + c * step}
            y={2 + r * step}
            width={size}
            height={size}
            rx={3}
            fill={owned ? 'rgba(4, 120, 87, 0.55)' : 'rgba(51, 65, 85, 0.5)'}
            stroke={owned ? '#059669' : 'rgb(71, 85, 105)'}
            strokeWidth="1.5"
          />
        ))
      )}
      {/* Ownership ring in center */}
      <circle
        cx={2 + 1 * step + size / 2}
        cy={2 + 1 * step + size / 2}
        r={8}
        fill="none"
        stroke="#34d399"
        strokeWidth="3"
      />
    </svg>
  );
}

/** Tiny 3x3 square grid for board geometry icon */
export function MiniSquareGridIcon() {
  return (
    <svg viewBox="0 0 28 28" className="w-6 h-6" aria-hidden="true">
      {[0, 1, 2].map((r) =>
        [0, 1, 2].map((c) => (
          <rect
            key={`${r}-${c}`}
            x={1 + c * 9}
            y={1 + r * 9}
            width={8}
            height={8}
            rx={1}
            fill="rgba(52, 211, 153, 0.3)"
            stroke="#34d399"
            strokeWidth="0.8"
          />
        ))
      )}
    </svg>
  );
}

/** Tiny hex cluster (3 cells) for board geometry icon */
export function MiniHexClusterIcon() {
  const r = 5;
  const hexPoints = (cx: number, cy: number) =>
    Array.from({ length: 6 }, (_, i) => {
      const angle = (Math.PI / 180) * (60 * i - 30);
      return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
    }).join(' ');

  return (
    <svg viewBox="0 0 28 28" className="w-6 h-6" aria-hidden="true">
      <polygon
        points={hexPoints(14, 8)}
        fill="rgba(52, 211, 153, 0.3)"
        stroke="#34d399"
        strokeWidth="0.8"
      />
      <polygon
        points={hexPoints(9, 18)}
        fill="rgba(52, 211, 153, 0.3)"
        stroke="#34d399"
        strokeWidth="0.8"
      />
      <polygon
        points={hexPoints(19, 18)}
        fill="rgba(52, 211, 153, 0.3)"
        stroke="#34d399"
        strokeWidth="0.8"
      />
    </svg>
  );
}
