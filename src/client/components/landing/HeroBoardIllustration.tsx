/**
 * Decorative hex board illustration for the landing page hero section.
 * Shows a small cluster of ~7 hex cells with colored ring stacks,
 * giving visitors an instant visual taste of the game.
 *
 * Uses exact player colors from BoardView.tsx:
 *   Player 1: emerald-400 (#34d399)
 *   Player 2: sky-600    (#0284c7)
 *   Player 3: amber-400  (#fbbf24)
 *   Player 4: fuchsia-400(#e879f9)
 */
export function HeroBoardIllustration() {
  // Hex cell positions for a small flower cluster (center + 6 around)
  // pointy-top hex, spacing ~52px horizontal, ~45px vertical offset
  const cells: Array<{
    cx: number;
    cy: number;
    rings?: Array<{ color: string; className?: string }>;
  }> = [
    { cx: 130, cy: 120 }, // center
    { cx: 130, cy: 72, rings: [{ color: '#34d399' }] }, // top
    {
      cx: 175,
      cy: 96,
      rings: [{ color: '#0284c7' }, { color: '#34d399', className: 'hero-pulse' }],
    }, // top-right
    { cx: 175, cy: 144, rings: [{ color: '#fbbf24' }] }, // bottom-right
    { cx: 130, cy: 168 }, // bottom
    { cx: 85, cy: 144, rings: [{ color: '#34d399' }, { color: '#e879f9' }] }, // bottom-left
    { cx: 85, cy: 96, rings: [{ color: '#0284c7', className: 'hero-pulse-delayed' }] }, // top-left
  ];

  const hexR = 24; // hex radius
  const hexPoints = (cx: number, cy: number) => {
    // pointy-top hexagon
    return Array.from({ length: 6 }, (_, i) => {
      const angle = (Math.PI / 180) * (60 * i - 30);
      return `${cx + hexR * Math.cos(angle)},${cy + hexR * Math.sin(angle)}`;
    }).join(' ');
  };

  return (
    <svg
      viewBox="0 0 260 240"
      className="w-full max-w-[280px] sm:max-w-[320px] h-auto"
      aria-hidden="true"
    >
      <defs>
        <filter id="hero-glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Hex cells */}
      {cells.map((cell, i) => (
        <g key={i}>
          <polygon
            points={hexPoints(cell.cx, cell.cy)}
            fill="rgba(51, 65, 85, 0.5)"
            stroke="rgb(71, 85, 105)"
            strokeWidth="1.5"
          />
          {/* Ring stacks */}
          {cell.rings?.map((ring, ri) => {
            const yOff = cell.cy - (cell.rings!.length - 1) * 5 + ri * 10;
            return (
              <circle
                key={ri}
                cx={cell.cx}
                cy={yOff}
                r={9 - ri * 0.5}
                fill="none"
                stroke={ring.color}
                strokeWidth="4"
                className={ring.className}
                filter={ring.className ? 'url(#hero-glow)' : undefined}
                opacity={0.9}
              />
            );
          })}
        </g>
      ))}

      <style>{`
        .hero-pulse {
          animation: hero-ring-pulse 3s ease-in-out infinite;
        }
        .hero-pulse-delayed {
          animation: hero-ring-pulse 3s ease-in-out 1.5s infinite;
        }
        @keyframes hero-ring-pulse {
          0%, 100% { opacity: 0.9; }
          50% { opacity: 0.5; }
        }
        @media (prefers-reduced-motion: reduce) {
          .hero-pulse, .hero-pulse-delayed {
            animation: none;
          }
        }
      `}</style>
    </svg>
  );
}
