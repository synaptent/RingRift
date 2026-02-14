/**
 * Small inline SVG illustrations for the onboarding modal steps.
 * Uses the game's visual language: hex cells, colored ring stacks, territory fills.
 *
 * Player colors (from BoardView.tsx):
 *   Player 1: #34d399 (emerald-400)
 *   Player 2: #0284c7 (sky-600)
 *   Player 3: #fbbf24 (amber-400)
 *   Player 4: #e879f9 (fuchsia-400)
 */

const HEX_R = 18;

function hexPoints(cx: number, cy: number, r = HEX_R) {
  return Array.from({ length: 6 }, (_, i) => {
    const angle = (Math.PI / 180) * (60 * i - 30);
    return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
  }).join(' ');
}

function HexCell({ cx, cy, highlight }: { cx: number; cy: number; highlight?: boolean }) {
  return (
    <polygon
      points={hexPoints(cx, cy)}
      fill={highlight ? 'rgba(52, 211, 153, 0.2)' : 'rgba(51, 65, 85, 0.5)'}
      stroke={highlight ? '#34d399' : 'rgb(71, 85, 105)'}
      strokeWidth="1.2"
    />
  );
}

function Ring({ cx, cy, color, r = 7 }: { cx: number; cy: number; color: string; r?: number }) {
  return <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="3" />;
}

/** Welcome step: mini hex board with colorful ring stacks */
export function WelcomeIllustration() {
  return (
    <svg viewBox="0 0 160 100" className="w-40 h-24 mx-auto" aria-hidden="true">
      {/* 5-cell hex cluster */}
      <HexCell cx={55} cy={50} />
      <HexCell cx={80} cy={36} />
      <HexCell cx={80} cy={64} />
      <HexCell cx={105} cy={50} highlight />
      <HexCell cx={105} cy={22} />

      {/* Ring stacks */}
      <Ring cx={55} cy={50} color="#0284c7" />
      <Ring cx={80} cy={36} color="#34d399" />
      <Ring cx={80} cy={60} color="#fbbf24" />
      <Ring cx={80} cy={68} color="#34d399" />
      <Ring cx={105} cy={50} color="#e879f9" />
      <Ring cx={105} cy={18} color="#0284c7" />
      <Ring cx={105} cy={26} color="#34d399" />
    </svg>
  );
}

/** Phase illustrations: placement, movement, capture as compact vignettes */
export function PlacementIllustration() {
  return (
    <svg viewBox="0 0 48 48" className="w-10 h-10 flex-shrink-0" aria-hidden="true">
      <rect
        x="4"
        y="4"
        width="40"
        height="40"
        rx="4"
        fill="rgba(51, 65, 85, 0.4)"
        stroke="rgb(71, 85, 105)"
        strokeWidth="1"
      />
      {/* Target cell highlight */}
      <rect
        x="15"
        y="15"
        width="18"
        height="18"
        rx="2"
        fill="rgba(52, 211, 153, 0.2)"
        stroke="#34d399"
        strokeWidth="1"
        strokeDasharray="3 2"
      />
      {/* Ring being placed */}
      <Ring cx={24} cy={24} color="#34d399" r={6} />
      {/* Down arrow */}
      <line x1="24" y1="6" x2="24" y2="14" stroke="#34d399" strokeWidth="1.2" opacity={0.6} />
      <polyline
        points="21,12 24,15 27,12"
        fill="none"
        stroke="#34d399"
        strokeWidth="1.2"
        opacity={0.6}
      />
    </svg>
  );
}

export function MovementIllustration() {
  return (
    <svg viewBox="0 0 48 48" className="w-10 h-10 flex-shrink-0" aria-hidden="true">
      <rect
        x="4"
        y="4"
        width="40"
        height="40"
        rx="4"
        fill="rgba(51, 65, 85, 0.4)"
        stroke="rgb(71, 85, 105)"
        strokeWidth="1"
      />
      {/* Start position */}
      <Ring cx={14} cy={24} color="#0284c7" r={5} />
      {/* Movement arrow */}
      <line
        x1="21"
        y1="24"
        x2="31"
        y2="24"
        stroke="#0284c7"
        strokeWidth="1.5"
        strokeDasharray="3 2"
      />
      <polyline points="29,21 33,24 29,27" fill="none" stroke="#0284c7" strokeWidth="1.5" />
      {/* End position (ghost) */}
      <circle
        cx="36"
        cy="24"
        r="5"
        fill="none"
        stroke="#0284c7"
        strokeWidth="2"
        opacity={0.4}
        strokeDasharray="2 2"
      />
    </svg>
  );
}

export function CaptureIllustration() {
  return (
    <svg viewBox="0 0 48 48" className="w-10 h-10 flex-shrink-0" aria-hidden="true">
      <rect
        x="4"
        y="4"
        width="40"
        height="40"
        rx="4"
        fill="rgba(51, 65, 85, 0.4)"
        stroke="rgb(71, 85, 105)"
        strokeWidth="1"
      />
      {/* Attacker */}
      <Ring cx={12} cy={24} color="#34d399" r={5} />
      {/* Target (opponent) */}
      <Ring cx={24} cy={24} color="#fbbf24" r={5} />
      {/* Jump arrow over target */}
      <path d="M18,20 Q24,12 30,20" fill="none" stroke="#34d399" strokeWidth="1.5" />
      <polyline points="28,18 31,20 28,22" fill="none" stroke="#34d399" strokeWidth="1.2" />
      {/* Landing spot */}
      <circle
        cx="36"
        cy="24"
        r="5"
        fill="none"
        stroke="#34d399"
        strokeWidth="2"
        opacity={0.4}
        strokeDasharray="2 2"
      />
    </svg>
  );
}

/** Victory condition icons */
export function EliminationIcon() {
  return (
    <svg viewBox="0 0 40 40" className="w-8 h-8 flex-shrink-0" aria-hidden="true">
      {/* Broken ring */}
      <path
        d="M20,8 A12,12 0 1,1 12,28"
        fill="none"
        stroke="#fbbf24"
        strokeWidth="3"
        strokeLinecap="round"
      />
      {/* Scatter fragments */}
      <line
        x1="10"
        y1="28"
        x2="6"
        y2="33"
        stroke="#fbbf24"
        strokeWidth="2"
        opacity={0.5}
        strokeLinecap="round"
      />
      <line
        x1="13"
        y1="30"
        x2="11"
        y2="35"
        stroke="#fbbf24"
        strokeWidth="1.5"
        opacity={0.4}
        strokeLinecap="round"
      />
    </svg>
  );
}

export function TerritoryIcon() {
  return (
    <svg viewBox="0 0 40 40" className="w-8 h-8 flex-shrink-0" aria-hidden="true">
      {/* Filled territory cells */}
      <rect
        x="4"
        y="4"
        width="14"
        height="14"
        rx="2"
        fill="rgba(4, 120, 87, 0.5)"
        stroke="#059669"
        strokeWidth="1.2"
      />
      <rect
        x="20"
        y="4"
        width="14"
        height="14"
        rx="2"
        fill="rgba(4, 120, 87, 0.5)"
        stroke="#059669"
        strokeWidth="1.2"
      />
      <rect
        x="4"
        y="20"
        width="14"
        height="14"
        rx="2"
        fill="rgba(4, 120, 87, 0.5)"
        stroke="#059669"
        strokeWidth="1.2"
      />
      <rect
        x="20"
        y="20"
        width="14"
        height="14"
        rx="2"
        fill="rgba(51, 65, 85, 0.5)"
        stroke="rgb(71, 85, 105)"
        strokeWidth="1.2"
      />
      {/* Ownership ring */}
      <Ring cx={11} cy={11} color="#34d399" r={4} />
    </svg>
  );
}

export function LastStandingIcon() {
  return (
    <svg viewBox="0 0 40 40" className="w-8 h-8 flex-shrink-0" aria-hidden="true">
      {/* Single standing ring */}
      <Ring cx={20} cy={20} color="#34d399" r={8} />
      {/* Faded eliminated rings */}
      <Ring cx={8} cy={12} color="#94a3b8" r={4} />
      <Ring cx={32} cy={12} color="#94a3b8" r={4} />
      <Ring cx={8} cy={32} color="#94a3b8" r={4} />
    </svg>
  );
}

/** Ready to play: board with highlighted "your turn" cell */
export function ReadyToPlayIllustration() {
  return (
    <svg viewBox="0 0 120 80" className="w-32 h-20 mx-auto" aria-hidden="true">
      {/* Small 3-hex cluster */}
      <HexCell cx={40} cy={40} />
      <HexCell cx={65} cy={28} highlight />
      <HexCell cx={65} cy={52} />
      <HexCell cx={90} cy={40} />

      {/* Rings on some cells */}
      <Ring cx={40} cy={40} color="#0284c7" />
      <Ring cx={65} cy={52} color="#fbbf24" />
      <Ring cx={90} cy={40} color="#34d399" />

      {/* "Your turn" cursor on highlighted cell */}
      <Ring cx={65} cy={28} color="#34d399" r={5} />
      <circle
        cx={65}
        cy={28}
        r={10}
        fill="none"
        stroke="#34d399"
        strokeWidth="1"
        opacity={0.4}
        strokeDasharray="3 2"
      />
    </svg>
  );
}
