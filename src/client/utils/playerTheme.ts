export type ColorVisionMode = 'normal' | 'deuteranopia' | 'protanopia' | 'tritanopia';

export type PlayerTheme = {
  ring: string;
  ringBorder: string;
  marker: string;
  territory: string;
  card: string;
  hex: string;
};

export const DEFAULT_PLAYER_THEME: PlayerTheme = {
  ring: 'bg-slate-300',
  ringBorder: 'border-slate-100',
  marker: 'border-slate-300',
  territory: 'bg-slate-800/70',
  card: 'bg-slate-500',
  hex: '#64748b',
} as const;

/**
 * Player color palettes optimized for different color vision deficiencies.
 *
 * Normal: Standard emerald/sky/amber/fuchsia palette
 * Deuteranopia/Protanopia: Blue/orange high-contrast palette (avoids red-green confusion)
 * Tritanopia: Magenta/cyan palette (avoids blue-yellow confusion)
 */
export const PLAYER_COLOR_PALETTES: Record<
  ColorVisionMode,
  readonly [string, string, string, string]
> = {
  normal: ['#10b981', '#0ea5e9', '#f59e0b', '#d946ef'], // emerald-500, sky-500, amber-500, fuchsia-500
  deuteranopia: ['#2563eb', '#ea580c', '#0891b2', '#7c3aed'], // blue-600, orange-600, cyan-600, violet-600
  protanopia: ['#2563eb', '#ea580c', '#0891b2', '#7c3aed'], // Same as deuteranopia
  tritanopia: ['#db2777', '#06b6d4', '#84cc16', '#f97316'], // pink-600, cyan-500, lime-500, orange-500
} as const;

export const PLAYER_COLOR_CLASSES: Record<
  ColorVisionMode,
  {
    bg: readonly [string, string, string, string];
    text: readonly [string, string, string, string];
    border: readonly [string, string, string, string];
    ring: readonly [string, string, string, string];
  }
> = {
  normal: {
    bg: ['bg-emerald-500', 'bg-sky-500', 'bg-amber-500', 'bg-fuchsia-500'],
    text: ['text-emerald-500', 'text-sky-500', 'text-amber-500', 'text-fuchsia-500'],
    border: ['border-emerald-500', 'border-sky-500', 'border-amber-500', 'border-fuchsia-500'],
    ring: ['ring-emerald-500', 'ring-sky-500', 'ring-amber-500', 'ring-fuchsia-500'],
  },
  deuteranopia: {
    bg: ['bg-blue-600', 'bg-orange-600', 'bg-cyan-600', 'bg-violet-600'],
    text: ['text-blue-600', 'text-orange-600', 'text-cyan-600', 'text-violet-600'],
    border: ['border-blue-600', 'border-orange-600', 'border-cyan-600', 'border-violet-600'],
    ring: ['ring-blue-600', 'ring-orange-600', 'ring-cyan-600', 'ring-violet-600'],
  },
  protanopia: {
    bg: ['bg-blue-600', 'bg-orange-600', 'bg-cyan-600', 'bg-violet-600'],
    text: ['text-blue-600', 'text-orange-600', 'text-cyan-600', 'text-violet-600'],
    border: ['border-blue-600', 'border-orange-600', 'border-cyan-600', 'border-violet-600'],
    ring: ['ring-blue-600', 'ring-orange-600', 'ring-cyan-600', 'ring-violet-600'],
  },
  tritanopia: {
    bg: ['bg-pink-600', 'bg-cyan-500', 'bg-lime-500', 'bg-orange-500'],
    text: ['text-pink-600', 'text-cyan-500', 'text-lime-500', 'text-orange-500'],
    border: ['border-pink-600', 'border-cyan-500', 'border-lime-500', 'border-orange-500'],
    ring: ['ring-pink-600', 'ring-cyan-500', 'ring-lime-500', 'ring-orange-500'],
  },
} as const;

export const PLAYER_THEMES: Record<ColorVisionMode, Record<number, PlayerTheme>> = {
  normal: {
    1: {
      ring: 'bg-emerald-400',
      ringBorder: 'border-emerald-200',
      marker: 'border-emerald-400',
      territory: 'bg-emerald-700/85',
      card: 'bg-emerald-500',
      hex: PLAYER_COLOR_PALETTES.normal[0],
    },
    2: {
      ring: 'bg-sky-600',
      ringBorder: 'border-sky-300',
      marker: 'border-sky-500',
      territory: 'bg-sky-700/85',
      card: 'bg-sky-500',
      hex: PLAYER_COLOR_PALETTES.normal[1],
    },
    3: {
      ring: 'bg-amber-400',
      ringBorder: 'border-amber-200',
      marker: 'border-amber-400',
      territory: 'bg-amber-600/85',
      card: 'bg-amber-500',
      hex: PLAYER_COLOR_PALETTES.normal[2],
    },
    4: {
      ring: 'bg-fuchsia-400',
      ringBorder: 'border-fuchsia-200',
      marker: 'border-fuchsia-400',
      territory: 'bg-fuchsia-700/85',
      card: 'bg-fuchsia-500',
      hex: PLAYER_COLOR_PALETTES.normal[3],
    },
  },
  deuteranopia: {
    1: {
      ring: 'bg-blue-500',
      ringBorder: 'border-blue-200',
      marker: 'border-blue-400',
      territory: 'bg-blue-800/85',
      card: 'bg-blue-600',
      hex: PLAYER_COLOR_PALETTES.deuteranopia[0],
    },
    2: {
      ring: 'bg-orange-500',
      ringBorder: 'border-orange-200',
      marker: 'border-orange-400',
      territory: 'bg-orange-800/85',
      card: 'bg-orange-600',
      hex: PLAYER_COLOR_PALETTES.deuteranopia[1],
    },
    3: {
      ring: 'bg-cyan-500',
      ringBorder: 'border-cyan-200',
      marker: 'border-cyan-400',
      territory: 'bg-cyan-800/85',
      card: 'bg-cyan-600',
      hex: PLAYER_COLOR_PALETTES.deuteranopia[2],
    },
    4: {
      ring: 'bg-violet-500',
      ringBorder: 'border-violet-200',
      marker: 'border-violet-400',
      territory: 'bg-violet-800/85',
      card: 'bg-violet-600',
      hex: PLAYER_COLOR_PALETTES.deuteranopia[3],
    },
  },
  protanopia: {
    1: {
      ring: 'bg-blue-500',
      ringBorder: 'border-blue-200',
      marker: 'border-blue-400',
      territory: 'bg-blue-800/85',
      card: 'bg-blue-600',
      hex: PLAYER_COLOR_PALETTES.protanopia[0],
    },
    2: {
      ring: 'bg-orange-500',
      ringBorder: 'border-orange-200',
      marker: 'border-orange-400',
      territory: 'bg-orange-800/85',
      card: 'bg-orange-600',
      hex: PLAYER_COLOR_PALETTES.protanopia[1],
    },
    3: {
      ring: 'bg-cyan-500',
      ringBorder: 'border-cyan-200',
      marker: 'border-cyan-400',
      territory: 'bg-cyan-800/85',
      card: 'bg-cyan-600',
      hex: PLAYER_COLOR_PALETTES.protanopia[2],
    },
    4: {
      ring: 'bg-violet-500',
      ringBorder: 'border-violet-200',
      marker: 'border-violet-400',
      territory: 'bg-violet-800/85',
      card: 'bg-violet-600',
      hex: PLAYER_COLOR_PALETTES.protanopia[3],
    },
  },
  tritanopia: {
    1: {
      ring: 'bg-pink-500',
      ringBorder: 'border-pink-200',
      marker: 'border-pink-400',
      territory: 'bg-pink-800/85',
      card: 'bg-pink-600',
      hex: PLAYER_COLOR_PALETTES.tritanopia[0],
    },
    2: {
      ring: 'bg-cyan-400',
      ringBorder: 'border-cyan-200',
      marker: 'border-cyan-400',
      territory: 'bg-cyan-800/85',
      card: 'bg-cyan-500',
      hex: PLAYER_COLOR_PALETTES.tritanopia[1],
    },
    3: {
      ring: 'bg-lime-400',
      ringBorder: 'border-lime-200',
      marker: 'border-lime-400',
      territory: 'bg-lime-800/85',
      card: 'bg-lime-500',
      hex: PLAYER_COLOR_PALETTES.tritanopia[2],
    },
    4: {
      ring: 'bg-orange-400',
      ringBorder: 'border-orange-200',
      marker: 'border-orange-400',
      territory: 'bg-orange-800/85',
      card: 'bg-orange-500',
      hex: PLAYER_COLOR_PALETTES.tritanopia[3],
    },
  },
} as const;

function clampPlayerIndex(playerIndex: number): 0 | 1 | 2 | 3 {
  if (!Number.isFinite(playerIndex)) return 0;
  const index = Math.floor(playerIndex);
  if (index <= 0) return 0;
  if (index >= 3) return 3;
  return index as 0 | 1 | 2 | 3;
}

function clampPlayerNumber(playerNumber: number): 1 | 2 | 3 | 4 {
  if (!Number.isFinite(playerNumber)) return 1;
  const value = Math.floor(playerNumber);
  if (value <= 1) return 1;
  if (value >= 4) return 4;
  return value as 1 | 2 | 3 | 4;
}

export function getPlayerColorHex(playerIndex: number, mode: ColorVisionMode): string {
  const palette = PLAYER_COLOR_PALETTES[mode] ?? PLAYER_COLOR_PALETTES.normal;
  return palette[clampPlayerIndex(playerIndex)];
}

export function getPlayerColorClass(
  playerIndex: number,
  mode: ColorVisionMode,
  type: keyof (typeof PLAYER_COLOR_CLASSES)['normal']
): string {
  const palette = PLAYER_COLOR_CLASSES[mode] ?? PLAYER_COLOR_CLASSES.normal;
  return palette[type][clampPlayerIndex(playerIndex)];
}

export function getPlayerTheme(
  playerNumber: number | undefined,
  mode: ColorVisionMode
): PlayerTheme {
  if (!playerNumber) {
    return DEFAULT_PLAYER_THEME;
  }
  const safePlayer = clampPlayerNumber(playerNumber);
  return (
    PLAYER_THEMES[mode]?.[safePlayer] ?? PLAYER_THEMES.normal[safePlayer] ?? DEFAULT_PLAYER_THEME
  );
}

export function getPlayerIndicatorPatternClass(playerNumber: number): string {
  const safePlayerNumber = Number.isFinite(playerNumber)
    ? Math.max(1, Math.floor(playerNumber))
    : 1;
  const index = ((safePlayerNumber - 1) % 4) as 0 | 1 | 2 | 3;
  return `player-indicator-pattern-${index}`;
}

export function isNonNormalColorVisionMode(mode: ColorVisionMode): boolean {
  return mode !== 'normal';
}
