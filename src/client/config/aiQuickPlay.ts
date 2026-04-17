import { BoardType } from '../../shared/types/game';

export type DifficultyTier = 'easy' | 'medium' | 'hard' | 'expert';

/**
 * Four heuristic personas (C2 / #80). Each is a ~51-weight delta
 * applied to the ladder's baseline heuristic profile on the Python
 * side. Same tactical engine, same think-time, visibly different play.
 * Kept mirrored with AIServiceClient.PersonaId on the server.
 */
export type PersonaId = 'balanced' | 'aggressive' | 'territorial' | 'defensive';

export const ALL_PERSONAS: readonly PersonaId[] = [
  'balanced',
  'aggressive',
  'territorial',
  'defensive',
] as const;

/**
 * Designer-facing descriptions — kept short enough for a UI tooltip.
 * Derived from the weight deltas in app.ai.heuristic_weights.py so
 * the UX label matches the actual play behaviour.
 */
export const PERSONA_COPY: Record<PersonaId, { label: string; tagline: string; detail: string }> = {
  balanced: {
    label: 'Balanced',
    tagline: 'Neutral, all-round play',
    detail: 'The reference opponent. Weighs captures, territory, defence, and mobility evenly.',
  },
  aggressive: {
    label: 'Aggressive',
    tagline: 'Favours captures and overtakes',
    detail:
      'Values eliminations and overtakes ~25% higher and risks ~15% more vulnerability than Balanced.',
  },
  territorial: {
    label: 'Territorial',
    tagline: 'Favours board control and closure',
    detail: 'Values territory and closure ~25% higher; less eager to trade rings for captures.',
  },
  defensive: {
    label: 'Defensive',
    tagline: 'Risk-averse, mobility-focused',
    detail: 'Weighs vulnerability ~20% higher and prefers mobile positions; slower to engage.',
  },
};

export interface AIQuickPlayOption {
  id: string;
  boardType: BoardType;
  numPlayers: number;
  difficultyTier: DifficultyTier;
  difficulty: number; // 1-10 ladder level
  displayName: string;
  description: string;
  estimatedElo: number; // Legacy model-training estimate; do not present as human Elo.
  /**
   * Optional persona selector (C2 phase 3). When unset, the server
   * uses the ladder's default heuristic profile. Multiplying the base
   * grid by personas is intentionally done at game-creation time
   * rather than by expanding the preset array so URLs, analytics, and
   * ladder test harnesses stay stable.
   */
  personaId?: PersonaId;
}

/**
 * Map difficulty ladder level to AI type for game creation.
 * Based on AI_DIFFICULTY_PRESETS in AIEngine.ts.
 */
export function getDifficultyAiType(difficulty: number): string {
  if (difficulty <= 1) return 'random';
  if (difficulty <= 2) return 'heuristic';
  if (difficulty <= 4) return 'minimax';
  if (difficulty <= 6) return 'descent';
  if (difficulty <= 8) return 'mcts';
  return 'gumbel_mcts';
}

/** Board display names for UI */
export const BOARD_DISPLAY_NAMES: Record<BoardType, string> = {
  square8: 'Square 8×8',
  square19: 'Square 19×19',
  hex8: 'Hex (Small)',
  hexagonal: 'Hexagonal',
};

/** Tier color classes for UI styling */
export const TIER_COLORS: Record<DifficultyTier, { border: string; bg: string; text: string }> = {
  easy: {
    border: 'border-green-500/50',
    bg: 'bg-green-500/10 hover:bg-green-500/20',
    text: 'text-green-400',
  },
  medium: {
    border: 'border-yellow-500/50',
    bg: 'bg-yellow-500/10 hover:bg-yellow-500/20',
    text: 'text-yellow-400',
  },
  hard: {
    border: 'border-orange-500/50',
    bg: 'bg-orange-500/10 hover:bg-orange-500/20',
    text: 'text-orange-400',
  },
  expert: {
    border: 'border-red-500/50',
    bg: 'bg-red-500/10 hover:bg-red-500/20',
    text: 'text-red-400',
  },
};

// Helper to generate options for a single board/player config
function makeOptions(
  boardType: BoardType,
  numPlayers: number,
  prefix: string
): AIQuickPlayOption[] {
  return [
    {
      id: `${prefix}-easy`,
      boardType,
      numPlayers,
      difficultyTier: 'easy',
      difficulty: 2,
      displayName: 'Easy',
      description: 'Learning the basics',
      estimatedElo: 600,
    },
    {
      id: `${prefix}-medium`,
      boardType,
      numPlayers,
      difficultyTier: 'medium',
      difficulty: 4,
      displayName: 'Medium',
      description: 'Fair challenge',
      estimatedElo: 900,
    },
    {
      id: `${prefix}-hard`,
      boardType,
      numPlayers,
      difficultyTier: 'hard',
      difficulty: 7,
      displayName: 'Hard',
      description: 'Strong opponent',
      estimatedElo: 1200,
    },
    {
      id: `${prefix}-expert`,
      boardType,
      numPlayers,
      difficultyTier: 'expert',
      difficulty: 9,
      displayName: 'Expert',
      description: 'Near-optimal play',
      estimatedElo: 1500,
    },
  ];
}

/**
 * All available AI quick-play options.
 * 4 boards × 3 player counts × 4 difficulty tiers = 48 options.
 */
export const AI_QUICK_PLAY_OPTIONS: AIQuickPlayOption[] = [
  // Square 8×8
  ...makeOptions('square8', 2, 'sq8-2p'),
  ...makeOptions('square8', 3, 'sq8-3p'),
  ...makeOptions('square8', 4, 'sq8-4p'),
  // Square 19×19
  ...makeOptions('square19', 2, 'sq19-2p'),
  ...makeOptions('square19', 3, 'sq19-3p'),
  ...makeOptions('square19', 4, 'sq19-4p'),
  // Hex (Small)
  ...makeOptions('hex8', 2, 'hex8-2p'),
  ...makeOptions('hex8', 3, 'hex8-3p'),
  ...makeOptions('hex8', 4, 'hex8-4p'),
  // Hexagonal (Large)
  ...makeOptions('hexagonal', 2, 'hexl-2p'),
  ...makeOptions('hexagonal', 3, 'hexl-3p'),
  ...makeOptions('hexagonal', 4, 'hexl-4p'),
];

/**
 * Filter options by board type and player count.
 */
export function getOptionsForConfig(boardType: BoardType, numPlayers: number): AIQuickPlayOption[] {
  return AI_QUICK_PLAY_OPTIONS.filter(
    (opt) => opt.boardType === boardType && opt.numPlayers === numPlayers
  );
}
