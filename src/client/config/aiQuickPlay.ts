import { BoardType } from '../../shared/types/game';

export type DifficultyTier = 'easy' | 'medium' | 'hard' | 'expert';

export interface AIQuickPlayOption {
  id: string;
  boardType: BoardType;
  numPlayers: number;
  difficultyTier: DifficultyTier;
  difficulty: number; // 1-10 ladder level
  displayName: string;
  description: string;
  estimatedElo: number; // Legacy model-training estimate; do not present as human Elo.
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
