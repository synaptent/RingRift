/**
 * Tutorial Hints Hook
 *
 * Manages contextual hints for the "Learn the Basics" tutorial mode.
 * Triggers hints when entering new game phases for the first time.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { GameState, GamePhase } from '../../shared/types/game';
import type { TeachingTopic } from '../components/TeachingOverlay';

export interface TutorialHint {
  /** Game phase that triggered this hint */
  phase: GamePhase;
  /** Title shown in the hint banner */
  title: string;
  /** Main message explaining what to do */
  message: string;
  /** Teaching topic to open when "Learn More" is clicked */
  topic: TeachingTopic;
  /** Icon for the hint */
  icon: string;
}

/** Maps game phases to their corresponding teaching topics */
const PHASE_TO_TOPIC: Partial<Record<GamePhase, TeachingTopic>> = {
  ring_placement: 'ring_placement',
  movement: 'stack_movement',
  capture: 'capturing',
  chain_capture: 'chain_capture',
  line_processing: 'line_bonus',
  territory_processing: 'territory',
  forced_elimination: 'forced_elimination',
};

/** Hint content for each phase */
const PHASE_HINTS: Partial<Record<GamePhase, Omit<TutorialHint, 'phase' | 'topic'>>> = {
  ring_placement: {
    title: 'Place Your Rings',
    message:
      'Click any empty cell to place up to 3 rings, or click an existing stack to add 1 ring on top. You cannot place if it would leave you with no legal moves. Place adjacent to your own rings to build connected groups for territory. You have limited rings - use them wisely!',
    icon: '🎯',
  },
  movement: {
    title: 'Move Your Stacks',
    message:
      'Click your stack, then a highlighted destination. Move in a straight line in any available direction at least as many spaces as your stack is tall (height 2 = move 2+ spaces). Collapsed territory blocks your path, and rings or stacks cannot be landed on and can only be jumped over and overtaken (capturing the top ring), but you can land on markers (this eliminates your top ring) or any empty space. Any markers you travel over will flip to your color, or if they are of your color they will be turned into a collapsed space claimed by you.',
    icon: '↗️',
  },
  capture: {
    title: 'Capture!',
    message:
      "Jump over any adjacent stack to capture it. Your stack's cap (consecutive top rings of your color) must be at least as tall as the target stack. Land on empty space or marker beyond. The captured top ring joins the bottom of your stack. Capturing is optional, but once started, you MUST continue any available chain captures.",
    icon: '⚔️',
  },
  chain_capture: {
    title: 'Chain Capture!',
    message:
      'Once you start capturing, you MUST continue until no more captures are possible. Choose your direction when multiple targets exist. Plan ahead - you cannot stop mid-chain!',
    icon: '⇉',
  },
  line_processing: {
    title: 'Line Formed!',
    message:
      'Your markers formed a line! Line length varies by board and player count (4+ for 2-player on 8×8, 3+ for 3-4 player). The line collapses into territory. For exact-length lines, eliminate 1 ring from any controlled stack. For overlength lines, choose: collapse all (costs 1 ring) or collapse minimum length (free).',
    icon: '━',
  },
  territory_processing: {
    title: 'Claim Territory',
    message:
      'A disconnected region has formed! To claim it, you must eliminate your ENTIRE CAP from any controlled stack OUTSIDE the region (including height-1 standalone rings). All interior rings are credited to you. Territory spaces count toward victory and block movement.',
    icon: '🏰',
  },
  forced_elimination: {
    title: 'No Legal Moves',
    message:
      'You have stacks but no legal moves available. You must eliminate your ENTIRE CAP from one controlled stack (including height-1 standalone rings). This keeps the game moving and counts toward ring elimination totals.',
    icon: '💎',
  },
};

export interface UseTutorialHintsOptions {
  /** Current game state */
  gameState: GameState | null;
  /** Whether tutorial mode is active */
  isLearnMode: boolean;
  /** Phases that have already been seen */
  seenPhases: string[];
  /** Whether hints are enabled by user preference */
  hintsEnabled: boolean;
}

export interface UseTutorialHintsResult {
  /** Current hint to display, or null if none */
  currentHint: TutorialHint | null;
  /** Dismiss the current hint */
  dismissHint: () => void;
  /** Get the teaching topic for the current hint (for "Learn More") */
  getTeachingTopic: () => TeachingTopic | null;
}

/**
 * Hook that manages tutorial hints based on game phase changes.
 *
 * Shows hints when entering a game phase for the first time in learn mode.
 * Hints can be dismissed and won't reappear for the same phase.
 */
export function useTutorialHints({
  gameState,
  isLearnMode,
  seenPhases,
  hintsEnabled,
}: UseTutorialHintsOptions): UseTutorialHintsResult {
  const [currentHint, setCurrentHint] = useState<TutorialHint | null>(null);
  const [dismissedPhase, setDismissedPhase] = useState<GamePhase | null>(null);
  const lastPhaseRef = useRef<GamePhase | null>(null);

  // Track phase changes and trigger hints
  useEffect(() => {
    if (!isLearnMode || !hintsEnabled || !gameState) {
      setCurrentHint(null);
      return;
    }

    const currentPhase = gameState.currentPhase;

    // Skip if same phase as before or already dismissed this phase
    if (currentPhase === lastPhaseRef.current || currentPhase === dismissedPhase) {
      return;
    }

    // Skip game_over phase
    if (currentPhase === 'game_over') {
      setCurrentHint(null);
      return;
    }

    lastPhaseRef.current = currentPhase;

    // Check if this phase has already been seen
    if (seenPhases.includes(currentPhase)) {
      setCurrentHint(null);
      return;
    }

    // Get hint content for this phase
    const hintContent = PHASE_HINTS[currentPhase];
    const topic = PHASE_TO_TOPIC[currentPhase];

    if (!hintContent || !topic) {
      setCurrentHint(null);
      return;
    }

    // Show the hint
    setCurrentHint({
      phase: currentPhase,
      topic,
      ...hintContent,
    });
  }, [gameState, isLearnMode, seenPhases, hintsEnabled, dismissedPhase]);

  // Clear dismissed phase when phase changes
  useEffect(() => {
    if (gameState && gameState.currentPhase !== dismissedPhase) {
      setDismissedPhase(null);
    }
  }, [gameState, dismissedPhase]);

  const dismissHint = useCallback(() => {
    if (currentHint) {
      setDismissedPhase(currentHint.phase);
    }
    setCurrentHint(null);
  }, [currentHint]);

  const getTeachingTopic = useCallback(() => {
    return currentHint?.topic ?? null;
  }, [currentHint]);

  return {
    currentHint,
    dismissHint,
    getTeachingTopic,
  };
}
