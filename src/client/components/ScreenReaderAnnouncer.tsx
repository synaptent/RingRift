import React, { useEffect, useState, useRef, useCallback } from 'react';

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Announcement priority levels.
 * Higher priority announcements will be queued before lower priority ones
 * and may use assertive mode for immediate interruption.
 */
export type AnnouncementPriority = 'high' | 'medium' | 'low';

/**
 * Announcement categories for game events.
 * Each category has default priority and politeness settings.
 */
export type AnnouncementCategory =
  | 'victory' // Game over announcements - highest priority, assertive
  | 'defeat' // Player eliminated - high priority, assertive
  | 'your_turn' // It's the player's turn - high priority, polite
  | 'turn_change' // Someone else's turn - medium priority, polite
  | 'phase_transition' // Phase changes - medium priority, polite
  | 'move' // Move made by any player - low priority, polite
  | 'capture' // Capture occurred - medium priority, polite
  | 'line_formed' // Line completed - medium priority, polite
  | 'territory' // Territory claimed - medium priority, polite
  | 'timer_warning' // Timer running low - high priority, assertive
  | 'selection' // Cell/piece selection - low priority, polite
  | 'error' // Error message - high priority, assertive
  | 'info'; // General information - low priority, polite

/**
 * Configuration for each announcement category.
 */
interface CategoryConfig {
  priority: AnnouncementPriority;
  politeness: 'polite' | 'assertive';
  /** Minimum time in ms between announcements of this category (debounce) */
  debounceMs: number;
}

const CATEGORY_CONFIG: Record<AnnouncementCategory, CategoryConfig> = {
  victory: { priority: 'high', politeness: 'assertive', debounceMs: 0 },
  defeat: { priority: 'high', politeness: 'assertive', debounceMs: 0 },
  your_turn: { priority: 'high', politeness: 'polite', debounceMs: 500 },
  turn_change: { priority: 'medium', politeness: 'polite', debounceMs: 500 },
  phase_transition: { priority: 'medium', politeness: 'polite', debounceMs: 300 },
  move: { priority: 'low', politeness: 'polite', debounceMs: 200 },
  capture: { priority: 'medium', politeness: 'polite', debounceMs: 200 },
  line_formed: { priority: 'medium', politeness: 'polite', debounceMs: 300 },
  territory: { priority: 'medium', politeness: 'polite', debounceMs: 300 },
  timer_warning: { priority: 'high', politeness: 'assertive', debounceMs: 5000 },
  selection: { priority: 'low', politeness: 'polite', debounceMs: 100 },
  error: { priority: 'high', politeness: 'assertive', debounceMs: 0 },
  info: { priority: 'low', politeness: 'polite', debounceMs: 200 },
};

/**
 * A queued announcement with metadata.
 */
export interface QueuedAnnouncement {
  id: string;
  message: string;
  category: AnnouncementCategory;
  priority: AnnouncementPriority;
  politeness: 'polite' | 'assertive';
  timestamp: number;
}

export interface ScreenReaderAnnouncerProps {
  /**
   * Message to announce. When this changes, the new message is announced.
   * Set to empty string to clear the announcement.
   * @deprecated Use the useGameAnnouncements hook instead for new code.
   */
  message?: string;
  /**
   * Politeness level for the announcement.
   * - 'polite': Waits for user to finish their current task (default)
   * - 'assertive': Interrupts immediately (use sparingly for urgent info)
   */
  politeness?: 'polite' | 'assertive';
  /**
   * Queue of announcements to process. When using the priority queue,
   * provide this instead of message.
   */
  queue?: QueuedAnnouncement[];
  /**
   * Callback when an announcement is spoken (removed from queue).
   */
  onAnnouncementSpoken?: (id: string) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Generate announcements for common game events.
 */
export const GameAnnouncements = {
  /** Announce that it's a player's turn */
  turnChange: (playerName: string, isYourTurn: boolean): string => {
    return isYourTurn ? "It's your turn!" : `It's ${playerName}'s turn.`;
  },

  /** Announce a phase transition */
  phaseTransition: (phaseName: string, description?: string): string => {
    return description ? `${phaseName}. ${description}` : phaseName;
  },

  /** Announce a move */
  move: (
    playerName: string,
    fromPosition: string,
    toPosition: string,
    isYourMove: boolean
  ): string => {
    if (isYourMove) {
      return `You moved from ${fromPosition} to ${toPosition}.`;
    }
    return `${playerName} moved from ${fromPosition} to ${toPosition}.`;
  },

  /** Announce a ring placement */
  placement: (
    playerName: string,
    position: string,
    stackHeight: number,
    isYourMove: boolean
  ): string => {
    const stackInfo = stackHeight > 1 ? ` Stack height is now ${stackHeight}.` : '';
    if (isYourMove) {
      return `You placed a ring at ${position}.${stackInfo}`;
    }
    return `${playerName} placed a ring at ${position}.${stackInfo}`;
  },

  /** Announce a capture */
  capture: (
    playerName: string,
    capturedPosition: string,
    landingPosition: string,
    ringsGained: number,
    isYourCapture: boolean
  ): string => {
    const ringsInfo =
      ringsGained > 0 ? ` Gained ${ringsGained} ring${ringsGained !== 1 ? 's' : ''}.` : '';
    if (isYourCapture) {
      return `You captured at ${capturedPosition}, landing at ${landingPosition}.${ringsInfo}`;
    }
    return `${playerName} captured at ${capturedPosition}, landing at ${landingPosition}.${ringsInfo}`;
  },

  /** Announce a chain capture continuation */
  chainCapture: (positions: string[]): string => {
    return `Chain capture in progress. Path: ${positions.join(' to ')}.`;
  },

  /** Announce a line formation */
  lineFormed: (playerName: string, lineLength: number, isYourLine: boolean): string => {
    if (isYourLine) {
      return `You formed a line of ${lineLength}! Choose your reward.`;
    }
    return `${playerName} formed a line of ${lineLength}.`;
  },

  /** Announce territory claim */
  territoryClaimed: (
    playerName: string,
    spacesCount: number,
    totalTerritory: number,
    isYours: boolean
  ): string => {
    if (isYours) {
      return `You claimed ${spacesCount} territory space${spacesCount !== 1 ? 's' : ''}. Total: ${totalTerritory}.`;
    }
    return `${playerName} claimed ${spacesCount} territory space${spacesCount !== 1 ? 's' : ''}. Total: ${totalTerritory}.`;
  },

  /** Announce victory */
  victory: (
    winnerName: string,
    condition: 'elimination' | 'territory' | 'last_player_standing',
    isYou: boolean
  ): string => {
    const conditionText =
      condition === 'elimination'
        ? 'by ring elimination'
        : condition === 'territory'
          ? 'by territory control'
          : 'as last player standing';

    if (isYou) {
      return `Victory! You won the game ${conditionText}!`;
    }
    return `Game over. ${winnerName} won ${conditionText}.`;
  },

  /** Announce player elimination */
  playerEliminated: (playerName: string, isYou: boolean): string => {
    if (isYou) {
      return "You've been eliminated from the game.";
    }
    return `${playerName} has been eliminated.`;
  },

  /** Announce timer warning */
  timerWarning: (secondsRemaining: number): string => {
    if (secondsRemaining <= 10) {
      return `Warning! ${secondsRemaining} seconds remaining!`;
    }
    return `${secondsRemaining} seconds remaining.`;
  },

  /** Announce selection */
  cellSelected: (position: string, stackInfo?: string): string => {
    if (stackInfo) {
      return `Selected ${position}. ${stackInfo}`;
    }
    return `Selected ${position}.`;
  },

  /** Announce valid moves available */
  validMoves: (count: number): string => {
    if (count === 0) {
      return 'No valid moves available.';
    }
    return `${count} valid move${count !== 1 ? 's' : ''} available.`;
  },

  /** Announce decision required */
  decisionRequired: (decisionType: string, optionsCount: number): string => {
    return `Decision required: ${decisionType}. ${optionsCount} option${optionsCount !== 1 ? 's' : ''} available.`;
  },

  /** Announce ring statistics */
  ringStats: (playerName: string, inHand: number, onBoard: number, eliminated: number): string => {
    return `${playerName}: ${inHand} rings in hand, ${onBoard} on board, ${eliminated} eliminated.`;
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// Priority Comparison
// ═══════════════════════════════════════════════════════════════════════════

const PRIORITY_ORDER: Record<AnnouncementPriority, number> = {
  high: 3,
  medium: 2,
  low: 1,
};

function comparePriority(a: QueuedAnnouncement, b: QueuedAnnouncement): number {
  // Higher priority first
  const priorityDiff = PRIORITY_ORDER[b.priority] - PRIORITY_ORDER[a.priority];
  if (priorityDiff !== 0) return priorityDiff;
  // Same priority: earlier timestamp first
  return a.timestamp - b.timestamp;
}

// ═══════════════════════════════════════════════════════════════════════════
// Component
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Visually hidden component that announces messages to screen readers.
 *
 * Uses aria-live regions to communicate dynamic content changes.
 * The component is positioned off-screen but remains accessible to
 * assistive technology.
 *
 * Supports two modes:
 * 1. Simple mode: Pass a `message` prop for basic announcements
 * 2. Queue mode: Pass a `queue` prop for priority-based announcements
 *
 * Usage (Simple):
 * ```tsx
 * const [announcement, setAnnouncement] = useState('');
 * setAnnouncement(`It's now Player 2's turn`);
 * <ScreenReaderAnnouncer message={announcement} />
 * ```
 *
 * Usage (Queue):
 * ```tsx
 * const { queue, announce, removeAnnouncement } = useGameAnnouncements();
 * announce('Your turn!', 'your_turn');
 * <ScreenReaderAnnouncer queue={queue} onAnnouncementSpoken={removeAnnouncement} />
 * ```
 */
export function ScreenReaderAnnouncer({
  message,
  politeness = 'polite',
  queue,
  onAnnouncementSpoken,
}: ScreenReaderAnnouncerProps) {
  // Use alternating regions to ensure announcement is triggered on duplicate messages
  const [currentMessage, setCurrentMessage] = useState('');
  const [currentPoliteness, setCurrentPoliteness] = useState<'polite' | 'assertive'>('polite');
  const [isFirst, setIsFirst] = useState(true);
  const prevMessageRef = useRef<string>('');
  const isProcessingRef = useRef(false);

  // Process queue announcements
  useEffect(() => {
    if (!queue || queue.length === 0 || isProcessingRef.current) return;

    // Sort queue by priority
    const sortedQueue = [...queue].sort(comparePriority);
    const nextAnnouncement = sortedQueue[0];

    if (!nextAnnouncement) return;

    isProcessingRef.current = true;

    // Set the announcement
    setCurrentMessage(nextAnnouncement.message);
    setCurrentPoliteness(nextAnnouncement.politeness);
    setIsFirst((prev) => !prev);

    // Notify that announcement was spoken
    const timeoutId = setTimeout(() => {
      onAnnouncementSpoken?.(nextAnnouncement.id);
      isProcessingRef.current = false;
      // Clear message after speaking
      setCurrentMessage('');
    }, 1000);

    return () => {
      clearTimeout(timeoutId);
      isProcessingRef.current = false;
    };
  }, [queue, onAnnouncementSpoken]);

  // Handle simple message prop (backward compatibility)
  useEffect(() => {
    if (queue && queue.length > 0) return; // Queue mode takes precedence

    let timeout: ReturnType<typeof setTimeout> | undefined;

    if (message && message !== prevMessageRef.current) {
      setCurrentMessage(message);
      setCurrentPoliteness(politeness);
      setIsFirst((prev) => !prev);
      prevMessageRef.current = message;

      // Clear message after announcement to allow re-announcement of same message
      timeout = setTimeout(() => {
        setCurrentMessage('');
      }, 1000);
    }

    return () => {
      if (timeout !== undefined) {
        clearTimeout(timeout);
      }
    };
  }, [message, politeness, queue]);

  // Visually hidden but accessible to screen readers
  const hiddenStyle: React.CSSProperties = {
    position: 'absolute',
    width: '1px',
    height: '1px',
    padding: 0,
    margin: '-1px',
    overflow: 'hidden',
    clip: 'rect(0, 0, 0, 0)',
    whiteSpace: 'nowrap',
    border: 0,
  };

  return (
    <>
      {/* Polite region */}
      <div
        aria-live="polite"
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-polite-1"
      >
        {isFirst && currentPoliteness === 'polite' ? currentMessage : ''}
      </div>
      <div
        aria-live="polite"
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-polite-2"
      >
        {!isFirst && currentPoliteness === 'polite' ? currentMessage : ''}
      </div>

      {/* Assertive region for urgent announcements */}
      <div
        aria-live="assertive"
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-assertive-1"
      >
        {isFirst && currentPoliteness === 'assertive' ? currentMessage : ''}
      </div>
      <div
        aria-live="assertive"
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-assertive-2"
      >
        {!isFirst && currentPoliteness === 'assertive' ? currentMessage : ''}
      </div>
    </>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Hooks
// ═══════════════════════════════════════════════════════════════════════════

let announcementIdCounter = 0;

/**
 * Hook to manage screen reader announcements with a simple interface.
 *
 * @deprecated Use useGameAnnouncements for new code.
 *
 * Usage:
 * ```tsx
 * const { message, announce } = useScreenReaderAnnouncement();
 * announce(`Game over! Player 1 wins by elimination.`);
 * <ScreenReaderAnnouncer message={message} />
 * ```
 */
export function useScreenReaderAnnouncement() {
  const [message, setMessage] = useState('');

  const announce = useCallback((text: string) => {
    setMessage(text);
  }, []);

  return { message, announce };
}

/**
 * Hook to manage game announcements with priority queue support.
 *
 * Provides automatic prioritization based on announcement category,
 * debouncing to prevent announcement spam, and a queue system that
 * ensures important messages are heard first.
 *
 * Usage:
 * ```tsx
 * const { queue, announce, removeAnnouncement, clearQueue } = useGameAnnouncements();
 *
 * // When game events occur:
 * announce('Your turn!', 'your_turn');
 * announce('You captured 3 rings!', 'capture');
 * announce('Victory! You won!', 'victory');
 *
 * <ScreenReaderAnnouncer queue={queue} onAnnouncementSpoken={removeAnnouncement} />
 * ```
 */
export function useGameAnnouncements() {
  const [queue, setQueue] = useState<QueuedAnnouncement[]>([]);
  const lastAnnouncementTimeRef = useRef<Map<AnnouncementCategory, number>>(new Map());

  const announce = useCallback((message: string, category: AnnouncementCategory) => {
    const config = CATEGORY_CONFIG[category];
    const now = Date.now();

    // Check debounce
    const lastTime = lastAnnouncementTimeRef.current.get(category) ?? 0;
    if (now - lastTime < config.debounceMs) {
      return; // Skip debounced announcement
    }

    lastAnnouncementTimeRef.current.set(category, now);

    const announcement: QueuedAnnouncement = {
      id: `announcement-${++announcementIdCounter}`,
      message,
      category,
      priority: config.priority,
      politeness: config.politeness,
      timestamp: now,
    };

    setQueue((prev) => [...prev, announcement]);
  }, []);

  const removeAnnouncement = useCallback((id: string) => {
    setQueue((prev) => prev.filter((a) => a.id !== id));
  }, []);

  const clearQueue = useCallback(() => {
    setQueue([]);
  }, []);

  return {
    queue,
    announce,
    removeAnnouncement,
    clearQueue,
  };
}

/**
 * Hook that generates game-specific announcements based on game state changes.
 * Integrates with useGameAnnouncements to automatically announce:
 * - Turn changes
 * - Phase transitions
 * - Victory/defeat
 * - Timer warnings
 *
 * Usage:
 * ```tsx
 * const announcements = useGameAnnouncements();
 * useGameStateAnnouncements({
 *   currentPlayerName: 'Player 2',
 *   isYourTurn: true,
 *   phase: 'movement',
 *   previousPhase: 'ring_placement',
 *   timeRemaining: 30000,
 *   isGameOver: false,
 *   announce: announcements.announce,
 * });
 * ```
 */
export interface GameStateAnnouncementOptions {
  /** Name of the current player */
  currentPlayerName?: string;
  /** Whether it's the local user's turn */
  isYourTurn?: boolean;
  /** Current game phase */
  phase?: string;
  /** Previous game phase (for detecting transitions) */
  previousPhase?: string;
  /** Phase description for announcements */
  phaseDescription?: string;
  /** Time remaining in milliseconds */
  timeRemaining?: number | null;
  /** Whether the game is over */
  isGameOver?: boolean;
  /** Winner name (when game is over) */
  winnerName?: string;
  /** Victory condition */
  victoryCondition?: 'elimination' | 'territory' | 'last_player_standing';
  /** Whether the local user is the winner */
  isWinner?: boolean;
  /** Announce function from useGameAnnouncements */
  announce: (message: string, category: AnnouncementCategory) => void;
}

export function useGameStateAnnouncements(options: GameStateAnnouncementOptions) {
  const {
    currentPlayerName,
    isYourTurn,
    phase,
    previousPhase,
    phaseDescription,
    timeRemaining,
    isGameOver,
    winnerName,
    victoryCondition,
    isWinner,
    announce,
  } = options;

  const prevTurnRef = useRef<boolean | undefined>(undefined);
  const prevPhaseRef = useRef<string | undefined>(undefined);
  const prevTimeWarningRef = useRef<number | null>(null);
  const prevGameOverRef = useRef<boolean | undefined>(undefined);

  // Announce turn changes
  useEffect(() => {
    if (isYourTurn !== prevTurnRef.current && isYourTurn !== undefined) {
      if (currentPlayerName) {
        announce(
          GameAnnouncements.turnChange(currentPlayerName, isYourTurn),
          isYourTurn ? 'your_turn' : 'turn_change'
        );
      }
    }
    prevTurnRef.current = isYourTurn;
  }, [isYourTurn, currentPlayerName, announce]);

  // Announce phase transitions
  useEffect(() => {
    if (phase && phase !== prevPhaseRef.current && previousPhase !== undefined) {
      announce(GameAnnouncements.phaseTransition(phase, phaseDescription), 'phase_transition');
    }
    prevPhaseRef.current = phase;
  }, [phase, previousPhase, phaseDescription, announce]);

  // Announce timer warnings
  useEffect(() => {
    if (timeRemaining === null || timeRemaining === undefined) return;

    const seconds = Math.floor(timeRemaining / 1000);
    const warningThresholds = [60, 30, 10, 5];

    for (const threshold of warningThresholds) {
      if (seconds <= threshold && (prevTimeWarningRef.current ?? Infinity) > threshold) {
        announce(GameAnnouncements.timerWarning(seconds), 'timer_warning');
        prevTimeWarningRef.current = seconds;
        break;
      }
    }
  }, [timeRemaining, announce]);

  // Announce game over
  useEffect(() => {
    if (isGameOver && !prevGameOverRef.current && winnerName && victoryCondition !== undefined) {
      announce(
        GameAnnouncements.victory(winnerName, victoryCondition, isWinner ?? false),
        isWinner ? 'victory' : 'defeat'
      );
    }
    prevGameOverRef.current = isGameOver;
  }, [isGameOver, winnerName, victoryCondition, isWinner, announce]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional Types for Integration
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Helper type for components that consume announcements.
 */
export interface AnnouncementConsumer {
  queue: QueuedAnnouncement[];
  announce: (message: string, category: AnnouncementCategory) => void;
  removeAnnouncement: (id: string) => void;
  clearQueue: () => void;
}

/**
 * Combine multiple announcement states (useful for composing hooks).
 */
export function mergeAnnouncementQueues(...queues: QueuedAnnouncement[][]): QueuedAnnouncement[] {
  return queues.flat().sort(comparePriority);
}
