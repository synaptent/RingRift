import React, { useState, useEffect } from 'react';
import type { GamePhase, Move } from '../../shared/types/game';

export type TeachingTopic =
  | 'ring_placement'
  | 'stack_movement'
  | 'capturing'
  | 'chain_capture'
  | 'line_bonus'
  | 'territory'
  | 'victory_elimination'
  | 'victory_territory'
  | 'victory_stalemate';

export interface TeachingContent {
  title: string;
  icon: string;
  description: string;
  tips: string[];
  relatedPhases?: GamePhase[];
}

const TEACHING_CONTENT: Record<TeachingTopic, TeachingContent> = {
  ring_placement: {
    title: 'Ring Placement',
    icon: '◯',
    description:
      'Players take turns placing rings from their hand onto empty board spaces. Rings stack on top of any existing rings at that position.',
    tips: [
      'Placing adjacent to your existing rings helps build territory',
      'Stacking on opponents can set up future captures',
      'The placement phase continues until all players have placed all rings',
    ],
    relatedPhases: ['ring_placement'],
  },
  stack_movement: {
    title: 'Stack Movement',
    icon: '→',
    description:
      'Move a stack you control (your ring on top) exactly as many spaces as the stack height. Movement can be horizontal, vertical, or diagonal in a straight line.',
    tips: [
      'Taller stacks move further but are harder to control',
      'Moving onto an opponent creates a capture opportunity',
      'You cannot pass through other stacks',
    ],
    relatedPhases: ['movement'],
  },
  capturing: {
    title: 'Capturing',
    icon: '×',
    description:
      "When you land on an opponent's stack, you capture (eliminate) one ring from the bottom of their stack. Your ring stays on top.",
    tips: [
      'Capturing removes rings from the game permanently',
      "Eliminate >50% of an opponent's rings to win",
      'Each capture earns one elimination point',
    ],
    relatedPhases: ['capturing'],
  },
  chain_capture: {
    title: 'Chain Capture',
    icon: '⇉',
    description:
      'After capturing, if your new stack position allows another valid capture, you may continue capturing. This creates a chain of captures in a single turn.',
    tips: [
      'Plan chain captures for maximum impact',
      'Chain captures can eliminate multiple rings in one turn',
      'The chain ends when no valid capture targets remain',
    ],
    relatedPhases: ['chain_capture'],
  },
  line_bonus: {
    title: 'Line Bonus',
    icon: '═',
    description:
      'When you form a line of 3+ consecutive spaces containing your rings, you earn a bonus. Choose to either retrieve a ring to hand or claim territory.',
    tips: [
      'Horizontal, vertical, and diagonal lines all count',
      'Retrieved rings can be placed again later',
      'Territory lines mark those spaces as your controlled area',
    ],
    relatedPhases: ['line_processing'],
  },
  territory: {
    title: 'Territory Control',
    icon: '▣',
    description:
      'Territory spaces are permanently owned by a player. Controlling >50% of all board spaces wins the game. Territory cannot be captured or removed.',
    tips: [
      'Territory from line bonuses is permanent',
      'Plan lines to claim strategic board positions',
      'Territory provides a path to victory without elimination',
    ],
    relatedPhases: ['territory_processing'],
  },
  victory_elimination: {
    title: 'Victory: Elimination',
    icon: '💎',
    description:
      "Win by capturing more than half of any single opponent's total rings. In a 2-player game with 24 rings each, capture 13+ rings to win.",
    tips: [
      'Focus captures on one opponent for fastest victory',
      'Track opponent ring counts throughout the game',
      'Chain captures accelerate elimination victories',
    ],
  },
  victory_territory: {
    title: 'Victory: Territory',
    icon: '🏰',
    description:
      'Win by controlling more than half of all board spaces through territory. This requires consistent line formations throughout the game.',
    tips: [
      'Territory cannot be lost once claimed',
      'On an 8×8 board, control 33+ spaces to win',
      'Balance territory gains with defensive play',
    ],
  },
  victory_stalemate: {
    title: 'Victory: Last Standing',
    icon: '👑',
    description:
      'Win if, after a full round of turns, you are the only player able to make a meaningful move (placement, movement, or capture).',
    tips: [
      'Occurs when opponents run out of valid moves',
      'Can happen through elimination or board control',
      'Usually a late-game condition',
    ],
  },
};

export interface TeachingOverlayProps {
  /** The topic to display */
  topic: TeachingTopic;
  /** Whether the overlay is visible */
  isOpen: boolean;
  /** Callback when overlay is closed */
  onClose: () => void;
  /** Position hint for the overlay */
  position?: 'center' | 'bottom-right';
  /** Additional CSS classes */
  className?: string;
}

/**
 * Overlay component that displays teaching content for game concepts.
 * Used during analysis, replay, or when players need help understanding a mechanic.
 */
export function TeachingOverlay({
  topic,
  isOpen,
  onClose,
  position = 'center',
  className = '',
}: TeachingOverlayProps) {
  const content = TEACHING_CONTENT[topic];

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !content) return null;

  const positionClasses =
    position === 'center'
      ? 'fixed inset-0 flex items-center justify-center'
      : 'fixed bottom-4 right-4';

  return (
    <div className={positionClasses} style={{ zIndex: 60 }}>
      {/* Backdrop for center position */}
      {position === 'center' && (
        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      )}

      {/* Content Card */}
      <div
        className={`relative bg-slate-900 border border-slate-600 rounded-xl shadow-2xl max-w-md w-full mx-4 overflow-hidden ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="teaching-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{content.icon}</span>
            <h2 id="teaching-title" className="text-lg font-bold text-slate-100">
              {content.title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Description */}
        <div className="px-4 py-3">
          <p className="text-sm text-slate-300 leading-relaxed">{content.description}</p>
        </div>

        {/* Tips */}
        <div className="px-4 pb-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Tips
          </h3>
          <ul className="space-y-2">
            {content.tips.map((tip, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-slate-400">
                <span className="text-emerald-400 mt-0.5">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Related Phases Badge */}
        {content.relatedPhases && content.relatedPhases.length > 0 && (
          <div className="px-4 pb-3 flex items-center gap-2">
            <span className="text-[10px] text-slate-500 uppercase">Applies to:</span>
            {content.relatedPhases.map((phase) => (
              <span
                key={phase}
                className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700"
              >
                {phase.replace('_', ' ')}
              </span>
            ))}
          </div>
        )}

        {/* Footer hint */}
        <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700/50 text-center">
          <span className="text-[10px] text-slate-500">
            Press <kbd className="px-1 py-0.5 bg-slate-700 rounded text-[9px]">Esc</kbd> or click
            outside to close
          </span>
        </div>
      </div>
    </div>
  );
}

/**
 * Hook to manage teaching overlay state
 */
export function useTeachingOverlay() {
  const [currentTopic, setCurrentTopic] = useState<TeachingTopic | null>(null);

  const showTopic = (topic: TeachingTopic) => setCurrentTopic(topic);
  const hideTopic = () => setCurrentTopic(null);

  return {
    currentTopic,
    isOpen: currentTopic !== null,
    showTopic,
    hideTopic,
  };
}

/**
 * Determine which teaching topic is relevant for a given move
 */
export function getTeachingTopicForMove(move: Move): TeachingTopic | null {
  switch (move.phase) {
    case 'ring_placement':
      return 'ring_placement';
    case 'movement':
      return 'stack_movement';
    case 'capturing':
      return 'capturing';
    case 'chain_capture':
      return 'chain_capture';
    case 'line_processing':
      return 'line_bonus';
    case 'territory_processing':
      return 'territory';
    default:
      return null;
  }
}

/**
 * Quick-access buttons for all teaching topics
 */
export function TeachingTopicButtons({
  onSelectTopic,
  className = '',
}: {
  onSelectTopic: (topic: TeachingTopic) => void;
  className?: string;
}) {
  const topics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'ring_placement', label: 'Placement', icon: '◯' },
    { topic: 'stack_movement', label: 'Movement', icon: '→' },
    { topic: 'capturing', label: 'Capture', icon: '×' },
    { topic: 'chain_capture', label: 'Chain', icon: '⇉' },
    { topic: 'line_bonus', label: 'Lines', icon: '═' },
    { topic: 'territory', label: 'Territory', icon: '▣' },
  ];

  const victoryTopics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'victory_elimination', label: 'Elimination', icon: '💎' },
    { topic: 'victory_territory', label: 'Territory', icon: '🏰' },
    { topic: 'victory_stalemate', label: 'Stalemate', icon: '👑' },
  ];

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide">Game Mechanics</div>
      <div className="flex flex-wrap gap-1">
        {topics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>

      <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-3">
        Victory Conditions
      </div>
      <div className="flex flex-wrap gap-1">
        {victoryTopics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>
    </div>
  );
}
