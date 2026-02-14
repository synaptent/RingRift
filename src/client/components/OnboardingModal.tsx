import React, { useEffect, useRef, useState } from 'react';
import type { BoardType } from '../../shared/types/game';
import {
  ONBOARDING_COPY,
  logHelpOpenEvent,
  newHelpSessionId,
  sendRulesUxEvent,
} from '../utils/rulesUxTelemetry';
import { Dialog } from './ui/Dialog';
import {
  WelcomeIllustration,
  PlacementIllustration,
  MovementIllustration,
  CaptureIllustration,
  EliminationIcon,
  TerritoryIcon,
  LastStandingIcon,
  ReadyToPlayIllustration,
} from './onboarding/OnboardingIllustrations';

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartTutorial: () => void;
}

interface StepProps {
  onVictoryConceptClick?: (conceptId: string) => void;
}

const ONBOARDING_BOARD_TYPE: BoardType = 'square8';
const ONBOARDING_NUM_PLAYERS = 2;
const ONBOARDING_RULES_CONCEPT = 'board_intro_square8';
const ONBOARDING_TOPIC = 'onboarding';

/**
 * Step indicator for the onboarding flow
 */
function StepIndicator({ currentStep, totalSteps }: { currentStep: number; totalSteps: number }) {
  return (
    <div className="flex justify-center gap-2 mt-4">
      {Array.from({ length: totalSteps }, (_, i) => (
        <div
          key={i}
          className={`w-2 h-2 rounded-full transition-colors duration-300 ${
            i === currentStep ? 'bg-blue-500' : 'bg-slate-600'
          }`}
        />
      ))}
    </div>
  );
}

/**
 * Welcome step content
 */
function WelcomeStep(_: StepProps) {
  const { title, body } = ONBOARDING_COPY.intro;
  return (
    <div className="text-center space-y-4">
      <WelcomeIllustration />
      <h2 className="text-2xl font-bold text-slate-100">{title}</h2>
      <p className="text-slate-300 max-w-md mx-auto">{body}</p>
    </div>
  );
}

/**
 * Game phases overview step
 */
function PhasesStep(_: StepProps) {
  const phases = ONBOARDING_COPY.phases;

  const getPhaseIllustration = (id: string) => {
    switch (id) {
      case 'sandbox.phase.ring_placement':
        return <PlacementIllustration />;
      case 'sandbox.phase.movement':
        return <MovementIllustration />;
      case 'sandbox.phase.capture':
        return <CaptureIllustration />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      <div className="space-y-1 text-center">
        <h2 className="text-xl font-bold text-slate-100">Three Core Phases</h2>
        <p className="text-sm text-slate-400">
          Other phases (chain captures, line/territory processing, forced elimination) appear when
          triggered.
        </p>
      </div>
      <div className="space-y-3">
        {phases.map((phase) => (
          <div key={phase.id} className="flex items-center gap-3 bg-slate-800/50 rounded-lg p-3">
            {getPhaseIllustration(phase.id)}
            <div>
              <div className="font-semibold text-slate-100">{phase.title}</div>
              <div className="text-sm text-slate-400">{phase.body}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Victory conditions step
 */
const VICTORY_SHORT_DESCRIPTIONS: Record<string, string> = {
  'onboarding.victory.elimination':
    'Eliminate enough of your opponents\u2019 rings to hit the elimination threshold and win instantly.',
  'onboarding.victory.territory':
    'Control more territory than all opponents combined by collapsing lines and claiming regions.',
  'onboarding.victory.lps':
    'Be the last player who can still make moves for three consecutive rounds.',
};

function VictoryStep({ onVictoryConceptClick }: StepProps) {
  const concepts = ONBOARDING_COPY.victoryConcepts;

  const getConceptIcon = (id: string) => {
    switch (id) {
      case 'onboarding.victory.elimination':
        return <EliminationIcon />;
      case 'onboarding.victory.territory':
        return <TerritoryIcon />;
      case 'onboarding.victory.lps':
        return <LastStandingIcon />;
      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-100 text-center">Ways to Win</h2>
      <div className="space-y-3">
        {concepts.map((concept) => (
          <button
            key={concept.id}
            type="button"
            onClick={() => onVictoryConceptClick?.(concept.id)}
            className="w-full flex items-center gap-3 bg-slate-800/50 rounded-lg p-3 text-left hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            {getConceptIcon(concept.id)}
            <div>
              <div className="font-semibold text-slate-100">{concept.title}</div>
              <div className="text-sm text-slate-400">
                {VICTORY_SHORT_DESCRIPTIONS[concept.id] ?? concept.body}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Ready to play step
 */
function ReadyStep(_: StepProps) {
  return (
    <div className="text-center space-y-4">
      <ReadyToPlayIllustration />
      <h2 className="text-2xl font-bold text-slate-100">Ready to Play!</h2>
      <p className="text-slate-300 max-w-md mx-auto">
        We recommend starting with{' '}
        <span className="text-blue-400 font-semibold">"Learn the Basics"</span> — a quick game
        against a beginner-friendly AI on a compact board.
      </p>
      <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4 mt-4">
        <p className="text-sm text-blue-200">
          <strong>Tip:</strong> Press{' '}
          <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">?</kbd> anytime during the
          game to see keyboard shortcuts and controls.
        </p>
      </div>
    </div>
  );
}

const STEPS: Array<(props: StepProps) => React.ReactElement> = [
  WelcomeStep,
  PhasesStep,
  VictoryStep,
  ReadyStep,
];

/**
 * Onboarding Modal for first-time players
 *
 * Displays a multi-step introduction to RingRift's core concepts.
 */
export function OnboardingModal({ isOpen, onClose, onStartTutorial }: OnboardingModalProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const primaryButtonRef = useRef<HTMLButtonElement | null>(null);

  const isLastStep = currentStep === STEPS.length - 1;
  const isFirstStep = currentStep === 0;

  const helpSessionIdRef = useRef<string | null>(null);
  const openedAtRef = useRef<number | null>(null);
  const prevIsOpenRef = useRef(false);

  // Emit rules-UX telemetry when the onboarding modal is opened/closed.
  useEffect(() => {
    const prev = prevIsOpenRef.current;
    prevIsOpenRef.current = isOpen;

    if (isOpen && !prev) {
      openedAtRef.current = Date.now();
      if (!helpSessionIdRef.current) {
        helpSessionIdRef.current = newHelpSessionId();
      }
      const helpSessionId = helpSessionIdRef.current;
      const boardType = ONBOARDING_BOARD_TYPE;
      const numPlayers = ONBOARDING_NUM_PLAYERS;

      const baseEvent = {
        boardType,
        numPlayers,
        topic: ONBOARDING_TOPIC,
        rulesConcept: ONBOARDING_RULES_CONCEPT,
        isSandbox: true,
      } as const;

      void logHelpOpenEvent({
        boardType,
        numPlayers,
        difficulty: 'tutorial',
        rulesConcept: ONBOARDING_RULES_CONCEPT,
        topic: ONBOARDING_TOPIC,
        source: 'sandbox',
        entrypoint: 'sandbox_toolbar_help',
        isRanked: false,
        isCalibrationGame: false,
        isSandbox: true,
        helpSessionId,
      });

      void sendRulesUxEvent({
        type: 'rules_help_open',
        ...baseEvent,
      });
    } else if (!isOpen && prev && openedAtRef.current != null) {
      const openedAt = openedAtRef.current;
      openedAtRef.current = null;
      const msSinceOpen = Date.now() - openedAt;
      const boardType = ONBOARDING_BOARD_TYPE;
      const numPlayers = ONBOARDING_NUM_PLAYERS;
      const helpSessionId = helpSessionIdRef.current ?? newHelpSessionId();
      helpSessionIdRef.current = helpSessionId;

      void sendRulesUxEvent({
        type: 'help_topic_view',
        boardType,
        numPlayers,
        topic: ONBOARDING_TOPIC,
        rulesConcept: ONBOARDING_RULES_CONCEPT,
        isSandbox: true,
        helpSessionId,
        payload: {
          topic_id: 'onboarding.summary',
          ms_since_help_open: msSinceOpen,
        },
      });
    }
  }, [isOpen]);

  // Reset step when modal opens
  useEffect(() => {
    if (isOpen) {
      setCurrentStep(0);
    }
  }, [isOpen]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowRight' || e.key === 'Enter') {
        if (isLastStep) {
          onStartTutorial();
        } else {
          setCurrentStep((s) => Math.min(s + 1, STEPS.length - 1));
        }
      } else if (e.key === 'ArrowLeft' && !isFirstStep) {
        setCurrentStep((s) => Math.max(s - 1, 0));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isLastStep, isFirstStep, onStartTutorial]);

  const handleVictoryConceptClick = (conceptId: string) => {
    const boardType = ONBOARDING_BOARD_TYPE;
    const numPlayers = ONBOARDING_NUM_PLAYERS;
    const helpSessionId = helpSessionIdRef.current ?? newHelpSessionId();
    helpSessionIdRef.current = helpSessionId;

    const openedAt = openedAtRef.current;
    const msSinceHelpOpen =
      typeof openedAt === 'number' ? Math.max(0, Date.now() - openedAt) : undefined;

    const payload: Record<string, unknown> = {
      topic_id: conceptId,
    };
    if (msSinceHelpOpen !== undefined) {
      payload.ms_since_help_open = msSinceHelpOpen;
    }

    void sendRulesUxEvent({
      type: 'help_topic_view',
      boardType,
      numPlayers,
      topic: ONBOARDING_TOPIC,
      rulesConcept: ONBOARDING_RULES_CONCEPT,
      isSandbox: true,
      helpSessionId,
      payload,
    });
  };

  if (!isOpen) return null;

  const CurrentStepComponent = STEPS[currentStep];

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      labelledBy="onboarding-title"
      initialFocusRef={primaryButtonRef}
      backdropClassName="bg-black/70 backdrop-blur-sm"
      className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-xl w-full mx-4 p-6 relative overflow-hidden"
    >
      <h2 id="onboarding-title" className="sr-only">
        RingRift onboarding
      </h2>

      {/* Skip button */}
      <button
        onClick={onClose}
        className="absolute top-4 right-4 text-slate-400 hover:text-slate-200 text-sm transition-colors"
        aria-label="Skip introduction"
      >
        Skip
      </button>

      {/* Content */}
      <div className="min-h-[280px] flex flex-col justify-center py-4">
        <CurrentStepComponent onVictoryConceptClick={handleVictoryConceptClick} />
      </div>

      {/* Step indicator */}
      <StepIndicator currentStep={currentStep} totalSteps={STEPS.length} />

      {/* Navigation buttons */}
      <div className="flex justify-between items-center mt-6">
        <button
          onClick={() => setCurrentStep((s) => Math.max(s - 1, 0))}
          disabled={isFirstStep}
          className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
            isFirstStep
              ? 'text-slate-600 cursor-not-allowed'
              : 'text-slate-300 hover:text-slate-100 hover:bg-slate-800'
          }`}
        >
          Back
        </button>

        {isLastStep ? (
          <button
            ref={primaryButtonRef}
            onClick={onStartTutorial}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Start Playing
          </button>
        ) : (
          <button
            ref={primaryButtonRef}
            onClick={() => setCurrentStep((s) => Math.min(s + 1, STEPS.length - 1))}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
          >
            Next
          </button>
        )}
      </div>

      {/* Keyboard hint */}
      <p className="text-center text-xs text-slate-500 mt-4">
        Use arrow keys to navigate, Enter to continue
      </p>
    </Dialog>
  );
}
