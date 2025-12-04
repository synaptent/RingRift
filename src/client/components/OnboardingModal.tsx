import React, { useEffect, useRef, useState } from 'react';

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartTutorial: () => void;
}

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

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
function WelcomeStep() {
  return (
    <div className="text-center space-y-4">
      <div className="text-6xl mb-4">🎮</div>
      <h2 className="text-2xl font-bold text-slate-100">Welcome to RingRift!</h2>
      <p className="text-slate-300 max-w-md mx-auto">
        A strategic board game where you place rings, build stacks, and compete for territory.
      </p>
    </div>
  );
}

/**
 * Game phases overview step
 */
function PhasesStep() {
  const phases = [
    { icon: '🎯', name: 'Ring Placement', desc: 'Place rings on the board to control territory' },
    { icon: '⚡', name: 'Movement', desc: 'Move your stacks based on their height' },
    { icon: '⚔️', name: 'Capture', desc: 'Land on opponents to capture their rings' },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-100 text-center">Three Simple Phases</h2>
      <div className="space-y-3">
        {phases.map((phase) => (
          <div key={phase.name} className="flex items-center gap-3 bg-slate-800/50 rounded-lg p-3">
            <span className="text-2xl">{phase.icon}</span>
            <div>
              <div className="font-semibold text-slate-100">{phase.name}</div>
              <div className="text-sm text-slate-400">{phase.desc}</div>
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
function VictoryStep() {
  const conditions = [
    {
      icon: '💎',
      name: 'Ring Elimination',
      desc: "Capture more than half of any opponent's rings",
    },
    { icon: '🏰', name: 'Territory Control', desc: 'Control more than half the board spaces' },
    { icon: '👑', name: 'Last Standing', desc: 'Be the only player who can still move' },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-slate-100 text-center">Ways to Win</h2>
      <div className="space-y-3">
        {conditions.map((condition) => (
          <div
            key={condition.name}
            className="flex items-center gap-3 bg-slate-800/50 rounded-lg p-3"
          >
            <span className="text-2xl">{condition.icon}</span>
            <div>
              <div className="font-semibold text-slate-100">{condition.name}</div>
              <div className="text-sm text-slate-400">{condition.desc}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Ready to play step
 */
function ReadyStep() {
  return (
    <div className="text-center space-y-4">
      <div className="text-6xl mb-4">🚀</div>
      <h2 className="text-2xl font-bold text-slate-100">Ready to Play!</h2>
      <p className="text-slate-300 max-w-md mx-auto">
        We recommend starting with{' '}
        <span className="text-blue-400 font-semibold">"Learn the Basics"</span> - a quick game
        against a beginner-friendly AI on a compact board.
      </p>
      <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4 mt-4">
        <p className="text-sm text-blue-200">
          💡 <strong>Tip:</strong> Press{' '}
          <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-xs">?</kbd> anytime during the
          game to see keyboard shortcuts and controls.
        </p>
      </div>
    </div>
  );
}

const STEPS = [WelcomeStep, PhasesStep, VictoryStep, ReadyStep];

/**
 * Onboarding Modal for first-time players
 *
 * Displays a multi-step introduction to RingRift's core concepts.
 */
export function OnboardingModal({ isOpen, onClose, onStartTutorial }: OnboardingModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  const isLastStep = currentStep === STEPS.length - 1;
  const isFirstStep = currentStep === 0;

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

      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowRight' || e.key === 'Enter') {
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
  }, [isOpen, isLastStep, isFirstStep, onClose, onStartTutorial]);

  // Focus trap
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    previouslyFocusedElementRef.current = (document.activeElement as HTMLElement | null) ?? null;

    if (!dialogEl) return;

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (first) {
      first.focus();
    }

    const handleTabKey = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || focusable.length === 0) return;

      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      const isShift = event.shiftKey;

      if (isShift && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!isShift && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    dialogEl.addEventListener('keydown', handleTabKey);

    return () => {
      dialogEl.removeEventListener('keydown', handleTabKey);
      if (previouslyFocusedElementRef.current) {
        previouslyFocusedElementRef.current.focus();
      }
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const CurrentStepComponent = STEPS[currentStep];

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-title"
      onClick={handleBackdropClick}
    >
      <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-lg w-full mx-4 p-6 relative overflow-hidden">
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
          <CurrentStepComponent />
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
              onClick={onStartTutorial}
              className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
            >
              Start Playing
            </button>
          ) : (
            <button
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
      </div>
    </div>
  );
}
