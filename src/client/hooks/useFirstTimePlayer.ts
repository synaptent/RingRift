/**
 * ═══════════════════════════════════════════════════════════════════════════
 * First-Time Player Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Tracks onboarding state for new players using localStorage.
 * Helps surface tutorial content and guided experiences for first-time users.
 */

import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'ringrift_onboarding';

export interface OnboardingState {
  /** Whether the user has seen the welcome modal */
  hasSeenWelcome: boolean;
  /** Whether the user has completed their first game */
  hasCompletedFirstGame: boolean;
  /** Whether the user has seen the controls help */
  hasSeenControlsHelp: boolean;
  /** Number of games played */
  gamesPlayed: number;
  /** Timestamp of first visit */
  firstVisit: number;
}

const DEFAULT_STATE: OnboardingState = {
  hasSeenWelcome: false,
  hasCompletedFirstGame: false,
  hasSeenControlsHelp: false,
  gamesPlayed: 0,
  firstVisit: Date.now(),
};

function loadState(): OnboardingState {
  if (typeof window === 'undefined') return DEFAULT_STATE;

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...DEFAULT_STATE, ...JSON.parse(stored) };
    }
  } catch {
    // Invalid JSON or storage error - use defaults
  }

  return { ...DEFAULT_STATE, firstVisit: Date.now() };
}

function saveState(state: OnboardingState): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Storage full or disabled - ignore
  }
}

export interface UseFirstTimePlayerResult {
  /** Current onboarding state */
  state: OnboardingState;
  /** Whether this is a first-time player (hasn't completed first game) */
  isFirstTimePlayer: boolean;
  /** Whether to show the welcome modal */
  shouldShowWelcome: boolean;
  /** Mark the welcome modal as seen */
  markWelcomeSeen: () => void;
  /** Mark a game as completed */
  markGameCompleted: () => void;
  /** Mark controls help as seen */
  markControlsHelpSeen: () => void;
  /** Reset onboarding state (for testing) */
  resetOnboarding: () => void;
}

/**
 * Hook to track and manage first-time player onboarding state.
 *
 * @example
 * ```tsx
 * const { isFirstTimePlayer, shouldShowWelcome, markWelcomeSeen } = useFirstTimePlayer();
 *
 * if (shouldShowWelcome) {
 *   return <OnboardingModal onClose={markWelcomeSeen} />;
 * }
 * ```
 */
export function useFirstTimePlayer(): UseFirstTimePlayerResult {
  const [state, setState] = useState<OnboardingState>(loadState);

  // Sync state to localStorage on changes
  useEffect(() => {
    saveState(state);
  }, [state]);

  const markWelcomeSeen = useCallback(() => {
    setState((prev) => ({ ...prev, hasSeenWelcome: true }));
  }, []);

  const markGameCompleted = useCallback(() => {
    setState((prev) => ({
      ...prev,
      hasCompletedFirstGame: true,
      gamesPlayed: prev.gamesPlayed + 1,
    }));
  }, []);

  const markControlsHelpSeen = useCallback(() => {
    setState((prev) => ({ ...prev, hasSeenControlsHelp: true }));
  }, []);

  const resetOnboarding = useCallback(() => {
    const newState = { ...DEFAULT_STATE, firstVisit: Date.now() };
    setState(newState);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const isFirstTimePlayer = !state.hasCompletedFirstGame;
  const shouldShowWelcome = !state.hasSeenWelcome;

  return {
    state,
    isFirstTimePlayer,
    shouldShowWelcome,
    markWelcomeSeen,
    markGameCompleted,
    markControlsHelpSeen,
    resetOnboarding,
  };
}
