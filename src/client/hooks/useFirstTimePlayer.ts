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
  /** Whether the user has seen a victory modal */
  hasSeenVictoryModal: boolean;
  /** Whether the user has won at least one game */
  hasWonAGame: boolean;
  /** Number of games played */
  gamesPlayed: number;
  /** Number of games won */
  gamesWon: number;
  /** Last game board type (e.g., 'square8', 'hex8') */
  lastGameBoardType: string | null;
  /** Last game number of players */
  lastGameNumPlayers: number | null;
  /** Timestamp of first visit */
  firstVisit: number;
  /** Timestamp of last game */
  lastGameTimestamp: number | null;
}

const DEFAULT_STATE: OnboardingState = {
  hasSeenWelcome: false,
  hasCompletedFirstGame: false,
  hasSeenControlsHelp: false,
  hasSeenVictoryModal: false,
  hasWonAGame: false,
  gamesPlayed: 0,
  gamesWon: 0,
  lastGameBoardType: null,
  lastGameNumPlayers: null,
  firstVisit: Date.now(),
  lastGameTimestamp: null,
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

export interface GameCompletionInfo {
  /** Whether the player won */
  won: boolean;
  /** Board type (e.g., 'square8', 'hex8') */
  boardType: string;
  /** Number of players */
  numPlayers: number;
}

export interface UseFirstTimePlayerResult {
  /** Current onboarding state */
  state: OnboardingState;
  /** Whether this is a first-time player (hasn't completed first game) */
  isFirstTimePlayer: boolean;
  /** Whether this is a new player (< 5 games) */
  isNewPlayer: boolean;
  /** Whether to show the welcome modal */
  shouldShowWelcome: boolean;
  /** Mark the welcome modal as seen */
  markWelcomeSeen: () => void;
  /** Mark a game as completed with optional details */
  markGameCompleted: (info?: GameCompletionInfo) => void;
  /** Mark controls help as seen */
  markControlsHelpSeen: () => void;
  /** Mark victory modal as seen */
  markVictoryModalSeen: () => void;
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

  const markGameCompleted = useCallback((info?: GameCompletionInfo) => {
    setState((prev) => ({
      ...prev,
      hasCompletedFirstGame: true,
      hasWonAGame: prev.hasWonAGame || (info?.won ?? false),
      gamesPlayed: prev.gamesPlayed + 1,
      gamesWon: prev.gamesWon + (info?.won ? 1 : 0),
      lastGameBoardType: info?.boardType ?? prev.lastGameBoardType,
      lastGameNumPlayers: info?.numPlayers ?? prev.lastGameNumPlayers,
      lastGameTimestamp: Date.now(),
    }));
  }, []);

  const markControlsHelpSeen = useCallback(() => {
    setState((prev) => ({ ...prev, hasSeenControlsHelp: true }));
  }, []);

  const markVictoryModalSeen = useCallback(() => {
    setState((prev) => ({ ...prev, hasSeenVictoryModal: true }));
  }, []);

  const resetOnboarding = useCallback(() => {
    const newState = { ...DEFAULT_STATE, firstVisit: Date.now() };
    setState(newState);
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const isFirstTimePlayer = !state.hasCompletedFirstGame;
  const isNewPlayer = state.gamesPlayed < 5;
  const shouldShowWelcome = !state.hasSeenWelcome;

  return {
    state,
    isFirstTimePlayer,
    isNewPlayer,
    shouldShowWelcome,
    markWelcomeSeen,
    markGameCompleted,
    markControlsHelpSeen,
    markVictoryModalSeen,
    resetOnboarding,
  };
}
