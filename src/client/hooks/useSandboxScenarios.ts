/**
 * useSandboxScenarios - Scenario loading and replay management for sandbox mode
 *
 * This hook extracts scenario/replay functionality from SandboxGameHost:
 * - Loading curated teaching scenarios
 * - Loading self-play games from database
 * - Replaying move history with animations
 * - Forking from a replay position to start a new game
 * - History scrubbing and snapshot detection
 *
 * @module hooks/useSandboxScenarios
 */

import React, { useState, useCallback, useRef } from 'react';
import { toast } from 'react-hot-toast';
import type { GameState, Position, Move } from '../../shared/types/game';
import type { ClientSandboxEngine } from '../sandbox/ClientSandboxEngine';
import type { MoveAnimationData as MoveAnimation } from '../components/BoardView';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Loaded scenario metadata.
 */
export interface LoadedScenario {
  id: string;
  name: string;
  description?: string;
  /** Whether this is an onboarding/teaching scenario */
  onboarding?: boolean;
  /** Rules concept being taught */
  rulesConcept?: string;
  /** Rules snippet to display */
  rulesSnippet?: string;
  /** Source: 'builtin', 'user', 'selfplay' */
  source?: string;
}

/**
 * Scenario data loaded from picker or browser.
 */
export interface ScenarioData {
  id: string;
  name: string;
  description?: string;
  gameState: GameState;
  /** Move history to replay (for self-play games) */
  moveHistory?: Move[];
  /** Metadata */
  onboarding?: boolean;
  rulesConcept?: string;
  rulesSnippet?: string;
  source?: string;
}

/**
 * Options for the scenarios hook.
 */
export interface SandboxScenariosOptions<T = ScenarioData> {
  /** Initialize sandbox engine with scenario - parent handles all complex engine creation logic */
  initSandboxWithScenario: (scenario: T) => ClientSandboxEngine | null;
  /** Callback when scenario is loaded (for telemetry, etc.) */
  onScenarioLoaded?: (scenario: LoadedScenario) => void;
  /** Callback when state version should be bumped */
  onStateVersionChange?: () => void;
  /** Callback to reset parent's UI state (selection, pending choices, etc.) */
  onUIStateReset?: () => void;
}

/**
 * Return type for useSandboxScenarios.
 */
export interface SandboxScenariosState<T = ScenarioData> {
  // Scenario state
  lastLoadedScenario: LoadedScenario | null;
  showScenarioPicker: boolean;
  setShowScenarioPicker: (show: boolean) => void;
  showSelfPlayBrowser: boolean;
  setShowSelfPlayBrowser: (show: boolean) => void;

  // Replay state
  isInReplayMode: boolean;
  setIsInReplayMode: (inReplay: boolean) => void;
  replayState: GameState | null;
  setReplayState: (state: GameState | null) => void;
  replayAnimation: MoveAnimation | null;
  setReplayAnimation: (anim: MoveAnimation | null) => void;

  // History playback
  isViewingHistory: boolean;
  setIsViewingHistory: (viewing: boolean) => void;
  historyViewIndex: number;
  setHistoryViewIndex: (index: number) => void;
  hasHistorySnapshots: boolean;
  setHasHistorySnapshots: (has: boolean) => void;

  // Handlers
  handleLoadScenario: (scenario: T) => ClientSandboxEngine | null;
  handleForkFromReplay: (state: GameState, moveIndex: number) => void;
  handleResetScenario: () => void;
  clearScenarioContext: () => void;

  // Ref to original scenario for reset
  originalScenarioRef: React.RefObject<T | null>;
  setLastLoadedScenario: (scenario: LoadedScenario | null) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing scenario loading and replay in sandbox mode.
 *
 * This hook provides state management for scenarios, replay, and history playback.
 * Complex logic like engine creation, config updates, and move replay should be
 * handled by the parent component via callbacks.
 *
 * @template T - The scenario type (defaults to ScenarioData)
 */
export function useSandboxScenarios<
  T extends {
    id: string;
    name: string;
    description?: string;
    onboarding?: boolean;
    rulesConcept?: string;
    rulesSnippet?: string;
    source?: string;
  } = ScenarioData,
>(options: SandboxScenariosOptions<T>): SandboxScenariosState<T> {
  const { initSandboxWithScenario, onScenarioLoaded, onStateVersionChange, onUIStateReset } =
    options;

  // Scenario state
  const [lastLoadedScenario, setLastLoadedScenario] = useState<LoadedScenario | null>(null);
  const [showScenarioPicker, setShowScenarioPicker] = useState(false);
  const [showSelfPlayBrowser, setShowSelfPlayBrowser] = useState(false);

  // Replay state
  const [isInReplayMode, setIsInReplayMode] = useState(false);
  const [replayState, setReplayState] = useState<GameState | null>(null);
  const [replayAnimation, setReplayAnimation] = useState<MoveAnimation | null>(null);

  // History playback
  const [isViewingHistory, setIsViewingHistory] = useState(false);
  const [historyViewIndex, setHistoryViewIndex] = useState(0);
  const [hasHistorySnapshots, setHasHistorySnapshots] = useState(true);

  // Ref to store the original scenario for reset
  const originalScenarioRef = useRef<T | null>(null);

  // Load a scenario - simplified to delegate complex logic to parent
  // Returns the created engine so the caller can use it for replay without
  // waiting for React state updates (which are asynchronous)
  const handleLoadScenario = useCallback(
    (scenario: T): ClientSandboxEngine | null => {
      // Store for potential reset
      originalScenarioRef.current = scenario;

      // Initialize engine with scenario (parent handles complex logic)
      const engine = initSandboxWithScenario(scenario);

      if (!engine) {
        toast.error('Failed to load scenario');
        return null;
      }

      // Reset parent's UI state (selection, pending choices, etc.)
      onUIStateReset?.();

      // Track loaded scenario metadata
      const loadedScenario: LoadedScenario = {
        id: scenario.id,
        name: scenario.name,
        description: scenario.description,
        onboarding: scenario.onboarding,
        rulesConcept: scenario.rulesConcept,
        rulesSnippet: scenario.rulesSnippet,
        source: scenario.source,
      };

      setLastLoadedScenario(loadedScenario);
      onScenarioLoaded?.(loadedScenario);

      // Close pickers
      setShowScenarioPicker(false);
      setShowSelfPlayBrowser(false);

      // Reset history state
      setIsViewingHistory(false);
      setHistoryViewIndex(0);
      setHasHistorySnapshots(true);

      // Exit replay mode if active
      setIsInReplayMode(false);
      setReplayState(null);

      // Note: Move replay for self-play scenarios is handled by the parent
      // after receiving the onScenarioLoaded callback

      onStateVersionChange?.();

      // Return the engine so caller can use it immediately for replay
      return engine;
    },
    [initSandboxWithScenario, onScenarioLoaded, onStateVersionChange, onUIStateReset]
  );

  // Fork from a replay position to start a new playable game
  const handleForkFromReplay = useCallback(
    (state: GameState, moveIndex: number) => {
      // For forking, we create a minimal scenario-like object
      // The parent's initSandboxWithScenario must handle this case
      const forkScenario = {
        id: `fork-${state.id}-${moveIndex}`,
        name: `Fork from move ${moveIndex}`,
        description: `Forked from replay at move ${moveIndex}`,
        source: 'fork',
        // Extract boardType and playerCount from state for parent's engine init
        boardType: state.boardType,
        playerCount: state.players?.length,
        // Include the state for the parent to use
        state,
      } as unknown as T;

      // Initialize engine with forked state (parent handles complex logic)
      const engine = initSandboxWithScenario(forkScenario);

      if (!engine) {
        toast.error('Failed to fork from replay');
        return;
      }

      // Reset parent's UI state
      onUIStateReset?.();

      // Clear scenario context (fork is not a teaching scenario)
      setLastLoadedScenario(null);
      originalScenarioRef.current = null;

      // Exit replay mode
      setIsInReplayMode(false);
      setReplayState(null);

      toast.success(`Forked from move ${moveIndex}`);
      onStateVersionChange?.();
    },
    [initSandboxWithScenario, onStateVersionChange, onUIStateReset]
  );

  // Reset to last loaded scenario
  const handleResetScenario = useCallback(() => {
    const scenario = originalScenarioRef.current;

    if (!scenario) {
      toast.error('No scenario to reset');
      return;
    }

    // Re-load the original scenario
    handleLoadScenario(scenario);
  }, [handleLoadScenario]);

  // Clear scenario context
  const clearScenarioContext = useCallback(() => {
    setLastLoadedScenario(null);
    originalScenarioRef.current = null;
    setIsViewingHistory(false);
    setHistoryViewIndex(0);
    setHasHistorySnapshots(true);
    setIsInReplayMode(false);
    setReplayState(null);
  }, []);

  return {
    // Scenario state
    lastLoadedScenario,
    showScenarioPicker,
    setShowScenarioPicker,
    showSelfPlayBrowser,
    setShowSelfPlayBrowser,

    // Replay state
    isInReplayMode,
    setIsInReplayMode,
    replayState,
    setReplayState,
    replayAnimation,
    setReplayAnimation,

    // History playback
    isViewingHistory,
    setIsViewingHistory,
    historyViewIndex,
    setHistoryViewIndex,
    hasHistorySnapshots,
    setHasHistorySnapshots,

    // Handlers
    handleLoadScenario,
    handleForkFromReplay,
    handleResetScenario,
    clearScenarioContext,

    // Expose ref and setter for parent access
    originalScenarioRef,
    setLastLoadedScenario,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// CHAIN CAPTURE PATH UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Extract chain capture path from game state.
 *
 * When in chain_capture phase, walks backwards through move history
 * to build the visualization path.
 */
export function extractChainCapturePath(gameState: GameState | null): Position[] | undefined {
  if (!gameState || gameState.currentPhase !== 'chain_capture') {
    return undefined;
  }

  const moveHistory = gameState.moveHistory;
  if (!moveHistory || moveHistory.length === 0) {
    return undefined;
  }

  const currentPlayer = gameState.currentPlayer;
  const path: Position[] = [];

  // Walk backwards to find all chain capture moves by the current player
  for (let i = moveHistory.length - 1; i >= 0; i--) {
    const move = moveHistory[i];
    if (!move) continue;

    // Stop if we hit a move by a different player or a non-capture move
    if (
      move.player !== currentPlayer ||
      (move.type !== 'overtaking_capture' && move.type !== 'continue_capture_segment')
    ) {
      break;
    }

    // Add the landing position to the front of the path
    if (move.to) {
      path.unshift(move.to);
    }

    // If this is the first capture in the chain, add the starting position
    if (move.type === 'overtaking_capture' && move.from) {
      path.unshift(move.from);
    }
  }

  // Need at least 2 positions to show a path
  return path.length >= 2 ? path : undefined;
}
