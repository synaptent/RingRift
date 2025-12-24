/**
 * useSandboxInteractions Hook
 *
 * Coordinator hook for sandbox mode interactions. Composes the following
 * sub-hooks for better maintainability:
 * - useSandboxAILoop: AI turn management
 * - useSandboxDecisionHandlers: Player choice handling
 * - useSandboxRingPlacement: Ring placement interactions
 * - useSandboxMoveHandlers: Movement/capture click handling
 *
 * This hook orchestrates the main cell click routing logic.
 */

import type { Position, PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { positionsEqual } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';
import { useInvalidMoveFeedback } from './useInvalidMoveFeedback';
import { useSandboxAILoop } from './useSandboxAILoop';
import { useSandboxDecisionHandlers } from './useSandboxDecisionHandlers';
import { useSandboxRingPlacement } from './useSandboxRingPlacement';
import { useSandboxMoveHandlers } from './useSandboxMoveHandlers';

interface UseSandboxInteractionsOptions {
  selected: Position | undefined;
  setSelected: React.Dispatch<React.SetStateAction<Position | undefined>>;
  validTargets: Position[];
  setValidTargets: React.Dispatch<React.SetStateAction<Position[]>>;
  choiceResolverRef: React.MutableRefObject<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >;
}

export function useSandboxInteractions({
  selected,
  setSelected,
  validTargets,
  setValidTargets,
  choiceResolverRef,
}: UseSandboxInteractionsOptions) {
  const { sandboxEngine, setSandboxStateVersion } = useSandbox();

  // Invalid move feedback
  const { shakingCellKey, triggerInvalidMove, analyzeInvalidMove } = useInvalidMoveFeedback();

  // AI turn management
  const { bumpSandboxTurn, maybeRunSandboxAiIfNeeded } = useSandboxAILoop({
    setSelected,
    setValidTargets,
  });

  // Decision handlers
  const {
    territoryRegionPrompt,
    closeTerritoryRegionPrompt,
    confirmTerritoryRegionPrompt,
    recoveryChoicePromptOpen,
    resolveRecoveryChoice,
    requestRecoveryChoice,
    handleDecisionClick,
  } = useSandboxDecisionHandlers({
    choiceResolverRef,
    maybeRunSandboxAiIfNeeded,
    bumpSandboxTurn,
  });

  // Ring placement handlers
  const {
    ringPlacementCountPrompt,
    closeRingPlacementCountPrompt,
    confirmRingPlacementCountPrompt,
    handleCellDoubleClick,
    handleCellContextMenu,
  } = useSandboxRingPlacement({
    setSelected,
    setValidTargets,
    bumpSandboxTurn,
    ringPlacementCountPrompt: null,
    recoveryChoicePromptOpen,
  });

  // Move handlers
  const {
    handleRingPlacementClick,
    handleChainCaptureClick,
    handleFirstClick,
    handleTargetClick,
  } = useSandboxMoveHandlers({
    sandboxEngine,
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    bumpSandboxTurn,
    setSandboxStateVersion,
    maybeRunSandboxAiIfNeeded,
    requestRecoveryChoice,
    analyzeInvalidMove,
    triggerInvalidMove,
  });

  /** Explicit selection clearer for touch-centric controls */
  const clearSelection = () => {
    setSelected(undefined);
    setValidTargets([]);
    sandboxEngine?.clearSelection();
  };

  /** Main cell click handler for sandbox mode */
  const handleCellClick = (pos: Position) => {
    // Block clicks when prompts are open
    if (ringPlacementCountPrompt || recoveryChoicePromptOpen || territoryRegionPrompt) return;

    // Check if this is a decision click
    if (handleDecisionClick(pos)) return;

    const engine = sandboxEngine;
    if (!engine) return;

    const stateBefore = engine.getGameState();
    const current = stateBefore.players.find((p: any) => p.playerNumber === stateBefore.currentPlayer);

    // If it's an AI player's turn, run AI instead
    if (stateBefore.gameStatus === 'active' && current?.type === 'ai') {
      maybeRunSandboxAiIfNeeded();
      return;
    }

    const phaseBefore = stateBefore.currentPhase;

    // Ring placement phase
    if (phaseBefore === 'ring_placement') {
      handleRingPlacementClick(pos);
      return;
    }

    // Chain-capture phase
    if (phaseBefore === 'chain_capture') {
      handleChainCaptureClick(pos);
      return;
    }

    const isTarget = validTargets.some((t) => positionsEqual(t, pos));

    // No selection: first click
    if (!selected) {
      handleFirstClick(pos);
      return;
    }

    // Clicking same cell clears selection
    if (positionsEqual(selected, pos)) {
      setSelected(undefined);
      setValidTargets([]);
      engine.clearSelection();
      return;
    }

    // Click on valid target: execute move
    if (isTarget) {
      handleTargetClick(pos);
      return;
    }

    // Invalid click while selection is active
    if (phaseBefore === 'movement' || phaseBefore === 'capture') {
      const validMoves = engine.getValidMoves(stateBefore.currentPlayer);
      const reason = analyzeInvalidMove(stateBefore, pos, {
        isPlayer: true,
        isMyTurn: true,
        isConnected: true,
        selectedPosition: selected,
        validMoves,
      });
      triggerInvalidMove(pos, reason);
    }
  };

  return {
    shakingCellKey,
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection,
    ringPlacementCountPrompt,
    closeRingPlacementCountPrompt,
    confirmRingPlacementCountPrompt,
    recoveryChoicePromptOpen,
    resolveRecoveryChoice,
    territoryRegionPrompt,
    closeTerritoryRegionPrompt,
    confirmTerritoryRegionPrompt,
  };
}
