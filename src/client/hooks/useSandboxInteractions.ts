/**
 * @fileoverview useSandboxInteractions Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It coordinates UI interactions, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 *
 * This adapter:
 * - Composes sub-hooks for AI turns, decisions, placement, movement
 * - Orchestrates main cell click routing logic
 * - Manages selection state and UI feedback
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
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
  setSelected: (pos: Position | undefined) => void;
  validTargets: Position[];
  setValidTargets: (cells: Position[]) => void;
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
  const { handleRingPlacementClick, handleChainCaptureClick, handleFirstClick, handleTargetClick } =
    useSandboxMoveHandlers({
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
    const current = stateBefore.players.find(
      (p: { playerNumber: number; type?: string }) => p.playerNumber === stateBefore.currentPlayer
    );

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

    // Territory processing phase - auto-apply no_territory_action when no regions to process
    // RR-FIX-2025-01-10: Handle stuck territory_processing phase for human players
    if (phaseBefore === 'territory_processing') {
      const validMoves = engine.getValidMoves(stateBefore.currentPlayer);

      // Check if the only valid move is no_territory_action (no regions to process)
      const hasOnlyNoTerritoryAction =
        validMoves.length === 1 && validMoves[0].type === 'no_territory_action';

      if (hasOnlyNoTerritoryAction) {
        // Auto-apply the no_territory_action move to advance the game
        void (async () => {
          await engine.applyCanonicalMove(validMoves[0]);
          bumpSandboxTurn();
          setSandboxStateVersion((v) => v + 1);
          maybeRunSandboxAiIfNeeded();
        })();
        return;
      }

      // Check for elimination moves - let player click on their stacks
      const eliminationMoves = validMoves.filter((m) => m.type === 'eliminate_rings_from_stack');
      if (eliminationMoves.length > 0) {
        const matchingElim = eliminationMoves.find((m) => m.to && positionsEqual(m.to, pos));
        if (matchingElim) {
          void (async () => {
            await engine.applyCanonicalMove(matchingElim);
            bumpSandboxTurn();
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          })();
          return;
        }
        // Click wasn't on an elimination target - don't process further
        return;
      }

      // RR-FIX-2026-01-11: Handle choose_territory_option moves for disconnected regions.
      // When there are disconnected territory regions to process, let the player click
      // anywhere to claim them. Auto-apply if there's only one option.
      const territoryOptionMoves = validMoves.filter((m) => m.type === 'choose_territory_option');
      if (territoryOptionMoves.length > 0) {
        if (territoryOptionMoves.length === 1) {
          // Only one region to claim - auto-apply on any click
          // Clear validTargets so elimination targets can be computed after the move
          setValidTargets([]);
          void (async () => {
            await engine.applyCanonicalMove(territoryOptionMoves[0]);
            bumpSandboxTurn();
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          })();
          return;
        }

        // Multiple regions - check if click is within any region's territory
        const territories = stateBefore.board.territories;
        for (const move of territoryOptionMoves) {
          // Find which region this move corresponds to by checking if the move's
          // target position is the representative position of a territory
          const moveTargetKey = move.to ? `${move.to.x},${move.to.y},${move.to.z}` : null;
          if (!moveTargetKey) continue;

          // Check each territory region to see if the clicked position is inside
          for (const [regionId, territory] of territories.entries()) {
            const spaces = territory.spaces ?? [];
            const clickedInRegion = spaces.some((space) => positionsEqual(space, pos));
            if (clickedInRegion) {
              // Check if this move's target is the representative of this region
              const representativeMatch = spaces.some(
                (space) => `${space.x},${space.y},${space.z}` === moveTargetKey
              );
              if (representativeMatch) {
                // Clear validTargets so elimination targets can be computed after the move
                setValidTargets([]);
                void (async () => {
                  await engine.applyCanonicalMove(move);
                  bumpSandboxTurn();
                  setSandboxStateVersion((v) => v + 1);
                  maybeRunSandboxAiIfNeeded();
                })();
                return;
              }
            }
          }
        }
        // Click wasn't in any claimable region - don't process further
        return;
      }

      // Fall through to normal handling if there are region choices (handled by handleDecisionClick)
    }

    // Line processing phase - auto-apply no_line_action when no lines to process
    // RR-FIX-2025-01-10: Handle stuck line_processing phase for human players
    if (phaseBefore === 'line_processing') {
      const validMoves = engine.getValidMoves(stateBefore.currentPlayer);

      // Check if the only valid move is no_line_action (no lines to process)
      const hasOnlyNoLineAction =
        validMoves.length === 1 && validMoves[0].type === 'no_line_action';

      if (hasOnlyNoLineAction) {
        // Auto-apply the no_line_action move to advance the game
        void (async () => {
          await engine.applyCanonicalMove(validMoves[0]);
          bumpSandboxTurn();
          setSandboxStateVersion((v) => v + 1);
          maybeRunSandboxAiIfNeeded();
        })();
        return;
      }

      // Check for elimination moves - let player click on their stacks
      const eliminationMoves = validMoves.filter((m) => m.type === 'eliminate_rings_from_stack');
      if (eliminationMoves.length > 0) {
        const matchingElim = eliminationMoves.find((m) => m.to && positionsEqual(m.to, pos));
        if (matchingElim) {
          void (async () => {
            await engine.applyCanonicalMove(matchingElim);
            bumpSandboxTurn();
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          })();
          return;
        }
        // Click wasn't on an elimination target - don't process further
        return;
      }

      // Fall through to normal handling if there are line choices
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
