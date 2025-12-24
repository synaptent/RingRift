/**
 * @fileoverview useSandboxRingPlacement Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It handles UI for ring placement interactions, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Placement logic: `src/shared/engine/aggregates/Placement.ts`
 *
 * This adapter:
 * - Double-click for quick 2-ring placement
 * - Context menu for custom ring count placement
 * - Ring placement count prompt state
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useCallback } from 'react';
import type { Position } from '../../shared/types/game';
import { positionToString } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';

export interface RingPlacementCountPromptState {
  position: Position;
  maxCount: number;
  isStackPlacement: boolean;
}

export interface UseSandboxRingPlacementOptions {
  setSelected: React.Dispatch<React.SetStateAction<Position | undefined>>;
  setValidTargets: React.Dispatch<React.SetStateAction<Position[]>>;
  bumpSandboxTurn: () => void;
  ringPlacementCountPrompt: RingPlacementCountPromptState | null;
  recoveryChoicePromptOpen: boolean;
}

export interface UseSandboxRingPlacementReturn {
  /** Ring placement count prompt state */
  ringPlacementCountPrompt: RingPlacementCountPromptState | null;
  /** Set the ring placement count prompt */
  setRingPlacementCountPrompt: React.Dispatch<
    React.SetStateAction<RingPlacementCountPromptState | null>
  >;
  /** Close the ring placement count prompt */
  closeRingPlacementCountPrompt: () => void;
  /** Confirm ring placement with specified count */
  confirmRingPlacementCountPrompt: (count: number) => void;
  /** Handle double-click for quick placement */
  handleCellDoubleClick: (pos: Position) => void;
  /** Handle context menu for custom placement */
  handleCellContextMenu: (pos: Position) => void;
}

/**
 * Hook for handling ring placement interactions in sandbox mode.
 */
export function useSandboxRingPlacement({
  setSelected,
  setValidTargets,
  bumpSandboxTurn,
  ringPlacementCountPrompt: externalPrompt,
  recoveryChoicePromptOpen,
}: UseSandboxRingPlacementOptions): UseSandboxRingPlacementReturn {
  const { sandboxEngine } = useSandbox();

  const [ringPlacementCountPrompt, setRingPlacementCountPrompt] =
    useState<RingPlacementCountPromptState | null>(null);

  const closeRingPlacementCountPrompt = useCallback(() => {
    setRingPlacementCountPrompt(null);
  }, []);

  const confirmRingPlacementCountPrompt = useCallback(
    (count: number) => {
      const engine = sandboxEngine;
      const prompt = ringPlacementCountPrompt;
      if (!engine || !prompt) {
        setRingPlacementCountPrompt(null);
        return;
      }

      setRingPlacementCountPrompt(null);

      void (async () => {
        const placed = await engine.tryPlaceRings(prompt.position, count);
        if (!placed) {
          return;
        }

        setSelected(prompt.position);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(prompt.position);
        setValidTargets(targets);
        bumpSandboxTurn();
      })();
    },
    [sandboxEngine, ringPlacementCountPrompt, setSelected, setValidTargets, bumpSandboxTurn]
  );

  /**
   * Sandbox double-click handler: implements the richer placement semantics
   * for the local sandbox during the ring_placement phase.
   *
   * - Empty cells: attempt a 2-ring placement (falling back to 1 ring if
   *   necessary) and then highlight movement targets from the new stack.
   * - Occupied cells: attempt a single-ring placement onto the stack and
   *   then highlight movement targets from that stack.
   */
  const handleCellDoubleClick = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine) return;

      // Use external prompt check if available, otherwise use internal state
      const activePrompt = externalPrompt ?? ringPlacementCountPrompt;
      if (activePrompt || recoveryChoicePromptOpen) {
        return;
      }

      const state = engine.getGameState();
      if (state.currentPhase !== 'ring_placement') {
        return;
      }

      const board = state.board;
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!player || player.ringsInHand <= 0) {
        return;
      }

      const isOccupied = !!stack && stack.rings.length > 0;
      const maxFromHand = player.ringsInHand;
      const maxPerPlacement = isOccupied ? 1 : maxFromHand;

      if (maxPerPlacement <= 0) {
        return;
      }

      void (async () => {
        let placed = false;

        if (!isOccupied) {
          // Empty cell: treat as a request to place 2 rings here in a single
          // placement action when possible.
          const desiredCount = Math.min(2, maxFromHand);
          placed = await engine.tryPlaceRings(pos, desiredCount);

          // If the desired multi-ring placement fails no-dead-placement checks,
          // fall back to a single-ring placement.
          if (!placed && desiredCount > 1) {
            placed = await engine.tryPlaceRings(pos, 1);
          }
        } else {
          // Existing stack: canonical rule is exactly 1 ring per placement.
          placed = await engine.tryPlaceRings(pos, 1);
        }

        if (!placed) {
          return;
        }

        // After a successful placement, we are now in the movement step for
        // this player, and the placed/updated stack must move. Highlight its
        // legal landing targets so the user can complete the turn.
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        bumpSandboxTurn();
      })();
    },
    [
      sandboxEngine,
      externalPrompt,
      ringPlacementCountPrompt,
      recoveryChoicePromptOpen,
      setSelected,
      setValidTargets,
      bumpSandboxTurn,
    ]
  );

  /**
   * Sandbox context-menu handler (right-click / long-press proxy): prompts
   * the user for a ring-count to place at the clicked position, then applies
   * that placement via tryPlaceRings when legal.
   */
  const handleCellContextMenu = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine) return;

      // Use external prompt check if available, otherwise use internal state
      const activePrompt = externalPrompt ?? ringPlacementCountPrompt;
      if (activePrompt || recoveryChoicePromptOpen) {
        return;
      }

      const state = engine.getGameState();
      if (state.currentPhase !== 'ring_placement') {
        return;
      }

      const board = state.board;
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!player || player.ringsInHand <= 0) {
        return;
      }

      const isOccupied = !!stack && stack.rings.length > 0;
      const maxFromHand = player.ringsInHand;
      const maxPerPlacement = isOccupied ? 1 : maxFromHand;

      if (maxPerPlacement <= 0) {
        return;
      }

      // If only 1 ring can be placed, do it directly without showing prompt
      if (maxPerPlacement <= 1) {
        void (async () => {
          const placed = await engine.tryPlaceRings(pos, 1);
          if (!placed) {
            return;
          }

          setSelected(pos);
          const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
          setValidTargets(targets);
          bumpSandboxTurn();
        })();
        return;
      }

      // Show prompt for custom ring count selection
      setRingPlacementCountPrompt({
        position: pos,
        maxCount: maxPerPlacement,
        isStackPlacement: isOccupied,
      });
    },
    [
      sandboxEngine,
      externalPrompt,
      ringPlacementCountPrompt,
      recoveryChoicePromptOpen,
      setSelected,
      setValidTargets,
      bumpSandboxTurn,
    ]
  );

  return {
    ringPlacementCountPrompt,
    setRingPlacementCountPrompt,
    closeRingPlacementCountPrompt,
    confirmRingPlacementCountPrompt,
    handleCellDoubleClick,
    handleCellContextMenu,
  };
}

export default useSandboxRingPlacement;
