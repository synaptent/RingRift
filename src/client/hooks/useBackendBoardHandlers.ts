/**
 * @fileoverview useBackendBoardHandlers Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** for backend game interactions.
 * It manages board click/interaction handlers, not rules logic.
 *
 * Canonical SSoT:
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - Movement logic: `src/shared/engine/aggregates/Movement.ts`
 * - Capture logic: `src/shared/engine/aggregates/Capture.ts`
 * - Placement logic: `src/shared/engine/aggregates/Placement.ts`
 *
 * This adapter:
 * - Handles cell click interactions for backend games
 * - Handles double-click for 2-ring placement
 * - Handles context-menu for ring placement count selection
 * - Manages ring placement count prompt state
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/BACKEND_GAME_HOST_DECOMPOSITION_PLAN.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useState, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import type { Position, GameState, Move, BoardState } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import type { PartialMove } from './useGameActions';
import { analyzeInvalidMove as analyzeInvalid } from './useInvalidMoveFeedback';

/**
 * Ring placement prompt state for the context-menu dialog.
 */
export interface RingPlacementPrompt {
  maxCount: number;
  hasStack: boolean;
  placeMovesAtPos: Move[];
}

/**
 * Dependencies required by the board handlers hook.
 */
export interface UseBackendBoardHandlersDeps {
  /** Current game state */
  gameState: GameState | null;
  /** Valid moves from the server */
  validMoves: Move[] | null;
  /** Currently selected position */
  selected: Position | undefined;
  /** Valid target positions */
  validTargets: Position[];
  /** Must-move-from position (when all moves from same source) */
  mustMoveFrom: Position | undefined;
  /** Function to set the selected position */
  setSelected: (pos: Position | undefined) => void;
  /** Function to set valid targets */
  setValidTargets: (targets: Position[]) => void;
  /** Function to submit a move to the backend */
  submitMove: (move: PartialMove) => void;
  /** Whether the current user is a player (not spectator) */
  isPlayer: boolean;
  /** Whether the WebSocket connection is active */
  isConnectionActive: boolean;
  /** Whether it's the current user's turn */
  isMyTurn: boolean;
  /** Function to trigger invalid move feedback */
  triggerInvalidMove: (pos: Position, reason: string) => void;
}

/**
 * Return type for useBackendBoardHandlers hook.
 */
export interface UseBackendBoardHandlersReturn {
  /** Current ring placement count prompt state */
  ringPlacementCountPrompt: RingPlacementPrompt | null;
  /** Handle cell click */
  handleCellClick: (pos: Position, board: BoardState) => void;
  /** Handle cell double-click */
  handleCellDoubleClick: (pos: Position, board: BoardState) => void;
  /** Handle cell context-menu (right-click) */
  handleCellContextMenu: (pos: Position, board: BoardState) => void;
  /** Handle confirming ring placement count from dialog */
  handleConfirmRingPlacementCount: (count: number) => void;
  /** Close the ring placement prompt dialog */
  closeRingPlacementPrompt: () => void;
}

/**
 * Custom hook for managing backend game board interaction handlers.
 *
 * Handles:
 * - Cell click for selection and move execution
 * - Double-click for 2-ring placement
 * - Context-menu for custom ring placement counts
 * - Ring placement count dialog state
 *
 * Extracted from BackendGameHost to reduce component complexity.
 *
 * @param deps - Dependencies including game state, selection state, and actions
 * @returns Object with handlers and prompt state
 */
export function useBackendBoardHandlers(
  deps: UseBackendBoardHandlersDeps
): UseBackendBoardHandlersReturn {
  const {
    gameState,
    validMoves,
    selected,
    mustMoveFrom,
    setSelected,
    setValidTargets,
    submitMove,
    isPlayer,
    isConnectionActive,
    isMyTurn,
    triggerInvalidMove,
  } = deps;

  // Ring placement count prompt state
  const [ringPlacementCountPrompt, setRingPlacementCountPrompt] =
    useState<RingPlacementPrompt | null>(null);

  // Close the ring placement prompt
  const closeRingPlacementPrompt = useCallback(() => {
    setRingPlacementCountPrompt(null);
  }, []);

  // Handle cell click
  const handleCellClick = useCallback(
    (pos: Position, board: BoardState) => {
      if (!gameState) return;
      const posKey = positionToString(pos);

      if (!isPlayer) {
        toast.error('Spectators cannot submit moves', { id: 'interaction-locked' });
        return;
      }

      if (!isConnectionActive) {
        toast.error('Moves paused while disconnected', { id: 'interaction-locked' });
        return;
      }

      // Ring placement phase: attempt canonical 1-ring placement on empties
      if (gameState.currentPhase === 'ring_placement') {
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          return;
        }

        const hasStack = !!board.stacks.get(posKey);

        if (!hasStack) {
          const placeMovesAtPos = validMoves.filter(
            (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
          );
          if (placeMovesAtPos.length === 0) {
            // Use enhanced invalid move feedback with shake animation and explanatory toast
            const reason = analyzeInvalid(gameState, pos, {
              isPlayer,
              isMyTurn,
              isConnected: isConnectionActive,
              selectedPosition: selected,
              validMoves: validMoves ?? undefined,
              mustMoveFrom,
            });
            triggerInvalidMove(pos, reason);
            return;
          }

          const preferred =
            placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];

          submitMove({
            type: 'place_ring',
            to: preferred.to,
            placementCount: preferred.placementCount,
            placedOnStack: preferred.placedOnStack,
          } as PartialMove);

          setSelected(undefined);
          setValidTargets([]);
          return;
        }

        // Clicking stacks in placement phase just selects them.
        setSelected(pos);
        setValidTargets([]);
        return;
      }

      // Movement/capture phases: select source, then target.
      if (!selected) {
        // When there are no valid moves at all, keep the previous behaviour and
        // simply allow selection without special feedback.
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          setSelected(pos);
          setValidTargets([]);
          return;
        }

        const hasStack = !!board.stacks.get(posKey);
        const hasMovesFromHere = validMoves.some(
          (m) => m.from && positionsEqual(m.from as Position, pos)
        );

        if (hasStack && hasMovesFromHere) {
          setSelected(pos);
          const targets = validMoves
            .filter((m) => m.from && positionsEqual(m.from as Position, pos))
            .map((m) => m.to);
          setValidTargets(targets);
        } else {
          const reason = analyzeInvalid(gameState, pos, {
            isPlayer,
            isMyTurn,
            isConnected: isConnectionActive,
            selectedPosition: null,
            validMoves: validMoves ?? undefined,
            mustMoveFrom,
          });
          triggerInvalidMove(pos, reason);
        }
        return;
      }

      // Clicking the same cell clears selection.
      if (positionsEqual(selected, pos)) {
        setSelected(undefined);
        setValidTargets([]);
        return;
      }

      // If highlighted and a matching move exists, submit.
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const matching = validMoves.find(
          (m) => m.from && positionsEqual(m.from, selected) && positionsEqual(m.to, pos)
        );

        if (matching) {
          submitMove({
            type: matching.type,
            from: matching.from,
            to: matching.to,
          } as PartialMove);

          setSelected(undefined);
          setValidTargets([]);
          return;
        }
      }

      // Otherwise, treat either as a new (valid) selection or as an invalid
      // landing/selection and surface feedback.
      const hasStack = !!board.stacks.get(posKey);
      const hasMovesFromHere =
        Array.isArray(validMoves) &&
        validMoves.some((m) => m.from && positionsEqual(m.from as Position, pos));

      if (hasStack && hasMovesFromHere) {
        setSelected(pos);
        if (Array.isArray(validMoves) && validMoves.length > 0) {
          const targets = validMoves
            .filter((m) => m.from && positionsEqual(m.from as Position, pos))
            .map((m) => m.to);
          setValidTargets(targets);
        } else {
          setValidTargets([]);
        }
      } else {
        const reason = analyzeInvalid(gameState, pos, {
          isPlayer,
          isMyTurn,
          isConnected: isConnectionActive,
          selectedPosition: selected ?? null,
          validMoves: validMoves ?? undefined,
          mustMoveFrom,
        });
        triggerInvalidMove(pos, reason);
      }
    },
    [
      gameState,
      validMoves,
      selected,
      mustMoveFrom,
      setSelected,
      setValidTargets,
      submitMove,
      isPlayer,
      isConnectionActive,
      isMyTurn,
      triggerInvalidMove,
    ]
  );

  // Handle double-click for 2-ring placement
  const handleCellDoubleClick = useCallback(
    (pos: Position, board: BoardState) => {
      if (!gameState) return;
      if (!isPlayer || !isConnectionActive) {
        toast.error('Cannot modify placements while disconnected or spectating', {
          id: 'interaction-locked',
        });
        return;
      }
      if (gameState.currentPhase !== 'ring_placement') {
        return;
      }

      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        return;
      }

      const key = positionToString(pos);
      const hasStack = !!board.stacks.get(key);

      const placeMovesAtPos = validMoves.filter(
        (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
      );
      if (placeMovesAtPos.length === 0) {
        return;
      }

      let chosen: Move | undefined;

      if (!hasStack) {
        const twoRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 2);
        const oneRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1);
        chosen = twoRing || oneRing || placeMovesAtPos[0];
      } else {
        chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
      }

      if (!chosen) {
        return;
      }

      submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount: chosen.placementCount,
        placedOnStack: chosen.placedOnStack,
      } as PartialMove);

      setSelected(undefined);
      setValidTargets([]);
    },
    [gameState, validMoves, isPlayer, isConnectionActive, submitMove, setSelected, setValidTargets]
  );

  // Handle context-menu for ring placement count selection
  const handleCellContextMenu = useCallback(
    (pos: Position, board: BoardState) => {
      if (!gameState) return;
      if (!isPlayer || !isConnectionActive) {
        toast.error('Cannot modify placements while disconnected or spectating', {
          id: 'interaction-locked',
        });
        return;
      }
      if (gameState.currentPhase !== 'ring_placement') {
        return;
      }

      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        return;
      }

      const key = positionToString(pos);
      const hasStack = !!board.stacks.get(key);

      const placeMovesAtPos = validMoves.filter(
        (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
      );
      if (placeMovesAtPos.length === 0) {
        return;
      }

      const counts = placeMovesAtPos.map((m) => m.placementCount ?? 1);
      const maxCount = Math.max(...counts);

      if (maxCount <= 1) {
        const chosen =
          placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
        if (!chosen) return;

        submitMove({
          type: 'place_ring',
          to: chosen.to,
          placementCount: chosen.placementCount,
          placedOnStack: chosen.placedOnStack,
        } as PartialMove);

        setSelected(undefined);
        setValidTargets([]);
        return;
      }

      setRingPlacementCountPrompt({
        maxCount,
        hasStack,
        placeMovesAtPos,
      });
    },
    [gameState, validMoves, isPlayer, isConnectionActive, submitMove, setSelected, setValidTargets]
  );

  // Handle confirming ring placement count from dialog
  const handleConfirmRingPlacementCount = useCallback(
    (count: number) => {
      const prompt = ringPlacementCountPrompt;
      if (!prompt) return;

      const chosen = prompt.placeMovesAtPos.find((m) => (m.placementCount ?? 1) === count);
      if (!chosen) {
        setRingPlacementCountPrompt(null);
        return;
      }

      submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount: chosen.placementCount,
        placedOnStack: chosen.placedOnStack,
      } as PartialMove);

      setSelected(undefined);
      setValidTargets([]);
      setRingPlacementCountPrompt(null);
    },
    [ringPlacementCountPrompt, submitMove, setSelected, setValidTargets]
  );

  return {
    ringPlacementCountPrompt,
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    handleConfirmRingPlacementCount,
    closeRingPlacementPrompt,
  };
}

export default useBackendBoardHandlers;
