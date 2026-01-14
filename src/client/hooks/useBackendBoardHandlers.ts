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

import { useState, useCallback, useEffect, useRef } from 'react';
import { toast } from 'react-hot-toast';
import type {
  Position,
  GameState,
  Move,
  BoardState,
  PlayerChoice,
  BoardType,
} from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import {
  enumerateSimpleMoveTargetsFromStack,
  isValidPosition,
  type MovementBoardView,
} from '../../shared/engine';
import type { PartialMove } from './useGameActions';
import {
  analyzeInvalidMove as analyzeInvalid,
  type InvalidMoveReason,
} from './useInvalidMoveFeedback';
import type { TerritoryRegionOption } from '../components/TerritoryRegionChoiceDialog';

/**
 * Create a MovementBoardView adapter from a BoardState for computing valid landing positions.
 * Used by skip_placement shortcut to validate targets before sending to server.
 */
function createMovementBoardView(board: BoardState): MovementBoardView {
  const boardType = board.type as BoardType;
  const size = board.size;

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      const stack = board.stacks.get(key);
      if (!stack) {
        return undefined;
      }
      return {
        controllingPlayer: stack.controllingPlayer,
        capHeight: stack.capHeight,
        stackHeight: stack.stackHeight,
      };
    },
    getMarkerOwner: (pos: Position) => {
      const key = positionToString(pos);
      const marker = board.markers.get(key);
      return marker?.player;
    },
  };
}

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
  triggerInvalidMove: (pos: Position, reason: InvalidMoveReason) => void;
  /** Function to request a recovery choice from the user (for overlength recovery lines) */
  requestRecoveryChoice?: () => Promise<'option1' | 'option2' | null>;
  /** Pending player choice from server (for territory region selection, etc.) */
  pendingChoice?: PlayerChoice | null;
  /** Function to respond to a player choice */
  onRespondToChoice?: <TChoice extends PlayerChoice>(
    choice: TChoice,
    option: TChoice['options'][number]
  ) => void;
}

/**
 * Territory region prompt state for disambiguation dialog.
 */
export interface TerritoryRegionPromptState {
  options: TerritoryRegionOption[];
  clickedPosition: Position;
}

/**
 * Return type for useBackendBoardHandlers hook.
 */
export interface UseBackendBoardHandlersReturn {
  /** Current ring placement count prompt state */
  ringPlacementCountPrompt: RingPlacementPrompt | null;
  /** Current territory region prompt state (for disambiguation) */
  territoryRegionPrompt: TerritoryRegionPromptState | null;
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
  /** Handle confirming territory region selection from dialog */
  handleConfirmTerritoryRegion: (option: TerritoryRegionOption) => void;
  /** Close the territory region prompt dialog */
  closeTerritoryRegionPrompt: () => void;
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
    requestRecoveryChoice,
    pendingChoice,
    onRespondToChoice,
  } = deps;

  // Ring placement count prompt state
  const [ringPlacementCountPrompt, setRingPlacementCountPrompt] =
    useState<RingPlacementPrompt | null>(null);

  // Territory region prompt state (for disambiguation when multiple regions overlap)
  const [territoryRegionPrompt, setTerritoryRegionPrompt] =
    useState<TerritoryRegionPromptState | null>(null);

  // Pending movement target for skip_placement + movement shortcut
  // When user clicks a valid landing during placement phase, we submit skip_placement
  // and store the intended movement here. After phase changes to movement, we auto-submit.
  const pendingMovementRef = useRef<{ from: Position; to: Position; timestamp: number } | null>(
    null
  );
  const prevPhaseRef = useRef<string | null>(null);
  const pendingMovementRetryCount = useRef<number>(0);

  // Effect to handle pending movement after skip_placement
  // Includes retry logic for when validMoves might be stale after phase change
  useEffect(() => {
    const currentPhase = gameState?.currentPhase;
    const prevPhase = prevPhaseRef.current;
    prevPhaseRef.current = currentPhase ?? null;

    // Clear stale pending movements (older than 5 seconds)
    if (pendingMovementRef.current) {
      const age = Date.now() - pendingMovementRef.current.timestamp;
      if (age > 5000) {
        console.warn('[PendingMovement] Clearing stale pending movement after 5s timeout');
        pendingMovementRef.current = null;
        pendingMovementRetryCount.current = 0;
        return;
      }
    }

    // If phase changed to movement and we have a pending target
    const shouldTryPendingMovement =
      currentPhase === 'movement' && pendingMovementRef.current && validMoves;

    if (!shouldTryPendingMovement) {
      // Reset retry count if we're no longer in movement phase or no pending movement
      if (currentPhase !== 'movement' || !pendingMovementRef.current) {
        pendingMovementRetryCount.current = 0;
      }
      return;
    }

    const pending = pendingMovementRef.current!;

    // Find the matching move_stack move
    const moveStackMove = validMoves.find(
      (m) =>
        m.type === 'move_stack' &&
        m.from &&
        positionsEqual(m.from, pending.from) &&
        m.to &&
        positionsEqual(m.to, pending.to)
    );

    if (moveStackMove) {
      // Success! Submit the move and clear pending state
      pendingMovementRef.current = null;
      pendingMovementRetryCount.current = 0;
      submitMove({
        type: moveStackMove.type,
        from: moveStackMove.from,
        to: moveStackMove.to,
      } as PartialMove);
      setSelected(undefined);
      setValidTargets([]);
    } else if (pendingMovementRetryCount.current < 3) {
      // Move not found in validMoves - this may be because validMoves is stale
      // Schedule a retry after a brief delay to allow state to settle
      pendingMovementRetryCount.current += 1;
      console.log(
        '[PendingMovement] Move not found, scheduling retry',
        pendingMovementRetryCount.current
      );
      // The effect will re-run when validMoves updates
    } else {
      // Max retries reached, clear the pending movement
      console.warn('[PendingMovement] Max retries reached, clearing pending movement');
      pendingMovementRef.current = null;
      pendingMovementRetryCount.current = 0;
    }
  }, [gameState?.currentPhase, validMoves, submitMove, setSelected, setValidTargets]);

  // Close the ring placement prompt
  const closeRingPlacementPrompt = useCallback(() => {
    setRingPlacementCountPrompt(null);
  }, []);

  // Close the territory region prompt
  const closeTerritoryRegionPrompt = useCallback(() => {
    setTerritoryRegionPrompt(null);
  }, []);

  // Handle confirming territory region selection from dialog
  const handleConfirmTerritoryRegion = useCallback(
    (selectedOption: TerritoryRegionOption) => {
      if (!pendingChoice || pendingChoice.type !== 'region_order' || !onRespondToChoice) {
        setTerritoryRegionPrompt(null);
        return;
      }

      onRespondToChoice(pendingChoice, selectedOption);
      setTerritoryRegionPrompt(null);
    },
    [pendingChoice, onRespondToChoice]
  );

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

      // Handle pending player choice (region_order, ring_elimination, line_reward_option)
      if (pendingChoice && onRespondToChoice) {
        // Handle region_order choice (territory region selection)
        if (pendingChoice.type === 'region_order') {
          const options = (pendingChoice.options ?? []) as TerritoryRegionOption[];
          if (options.length === 0) {
            // No options - let other handlers try
          } else {
            // Find which options contain the clicked position using spaces
            const optionsWithSpaces = options.filter((opt) => opt.spaces && opt.spaces.length > 0);
            if (optionsWithSpaces.length > 0) {
              const matchingBySpaces = optionsWithSpaces.filter((opt) =>
                opt.spaces!.some((space) => positionsEqual(space, pos))
              );

              if (matchingBySpaces.length === 1) {
                // Single match - respond directly
                onRespondToChoice(pendingChoice, matchingBySpaces[0]);
                return;
              } else if (matchingBySpaces.length > 1) {
                // Multiple regions overlap - show disambiguation dialog
                setTerritoryRegionPrompt({
                  options: matchingBySpaces,
                  clickedPosition: pos,
                });
                return;
              }
            }

            // Fallback: check representative positions
            const matchingByRepresentative = options.find((opt) =>
              positionsEqual(opt.representativePosition, pos)
            );
            if (matchingByRepresentative) {
              onRespondToChoice(pendingChoice, matchingByRepresentative);
              return;
            }

            // Click was not on any region - don't block other handlers
          }
        }

        // Handle ring_elimination choice
        if (pendingChoice.type === 'ring_elimination') {
          const options = (pendingChoice.options ?? []) as Array<{ stackPosition: Position }>;
          const matching = options.find((opt) => positionsEqual(opt.stackPosition, pos));
          if (matching) {
            onRespondToChoice(pendingChoice, matching);
            return;
          }
          // If valid options exist, block fall-through (must click a valid target)
          if (options.length > 0) {
            return;
          }
        }

        // Handle line_reward_option choice with segments
        if (
          pendingChoice.type === 'line_reward_option' &&
          pendingChoice.segments &&
          pendingChoice.segments.length > 0
        ) {
          const segments = pendingChoice.segments;
          const clickedSegment = segments.find((segment) =>
            segment.positions.some((p) => positionsEqual(p, pos))
          );
          if (clickedSegment) {
            onRespondToChoice(pendingChoice, { optionId: clickedSegment.optionId });
            return;
          }
          // Click was not on a highlighted segment - don't block other handlers
        }
      }

      // Ring placement phase: attempt canonical 1-ring placement on empties
      if (gameState.currentPhase === 'ring_placement') {
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          return;
        }

        const hasStack = !!board.stacks.get(posKey);

        // Skip placement + movement shortcut: if user has selected a stack they control
        // and clicks on a valid landing position, chain skip_placement + move_stack
        if (selected && !positionsEqual(selected, pos)) {
          const selectedKey = positionToString(selected);
          const selectedStack = board.stacks.get(selectedKey);

          if (selectedStack && selectedStack.controllingPlayer === gameState.currentPlayer) {
            // Check if there's a skip_placement move available
            const skipPlacementMove = validMoves.find((m) => m.type === 'skip_placement');

            // Compute valid landing positions using the shared movement logic
            // This validates direction, distance, and path clearance properly
            const boardView = createMovementBoardView(board);
            const validLandings = enumerateSimpleMoveTargetsFromStack(
              board.type as BoardType,
              selected,
              gameState.currentPlayer,
              boardView
            );
            const isValidLanding = validLandings.some((t) => positionsEqual(t.to, pos));

            if (skipPlacementMove && isValidLanding) {
              // Store the pending movement and submit skip_placement
              pendingMovementRef.current = { from: selected, to: pos, timestamp: Date.now() };

              submitMove({
                type: 'skip_placement',
                to: { x: 0, y: 0 }, // skip_placement doesn't use 'to' meaningfully
              } as PartialMove);

              // Keep selection for the movement phase
              return;
            }
          }
        }

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

      // Chain capture phase: clicking valid landing positions continues the chain
      if (gameState.currentPhase === 'chain_capture') {
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          return;
        }

        // Look for continue_capture_segment or overtaking_capture moves to this position
        const chainMoves = validMoves.filter(
          (m) =>
            (m.type === 'continue_capture_segment' || m.type === 'overtaking_capture') &&
            m.to &&
            positionsEqual(m.to, pos)
        );

        if (chainMoves.length > 0) {
          const chainMove = chainMoves[0];
          submitMove({
            type: chainMove.type,
            from: chainMove.from,
            to: chainMove.to,
          } as PartialMove);
          // Don't clear selection - let the auto-selection effect handle the next capture
          return;
        }

        // Also check for skip_capture move if clicking outside valid targets
        const skipMove = validMoves.find((m) => m.type === 'skip_capture');
        if (skipMove) {
          // Let user click elsewhere to potentially skip (or just don't submit)
          // For now, show feedback that this isn't a valid chain capture target
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
        return;
      }

      // Elimination phases: clicking valid elimination targets submits the elimination
      if (
        gameState.currentPhase === 'forced_elimination' ||
        gameState.currentPhase === 'line_processing' ||
        gameState.currentPhase === 'territory_processing'
      ) {
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          return;
        }

        // Check for eliminate_rings_from_stack moves targeting this position
        const elimMoves = validMoves.filter(
          (m) => m.type === 'eliminate_rings_from_stack' && m.to && positionsEqual(m.to, pos)
        );

        if (elimMoves.length > 0) {
          const elimMove = elimMoves[0];
          submitMove({
            type: elimMove.type,
            to: elimMove.to,
            eliminationCount: elimMove.eliminationCount,
            targetPlayer: elimMove.targetPlayer,
          } as PartialMove);
          setSelected(undefined);
          setValidTargets([]);
          return;
        }

        // If clicking elsewhere during elimination, show feedback
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

      // Check for recovery_slide moves first (may need disambiguation)
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const recoveryMoves = validMoves.filter(
          (m) =>
            m.type === 'recovery_slide' &&
            m.from &&
            positionsEqual(m.from, selected) &&
            m.to &&
            positionsEqual(m.to, pos)
        );

        if (recoveryMoves.length > 0) {
          // Check if we have multiple recovery options (option1 vs option2)
          const option1Move = recoveryMoves.find((m) => m.recoveryOption === 1);
          const option2Move = recoveryMoves.find((m) => m.recoveryOption === 2);

          if (option1Move && option2Move && requestRecoveryChoice) {
            // Multiple options - request choice from user
            void requestRecoveryChoice().then((choice) => {
              if (choice === 'option1') {
                submitMove({
                  type: option1Move.type,
                  from: option1Move.from,
                  to: option1Move.to,
                  recoveryOption: option1Move.recoveryOption,
                } as PartialMove);
              } else if (choice === 'option2') {
                submitMove({
                  type: option2Move.type,
                  from: option2Move.from,
                  to: option2Move.to,
                  recoveryOption: option2Move.recoveryOption,
                } as PartialMove);
              }
              // If null (cancelled), do nothing - keep selection
              if (choice !== null) {
                setSelected(undefined);
                setValidTargets([]);
              }
            });
            return;
          }

          // Single option or no requestRecoveryChoice - submit directly
          const selectedMove = recoveryMoves[0];
          submitMove({
            type: selectedMove.type,
            from: selectedMove.from,
            to: selectedMove.to,
            recoveryOption: selectedMove.recoveryOption,
          } as PartialMove);
          setSelected(undefined);
          setValidTargets([]);
          return;
        }
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
      requestRecoveryChoice,
      pendingChoice,
      onRespondToChoice,
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
    territoryRegionPrompt,
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    handleConfirmRingPlacementCount,
    closeRingPlacementPrompt,
    handleConfirmTerritoryRegion,
    closeTerritoryRegionPrompt,
  };
}

export default useBackendBoardHandlers;
