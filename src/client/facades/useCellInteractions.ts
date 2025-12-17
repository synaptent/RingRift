/**
 * useCellInteractions - Shared cell interaction logic for game hosts
 *
 * This hook encapsulates the common cell click handling logic that was
 * previously duplicated between BackendGameHost and SandboxGameHost:
 * - Selection state management
 * - Valid target highlighting
 * - Move construction from clicks
 * - Invalid move detection and feedback
 *
 * @module facades/useCellInteractions
 */

import { useState, useCallback } from 'react';
import type { Move, Position, BoardState } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import type { GameFacade, PartialMove } from './GameFacade';
import { canInteract } from './GameFacade';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Options for cell interactions.
 */
export interface CellInteractionOptions {
  /** Callback when an invalid move is detected */
  onInvalidMove?: (position: Position, reason: string) => void;
  /** Callback when interaction is blocked (spectator, disconnected) */
  onInteractionBlocked?: (reason: string) => void;
  /**
   * Optional async prompt for selecting a custom ring placement count.
   * When omitted, the handler falls back to a sensible default (2 on empty
   * cells when legal, otherwise 1).
   */
  requestRingPlacementCount?: (context: {
    maxCount: number;
    hasStack: boolean;
    defaultCount: number;
  }) => Promise<number | null>;
}

/**
 * State returned by useCellInteractions.
 */
export interface CellInteractionState {
  /** Currently selected position */
  selected: Position | undefined;
  /** Valid target positions for the selected piece */
  validTargets: Position[];
  /** Effective selected position (includes must-move-from) */
  effectiveSelected: Position | undefined;
}

/**
 * Handlers returned by useCellInteractions.
 */
export interface CellInteractionHandlers {
  /** Handle cell click */
  handleCellClick: (position: Position, board: BoardState) => void;
  /** Handle cell double-click (for 2-ring placement) */
  handleCellDoubleClick: (position: Position, board: BoardState) => void;
  /** Handle cell context menu (for custom ring count) */
  handleCellContextMenu: (position: Position, board: BoardState) => void;
  /** Clear current selection */
  clearSelection: () => void;
  /** Set selection explicitly */
  setSelected: (position: Position | undefined) => void;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Shared cell interaction logic for game hosts.
 *
 * This hook manages selection state and translates cell clicks into moves,
 * handling the common patterns for ring placement, movement, and capture phases.
 */
export function useCellInteractions(
  facade: GameFacade | null,
  options: CellInteractionOptions = {}
): CellInteractionState & CellInteractionHandlers {
  const { onInvalidMove, onInteractionBlocked, requestRingPlacementCount } = options;

  // Selection state
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  // Effective selected position (includes must-move-from)
  const effectiveSelected = selected ?? facade?.mustMoveFrom;

  // Clear selection
  const clearSelection = useCallback(() => {
    setSelected(undefined);
    setValidTargets([]);
  }, []);

  // Handle cell click
  const handleCellClick = useCallback(
    (position: Position, board: BoardState) => {
      if (!facade?.gameState) {
        return;
      }

      const posKey = positionToString(position);

      // Check if interaction is allowed
      if (!canInteract(facade)) {
        const reason =
          facade.mode === 'spectator'
            ? 'Spectators cannot submit moves'
            : 'Moves paused while disconnected';
        onInteractionBlocked?.(reason);
        return;
      }

      const { gameState, validMoves, mustMoveFrom } = facade;

      // ─────────────────────────────────────────────────────────────────────
      // Ring Placement Phase
      // ─────────────────────────────────────────────────────────────────────
      if (gameState.currentPhase === 'ring_placement') {
        if (!validMoves || validMoves.length === 0) {
          return;
        }

        const hasStack = !!board.stacks.get(posKey);

        if (!hasStack) {
          // Clicking empty cell: attempt placement
          const placeMovesAtPos = validMoves.filter(
            (m) => m.type === 'place_ring' && positionsEqual(m.to, position)
          );

          if (placeMovesAtPos.length === 0) {
            onInvalidMove?.(position, 'Cannot place ring here');
            return;
          }

          // Prefer single-ring placement
          const preferred =
            placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];

          facade.submitMove({
            type: 'place_ring',
            to: preferred.to,
            placementCount: preferred.placementCount,
            placedOnStack: preferred.placedOnStack,
          } as PartialMove);

          clearSelection();
          return;
        }

        // Clicking stack in placement: just select it
        setSelected(position);
        setValidTargets([]);
        return;
      }

      // ─────────────────────────────────────────────────────────────────────
      // Movement/Capture Phases
      // ─────────────────────────────────────────────────────────────────────

      // No selection yet: try to select a stack with valid moves
      if (!selected) {
        if (!validMoves || validMoves.length === 0) {
          setSelected(position);
          setValidTargets([]);
          return;
        }

        const hasStack = !!board.stacks.get(posKey);
        const hasMovesFromHere = validMoves.some(
          (m) => m.from && positionsEqual(m.from as Position, position)
        );

        if (hasStack && hasMovesFromHere) {
          setSelected(position);
          const targets = validMoves
            .filter((m) => m.from && positionsEqual(m.from as Position, position))
            .map((m) => m.to);
          setValidTargets(targets);
        } else {
          onInvalidMove?.(
            position,
            analyzeInvalidSelection(gameState, position, {
              hasStack,
              hasMovesFromHere,
              mustMoveFrom,
            })
          );
        }
        return;
      }

      // Clicking same cell: clear selection
      if (positionsEqual(selected, position)) {
        clearSelection();
        return;
      }

      // Check for matching move from selected to clicked position
      if (validMoves && validMoves.length > 0) {
        const matching = validMoves.find(
          (m) => m.from && positionsEqual(m.from, selected) && positionsEqual(m.to, position)
        );

        if (matching) {
          facade.submitMove({
            type: matching.type,
            from: matching.from,
            to: matching.to,
          } as PartialMove);

          clearSelection();
          return;
        }
      }

      // Check if clicked position can become new selection
      const hasStack = !!board.stacks.get(posKey);
      const hasMovesFromHere =
        validMoves &&
        validMoves.some((m) => m.from && positionsEqual(m.from as Position, position));

      if (hasStack && hasMovesFromHere) {
        setSelected(position);
        if (validMoves && validMoves.length > 0) {
          const targets = validMoves
            .filter((m) => m.from && positionsEqual(m.from as Position, position))
            .map((m) => m.to);
          setValidTargets(targets);
        } else {
          setValidTargets([]);
        }
      } else {
        onInvalidMove?.(position, analyzeInvalidMove(gameState, selected, position, validMoves));
      }
    },
    [facade, selected, clearSelection, onInvalidMove, onInteractionBlocked]
  );

  // Handle cell double-click (prefer 2-ring placement)
  const handleCellDoubleClick = useCallback(
    (position: Position, board: BoardState) => {
      if (!facade?.gameState) {
        return;
      }

      if (!canInteract(facade)) {
        const reason =
          facade.mode === 'spectator'
            ? 'Cannot modify placements while spectating'
            : 'Cannot modify placements while disconnected';
        onInteractionBlocked?.(reason);
        return;
      }

      const { gameState, validMoves } = facade;

      if (gameState.currentPhase !== 'ring_placement') {
        return;
      }

      if (!validMoves || validMoves.length === 0) {
        return;
      }

      const key = positionToString(position);
      const hasStack = !!board.stacks.get(key);

      const placeMovesAtPos = validMoves.filter(
        (m) => m.type === 'place_ring' && positionsEqual(m.to, position)
      );

      if (placeMovesAtPos.length === 0) {
        return;
      }

      let chosen: Move | undefined;

      if (!hasStack) {
        // Prefer 2-ring on empty cell
        const twoRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 2);
        const oneRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1);
        chosen = twoRing || oneRing || placeMovesAtPos[0];
      } else {
        // Single ring on stack
        chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
      }

      if (!chosen) {
        return;
      }

      facade.submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount: chosen.placementCount,
        placedOnStack: chosen.placedOnStack,
      } as PartialMove);

      clearSelection();
    },
    [facade, clearSelection, onInteractionBlocked]
  );

  // Handle cell context menu (custom ring count)
  const handleCellContextMenu = useCallback(
    (position: Position, board: BoardState) => {
      if (!facade?.gameState) {
        return;
      }

      if (!canInteract(facade)) {
        const reason =
          facade.mode === 'spectator'
            ? 'Cannot modify placements while spectating'
            : 'Cannot modify placements while disconnected';
        onInteractionBlocked?.(reason);
        return;
      }

      const { gameState, validMoves } = facade;

      if (gameState.currentPhase !== 'ring_placement') {
        return;
      }

      if (!validMoves || validMoves.length === 0) {
        return;
      }

      const key = positionToString(position);
      const hasStack = !!board.stacks.get(key);

      const placeMovesAtPos = validMoves.filter(
        (m) => m.type === 'place_ring' && positionsEqual(m.to, position)
      );

      if (placeMovesAtPos.length === 0) {
        return;
      }

      const counts = placeMovesAtPos.map((m) => m.placementCount ?? 1);
      const maxCount = Math.max(...counts);
      const defaultCount = hasStack ? 1 : Math.min(2, maxCount);

      const submitPlacement = (count: number) => {
        const chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === count);
        if (!chosen) {
          return;
        }

        facade.submitMove({
          type: 'place_ring',
          to: chosen.to,
          placementCount: chosen.placementCount,
          placedOnStack: chosen.placedOnStack,
        } as PartialMove);

        clearSelection();
      };

      if (maxCount <= 1) {
        submitPlacement(1);
        return;
      }

      if (requestRingPlacementCount) {
        void requestRingPlacementCount({ maxCount, hasStack, defaultCount }).then((count) => {
          if (count === null || !Number.isFinite(count)) {
            return;
          }
          submitPlacement(Math.max(1, Math.min(maxCount, Math.floor(count))));
        });
        return;
      }

      submitPlacement(defaultCount);
    },
    [facade, clearSelection, onInteractionBlocked, requestRingPlacementCount]
  );

  return {
    // State
    selected,
    validTargets,
    effectiveSelected,
    // Handlers
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    clearSelection,
    setSelected,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Analyze why a selection is invalid.
 */
function analyzeInvalidSelection(
  gameState: { currentPhase: string; currentPlayer: number },
  position: Position,
  context: {
    hasStack: boolean;
    hasMovesFromHere: boolean;
    mustMoveFrom: Position | undefined;
  }
): string {
  const { hasStack, hasMovesFromHere, mustMoveFrom } = context;

  if (!hasStack) {
    return 'No stack at this position';
  }

  if (mustMoveFrom && !positionsEqual(position, mustMoveFrom)) {
    return 'You must move from the highlighted stack';
  }

  if (!hasMovesFromHere) {
    return 'No valid moves from this stack';
  }

  return 'Invalid selection';
}

/**
 * Analyze why a move is invalid.
 */
function analyzeInvalidMove(
  gameState: { currentPhase: string; currentPlayer: number },
  from: Position,
  to: Position,
  validMoves: Move[] | undefined
): string {
  if (!validMoves || validMoves.length === 0) {
    return 'No valid moves available';
  }

  const hasMatchingOrigin = validMoves.some((m) => m.from && positionsEqual(m.from, from));

  if (!hasMatchingOrigin) {
    return 'Cannot move from the selected position';
  }

  // Check if the destination is valid for any move from this origin
  const movesFromOrigin = validMoves.filter((m) => m.from && positionsEqual(m.from, from));

  if (movesFromOrigin.length > 0) {
    return 'Not a valid destination for this piece';
  }

  return 'Invalid move';
}
