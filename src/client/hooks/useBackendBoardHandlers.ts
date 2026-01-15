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

import { useState, useCallback, useEffect, useLayoutEffect, useRef } from 'react';
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
  enumerateCaptureMoves,
  isValidPosition,
  type MovementBoardView,
  type CaptureBoardAdapters,
} from '../../shared/engine';
import type { PartialMove } from './useGameActions';
import {
  analyzeInvalidMove as analyzeInvalid,
  type InvalidMoveReason,
} from './useInvalidMoveFeedback';
import type { TerritoryRegionOption } from '../components/TerritoryRegionChoiceDialog';

/**
 * Extract captureTarget from a move object if it's a capture move.
 * Mirrors how sandbox naturally gets captureTarget from aggregates.
 */
function extractCaptureTarget(move: Move | PartialMove): Position | undefined {
  if ('captureTarget' in move && move.captureTarget) {
    return move.captureTarget as Position;
  }
  return undefined;
}

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
 * Create a MovementBoardView adapter with a simulated stack at a given position.
 * Used for computing valid landing positions during pending ring placement,
 * where no actual stack exists yet on the board.
 */
function createMovementBoardViewWithSimulatedStack(
  board: BoardState,
  simulatedPos: Position,
  simulatedPlayer: number,
  simulatedStackHeight: number
): MovementBoardView {
  const boardType = board.type as BoardType;
  const size = board.size;
  const simulatedPosKey = positionToString(simulatedPos);

  return {
    isValidPosition: (pos: Position) => isValidPosition(pos, boardType, size),
    isCollapsedSpace: (pos: Position) => board.collapsedSpaces.has(positionToString(pos)),
    getStackAt: (pos: Position) => {
      const key = positionToString(pos);
      // Return simulated stack for the pending position
      if (key === simulatedPosKey) {
        return {
          controllingPlayer: simulatedPlayer,
          capHeight: simulatedStackHeight,
          stackHeight: simulatedStackHeight,
        };
      }
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
 * Compute all valid landing positions from a simulated stack, including both
 * simple movement targets and capture landing positions.
 *
 * Per RR-CANON rules, direct capture is allowed immediately after placement -
 * the player doesn't need to make a simple move first.
 */
function computeValidLandingsWithCaptures(
  board: BoardState,
  fromPos: Position,
  playerNumber: number,
  simulatedStackHeight: number
): Position[] {
  const boardType = board.type as BoardType;

  // Create simulated board view for this pending placement
  const simulatedBoardView = createMovementBoardViewWithSimulatedStack(
    board,
    fromPos,
    playerNumber,
    simulatedStackHeight
  );

  // Get simple movement targets
  const simpleLandings = enumerateSimpleMoveTargetsFromStack(
    boardType,
    fromPos,
    playerNumber,
    simulatedBoardView
  );

  // Get capture landing targets using the same simulated view
  // (CaptureBoardAdapters is compatible with MovementBoardView)
  const captureMoves = enumerateCaptureMoves(
    boardType,
    fromPos,
    playerNumber,
    simulatedBoardView as CaptureBoardAdapters,
    0 // moveNumber not used for enumeration
  );

  // Combine both sets of landing positions, deduplicating
  const landingSet = new Set<string>();
  const allLandings: Position[] = [];

  for (const landing of simpleLandings) {
    const key = positionToString(landing.to);
    if (!landingSet.has(key)) {
      landingSet.add(key);
      allLandings.push(landing.to);
    }
  }

  for (const capture of captureMoves) {
    const key = positionToString(capture.to);
    if (!landingSet.has(key)) {
      landingSet.add(key);
      allLandings.push(capture.to);
    }
  }

  return allLandings;
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
 * Pending ring placement state for click-to-increment feature.
 * Each click on the same cell increments the count before submitting.
 */
export interface PendingRingPlacement {
  position: Position;
  positionKey: string;
  currentCount: number;
  maxCount: number;
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
  /** Pending ring placement for click-to-increment feature */
  pendingRingPlacement: PendingRingPlacement | null;
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
  /** Confirm the pending ring placement */
  confirmPendingRingPlacement: () => void;
  /** Cancel the pending ring placement */
  clearPendingRingPlacement: () => void;
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

  // Pending ring placement for click-to-increment feature
  // Each click on the same cell increments the count before confirming
  const [pendingRingPlacement, setPendingRingPlacement] = useState<PendingRingPlacement | null>(
    null
  );

  // Pending movement target for skip_placement + movement shortcut
  // When user clicks a valid landing during placement phase, we submit skip_placement
  // and store the intended movement here. After phase changes to movement, we auto-submit.
  const pendingMovementRef = useRef<{ from: Position; to: Position; timestamp: number } | null>(
    null
  );
  const prevPhaseRef = useRef<string | null>(null);
  const pendingMovementRetryCount = useRef<number>(0);

  // Refs for pendingChoice and onRespondToChoice to avoid stale closure issues
  // The handleCellClick callback can capture stale values; refs always have current values
  // Use useLayoutEffect to ensure refs are updated synchronously before any user interaction
  const pendingChoiceRef = useRef(pendingChoice);
  const onRespondToChoiceRef = useRef(onRespondToChoice);
  useLayoutEffect(() => {
    pendingChoiceRef.current = pendingChoice;
    onRespondToChoiceRef.current = onRespondToChoice;
  }, [pendingChoice, onRespondToChoice]);

  // Effect to handle pending movement after skip_placement
  // Includes retry logic for when validMoves might be stale after phase change
  useEffect(() => {
    const currentPhase = gameState?.currentPhase;
    const _prevPhase = prevPhaseRef.current;
    prevPhaseRef.current = currentPhase ?? null;

    // Clear stale pending movements (older than 10 seconds, forgiving of slow networks)
    if (pendingMovementRef.current) {
      const age = Date.now() - pendingMovementRef.current.timestamp;
      if (age > 10000) {
        console.warn('[PendingMovement] Clearing stale pending movement after 10s timeout');
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

    const pending = pendingMovementRef.current;
    if (!pending) {
      return;
    }

    // Find the matching move_stack or overtaking_capture move
    // (capture targets use overtaking_capture type, not move_stack)
    const pendingMove = validMoves.find(
      (m) =>
        (m.type === 'move_stack' || m.type === 'overtaking_capture') &&
        m.from &&
        positionsEqual(m.from, pending.from) &&
        m.to &&
        positionsEqual(m.to, pending.to)
    );

    if (pendingMove) {
      // Success! Submit the move and clear pending state
      // For capture moves, include captureTarget from the server's move
      const captureTarget = extractCaptureTarget(pendingMove);
      pendingMovementRef.current = null;
      pendingMovementRetryCount.current = 0;
      submitMove({
        type: pendingMove.type,
        from: pendingMove.from,
        to: pendingMove.to,
        captureTarget,
      } as PartialMove);
      setSelected(undefined);
      setValidTargets([]);
    } else if (pendingMovementRetryCount.current < 5) {
      // Move not found in validMoves - this may be because validMoves is stale
      // Schedule a retry after a brief delay to allow state to settle
      pendingMovementRetryCount.current += 1;
      // The effect will re-run when validMoves updates
    } else {
      // Max retries reached, clear the pending movement
      console.warn('[PendingMovement] Max retries reached, clearing pending movement');
      pendingMovementRef.current = null;
      pendingMovementRetryCount.current = 0;
    }
  }, [gameState?.currentPhase, validMoves, submitMove, setSelected, setValidTargets]);

  // Effect to clear pending ring placement when phase changes away from ring_placement
  useEffect(() => {
    if (gameState?.currentPhase !== 'ring_placement' && pendingRingPlacement) {
      setPendingRingPlacement(null);
    }
  }, [gameState?.currentPhase, pendingRingPlacement]);

  // Effect to auto-select capture source when entering capture or chain_capture phase
  // Mirrors sandbox's direct landing click pattern - user only needs to click the target
  useEffect(() => {
    const phase = gameState?.currentPhase;
    if (
      (phase === 'chain_capture' || phase === 'capture') &&
      Array.isArray(validMoves) &&
      validMoves.length > 0
    ) {
      // Filter for capture moves based on phase
      const captureMoves = validMoves.filter((m) =>
        phase === 'chain_capture'
          ? m.type === 'continue_capture_segment' || m.type === 'overtaking_capture'
          : m.type === 'overtaking_capture'
      );

      if (captureMoves.length > 0 && captureMoves[0].from) {
        const from = captureMoves[0].from as Position;
        const fromKey = positionToString(from);

        // RR-FIX-2026-01-15: Filter to only moves from the expected position.
        // During chain_capture, all moves MUST come from the chain capture position.
        // This defensive check ensures we don't highlight wrong landing positions
        // if the server ever returns moves from multiple stacks.
        const movesFromPosition = captureMoves.filter(
          (m) => m.from && positionToString(m.from) === fromKey
        );

        // Deduplicate landing positions
        const landingSet = new Set<string>();
        const landings: Position[] = [];
        for (const m of movesFromPosition) {
          if (m.to) {
            const key = positionToString(m.to);
            if (!landingSet.has(key)) {
              landingSet.add(key);
              landings.push(m.to);
            }
          }
        }
        setSelected(from);
        setValidTargets(landings);
      }
    }
  }, [gameState?.currentPhase, validMoves, setSelected, setValidTargets]);

  // Confirm the pending ring placement
  const confirmPendingRingPlacement = useCallback(() => {
    if (!pendingRingPlacement) return;

    const chosen = pendingRingPlacement.placeMovesAtPos.find(
      (m) => (m.placementCount ?? 1) === pendingRingPlacement.currentCount
    );

    if (chosen) {
      submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount: chosen.placementCount,
        placedOnStack: chosen.placedOnStack,
      } as PartialMove);
    }

    setPendingRingPlacement(null);
    setSelected(undefined);
    setValidTargets([]);
  }, [pendingRingPlacement, submitMove, setSelected, setValidTargets]);

  // Cancel/clear the pending ring placement
  const clearPendingRingPlacement = useCallback(() => {
    setPendingRingPlacement(null);
    setSelected(undefined);
  }, [setSelected]);

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
      // Use refs to get current values (avoid stale closure)
      const choice = pendingChoiceRef.current;
      const respond = onRespondToChoiceRef.current;
      if (!choice || choice.type !== 'region_order' || !respond) {
        setTerritoryRegionPrompt(null);
        return;
      }

      respond(choice, selectedOption);
      setTerritoryRegionPrompt(null);
    },
    [] // No deps needed - uses refs
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
      // Use refs to avoid stale closure issues - the callback may have captured old values
      const currentPendingChoice = pendingChoiceRef.current;
      const currentRespondToChoice = onRespondToChoiceRef.current;
      if (currentPendingChoice && currentRespondToChoice) {
        // Handle region_order choice (territory region selection)
        if (currentPendingChoice.type === 'region_order') {
          const options = (currentPendingChoice.options ?? []) as TerritoryRegionOption[];
          if (options.length === 0) {
            // No options - let other handlers try
          } else {
            // Find which options contain the clicked position using spaces
            const optionsWithSpaces = options.filter(
              (opt): opt is TerritoryRegionOption & { spaces: Position[] } =>
                opt.spaces != null && opt.spaces.length > 0
            );
            if (optionsWithSpaces.length > 0) {
              const matchingBySpaces = optionsWithSpaces.filter((opt) =>
                opt.spaces.some((space) => positionsEqual(space, pos))
              );

              if (matchingBySpaces.length === 1) {
                // Single match - respond directly
                currentRespondToChoice(currentPendingChoice, matchingBySpaces[0]);
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

            // 2nd tier fallback: Look up from territories map (may have stale regionIds)
            // This mirrors the sandbox's more robust territory matching pattern
            const territories = board.territories;
            if (territories && territories.size > 0) {
              // Identify which territory region(s) contain the clicked cell
              const clickedRegionIds: string[] = [];
              territories.forEach((territory, regionId) => {
                const spaces = territory.spaces ?? [];
                if (spaces.some((space) => positionsEqual(space, pos))) {
                  clickedRegionIds.push(regionId);
                }
              });

              // Find all options whose regions contain the clicked cell
              const matchingByRegion = options.filter((opt) =>
                clickedRegionIds.includes(opt.regionId)
              );

              if (matchingByRegion.length === 1) {
                currentRespondToChoice(currentPendingChoice, matchingByRegion[0]);
                return;
              } else if (matchingByRegion.length > 1) {
                // Multiple regions overlap at this cell - show disambiguation dialog
                setTerritoryRegionPrompt({
                  options: matchingByRegion,
                  clickedPosition: pos,
                });
                return;
              }

              // 3rd tier: Representative position heuristic from territories
              let matchedOption: TerritoryRegionOption | undefined;
              territories.forEach((territory, regionId) => {
                if (matchedOption) return;
                const spaces = territory.spaces ?? [];
                const containsClick = spaces.some((space) => positionsEqual(space, pos));
                if (!containsClick) return;

                const hasRepresentative = spaces.some((space) =>
                  options.some((opt) => positionsEqual(opt.representativePosition, space))
                );
                if (hasRepresentative) {
                  matchedOption = options.find((opt) =>
                    spaces.some((space) => positionsEqual(space, opt.representativePosition))
                  );
                } else {
                  matchedOption = options.find((opt) => opt.regionId === regionId);
                }
              });

              if (matchedOption) {
                currentRespondToChoice(currentPendingChoice, matchedOption);
                return;
              }
            }

            // Final fallback: check direct representative position match
            const matchingByRepresentative = options.find((opt) =>
              positionsEqual(opt.representativePosition, pos)
            );
            if (matchingByRepresentative) {
              currentRespondToChoice(currentPendingChoice, matchingByRepresentative);
              return;
            }

            // Click was not on any region - don't block other handlers
          }
        }

        // Handle ring_elimination choice
        if (currentPendingChoice.type === 'ring_elimination') {
          const options = (currentPendingChoice.options ?? []) as Array<{
            stackPosition: Position;
          }>;
          const matching = options.find(
            (opt) => opt.stackPosition && positionsEqual(opt.stackPosition, pos)
          );
          if (matching) {
            currentRespondToChoice(currentPendingChoice, matching);
            return;
          }
          // If valid options exist, block fall-through (must click a valid target)
          if (options.length > 0) {
            return;
          }
        }

        // Handle line_reward_option choice with segments
        if (
          currentPendingChoice.type === 'line_reward_option' &&
          currentPendingChoice.segments &&
          currentPendingChoice.segments.length > 0
        ) {
          const segments = currentPendingChoice.segments;
          const clickedSegment = segments.find((segment) =>
            segment.positions.some((p) => positionsEqual(p, pos))
          );
          if (clickedSegment) {
            currentRespondToChoice(currentPendingChoice, { optionId: clickedSegment.optionId });
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

        // Click-to-accumulate ring placement:
        // - First click on empty cell: enter pending state with count=1
        // - Additional clicks on same cell: increment count (up to max)
        // - Click on valid landing position: submit placement + movement chain
        // - Click elsewhere: show error or start new pending at that position
        // - Press Enter/Escape: confirm or cancel pending (handled in keyboard effect)

        const placeMovesAtPos = validMoves.filter(
          (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
        );

        // Case 1: Clicking on same position as pending placement - increment count
        if (pendingRingPlacement && pendingRingPlacement.positionKey === posKey) {
          const newCount =
            pendingRingPlacement.currentCount >= pendingRingPlacement.maxCount
              ? 1 // Wrap around to 1 at max
              : pendingRingPlacement.currentCount + 1;
          setPendingRingPlacement({
            ...pendingRingPlacement,
            currentCount: newCount,
          });
          // Update valid landing positions based on new stack height
          // Includes both simple movement and capture targets
          const landings = computeValidLandingsWithCaptures(
            board,
            pendingRingPlacement.position,
            gameState.currentPlayer,
            newCount
          );
          setValidTargets(landings);
          return;
        }

        // Case 2: There's a pending placement at different position - check if this click is a valid landing
        if (pendingRingPlacement) {
          // Use any placement move for the position (server moves don't include placementCount,
          // they just indicate valid positions - we supply our own count)
          const pendingPlaceMove = pendingRingPlacement.placeMovesAtPos[0];

          if (pendingPlaceMove) {
            // After placement, we'd need to move from pendingRingPlacement.position
            // Check if clicked position is a valid landing (movement or capture)
            const validLandings = computeValidLandingsWithCaptures(
              board,
              pendingRingPlacement.position,
              gameState.currentPlayer,
              pendingRingPlacement.currentCount
            );
            const isValidLanding = validLandings.some((landing) => positionsEqual(landing, pos));

            if (isValidLanding) {
              // Submit placement, store pending movement for after phase transition
              pendingMovementRef.current = {
                from: pendingRingPlacement.position,
                to: pos,
                timestamp: Date.now(),
              };
              submitMove({
                type: 'place_ring',
                to: pendingPlaceMove.to,
                placementCount: pendingRingPlacement.currentCount,
                placedOnStack: pendingPlaceMove.placedOnStack,
              } as PartialMove);
              setPendingRingPlacement(null);
              setSelected(undefined);
              setValidTargets([]);
              return;
            }
          }

          // Not a valid landing - check if we can start new placement here instead
          if (!hasStack && placeMovesAtPos.length > 0) {
            // Start new pending at this position, abandoning previous
            // Server moves don't include placementCount, compute maxCount from game rules:
            // Empty cell: min(3, ringsInHand)
            const currentPlayer = gameState.players.find(
              (p) => p.playerNumber === gameState.currentPlayer
            );
            const ringsInHand = currentPlayer?.ringsInHand ?? 0;
            const maxCount = Math.min(3, ringsInHand);
            setPendingRingPlacement({
              position: pos,
              positionKey: posKey,
              currentCount: 1,
              maxCount,
              placeMovesAtPos,
            });
            setSelected(pos);
            // Show valid landings for this new position (movement + capture)
            const newLandings = computeValidLandingsWithCaptures(
              board,
              pos,
              gameState.currentPlayer,
              1 // Initial placement of 1 ring
            );
            setValidTargets(newLandings);
            return;
          }

          // Invalid click location - show error and keep pending
          const reason = analyzeInvalid(gameState, pos, {
            isPlayer,
            isMyTurn,
            isConnected: isConnectionActive,
            selectedPosition: pendingRingPlacement.position,
            validMoves: validMoves ?? undefined,
            mustMoveFrom,
          });
          triggerInvalidMove(pos, reason);
          return;
        }

        // Case 3: No pending - start new pending placement on empty cell
        if (!hasStack) {
          if (placeMovesAtPos.length === 0) {
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

          // Server moves don't include placementCount, compute maxCount from game rules:
          // Empty cell: min(3, ringsInHand)
          const currentPlayer = gameState.players.find(
            (p) => p.playerNumber === gameState.currentPlayer
          );
          const ringsInHand = currentPlayer?.ringsInHand ?? 0;
          const maxCount = Math.min(3, ringsInHand);

          // If only 1 ring can be placed, submit immediately (no accumulation possible)
          if (maxCount <= 1) {
            const chosen = placeMovesAtPos[0];
            submitMove({
              type: 'place_ring',
              to: chosen.to,
              placementCount: 1,
              placedOnStack: chosen.placedOnStack,
            } as PartialMove);
            setSelected(undefined);
            setValidTargets([]);
            return;
          }

          // Start pending state for click-to-accumulate
          setPendingRingPlacement({
            position: pos,
            positionKey: posKey,
            currentCount: 1,
            maxCount,
            placeMovesAtPos,
          });
          setSelected(pos);
          // Show valid landings from this position (movement + capture, with 1 ring initially)
          const landings = computeValidLandingsWithCaptures(board, pos, gameState.currentPlayer, 1);
          setValidTargets(landings);
          return;
        }

        // Case 4: Clicking on existing stack - place 1 ring immediately (stacks max at 1 per placement)
        const stackPlaceMoves = validMoves.filter(
          (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
        );
        if (stackPlaceMoves.length > 0) {
          const chosen =
            stackPlaceMoves.find((m) => (m.placementCount ?? 1) === 1) || stackPlaceMoves[0];
          submitMove({
            type: 'place_ring',
            to: chosen.to,
            placementCount: chosen.placementCount,
            placedOnStack: chosen.placedOnStack,
          } as PartialMove);
          setSelected(undefined);
          setValidTargets([]);
          setPendingRingPlacement(null);
          return;
        }

        // No placement moves on this stack - just select it (for skip_placement + move)
        setSelected(pos);
        setValidTargets([]);
        setPendingRingPlacement(null);
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
          // Include captureTarget for capture moves
          const captureTarget = extractCaptureTarget(chainMove);
          submitMove({
            type: chainMove.type,
            from: chainMove.from,
            to: chainMove.to,
            captureTarget,
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

      // Capture phase: direct landing click support (like sandbox)
      // The auto-selection effect already sets selected and validTargets
      if (gameState.currentPhase === 'capture') {
        if (!Array.isArray(validMoves) || validMoves.length === 0) {
          return;
        }

        // Look for overtaking_capture moves to this position (direct landing click)
        const captureMoves = validMoves.filter(
          (m) => m.type === 'overtaking_capture' && m.to && positionsEqual(m.to, pos)
        );

        if (captureMoves.length > 0) {
          const captureMove = captureMoves[0];
          const captureTarget = extractCaptureTarget(captureMove);
          submitMove({
            type: captureMove.type,
            from: captureMove.from,
            to: captureMove.to,
            captureTarget,
          } as PartialMove);
          // Don't clear selection - let the auto-selection effect handle the next state
          return;
        }

        // Clicking elsewhere during capture phase - show feedback
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
          // Include captureTarget for capture moves
          const captureTarget = extractCaptureTarget(matching);
          submitMove({
            type: matching.type,
            from: matching.from,
            to: matching.to,
            captureTarget,
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
      pendingRingPlacement,
      setPendingRingPlacement,
    ]
  );

  // Handle double-click: confirm pending placement with accumulated count, or place 2 rings on empty cells
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

      const posKey = positionToString(pos);

      // If there's a pending placement at this position, confirm it with accumulated count
      if (pendingRingPlacement && pendingRingPlacement.positionKey === posKey) {
        const chosen = pendingRingPlacement.placeMovesAtPos[0];
        if (chosen) {
          submitMove({
            type: 'place_ring',
            to: chosen.to,
            placementCount: pendingRingPlacement.currentCount,
            placedOnStack: chosen.placedOnStack,
          } as PartialMove);
          setPendingRingPlacement(null);
          setSelected(undefined);
          setValidTargets([]);
        }
        return;
      }

      // No pending placement - use default double-click behavior (place 2 rings on empty cell)
      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        return;
      }

      const hasStack = !!board.stacks.get(posKey);

      const placeMovesAtPos = validMoves.filter(
        (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
      );
      if (placeMovesAtPos.length === 0) {
        return;
      }

      // Compute max count based on game rules (same logic as click handler)
      const currentPlayer = gameState.players.find(
        (p) => p.playerNumber === gameState.currentPlayer
      );
      const ringsInHand = currentPlayer?.ringsInHand ?? 0;
      const maxCount = hasStack ? 1 : Math.min(3, ringsInHand);

      // Default double-click places 2 rings (or max available)
      const placementCount = Math.min(2, maxCount);

      const chosen = placeMovesAtPos[0];
      if (!chosen) {
        return;
      }

      submitMove({
        type: 'place_ring',
        to: chosen.to,
        placementCount,
        placedOnStack: hasStack,
      } as PartialMove);

      setPendingRingPlacement(null);
      setSelected(undefined);
      setValidTargets([]);
    },
    [
      gameState,
      validMoves,
      isPlayer,
      isConnectionActive,
      submitMove,
      setSelected,
      setValidTargets,
      pendingRingPlacement,
      setPendingRingPlacement,
    ]
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
    pendingRingPlacement,
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    handleConfirmRingPlacementCount,
    closeRingPlacementPrompt,
    handleConfirmTerritoryRegion,
    closeTerritoryRegionPrompt,
    confirmPendingRingPlacement,
    clearPendingRingPlacement,
  };
}

export default useBackendBoardHandlers;
