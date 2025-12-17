/**
 * useBoardViewProps - Consolidates BoardView props derivation
 *
 * This hook extracts and consolidates all the props needed for BoardView
 * from game state and interaction state. It handles:
 * - View model derivation from game state
 * - Selected position and valid targets
 * - Animation state management
 * - Overlay toggle state
 * - Chain capture path extraction
 * - Shaking cell feedback
 *
 * @module hooks/useBoardViewProps
 */

import { useMemo, useState, useCallback } from 'react';
import type { BoardState, GameState, Position, BoardType } from '../../shared/types/game';
import type { BoardViewModel } from '../adapters/gameViewModels';
import { toBoardViewModel, deriveBoardDecisionHighlights } from '../adapters/gameViewModels';
import { useAccessibility } from '../contexts/AccessibilityContext';
import type { MoveAnimationData as MoveAnimation } from '../components/BoardView';
import type { PlayerChoice } from '../../shared/types/game';

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Overlay visibility configuration.
 */
export interface BoardOverlayConfig {
  /** Show movement grid overlay */
  showMovementGrid: boolean;
  /** Show valid target highlighting */
  showValidTargets: boolean;
  /** Show coordinate labels (for square boards) */
  showCoordinateLabels: boolean;
  /** Use chess-style rank numbering (1 at bottom) */
  squareRankFromBottom: boolean;
  /** Show line detection overlays (dev tools) */
  showLineOverlays: boolean;
  /** Show territory region overlays (dev tools) */
  showTerritoryOverlays: boolean;
}

/**
 * Options for the BoardViewProps hook.
 */
export interface UseBoardViewPropsOptions {
  /** Current game state */
  gameState: GameState | null;
  /** Currently selected position */
  selectedPosition: Position | undefined;
  /** Valid move targets for selected position */
  validTargets: Position[];
  /** Pending animation data */
  pendingAnimation: MoveAnimation | null;
  /** Callback when animation completes */
  onAnimationComplete: () => void;
  /** Pending player choice (for decision highlights) */
  pendingChoice: PlayerChoice | null;
  /** Position key of cell to shake */
  shakingCellKey: string | null;
  /** Overlay configuration */
  overlays?: Partial<BoardOverlayConfig>;
  /** Whether viewer is spectator */
  isSpectator?: boolean;
  /** Must-move-from position (from valid moves) */
  mustMoveFrom?: Position;
}

/**
 * Props ready to spread onto BoardView.
 */
export interface BoardViewPropsResult {
  /** Board type */
  boardType: BoardType;
  /** Board state */
  board: BoardState;
  /** Precomputed view model */
  viewModel: BoardViewModel | undefined;
  /** Selected position (may include mustMoveFrom) */
  selectedPosition: Position | undefined;
  /** Valid targets */
  validTargets: Position[];
  /** Spectator flag */
  isSpectator: boolean;
  /** Movement grid */
  showMovementGrid: boolean;
  /** Coordinate labels */
  showCoordinateLabels: boolean;
  /** Rank from bottom */
  squareRankFromBottom: boolean;
  /** Line overlays */
  showLineOverlays: boolean;
  /** Territory overlays */
  showTerritoryRegionOverlays: boolean;
  /** Pending animation */
  pendingAnimation: MoveAnimation | undefined;
  /** Animation complete handler */
  onAnimationComplete: () => void;
  /** Chain capture path */
  chainCapturePath: Position[] | undefined;
  /** Shaking cell */
  shakingCellKey: string | null;
}

// ═══════════════════════════════════════════════════════════════════════════
// DEFAULT OVERLAY CONFIG
// ═══════════════════════════════════════════════════════════════════════════

const DEFAULT_OVERLAYS: BoardOverlayConfig = {
  showMovementGrid: false,
  showValidTargets: true,
  showCoordinateLabels: false,
  squareRankFromBottom: false,
  showLineOverlays: false,
  showTerritoryOverlays: false,
};

// ═══════════════════════════════════════════════════════════════════════════
// MAIN HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook that consolidates BoardView props derivation.
 */
export function useBoardViewProps(options: UseBoardViewPropsOptions): BoardViewPropsResult | null {
  const { colorVisionMode } = useAccessibility();
  const {
    gameState,
    selectedPosition,
    validTargets,
    pendingAnimation,
    onAnimationComplete,
    pendingChoice,
    shakingCellKey,
    overlays = {},
    isSpectator = false,
    mustMoveFrom,
  } = options;

  // Merge overlay config with defaults
  const overlayConfig = useMemo(() => ({ ...DEFAULT_OVERLAYS, ...overlays }), [overlays]);

  // Derive decision highlights from pending choice
  const decisionHighlights = useMemo(() => {
    if (!gameState || !pendingChoice) {
      return undefined;
    }
    return deriveBoardDecisionHighlights(gameState, pendingChoice);
  }, [gameState, pendingChoice]);

  // Compute effective selected position (includes mustMoveFrom)
  const effectiveSelectedPosition = selectedPosition ?? mustMoveFrom;

  // Derive board view model
  const viewModel = useMemo<BoardViewModel | undefined>(() => {
    if (!gameState?.board) {
      return undefined;
    }

    return toBoardViewModel(gameState.board, {
      selectedPosition: effectiveSelectedPosition,
      validTargets,
      decisionHighlights,
      colorVisionMode,
    });
  }, [
    gameState?.board,
    effectiveSelectedPosition,
    validTargets,
    decisionHighlights,
    colorVisionMode,
  ]);

  // Extract chain capture path
  const chainCapturePath = useMemo(() => {
    return extractChainCapturePath(gameState);
  }, [gameState]);

  // Determine coordinate label visibility (only for square boards)
  const showCoordinateLabels = useMemo(() => {
    if (!overlayConfig.showCoordinateLabels) {
      return false;
    }
    return gameState?.boardType === 'square8' || gameState?.boardType === 'square19';
  }, [overlayConfig.showCoordinateLabels, gameState?.boardType]);

  // Determine rank from bottom (only for square boards)
  const squareRankFromBottom = useMemo(() => {
    if (!overlayConfig.squareRankFromBottom) {
      return false;
    }
    return gameState?.boardType === 'square8' || gameState?.boardType === 'square19';
  }, [overlayConfig.squareRankFromBottom, gameState?.boardType]);

  // Return null if no game state
  if (!gameState?.board) {
    return null;
  }

  return {
    boardType: gameState.boardType,
    board: gameState.board,
    viewModel,
    selectedPosition: effectiveSelectedPosition,
    validTargets,
    isSpectator,
    showMovementGrid: overlayConfig.showMovementGrid,
    showCoordinateLabels,
    squareRankFromBottom,
    showLineOverlays: overlayConfig.showLineOverlays,
    showTerritoryRegionOverlays: overlayConfig.showTerritoryOverlays,
    pendingAnimation: pendingAnimation ?? undefined,
    onAnimationComplete,
    chainCapturePath,
    shakingCellKey,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Extract chain capture path from game state.
 *
 * When in chain_capture phase, walks backwards through move history
 * to build the visualization path.
 */
function extractChainCapturePath(gameState: GameState | null): Position[] | undefined {
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

// ═══════════════════════════════════════════════════════════════════════════
// OVERLAY STATE HOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing overlay toggle state.
 *
 * This is a separate hook for use when the parent component needs
 * direct access to toggle the overlays.
 */
export function useBoardOverlays(defaults?: Partial<BoardOverlayConfig>): {
  overlays: BoardOverlayConfig;
  setShowMovementGrid: (show: boolean) => void;
  setShowValidTargets: (show: boolean) => void;
  setShowCoordinateLabels: (show: boolean) => void;
  setSquareRankFromBottom: (show: boolean) => void;
  setShowLineOverlays: (show: boolean) => void;
  setShowTerritoryOverlays: (show: boolean) => void;
  resetOverlays: () => void;
} {
  const [overlays, setOverlays] = useState<BoardOverlayConfig>({
    ...DEFAULT_OVERLAYS,
    ...defaults,
  });

  const setShowMovementGrid = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, showMovementGrid: show }));
  }, []);

  const setShowValidTargets = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, showValidTargets: show }));
  }, []);

  const setShowCoordinateLabels = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, showCoordinateLabels: show }));
  }, []);

  const setSquareRankFromBottom = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, squareRankFromBottom: show }));
  }, []);

  const setShowLineOverlays = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, showLineOverlays: show }));
  }, []);

  const setShowTerritoryOverlays = useCallback((show: boolean) => {
    setOverlays((prev: BoardOverlayConfig) => ({ ...prev, showTerritoryOverlays: show }));
  }, []);

  const resetOverlays = useCallback(() => {
    setOverlays({ ...DEFAULT_OVERLAYS, ...defaults });
  }, [defaults]);

  return {
    overlays,
    setShowMovementGrid,
    setShowValidTargets,
    setShowCoordinateLabels,
    setSquareRankFromBottom,
    setShowLineOverlays,
    setShowTerritoryOverlays,
    resetOverlays,
  };
}
