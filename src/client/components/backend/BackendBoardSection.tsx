import React from 'react';
import { BoardView, type MoveAnimationData } from '../BoardView';
import type { BoardState, BoardType, Position } from '../../../shared/types/game';
import type { BoardViewModel } from '../../adapters/gameViewModels';

export interface BackendBoardSectionProps {
  /** Board type for rendering */
  boardType: BoardType;
  /** Board state for rendering */
  board: BoardState;
  /** Transformed view model for the board */
  viewModel: BoardViewModel;
  /** Selected cell position */
  selectedPosition: Position | undefined;
  /** Valid move target positions */
  validTargets: Position[];
  /** Whether user is a spectator (not a player) */
  isSpectator: boolean;
  /** Pending move animation */
  pendingAnimation: MoveAnimationData | undefined;
  /** Chain capture path for visualization */
  chainCapturePath: Position[] | undefined;
  /** Position key of cell currently shaking (invalid move feedback) */
  shakingCellKey: string | null;

  // Handlers
  onCellClick: (pos: Position) => void;
  onCellDoubleClick: (pos: Position) => void;
  onCellContextMenu: (pos: Position) => void;
  onAnimationComplete: () => void;
  onShowBoardControls: () => void;
}

/**
 * BackendBoardSection - Extracted board area component for the backend game host.
 *
 * Contains:
 * - BoardView rendering with all interaction handlers
 * - Centralized board container styling
 *
 * This component is intentionally simpler than SandboxBoardSection as the backend
 * game host has fewer board-level controls (no replay mode, debug tools, etc.).
 */
export const BackendBoardSection: React.FC<BackendBoardSectionProps> = ({
  boardType,
  board,
  viewModel,
  selectedPosition,
  validTargets,
  isSpectator,
  pendingAnimation,
  chainCapturePath,
  shakingCellKey,
  onCellClick,
  onCellDoubleClick,
  onCellContextMenu,
  onAnimationComplete,
  onShowBoardControls,
}) => {
  return (
    <section className="flex-shrink-0 flex justify-center lg:justify-start">
      <BoardView
        boardType={boardType}
        board={board}
        viewModel={viewModel}
        selectedPosition={selectedPosition}
        validTargets={validTargets}
        onCellClick={onCellClick}
        onCellDoubleClick={onCellDoubleClick}
        onCellContextMenu={onCellContextMenu}
        isSpectator={isSpectator}
        pendingAnimation={pendingAnimation}
        onAnimationComplete={onAnimationComplete}
        chainCapturePath={chainCapturePath}
        shakingCellKey={shakingCellKey}
        onShowKeyboardHelp={onShowBoardControls}
      />
    </section>
  );
};

export default BackendBoardSection;
