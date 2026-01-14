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
  /** Whether to show movement grid overlay */
  showMovementGrid?: boolean;
  /** Whether to show coordinate labels (A-H, 1-8 for square boards) */
  showCoordinateLabels?: boolean;
  /** Whether to render square board ranks from bottom (1 at bottom) */
  squareRankFromBottom?: boolean;
  /** Whether to show line overlay visualization (debug) */
  showLineOverlays?: boolean;
  /** Whether to show territory region overlays (debug) */
  showTerritoryRegionOverlays?: boolean;

  // Game state for info panel
  /** Current phase label */
  phaseLabel: string;
  /** Players list for display */
  players: Array<{
    playerNumber: number;
    username?: string;
    type: 'human' | 'ai';
  }>;
  /** Current player number */
  currentPlayerNumber: number;

  // Handlers
  onCellClick: (pos: Position) => void;
  onCellDoubleClick: (pos: Position) => void;
  onCellContextMenu: (pos: Position) => void;
  onAnimationComplete: () => void;
  onShowBoardControls: () => void;
}

// Player type styling
const PLAYER_TYPE_STYLES = {
  human: 'bg-emerald-900/40 text-emerald-200 border-emerald-700/50',
  ai: 'bg-sky-900/40 text-sky-200 border-sky-700/50',
};

/**
 * BackendBoardSection - Extracted board area component for the backend game host.
 *
 * Contains:
 * - Board header with phase indicator and help button
 * - BoardView rendering with all interaction handlers
 * - Board info panel with player indicators
 *
 * Updated to match sandbox styling for visual consistency.
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
  showMovementGrid = false,
  showCoordinateLabels = false,
  squareRankFromBottom = false,
  showLineOverlays = false,
  showTerritoryRegionOverlays = false,
  phaseLabel,
  players,
  currentPlayerNumber,
  onCellClick,
  onCellDoubleClick,
  onCellContextMenu,
  onAnimationComplete,
  onShowBoardControls,
}) => {
  return (
    <section className="flex-shrink-0">
      {/* Grid layout: board determines column width, panels constrained to match */}
      <div className="grid gap-2" style={{ gridTemplateColumns: 'min-content' }}>
        {/* Board header with phase and help */}
        <div className="flex items-center justify-between p-2 sm:p-2.5 rounded-xl border border-slate-700 bg-slate-900/70">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Phase:</span>
            <span className="px-2 py-1 text-xs font-medium rounded-lg bg-slate-800/80 border border-slate-600 text-slate-200">
              {phaseLabel}
            </span>
          </div>
          <button
            type="button"
            onClick={onShowBoardControls}
            className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80 transition"
            title="Keyboard shortcuts"
          >
            ?
          </button>
        </div>

        {/* Board view */}
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
          showMovementGrid={showMovementGrid}
          showCoordinateLabels={showCoordinateLabels}
          squareRankFromBottom={squareRankFromBottom}
          showLineOverlays={showLineOverlays}
          showTerritoryRegionOverlays={showTerritoryRegionOverlays}
          onShowKeyboardHelp={onShowBoardControls}
        />

        {/* Board info panel with player chips */}
        <div className="p-3 rounded-2xl border border-slate-700 bg-slate-900/70 text-xs">
          <div className="flex flex-wrap gap-1.5">
            {players.map((player) => {
              const isCurrent = player.playerNumber === currentPlayerNumber;
              const typeStyle = PLAYER_TYPE_STYLES[player.type];
              const nameLabel = player.username || `Player ${player.playerNumber}`;
              const typeLabel = player.type === 'ai' ? 'AI' : 'Human';

              return (
                <span
                  key={player.playerNumber}
                  className={`px-3 py-1 rounded-full border transition ${
                    isCurrent ? 'border-white text-white bg-white/15 font-medium' : typeStyle
                  }`}
                >
                  P{player.playerNumber} • {nameLabel} ({typeLabel})
                </span>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default BackendBoardSection;
