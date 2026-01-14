import React from 'react';
import { BoardView, type MoveAnimationData } from '../BoardView';
import { VictoryConditionsPanel } from '../GameHUD';
import type { BoardState, BoardType, Position } from '../../../shared/types/game';
import type {
  BoardViewModel,
  BoardDecisionHighlightsViewModel,
} from '../../adapters/gameViewModels';

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
  /** Decision phase highlights (ring elimination, territory regions, capture direction, etc.) */
  decisionHighlights?: BoardDecisionHighlightsViewModel;

  // Decision phase status indicators
  /** Whether ring elimination choice is pending */
  isRingEliminationChoice?: boolean;
  /** Whether region order choice is pending */
  isRegionOrderChoice?: boolean;
  /** Whether chain capture continuation is active */
  isChainCaptureContinuationStep?: boolean;
  /** Board display subtitle (e.g., "Turn 5 • Active") */
  boardDisplaySubtitle?: string;
  /** Board display label (e.g., "8×8 Square") */
  boardDisplayLabel?: string;

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
  decisionHighlights,
  isRingEliminationChoice = false,
  isRegionOrderChoice = false,
  isChainCaptureContinuationStep = false,
  boardDisplaySubtitle,
  boardDisplayLabel,
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
        {/* Board header with title, phase, and help */}
        <div className="p-2 sm:p-2.5 rounded-xl border border-slate-700 bg-slate-900/70 shadow-lg overflow-x-auto">
          <div className="flex flex-wrap items-center justify-between gap-1.5 sm:gap-2">
            <h1 className="text-sm sm:text-base font-bold text-white">
              <span className="text-slate-400 font-medium">Game</span>
              <span className="mx-2 text-slate-600">–</span>
              {boardDisplayLabel}
            </h1>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400">Phase:</span>
              <span className="px-2 py-1 text-xs font-medium rounded-lg bg-slate-800/80 border border-slate-600 text-slate-200">
                {phaseLabel}
              </span>
              <button
                type="button"
                onClick={onShowBoardControls}
                className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80 transition"
                title="Keyboard shortcuts"
              >
                ?
              </button>
            </div>
          </div>
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
          decisionHighlights={decisionHighlights}
          onShowKeyboardHelp={onShowBoardControls}
          scaleAdjustment={boardType === 'square8' ? 0.9 : 1.0}
        />

        {/* Board info panel with status chips and player chips */}
        <section className="p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col gap-2 text-xs text-slate-200 overflow-x-auto">
          {/* Status chips row */}
          <div className="flex flex-wrap items-center gap-2">
            {(() => {
              let primarySubtitleText = boardDisplaySubtitle || `Phase: ${phaseLabel}`;
              let primarySubtitleClass =
                'px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600';

              if (isRingEliminationChoice) {
                primarySubtitleText =
                  '⚠️ SELF-ELIMINATION REQUIRED – Select stack cap to eliminate';
                primarySubtitleClass =
                  'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-lg shadow-amber-500/50 animate-pulse';
              } else if (isRegionOrderChoice) {
                primarySubtitleText = 'Territory claimed – choose region to process';
                primarySubtitleClass =
                  'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-sm shadow-amber-500/40';
              } else if (isChainCaptureContinuationStep) {
                primarySubtitleText = 'Continue Chain Capture';
                primarySubtitleClass =
                  'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-sm shadow-amber-500/40';
              }

              return <span className={primarySubtitleClass}>{primarySubtitleText}</span>;
            })()}
            {(() => {
              const humanCount = players.filter((p) => p.type === 'human').length;
              const aiCount = players.filter((p) => p.type === 'ai').length;
              return (
                <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
                  Players: {players.length} ({humanCount} human, {aiCount} AI)
                </span>
              );
            })()}
            <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600 min-w-[10rem] inline-flex justify-center text-center">
              Phase: {phaseLabel}
            </span>
          </div>
          {/* Player chips row */}
          <div className="flex flex-wrap gap-2">
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
        </section>

        {/* Victory Conditions - placed below game info panel */}
        <VictoryConditionsPanel className="" />
      </div>
    </section>
  );
};

export default BackendBoardSection;
