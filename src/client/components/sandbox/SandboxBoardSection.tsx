import React from 'react';
import { BoardView, type MoveAnimationData } from '../BoardView';
import { VictoryConditionsPanel } from '../GameHUD';
import type { BoardState, Position, GameState } from '../../../shared/types/game';
import type { GameEndExplanation } from '../../../shared/engine/gameEndExplanation';
import type { BoardViewModel } from '../../adapters/gameViewModels';
import type { LocalConfig, LocalPlayerType } from '../../contexts/SandboxContext';
import type { LoadedScenario } from '../../hooks/useSandboxScenarios';

// Constant for player type display metadata
const PLAYER_TYPE_META: Record<LocalPlayerType, { label: string; chip: string }> = {
  human: {
    label: 'Human',
    chip: 'bg-emerald-900/40 text-emerald-200',
  },
  ai: {
    label: 'Computer',
    chip: 'bg-sky-900/40 text-sky-200',
  },
};

export interface SandboxBoardSectionProps {
  /** Board state for rendering */
  boardState: BoardState;
  /** Transformed view model for the board */
  boardViewModel: BoardViewModel;
  /** Game state (for player list, current phase, etc.) */
  gameState: GameState;
  /** Selected cell position */
  selectedPosition: Position | undefined;
  /** Valid move target positions */
  validTargets: Position[];
  /** Whether sandbox is in replay mode (board is read-only) */
  isInReplayMode: boolean;
  /** Pending move animation */
  pendingAnimation: MoveAnimationData | null;
  /** Replay mode animation (if applicable) */
  replayAnimation: MoveAnimationData | null;
  /** Chain capture path for visualization */
  chainCapturePath: Position[] | undefined;
  /** Position key of cell currently shaking (invalid move feedback) */
  shakingCellKey: string | null;
  /** Overlay visibility settings */
  overlays: {
    showMovementGrid: boolean;
    showLineOverlays: boolean;
    showTerritoryOverlays: boolean;
  };
  /** Board display label (e.g., "8×8 Square") */
  boardDisplayLabel: string;
  /** Board display subtitle */
  boardDisplaySubtitle: string;
  /** Configuration object */
  config: LocalConfig;
  /** Players list with player data */
  playersList: Array<{
    playerNumber: number;
    username?: string;
    type: string;
    ringsInHand: number;
    eliminatedRings: number;
    territorySpaces: number;
  }>;
  /** Current player number */
  currentPlayerNumber: number;
  /** Phase details for the current phase */
  phaseDetails: { label: string; summary: string };
  /** Human seat count */
  humanSeatCount: number;
  /** AI seat count */
  aiSeatCount: number;
  /** Whether beginner mode is active */
  isBeginnerMode: boolean;
  /** Whether developer tools are enabled */
  developerToolsEnabled: boolean;
  /** Last loaded scenario (if any) */
  lastLoadedScenario: LoadedScenario | null;
  /** Game end explanation (for debug display) */
  gameEndExplanation: GameEndExplanation | null;

  // Status chip flags
  /** Whether ring elimination choice is active */
  isRingEliminationChoice: boolean;
  /** Whether region order choice is active */
  isRegionOrderChoice: boolean;
  /** Whether chain capture continuation is active */
  isChainCaptureContinuationStep: boolean;

  // Handlers
  onCellClick: (pos: Position) => void;
  onCellDoubleClick: (pos: Position) => void;
  onCellContextMenu: (pos: Position) => void;
  onAnimationComplete: () => void;
  onReplayAnimationComplete: () => void;
  onShowBoardControls: () => void;
  onSaveState: () => void;
  onExportScenario: () => void;
  onCopyFixture: () => void;
  onLoadScenario: () => void;
  onResetScenario: () => void;
  onChangeSetup: () => void;
  onModeChange: (mode: 'beginner' | 'debug') => void;
}

/**
 * SandboxBoardSection - Extracted board area component for the sandbox host.
 *
 * Contains:
 * - Board header with mode toggle and action buttons
 * - Replay mode indicator
 * - BoardView rendering
 * - Board info panel with phase and player indicators
 * - Victory conditions panel
 */
export const SandboxBoardSection: React.FC<SandboxBoardSectionProps> = ({
  boardState,
  boardViewModel,
  gameState: _gameState,
  selectedPosition,
  validTargets,
  isInReplayMode,
  pendingAnimation,
  replayAnimation,
  chainCapturePath,
  shakingCellKey,
  overlays,
  boardDisplayLabel,
  boardDisplaySubtitle,
  config,
  playersList,
  currentPlayerNumber,
  phaseDetails,
  humanSeatCount,
  aiSeatCount,
  isBeginnerMode,
  developerToolsEnabled,
  lastLoadedScenario,
  gameEndExplanation,
  isRingEliminationChoice,
  isRegionOrderChoice,
  isChainCaptureContinuationStep,
  onCellClick,
  onCellDoubleClick,
  onCellContextMenu,
  onAnimationComplete,
  onReplayAnimationComplete,
  onShowBoardControls,
  onSaveState,
  onExportScenario,
  onCopyFixture,
  onLoadScenario,
  onResetScenario,
  onChangeSetup,
  onModeChange,
}) => {
  return (
    <section className="flex-shrink-0">
      {/* Grid layout: board determines column width, panels constrained to match */}
      <div className="grid gap-4" style={{ gridTemplateColumns: 'min-content' }}>
        {/* Board header with mode toggle and action buttons */}
        <div className="p-3 sm:p-4 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg overflow-x-auto">
          <div className="flex flex-wrap items-center justify-between gap-2 sm:gap-3">
            <div>
              <p className="text-[10px] sm:text-xs uppercase tracking-wide text-slate-400">
                Local Sandbox
              </p>
              <h1 className="text-lg sm:text-2xl font-bold text-white">
                Game – {boardDisplayLabel}
              </h1>
            </div>
            <div className="flex items-center gap-3">
              {/* Beginner/Debug Mode Toggle */}
              <div className="inline-flex rounded-lg border border-slate-600 p-0.5 bg-slate-900/60">
                <button
                  type="button"
                  onClick={() => onModeChange('beginner')}
                  className={`px-2 py-1 text-[10px] font-medium rounded-md transition ${
                    isBeginnerMode
                      ? 'bg-emerald-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  aria-pressed={isBeginnerMode}
                  title="Hide debug panels, show teaching features"
                >
                  🎓 Beginner
                </button>
                <button
                  type="button"
                  onClick={() => onModeChange('debug')}
                  className={`px-2 py-1 text-[10px] font-medium rounded-md transition ${
                    !isBeginnerMode
                      ? 'bg-sky-600 text-white'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                  aria-pressed={!isBeginnerMode}
                  title="Show all developer tools and diagnostics"
                >
                  🔧 Debug
                </button>
              </div>
              <div className="flex items-center gap-2">
                {/* Save State - always visible for undo/redo */}
                <button
                  type="button"
                  onClick={onSaveState}
                  className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
                >
                  Save State
                </button>
                {/* Debug-only buttons - hidden in beginner mode */}
                {!isBeginnerMode && developerToolsEnabled && (
                  <>
                    <button
                      type="button"
                      onClick={onExportScenario}
                      className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
                    >
                      Export Scenario JSON
                    </button>
                    <button
                      type="button"
                      onClick={onCopyFixture}
                      className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                    >
                      Copy Test Fixture
                    </button>
                    {gameEndExplanation && (
                      <details
                        className="absolute top-full right-0 mt-2 w-96 p-4 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 text-xs font-mono overflow-auto max-h-96"
                        data-testid="sandbox-game-end-explanation-debug"
                      >
                        <summary className="cursor-pointer text-slate-400 hover:text-slate-200 mb-2">
                          Debug: GameEndExplanation
                        </summary>
                        <pre className="whitespace-pre-wrap text-emerald-300">
                          {JSON.stringify(gameEndExplanation, null, 2)}
                        </pre>
                      </details>
                    )}
                  </>
                )}
                {/* Load Scenario - hidden in beginner mode */}
                {!isBeginnerMode && (
                  <button
                    type="button"
                    onClick={onLoadScenario}
                    className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-amber-400 hover:text-amber-200 transition"
                  >
                    Load Scenario
                  </button>
                )}
                {lastLoadedScenario && (
                  <button
                    type="button"
                    onClick={onResetScenario}
                    className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                  >
                    Reset Scenario
                  </button>
                )}
                <button
                  type="button"
                  onClick={onChangeSetup}
                  className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                >
                  Change Setup
                </button>
                <button
                  type="button"
                  aria-label="Show board controls"
                  data-testid="board-controls-button"
                  onClick={onShowBoardControls}
                  className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80"
                >
                  ?
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Replay mode indicator */}
        {isInReplayMode && (
          <div className="mb-2 p-2 rounded-lg bg-emerald-900/40 border border-emerald-700/50 text-xs text-emerald-200 flex items-center justify-between">
            <span>Viewing replay - board is read-only</span>
            <span className="text-emerald-400/70">Use playback controls in sidebar</span>
          </div>
        )}

        {/* Board view */}
        <BoardView
          boardType={boardState.type}
          board={boardState}
          viewModel={boardViewModel}
          selectedPosition={isInReplayMode ? undefined : selectedPosition}
          validTargets={isInReplayMode ? [] : validTargets}
          onCellClick={isInReplayMode ? undefined : onCellClick}
          onCellDoubleClick={isInReplayMode ? undefined : onCellDoubleClick}
          onCellContextMenu={isInReplayMode ? undefined : onCellContextMenu}
          onShowKeyboardHelp={onShowBoardControls}
          showMovementGrid={overlays.showMovementGrid}
          showCoordinateLabels={boardState.type === 'square8' || boardState.type === 'square19'}
          squareRankFromBottom={boardState.type === 'square8' || boardState.type === 'square19'}
          showLineOverlays={overlays.showLineOverlays}
          showTerritoryRegionOverlays={overlays.showTerritoryOverlays}
          pendingAnimation={
            isInReplayMode ? (replayAnimation ?? undefined) : (pendingAnimation ?? undefined)
          }
          onAnimationComplete={isInReplayMode ? onReplayAnimationComplete : onAnimationComplete}
          chainCapturePath={isInReplayMode ? undefined : chainCapturePath}
          shakingCellKey={isInReplayMode ? null : shakingCellKey}
        />

        {/* Board info panel */}
        <section className="p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col gap-2 text-xs text-slate-200 overflow-x-auto">
          <div className="flex flex-wrap items-center gap-2">
            {(() => {
              let primarySubtitleText = boardDisplaySubtitle;
              let primarySubtitleClass =
                'px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600';

              if (isRingEliminationChoice) {
                primarySubtitleText = 'Select stack cap to eliminate';
                primarySubtitleClass =
                  'px-2 py-1 rounded-full bg-amber-500 text-slate-950 font-semibold border border-amber-300 shadow-sm shadow-amber-500/40';
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
            <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
              Players: {config.numPlayers} ({humanSeatCount} human, {aiSeatCount} AI)
            </span>
            <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600 min-w-[10rem] inline-flex justify-center text-center">
              Phase: {phaseDetails.label}
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {playersList.map((player) => {
              const typeKey = player.type === 'ai' ? 'ai' : 'human';
              const meta = PLAYER_TYPE_META[typeKey as LocalPlayerType];
              const isCurrent = player.playerNumber === currentPlayerNumber;
              const nameLabel = player.username || `Player ${player.playerNumber}`;
              return (
                <span
                  key={player.playerNumber}
                  className={`px-3 py-1 rounded-full border transition ${
                    isCurrent ? 'border-white text-white bg-white/15' : meta.chip
                  }`}
                >
                  P{player.playerNumber} • {nameLabel} ({meta.label})
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

export default SandboxBoardSection;
