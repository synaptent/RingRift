import React from 'react';
import { GameHUD } from '../GameHUD';
import { MobileGameHUD } from '../MobileGameHUD';
import { ChoiceDialog } from '../ChoiceDialog';
import { LineRewardPanel } from '../LineRewardPanel';
import { MoveHistory } from '../MoveHistory';
import { GameEventLog } from '../GameEventLog';
import { GameHistoryPanel } from '../GameHistoryPanel';
import { EvaluationPanel } from '../EvaluationPanel';
import { AIThinkTimeProgress } from '../AIThinkTimeProgress';
import { ResignButton } from '../ResignButton';
import { BackendTouchControlsPanel } from './BackendTouchControlsPanel';
import { VictoryConditionsPanel } from '../GameHUD';
import { Button } from '../ui/Button';
import { formatPosition } from '../../../shared/engine/notation';
import { describeDecisionAutoResolved } from '../../hooks/useBackendDiagnosticsLog';
import type {
  BoardType,
  GameState,
  Move,
  Position,
  TimeControl,
  PlayerChoice,
} from '../../../shared/types/game';
import type { HUDViewModel, EventLogViewModel } from '../../adapters/gameViewModels';
import type { ChoiceViewModel } from '../../adapters/choiceViewModels';
import type {
  DecisionAutoResolvedMeta,
  PositionEvaluationPayload,
} from '../../../shared/types/websocket';

export interface BackendGameSidebarProps {
  /** HUD view model for game state display */
  hudViewModel: HUDViewModel;
  /** Current game state */
  gameState: GameState;
  /** Board type for formatting */
  boardType: BoardType;
  /** Time control settings (if clock enabled) */
  timeControl: TimeControl | undefined;
  /** Whether mobile mode is active */
  isMobile: boolean;
  /** Whether user is a player (not spectator) */
  isPlayer: boolean;
  /** Whether it's the current user's turn */
  isMyTurn: boolean;
  /** Whether connection is active */
  isConnectionActive: boolean;
  /** Rules UX context for teaching/tooltips */
  rulesUxContext: {
    boardType: BoardType;
    numPlayers: number;
    aiDifficulty?: number;
  };

  // Selection state
  /** Currently selected position */
  selectedPosition: Position | undefined;
  /** Selected stack details */
  selectedStackDetails: {
    height: number;
    cap: number;
    controllingPlayer: number;
  } | null;
  /** Board interaction disabled message */
  boardInteractionMessage: string | null;
  /** Valid move targets for touch controls */
  validTargets: Position[];
  /** Whether capture direction selection is pending */
  isCaptureDirectionPending: boolean;
  /** Current phase label for touch controls */
  phaseLabel: string;
  /** Optional phase hint text */
  phaseHint?: string;
  /** Handler to clear current selection */
  onClearSelection: () => void;

  // Decision/Choice state
  /** Pending player choice */
  pendingChoice: PlayerChoice | null;
  /** Choice view model for styling */
  pendingChoiceView: { viewModel: ChoiceViewModel } | null;
  /** Choice deadline timestamp */
  choiceDeadline: number | null;
  /** Reconciled time remaining for choice */
  reconciledDecisionTimeRemainingMs: number | null;
  /** Whether decision countdown is server-capped */
  isDecisionServerCapped: boolean;
  /** Decision auto-resolved metadata */
  decisionAutoResolved: DecisionAutoResolvedMeta | null;

  // Move history state
  /** Move history for display */
  moveHistory: Move[];
  /** Current move index in history */
  currentMoveIndex: number;

  // Event log state
  /** Event log view model */
  eventLogViewModel: EventLogViewModel;
  /** Whether system events are shown in log */
  showSystemEventsInLog: boolean;

  // Resign state
  /** Whether user is currently resigning */
  isResigning: boolean;
  /** Whether resign confirmation is open */
  isResignConfirmOpen: boolean;

  // Advanced panels state
  /** Whether advanced sidebar panels are shown */
  showAdvancedSidebarPanels: boolean;
  /** Game ID for history panel */
  gameId: string;
  /** Victory state for evaluation panel visibility */
  hasVictoryState: boolean;
  /** Evaluation history for evaluation panel */
  evaluationHistory: PositionEvaluationPayload['data'][];

  // Swap sides state (for 2-player games with swap rule)
  /** Whether swap sides prompt should be shown */
  showSwapSidesPrompt: boolean;
  /** Current HUD player for swap rule display */
  hudCurrentPlayer: HUDViewModel['players'][0] | undefined;

  // Chat state
  /** Chat messages */
  chatMessages: Array<{ sender: string; text: string }>;
  /** Chat input value */
  chatInput: string;

  // Handlers - Choice
  onRespondToChoice: <TChoice extends PlayerChoice>(
    choice: TChoice,
    option: TChoice['options'][number]
  ) => void;

  // Handlers - Resign
  onResign: () => void;
  onResignConfirmOpenChange: (open: boolean) => void;

  // Handlers - Swap sides
  onSwapSides: () => void;

  // Handlers - Event log
  onToggleSystemEventsInLog: () => void;

  // Handlers - Advanced panels
  onAdvancedPanelsToggle: (open: boolean) => void;
  onHistoryError: (err: Error) => void;

  // Handlers - Chat
  onChatInputChange: (value: string) => void;
  onChatSubmit: (e: React.FormEvent) => void;

  // Handlers - Controls
  onShowBoardControls: () => void;

  // Skip action state
  /** Whether skip capture is available */
  canSkipCapture?: boolean;
  /** Whether skip territory processing is available */
  canSkipTerritory?: boolean;
  /** Whether skip recovery is available */
  canSkipRecovery?: boolean;

  // Handlers - Skip actions
  onSkipCapture?: () => void;
  onSkipTerritory?: () => void;
  onSkipRecovery?: () => void;

  // AI think time state
  /** Whether an AI is currently thinking */
  isAiThinking?: boolean;
  /** When the AI started thinking (timestamp) */
  aiThinkingStartedAt?: number | null;
  /** AI difficulty level (1-10) */
  aiDifficulty?: number;
  /** AI player name for display */
  aiPlayerName?: string;

  // Board overlay state
  /** Whether movement grid overlay is shown */
  showMovementGrid?: boolean;
  /** Handler to toggle movement grid */
  onToggleMovementGrid?: () => void;
  /** Whether valid targets overlay is shown */
  showValidTargets?: boolean;
  /** Handler to toggle valid targets overlay */
  onToggleValidTargets?: (next: boolean) => void;

  // Board display settings
  /** Whether coordinate labels are shown */
  showCoordinateLabels?: boolean;
  /** Handler to toggle coordinate labels */
  onToggleCoordinateLabels?: () => void;
  /** Whether line overlays are shown (debug) */
  showLineOverlays?: boolean;
  /** Handler to toggle line overlays */
  onToggleLineOverlays?: () => void;
  /** Whether territory region overlays are shown (debug) */
  showTerritoryRegionOverlays?: boolean;
  /** Handler to toggle territory overlays */
  onToggleTerritoryOverlays?: () => void;

  // Decision phase status indicators
  /** Whether ring elimination choice is pending */
  isRingEliminationChoice?: boolean;
}

/**
 * BackendGameSidebar - Extracted sidebar component for the backend game host.
 *
 * Contains:
 * - Game HUD (desktop or mobile variant)
 * - Choice dialog for pending decisions
 * - Selection info panel
 * - Move history
 * - Resign button
 * - Swap sides prompt (when applicable)
 * - Decision auto-resolved indicator
 * - Event log with system event toggle
 * - Advanced diagnostics (game history, evaluation)
 * - Chat panel
 */
export const BackendGameSidebar: React.FC<BackendGameSidebarProps> = ({
  hudViewModel,
  gameState,
  boardType,
  timeControl,
  isMobile,
  isPlayer,
  isMyTurn,
  isConnectionActive,
  rulesUxContext,
  selectedPosition,
  selectedStackDetails,
  boardInteractionMessage,
  validTargets,
  isCaptureDirectionPending,
  phaseLabel,
  phaseHint,
  onClearSelection,
  pendingChoice,
  pendingChoiceView,
  choiceDeadline,
  reconciledDecisionTimeRemainingMs,
  isDecisionServerCapped,
  decisionAutoResolved,
  moveHistory,
  currentMoveIndex,
  eventLogViewModel,
  showSystemEventsInLog,
  isResigning,
  isResignConfirmOpen,
  showAdvancedSidebarPanels,
  gameId,
  hasVictoryState,
  evaluationHistory,
  showSwapSidesPrompt,
  hudCurrentPlayer: _hudCurrentPlayer,
  chatMessages,
  chatInput,
  onRespondToChoice,
  onResign,
  onResignConfirmOpenChange,
  onSwapSides,
  onToggleSystemEventsInLog,
  onAdvancedPanelsToggle,
  onHistoryError,
  onChatInputChange,
  onChatSubmit,
  onShowBoardControls,
  canSkipCapture,
  canSkipTerritory,
  canSkipRecovery,
  onSkipCapture,
  onSkipTerritory,
  onSkipRecovery,
  isAiThinking,
  aiThinkingStartedAt,
  aiDifficulty,
  aiPlayerName,
  showMovementGrid,
  onToggleMovementGrid,
  showValidTargets,
  onToggleValidTargets,
  showCoordinateLabels,
  onToggleCoordinateLabels,
  showLineOverlays,
  onToggleLineOverlays,
  showTerritoryRegionOverlays,
  onToggleTerritoryOverlays,
  isRingEliminationChoice = false,
}) => {
  // Derive region_order choice state from pendingChoice
  const isRegionOrderChoice = pendingChoice?.type === 'region_order';

  return (
    <aside className="w-full max-w-md mx-auto lg:mx-0 lg:w-[256px] flex-shrink-0 space-y-2 text-xs text-slate-100">
      {/* Primary HUD band – placed at the top of the sidebar so phase/turn/time
          are always visible alongside the board. On mobile, render the
          compact MobileGameHUD; on larger screens, use the full GameHUD. */}
      {isMobile ? (
        <MobileGameHUD
          viewModel={hudViewModel}
          timeControl={timeControl}
          onShowBoardControls={onShowBoardControls}
          rulesUxContext={rulesUxContext}
        />
      ) : (
        <GameHUD
          viewModel={hudViewModel}
          timeControl={timeControl}
          onShowBoardControls={onShowBoardControls}
          rulesUxContext={rulesUxContext}
          hideVictoryConditions
        />
      )}

      {/* AI Think Time Progress - show when AI is thinking */}
      {isAiThinking && aiThinkingStartedAt && aiDifficulty !== undefined && (
        <AIThinkTimeProgress
          isAiThinking={isAiThinking}
          thinkingStartedAt={aiThinkingStartedAt}
          aiDifficulty={aiDifficulty}
          aiPlayerName={aiPlayerName}
        />
      )}

      {/* Self-elimination pending alert - matching sandbox pattern */}
      {isRingEliminationChoice && (
        <div className="p-3 border-2 border-amber-400 rounded-2xl bg-amber-900/60 text-xs animate-pulse shadow-lg shadow-amber-500/30">
          <div className="font-semibold text-amber-200 mb-1">⚠️ Self-Elimination Required</div>
          <p className="text-amber-100/90">
            Click one of the highlighted stacks on the board to eliminate.
          </p>
        </div>
      )}

      {/* Territory region order alert - direct board selection */}
      {isRegionOrderChoice && (
        <div className="p-3 border-2 border-emerald-400 rounded-2xl bg-emerald-900/60 text-xs animate-pulse shadow-lg shadow-emerald-500/30">
          <div className="font-semibold text-emerald-200 mb-1">Territory Order Choice</div>
          <p className="text-emerald-100/90">
            Click on one of the highlighted territory regions on the board to process first.
          </p>
        </div>
      )}

      {/* Phase Guide - educational panel showing current phase info */}
      {hudViewModel?.phase && (
        <div className="p-3 rounded-2xl border border-slate-700 bg-slate-900/70 space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wide text-slate-500">Phase</span>
            <span className="font-semibold text-white">{hudViewModel.phase.label}</span>
          </div>
          {hudViewModel.phase.description && (
            <p className="text-slate-300 text-[11px] leading-relaxed">
              {hudViewModel.phase.description}
            </p>
          )}
          <p className="text-[10px] text-slate-500 pt-0.5">
            Complete each phase requirement to advance your turn.
          </p>
        </div>
      )}

      {/* LineRewardPanel for overlength line choices with segments */}
      {isPlayer &&
        pendingChoice &&
        pendingChoice.type === 'line_reward_option' &&
        pendingChoice.segments &&
        pendingChoice.segments.length > 0 && (
          <LineRewardPanel
            choice={pendingChoice}
            onSelect={(optionId) => {
              // When using segments, optionId is a move ID that maps to one of the canonical options
              onRespondToChoice(
                pendingChoice,
                optionId as
                  | 'option_1_collapse_all_and_eliminate'
                  | 'option_2_min_collapse_no_elimination'
              );
            }}
          />
        )}

      {/* Generic ChoiceDialog for other choice types */}
      {isPlayer && (
        <ChoiceDialog
          choice={
            // Hide generic dialog for choices that are handled by direct board clicks
            // (like sandbox does) - ring_elimination, region_order, and line_reward_option with segments
            pendingChoice &&
            (pendingChoice.type === 'ring_elimination' ||
              pendingChoice.type === 'region_order' ||
              (pendingChoice.type === 'line_reward_option' &&
                pendingChoice.segments &&
                pendingChoice.segments.length > 0))
              ? null
              : pendingChoice
          }
          choiceViewModel={pendingChoiceView?.viewModel}
          deadline={choiceDeadline}
          timeRemainingMs={reconciledDecisionTimeRemainingMs}
          isServerCapped={isDecisionServerCapped}
          onSelectOption={(choice, option) => onRespondToChoice(choice, option)}
        />
      )}

      <div className="p-3 border border-slate-700 rounded-xl bg-slate-900/70">
        <h2 className="font-semibold mb-1.5 text-sm">Selection</h2>
        {selectedPosition ? (
          <div className="space-y-1.5">
            <div className="text-base font-mono font-semibold text-white">
              {formatPosition(selectedPosition, { boardType })}
            </div>
            {selectedStackDetails ? (
              <ul className="text-xs text-slate-300 space-y-1">
                <li>Stack height: {selectedStackDetails.height}</li>
                <li>Cap height: {selectedStackDetails.cap}</li>
                <li>Controlled by: P{selectedStackDetails.controllingPlayer}</li>
              </ul>
            ) : (
              <p className="text-xs text-slate-300">Empty cell – choose a placement target.</p>
            )}
            <p className="text-xs text-slate-400">
              Click a highlighted destination to commit the move, or select a new source.
            </p>
          </div>
        ) : (
          <div className="text-slate-200">Click a cell to inspect it.</div>
        )}
        {boardInteractionMessage && (
          <div className="mt-3 text-xs text-amber-300">{boardInteractionMessage}</div>
        )}
      </div>

      {/* Move History - compact notation display */}
      <MoveHistory
        moves={moveHistory}
        boardType={boardType}
        currentMoveIndex={currentMoveIndex}
        notationOptions={{
          squareRankFromBottom: boardType === 'square8' || boardType === 'square19',
        }}
      />

      {/* Skip action buttons - visible when skip is available but not the only option */}
      {isPlayer && isMyTurn && (canSkipCapture || canSkipTerritory || canSkipRecovery) && (
        <div className="flex flex-wrap gap-2 mt-2">
          {canSkipCapture && onSkipCapture && (
            <button
              type="button"
              onClick={onSkipCapture}
              disabled={!isConnectionActive}
              className="px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-900/30 text-sky-200 text-xs font-medium hover:bg-sky-900/50 hover:border-sky-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Skip Capture
            </button>
          )}
          {canSkipTerritory && onSkipTerritory && (
            <button
              type="button"
              onClick={onSkipTerritory}
              disabled={!isConnectionActive}
              className="px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-900/30 text-sky-200 text-xs font-medium hover:bg-sky-900/50 hover:border-sky-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Skip Territory
            </button>
          )}
          {canSkipRecovery && onSkipRecovery && (
            <button
              type="button"
              onClick={onSkipRecovery}
              disabled={!isConnectionActive}
              className="px-3 py-1.5 rounded-lg border border-sky-500/50 bg-sky-900/30 text-sky-200 text-xs font-medium hover:bg-sky-900/50 hover:border-sky-400 disabled:opacity-50 disabled:cursor-not-allowed transition"
            >
              Skip Recovery
            </button>
          )}
        </div>
      )}

      {/* Resign button - visible to players during active games */}
      {isPlayer && gameState.gameStatus === 'active' && (
        <div className="mt-2 flex justify-end">
          <ResignButton
            onResign={onResign}
            disabled={!isConnectionActive}
            isResigning={isResigning}
            isConfirmOpen={isResignConfirmOpen}
            onConfirmOpenChange={onResignConfirmOpenChange}
          />
        </div>
      )}

      {/* Swap sides prompt for 2-player games with swap rule */}
      {showSwapSidesPrompt && (
        <div className="mt-2 p-3 border border-amber-500/60 rounded-xl bg-amber-900/40 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="font-semibold text-amber-100">
              Pie rule available: swap colours with Player 1.
            </span>
            <button
              type="button"
              className="px-2 py-1 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold"
              onClick={onSwapSides}
            >
              Swap colours
            </button>
          </div>
          <p className="mt-1 text-amber-100/80">
            As Player 2, you may use this once, immediately after Player 1's first turn.
          </p>
        </div>
      )}

      {decisionAutoResolved && (
        <div className="mt-1 px-2 py-1 text-[11px] rounded-lg bg-emerald-900/30 border border-emerald-500/40 text-emerald-300">
          {describeDecisionAutoResolved(decisionAutoResolved)}
        </div>
      )}

      {/* Board display settings */}
      {(onToggleMovementGrid ||
        onToggleCoordinateLabels ||
        onToggleLineOverlays ||
        onToggleTerritoryOverlays) && (
        <div className="space-y-1 mt-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider font-medium">
            Board Display
          </div>
          {onToggleMovementGrid && (
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Movement grid</span>
              <button
                type="button"
                onClick={onToggleMovementGrid}
                className={`px-2 py-0.5 rounded border text-xs transition ${
                  showMovementGrid
                    ? 'border-emerald-500 bg-emerald-900/40 text-emerald-200'
                    : 'border-slate-600 bg-slate-900/70 hover:border-emerald-400 hover:text-emerald-200'
                }`}
              >
                {showMovementGrid ? 'On' : 'Off'}
              </button>
            </div>
          )}
          {onToggleCoordinateLabels && (
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Coordinates</span>
              <button
                type="button"
                onClick={onToggleCoordinateLabels}
                className={`px-2 py-0.5 rounded border text-xs transition ${
                  showCoordinateLabels
                    ? 'border-emerald-500 bg-emerald-900/40 text-emerald-200'
                    : 'border-slate-600 bg-slate-900/70 hover:border-emerald-400 hover:text-emerald-200'
                }`}
              >
                {showCoordinateLabels ? 'On' : 'Off'}
              </button>
            </div>
          )}
          {onToggleLineOverlays && (
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Line overlays</span>
              <button
                type="button"
                onClick={onToggleLineOverlays}
                className={`px-2 py-0.5 rounded border text-xs transition ${
                  showLineOverlays
                    ? 'border-sky-500 bg-sky-900/40 text-sky-200'
                    : 'border-slate-600 bg-slate-900/70 hover:border-sky-400 hover:text-sky-200'
                }`}
              >
                {showLineOverlays ? 'On' : 'Off'}
              </button>
            </div>
          )}
          {onToggleTerritoryOverlays && (
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span>Territory overlays</span>
              <button
                type="button"
                onClick={onToggleTerritoryOverlays}
                className={`px-2 py-0.5 rounded border text-xs transition ${
                  showTerritoryRegionOverlays
                    ? 'border-amber-500 bg-amber-900/40 text-amber-200'
                    : 'border-slate-600 bg-slate-900/70 hover:border-amber-400 hover:text-amber-200'
                }`}
              >
                {showTerritoryRegionOverlays ? 'On' : 'Off'}
              </button>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1">
        <span>Log view</span>
        <button
          type="button"
          onClick={onToggleSystemEventsInLog}
          className="px-2 py-0.5 rounded border border-slate-600 bg-slate-900/70 text-xs hover:border-emerald-400 hover:text-emerald-200 transition"
        >
          {showSystemEventsInLog ? 'Moves + system' : 'Moves only'}
        </button>
      </div>

      <GameEventLog viewModel={eventLogViewModel} />

      <details
        className="p-3 border border-slate-700 rounded-xl bg-slate-900/70"
        open={showAdvancedSidebarPanels}
        onToggle={(event) => {
          onAdvancedPanelsToggle(event.currentTarget.open);
        }}
        data-testid="backend-advanced-sidebar-panels"
      >
        <summary className="cursor-pointer select-none text-xs font-semibold text-slate-200">
          Advanced diagnostics
          <span className="ml-1.5 text-[10px] font-normal text-slate-400">
            (history, evaluation)
          </span>
        </summary>
        {showAdvancedSidebarPanels && (
          <div className="mt-2 space-y-2">
            {/* Full move history panel with expandable details */}
            <GameHistoryPanel gameId={gameId} defaultCollapsed={true} onError={onHistoryError} />

            {/* AI analysis/evaluation panel – enabled for spectators and finished games.
                When no evaluation data has been streamed yet, the panel renders a
                placeholder message instead of remaining hidden. */}
            {(!isPlayer || hasVictoryState) && gameState && (
              <EvaluationPanel
                evaluationHistory={evaluationHistory}
                players={gameState.players}
                className="mt-2"
              />
            )}
          </div>
        )}
      </details>

      {/* Touch Controls Panel - shown on mobile or when advanced panels are open */}
      {(isMobile || showAdvancedSidebarPanels) && (
        <BackendTouchControlsPanel
          selectedPosition={selectedPosition}
          selectedStackDetails={selectedStackDetails}
          validTargets={validTargets}
          isCaptureDirectionPending={isCaptureDirectionPending}
          phaseLabel={phaseLabel}
          phaseHint={phaseHint}
          isSpectator={!isPlayer}
          isMyTurn={isMyTurn}
          canSkipCapture={canSkipCapture}
          canSkipTerritoryProcessing={canSkipTerritory}
          canSkipRecovery={canSkipRecovery}
          onSkipCapture={onSkipCapture}
          onSkipTerritoryProcessing={onSkipTerritory}
          onSkipRecovery={onSkipRecovery}
          onClearSelection={onClearSelection}
          showMovementGrid={showMovementGrid ?? false}
          onToggleMovementGrid={(next) => {
            if (onToggleMovementGrid) onToggleMovementGrid();
          }}
          showValidTargets={showValidTargets ?? true}
          onToggleValidTargets={(next) => {
            if (onToggleValidTargets) onToggleValidTargets(next);
          }}
          showLineOverlays={showLineOverlays}
          onToggleLineOverlays={onToggleLineOverlays ? (next) => onToggleLineOverlays() : undefined}
          showTerritoryOverlays={showTerritoryRegionOverlays}
          onToggleTerritoryOverlays={
            onToggleTerritoryOverlays ? (next) => onToggleTerritoryOverlays() : undefined
          }
        />
      )}

      <div className="p-3 border border-slate-700 rounded-xl bg-slate-900/70 flex flex-col h-44">
        <h2 className="font-semibold mb-1 text-xs">Chat</h2>
        <div className="flex-1 overflow-y-auto mb-1 space-y-0.5">
          {chatMessages.length === 0 ? (
            <div className="text-slate-400 text-xs italic">No messages yet.</div>
          ) : (
            chatMessages.map((msg, idx) => (
              <div key={idx} className="text-xs">
                <span className="font-bold text-slate-300">{msg.sender}: </span>
                <span className="text-slate-200">{msg.text}</span>
              </div>
            ))
          )}
        </div>
        <form onSubmit={onChatSubmit} className="flex gap-1.5">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => onChatInputChange(e.target.value)}
            placeholder="Type a message..."
            aria-label="Chat message"
            className="flex-1 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-emerald-500"
          />
          <Button type="submit" size="sm">
            Send
          </Button>
        </form>
      </div>

      {/* Victory Conditions - below Chat for better layout */}
      <VictoryConditionsPanel />
    </aside>
  );
};

export default BackendGameSidebar;
