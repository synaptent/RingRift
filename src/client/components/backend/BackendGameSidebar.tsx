import React from 'react';
import { GameHUD } from '../GameHUD';
import { MobileGameHUD } from '../MobileGameHUD';
import { ChoiceDialog } from '../ChoiceDialog';
import { MoveHistory } from '../MoveHistory';
import { GameEventLog } from '../GameEventLog';
import { GameHistoryPanel } from '../GameHistoryPanel';
import { EvaluationPanel } from '../EvaluationPanel';
import { ResignButton } from '../ResignButton';
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
  hudCurrentPlayer,
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
}) => {
  return (
    <aside className="w-full lg:w-80 flex-shrink-0 space-y-3 text-sm text-slate-100">
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
        />
      )}

      {isPlayer && (
        <ChoiceDialog
          choice={pendingChoice}
          choiceViewModel={pendingChoiceView?.viewModel}
          deadline={choiceDeadline}
          timeRemainingMs={reconciledDecisionTimeRemainingMs}
          isServerCapped={isDecisionServerCapped}
          onSelectOption={(choice, option) => onRespondToChoice(choice, option)}
        />
      )}

      <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
        <h2 className="font-semibold mb-2">Selection</h2>
        {selectedPosition ? (
          <div className="space-y-2">
            <div className="text-lg font-mono font-semibold text-white">
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
      <MoveHistory moves={moveHistory} boardType={boardType} currentMoveIndex={currentMoveIndex} />

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
        <div className="mt-2 p-2 border border-amber-500/60 rounded bg-amber-900/40 text-xs">
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
        <div className="mt-1 text-[11px] text-amber-300">
          {describeDecisionAutoResolved(decisionAutoResolved)}
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
        className="p-3 border border-slate-700 rounded bg-slate-900/50"
        open={showAdvancedSidebarPanels}
        onToggle={(event) => {
          onAdvancedPanelsToggle(event.currentTarget.open);
        }}
        data-testid="backend-advanced-sidebar-panels"
      >
        <summary className="cursor-pointer select-none text-sm font-semibold text-slate-200">
          Advanced diagnostics
          <span className="ml-2 text-[11px] font-normal text-slate-400">(history, evaluation)</span>
        </summary>
        {showAdvancedSidebarPanels && (
          <div className="mt-3 space-y-3">
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

      <div className="p-3 border border-slate-700 rounded bg-slate-900/50 flex flex-col h-64">
        <h2 className="font-semibold mb-2">Chat</h2>
        <div className="flex-1 overflow-y-auto mb-2 space-y-1">
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
        <form onSubmit={onChatSubmit} className="flex gap-2">
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
    </aside>
  );
};

export default BackendGameSidebar;
