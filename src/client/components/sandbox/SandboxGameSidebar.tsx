import React from 'react';
import { GameHUD } from '../GameHUD';
import { MobileGameHUD } from '../MobileGameHUD';
import { AIThinkTimeProgress } from '../AIThinkTimeProgress';
import { ReplayPanel } from '../ReplayPanel';
import { HistoryPlaybackPanel } from '../HistoryPlaybackPanel';
import { MoveHistory } from '../MoveHistory';
import { GameEventLog } from '../GameEventLog';
import { SandboxDevTools } from '../SandboxDevTools';
import { SandboxTouchControlsPanel } from '../SandboxTouchControlsPanel';
import { GameSyncService } from '../../services/GameSyncService';
import type { MoveNotationOptions } from '../../../shared/engine/notation';
import type { BoardType, GameState, Position, TimeControl } from '../../../shared/types/game';
import type { HUDViewModel, EventLogViewModel } from '../../adapters/gameViewModels';
import type { LocalConfig } from '../../contexts/SandboxContext';
import type { LoadedScenario } from '../../hooks/useSandboxScenarios';
import type { ClientSandboxEngine } from '../../sandbox/ClientSandboxEngine';
import type { MoveAnimationData } from '../BoardView';
import type { PositionEvaluationPayload } from '../../../shared/types/websocket';

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'saved-local' | 'error';

export interface SyncState {
  status: 'idle' | 'syncing' | 'offline' | 'error';
  lastSyncAt?: Date;
  error?: string;
}

export interface SandboxGameSidebarProps {
  /** HUD view model for game state display */
  hudViewModel: HUDViewModel | null;
  /** Current game state */
  gameState: GameState | null;
  /** Game configuration */
  config: LocalConfig;
  /** Time control settings (if clock enabled) */
  timeControl: TimeControl | undefined;
  /** Board state for move history notation */
  boardState: { type: string; size?: number } | null;
  /** Whether mobile mode is active */
  isMobile: boolean;
  /** Whether user is logged in */
  isLoggedIn: boolean;
  /** Whether beginner mode is active */
  isBeginnerMode: boolean;
  /** Whether developer tools are enabled */
  developerToolsEnabled: boolean;
  /** Last loaded scenario context */
  lastLoadedScenario: LoadedScenario | null;
  /** Board type value for rules context */
  boardTypeValue: BoardType;
  /** Number of players */
  numPlayers: number;

  // Selection state for touch controls
  /** Currently selected position */
  selectedPosition: Position | undefined;
  /** Selected stack details */
  selectedStackDetails: {
    height: number;
    cap: number;
    controllingPlayer: number;
  } | null;
  /** Valid move targets */
  validTargets: Position[];
  /** Primary valid targets (capture targets take precedence) */
  primaryValidTargets: Position[];
  /** Whether capture direction choice is pending */
  isCaptureDirectionPending: boolean;
  /** Capture target positions */
  captureTargets: Position[];

  // Overlay visibility
  overlays: {
    showMovementGrid: boolean;
    showValidTargets: boolean;
    showLineOverlays: boolean;
    showTerritoryOverlays: boolean;
  };

  // Phase info for touch controls
  /** Current phase label */
  phaseLabel: string;
  /** Phase hint text */
  phaseHint: string | undefined;

  // Skip action availability
  canSkipCapture: boolean;
  canSkipTerritoryProcessing: boolean;
  canSkipRecovery: boolean;

  // Replay state
  /** Whether in replay mode */
  isInReplayMode: boolean;
  /** Whether viewing history */
  isViewingHistory: boolean;
  /** Current history view index */
  historyViewIndex: number;
  /** Whether history snapshots are available */
  hasHistorySnapshots: boolean;

  // Sidebar panel state
  /** Whether advanced panels are shown */
  showAdvancedSidebarPanels: boolean;

  // Recording state
  /** Whether auto-save is enabled */
  autoSaveGames: boolean;
  /** Current save status */
  gameSaveStatus: SaveStatus;
  /** Number of pending local games */
  pendingLocalGames: number;
  /** Sync state */
  syncState: SyncState | null;

  // AI tracking state
  /** AI thinking started timestamp */
  aiThinkingStartedAt: number | null;
  /** AI ladder health data */
  aiLadderHealth: Record<string, unknown> | null;
  /** AI ladder health error */
  aiLadderHealthError: string | null;
  /** AI ladder health loading state */
  aiLadderHealthLoading: boolean;

  // Evaluation state
  /** Evaluation history */
  evaluationHistory: PositionEvaluationPayload['data'][];
  /** Evaluation error */
  evaluationError: string | null;
  /** Whether evaluation is in progress */
  isEvaluating: boolean;

  // Event log
  /** Event log view model */
  eventLogViewModel: EventLogViewModel;

  // Sandbox engine reference (for dev tools)
  sandboxEngine: ClientSandboxEngine | null;

  // Swap sides callback
  onSwapSides: () => void;

  // Handlers - Replay/History
  onReplayStateChange: (state: GameState | null) => void;
  onReplayModeChange: (mode: boolean) => void;
  onReplayAnimationChange: (animation: MoveAnimationData | null) => void;
  onForkFromReplay: (state: GameState) => void;
  onHistoryIndexChange: (index: number) => void;
  onExitHistoryView: () => void;
  onEnterHistoryView: () => void;

  // Handlers - Selection/Controls
  onClearSelection: () => void;
  onShowBoardControls: () => void;

  // Handlers - Overlay toggles
  onToggleMovementGrid: (show: boolean) => void;
  onToggleValidTargets: (show: boolean) => void;
  onToggleLineOverlays?: (show: boolean) => void;
  onToggleTerritoryOverlays?: (show: boolean) => void;

  // Handlers - Skip actions
  onSkipCapture?: () => void;
  onSkipTerritoryProcessing?: () => void;
  onSkipRecovery?: () => void;

  // Handlers - Sidebar panel toggle
  onAdvancedPanelsToggle: (show: boolean) => void;

  // Handlers - Recording toggle
  onToggleAutoSave: (enabled: boolean) => void;

  // Handlers - AI tracking
  onRefreshLadderHealth: () => void;
  onCopyLadderHealth: () => void;

  // Handlers - Diagnostics
  onCopyAiTrace: () => void;
  onCopyAiMeta: () => void;
  onExportScenario: () => void;
  onCopyTestFixture: () => void;

  // Handlers - Evaluation
  onRequestEvaluation: () => void;
}

/**
 * SandboxGameSidebar - Extracted sidebar component for the sandbox host.
 *
 * Contains:
 * - Game HUD (desktop or mobile variant)
 * - AI think time progress
 * - Dynamic alerts zone (scenario context, swap sides)
 * - Advanced sidebar panels (replays, history, logs, recording, dev tools)
 * - Touch controls panel
 * - Phase guide
 * - Sandbox notes
 */
export const SandboxGameSidebar: React.FC<SandboxGameSidebarProps> = ({
  hudViewModel,
  gameState,
  config,
  timeControl,
  boardState,
  isMobile,
  isLoggedIn,
  isBeginnerMode,
  developerToolsEnabled,
  lastLoadedScenario,
  boardTypeValue,
  numPlayers,
  selectedPosition,
  selectedStackDetails,
  validTargets: _validTargets,
  primaryValidTargets,
  isCaptureDirectionPending,
  captureTargets,
  overlays,
  phaseLabel,
  phaseHint,
  canSkipCapture,
  canSkipTerritoryProcessing,
  canSkipRecovery,
  isInReplayMode,
  isViewingHistory,
  historyViewIndex,
  hasHistorySnapshots,
  showAdvancedSidebarPanels,
  autoSaveGames,
  gameSaveStatus,
  pendingLocalGames,
  syncState,
  aiThinkingStartedAt,
  aiLadderHealth,
  aiLadderHealthError,
  aiLadderHealthLoading,
  evaluationHistory,
  evaluationError,
  isEvaluating,
  eventLogViewModel,
  sandboxEngine,
  onSwapSides,
  onReplayStateChange,
  onReplayModeChange,
  onReplayAnimationChange,
  onForkFromReplay,
  onHistoryIndexChange,
  onExitHistoryView,
  onEnterHistoryView,
  onClearSelection,
  onShowBoardControls,
  onToggleMovementGrid,
  onToggleValidTargets,
  onToggleLineOverlays,
  onToggleTerritoryOverlays,
  onSkipCapture,
  onSkipTerritoryProcessing,
  onSkipRecovery,
  onAdvancedPanelsToggle,
  onToggleAutoSave,
  onRefreshLadderHealth,
  onCopyLadderHealth,
  onCopyAiTrace,
  onCopyAiMeta,
  onExportScenario,
  onCopyTestFixture,
  onRequestEvaluation,
}) => {
  // Derive sandbox mode notes
  const humanSeatCount =
    gameState?.players.filter((p) => p.type === 'human').length ??
    config.playerTypes.slice(0, config.numPlayers).filter((t) => t === 'human').length;
  const aiSeatCount = (gameState?.players.length ?? config.numPlayers) - humanSeatCount;
  const boardDisplayLabel = boardTypeValue;

  const sandboxModeNotes = [
    `Board: ${boardDisplayLabel}`,
    `${humanSeatCount} human seat${humanSeatCount === 1 ? '' : 's'} · ${aiSeatCount} AI`,
    sandboxEngine
      ? 'Engine parity mode with local AI and choice handler.'
      : 'Legacy local sandbox fallback (no backend).',
    'Runs entirely in-browser; use "Change Setup" to switch configurations.',
    !isLoggedIn ? "You're not logged in; this game runs as a local sandbox only." : null,
  ].filter(Boolean) as string[];

  // Derive AI player info for think time progress
  const currentAiPlayer =
    gameState && gameState.gameStatus === 'active'
      ? gameState.players.find((p) => p.playerNumber === gameState.currentPlayer && p.type === 'ai')
      : null;
  const aiDifficulty = currentAiPlayer
    ? (config.aiDifficulties[currentAiPlayer.playerNumber - 1] ?? 5)
    : 5;
  const aiPlayerName = currentAiPlayer
    ? currentAiPlayer.username || `AI Player ${currentAiPlayer.playerNumber}`
    : '';

  // Derive move history notation options
  const notationOptions = ((): MoveNotationOptions | undefined => {
    if (!gameState) return undefined;
    if (gameState.boardType === 'square8' || gameState.boardType === 'square19') {
      const size = boardState?.size ?? 0;
      return {
        boardType: gameState.boardType,
        boardSizeOverride: size > 0 ? size : undefined,
        squareRankFromBottom: true,
      };
    }
    return undefined;
  })();

  return (
    <aside className="w-full lg:w-80 flex-shrink-0 space-y-3 sm:space-y-4 text-sm text-slate-100">
      {/* Unified Game HUD - full HUD on desktop, compact HUD on mobile */}
      {hudViewModel &&
        (isMobile ? (
          <MobileGameHUD
            isLocalSandboxOnly={!isLoggedIn}
            viewModel={hudViewModel}
            timeControl={timeControl}
            onShowBoardControls={onShowBoardControls}
            rulesUxContext={{
              boardType: boardTypeValue,
              numPlayers,
              aiDifficulty: undefined,
              rulesConcept:
                lastLoadedScenario && lastLoadedScenario.onboarding
                  ? lastLoadedScenario.rulesConcept
                  : undefined,
              scenarioId:
                lastLoadedScenario && lastLoadedScenario.onboarding
                  ? lastLoadedScenario.id
                  : undefined,
            }}
          />
        ) : (
          <GameHUD
            isLocalSandboxOnly={!isLoggedIn}
            viewModel={hudViewModel}
            timeControl={timeControl}
            onShowBoardControls={onShowBoardControls}
            hideVictoryConditions={true}
            rulesUxContext={{
              boardType: boardTypeValue,
              numPlayers,
              aiDifficulty: undefined,
              rulesConcept:
                lastLoadedScenario && lastLoadedScenario.onboarding
                  ? lastLoadedScenario.rulesConcept
                  : undefined,
              scenarioId:
                lastLoadedScenario && lastLoadedScenario.onboarding
                  ? lastLoadedScenario.id
                  : undefined,
            }}
          />
        ))}

      {/* AI Think Time Progress Bar - shows when AI is thinking */}
      {currentAiPlayer && (
        <AIThinkTimeProgress
          isAiThinking={aiThinkingStartedAt !== null}
          thinkingStartedAt={aiThinkingStartedAt}
          aiDifficulty={aiDifficulty}
          aiPlayerName={aiPlayerName}
        />
      )}

      {/* Dynamic alerts zone */}
      <div className="flex flex-col justify-end transition-all duration-200 ease-in-out">
        {/* Onboarding scenario context for curated rules/FAQ scenarios */}
        {lastLoadedScenario && lastLoadedScenario.onboarding && lastLoadedScenario.rulesSnippet && (
          <div className="p-4 border border-emerald-700 rounded-2xl bg-emerald-950/60 space-y-2 mb-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="font-semibold text-sm text-emerald-100">
                Scenario: {lastLoadedScenario.name}
              </h2>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wide bg-emerald-900/80 text-emerald-300 border border-emerald-600/80">
                Rules context
              </span>
            </div>
            {lastLoadedScenario.description && (
              <p className="text-xs text-emerald-100/90">{lastLoadedScenario.description}</p>
            )}
            <p className="text-xs text-emerald-50">{lastLoadedScenario.rulesSnippet}</p>
          </div>
        )}

        {/* Swap sides prompt */}
        {sandboxEngine &&
          gameState &&
          gameState.gameStatus === 'active' &&
          gameState.players.length === 2 &&
          gameState.rulesOptions?.swapRuleEnabled === true &&
          sandboxEngine.canCurrentPlayerSwapSides() && (
            <div className="p-3 border border-amber-500/60 rounded-2xl bg-amber-900/40 text-xs space-y-2">
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
              <p className="text-amber-100/80">
                As Player 2, you may use this once, immediately after Player 1's first turn.
              </p>
            </div>
          )}
      </div>

      {/* Advanced sidebar panels */}
      <details
        className="p-3 border border-slate-700 rounded-2xl bg-slate-900/60"
        open={showAdvancedSidebarPanels || isInReplayMode}
        onToggle={(event) => {
          onAdvancedPanelsToggle(event.currentTarget.open);
        }}
        data-testid="sandbox-advanced-sidebar-panels"
      >
        <summary className="cursor-pointer select-none text-sm font-semibold text-slate-200">
          Advanced panels
          <span className="ml-2 text-[11px] font-normal text-slate-400">
            (replays, logs, recording)
          </span>
        </summary>
        {(showAdvancedSidebarPanels || isInReplayMode) && (
          <div className="mt-3 space-y-3">
            {/* Replay Panel - Game Database Browser */}
            <ReplayPanel
              onStateChange={onReplayStateChange}
              onReplayModeChange={onReplayModeChange}
              onForkFromPosition={onForkFromReplay}
              onAnimationChange={onReplayAnimationChange}
              defaultCollapsed={true}
            />

            {/* History Playback Controls */}
            {!isInReplayMode && gameState && gameState.moveHistory.length > 0 && (
              <HistoryPlaybackPanel
                totalMoves={gameState.moveHistory.length}
                currentMoveIndex={
                  isViewingHistory ? historyViewIndex : gameState.moveHistory.length
                }
                isViewingHistory={isViewingHistory}
                onMoveIndexChange={(index) => {
                  onHistoryIndexChange(index);
                  if (index >= gameState.moveHistory.length) {
                    onExitHistoryView();
                  }
                }}
                onExitHistoryView={onExitHistoryView}
                onEnterHistoryView={onEnterHistoryView}
                hasSnapshots={hasHistorySnapshots}
              />
            )}

            {/* Move History - compact notation display */}
            {!isInReplayMode && gameState && (
              <MoveHistory
                moves={gameState.moveHistory}
                boardType={gameState.boardType}
                currentMoveIndex={
                  isViewingHistory ? historyViewIndex - 1 : gameState.moveHistory.length - 1
                }
                onMoveClick={(index) => {
                  if (hasHistorySnapshots) {
                    onEnterHistoryView();
                    onHistoryIndexChange(index + 1);
                  }
                }}
                notationOptions={notationOptions}
              />
            )}

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60">
              <GameEventLog viewModel={eventLogViewModel} />
            </div>

            {/* Recording Status Panel */}
            <div className="p-3 border border-slate-700 rounded-2xl bg-slate-900/60">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Recording:</span>
                  {gameSaveStatus === 'idle' && autoSaveGames && (
                    <span className="flex items-center gap-1 text-xs text-slate-400">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      Ready
                    </span>
                  )}
                  {gameSaveStatus === 'idle' && !autoSaveGames && (
                    <span className="text-xs text-slate-500">Disabled</span>
                  )}
                  {gameSaveStatus === 'saving' && (
                    <span className="flex items-center gap-1 text-xs text-amber-400">
                      <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                      Saving...
                    </span>
                  )}
                  {gameSaveStatus === 'saved' && (
                    <span className="flex items-center gap-1 text-xs text-emerald-400">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      Saved to server
                    </span>
                  )}
                  {gameSaveStatus === 'saved-local' && (
                    <span className="flex items-center gap-1 text-xs text-amber-300">
                      <span className="w-2 h-2 rounded-full bg-amber-300" />
                      Saved locally
                    </span>
                  )}
                  {gameSaveStatus === 'error' && (
                    <span className="flex items-center gap-1 text-xs text-red-400">
                      <span className="w-2 h-2 rounded-full bg-red-400" />
                      Failed
                    </span>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => onToggleAutoSave(!autoSaveGames)}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition ${
                    autoSaveGames
                      ? 'bg-emerald-900/40 text-emerald-300 border border-emerald-700'
                      : 'bg-slate-800 text-slate-400 border border-slate-600'
                  }`}
                  title={autoSaveGames ? 'Click to disable recording' : 'Click to enable recording'}
                >
                  {autoSaveGames ? 'ON' : 'OFF'}
                </button>
              </div>
              {pendingLocalGames > 0 && (
                <div className="mt-2 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5">
                    {syncState?.status === 'syncing' ? (
                      <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
                    ) : syncState?.status === 'offline' ? (
                      <span className="w-2 h-2 rounded-full bg-slate-500" />
                    ) : syncState?.status === 'error' ? (
                      <span className="w-2 h-2 rounded-full bg-red-400" />
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-amber-400" />
                    )}
                    <span className="text-[10px] text-amber-400">
                      {pendingLocalGames} game{pendingLocalGames !== 1 ? 's' : ''}{' '}
                      {syncState?.status === 'syncing'
                        ? 'syncing...'
                        : syncState?.status === 'offline'
                          ? '(offline)'
                          : 'pending'}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => GameSyncService.triggerSync()}
                    disabled={syncState?.status === 'syncing' || syncState?.status === 'offline'}
                    className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-900/40 text-blue-300 border border-blue-700 hover:bg-blue-800/40 disabled:opacity-50 disabled:cursor-not-allowed transition"
                    title="Sync pending games to server"
                  >
                    Sync
                  </button>
                </div>
              )}
            </div>

            {!isBeginnerMode && developerToolsEnabled && (
              <SandboxDevTools
                aiLadderHealth={aiLadderHealth}
                aiLadderHealthError={aiLadderHealthError}
                aiLadderHealthLoading={aiLadderHealthLoading}
                onRefreshLadderHealth={onRefreshLadderHealth}
                onCopyLadderHealth={onCopyLadderHealth}
                onCopyAiTrace={onCopyAiTrace}
                onCopyAiMeta={onCopyAiMeta}
                onExportScenario={onExportScenario}
                onCopyTestFixture={onCopyTestFixture}
                evaluationHistory={evaluationHistory}
                evaluationError={evaluationError}
                isEvaluating={isEvaluating}
                onRequestEvaluation={onRequestEvaluation}
                gameState={gameState}
                sandboxEngine={sandboxEngine}
                aiDifficulties={config.aiDifficulties}
              />
            )}
          </div>
        )}
      </details>

      {/* Touch Controls Panel */}
      {(isMobile || showAdvancedSidebarPanels) && (
        <SandboxTouchControlsPanel
          selectedPosition={selectedPosition}
          selectedStackDetails={selectedStackDetails}
          validTargets={primaryValidTargets}
          isCaptureDirectionPending={isCaptureDirectionPending}
          captureTargets={captureTargets}
          canUndoSegment={false}
          onClearSelection={onClearSelection}
          onUndoSegment={() => {
            // no-op for now
          }}
          onApplyMove={onClearSelection}
          showMovementGrid={overlays.showMovementGrid}
          onToggleMovementGrid={onToggleMovementGrid}
          showValidTargets={overlays.showValidTargets}
          onToggleValidTargets={onToggleValidTargets}
          showLineOverlays={overlays.showLineOverlays}
          onToggleLineOverlays={
            developerToolsEnabled && onToggleLineOverlays ? onToggleLineOverlays : undefined
          }
          showTerritoryOverlays={overlays.showTerritoryOverlays}
          onToggleTerritoryOverlays={
            developerToolsEnabled && onToggleTerritoryOverlays
              ? onToggleTerritoryOverlays
              : undefined
          }
          phaseLabel={phaseLabel}
          phaseHint={phaseHint}
          canSkipTerritoryProcessing={canSkipTerritoryProcessing}
          onSkipTerritoryProcessing={onSkipTerritoryProcessing}
          canSkipCapture={canSkipCapture}
          onSkipCapture={onSkipCapture}
          canSkipRecovery={canSkipRecovery}
          onSkipRecovery={onSkipRecovery}
          autoSaveGames={autoSaveGames}
          onToggleAutoSave={onToggleAutoSave}
          gameSaveStatus={gameSaveStatus}
        />
      )}

      {/* Phase Guide */}
      <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
        <h2 className="font-semibold">Phase Guide</h2>
        <p className="text-xs uppercase tracking-wide text-slate-400">
          {hudViewModel?.phase.label ?? 'Initializing...'}
        </p>
        <p className="text-sm text-slate-200">
          {hudViewModel?.phase.description ?? 'Setting up game state.'}
        </p>
        <p className="text-xs text-slate-400">
          Complete the current requirement to advance the turn (chain captures, line rewards, etc.).
        </p>
      </div>

      {/* Sandbox Notes */}
      <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
        <h2 className="font-semibold">Sandbox Notes</h2>
        <ul className="list-disc list-inside text-slate-300 space-y-1 text-xs">
          {sandboxModeNotes.map((note, idx) => (
            <li key={idx}>{note}</li>
          ))}
        </ul>
      </div>
    </aside>
  );
};

export default SandboxGameSidebar;
