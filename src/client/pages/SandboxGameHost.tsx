/**
 * @fileoverview SandboxGameHost - ADAPTER/HOST, NOT CANONICAL
 *
 * SSoT alignment: This component is a **React host** that bridges the sandbox
 * engine with client UI. It orchestrates UI state and lifecycle, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Orchestrator: `src/shared/engine/orchestration/turnOrchestrator.ts`
 * - FSM: `src/shared/engine/fsm/TurnStateMachine.ts`
 *
 * This host:
 * - Owns sandbox configuration (board type, seats, player kinds) via SandboxContext
 * - Starts sandbox games using ClientSandboxEngine
 * - Wires sandbox board interactions and local AI via useSandboxInteractions
 * - Renders sandbox-specific HUD (players, selection, phase help, stall diagnostics)
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal, type TrainingSubmissionState } from '../components/VictoryModal';
import { getReplayService } from '../services/ReplayService';
import type { TrainingMoveRecord } from '../types/replay';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { ScenarioPickerModal } from '../components/ScenarioPickerModal';
import { SelfPlayBrowser } from '../components/SelfPlayBrowser';
import { SaveStateDialog } from '../components/SaveStateDialog';
import { RingPlacementCountDialog } from '../components/RingPlacementCountDialog';
import { RecoveryLineChoiceDialog } from '../components/RecoveryLineChoiceDialog';
import { TerritoryRegionChoiceDialog } from '../components/TerritoryRegionChoiceDialog';
import { LineRewardPanel } from '../components/LineRewardPanel';
import { StatusBanner } from '../components/ui/StatusBanner';
import { Button } from '../components/ui/Button';
import { OnboardingModal } from '../components/OnboardingModal';
// Sandbox components for decomposed view
import { SandboxBoardSection, SandboxGameSidebar } from '../components/sandbox';
import { useFirstTimePlayer } from '../hooks/useFirstTimePlayer';
import { useTutorialHints } from '../hooks/useTutorialHints';
import { TutorialHintBanner } from '../components/tutorial/TutorialHintBanner';
import { TeachingOverlay, type TeachingTopic } from '../components/TeachingOverlay';
import {
  SandboxGameConfig,
  BOARD_PRESETS,
  QUICK_START_PRESETS,
} from '../components/SandboxGameConfig';
import type { LoadableScenario } from '../sandbox/scenarioTypes';
import {
  BoardState,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  positionToString,
  TimeControl,
  RingEliminationChoice,
} from '../../shared/types/game';
import { enumerateTerritoryEliminationMoves } from '../../shared/engine';
import { useAuth } from '../contexts/AuthContext';
import { useAccessibility } from '../contexts/AccessibilityContext';
import { useSandbox, LocalConfig, LocalPlayerType } from '../contexts/SandboxContext';
import { useSandboxInteractions } from '../hooks/useSandboxInteractions';
import { useAutoMoveAnimation } from '../hooks/useMoveAnimation';
import {
  toBoardViewModel,
  toEventLogViewModel,
  toVictoryViewModel,
  toHUDViewModel,
  deriveBoardDecisionHighlights,
} from '../adapters/gameViewModels';
import {
  ScreenReaderAnnouncer,
  useGameAnnouncements,
  useGameStateAnnouncements,
} from '../components/ScreenReaderAnnouncer';
import type {
  ClientSandboxEngine,
  SandboxInteractionHandler,
} from '../sandbox/ClientSandboxEngine';
import {
  logSandboxScenarioLoaded,
  logSandboxScenarioCompleted,
} from '../sandbox/sandboxRulesUxTelemetry';
import { getGameOverBannerText } from '../utils/gameCopy';
import { useSandboxDiagnostics } from '../hooks/useSandboxDiagnostics';
import { useIsMobile } from '../hooks/useIsMobile';
import { useBoardOverlays } from '../hooks/useBoardViewProps';
import { useSandboxPersistence } from '../hooks/useSandboxPersistence';
import { useSandboxEvaluation } from '../hooks/useSandboxEvaluation';
import { useSandboxScenarios, type LoadedScenario } from '../hooks/useSandboxScenarios';
import { useGameSoundEffects } from '../hooks/useGameSoundEffects';
import { useSandboxClock } from '../hooks/useSandboxClock';
import { useSandboxAITracking } from '../hooks/useSandboxAITracking';
import { useSandboxAIServiceStatus } from '../hooks/useSandboxAIServiceStatus';
import { useSandboxBoardSelection } from '../hooks/useSandboxBoardSelection';
import { useSandboxGameLifecycle } from '../hooks/useSandboxGameLifecycle';
import { useGlobalGameShortcuts } from '../hooks/useKeyboardNavigation';
import { useSoundOptional } from '../contexts/SoundContext';
import { AIServiceStatusBanner } from '../components/AIServiceStatusBanner';

const PHASE_COPY: Record<
  string,
  {
    label: string;
    summary: string;
  }
> = {
  ring_placement: {
    label: 'Ring Placement',
    summary:
      'Place new rings or add to existing stacks while keeping at least one real move available for your next turn.',
  },
  movement: {
    label: 'Movement',
    summary:
      'Pick a stack and move it in a straight line at least as far as its height, and farther if the path stays clear (stacks and territory block; markers do not).',
  },
  // Support both 'capture' and 'chain_capture' phase keys for compatibility
  capture: {
    label: 'Capture',
    summary:
      'Start an overtaking capture by jumping over an adjacent stack and landing on an empty or marker space beyond it.',
  },
  chain_capture: {
    label: 'Chain Capture',
    summary:
      'Continue the capture chain: you must keep jumping while any capture exists, but you choose which capture direction to take next.',
  },
  line_processing: {
    label: 'Line Completion',
    summary:
      'Resolve completed marker lines into Territory; pay the line cost (one ring from any stack you control, including standalone rings).',
  },
  territory_processing: {
    label: 'Territory Claim',
    summary:
      'Collapse disconnected regions into Territory; pay the entire cap from any controlled stack outside each region (including height-1 standalone rings).',
  },
};

/**
 * Host component for the local sandbox experience.
 *
 * Responsibilities:
 * - Own sandbox configuration (board type, seats, player kinds) via SandboxContext
 * - Start sandbox games using ClientSandboxEngine, optionally attempting a backend game first
 * - Wire sandbox board interactions and local AI via useSandboxInteractions
 * - Render sandbox-specific HUD (players, selection, phase help, stall diagnostics)
 *
 * Rules semantics remain in the shared TS engine + orchestrator; this host only orchestrates
 * sandbox UI and engine lifecycle.
 */
export const SandboxGameHost: React.FC = () => {
  const _navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user } = useAuth();
  const { colorVisionMode, effectiveReducedMotion } = useAccessibility();
  const {
    shouldShowWelcome,
    markWelcomeSeen,
    markGameCompleted,
    isFirstTimePlayer,
    state: onboardingState,
    markPhaseHintSeen,
    setTutorialHintsEnabled,
  } = useFirstTimePlayer();
  const presetParam = searchParams.get('preset');

  const {
    config,
    setConfig,
    isConfigured,
    backendSandboxError,
    setBackendSandboxError,
    sandboxEngine,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    sandboxCaptureTargets,
    setSandboxCaptureTargets,
    sandboxLastProgressAt: _sandboxLastProgressAt,
    setSandboxLastProgressAt,
    sandboxStallWarning,
    setSandboxStallWarning,
    sandboxStateVersion: _sandboxStateVersion,
    setSandboxStateVersion,
    developerToolsEnabled,
    setDeveloperToolsEnabled: _setDeveloperToolsEnabled,
    sandboxMode: _sandboxMode,
    setSandboxMode,
    isBeginnerMode,
    initLocalSandboxEngine,
    resetSandboxEngine,
  } = useSandbox();

  // Local-only diagnostics / UX state
  const [isSandboxVictoryModalDismissed, setIsSandboxVictoryModalDismissed] = useState(false);

  // Training submission state (January 2026 - Human game training)
  const [trainingSubmissionState, setTrainingSubmissionState] = useState<TrainingSubmissionState>({
    isAvailable: false,
    isSubmitting: false,
    wasSubmitted: false,
    error: null,
  });

  // Game view once configured (local sandbox) - needed early for clock hook
  const sandboxGameStateForClock: GameState | null = sandboxEngine
    ? sandboxEngine.getGameState()
    : null;

  // Sandbox clock: time control state via extracted hook
  const {
    clockEnabled: sandboxClockEnabled,
    setClockEnabled: setSandboxClockEnabled,
    timeControl: sandboxTimeControl,
    playerTimes: sandboxPlayerTimes,
    resetPlayerTimes: resetSandboxPlayerTimes,
  } = useSandboxClock(sandboxGameStateForClock);

  // Selection + valid target highlighting (extracted to useSandboxBoardSelection hook)
  const [boardSelection, boardSelectionActions] = useSandboxBoardSelection();
  // Destructure for convenience - 'selected' is used extensively throughout the component
  // Note: internal 'selected' remains Position | undefined for compatibility with useSandboxInteractions
  const selected = boardSelection.selectedCell ?? undefined;
  const validTargets = boardSelection.highlightedCells;
  const setSelected = (pos: Position | undefined) =>
    boardSelectionActions.setSelectedCell(pos ?? null);
  const setValidTargets = boardSelectionActions.setHighlightedCells;

  // When a self-play scenario is loaded, this bridges the gameId into the
  // ReplayPanel so it can attempt to drive the board from the AI service's
  // /api/replay endpoints (Option A).
  const [_requestedReplayGameId, setRequestedReplayGameId] = useState<string | null>(null);

  // Sandbox diagnostics: save state dialog, export handlers, debug utilities
  // Extracted via useSandboxDiagnostics hook for cleaner component structure
  const { state: diagnosticsState, actions: diagnosticsActions } = useSandboxDiagnostics(
    sandboxEngine ? sandboxEngine.getGameState() : null
  );

  // Board overlay visibility configuration - using extracted hook
  // Start with movement grid overlay enabled by default; it helps
  // players understand valid moves and adjacency patterns.
  const {
    overlays,
    setShowMovementGrid,
    setShowValidTargets,
    setShowLineOverlays,
    setShowTerritoryOverlays,
  } = useBoardOverlays({
    showMovementGrid: true,
    showValidTargets: true,
    showLineOverlays: true,
    showTerritoryOverlays: true,
  });

  // Help / controls overlay for the active sandbox host
  const [showBoardControls, setShowBoardControls] = useState(false);

  // Sidebar density: keep replay/diagnostics tools available without
  // overwhelming onboarding and casual sandbox sessions.
  const sandboxAdvancedSidebarStorageKey = 'ringrift_sandbox_sidebar_show_advanced';
  const [showAdvancedSidebarPanels, setShowAdvancedSidebarPanels] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return localStorage.getItem(sandboxAdvancedSidebarStorageKey) === 'true';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(
        sandboxAdvancedSidebarStorageKey,
        showAdvancedSidebarPanels ? 'true' : 'false'
      );
    } catch {
      // Storage might be disabled; ignore.
    }
  }, [sandboxAdvancedSidebarStorageKey, showAdvancedSidebarPanels]);

  const sound = useSoundOptional();

  useGlobalGameShortcuts({
    onShowHelp: () => {
      if (!isConfigured || !sandboxEngine) {
        return;
      }
      setShowBoardControls((prev) => !prev);
    },
    onToggleMute: () => {
      if (!sound) {
        return;
      }
      const nextMuted = !sound.muted;
      sound.toggleMute();
      toast(nextMuted ? 'Muted' : 'Sound on', { id: 'mute-toggle' });
    },
    onToggleFullscreen: () => {
      if (typeof document === 'undefined') return;

      if (document.fullscreenElement) {
        void document.exitFullscreen?.();
        return;
      }

      void document.documentElement.requestFullscreen?.();
    },
  });

  // Ref for resolving pending player choices in sandbox mode
  const sandboxChoiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  // Factory for creating interaction handlers for sandbox engine instances
  const createSandboxInteractionHandler = useCallback(
    (playerTypesSnapshot: LocalPlayerType[]): SandboxInteractionHandler => {
      return {
        async requestChoice<TChoice extends PlayerChoice>(
          choice: TChoice
        ): Promise<PlayerChoiceResponseFor<TChoice>> {
          const playerKind = playerTypesSnapshot[choice.playerNumber - 1] ?? 'human';

          // AI players: pick a random option without involving the UI.
          if (playerKind === 'ai') {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any -- generic choice type narrowing
            const options = (choice as any).options as TChoice['options'];
            const optionsArray = (options as unknown[]) ?? [];
            if (optionsArray.length === 0) {
              throw new Error('SandboxInteractionHandler: no options available for AI choice');
            }
            const selectedOption = optionsArray[
              Math.floor(Math.random() * optionsArray.length)
            ] as TChoice['options'][number];

            return {
              choiceId: choice.id,
              playerNumber: choice.playerNumber,
              choiceType: choice.type,
              selectedOption,
            } as PlayerChoiceResponseFor<TChoice>;
          }

          // Human players with a single available option: auto-resolve the
          // choice without surfacing any blocking UI.
          // RR-FIX-2026-01-10: Exception for ring_elimination - always show UI for visual feedback
          // so human players see stack highlighting and confirmation even for single-option eliminations.
          // eslint-disable-next-line @typescript-eslint/no-explicit-any -- polymorphic choice options access
          const rawOptions = (choice as any).options as TChoice['options'] | undefined;
          const autoOptions = (rawOptions as unknown[]) ?? [];
          if (autoOptions.length === 1 && choice.type !== 'ring_elimination') {
            const onlyOption = autoOptions[0] as TChoice['options'][number];
            return {
              choiceId: choice.id,
              playerNumber: choice.playerNumber,
              choiceType: choice.type,
              selectedOption: onlyOption,
            } as PlayerChoiceResponseFor<TChoice>;
          }

          // Human players
          if (choice.type === 'capture_direction') {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any -- capture_direction type narrowing
            const anyChoice = choice as any;
            const options = (anyChoice.options ?? []) as Array<{ landingPosition: Position }>;
            const targets: Position[] = options.map((opt) => opt.landingPosition);
            setSandboxCaptureChoice(choice);
            setSandboxCaptureTargets(targets);
          } else {
            // RR-DEBUG-2026-01-10: Log pending choice for territory elimination debugging
            // eslint-disable-next-line no-console
            console.log('[SandboxInteractionHandler] Setting sandboxPendingChoice:', {
              choiceType: choice.type,
              choiceId: choice.id,
              playerNumber: choice.playerNumber,
              optionsCount:
                'options' in choice && Array.isArray(choice.options) ? choice.options.length : 0,
              options:
                'options' in choice && Array.isArray(choice.options)
                  ? choice.options.map((opt: { stackPosition?: unknown; regionId?: unknown }) => ({
                      stackPosition: opt.stackPosition,
                      regionId: opt.regionId,
                    }))
                  : undefined,
            });
            // RR-FIX-2026-01-10: Clear any stale capture choice to ensure activePendingChoice
            // uses sandboxPendingChoice. This prevents a leftover sandboxCaptureChoice from
            // blocking ring_elimination or region_order UI.
            setSandboxCaptureChoice(null);
            setSandboxCaptureTargets([]);
            setSandboxPendingChoice(choice);
          }

          return new Promise<PlayerChoiceResponseFor<TChoice>>((resolve) => {
            sandboxChoiceResolverRef.current = ((
              response: PlayerChoiceResponseFor<PlayerChoice>
            ) => {
              resolve(response as PlayerChoiceResponseFor<TChoice>);
            }) as (response: PlayerChoiceResponseFor<PlayerChoice>) => void;
          });
        },
      };
    },
    [setSandboxCaptureChoice, setSandboxCaptureTargets, setSandboxPendingChoice]
  );

  // ═══════════════════════════════════════════════════════════════════════════
  // SCENARIO/REPLAY STATE - using extracted scenarios hook
  // ═══════════════════════════════════════════════════════════════════════════

  // Callback to reset UI state when loading a new scenario
  const resetGameUIState = useCallback(() => {
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
  }, []);

  // Callback to initialize sandbox engine with a scenario
  const initSandboxWithScenario = useCallback(
    (scenario: LoadableScenario): ClientSandboxEngine | null => {
      // Update config to match scenario settings
      setConfig((prev) => ({
        ...prev,
        boardType: scenario.boardType,
        numPlayers: scenario.playerCount,
      }));

      // Determine player types. For general fixtures, default to human vs AI.
      // For recorded self-play games, treat all seats as human so we don't
      // auto-run local AI turns while replaying the canonical move sequence.
      let playerTypes: LocalPlayerType[] =
        scenario.playerCount === 2
          ? ['human', 'ai', 'human', 'human']
          : (config.playerTypes.slice(0, scenario.playerCount) as LocalPlayerType[]);

      if (scenario.selfPlayMeta) {
        playerTypes = Array.from(
          { length: scenario.playerCount },
          () => 'human' as LocalPlayerType
        );
      }

      // Create interaction handler
      const interactionHandler = createSandboxInteractionHandler(playerTypes);

      // Initialize sandbox engine (use config's aiDifficulties for scenarios)
      const engine = initLocalSandboxEngine({
        boardType: scenario.boardType,
        numPlayers: scenario.playerCount,
        playerTypes,
        aiDifficulties: config.aiDifficulties.slice(0, scenario.playerCount),
        interactionHandler,
      });

      if (!engine) return null;

      // Normalize terminal states to active (completed self-play snapshots)
      const scenarioState = scenario.state;
      const isTerminalStatus =
        scenarioState.gameStatus === 'completed' || scenarioState.gameStatus === 'finished';
      const normalizedState = isTerminalStatus
        ? {
            ...scenarioState,
            gameStatus: 'active',
            currentPhase: 'ring_placement',
            chainCapturePosition: undefined,
          }
        : scenarioState;

      engine.initFromSerializedState(
        normalizedState,
        playerTypes,
        interactionHandler,
        config.aiDifficulties.slice(0, scenario.playerCount)
      );
      return engine;
    },
    [
      config.aiDifficulties,
      config.playerTypes,
      createSandboxInteractionHandler,
      initLocalSandboxEngine,
    ]
  );

  // Callback when scenario is loaded (for telemetry)
  const handleScenarioLoadComplete = useCallback((scenario: LoadedScenario) => {
    void logSandboxScenarioLoaded(scenario as LoadableScenario);
  }, []);

  // Callback to bump state version
  const handleStateVersionChange = useCallback(() => {
    setSandboxStateVersion((v) => v + 1);
  }, []);

  // Scenarios hook - manages replay, history, and scenario state
  const {
    lastLoadedScenario,
    setLastLoadedScenario,
    showScenarioPicker,
    setShowScenarioPicker,
    showSelfPlayBrowser,
    setShowSelfPlayBrowser,
    isInReplayMode,
    setIsInReplayMode,
    replayState,
    setReplayState,
    replayAnimation,
    setReplayAnimation,
    isViewingHistory,
    setIsViewingHistory,
    historyViewIndex,
    setHistoryViewIndex,
    hasHistorySnapshots,
    setHasHistorySnapshots,
    handleLoadScenario: hookHandleLoadScenario,
    handleForkFromReplay: hookHandleForkFromReplay,
    handleResetScenario: _hookHandleResetScenario,
    clearScenarioContext,
    originalScenarioRef,
  } = useSandboxScenarios<LoadableScenario>({
    initSandboxWithScenario,
    onScenarioLoaded: handleScenarioLoadComplete,
    onStateVersionChange: handleStateVersionChange,
    onUIStateReset: resetGameUIState,
  });

  // AI evaluation state - using extracted evaluation hook
  const {
    evaluationHistory: sandboxEvaluationHistory,
    evaluationError: sandboxEvaluationError,
    isEvaluating: isSandboxAnalysisRunning,
    requestEvaluation: requestSandboxEvaluation,
    clearHistory: _clearEvaluationHistory,
  } = useSandboxEvaluation({
    engine: sandboxEngine,
    developerToolsEnabled,
    isInReplayMode,
    isViewingHistory,
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SCENARIO HANDLERS - wrap hook handlers with self-play specific logic
  // ═══════════════════════════════════════════════════════════════════════════

  /**
   * Load a scenario from the scenario picker.
   * Wraps the hook handler to add self-play replay logic.
   */
  const handleLoadScenario = useCallback(
    (scenario: LoadableScenario) => {
      try {
        // Reset state version before loading
        setSandboxStateVersion(0);

        // Call the hook handler (handles state management and engine creation)
        // The hook now returns the engine synchronously so we can use it for replay
        const engine = hookHandleLoadScenario(scenario);

        // Handle self-play specific logic
        if (scenario.selfPlayMeta) {
          const recordedMoves: Move[] | undefined = scenario.selfPlayMeta.moves;

          if (recordedMoves && recordedMoves.length > 0 && engine) {
            // Capture engine reference for the async closure
            const replayEngine = engine;

            void (async () => {
              // eslint-disable-next-line no-console
              console.log('[SandboxSelfPlayReplay] Replaying recorded self-play game', {
                gameId: scenario.selfPlayMeta?.gameId,
                totalRecordedMoves: recordedMoves.length,
              });

              let appliedCount = 0;

              try {
                for (let i = 0; i < recordedMoves.length; i += 1) {
                  const move = recordedMoves[i];

                  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- canonical move replay accepts polymorphic move type
                  await replayEngine.applyCanonicalMoveForReplay(move as any);
                  appliedCount += 1;
                }

                setSandboxStateVersion((v) => v + 1);
                setHasHistorySnapshots(true);

                // eslint-disable-next-line no-console
                console.log('[SandboxSelfPlayReplay] Finished local replay', {
                  gameId: scenario.selfPlayMeta?.gameId,
                  appliedMoves: appliedCount,
                  historyLength: replayEngine.getGameState().moveHistory.length,
                });
              } catch (err) {
                const failedMove =
                  appliedCount < recordedMoves.length ? recordedMoves[appliedCount] : null;
                console.error(
                  '[SandboxGameHost] Failed to replay recorded self-play game into sandbox engine',
                  {
                    errorMessage: err instanceof Error ? err.message : String(err),
                    errorStack: err instanceof Error ? err.stack : undefined,
                    gameId: scenario.selfPlayMeta?.gameId,
                    appliedMoves: appliedCount,
                    totalRecordedMoves: recordedMoves.length,
                    failedMoveIndex: appliedCount,
                    failedMove: failedMove
                      ? {
                          type: failedMove.type,
                          player: failedMove.player,
                          from: failedMove.from,
                          to: failedMove.to,
                        }
                      : null,
                  }
                );
                setHasHistorySnapshots(false);
              }
            })();
          } else {
            // No recorded moves or no engine; disable snapshot-driven history
            setHasHistorySnapshots(false);
          }

          // Bridge gameId into ReplayPanel for Option A
          setRequestedReplayGameId(scenario.selfPlayMeta.gameId);
        } else {
          setRequestedReplayGameId(null);
        }

        toast.success(`Loaded scenario: ${scenario.name}`);
      } catch (err) {
        console.error('[SandboxGameHost] Error loading scenario:', err);
        toast.error('Failed to load scenario');
      }
    },
    [hookHandleLoadScenario, setHasHistorySnapshots]
  );

  /**
   * Fork from a replay position - wraps the hook handler.
   */
  const handleForkFromReplay = useCallback(
    (state: GameState) => {
      // Reset state version
      setSandboxStateVersion(0);

      // Call hook handler (uses move index 0 for forking which is fine)
      hookHandleForkFromReplay(state, state.moveHistory?.length ?? 0);

      // Clear requestedReplayGameId
      setRequestedReplayGameId(null);
    },
    [hookHandleForkFromReplay]
  );

  /**
   * Reset to last loaded scenario - wraps the hook handler.
   */
  const handleResetScenario = useCallback(() => {
    const scenario = originalScenarioRef.current;
    if (!scenario) return;

    // Use handleLoadScenario to ensure self-play replay happens again
    handleLoadScenario(scenario);
  }, [handleLoadScenario, originalScenarioRef]);

  // Game storage state - using extracted persistence hook
  const {
    autoSaveGames,
    setAutoSaveGames,
    gameSaveStatus,
    pendingLocalGames,
    syncState,
    initialGameStateRef: _initialGameStateRef,
    gameSavedRef: _gameSavedRef,
    cloneInitialGameState: _cloneInitialGameState,
  } = useSandboxPersistence({
    engine: sandboxEngine,
    playerTypes: config.playerTypes as LocalPlayerType[],
    numPlayers: config.numPlayers,
    stateVersion: _sandboxStateVersion,
  });

  // Screen reader announcements for accessibility - using priority queue (mirrors BackendGameHost)
  const { queue: announcementQueue, announce, removeAnnouncement } = useGameAnnouncements();

  const lastSandboxPhaseRef = useRef<string | null>(null);
  // RR-FIX-2026-01-12: Track the last processed pending choice ID to prevent infinite re-render
  // when setting elimination targets. Without this, the useEffect that handles ring_elimination
  // choices would call setValidTargets on every render, triggering itself via validTargets.length.
  const lastProcessedPendingChoiceIdRef = useRef<string | null>(null);
  // RR-FIX-2026-01-12: Track chain capture initialization to prevent redundant effect runs.
  // Key format: `${fromX}-${fromY}-${landingsCount}-${moveHistoryLength}` to detect state changes.
  const lastChainCaptureInitRef = useRef<string | null>(null);

  // Sandbox-only visual cue: transient highlight of newly-collapsed line
  // segments, populated from the engine after automatic line processing.
  const [recentLineHighlights, setRecentLineHighlights] = useState<Position[]>([]);

  const {
    handleCellClick: handleSandboxCellClick,
    handleCellDoubleClick: handleSandboxCellDoubleClick,
    handleCellContextMenu: handleSandboxCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection: clearSandboxSelection,
    shakingCellKey: sandboxShakingCellKey,
    ringPlacementCountPrompt: sandboxRingPlacementCountPrompt,
    closeRingPlacementCountPrompt,
    confirmRingPlacementCountPrompt,
    recoveryChoicePromptOpen: sandboxRecoveryChoicePromptOpen,
    resolveRecoveryChoice: resolveSandboxRecoveryChoice,
    territoryRegionPrompt: sandboxTerritoryRegionPrompt,
    closeTerritoryRegionPrompt,
    confirmTerritoryRegionPrompt,
  } = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef: sandboxChoiceResolverRef,
  });

  // Game lifecycle: start, reset, rematch, and preset handlers via extracted hook
  const { actions: lifecycleActions } = useSandboxGameLifecycle({
    config,
    setConfig,
    user,
    initLocalSandboxEngine,
    resetSandboxEngine,
    createSandboxInteractionHandler,
    maybeRunSandboxAiIfNeeded,
    clearScenarioContext,
    resetGameUIState,
    markWelcomeSeen,
    setBackendSandboxError,
    setSelected,
    setValidTargets,
    setSandboxPendingChoice,
    setSandboxCaptureChoice,
    setSandboxCaptureTargets,
    setSandboxStallWarning,
    setSandboxLastProgressAt,
    setSandboxStateVersion,
    setLastLoadedScenario,
    setIsSandboxVictoryModalDismissed,
    setSandboxMode,
  });

  const isMobile = useIsMobile();

  // Consume any recent line highlights from the sandbox engine whenever the
  // sandbox state version advances. Highlights are cleared automatically
  // after a short delay so they behave as a brief visual cue rather than a
  // persistent overlay.
  useEffect(() => {
    if (!sandboxEngine) {
      setRecentLineHighlights([]);
      return;
    }

    const positions =
      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- duck-typing for optional engine method
      typeof (sandboxEngine as any).consumeRecentLineHighlights === 'function'
        ? sandboxEngine.consumeRecentLineHighlights()
        : [];

    if (positions.length === 0) {
      setRecentLineHighlights([]);
      return;
    }

    setRecentLineHighlights(positions);

    const timeoutId = window.setTimeout(() => {
      setRecentLineHighlights([]);
    }, 1800);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [sandboxEngine, _sandboxStateVersion]);

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig((prev) => {
      const numPlayers = partial.numPlayers;
      return {
        ...prev,
        ...partial,
        playerTypes: numPlayers
          ? prev.playerTypes.map((t, idx) => (idx < numPlayers ? t : prev.playerTypes[idx]))
          : prev.playerTypes,
        aiDifficulties: numPlayers
          ? prev.aiDifficulties.map((d, idx) => (idx < numPlayers ? d : prev.aiDifficulties[idx]))
          : prev.aiDifficulties,
      };
    });
  };

  const handlePlayerTypeChange = (index: number, type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      next[index] = type;
      return { ...prev, playerTypes: next };
    });
  };

  const handleAIDifficultyChange = (index: number, difficulty: number) => {
    setConfig((prev) => {
      const next = [...prev.aiDifficulties];
      next[index] = difficulty;
      return { ...prev, aiDifficulties: next };
    });
  };

  // Game lifecycle functions are now provided by useSandboxGameLifecycle hook:
  // - lifecycleActions.startLocalGame(config) - start local-only sandbox
  // - lifecycleActions.startGame(config) - attempt backend, fallback to local
  // - lifecycleActions.applyQuickStartPreset(preset) - apply preset and start
  // - lifecycleActions.resetToSetup() - return to setup screen
  // - lifecycleActions.rematch() - start new game with same config

  // Wrapper to match QUICK_START_PRESETS type with hook's QuickStartPreset type
  const handleQuickStartPreset = (preset: (typeof QUICK_START_PRESETS)[number]) => {
    lifecycleActions.applyQuickStartPreset(preset);
  };

  const presetHandledRef = useRef<string | null>(null);

  useEffect(() => {
    if (!presetParam) {
      presetHandledRef.current = null;
      return;
    }

    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete('preset');
        return next;
      },
      { replace: true }
    );

    if (presetHandledRef.current === presetParam) {
      return;
    }
    presetHandledRef.current = presetParam;

    const preset = QUICK_START_PRESETS.find((p) => p.id === presetParam);
    if (!preset) {
      toast.error(`Unknown sandbox preset: ${presetParam}`);
      return;
    }

    handleQuickStartPreset(preset);
  }, [presetParam, setSearchParams, handleQuickStartPreset]);

  // Handler for starting tutorial from onboarding modal - selects "Learn the Basics"
  const handleStartTutorial = useCallback(() => {
    const learnBasicsPreset = QUICK_START_PRESETS.find((p) => p.id === 'learn-basics');
    if (learnBasicsPreset) {
      handleQuickStartPreset(learnBasicsPreset);
    }
  }, [handleQuickStartPreset]);

  const handleStartLocalGame = async () => {
    await lifecycleActions.startGame(config);
  };

  // Game view once configured (local sandbox)
  const sandboxGameState: GameState | null = sandboxEngine ? sandboxEngine.getGameState() : null;

  // Tutorial hints for "Learn the Basics" mode
  // Active when in beginner mode and hints are enabled
  const isLearnBasicsMode = isBeginnerMode && onboardingState.tutorialHintsEnabled;
  const { currentHint, dismissHint, getTeachingTopic } = useTutorialHints({
    gameState: sandboxGameState,
    isLearnMode: isLearnBasicsMode,
    seenPhases: onboardingState.seenTutorialPhases,
    hintsEnabled: onboardingState.tutorialHintsEnabled,
  });

  // Teaching overlay state for "Learn More" from hints
  const [showTeachingOverlay, setShowTeachingOverlay] = useState(false);
  const [teachingTopic, setTeachingTopic] = useState<TeachingTopic | null>(null);

  // AI tracking: timing, diagnostics, ladder health via extracted hook
  const { state: aiTrackingState, actions: aiTrackingActions } = useSandboxAITracking(
    sandboxEngine,
    sandboxGameState,
    maybeRunSandboxAiIfNeeded
  );

  // AI service status: tracks connection status and fallback mode
  const { state: aiServiceState, actions: aiServiceActions } = useSandboxAIServiceStatus();

  const sandboxVictoryResult = sandboxEngine ? sandboxEngine.getVictoryResult() : null;
  const sandboxGameEndExplanation = sandboxEngine ? sandboxEngine.getGameEndExplanation() : null;

  // Derive current player info for announcements
  const currentPlayerForAnnouncements = sandboxGameState?.players.find(
    (p) => p.playerNumber === sandboxGameState.currentPlayer
  );
  const currentPlayerName =
    currentPlayerForAnnouncements?.username || `Player ${sandboxGameState?.currentPlayer ?? 1}`;
  // In sandbox, player 1 is typically the local human player
  const isLocalPlayerTurn = sandboxGameState?.currentPlayer === 1;

  // Sound effects for game events (phase changes, turns, moves, game end)
  // In sandbox mode, player 1 is the local human player
  useGameSoundEffects({
    gameState: sandboxGameState,
    victoryState: sandboxVictoryResult,
    currentUserId: undefined, // No user auth in sandbox
    myPlayerNumber: 1, // Local human is always player 1 in sandbox
  });

  // Map victory reason to the type expected by GameAnnouncements
  const mapVictoryCondition = (
    reason: string | undefined
  ): 'elimination' | 'territory' | 'last_player_standing' => {
    switch (reason) {
      case 'ring_elimination':
        return 'elimination';
      case 'territory_control':
        return 'territory';
      default:
        return 'last_player_standing';
    }
  };

  // Use the automatic game state announcements hook
  useGameStateAnnouncements({
    currentPlayerName,
    isYourTurn: isLocalPlayerTurn,
    phase: sandboxGameState?.currentPhase,
    previousPhase: undefined, // Let the hook track this internally
    phaseDescription: undefined,
    timeRemaining: null,
    isGameOver: !!sandboxVictoryResult,
    winnerName:
      sandboxVictoryResult?.winner !== undefined
        ? sandboxGameState?.players.find((p) => p.playerNumber === sandboxVictoryResult.winner)
            ?.username || `Player ${sandboxVictoryResult.winner}`
        : undefined,
    victoryCondition: sandboxVictoryResult
      ? mapVictoryCondition(sandboxVictoryResult.reason)
      : undefined,
    isWinner: sandboxVictoryResult?.winner === 1, // Player 1 is typically local human
    announce,
  });

  // Track victory for onboarding and telemetry (separate from announcements)
  const prevVictoryRef = useRef<boolean>(false);
  useEffect(() => {
    if (sandboxVictoryResult && !prevVictoryRef.current) {
      prevVictoryRef.current = true;

      // Mark first game completed for onboarding tracking
      markGameCompleted();

      // Emit sandbox_scenario_completed for curated teaching scenarios when applicable.
      if (lastLoadedScenario) {
        void logSandboxScenarioCompleted({
          scenario: lastLoadedScenario,
          victoryReason: sandboxVictoryResult.reason,
        });
      }
    }

    // Reset victory ref when starting a new game
    if (!sandboxVictoryResult && prevVictoryRef.current) {
      prevVictoryRef.current = false;
    }
  }, [sandboxVictoryResult, markGameCompleted, lastLoadedScenario]);

  // Update training submission availability when victory occurs (January 2026)
  useEffect(() => {
    if (!sandboxVictoryResult || !sandboxGameState) {
      setTrainingSubmissionState((prev) => ({
        ...prev,
        isAvailable: false,
        wasSubmitted: false,
        error: null,
      }));
      return;
    }

    // Check if this was a human vs AI game where human won
    const players = sandboxGameState.players;
    const humanPlayers = players.filter((p) => p.type === 'human');
    const aiPlayers = players.filter((p) => p.type === 'ai');
    const isHumanVsAI = humanPlayers.length > 0 && aiPlayers.length > 0;

    if (!isHumanVsAI) {
      setTrainingSubmissionState((prev) => ({ ...prev, isAvailable: false }));
      return;
    }

    // Check if human won (player 1 is typically human in sandbox)
    const humanPlayer = humanPlayers[0];
    const humanWon = sandboxVictoryResult.winner === humanPlayer?.playerNumber;

    // Training only accepts human wins and needs move history
    const hasMoveHistory = sandboxGameState.moveHistory && sandboxGameState.moveHistory.length > 0;

    setTrainingSubmissionState((prev) => ({
      ...prev,
      isAvailable: humanWon && hasMoveHistory,
      wasSubmitted: false,
      error: null,
    }));
  }, [sandboxVictoryResult, sandboxGameState]);

  // Callback to submit game for training (January 2026)
  const handleSubmitForTraining = useCallback(async () => {
    if (!sandboxGameState || !sandboxVictoryResult) return;

    const players = sandboxGameState.players;
    const humanPlayers = players.filter((p) => p.type === 'human');
    const aiPlayers = players.filter((p) => p.type === 'ai');
    const humanPlayer = humanPlayers[0];
    const aiPlayer = aiPlayers[0];

    if (!humanPlayer || !aiPlayer) return;

    setTrainingSubmissionState((prev) => ({
      ...prev,
      isSubmitting: true,
      error: null,
    }));

    try {
      const replayService = getReplayService();

      // Convert moves to training format
      const trainingMoves: TrainingMoveRecord[] = sandboxGameState.moveHistory.map((move) => ({
        type: move.type,
        player: move.player,
        from: move.from ? { x: move.from.x, y: move.from.y } : undefined,
        to: move.to ? { x: move.to.x, y: move.to.y } : undefined,
        captureTarget: move.captureTarget
          ? { x: move.captureTarget.x, y: move.captureTarget.y }
          : undefined,
      }));

      const result = await replayService.submitForTraining({
        board_type: sandboxGameState.boardType,
        num_players: players.length,
        moves: trainingMoves,
        winner: sandboxVictoryResult.winner ?? humanPlayer.playerNumber,
        human_player: humanPlayer.playerNumber,
        human_won: true,
        ai_difficulty: aiPlayer.aiDifficulty,
      });

      if (result.success) {
        setTrainingSubmissionState((prev) => ({
          ...prev,
          isSubmitting: false,
          wasSubmitted: true,
          error: null,
        }));
        toast.success('Game submitted for AI training!');
      } else {
        setTrainingSubmissionState((prev) => ({
          ...prev,
          isSubmitting: false,
          error: result.message || 'Submission failed',
        }));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setTrainingSubmissionState((prev) => ({
        ...prev,
        isSubmitting: false,
        error: message,
      }));
    }
  }, [sandboxGameState, sandboxVictoryResult]);

  // Get historical state when viewing history (for fixture/scenario playback)
  const historyState: GameState | null =
    isViewingHistory && sandboxEngine ? sandboxEngine.getStateAtMoveIndex(historyViewIndex) : null;

  // If history playback is enabled but the engine reports no snapshot for
  // the requested index (e.g. for imported self-play snapshots), mark
  // hasHistorySnapshots=false so the UI can disable the slider with a hint
  // instead of appearing to scrub while showing a static board.
  useEffect(() => {
    if (!isViewingHistory || !sandboxEngine) {
      return;
    }
    const stateAtIndex = sandboxEngine.getStateAtMoveIndex(historyViewIndex);
    if (!stateAtIndex && sandboxEngine.getGameState().moveHistory.length > 0) {
      setHasHistorySnapshots(false);
    }
  }, [isViewingHistory, historyViewIndex, sandboxEngine]);

  // When in replay mode, show the replay state instead of the sandbox state
  // When viewing history (for fixtures), show the historical state
  const displayGameState =
    isInReplayMode && replayState
      ? replayState
      : isViewingHistory && historyState
        ? historyState
        : sandboxGameState;
  const sandboxBoardState: BoardState | null = displayGameState?.board ?? null;

  // Move animations - auto-detects moves from game state changes
  const { pendingAnimation, clearAnimation } = useAutoMoveAnimation(sandboxGameState, {
    enabled: !effectiveReducedMotion,
  });

  const sandboxGameOverBannerText =
    sandboxVictoryResult && isSandboxVictoryModalDismissed && sandboxVictoryResult.reason
      ? getGameOverBannerText(sandboxVictoryResult.reason)
      : null;

  const boardTypeValue = sandboxBoardState?.type ?? config.boardType;
  const boardPresetInfo = BOARD_PRESETS.find((preset) => preset.value === boardTypeValue);
  const boardDisplayLabel = boardPresetInfo?.label ?? boardTypeValue;
  const boardDisplaySubtitle = boardPresetInfo?.subtitle ?? 'Custom configuration';
  const _boardDisplayBlurb =
    boardPresetInfo?.blurb ?? 'Custom layout selected for this local sandbox match.';

  // When a ring elimination decision is active in the sandbox, repurpose the
  // heuristic/status chip under the board as an explicit elimination prompt so
  // it mirrors the backend HUD directive.
  const isRingEliminationChoice =
    (sandboxCaptureChoice ?? sandboxPendingChoice)?.type === 'ring_elimination';

  // RR-DEBUG-2026-01-11: Log isRingEliminationChoice state
  React.useEffect(() => {
    if (sandboxPendingChoice?.type === 'ring_elimination' || isRingEliminationChoice) {
      // eslint-disable-next-line no-console
      console.log('[SandboxGameHost] isRingEliminationChoice debug:', {
        isRingEliminationChoice,
        sandboxPendingChoiceType: sandboxPendingChoice?.type,
        sandboxCaptureChoiceType: sandboxCaptureChoice?.type,
        activePendingChoiceType: (sandboxCaptureChoice ?? sandboxPendingChoice)?.type,
      });
    }
  }, [isRingEliminationChoice, sandboxPendingChoice, sandboxCaptureChoice]);

  // When a territory region-order decision is active, surface a territory-
  // specific prompt chip so territory-processing phases receive the same
  // high-attention treatment as line-formation eliminations.
  const isRegionOrderChoice =
    (sandboxCaptureChoice ?? sandboxPendingChoice)?.type === 'region_order';

  // RR-FIX-2026-01-12: Memoized valid moves to avoid redundant expensive calls.
  // Previously, getValidMoves() was called 11+ times per render cycle, each taking
  // 500ms-2000ms on large boards. This single memoized call provides the base for
  // all move-type filtering throughout the component.
  const allValidMoves = React.useMemo(() => {
    if (!sandboxEngine || !sandboxGameState || sandboxGameState.gameStatus !== 'active') {
      return [];
    }
    return sandboxEngine.getValidMoves(sandboxGameState.currentPlayer);
  }, [
    sandboxEngine,
    _sandboxStateVersion,
    sandboxGameState?.currentPlayer,
    sandboxGameState?.currentPhase,
    sandboxGameState?.gameStatus,
  ]);

  // Derived memoized move lists - filter once, use everywhere
  const chainContinueMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'continue_capture_segment'),
    [allValidMoves]
  );
  const captureMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'overtaking_capture'),
    [allValidMoves]
  );
  const skipCaptureMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'skip_capture'),
    [allValidMoves]
  );
  const skipRecoveryMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'skip_recovery'),
    [allValidMoves]
  );
  const eliminationMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'eliminate_rings_from_stack'),
    [allValidMoves]
  );
  const territoryOptionMoves = React.useMemo(
    () => allValidMoves.filter((m) => m.type === 'choose_territory_option'),
    [allValidMoves]
  );

  // When in chain_capture with available continuation segments, surface an
  // attention-style chip prompting the user to continue the chain. This mirrors
  // backend HUD semantics for mandatory chain continuation.
  const isChainCaptureContinuationStep = !!(
    sandboxGameState &&
    sandboxGameState.currentPhase === 'chain_capture' &&
    chainContinueMoves.length > 0
  );

  // Extract chain capture path for visualization during chain_capture phase.
  // The path includes the starting position and all landing positions visited.
  const chainCapturePath = React.useMemo(() => {
    if (!sandboxGameState || sandboxGameState.currentPhase !== 'chain_capture') {
      return undefined;
    }

    const moveHistory = sandboxGameState.moveHistory;
    if (!moveHistory || moveHistory.length === 0) {
      return undefined;
    }

    // Walk backwards from the end to find all chain capture moves by the current player
    const currentPlayer = sandboxGameState.currentPlayer;
    const path: Position[] = [];

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
  }, [sandboxGameState]);

  const sandboxPlayersList =
    sandboxGameState?.players ??
    Array.from({ length: config.numPlayers }, (_, idx) => ({
      playerNumber: idx + 1,
      username: `Player ${idx + 1}`,
      type: config.playerTypes[idx] ?? 'human',
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    }));

  const sandboxCurrentPlayerNumber = sandboxGameState?.currentPlayer ?? 1;
  const _sandboxCurrentPlayer =
    sandboxPlayersList.find((p) => p.playerNumber === sandboxCurrentPlayerNumber) ??
    sandboxPlayersList[0];

  const sandboxPhaseKey = sandboxGameState?.currentPhase ?? 'ring_placement';
  const sandboxPhaseDetails = PHASE_COPY[sandboxPhaseKey] ?? PHASE_COPY.ring_placement;

  useEffect(() => {
    if (!sandboxGameState) {
      lastSandboxPhaseRef.current = null;
      return;
    }

    const previousPhase = lastSandboxPhaseRef.current;
    const nextPhase = sandboxGameState.currentPhase;

    if (previousPhase !== nextPhase) {
      const stacksSnapshot = Array.from(sandboxGameState.board.stacks.entries()).map(
        ([key, stack]) => ({
          key,
          height: stack.stackHeight,
          cap: stack.capHeight,
          controllingPlayer: stack.controllingPlayer,
        })
      );

      // eslint-disable-next-line no-console
      console.log('[SandboxPhaseDebug][SandboxGameHost] Phase change in sandbox', {
        from: previousPhase,
        to: nextPhase,
        currentPlayer: sandboxGameState.currentPlayer,
        gameStatus: sandboxGameState.gameStatus,
        stacks: stacksSnapshot,
      });

      if (nextPhase === 'line_processing') {
        const formedLinesSnapshot =
          sandboxGameState.board.formedLines?.map((line) => ({
            player: line.player,
            length: line.length,
            positions: line.positions.map((pos) => positionToString(pos)),
          })) ?? [];

        // eslint-disable-next-line no-console
        console.log(
          '[SandboxPhaseDebug][SandboxGameHost] Entered line_processing with formedLines',
          formedLinesSnapshot
        );
      }

      // RR-FIX-2026-01-11: Clear movement selection/targets when entering decision phases.
      // Otherwise, leftover movement highlights from previous phase appear during
      // line_processing/territory_processing, confusing the user.
      const isDecisionPhase =
        nextPhase === 'line_processing' || nextPhase === 'territory_processing';
      if (isDecisionPhase) {
        setSelected(undefined);
        setValidTargets([]);

        // RR-FIX-2026-01-11: Compute and display elimination targets for decision phases.
        // After clearing old movement highlights, check if there are elimination moves
        // available and highlight their target positions so the player knows where to click.
        // RR-FIX-2026-01-12: Use memoized eliminationMoves instead of calling getValidMoves.
        if (eliminationMoves.length > 0) {
          const eliminationTargets = eliminationMoves
            .map((m) => m.to)
            .filter((pos): pos is Position => pos !== undefined);

          if (eliminationTargets.length > 0) {
            // eslint-disable-next-line no-console
            console.log('[SandboxPhaseDebug] Setting elimination targets:', {
              phase: nextPhase,
              targetCount: eliminationTargets.length,
              targets: eliminationTargets.map((p) => positionToString(p)),
            });
            setValidTargets(eliminationTargets);
          }
        }
      }

      lastSandboxPhaseRef.current = nextPhase;
    }
  }, [sandboxGameState, sandboxEngine, setSelected, setValidTargets, eliminationMoves]);

  // Initialize chain capture valid targets when loading a game directly into chain_capture phase.
  // The normal interaction handlers only set valid targets when transitioning INTO chain_capture
  // via a move, but when loading a fixture/scenario that starts in chain_capture, targets need
  // to be initialized explicitly.
  // RR-FIX-2026-01-12: Use ref tracking instead of validTargets.length to prevent effect cascade.
  useEffect(() => {
    if (
      !sandboxEngine ||
      !sandboxGameState ||
      sandboxGameState.gameStatus !== 'active' ||
      sandboxGameState.currentPhase !== 'chain_capture'
    ) {
      // Reset ref when not in chain_capture phase
      lastChainCaptureInitRef.current = null;
      return;
    }

    // Only initialize if validTargets is empty (user hasn't selected anything yet)
    if (validTargets.length > 0) {
      return;
    }

    const ctx = sandboxEngine.getChainCaptureContextForCurrentPlayer();
    if (!ctx || ctx.landings.length === 0) {
      return;
    }

    // Create a unique key for this chain capture context to prevent re-initialization
    const ctxKey = `${ctx.from.x}-${ctx.from.y}-${ctx.landings.length}-${sandboxGameState.moveHistory.length}`;
    if (lastChainCaptureInitRef.current === ctxKey) {
      return; // Already initialized for this context
    }

    lastChainCaptureInitRef.current = ctxKey;
    setSelected(ctx.from);
    setValidTargets(ctx.landings);
  }, [sandboxEngine, sandboxGameState, setSelected, setValidTargets, validTargets.length]);

  // RR-FIX-2026-01-11: Derive ring_elimination choice from game state when needed.
  // This handles the case where a game state is loaded (fixture, replay, etc.) and the
  // UI needs to show elimination highlights without the engine having actively called
  // requestChoice(). When:
  // 1. We're in territory_processing phase
  // 2. The last move was choose_territory_option (mandatory elimination pending)
  // 3. No sandboxPendingChoice is currently set
  // Then we derive and set a RingEliminationChoice to trigger the visual highlights.
  useEffect(() => {
    // Only derive if we don't already have a pending choice
    if (sandboxPendingChoice) {
      return;
    }

    if (
      !sandboxGameState ||
      sandboxGameState.gameStatus !== 'active' ||
      sandboxGameState.currentPhase !== 'territory_processing'
    ) {
      return;
    }

    // Check if the last move was choose_territory_option (indicating pending elimination)
    const moveHistory = sandboxGameState.moveHistory;
    if (moveHistory.length === 0) {
      return;
    }

    const lastMove = moveHistory[moveHistory.length - 1];
    if (lastMove.type !== 'choose_territory_option') {
      return;
    }

    // Verify the last move was by the current player
    if (lastMove.player !== sandboxGameState.currentPlayer) {
      return;
    }

    // Derive elimination moves from the game state
    const currentPlayer = sandboxGameState.currentPlayer;
    const eliminationMoves = enumerateTerritoryEliminationMoves(sandboxGameState, currentPlayer);

    if (eliminationMoves.length === 0) {
      return;
    }

    // Determine elimination context (territory vs recovery)
    let territoryEliminationContext: 'territory' | 'recovery' = 'territory';
    for (let i = moveHistory.length - 1; i >= 0; i--) {
      const move = moveHistory[i];
      if (move.player !== currentPlayer) {
        break;
      }
      if (move.type === 'recovery_slide') {
        territoryEliminationContext = 'recovery';
        break;
      }
    }

    // Build descriptive prompt from the processed region
    const regionSpaces = lastMove.disconnectedRegions?.[0]?.spaces ?? [];
    const spacesList = regionSpaces
      .slice(0, 4)
      .map((p) => `(${p.x},${p.y})`)
      .join(', ');
    const truncated = regionSpaces.length > 4 ? ` +${regionSpaces.length - 4} more` : '';
    const territoryPrompt =
      territoryEliminationContext === 'recovery'
        ? `Territory claimed at ${spacesList}${truncated}. You must extract ONE buried ring from a stack outside the region.`
        : `Territory claimed at ${spacesList}${truncated}. You must eliminate your ENTIRE CAP from an eligible stack outside the region.`;

    // Build the RingEliminationChoice
    const choice: RingEliminationChoice = {
      id: `derived-territory-elim-${Date.now()}`,
      gameId: sandboxGameState.id,
      playerNumber: currentPlayer,
      type: 'ring_elimination',
      eliminationContext: territoryEliminationContext,
      prompt: territoryPrompt,
      options: eliminationMoves.map((opt: Move) => {
        const pos = opt.to as Position;
        const key = positionToString(pos);
        const stack = sandboxGameState.board.stacks.get(key);

        const capHeight =
          (opt.eliminationFromStack && opt.eliminationFromStack.capHeight) ||
          (stack ? stack.capHeight : 1);
        const totalHeight =
          (opt.eliminationFromStack && opt.eliminationFromStack.totalHeight) ||
          (stack ? stack.stackHeight : capHeight || 1);

        const ringsToEliminate =
          typeof opt.eliminatedRings?.[0]?.count === 'number'
            ? opt.eliminatedRings[0].count
            : capHeight;

        return {
          stackPosition: pos,
          capHeight,
          totalHeight,
          ringsToEliminate,
          moveId: opt.id || key,
        };
      }),
    };

    // eslint-disable-next-line no-console
    console.log('[SandboxGameHost] Derived ring_elimination choice from game state:', {
      phase: sandboxGameState.currentPhase,
      lastMoveType: lastMove.type,
      choiceId: choice.id,
      optionsCount: choice.options.length,
      options: choice.options.map((opt) => ({
        stackPosition: opt.stackPosition,
        capHeight: opt.capHeight,
        ringsToEliminate: opt.ringsToEliminate,
      })),
    });

    setSandboxPendingChoice(choice);
  }, [sandboxGameState, sandboxPendingChoice, setSandboxPendingChoice]);

  // RR-FIX-2026-01-11: Initialize targets when in decision phases.
  // This handles:
  // 1. Loading a fixture directly into an elimination state
  // 2. When a player makes a territory_option choice (staying in territory_processing)
  //    that triggers elimination moves
  // 3. When there are choose_territory_option moves - highlight the claimable territory spaces
  // 4. RR-FIX-2026-01-11: When a ring_elimination pending choice is set, override
  //    any stale targets (e.g., leftover territory claim targets) with elimination targets
  // RR-FIX-2026-01-12: Use memoized moves instead of calling getValidMoves() in effect.
  useEffect(() => {
    if (
      !sandboxGameState ||
      sandboxGameState.gameStatus !== 'active' ||
      (sandboxGameState.currentPhase !== 'territory_processing' &&
        sandboxGameState.currentPhase !== 'line_processing')
    ) {
      return;
    }

    const currentPlayer = sandboxGameState.currentPlayer;

    // RR-FIX-2026-01-11: When a ring_elimination pending choice is active,
    // force update targets from the choice options. This handles the case where
    // territory claim targets are set, then the player claims a territory, and
    // a ring_elimination choice is returned but the old targets remain.
    // RR-FIX-2026-01-12: Only process each pending choice once to prevent infinite re-render.
    if (sandboxPendingChoice?.type === 'ring_elimination') {
      // Skip if we've already processed this pending choice
      if (lastProcessedPendingChoiceIdRef.current === sandboxPendingChoice.id) {
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing choice options
      const options = (sandboxPendingChoice as any).options ?? [];
      const eliminationTargets: Position[] = options
        .map((opt: { stackPosition?: Position }) => opt.stackPosition)
        .filter((pos: Position | undefined): pos is Position => pos !== undefined);

      if (eliminationTargets.length > 0) {
        // eslint-disable-next-line no-console
        console.log('[SandboxGameHost] Setting elimination targets from pending choice:', {
          phase: sandboxGameState.currentPhase,
          choiceId: sandboxPendingChoice.id,
          targetCount: eliminationTargets.length,
          targets: eliminationTargets.map((p) => positionToString(p)),
        });
        lastProcessedPendingChoiceIdRef.current = sandboxPendingChoice.id;
        setValidTargets(eliminationTargets);
        return;
      }
    } else if (sandboxPendingChoice?.type === 'region_order') {
      // RR-FIX-2026-01-18: When a region_order pending choice is active,
      // force update targets from the choice options. This handles the case where
      // a player has multiple territories to process - after processing the first one,
      // a new region_order choice is returned but the old targets remain.
      // Skip if we've already processed this pending choice
      if (lastProcessedPendingChoiceIdRef.current === sandboxPendingChoice.id) {
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing choice options
      const options = (sandboxPendingChoice as any).options ?? [];
      const territorySpaces: Position[] = [];

      for (const opt of options) {
        // Each option has a representativePosition and optionally spaces array
        const spaces = opt.spaces as Position[] | undefined;
        if (spaces && spaces.length > 0) {
          territorySpaces.push(...spaces);
        } else if (opt.representativePosition) {
          // Fallback to representative position if spaces not available
          territorySpaces.push(opt.representativePosition);
        }
      }

      if (territorySpaces.length > 0) {
        // eslint-disable-next-line no-console
        console.log('[SandboxGameHost] Setting territory targets from pending choice:', {
          phase: sandboxGameState.currentPhase,
          choiceId: sandboxPendingChoice.id,
          targetCount: territorySpaces.length,
          targets: territorySpaces.map((p) => positionToString(p)),
        });
        lastProcessedPendingChoiceIdRef.current = sandboxPendingChoice.id;
        setValidTargets(territorySpaces);
        return;
      }
    } else {
      // Reset ref when there's no ring_elimination or region_order choice active
      lastProcessedPendingChoiceIdRef.current = null;
    }

    // Only proceed with automatic target initialization if validTargets is empty
    if (validTargets.length > 0) {
      return;
    }

    // Check for elimination moves first (using memoized eliminationMoves)
    if (eliminationMoves.length > 0) {
      const eliminationTargets = eliminationMoves
        .map((m) => m.to)
        .filter((pos): pos is Position => pos !== undefined);

      if (eliminationTargets.length > 0) {
        // eslint-disable-next-line no-console
        console.log('[SandboxGameHost] Initializing elimination targets:', {
          phase: sandboxGameState.currentPhase,
          targetCount: eliminationTargets.length,
          targets: eliminationTargets.map((p) => positionToString(p)),
        });
        setValidTargets(eliminationTargets);
        return;
      }
    }

    // Check for choose_territory_option moves - highlight claimable territory spaces (using memoized)
    if (territoryOptionMoves.length > 0) {
      // Collect all spaces from claimable territories owned by current player
      const territories = sandboxGameState.board.territories;
      const claimableSpaces: Position[] = [];

      for (const [, territory] of territories.entries()) {
        // Only show territories owned by the current player
        if (territory.controllingPlayer !== currentPlayer) continue;
        if (!territory.isDisconnected) continue;

        const spaces = territory.spaces ?? [];
        claimableSpaces.push(...spaces);
      }

      if (claimableSpaces.length > 0) {
        // eslint-disable-next-line no-console
        console.log('[SandboxGameHost] Initializing territory claim targets:', {
          phase: sandboxGameState.currentPhase,
          targetCount: claimableSpaces.length,
          territories: Array.from(territories.keys()),
        });
        setValidTargets(claimableSpaces);
      }
    }
  }, [
    sandboxGameState,
    sandboxPendingChoice,
    validTargets.length,
    setValidTargets,
    eliminationMoves,
    territoryOptionMoves,
  ]);

  const humanSeatCount = sandboxPlayersList.filter((p) => p.type === 'human').length;
  const aiSeatCount = sandboxPlayersList.length - humanSeatCount;

  // AI tracking effects (thinking state, auto-trigger) are now managed by useSandboxAITracking hook

  // Clock state and effects are now managed by useSandboxClock hook
  // (initialization, turn change handling, and decrement interval)

  // Derive board VM + HUD-like summaries
  const primaryValidTargets =
    sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets;

  const displayedValidTargets = overlays.showValidTargets ? primaryValidTargets : [];

  // Derive decision-phase highlights from the current sandbox GameState and
  // whichever PlayerChoice is currently active. Capture-direction choices
  // take precedence over generic pending choices so that landing/target
  // geometry is always visible while the capture UI is open.
  const activePendingChoice: PlayerChoice | null = sandboxCaptureChoice ?? sandboxPendingChoice;

  // For sandbox hosts, surface a simple decision countdown when the underlying
  // PlayerChoice exposes a timeoutMs. This keeps HUD time-pressure semantics
  // aligned with backend games without requiring server-side timeout warnings.
  const sandboxDecisionTimeRemainingMs = React.useMemo(() => {
    if (!activePendingChoice) return null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing optional timeoutMs on polymorphic PlayerChoice
    const rawTimeout = (activePendingChoice as any).timeoutMs as number | undefined;
    if (typeof rawTimeout !== 'number' || Number.isNaN(rawTimeout)) {
      return null;
    }
    return rawTimeout >= 0 ? rawTimeout : 0;
  }, [activePendingChoice]);

  const baseDecisionHighlights =
    sandboxGameState && activePendingChoice
      ? deriveBoardDecisionHighlights(sandboxGameState, activePendingChoice)
      : undefined;

  // RR-DEBUG-2026-01-10: Log decision highlights for territory elimination debugging
  React.useEffect(() => {
    if (activePendingChoice?.type === 'ring_elimination') {
      // eslint-disable-next-line no-console
      console.log('[SandboxGameHost] Ring elimination decision state:', {
        hasGameState: !!sandboxGameState,
        activePendingChoiceType: activePendingChoice?.type,
        activePendingChoiceId: activePendingChoice?.id,
        optionsCount:
          activePendingChoice &&
          'options' in activePendingChoice &&
          Array.isArray(activePendingChoice.options)
            ? activePendingChoice.options.length
            : 0,
        baseDecisionHighlightsChoiceKind: baseDecisionHighlights?.choiceKind,
        baseDecisionHighlightsCount: baseDecisionHighlights?.highlights?.length ?? 0,
        highlightPositions: baseDecisionHighlights?.highlights?.map((h) => h.positionKey),
      });
    }
  }, [activePendingChoice, sandboxGameState, baseDecisionHighlights]);

  // Merge transient line highlights into the decision highlight model so
  // recently-collapsed lines receive a brief visual cue even when no
  // explicit line-order/reward choice is surfaced.
  let decisionHighlights = baseDecisionHighlights;
  if (recentLineHighlights.length > 0) {
    const recentKeys = new Set(recentLineHighlights.map((pos) => positionToString(pos)));
    const existing = baseDecisionHighlights?.highlights ?? [];

    const extraHighlights = Array.from(recentKeys)
      .filter((key) => !existing.some((h) => h.positionKey === key))
      .map((key) => ({
        positionKey: key,
        intensity: 'primary' as const,
      }));

    if (extraHighlights.length > 0) {
      decisionHighlights = {
        choiceKind: baseDecisionHighlights?.choiceKind ?? 'line_order',
        highlights: [...existing, ...extraHighlights],
      };
    }
  }

  // Optional-capture visibility: when the sandbox is in capture phase with no
  // explicit PlayerChoice but canonical overtaking_capture moves exist from the
  // current player's stacks, surface stronger capture-direction highlights so
  // both landing cells and overtaken stacks pulse clearly.
  if (
    !decisionHighlights &&
    sandboxGameState &&
    sandboxGameState.gameStatus === 'active' &&
    sandboxGameState.currentPhase === 'capture'
  ) {
    // RR-FIX-2026-01-12: Use memoized captureMoves instead of calling getValidMoves
    if (captureMoves.length > 0) {
      type CaptureHighlight = { positionKey: string; intensity: 'primary' | 'secondary' };
      const highlights: CaptureHighlight[] = [];
      const seenPrimary = new Set<string>();
      const seenAny = new Set<string>();

      for (const move of captureMoves) {
        const target = move.captureTarget as Position | undefined;
        const landing = move.to as Position | undefined;

        if (landing) {
          const key = positionToString(landing);
          if (!seenPrimary.has(key)) {
            seenPrimary.add(key);
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'primary' });
          }
        }

        if (target) {
          const key = positionToString(target);
          if (!seenAny.has(key)) {
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'secondary' });
          }
        }
      }

      if (highlights.length > 0) {
        decisionHighlights = {
          choiceKind: 'capture_direction',
          highlights,
        };
      }
    }
  }

  // Chain-capture visibility: when the sandbox is in chain_capture phase,
  // surface similar capture-direction highlights so landing cells and
  // overtaken stacks are visible.
  // RR-FIX-2026-01-12: Use memoized chainContinueMoves instead of calling getValidMoves
  if (
    !decisionHighlights &&
    sandboxGameState &&
    sandboxGameState.gameStatus === 'active' &&
    sandboxGameState.currentPhase === 'chain_capture'
  ) {
    if (chainContinueMoves.length > 0) {
      type CaptureHighlight = { positionKey: string; intensity: 'primary' | 'secondary' };
      const highlights: CaptureHighlight[] = [];
      const seenPrimary = new Set<string>();
      const seenAny = new Set<string>();

      for (const move of chainContinueMoves) {
        const target = move.captureTarget as Position | undefined;
        const landing = move.to as Position | undefined;

        if (landing) {
          const key = positionToString(landing);
          if (!seenPrimary.has(key)) {
            seenPrimary.add(key);
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'primary' });
          }
        }

        if (target) {
          const key = positionToString(target);
          if (!seenAny.has(key)) {
            seenAny.add(key);
            highlights.push({ positionKey: key, intensity: 'secondary' });
          }
        }
      }

      if (highlights.length > 0) {
        decisionHighlights = {
          choiceKind: 'capture_direction',
          highlights,
        };
      }
    }
  }

  const sandboxBoardViewModel = sandboxBoardState
    ? toBoardViewModel(sandboxBoardState, {
        selectedPosition: selected,
        validTargets: displayedValidTargets,
        decisionHighlights,
        colorVisionMode,
      })
    : null;

  useEffect(() => {
    if (!sandboxBoardState || !sandboxBoardViewModel) {
      return;
    }

    if (sandboxPhaseKey !== 'line_processing') {
      return;
    }

    const stacksSnapshot = Array.from(sandboxBoardState.stacks.entries()).map(([key, stack]) => ({
      key,
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    }));

    const decisionHighlightsSnapshot =
      decisionHighlights?.highlights?.map((h) => h.positionKey) ?? [];

    // eslint-disable-next-line no-console
    console.log('[SandboxPhaseDebug][SandboxGameHost] BoardView props in line_processing', {
      boardType: sandboxBoardState.type,
      phase: sandboxPhaseKey,
      stacks: stacksSnapshot,
      selectedPosition: selected ? positionToString(selected) : null,
      validTargets: displayedValidTargets.map((pos) => positionToString(pos)),
      decisionHighlights: decisionHighlightsSnapshot,
    });
  }, [
    sandboxBoardState,
    sandboxBoardViewModel,
    sandboxPhaseKey,
    selected,
    displayedValidTargets,
    decisionHighlights,
  ]);

  const sandboxVictoryViewModel = sandboxVictoryResult
    ? toVictoryViewModel(
        sandboxVictoryResult,
        sandboxGameState?.players ?? [],
        sandboxGameState ?? undefined,
        {
          currentUserId: user?.id,
          isDismissed: isSandboxVictoryModalDismissed,
          colorVisionMode,
          gameEndExplanation: sandboxGameEndExplanation,
        }
      )
    : null;

  // Create unified HUD view model using toHUDViewModel (matches backend host pattern)
  const sandboxLpsTracking = sandboxEngine?.getLpsTrackingState();
  let sandboxHudVM = sandboxGameState
    ? toHUDViewModel(sandboxGameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        currentUserId: user?.id,
        colorVisionMode,
        pendingChoice: activePendingChoice,
        choiceDeadline: null,
        choiceTimeRemainingMs: sandboxDecisionTimeRemainingMs,
        victoryState: sandboxVictoryResult,
        gameEndExplanation: sandboxGameEndExplanation,
        lpsTracking: sandboxLpsTracking,
      })
    : null;

  // Override player timeRemaining with sandbox clock times if enabled
  if (sandboxHudVM && sandboxClockEnabled) {
    sandboxHudVM = {
      ...sandboxHudVM,
      players: sandboxHudVM.players.map((p) => ({
        ...p,
        timeRemaining: sandboxPlayerTimes[p.playerNumber] ?? sandboxTimeControl.initialTimeMs,
      })),
    };
  }

  // Create TimeControl object for HUD when clock is enabled
  const sandboxTimeControlForHud: TimeControl | undefined = sandboxClockEnabled
    ? {
        initialTime: Math.round(sandboxTimeControl.initialTimeMs / 1000),
        increment: Math.round(sandboxTimeControl.incrementMs / 1000),
        type: 'rapid' as const,
      }
    : undefined;

  // Optional-capture HUD chip: when capture is available directly from the
  // capture phase (with skip_capture as an option) but no explicit decision
  // choice is active, surface a bright attention chip so players do not miss
  // the opportunity.
  // RR-FIX-2026-01-12: Use memoized captureMoves/skipCaptureMoves instead of getValidMoves
  if (
    sandboxHudVM &&
    !sandboxHudVM.decisionPhase &&
    sandboxGameState &&
    sandboxGameState.gameStatus === 'active' &&
    sandboxGameState.currentPhase === 'capture'
  ) {
    const hasCaptureMove = captureMoves.length > 0;
    const hasSkipCaptureMove = skipCaptureMoves.length > 0;

    if (hasCaptureMove) {
      const actingPlayer =
        sandboxGameState.players.find((p) => p.playerNumber === sandboxGameState.currentPlayer) ??
        sandboxGameState.players[0];
      const actingPlayerName = actingPlayer.username || `Player ${actingPlayer.playerNumber}`;

      sandboxHudVM = {
        ...sandboxHudVM,
        decisionPhase: {
          isActive: true,
          actingPlayerNumber: actingPlayer.playerNumber,
          actingPlayerName,
          isLocalActor: true,
          label: hasSkipCaptureMove
            ? 'Optional capture available'
            : 'Capture available from this stack',
          description: hasSkipCaptureMove
            ? 'You may jump over a neighbouring stack for an overtaking capture, or skip capture to continue this turn.'
            : 'You may jump over a neighbouring stack for an overtaking capture from your last move.',
          shortLabel: 'Capture opportunity',
          timeRemainingMs: null,
          showCountdown: false,
          warningThresholdMs: undefined,
          isServerCapped: undefined,
          spectatorLabel: hasSkipCaptureMove
            ? `${actingPlayerName} may choose an overtaking capture or skip.`
            : `${actingPlayerName} may choose an overtaking capture.`,
          statusChip: {
            text: hasSkipCaptureMove
              ? 'Capture available – tap a landing or skip'
              : 'Capture available – tap a landing',
            tone: 'attention',
          },
          canSkip: hasSkipCaptureMove,
        },
      };
    }
  }

  // Short, phase-specific hint for touch controls, derived from the HUD
  // decision view model where applicable. This keeps SandboxTouchControlsPanel
  // rules-agnostic while still surfacing decision context for touch users.
  const isDecisionPhaseForTouchHint =
    sandboxPhaseKey === 'line_processing' ||
    sandboxPhaseKey === 'territory_processing' ||
    sandboxPhaseKey === 'chain_capture';

  const sandboxTouchPhaseHint =
    isDecisionPhaseForTouchHint && sandboxHudVM?.decisionPhase
      ? (sandboxHudVM.decisionPhase.statusChip?.text ?? sandboxHudVM.decisionPhase.shortLabel)
      : undefined;
  const canSkipTerritoryProcessing =
    !!sandboxHudVM?.decisionPhase?.canSkip &&
    !!sandboxPendingChoice &&
    sandboxPendingChoice.type === 'region_order';

  const sandboxSquareRankFromBottom = boardTypeValue === 'square8' || boardTypeValue === 'square19';
  const sandboxEventLogViewModel = toEventLogViewModel(
    sandboxGameState?.history ?? [],
    [],
    sandboxVictoryResult,
    { maxEntries: 40, boardType: boardTypeValue, squareRankFromBottom: sandboxSquareRankFromBottom }
  );

  const selectedStackDetails = (() => {
    if (!sandboxBoardState || !selected) return null;
    const key = positionToString(selected);
    const stack = sandboxBoardState.stacks.get(key);
    if (!stack) return null;
    return {
      height: stack.stackHeight,
      cap: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  })();

  // RR-FIX-2026-01-12: Use memoized skipCaptureMoves/skipRecoveryMoves instead of getValidMoves
  const canSkipCaptureForTouch =
    !!sandboxGameState &&
    sandboxGameState.currentPhase === 'capture' &&
    skipCaptureMoves.length > 0;

  const canSkipRecoveryForTouch =
    !!sandboxGameState &&
    sandboxGameState.currentPhase === 'movement' &&
    skipRecoveryMoves.length > 0;

  // Most global keyboard shortcuts are handled via useGlobalGameShortcuts.
  // Keep a narrow Escape handler here so the overlay closes even if it is
  // mocked in tests or rendered outside the focus trap.
  useEffect(() => {
    if (!isConfigured || !sandboxEngine) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;

      const target = event.target as HTMLElement | null;
      if (target) {
        const tagName = target.tagName;
        const isEditableTag = tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT';
        const isContentEditable = target.isContentEditable;
        if (isEditableTag || isContentEditable) {
          return;
        }
      }

      if (event.key === 'Escape' && showBoardControls) {
        event.preventDefault();
        setShowBoardControls(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isConfigured, sandboxEngine, showBoardControls]);

  // Note: Evaluation state/effects now handled by useSandboxEvaluation hook
  // Note: Persistence state/sync now handled by useSandboxPersistence hook

  // Pre-game setup view
  if (!isConfigured || !sandboxEngine) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <SandboxGameConfig
          config={config}
          onConfigChange={handleSetupChange}
          onPlayerTypeChange={handlePlayerTypeChange}
          onAIDifficultyChange={handleAIDifficultyChange}
          clockEnabled={sandboxClockEnabled}
          onClockEnabledChange={setSandboxClockEnabled}
          timeControl={sandboxTimeControl}
          onResetPlayerTimes={resetSandboxPlayerTimes}
          onStartGame={handleStartLocalGame}
          onQuickStartPreset={handleQuickStartPreset}
          onShowScenarioPicker={() => setShowScenarioPicker(true)}
          onShowSelfPlayBrowser={() => setShowSelfPlayBrowser(true)}
          isBeginnerMode={isBeginnerMode}
          onModeChange={setSandboxMode}
          developerToolsEnabled={developerToolsEnabled}
          isFirstTimePlayer={isFirstTimePlayer}
          isLoggedIn={!!user}
          backendSandboxError={backendSandboxError}
        />
        <ScenarioPickerModal
          isOpen={showScenarioPicker}
          onClose={() => setShowScenarioPicker(false)}
          onSelectScenario={handleLoadScenario}
          developerToolsEnabled={developerToolsEnabled}
        />

        {developerToolsEnabled && (
          <SelfPlayBrowser
            isOpen={showSelfPlayBrowser}
            onClose={() => setShowSelfPlayBrowser(false)}
            onSelectGame={handleLoadScenario}
          />
        )}

        {/* First-time player onboarding modal */}
        <OnboardingModal
          isOpen={shouldShowWelcome}
          onClose={markWelcomeSeen}
          onStartTutorial={handleStartTutorial}
        />
      </div>
    );
  }

  // === Active sandbox game ===
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="container mx-auto px-2 sm:px-4 py-2 sm:py-3 space-y-2 sm:space-y-2">
        {/* Screen reader live region for game announcements (accessibility parity with BackendGameHost) */}
        <ScreenReaderAnnouncer
          queue={announcementQueue}
          onAnnouncementSpoken={removeAnnouncement}
        />

        {sandboxStallWarning && (
          <StatusBanner
            variant="warning"
            title="Sandbox warning"
            actions={
              <>
                {developerToolsEnabled && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={diagnosticsActions.copyAiTrace}
                    className="text-[11px] px-3 py-1"
                  >
                    Copy AI trace
                  </Button>
                )}
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  onClick={() => setSandboxStallWarning(null)}
                  className="text-[11px] px-3 py-1"
                >
                  Dismiss
                </Button>
              </>
            }
          >
            <span className="text-xs">{sandboxStallWarning}</span>
          </StatusBanner>
        )}

        {/* AI service fallback warning - shown when using local heuristics instead of service */}
        <AIServiceStatusBanner
          status={aiServiceState.status}
          message={aiServiceState.message}
          isServiceConfigured={aiServiceState.isServiceConfigured}
          onRetry={aiServiceActions.retryConnection}
          onDismiss={aiServiceActions.dismissMessage}
        />

        {sandboxGameOverBannerText && (
          <StatusBanner variant="success" title="Game over">
            <span className="text-xs">{sandboxGameOverBannerText}</span>
          </StatusBanner>
        )}

        {sandboxGameState && (
          <VictoryModal
            isOpen={!!sandboxVictoryResult && !isSandboxVictoryModalDismissed}
            viewModel={sandboxVictoryViewModel}
            gameEndExplanation={sandboxGameEndExplanation}
            isSandbox
            onClose={() => {
              setIsSandboxVictoryModalDismissed(true);
            }}
            onReturnToLobby={lifecycleActions.resetToSetup}
            onRematch={lifecycleActions.rematch}
            // January 2026 - Human game training
            onSubmitForTraining={handleSubmitForTraining}
            trainingSubmission={trainingSubmissionState}
          />
        )}

        {/* RR-FIX-2026-01-12: Render LineRewardPanel for overlength line choices with segments */}
        {sandboxPendingChoice &&
          sandboxPendingChoice.type === 'line_reward_option' &&
          sandboxPendingChoice.segments &&
          sandboxPendingChoice.segments.length > 0 && (
            <LineRewardPanel
              choice={sandboxPendingChoice}
              onSelect={(optionId) => {
                const resolver = sandboxChoiceResolverRef.current;
                if (resolver) {
                  resolver({
                    choiceId: sandboxPendingChoice.id,
                    playerNumber: sandboxPendingChoice.playerNumber,
                    choiceType: sandboxPendingChoice.type,
                    selectedOption: { optionId },
                  } as PlayerChoiceResponseFor<PlayerChoice>);
                  sandboxChoiceResolverRef.current = null;
                }
                setSandboxPendingChoice(null);
                setSandboxStateVersion((v) => v + 1);
              }}
            />
          )}

        <ChoiceDialog
          choice={
            sandboxPendingChoice &&
            sandboxPendingChoice.type !== 'ring_elimination' &&
            sandboxPendingChoice.type !== 'region_order' &&
            // Don't show ChoiceDialog for line_reward_option with segments (use LineRewardPanel instead)
            !(
              sandboxPendingChoice.type === 'line_reward_option' &&
              sandboxPendingChoice.segments &&
              sandboxPendingChoice.segments.length > 0
            )
              ? sandboxPendingChoice
              : null
          }
          deadline={null}
          onSelectOption={(choice, option) => {
            const resolver = sandboxChoiceResolverRef.current;
            if (resolver) {
              resolver({
                choiceId: choice.id,
                playerNumber: choice.playerNumber,
                choiceType: choice.type,
                selectedOption: option,
              } as PlayerChoiceResponseFor<PlayerChoice>);
              sandboxChoiceResolverRef.current = null;
            }
            setSandboxPendingChoice(null);
            // Bump sandbox state version so the AI turn loop
            // and any derived view models observe the post-
            // decision state (including advancing to an AI
            // turn after line/territory decisions).
            setSandboxStateVersion((v) => v + 1);
          }}
        />

        <main className="flex flex-col lg:flex-row lg:gap-6 gap-4">
          {/* Tutorial Hint Banner - shown in "Learn the Basics" mode */}
          {/* RR-FIX-2026-01-14: Restructured to prevent overlap with board */}
          {currentHint && isLearnBasicsMode && (
            <aside className="flex-shrink-0 w-full lg:w-72 xl:w-80">
              <TutorialHintBanner
                hint={currentHint}
                onDismiss={() => {
                  markPhaseHintSeen(currentHint.phase);
                  dismissHint();
                }}
                onLearnMore={() => {
                  const topic = getTeachingTopic();
                  if (topic) {
                    setTeachingTopic(topic);
                    setShowTeachingOverlay(true);
                  }
                }}
                onDisableHints={() => {
                  setTutorialHintsEnabled(false);
                  dismissHint();
                }}
              />
            </aside>
          )}

          {/* Board Section - extracted component */}
          {sandboxBoardState && sandboxBoardViewModel && sandboxGameState && (
            <SandboxBoardSection
              boardState={sandboxBoardState}
              boardViewModel={sandboxBoardViewModel}
              gameState={sandboxGameState}
              selectedPosition={selected}
              validTargets={displayedValidTargets}
              isInReplayMode={isInReplayMode}
              pendingAnimation={pendingAnimation}
              replayAnimation={replayAnimation}
              chainCapturePath={chainCapturePath}
              shakingCellKey={sandboxShakingCellKey}
              overlays={{
                showMovementGrid: overlays.showMovementGrid,
                showLineOverlays: overlays.showLineOverlays,
                showTerritoryOverlays: overlays.showTerritoryOverlays,
              }}
              boardDisplayLabel={boardDisplayLabel}
              boardDisplaySubtitle={boardDisplaySubtitle}
              config={config}
              playersList={sandboxPlayersList}
              currentPlayerNumber={sandboxCurrentPlayerNumber}
              phaseDetails={sandboxPhaseDetails}
              humanSeatCount={humanSeatCount}
              aiSeatCount={aiSeatCount}
              isBeginnerMode={isBeginnerMode}
              developerToolsEnabled={developerToolsEnabled}
              lastLoadedScenario={lastLoadedScenario}
              gameEndExplanation={sandboxGameEndExplanation}
              isRingEliminationChoice={isRingEliminationChoice}
              isRegionOrderChoice={isRegionOrderChoice}
              isChainCaptureContinuationStep={isChainCaptureContinuationStep}
              decisionHighlights={decisionHighlights}
              onCellClick={handleSandboxCellClick}
              onCellDoubleClick={handleSandboxCellDoubleClick}
              onCellContextMenu={handleSandboxCellContextMenu}
              onCellLongPress={handleSandboxCellContextMenu}
              onAnimationComplete={clearAnimation}
              onReplayAnimationComplete={() => setReplayAnimation(null)}
              onShowBoardControls={() => setShowBoardControls(true)}
              onSaveState={() => diagnosticsActions.setShowSaveStateDialog(true)}
              onExportScenario={diagnosticsActions.exportScenarioJson}
              onCopyFixture={diagnosticsActions.copyTestFixture}
              onLoadScenario={() => setShowScenarioPicker(true)}
              onResetScenario={handleResetScenario}
              onChangeSetup={() => {
                resetSandboxEngine();
                setSelected(undefined);
                setValidTargets([]);
                setBackendSandboxError(null);
                setSandboxPendingChoice(null);
                setSandboxStallWarning(null);
                setSandboxLastProgressAt(null);
                setIsSandboxVictoryModalDismissed(false);
              }}
              onModeChange={setSandboxMode}
            />
          )}

          {/* Sidebar - extracted component */}
          <SandboxGameSidebar
            hudViewModel={sandboxHudVM}
            gameState={sandboxGameState}
            config={config}
            timeControl={sandboxTimeControlForHud}
            boardState={
              sandboxBoardState
                ? { type: sandboxBoardState.type, size: sandboxBoardState.size }
                : null
            }
            isMobile={isMobile}
            isLoggedIn={!!user}
            isBeginnerMode={isBeginnerMode}
            developerToolsEnabled={developerToolsEnabled}
            lastLoadedScenario={lastLoadedScenario}
            boardTypeValue={boardTypeValue}
            numPlayers={sandboxPlayersList.length}
            selectedPosition={selected}
            selectedStackDetails={selectedStackDetails}
            validTargets={validTargets}
            primaryValidTargets={primaryValidTargets}
            isCaptureDirectionPending={!!sandboxCaptureChoice}
            captureTargets={sandboxCaptureTargets}
            overlays={overlays}
            phaseLabel={sandboxPhaseDetails.label}
            phaseHint={sandboxTouchPhaseHint}
            canSkipCapture={canSkipCaptureForTouch}
            canSkipTerritoryProcessing={canSkipTerritoryProcessing}
            canSkipRecovery={canSkipRecoveryForTouch}
            isRingEliminationChoice={isRingEliminationChoice}
            isInReplayMode={isInReplayMode}
            isViewingHistory={isViewingHistory}
            historyViewIndex={historyViewIndex}
            hasHistorySnapshots={hasHistorySnapshots}
            showAdvancedSidebarPanels={showAdvancedSidebarPanels}
            autoSaveGames={autoSaveGames}
            gameSaveStatus={gameSaveStatus}
            pendingLocalGames={pendingLocalGames}
            syncState={syncState}
            aiThinkingStartedAt={aiTrackingState.aiThinkingStartedAt}
            aiLadderHealth={aiTrackingState.aiLadderHealth}
            aiLadderHealthError={aiTrackingState.aiLadderHealthError}
            aiLadderHealthLoading={aiTrackingState.aiLadderHealthLoading}
            evaluationHistory={sandboxEvaluationHistory}
            evaluationError={sandboxEvaluationError}
            isEvaluating={isSandboxAnalysisRunning}
            eventLogViewModel={sandboxEventLogViewModel}
            sandboxEngine={sandboxEngine}
            onSwapSides={() => {
              if (sandboxEngine) {
                sandboxEngine.applySwapSidesForCurrentPlayer();
                setSelected(undefined);
                setValidTargets([]);
                setSandboxPendingChoice(null);
                setSandboxStateVersion((v) => v + 1);
              }
            }}
            onReplayStateChange={setReplayState}
            onReplayModeChange={setIsInReplayMode}
            onReplayAnimationChange={setReplayAnimation}
            onForkFromReplay={handleForkFromReplay}
            onHistoryIndexChange={(index) => {
              setHistoryViewIndex(index);
              if (sandboxGameState && index >= sandboxGameState.moveHistory.length) {
                setIsViewingHistory(false);
              }
            }}
            onExitHistoryView={() => {
              setIsViewingHistory(false);
              if (sandboxGameState) {
                setHistoryViewIndex(sandboxGameState.moveHistory.length);
              }
            }}
            onEnterHistoryView={() => setIsViewingHistory(true)}
            onClearSelection={clearSandboxSelection}
            onShowBoardControls={() => setShowBoardControls(true)}
            onToggleMovementGrid={setShowMovementGrid}
            onToggleValidTargets={setShowValidTargets}
            onToggleLineOverlays={developerToolsEnabled ? setShowLineOverlays : undefined}
            onToggleTerritoryOverlays={developerToolsEnabled ? setShowTerritoryOverlays : undefined}
            onSkipCapture={
              // RR-FIX-2026-01-12: Use memoized skipCaptureMoves instead of getValidMoves
              canSkipCaptureForTouch && sandboxEngine && sandboxGameState
                ? async () => {
                    const skipMove = skipCaptureMoves[0];
                    if (!skipMove) return;
                    await sandboxEngine.applyCanonicalMove(skipMove as Move);
                    setSandboxStateVersion((v) => v + 1);
                    clearSandboxSelection();
                    maybeRunSandboxAiIfNeeded();
                  }
                : undefined
            }
            onSkipTerritoryProcessing={() => {
              if (!sandboxPendingChoice || sandboxPendingChoice.type !== 'region_order') return;
              const currentChoice = sandboxPendingChoice;
              const options = (currentChoice.options ?? []) as Array<{
                regionId: string;
                size: number;
                representativePosition: Position;
                moveId: string;
              }>;
              const skipOption =
                options.find((opt) => opt.regionId === 'skip' || opt.size <= 0) ?? options[0];
              const resolver = sandboxChoiceResolverRef.current;
              if (resolver) {
                resolver({
                  choiceId: currentChoice.id,
                  playerNumber: currentChoice.playerNumber,
                  choiceType: currentChoice.type,
                  selectedOption: skipOption,
                } as PlayerChoiceResponseFor<PlayerChoice>);
                sandboxChoiceResolverRef.current = null;
              }
              setSandboxPendingChoice(null);
              setSandboxStateVersion((v) => v + 1);
              maybeRunSandboxAiIfNeeded();
            }}
            onSkipRecovery={
              // RR-FIX-2026-01-12: Use memoized skipRecoveryMoves instead of getValidMoves
              canSkipRecoveryForTouch && sandboxEngine && sandboxGameState
                ? async () => {
                    const skipMove = skipRecoveryMoves[0];
                    if (!skipMove) return;
                    await sandboxEngine.applyCanonicalMove(skipMove as Move);
                    setSandboxStateVersion((v) => v + 1);
                    clearSandboxSelection();
                    maybeRunSandboxAiIfNeeded();
                  }
                : undefined
            }
            onAdvancedPanelsToggle={setShowAdvancedSidebarPanels}
            onToggleAutoSave={setAutoSaveGames}
            onRefreshLadderHealth={aiTrackingActions.refreshLadderHealth}
            onCopyLadderHealth={aiTrackingActions.copyLadderHealth}
            onCopyAiTrace={diagnosticsActions.copyAiTrace}
            onCopyAiMeta={diagnosticsActions.copyAiMeta}
            onExportScenario={diagnosticsActions.exportScenarioJson}
            onCopyTestFixture={diagnosticsActions.copyTestFixture}
            onRequestEvaluation={requestSandboxEvaluation}
          />
        </main>

        <RingPlacementCountDialog
          isOpen={!!sandboxRingPlacementCountPrompt}
          maxCount={sandboxRingPlacementCountPrompt?.maxCount ?? 1}
          defaultCount={
            sandboxRingPlacementCountPrompt?.isStackPlacement
              ? 1
              : Math.min(2, sandboxRingPlacementCountPrompt?.maxCount ?? 1)
          }
          isStackPlacement={sandboxRingPlacementCountPrompt?.isStackPlacement ?? false}
          onClose={closeRingPlacementCountPrompt}
          onConfirm={confirmRingPlacementCountPrompt}
        />

        <RecoveryLineChoiceDialog
          isOpen={sandboxRecoveryChoicePromptOpen}
          onChooseOption1={() => resolveSandboxRecoveryChoice('option1')}
          onChooseOption2={() => resolveSandboxRecoveryChoice('option2')}
          onClose={() => resolveSandboxRecoveryChoice(null)}
        />

        <TerritoryRegionChoiceDialog
          isOpen={!!sandboxTerritoryRegionPrompt}
          options={sandboxTerritoryRegionPrompt?.options ?? []}
          onClose={closeTerritoryRegionPrompt}
          onConfirm={confirmTerritoryRegionPrompt}
        />

        {showBoardControls && (
          <BoardControlsOverlay
            mode="sandbox"
            hasTouchControlsPanel
            onClose={() => setShowBoardControls(false)}
          />
        )}

        <ScenarioPickerModal
          isOpen={showScenarioPicker}
          onClose={() => setShowScenarioPicker(false)}
          onSelectScenario={handleLoadScenario}
          developerToolsEnabled={developerToolsEnabled}
        />

        <SaveStateDialog
          isOpen={diagnosticsState.showSaveStateDialog}
          onClose={() => diagnosticsActions.setShowSaveStateDialog(false)}
          gameState={sandboxGameState}
          onSaved={(scenario) => {
            toast.success(`Saved state: ${scenario.name}`);
          }}
        />

        {/* Teaching Overlay - shown when "Learn More" is clicked in tutorial hints */}
        {teachingTopic && (
          <TeachingOverlay
            topic={teachingTopic}
            isOpen={showTeachingOverlay}
            onClose={() => {
              setShowTeachingOverlay(false);
              setTeachingTopic(null);
            }}
          />
        )}
      </div>
    </div>
  );
};
