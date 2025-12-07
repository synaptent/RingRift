import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameEventLog } from '../components/GameEventLog';
import { MoveHistory } from '../components/MoveHistory';
import type { MoveNotationOptions } from '../../shared/engine/notation';
import { SandboxTouchControlsPanel } from '../components/SandboxTouchControlsPanel';
import { BoardControlsOverlay } from '../components/BoardControlsOverlay';
import { ScenarioPickerModal } from '../components/ScenarioPickerModal';
import { SelfPlayBrowser } from '../components/SelfPlayBrowser';
import { EvaluationPanel } from '../components/EvaluationPanel';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';
import { SaveStateDialog } from '../components/SaveStateDialog';
import { ReplayPanel } from '../components/ReplayPanel';
import { HistoryPlaybackPanel } from '../components/HistoryPlaybackPanel';
import { OnboardingModal } from '../components/OnboardingModal';
import { useFirstTimePlayer } from '../hooks/useFirstTimePlayer';
import type { LoadableScenario } from '../sandbox/scenarioTypes';
import {
  BoardState,
  BoardType,
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
  Position,
  positionToString,
  CreateGameRequest,
} from '../../shared/types/game';
import { useAuth } from '../contexts/AuthContext';
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
import { GameHUD, VictoryConditionsPanel } from '../components/GameHUD';
import { MobileGameHUD } from '../components/MobileGameHUD';
import {
  ScreenReaderAnnouncer,
  useGameAnnouncements,
  useGameStateAnnouncements,
  GameAnnouncements,
} from '../components/ScreenReaderAnnouncer';
import { gameApi } from '../services/api';
import { getReplayService } from '../services/ReplayService';
import { storeGameLocally, getPendingCount } from '../services/LocalGameStorage';
import { GameSyncService, type SyncState } from '../services/GameSyncService';
import type {
  ClientSandboxEngine,
  SandboxInteractionHandler,
} from '../sandbox/ClientSandboxEngine';
import {
  logSandboxScenarioLoaded,
  logSandboxScenarioCompleted,
} from '../sandbox/sandboxRulesUxTelemetry';
import { getGameOverBannerText } from '../utils/gameCopy';
import { serializeGameState } from '../../shared/engine/contracts/serialization';
import { buildTestFixtureFromGameState, exportGameStateToFile } from '../sandbox/statePersistence';
import { useIsMobile } from '../hooks/useIsMobile';

const BOARD_PRESETS: Array<{
  value: BoardType;
  label: string;
  subtitle: string;
  blurb: string;
}> = [
  {
    value: 'square8',
    label: '8×8 Compact',
    subtitle: 'Fast tactical battles',
    blurb: 'Ideal for quick tests, fewer territories, emphasizes captures.',
  },
  {
    value: 'square19',
    label: '19×19 Classic',
    subtitle: 'Full RingRift experience',
    blurb: 'All line lengths and ring counts enabled for marathon sessions.',
  },
  {
    value: 'hexagonal',
    label: 'Full Hex',
    subtitle: 'High-mobility frontier',
    blurb: 'Hex adjacency, sweeping captures, and large territory swings.',
  },
];

/** Quick-start scenario presets that configure multiple settings at once */
const QUICK_START_PRESETS: Array<{
  id: string;
  label: string;
  description: string;
  /** Extended description shown on hover or in expanded view */
  learnMoreText?: string;
  icon: string;
  /** Badge shown next to preset (e.g., "Recommended", "New") */
  badge?: string;
  config: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
  };
}> = [
  {
    id: 'learn-basics',
    label: 'Learn the Basics',
    description: 'Best for first-time players',
    learnMoreText:
      'Start here to learn ring placement, movement, and captures on a compact board. The AI plays at a beginner-friendly level.',
    icon: '📚',
    badge: 'Recommended',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'human-vs-ai',
    label: 'Human vs AI',
    description: 'Quick 1v1 against the computer',
    learnMoreText:
      'A standard match against the AI. Practice your strategies and see how different tactics play out.',
    icon: '🤖',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'square19-human-vs-ai',
    label: 'Full Board vs AI',
    description: '19×19 human vs AI marathon',
    learnMoreText:
      'Play a full-length RingRift game on the 19×19 board against the AI. Best once you are comfortable on 8×8.',
    icon: '🏰',
    config: {
      boardType: 'square19',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'ai-battle',
    label: 'AI Battle',
    description: 'Watch two AIs compete',
    learnMoreText:
      'Observe AI strategies without playing. Great for learning tactics and understanding game flow.',
    icon: '⚔️',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'square19-ai-battle',
    label: 'Full Board AI Battle',
    description: 'Watch two AIs on 19×19',
    learnMoreText:
      'Let two AIs play a long game on the classic 19×19 board. Good for observing late-game territory and elimination patterns.',
    icon: '🎥',
    config: {
      boardType: 'square19',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hotseat',
    label: 'Hotseat',
    description: 'Two humans on one device',
    learnMoreText:
      'Pass-and-play mode for two players sharing a device. Take turns and compete face-to-face.',
    icon: '👥',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'hex-challenge',
    label: 'Hex Challenge',
    description: 'Human vs AI on hex board',
    learnMoreText:
      'Hexagonal boards offer 6-way movement and unique tactical possibilities. A step up in complexity.',
    icon: '⬡',
    config: {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'four-player',
    label: '4-Player Free-for-All',
    description: 'Chaotic multiplayer on hex',
    learnMoreText:
      'Compete against three AI opponents. Alliances shift, territory control is crucial, and no one is safe.',
    icon: '🎲',
    config: {
      boardType: 'hexagonal',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
];

const PLAYER_TYPE_META: Record<
  LocalPlayerType,
  { label: string; description: string; accent: string; chip: string }
> = {
  human: {
    label: 'Human',
    description: 'You control every move',
    accent: 'border-emerald-500 text-emerald-200',
    chip: 'bg-emerald-900/40 text-emerald-200',
  },
  ai: {
    label: 'Computer',
    description: 'Local heuristic AI',
    accent: 'border-sky-500 text-sky-200',
    chip: 'bg-sky-900/40 text-sky-200',
  },
};

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
      'Resolve completed marker lines into Territory and choose whether to take or skip any ring-elimination reward.',
  },
  territory_processing: {
    label: 'Territory Claim',
    summary:
      'Evaluate disconnected regions, collapse them into Territory, and pay any required self-elimination cost; territory wins are checked here.',
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
  const navigate = useNavigate();
  const { user } = useAuth();
  const { shouldShowWelcome, markWelcomeSeen, markGameCompleted, isFirstTimePlayer } =
    useFirstTimePlayer();

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
    setDeveloperToolsEnabled,
    initLocalSandboxEngine,
    resetSandboxEngine,
  } = useSandbox();

  const [sandboxEvaluationHistory, setSandboxEvaluationHistory] = useState<
    PositionEvaluationPayload['data'][]
  >([]);
  const [sandboxEvaluationError, setSandboxEvaluationError] = useState<string | null>(null);
  const [isSandboxAnalysisRunning, setIsSandboxAnalysisRunning] = useState(false);

  // Local-only diagnostics / UX state
  const [isSandboxVictoryModalDismissed, setIsSandboxVictoryModalDismissed] = useState(false);

  // Replay mode state (for database-loaded games via ReplayPanel)
  const [isInReplayMode, setIsInReplayMode] = useState(false);
  const [replayState, setReplayState] = useState<GameState | null>(null);
  const [replayAnimation, setReplayAnimation] = useState<
    import('../components/BoardView').MoveAnimationData | null
  >(null);
  // When a self-play scenario is loaded, this bridges the gameId into the
  // ReplayPanel so it can attempt to drive the board from the AI service's
  // /api/replay endpoints (Option A).
  const [requestedReplayGameId, setRequestedReplayGameId] = useState<string | null>(null);

  // History scrubbing state (for locally loaded fixtures/scenarios)
  const [isViewingHistory, setIsViewingHistory] = useState(false);
  const [historyViewIndex, setHistoryViewIndex] = useState(0);
  const [hasHistorySnapshots, setHasHistorySnapshots] = useState(true);

  // Selection + valid target highlighting
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);
  // Start with the movement grid overlay enabled by default; it helps
  // players understand valid moves and adjacency patterns.
  const [showMovementGrid, setShowMovementGrid] = useState(true);
  const [showValidTargetsOverlay, setShowValidTargetsOverlay] = useState(true);
  // Debug overlays for visualizing detected lines and territory regions
  const [showLineOverlays, setShowLineOverlays] = useState(true);
  const [showTerritoryOverlays, setShowTerritoryOverlays] = useState(true);

  // Help / controls overlay for the active sandbox host
  const [showBoardControls, setShowBoardControls] = useState(false);

  // Scenario picker, self-play browser, and save state dialogs
  const [showScenarioPicker, setShowScenarioPicker] = useState(false);
  const [showSelfPlayBrowser, setShowSelfPlayBrowser] = useState(false);
  const [showSaveStateDialog, setShowSaveStateDialog] = useState(false);
  const [lastLoadedScenario, setLastLoadedScenario] = useState<LoadableScenario | null>(null);

  // Game storage state - auto-save completed games to replay database
  const [autoSaveGames, setAutoSaveGames] = useState(true);
  const [gameSaveStatus, setGameSaveStatus] = useState<
    'idle' | 'saving' | 'saved' | 'saved-local' | 'error'
  >('idle');
  const [pendingLocalGames, setPendingLocalGames] = useState(0);
  const [syncState, setSyncState] = useState<SyncState | null>(null);
  const initialGameStateRef = useRef<GameState | null>(null);
  const gameSavedRef = useRef(false);
  const lastEvaluatedMoveRef = useRef<number | null>(null);

  // Screen reader announcements for accessibility - using priority queue (mirrors BackendGameHost)
  const { queue: announcementQueue, announce, removeAnnouncement } = useGameAnnouncements();

  // Show/hide advanced options - collapsed by default for first-time players
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(!isFirstTimePlayer);

  // Safari and some older runtimes may not expose structuredClone; provide a
  // small defensive wrapper so initial sandbox snapshots remain available
  // without crashing the host. For replay storage we only require a
  // JSON-serializable clone.
  const cloneInitialGameState = useCallback((state: GameState): GameState => {
    const globalClone = (globalThis as any).structuredClone as
      | ((value: unknown) => unknown)
      | undefined;
    if (typeof globalClone === 'function') {
      return globalClone(state) as GameState;
    }
    return JSON.parse(JSON.stringify(state)) as GameState;
  }, []);

  const sandboxChoiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);

  const lastSandboxPhaseRef = useRef<string | null>(null);

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
  } = useSandboxInteractions({
    selected,
    setSelected,
    validTargets,
    setValidTargets,
    choiceResolverRef: sandboxChoiceResolverRef,
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

  const requestSandboxEvaluation = useCallback(async () => {
    // Get game state from engine directly to avoid forward reference issues
    const gameState = sandboxEngine?.getGameState();
    if (!sandboxEngine || !gameState) {
      return;
    }

    try {
      setIsSandboxAnalysisRunning(true);
      setSandboxEvaluationError(null);

      const serialized = sandboxEngine.getSerializedState();

      const response = await fetch('/api/games/sandbox/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ state: serialized }),
      });

      if (!response.ok) {
        let message = 'Sandbox evaluation request failed.';
        try {
          const errorBody = (await response.json()) as { error?: string } | null;
          if (errorBody && typeof errorBody.error === 'string') {
            message = errorBody.error;
          }
        } catch {
          // Ignore JSON parse errors (HTML or empty responses)
        }

        const statusHint =
          response.status === 404
            ? 'AI evaluation is not enabled for this environment.'
            : response.status === 503
              ? 'Sandbox AI evaluation service is unavailable. Ensure the AI service is running.'
              : `HTTP ${response.status}`;

        setSandboxEvaluationError(`${message} ${statusHint}`.trim());
        return;
      }

      const data: PositionEvaluationPayload['data'] = await response.json();
      setSandboxEvaluationHistory((prev) => [...prev, data]);
    } catch (err) {
      console.warn('Sandbox evaluation request threw', err);
      const message =
        err instanceof Error ? err.message : 'Unknown error during sandbox evaluation request';
      setSandboxEvaluationError(`Sandbox evaluation failed: ${message}`);
    } finally {
      setIsSandboxAnalysisRunning(false);
    }
  }, [sandboxEngine]);

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig((prev) => {
      const numPlayers = partial.numPlayers;
      return {
        ...prev,
        ...partial,
        playerTypes: numPlayers
          ? prev.playerTypes.map((t, idx) => (idx < numPlayers ? t : prev.playerTypes[idx]))
          : prev.playerTypes,
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

  const startLocalSandboxGame = (snapshot: LocalConfig) => {
    const nextBoardType = snapshot.boardType;

    // Fallback: local sandbox engine using orchestrator-first semantics.
    const interactionHandler = createSandboxInteractionHandler(
      snapshot.playerTypes.slice(0, snapshot.numPlayers)
    );
    const engine = initLocalSandboxEngine({
      boardType: nextBoardType,
      numPlayers: snapshot.numPlayers,
      playerTypes: snapshot.playerTypes.slice(0, snapshot.numPlayers) as LocalPlayerType[],
      interactionHandler,
    });

    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);

    // If the first player is an AI, immediately start the sandbox AI turn loop.
    if (engine) {
      const state = engine.getGameState();
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (current && current.type === 'ai') {
        maybeRunSandboxAiIfNeeded();
      }
    }
  };

  const startSandboxGame = async (snapshot: LocalConfig) => {
    const nextBoardType = snapshot.boardType;

    // Starting a non-scenario sandbox game; clear any prior scenario context so
    // scenario-specific telemetry does not attribute future victories here.
    setLastLoadedScenario(null);

    // When not authenticated, skip backend game creation entirely and go
    // straight to the local sandbox engine to avoid expected 401 noise.
    if (!user) {
      startLocalSandboxGame(snapshot);
      return;
    }

    // First, attempt to create a real backend game using the same CreateGameRequest
    // shape as the lobby. On success, navigate into the real backend game route.
    try {
      const payload: CreateGameRequest = {
        boardType: nextBoardType,
        maxPlayers: snapshot.numPlayers,
        isRated: false,
        isPrivate: true,
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        aiOpponents: (() => {
          const aiSeats = snapshot.playerTypes
            .slice(0, snapshot.numPlayers)
            .filter((t) => t === 'ai').length;
          if (aiSeats <= 0) return undefined;
          return {
            count: aiSeats,
            difficulty: Array(aiSeats).fill(5),
            mode: 'service',
            aiType: 'heuristic',
          };
        })(),
        // Mirror lobby behaviour: default-enable the pie rule for 2-player
        // backend sandbox games. Local-only sandbox games (fallback path)
        // continue to use the shared engine's defaults.
        rulesOptions: snapshot.numPlayers === 2 ? { swapRuleEnabled: true } : undefined,
      };

      const game = await gameApi.createGame(payload);
      navigate(`/game/${game.id}`);
      return;
    } catch (err) {
      console.error('Failed to create backend sandbox game, falling back to local-only board', err);
      setBackendSandboxError(
        'Backend sandbox game could not be created; falling back to local-only board only.'
      );
    }

    startLocalSandboxGame(snapshot);
  };

  const handleQuickStartPreset = (preset: (typeof QUICK_START_PRESETS)[number]) => {
    // Build an explicit snapshot so we can both update config and launch a
    // game immediately without relying on async state updates.
    const baseTypes = [...config.playerTypes];
    const updatedTypes = baseTypes.map((t, idx) =>
      idx < preset.config.playerTypes.length ? preset.config.playerTypes[idx] : t
    );

    const snapshot: LocalConfig = {
      boardType: preset.config.boardType,
      numPlayers: preset.config.numPlayers,
      playerTypes: updatedTypes,
    };

    setConfig(snapshot);
    void startSandboxGame(snapshot);
  };

  // Handler for starting tutorial from onboarding modal - selects "Learn the Basics"
  const handleStartTutorial = useCallback(() => {
    markWelcomeSeen();
    const learnBasicsPreset = QUICK_START_PRESETS.find((p) => p.id === 'learn-basics');
    if (learnBasicsPreset) {
      handleQuickStartPreset(learnBasicsPreset);
    }
  }, [markWelcomeSeen]);

  const setAllPlayerTypes = (type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      for (let i = 0; i < prev.numPlayers; i += 1) {
        next[i] = type;
      }
      return { ...prev, playerTypes: next };
    });
  };

  const createSandboxInteractionHandler = (
    playerTypesSnapshot: LocalPlayerType[]
  ): SandboxInteractionHandler => {
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
        // choice without surfacing any blocking UI. This keeps sandbox
        // decisions (including territory region and elimination) flowing
        // without unnecessary dialogs when there is nothing to choose.
        const rawOptions = (choice as any).options as TChoice['options'] | undefined;
        const autoOptions = (rawOptions as unknown[]) ?? [];
        if (autoOptions.length === 1) {
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
          setSandboxPendingChoice(choice);
        }

        return new Promise<PlayerChoiceResponseFor<TChoice>>((resolve) => {
          sandboxChoiceResolverRef.current = ((response: PlayerChoiceResponseFor<PlayerChoice>) => {
            resolve(response as PlayerChoiceResponseFor<TChoice>);
          }) as (response: PlayerChoiceResponseFor<PlayerChoice>) => void;
        });
      },
    };
  };

  /**
   * Load a scenario from the scenario picker.
   * This initializes the sandbox engine from a pre-existing serialized game state.
   */
  const handleLoadScenario = (scenario: LoadableScenario) => {
    // Update config to match scenario settings
    setConfig((prev) => ({
      ...prev,
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
    }));

    // Get player types. For general fixtures, default to human vs AI for 2
    // players. For recorded self-play games, treat all seats as human so we
    // do not auto-run local AI turns while we are replaying the canonical
    // move sequence from the recorder.
    let playerTypes: LocalPlayerType[] =
      scenario.playerCount === 2
        ? ['human', 'ai', 'human', 'human']
        : (config.playerTypes.slice(0, scenario.playerCount) as LocalPlayerType[]);

    if (scenario.selfPlayMeta) {
      playerTypes = Array.from({ length: scenario.playerCount }, () => 'human' as LocalPlayerType);
    }

    // Create interaction handler
    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    // Initialize sandbox engine with the scenario state
    const engine = initLocalSandboxEngine({
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
      playerTypes,
      interactionHandler,
    });

    // Load the serialized state into the engine. When scenarios carry a
    // terminal gameStatus (e.g. completed self-play snapshots captured via
    // ringrift_sandbox_fixture_v1), normalise them into a fresh, playable
    // sandbox snapshot by re-opening the game as active from ring_placement.
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

    engine.initFromSerializedState(normalizedState, playerTypes, interactionHandler);

    // Reset UI state
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setSandboxStateVersion(0);
    setLastLoadedScenario(scenario);

    // Emit a sandbox_scenario_loaded event for curated teaching scenarios.
    void logSandboxScenarioLoaded(scenario);

    // Reset history-playback availability; fixtures loaded via ScenarioPicker
    // may or may not include internal snapshots. We conservatively assume
    // snapshots exist for local fixtures and will downgrade this flag when
    // getStateAtMoveIndex reports null for self-play snapshots.
    setHasHistorySnapshots(true);
    setIsViewingHistory(false);
    setHistoryViewIndex(0);

    // If this scenario originated from a recorded self-play game, attempt to
    // reconstruct the full move trajectory locally (Option B) so the history
    // slider under the board can scrub through every recorded move even when
    // the AI service replay DB is unavailable.
    if (scenario.selfPlayMeta) {
      const recordedMoves: Move[] | undefined = scenario.selfPlayMeta.moves;

      if (recordedMoves && recordedMoves.length > 0) {
        void (async () => {
          // Best-effort debug instrumentation so we can understand how many
          // canonical moves successfully replay into the sandbox engine for
          // recorded self-play games.
          // eslint-disable-next-line no-console
          console.log('[SandboxSelfPlayReplay] Replaying recorded self-play game', {
            gameId: scenario.selfPlayMeta?.gameId,
            totalRecordedMoves: recordedMoves.length,
          });

          let appliedCount = 0;

          try {
            for (let i = 0; i < recordedMoves.length; i += 1) {
              const move = recordedMoves[i];
              // Moves come from the recorder and already match the canonical
              // Move surface; we cast through any to avoid over-constraining
              // the type here.
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              await (engine as ClientSandboxEngine).applyCanonicalMoveForReplay(move as any);
              appliedCount += 1;
            }

            // Bump state version so derived views (board, HUD, history) pick
            // up the final replayed state and the newly-populated snapshots.
            setSandboxStateVersion((v) => v + 1);
            setHasHistorySnapshots(true);

            // eslint-disable-next-line no-console
            console.log('[SandboxSelfPlayReplay] Finished local replay', {
              gameId: scenario.selfPlayMeta?.gameId,
              appliedMoves: appliedCount,
              historyLength: (engine as ClientSandboxEngine).getGameState().moveHistory.length,
            });
          } catch (err) {
            console.error(
              '[SandboxGameHost] Failed to replay recorded self-play game into sandbox engine',
              {
                error: err,
                gameId: scenario.selfPlayMeta?.gameId,
                appliedMoves: appliedCount,
                totalRecordedMoves: recordedMoves.length,
              }
            );
            setHasHistorySnapshots(false);
          }
        })();
      } else {
        // No recorded moves were attached; disable snapshot-driven history so
        // the slider renders with an explicit "unavailable" hint instead of
        // appearing to scrub a static board.
        setHasHistorySnapshots(false);
      }

      // Best-effort Option A: bridge this gameId into the ReplayPanel so,
      // when the AI service is pointed at a compatible GameReplayDB, the
      // board can also be driven directly from /api/replay.
      setRequestedReplayGameId(scenario.selfPlayMeta.gameId);
      setIsInReplayMode(false);
      setReplayState(null);
    } else {
      setRequestedReplayGameId(null);
      setIsInReplayMode(false);
      setReplayState(null);
    }

    toast.success(`Loaded scenario: ${scenario.name}`);
  };

  /**
   * Fork from a replay position - loads the replay state into the sandbox engine
   * as a new playable game.
   */
  const handleForkFromReplay = (state: GameState) => {
    // Update config to match the replay state
    const numPlayers = state.players.length;
    setConfig((prev) => ({
      ...prev,
      boardType: state.board.type,
      numPlayers,
    }));

    // Default player types for forked game
    const playerTypes: LocalPlayerType[] = state.players.map((_, idx) =>
      idx === 0 ? 'human' : 'ai'
    );

    // Create interaction handler
    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    // Initialize sandbox engine
    const engine = initLocalSandboxEngine({
      boardType: state.board.type,
      numPlayers,
      playerTypes,
      interactionHandler,
    });

    // Load the state into the engine (convert GameState to SerializedGameState)
    const serialized = serializeGameState(state);
    const isTerminalStatus =
      serialized.gameStatus === 'completed' || serialized.gameStatus === 'finished';
    const normalizedSerialized = isTerminalStatus
      ? {
          ...serialized,
          gameStatus: 'active',
          currentPhase: 'ring_placement',
          chainCapturePosition: undefined,
        }
      : serialized;

    engine.initFromSerializedState(normalizedSerialized, playerTypes, interactionHandler);

    // Reset UI state
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setSandboxStateVersion(0);
    setLastLoadedScenario(null);

    // Exit replay mode
    setIsInReplayMode(false);
    setReplayState(null);

    toast.success('Forked from replay position');
  };

  const handleResetScenario = () => {
    if (!lastLoadedScenario) {
      return;
    }

    const scenario = lastLoadedScenario;

    // Mirror the logic from handleLoadScenario so reset behaves the same as a
    // fresh scenario load.
    setConfig((prev) => ({
      ...prev,
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
    }));

    let playerTypes: LocalPlayerType[] =
      scenario.playerCount === 2
        ? ['human', 'ai', 'human', 'human']
        : (config.playerTypes.slice(0, scenario.playerCount) as LocalPlayerType[]);

    if (scenario.selfPlayMeta) {
      playerTypes = Array.from({ length: scenario.playerCount }, () => 'human' as LocalPlayerType);
    }

    const interactionHandler = createSandboxInteractionHandler(playerTypes);

    const engine = initLocalSandboxEngine({
      boardType: scenario.boardType,
      numPlayers: scenario.playerCount,
      playerTypes,
      interactionHandler,
    });

    engine.initFromSerializedState(scenario.state, playerTypes, interactionHandler);

    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setSandboxCaptureChoice(null);
    setSandboxCaptureTargets([]);
    setSandboxStallWarning(null);
    setSandboxLastProgressAt(null);
    setIsSandboxVictoryModalDismissed(false);
    setBackendSandboxError(null);
    setSandboxStateVersion((v) => v + 1);
    setHasHistorySnapshots(true);
    setIsViewingHistory(false);
    setHistoryViewIndex(0);

    // Treat a scenario reset as a fresh load from a telemetry perspective.
    void logSandboxScenarioLoaded(scenario);

    // Re-run local reconstruction for recorded self-play scenarios so the
    // history slider remains meaningful after a reset.
    if (
      scenario.selfPlayMeta &&
      scenario.selfPlayMeta.moves &&
      scenario.selfPlayMeta.moves.length
    ) {
      void (async () => {
        let appliedCount = 0;
        try {
          for (const move of scenario.selfPlayMeta?.moves as Move[]) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            await (engine as ClientSandboxEngine).applyCanonicalMoveForReplay(move as any);
            appliedCount += 1;
          }
          setSandboxStateVersion((v) => v + 1);
          setHasHistorySnapshots(true);
          // eslint-disable-next-line no-console
          console.log('[SandboxSelfPlayReplay] Finished local replay on scenario reset', {
            gameId: scenario.selfPlayMeta?.gameId,
            appliedMoves: appliedCount,
            historyLength: (engine as ClientSandboxEngine).getGameState().moveHistory.length,
          });
        } catch (err) {
          console.error(
            '[SandboxGameHost] Failed to replay recorded self-play game on scenario reset',
            { error: err, gameId: scenario.selfPlayMeta?.gameId, appliedMoves: appliedCount }
          );
          setHasHistorySnapshots(false);
        }
      })();
    } else if (scenario.selfPlayMeta) {
      setHasHistorySnapshots(false);
    }

    toast.success(`Scenario reset: ${scenario.name}`);
  };

  const handleStartLocalGame = async () => {
    await startSandboxGame(config);
  };

  const handleCopySandboxTrace = async () => {
    try {
      if (typeof window === 'undefined') {
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing debug global
      const anyWindow = window as any;
      const trace = anyWindow.__RINGRIFT_SANDBOX_TRACE__ ?? [];
      const payload = JSON.stringify(trace, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox AI trace copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox AI trace', trace);
        toast.success('Sandbox AI trace logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox AI trace', err);
      toast.error('Failed to export sandbox AI trace; see console for details.');
    }
  };

  const handleCopySandboxFixture = async () => {
    try {
      if (!sandboxGameState) {
        toast.error('No sandbox game is currently active.');
        return;
      }

      const fixture = buildTestFixtureFromGameState(sandboxGameState);
      const payload = JSON.stringify(fixture, null, 2);

      if (
        typeof navigator !== 'undefined' &&
        navigator.clipboard &&
        navigator.clipboard.writeText
      ) {
        await navigator.clipboard.writeText(payload);
        toast.success('Sandbox test fixture copied to clipboard');
      } else {
        // eslint-disable-next-line no-console
        console.log('Sandbox test fixture', fixture);
        toast.success('Sandbox test fixture logged to console (clipboard API unavailable).');
      }
    } catch (err) {
      console.error('Failed to export sandbox test fixture', err);
      toast.error('Failed to export sandbox test fixture; see console for details.');
    }
  };

  const handleExportScenarioJson = () => {
    try {
      if (!sandboxGameState) {
        toast.error('No sandbox game is currently active.');
        return;
      }

      const turnLabel = sandboxGameState.moveHistory.length + 1;
      const name = `Sandbox Scenario - Turn ${turnLabel}`;
      exportGameStateToFile(sandboxGameState, name);
      toast.success('Sandbox scenario JSON downloaded');
    } catch (err) {
      console.error('Failed to export sandbox scenario JSON', err);
      toast.error('Failed to export sandbox scenario; see console for details.');
    }
  };

  // Game view once configured (local sandbox)
  const sandboxGameState: GameState | null = sandboxEngine ? sandboxEngine.getGameState() : null;
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
  const { pendingAnimation, clearAnimation } = useAutoMoveAnimation(sandboxGameState);

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

  // When a territory region-order decision is active, surface a territory-
  // specific prompt chip so territory-processing phases receive the same
  // high-attention treatment as line-formation eliminations.
  const isRegionOrderChoice =
    (sandboxCaptureChoice ?? sandboxPendingChoice)?.type === 'region_order';

  // When in chain_capture with available continuation segments, surface an
  // attention-style chip prompting the user to continue the chain. This mirrors
  // backend HUD semantics for mandatory chain continuation.
  const isChainCaptureContinuationStep = !!(
    sandboxGameState &&
    sandboxGameState.currentPhase === 'chain_capture' &&
    sandboxEngine &&
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- duck-typing for optional engine method
    typeof (sandboxEngine as any).getValidMoves === 'function' &&
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- accessing internal engine method
    (sandboxEngine as any)
      .getValidMoves(sandboxGameState.currentPlayer)
      .some((m: { type: string }) => m.type === 'continue_capture_segment')
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

      lastSandboxPhaseRef.current = nextPhase;
    }
  }, [sandboxGameState]);

  const humanSeatCount = sandboxPlayersList.filter((p) => p.type === 'human').length;
  const aiSeatCount = sandboxPlayersList.length - humanSeatCount;

  // Whenever the sandbox state reflects an active AI turn, trigger the
  // sandbox AI loop after a short delay. This keeps AI progression in
  // sync with orchestrator-driven state changes (including line/territory
  // processing and elimination decisions) without requiring an extra
  // board click from the user.
  useEffect(() => {
    if (!sandboxEngine || !sandboxGameState) {
      return;
    }

    const current = sandboxGameState.players.find(
      (p) => p.playerNumber === sandboxGameState.currentPlayer
    );

    if (sandboxGameState.gameStatus !== 'active' || !current || current.type !== 'ai') {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      maybeRunSandboxAiIfNeeded();
    }, 60);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [sandboxEngine, sandboxGameState, maybeRunSandboxAiIfNeeded]);

  // Derive board VM + HUD-like summaries
  const primaryValidTargets =
    sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets;

  const displayedValidTargets = showValidTargetsOverlay ? primaryValidTargets : [];

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
    sandboxGameState.currentPhase === 'capture' &&
    sandboxEngine
  ) {
    const moves = sandboxEngine.getValidMoves(sandboxGameState.currentPlayer);
    const captureMoves = moves.filter((m) => m.type === 'overtaking_capture');

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

  const sandboxBoardViewModel = sandboxBoardState
    ? toBoardViewModel(sandboxBoardState, {
        selectedPosition: selected,
        validTargets: displayedValidTargets,
        decisionHighlights,
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
          gameEndExplanation: sandboxGameEndExplanation,
        }
      )
    : null;

  const sandboxModeNotes = [
    `Board: ${boardDisplayLabel}`,
    `${humanSeatCount} human seat${humanSeatCount === 1 ? '' : 's'} · ${aiSeatCount} AI`,
    sandboxEngine
      ? 'Engine parity mode with local AI and choice handler.'
      : 'Legacy local sandbox fallback (no backend).',
    'Runs entirely in-browser; use "Change Setup" to switch configurations.',
    !user ? 'You’re not logged in; this game runs as a local sandbox only.' : null,
  ].filter(Boolean) as string[];

  // Create unified HUD view model using toHUDViewModel (matches backend host pattern)
  let sandboxHudVM = sandboxGameState
    ? toHUDViewModel(sandboxGameState, {
        connectionStatus: 'connected',
        lastHeartbeatAt: null,
        isSpectator: false,
        currentUserId: user?.id,
        pendingChoice: activePendingChoice,
        choiceDeadline: null,
        choiceTimeRemainingMs: sandboxDecisionTimeRemainingMs,
        victoryState: sandboxVictoryResult,
        gameEndExplanation: sandboxGameEndExplanation,
      })
    : null;

  // Optional-capture HUD chip: when capture is available directly from the
  // capture phase (with skip_capture as an option) but no explicit decision
  // choice is active, surface a bright attention chip so players do not miss
  // the opportunity.
  if (
    sandboxHudVM &&
    !sandboxHudVM.decisionPhase &&
    sandboxGameState &&
    sandboxGameState.gameStatus === 'active' &&
    sandboxGameState.currentPhase === 'capture' &&
    sandboxEngine
  ) {
    const moves = sandboxEngine.getValidMoves(sandboxGameState.currentPlayer);
    const hasCaptureMove = moves.some((m) => m.type === 'overtaking_capture');
    const hasSkipCaptureMove = moves.some((m) => m.type === 'skip_capture');

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

  const sandboxEventLogViewModel = toEventLogViewModel(
    sandboxGameState?.history ?? [],
    [],
    sandboxVictoryResult,
    { maxEntries: 40 }
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

  const activePlayerTypes = config.playerTypes.slice(0, config.numPlayers);
  const setupHumanSeatCount = activePlayerTypes.filter((t) => t === 'human').length;
  const setupAiSeatCount = activePlayerTypes.length - setupHumanSeatCount;
  const selectedBoardPreset =
    BOARD_PRESETS.find((preset) => preset.value === config.boardType) ?? BOARD_PRESETS[0];

  const canSkipCaptureForTouch =
    !!sandboxGameState &&
    sandboxGameState.currentPhase === 'capture' &&
    !!sandboxEngine &&
    sandboxEngine
      .getValidMoves(sandboxGameState.currentPlayer)
      .some((m) => m.type === 'skip_capture');

  // Keyboard shortcuts for sandbox overlay:
  // - "?" (Shift + "/") toggles the Board Controls overlay when a sandbox game is active.
  // - "Escape" closes the overlay when open.
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

      if (event.key === '?' || (event.key === '/' && event.shiftKey)) {
        event.preventDefault();
        setShowBoardControls((prev) => !prev);
        return;
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

  // Capture initial game state when engine is created for game storage
  useEffect(() => {
    if (!sandboxEngine) {
      // Reset refs when engine is destroyed
      initialGameStateRef.current = null;
      gameSavedRef.current = false;
      lastEvaluatedMoveRef.current = null;
      setGameSaveStatus('idle');
      return;
    }
    // Capture initial state only once per game (when moveHistory is empty)
    const currentState = sandboxEngine.getGameState();
    if (currentState.moveHistory.length === 0 && !initialGameStateRef.current) {
      initialGameStateRef.current = cloneInitialGameState(currentState);
      gameSavedRef.current = false;
      setGameSaveStatus('idle');
    }
  }, [sandboxEngine, cloneInitialGameState]);

  // Start game sync service and subscribe to state updates
  useEffect(() => {
    GameSyncService.start();
    const unsubscribe = GameSyncService.subscribe((state) => {
      setSyncState(state);
      setPendingLocalGames(state.pendingCount);
    });
    return () => {
      unsubscribe();
      GameSyncService.stop();
    };
  }, []);

  // Auto-save completed games to replay database when victory is detected
  useEffect(() => {
    if (!autoSaveGames || !sandboxVictoryResult || gameSavedRef.current) {
      return;
    }

    const saveCompletedGame = async () => {
      const finalState = sandboxEngine?.getGameState();
      const initialState = initialGameStateRef.current;

      if (!finalState || !initialState) {
        console.warn('[SandboxGameHost] Cannot save game: missing state');
        return;
      }

      const metadata = {
        source: 'sandbox',
        boardType: finalState.board.type,
        numPlayers: finalState.players.length,
        playerTypes: config.playerTypes.slice(0, config.numPlayers),
        victoryReason: sandboxVictoryResult.reason,
        winnerPlayerNumber: sandboxVictoryResult.winner,
      };

      try {
        setGameSaveStatus('saving');
        const replayService = getReplayService();
        const result = await replayService.storeGame({
          initialState,
          finalState,
          moves: finalState.moveHistory as unknown as Record<string, unknown>[],
          metadata,
        });

        if (result.success) {
          gameSavedRef.current = true;
          setGameSaveStatus('saved');
          toast.success(`Game saved (${result.totalMoves} moves)`);
        } else {
          // Server rejected - try local fallback
          throw new Error('Server rejected game storage');
        }
      } catch (error) {
        console.warn('[SandboxGameHost] Server save failed, trying local storage:', error);

        // Fallback to IndexedDB local storage
        try {
          const localResult = await storeGameLocally(
            initialState,
            finalState,
            finalState.moveHistory as unknown[],
            metadata
          );

          if (localResult.success) {
            gameSavedRef.current = true;
            setGameSaveStatus('saved-local');
            const newCount = await getPendingCount();
            setPendingLocalGames(newCount);
            toast.success('Game saved locally (will sync when server available)', {
              icon: '💾',
            });
          } else {
            setGameSaveStatus('error');
            toast.error('Failed to save game');
          }
        } catch (localError) {
          console.error('[SandboxGameHost] Local storage also failed:', localError);
          setGameSaveStatus('error');
          toast.error('Failed to save game (storage unavailable)');
        }
      }
    };

    saveCompletedGame();
  }, [autoSaveGames, sandboxVictoryResult, sandboxEngine, config.playerTypes, config.numPlayers]);

  // When developer tools are enabled, automatically request a sandbox AI
  // evaluation after each new move so the EvaluationPanel can render a
  // lightweight sparkline over the turn history.
  useEffect(() => {
    if (!developerToolsEnabled || !sandboxEngine || !sandboxGameState) {
      return;
    }

    // Skip when viewing historical states via replay/fixtures.
    if (isInReplayMode || isViewingHistory) {
      return;
    }

    const moveNumber = sandboxGameState.moveHistory.length;
    if (moveNumber <= 0) {
      return;
    }

    if (lastEvaluatedMoveRef.current === moveNumber) {
      return;
    }

    lastEvaluatedMoveRef.current = moveNumber;
    // Fire and forget; requestSandboxEvaluation manages its own loading state.
    requestSandboxEvaluation();
  }, [
    developerToolsEnabled,
    sandboxEngine,
    sandboxGameState,
    isInReplayMode,
    isViewingHistory,
    requestSandboxEvaluation,
  ]);

  // Pre-game setup view
  if (!isConfigured || !sandboxEngine) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-100">
        <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-4 sm:space-y-6">
          <header className="flex flex-col gap-2 sm:gap-3 sm:flex-row sm:items-baseline sm:justify-between">
            <div>
              <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold mb-1 flex items-center gap-2">
                <img
                  src="/ringrift-icon.png"
                  alt="RingRift"
                  className="w-6 h-6 sm:w-8 sm:h-8 flex-shrink-0"
                />
                <span>RingRift – Start a Game (Sandbox)</span>
              </h1>
              <p className="text-sm text-slate-400">
                This mode runs entirely in the browser using a local board. To view or play a real
                server-backed game, navigate to a URL with a game ID (e.g.
                <code className="ml-1 text-xs text-slate-300">/game/:gameId</code>).
              </p>
            </div>
            <label className="inline-flex items-center gap-2 text-xs text-slate-400 select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
                checked={developerToolsEnabled}
                onChange={(e) => setDeveloperToolsEnabled(e.target.checked)}
              />
              <span>Developer Tools</span>
            </label>
          </header>

          {/* Quick-start presets */}
          <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Quick Start</p>
                <h2 className="text-lg font-semibold text-white">Choose a preset</h2>
                <p className="text-xs text-slate-400 mt-1">
                  Click a preset to launch a local sandbox game immediately.
                </p>
              </div>
              {isFirstTimePlayer && (
                <span className="text-sm text-emerald-400 animate-pulse flex items-center gap-1">
                  <span aria-hidden="true">👇</span> Start here
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {QUICK_START_PRESETS.map((preset) => {
                const isLearnBasics = preset.id === 'learn-basics';
                const shouldHighlight = isLearnBasics && isFirstTimePlayer;

                return (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => handleQuickStartPreset(preset)}
                    title={preset.learnMoreText}
                    className={`relative flex items-center gap-2 px-3 py-2 rounded-xl border text-slate-200 transition text-sm ${
                      shouldHighlight
                        ? 'border-emerald-400 bg-emerald-900/40 ring-2 ring-emerald-500/50 ring-offset-2 ring-offset-slate-900 animate-pulse hover:animate-none hover:bg-emerald-900/50'
                        : preset.badge
                          ? 'border-emerald-500/50 bg-emerald-900/20 hover:border-emerald-400 hover:bg-emerald-900/30'
                          : 'border-slate-600 bg-slate-900/60 hover:border-emerald-400'
                    } hover:text-emerald-200`}
                  >
                    {preset.badge && (
                      <span
                        className={`absolute -top-2 -right-2 px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide ${
                          shouldHighlight
                            ? 'bg-emerald-400 text-slate-950'
                            : 'bg-emerald-500 text-slate-950'
                        }`}
                      >
                        {shouldHighlight ? '✨ Start Here' : preset.badge}
                      </span>
                    )}
                    <span className="text-lg" role="img" aria-hidden="true">
                      {preset.icon}
                    </span>
                    <div className="text-left">
                      <p className="font-semibold">{preset.label}</p>
                      <p className="text-xs text-slate-400">{preset.description}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </section>

          {/* Show/Hide Advanced Options Toggle */}
          <button
            type="button"
            onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition"
          >
            <span
              className={`transform transition-transform ${showAdvancedOptions ? 'rotate-90' : ''}`}
            >
              ▶
            </span>
            {showAdvancedOptions ? 'Hide advanced options' : 'Show advanced options'}
            {!showAdvancedOptions && (
              <span className="text-xs text-slate-500">(scenarios, manual setup, AI training)</span>
            )}
          </button>

          {showAdvancedOptions && (
            <>
              {/* Load Scenario section */}
              <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Scenarios</p>
                    <h2 className="text-lg font-semibold text-white">Load a saved scenario</h2>
                  </div>
                </div>
                <p className="text-sm text-slate-400 mb-3">
                  Load test vectors, curated learning scenarios, or your own saved game states.
                </p>
                <button
                  type="button"
                  onClick={() => setShowScenarioPicker(true)}
                  className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
                >
                  Browse Scenarios
                </button>
              </section>

              {/* Self-Play Games section (developer tools only) */}
              {developerToolsEnabled && (
                <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-slate-400">AI Training</p>
                      <h2 className="text-lg font-semibold text-white">Browse self-play games</h2>
                    </div>
                  </div>
                  <p className="text-sm text-slate-400 mb-3">
                    Load and replay games recorded during CMA-ES training, self-play soaks, and
                    other AI training activities.
                  </p>
                  <button
                    type="button"
                    onClick={() => setShowSelfPlayBrowser(true)}
                    className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition text-sm font-medium"
                  >
                    Browse Self-Play Games
                  </button>
                </section>
              )}

              <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
                <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-6 text-slate-100 shadow-lg">
                  {backendSandboxError && (
                    <div className="p-3 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded-lg">
                      {backendSandboxError}
                    </div>
                  )}

                  <div className="space-y-3">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div>
                        <p className="text-xs uppercase tracking-wide text-slate-400">Players</p>
                        <h2 className="text-lg font-semibold text-white">Seats & control</h2>
                      </div>
                      <div className="flex gap-2 text-xs">
                        {[2, 3, 4].map((count) => (
                          <button
                            key={count}
                            type="button"
                            onClick={() => handleSetupChange({ numPlayers: count })}
                            className={`px-2 py-1 rounded-full border ${
                              config.numPlayers === count
                                ? 'border-emerald-400 text-emerald-200 bg-emerald-900/30'
                                : 'border-slate-600 text-slate-300 hover:border-slate-400'
                            }`}
                          >
                            {count} Players
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-3">
                      {Array.from({ length: config.numPlayers }, (_, i) => {
                        const type = config.playerTypes[i];
                        const meta = PLAYER_TYPE_META[type];
                        return (
                          <div
                            key={i}
                            className={`rounded-xl border bg-slate-900/60 px-4 py-3 flex items-center justify-between gap-4 ${meta.accent}`}
                          >
                            <div>
                              <p className="text-sm font-semibold text-white">Player {i + 1}</p>
                              <p className="text-xs text-slate-300">{meta.description}</p>
                            </div>
                            <div className="flex gap-2">
                              {(['human', 'ai'] as LocalPlayerType[]).map((candidate) => {
                                const isActive = type === candidate;
                                return (
                                  <button
                                    key={candidate}
                                    type="button"
                                    onClick={() => handlePlayerTypeChange(i, candidate)}
                                    className={`px-3 py-1 rounded-full border text-xs font-semibold transition ${
                                      isActive
                                        ? 'border-white/80 text-white bg-white/10'
                                        : 'border-slate-600 text-slate-300 hover:border-slate-400'
                                    }`}
                                  >
                                    {PLAYER_TYPE_META[candidate].label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="flex flex-wrap gap-2 text-xs">
                      <button
                        type="button"
                        onClick={() => setAllPlayerTypes('human')}
                        className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition"
                      >
                        All Human
                      </button>
                      <button
                        type="button"
                        onClick={() => setAllPlayerTypes('ai')}
                        className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition"
                      >
                        All AI
                      </button>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div>
                        <p className="text-xs uppercase tracking-wide text-slate-400">Board</p>
                        <h2 className="text-lg font-semibold text-white">Choose a layout</h2>
                      </div>
                    </div>

                    <div className="grid gap-3 sm:grid-cols-2">
                      {BOARD_PRESETS.map((preset) => {
                        const isSelected = preset.value === config.boardType;
                        return (
                          <button
                            key={preset.value}
                            type="button"
                            onClick={() => handleSetupChange({ boardType: preset.value })}
                            className={`p-4 text-left rounded-2xl border transition shadow-sm ${
                              isSelected
                                ? 'border-emerald-400 bg-emerald-900/20 text-white'
                                : 'border-slate-600 bg-slate-900/60 text-slate-200 hover:border-slate-400'
                            }`}
                          >
                            <span className="text-xs uppercase tracking-wide text-slate-400">
                              {preset.subtitle}
                            </span>
                            <p className="text-lg font-semibold">{preset.label}</p>
                            <p className="text-xs text-slate-300 mt-1">{preset.blurb}</p>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 text-slate-100 shadow-lg space-y-4">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Summary</p>
                    <h2 className="text-xl font-bold text-white">{selectedBoardPreset.label}</h2>
                    <p className="text-sm text-slate-300">{selectedBoardPreset.blurb}</p>
                  </div>

                  <div className="space-y-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Humans</span>
                      <span className="font-semibold">{setupHumanSeatCount}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">AI opponents</span>
                      <span className="font-semibold">{setupAiSeatCount}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-300">Total seats</span>
                      <span className="font-semibold">{config.numPlayers}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <p className="text-xs text-slate-400">
                      We first attempt to stand up a backend game with these settings. If that
                      fails, we fall back to a purely client-local sandbox so you can still test
                      moves offline.
                    </p>
                    <button
                      type="button"
                      onClick={handleStartLocalGame}
                      className="w-full px-4 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white shadow-lg shadow-emerald-900/40 transition"
                    >
                      Launch Game
                    </button>
                  </div>
                </div>
              </section>
            </>
          )}
        </div>
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
      <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-3 sm:space-y-4">
        {/* Screen reader live region for game announcements (accessibility parity with BackendGameHost) */}
        <ScreenReaderAnnouncer
          queue={announcementQueue}
          onAnnouncementSpoken={removeAnnouncement}
        />

        {sandboxStallWarning && (
          <div className="p-3 rounded-xl border border-amber-500/70 bg-amber-900/40 text-amber-100 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <span>{sandboxStallWarning}</span>
            <div className="flex gap-2">
              {developerToolsEnabled && (
                <button
                  type="button"
                  onClick={handleCopySandboxTrace}
                  className="px-3 py-1 rounded-lg border border-amber-300 bg-amber-800/70 text-[11px] font-semibold hover:border-amber-100 hover:bg-amber-700/80"
                >
                  Copy AI trace
                </button>
              )}
              <button
                type="button"
                onClick={() => setSandboxStallWarning(null)}
                className="px-2 py-1 rounded-lg border border-slate-500 text-[11px] hover:border-slate-300"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {sandboxGameOverBannerText && (
          <div className="p-3 rounded-xl border border-emerald-500/70 bg-emerald-900/40 text-emerald-100 text-xs">
            {sandboxGameOverBannerText}
          </div>
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
            onReturnToLobby={() => {
              resetSandboxEngine();
              setSelected(undefined);
              setValidTargets([]);
              setBackendSandboxError(null);
              setSandboxPendingChoice(null);
              setIsSandboxVictoryModalDismissed(false);
              setLastLoadedScenario(null);
            }}
            onRematch={() => {
              // Reset state and start a new game with the same configuration.
              // Rematches are treated as generic sandbox games rather than
              // curated teaching scenarios, so we clear any scenario context.
              setIsSandboxVictoryModalDismissed(false);
              setSelected(undefined);
              setValidTargets([]);
              setSandboxPendingChoice(null);
              setSandboxCaptureChoice(null);
              setSandboxCaptureTargets([]);
              setSandboxStallWarning(null);
              setSandboxLastProgressAt(null);
              setLastLoadedScenario(null);

              // Re-initialize with the same config
              const interactionHandler = createSandboxInteractionHandler(
                config.playerTypes.slice(0, config.numPlayers)
              );
              const engine = initLocalSandboxEngine({
                boardType: config.boardType,
                numPlayers: config.numPlayers,
                playerTypes: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
                interactionHandler,
              });

              // If the first player is AI, start the AI turn loop
              if (engine) {
                const state = engine.getGameState();
                const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
                if (current && current.type === 'ai') {
                  maybeRunSandboxAiIfNeeded();
                }
              }

              toast.success('New game started with the same settings!');
            }}
          />
        )}

        <ChoiceDialog
          choice={
            sandboxPendingChoice &&
            sandboxPendingChoice.type !== 'ring_elimination' &&
            sandboxPendingChoice.type !== 'region_order'
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

        <main className="flex flex-col lg:flex-row lg:gap-8 gap-4">
          {/* Board container - centers on mobile, takes available space on desktop */}
          <section className="flex-shrink-0 flex justify-center lg:justify-start">
            {sandboxBoardState && (
              <div className="inline-block space-y-3">
                <div className="p-3 sm:p-4 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg">
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
                      <label className="inline-flex items-center gap-2 text-[11px] text-slate-400 select-none">
                        <input
                          type="checkbox"
                          className="rounded border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500"
                          checked={developerToolsEnabled}
                          onChange={(e) => setDeveloperToolsEnabled(e.target.checked)}
                        />
                        <span>Developer Tools</span>
                      </label>
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => setShowSaveStateDialog(true)}
                          className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
                        >
                          Save State
                        </button>
                        {developerToolsEnabled && (
                          <>
                            <button
                              type="button"
                              onClick={handleExportScenarioJson}
                              className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
                            >
                              Export Scenario JSON
                            </button>
                            <button
                              type="button"
                              onClick={handleCopySandboxFixture}
                              className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                            >
                              Copy Test Fixture
                            </button>
                            {sandboxGameEndExplanation && (
                              <details
                                className="absolute top-full right-0 mt-2 w-96 p-4 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 text-xs font-mono overflow-auto max-h-96"
                                data-testid="sandbox-game-end-explanation-debug"
                              >
                                <summary className="cursor-pointer text-slate-400 hover:text-slate-200 mb-2">
                                  Debug: GameEndExplanation
                                </summary>
                                <pre className="whitespace-pre-wrap text-emerald-300">
                                  {JSON.stringify(sandboxGameEndExplanation, null, 2)}
                                </pre>
                              </details>
                            )}
                          </>
                        )}
                        <button
                          type="button"
                          onClick={() => setShowScenarioPicker(true)}
                          className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-amber-400 hover:text-amber-200 transition"
                        >
                          Load Scenario
                        </button>
                        {lastLoadedScenario && (
                          <button
                            type="button"
                            onClick={handleResetScenario}
                            className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                          >
                            Reset Scenario
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => {
                            resetSandboxEngine();
                            setSelected(undefined);
                            setValidTargets([]);
                            setBackendSandboxError(null);
                            setSandboxPendingChoice(null);
                            setSandboxStallWarning(null);
                            setSandboxLastProgressAt(null);
                            setIsSandboxVictoryModalDismissed(false);
                          }}
                          className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
                        >
                          Change Setup
                        </button>
                        <button
                          type="button"
                          aria-label="Show board controls"
                          data-testid="board-controls-button"
                          onClick={() => setShowBoardControls(true)}
                          className="h-8 w-8 rounded-full border border-slate-600 text-[11px] leading-none text-slate-200 hover:bg-slate-800/80"
                        >
                          ?
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {isInReplayMode && (
                  <div className="mb-2 p-2 rounded-lg bg-emerald-900/40 border border-emerald-700/50 text-xs text-emerald-200 flex items-center justify-between">
                    <span>Viewing replay - board is read-only</span>
                    <span className="text-emerald-400/70">Use playback controls in sidebar</span>
                  </div>
                )}

                {sandboxBoardState && sandboxBoardViewModel && (
                  <BoardView
                    boardType={sandboxBoardState.type}
                    board={sandboxBoardState}
                    viewModel={sandboxBoardViewModel}
                    selectedPosition={isInReplayMode ? undefined : selected}
                    validTargets={isInReplayMode ? [] : displayedValidTargets}
                    onCellClick={isInReplayMode ? undefined : (pos) => handleSandboxCellClick(pos)}
                    onCellDoubleClick={
                      isInReplayMode ? undefined : (pos) => handleSandboxCellDoubleClick(pos)
                    }
                    onCellContextMenu={
                      isInReplayMode ? undefined : (pos) => handleSandboxCellContextMenu(pos)
                    }
                    showMovementGrid={showMovementGrid}
                    showCoordinateLabels={
                      sandboxBoardState.type === 'square8' || sandboxBoardState.type === 'square19'
                    }
                    squareRankFromBottom={
                      sandboxBoardState.type === 'square8' || sandboxBoardState.type === 'square19'
                        ? true
                        : false
                    }
                    showLineOverlays={showLineOverlays}
                    showTerritoryRegionOverlays={showTerritoryOverlays}
                    pendingAnimation={
                      isInReplayMode
                        ? (replayAnimation ?? undefined)
                        : (pendingAnimation ?? undefined)
                    }
                    onAnimationComplete={
                      isInReplayMode ? () => setReplayAnimation(null) : clearAnimation
                    }
                    chainCapturePath={isInReplayMode ? undefined : chainCapturePath}
                    shakingCellKey={isInReplayMode ? null : sandboxShakingCellKey}
                  />
                )}

                <section className="mt-1 p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-xs text-slate-200">
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
                      Phase: {sandboxPhaseDetails.label}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {sandboxPlayersList.map((player) => {
                      const typeKey = player.type === 'ai' ? 'ai' : 'human';
                      const meta = PLAYER_TYPE_META[typeKey as LocalPlayerType];
                      const isCurrent = player.playerNumber === sandboxCurrentPlayerNumber;
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
                <VictoryConditionsPanel className="mt-3" />
              </div>
            )}
          </section>

          <aside className="w-full lg:w-80 flex-shrink-0 space-y-3 sm:space-y-4 text-sm text-slate-100">
            {/* Replay Panel - Game Database Browser */}
            <ReplayPanel
              onStateChange={setReplayState}
              onReplayModeChange={setIsInReplayMode}
              onForkFromPosition={handleForkFromReplay}
              onAnimationChange={setReplayAnimation}
              defaultCollapsed={true}
            />

            {/* Unified Game HUD - full HUD on desktop, compact HUD on mobile */}
            {sandboxHudVM &&
              (isMobile ? (
                <MobileGameHUD
                  isLocalSandboxOnly={!user}
                  viewModel={sandboxHudVM}
                  onShowBoardControls={() => setShowBoardControls(true)}
                  rulesUxContext={{
                    boardType: boardTypeValue,
                    numPlayers: sandboxPlayersList.length,
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
                  isLocalSandboxOnly={!user}
                  viewModel={sandboxHudVM}
                  onShowBoardControls={() => setShowBoardControls(true)}
                  hideVictoryConditions={true}
                  rulesUxContext={{
                    boardType: boardTypeValue,
                    numPlayers: sandboxPlayersList.length,
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

            {/* Onboarding scenario context for curated rules/FAQ scenarios */}
            {lastLoadedScenario &&
              lastLoadedScenario.onboarding &&
              lastLoadedScenario.rulesSnippet && (
                <div className="p-4 border border-emerald-700 rounded-2xl bg-emerald-950/60 space-y-2">
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

            {sandboxEngine &&
              sandboxGameState &&
              sandboxGameState.gameStatus === 'active' &&
              sandboxGameState.players.length === 2 &&
              sandboxGameState.rulesOptions?.swapRuleEnabled === true &&
              sandboxEngine.canCurrentPlayerSwapSides() && (
                <div className="p-3 border border-amber-500/60 rounded-2xl bg-amber-900/40 text-xs space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-amber-100">
                      Pie rule available: swap colours with Player 1.
                    </span>
                    <button
                      type="button"
                      className="px-2 py-1 rounded bg-amber-500 hover:bg-amber-400 text-black font-semibold"
                      onClick={() => {
                        sandboxEngine.applySwapSidesForCurrentPlayer();
                        setSelected(undefined);
                        setValidTargets([]);
                        setSandboxPendingChoice(null);
                        setSandboxStateVersion((v) => v + 1);
                      }}
                    >
                      Swap colours
                    </button>
                  </div>
                  <p className="text-amber-100/80">
                    As Player 2, you may use this once, immediately after Player 1’s first turn.
                  </p>
                </div>
              )}

            {/* History Playback Controls - visible when there's move history.
                When no internal snapshots are available (e.g. imported self-play
                snapshots), we render the panel in a disabled state with a hint
                instead of allowing a no-op slider. */}
            {!isInReplayMode && sandboxGameState && sandboxGameState.moveHistory.length > 0 && (
              <HistoryPlaybackPanel
                totalMoves={sandboxGameState.moveHistory.length}
                currentMoveIndex={
                  isViewingHistory ? historyViewIndex : sandboxGameState.moveHistory.length
                }
                isViewingHistory={isViewingHistory}
                onMoveIndexChange={(index) => {
                  setHistoryViewIndex(index);
                  // If at end, exit history view
                  if (index >= sandboxGameState.moveHistory.length) {
                    setIsViewingHistory(false);
                  }
                }}
                onExitHistoryView={() => {
                  setIsViewingHistory(false);
                  setHistoryViewIndex(sandboxGameState.moveHistory.length);
                }}
                onEnterHistoryView={() => {
                  setIsViewingHistory(true);
                }}
                // New: pass a flag so the panel can disable scrubbing for
                // scenarios without snapshots (self-play snapshots, some fixtures).
                hasSnapshots={hasHistorySnapshots}
              />
            )}

            {/* Move History - compact notation display */}
            {!isInReplayMode && sandboxGameState && (
              <MoveHistory
                moves={sandboxGameState.moveHistory}
                boardType={sandboxGameState.boardType}
                currentMoveIndex={
                  isViewingHistory ? historyViewIndex - 1 : sandboxGameState.moveHistory.length - 1
                }
                onMoveClick={(index) => {
                  // Jump to the state after this move (index+1)
                  setIsViewingHistory(true);
                  setHistoryViewIndex(index + 1);
                }}
                notationOptions={((): MoveNotationOptions | undefined => {
                  if (
                    sandboxGameState.boardType === 'square8' ||
                    sandboxGameState.boardType === 'square19'
                  ) {
                    const size = sandboxBoardState?.size ?? 0;
                    return {
                      boardType: sandboxGameState.boardType,
                      boardSizeOverride: size > 0 ? size : undefined,
                      squareRankFromBottom: true,
                    };
                  }
                  return undefined;
                })()}
              />
            )}

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60">
              <GameEventLog viewModel={sandboxEventLogViewModel} />
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
                  onClick={() => setAutoSaveGames(!autoSaveGames)}
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

            {developerToolsEnabled && (
              <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="font-semibold">AI Evaluation (sandbox)</h2>
                  <button
                    type="button"
                    onClick={requestSandboxEvaluation}
                    disabled={!sandboxEngine || !sandboxGameState || isSandboxAnalysisRunning}
                    className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 disabled:opacity-60 disabled:cursor-not-allowed transition"
                  >
                    {isSandboxAnalysisRunning ? 'Evaluating…' : 'Request evaluation'}
                  </button>
                </div>
                <EvaluationPanel
                  evaluationHistory={sandboxEvaluationHistory}
                  players={sandboxGameState?.players ?? []}
                />
                {sandboxEvaluationError && (
                  <p className="text-xs text-amber-400" data-testid="sandbox-evaluation-error">
                    {sandboxEvaluationError}
                  </p>
                )}
              </div>
            )}

            <SandboxTouchControlsPanel
              selectedPosition={selected}
              selectedStackDetails={selectedStackDetails}
              validTargets={primaryValidTargets}
              isCaptureDirectionPending={!!sandboxCaptureChoice}
              captureTargets={sandboxCaptureTargets}
              // Multi-segment capture undo is not yet exposed by the sandbox
              // engine; this remains a no-op until the underlying rules
              // pipeline supports segment-level rewind.
              canUndoSegment={false}
              onClearSelection={() => {
                clearSandboxSelection();
              }}
              onUndoSegment={() => {
                // no-op for now
              }}
              // For now, treat "Finish move" as an explicit selection reset that
              // clears highlights without issuing additional engine actions.
              onApplyMove={() => {
                clearSandboxSelection();
              }}
              showMovementGrid={showMovementGrid}
              onToggleMovementGrid={(next) => setShowMovementGrid(next)}
              showValidTargets={showValidTargetsOverlay}
              onToggleValidTargets={(next) => setShowValidTargetsOverlay(next)}
              showLineOverlays={showLineOverlays}
              onToggleLineOverlays={
                developerToolsEnabled ? (next) => setShowLineOverlays(next) : undefined
              }
              showTerritoryOverlays={showTerritoryOverlays}
              onToggleTerritoryOverlays={
                developerToolsEnabled ? (next) => setShowTerritoryOverlays(next) : undefined
              }
              phaseLabel={sandboxPhaseDetails.label}
              phaseHint={sandboxTouchPhaseHint}
              canSkipTerritoryProcessing={canSkipTerritoryProcessing}
              onSkipTerritoryProcessing={() => {
                if (!sandboxPendingChoice || sandboxPendingChoice.type !== 'region_order') {
                  return;
                }
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
              canSkipCapture={canSkipCaptureForTouch}
              onSkipCapture={
                canSkipCaptureForTouch && sandboxEngine && sandboxGameState
                  ? async () => {
                      const moves = sandboxEngine.getValidMoves(sandboxGameState.currentPlayer);
                      const skipMove = moves.find((m) => m.type === 'skip_capture');
                      if (!skipMove) {
                        return;
                      }
                      await sandboxEngine.applyCanonicalMove(skipMove as any);
                      setSandboxStateVersion((v) => v + 1);
                      clearSandboxSelection();
                      maybeRunSandboxAiIfNeeded();
                    }
                  : undefined
              }
              autoSaveGames={autoSaveGames}
              onToggleAutoSave={(next) => setAutoSaveGames(next)}
              gameSaveStatus={gameSaveStatus}
            />

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
              <h2 className="font-semibold">Phase Guide</h2>
              <p className="text-xs uppercase tracking-wide text-slate-400">
                {sandboxHudVM?.phase.label ?? 'Initializing...'}
              </p>
              <p className="text-sm text-slate-200">
                {sandboxHudVM?.phase.description ?? 'Setting up game state.'}
              </p>
              <p className="text-xs text-slate-400">
                Complete the current requirement to advance the turn (chain captures, line rewards,
                etc.).
              </p>
            </div>

            <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
              <h2 className="font-semibold">Sandbox Notes</h2>
              <ul className="list-disc list-inside text-slate-300 space-y-1 text-xs">
                {sandboxModeNotes.map((note, idx) => (
                  <li key={idx}>{note}</li>
                ))}
              </ul>
            </div>
          </aside>
        </main>

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
          isOpen={showSaveStateDialog}
          onClose={() => setShowSaveStateDialog(false)}
          gameState={sandboxGameState}
          onSaved={(scenario) => {
            toast.success(`Saved state: ${scenario.name}`);
          }}
        />
      </div>
    </div>
  );
};
