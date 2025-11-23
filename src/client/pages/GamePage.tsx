import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameHUD } from '../components/GameHUD';
import { GameEventLog } from '../components/GameEventLog';
import { LocalSandboxState, handleLocalSandboxCellClick } from '../sandbox/localSandboxController';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../sandbox/ClientSandboxEngine';
import {
  BoardState,
  BoardType,
  GameState,
  GameResult,
  Position,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
  positionsEqual,
  CreateGameRequest,
} from '../../shared/types/game';
import { toast } from 'react-hot-toast';
import { useGame, ConnectionStatus } from '../contexts/GameContext';
import { useAuth } from '../contexts/AuthContext';
import { gameApi } from '../services/api';
import { isSandboxAiStallDiagnosticsEnabled } from '../../shared/utils/envFlags';

type LocalPlayerType = 'human' | 'ai';

interface LocalConfig {
  numPlayers: number;
  boardType: BoardType;
  playerTypes: LocalPlayerType[]; // indexed 0..3 for players 1..4
}

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
    summary: 'Place fresh stacks or build existing ones while keeping a legal move available.',
  },
  movement: {
    label: 'Movement',
    summary:
      'Pick a stack and travel a distance equal to its height, respecting board blocking rules.',
  },
  capture: {
    label: 'Capture',
    summary: 'Chain overtaking captures until no follow-up exists or a choice resolves.',
  },
  line_processing: {
    label: 'Line Processing',
    summary: 'Resolve completed lines and apply marker collapses/reward decisions.',
  },
  territory_processing: {
    label: 'Territory Processing',
    summary:
      'Evaluate disconnected regions, collapsing captured territory and enforcing self-elimination.',
  },
};

function getGameOverBannerText(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Game over – victory by ring elimination.';
    case 'territory_control':
      return 'Game over – victory by territory control.';
    case 'last_player_standing':
      return 'Game over – last player standing.';
    case 'timeout':
      return 'Game over – victory on time.';
    case 'resignation':
      return 'Game over – victory by resignation.';
    case 'abandonment':
      return 'Game over – game abandoned.';
    case 'draw':
      return 'Game over – draw.';
    default:
      return 'Game over.';
  }
}

/**
 * Get friendly display name for AI difficulty level with description
 */
function getAIDifficultyLabel(difficulty: number): { label: string; color: string } {
  if (difficulty <= 2) return { label: 'Beginner', color: 'text-green-400' };
  if (difficulty <= 5) return { label: 'Intermediate', color: 'text-blue-400' };
  if (difficulty <= 8) return { label: 'Advanced', color: 'text-purple-400' };
  return { label: 'Expert', color: 'text-red-400' };
}

function renderGameHeader(gameState: GameState) {
  const playerSummary = gameState.players
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map((p) => {
      if (p.type === 'ai') {
        const difficulty = p.aiProfile?.difficulty ?? p.aiDifficulty ?? 5;
        const aiType = p.aiProfile?.aiType ?? 'heuristic';
        const diffLabel = getAIDifficultyLabel(difficulty);
        return `${p.username || `AI-${p.playerNumber}`} (AI ${diffLabel.label} Lv${difficulty})`;
      }
      return `${p.username || `P${p.playerNumber}`} (Human)`;
    })
    .join(', ');

  return (
    <>
      <h1 className="text-3xl font-bold mb-1">Game</h1>
      <p className="text-sm text-gray-500">
        Game ID: {gameState.id} • Board: {gameState.boardType} • Players: {playerSummary}
      </p>
    </>
  );
}

export default function GamePage() {
  const navigate = useNavigate();
  const params = useParams<{ gameId?: string }>();
  const routeGameId = params.gameId;

  // Backend game context (used when viewing a real server game)
  const {
    gameId,
    gameState,
    validMoves,
    isConnecting,
    error,
    victoryState,
    connectToGame,
    disconnect,
    pendingChoice,
    choiceDeadline,
    respondToChoice,
    submitMove,
    sendChatMessage,
    chatMessages: backendChatMessages,
    connectionStatus,
    lastHeartbeatAt,
  } = useGame();

  const { user } = useAuth();

  // Derived state for HUD
  const currentPlayer = gameState?.players.find((p) => p.playerNumber === gameState.currentPlayer);
  const isPlayer = !!gameState?.players.some((p) => p.id === user?.id);
  const isMyTurn = currentPlayer?.id === user?.id;
  const isConnectionActive = connectionStatus === 'connected';
  const boardInteractionDisabled = !isConnectionActive || !isPlayer || !!victoryState;
  const boardInteractionMessage = (() => {
    if (!isPlayer) {
      return 'Moves disabled while spectating.';
    }
    if (!isConnectionActive) {
      return connectionStatus === 'reconnecting' || connectionStatus === 'connecting'
        ? 'Reconnecting to server…'
        : 'Disconnected from server.';
    }
    if (victoryState) {
      return 'Game completed.';
    }
    return null;
  })();

  const getInstruction = () => {
    if (!gameState || !currentPlayer) return undefined;
    if (!isPlayer)
      return `Spectating: ${currentPlayer.username || `Player ${currentPlayer.playerNumber}`}'s turn`;
    if (!isMyTurn)
      return `Waiting for ${currentPlayer.username || `Player ${currentPlayer.playerNumber}`}...`;

    switch (gameState.currentPhase) {
      case 'ring_placement':
        return 'Place a ring on an empty edge space.';
      case 'movement':
        return 'Select a stack to move.';
      case 'capture':
        return 'Select a stack to capture with.';
      case 'line_processing':
        return 'Choose a line to collapse.';
      case 'territory_processing':
        return 'Choose a region to claim.';
      default:
        return 'Make your move.';
    }
  };

  // Choice/phase diagnostics
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [showSystemEventsInLog, setShowSystemEventsInLog] = useState(true);
  const [fatalGameError, setFatalGameError] = useState<{
    message: string;
    technical?: string;
  } | null>(null);
  const [isVictoryModalDismissed, setIsVictoryModalDismissed] = useState(false);
  const [isSandboxVictoryModalDismissed, setIsSandboxVictoryModalDismissed] = useState(false);
  // Use backend chat messages if available, otherwise local state (for sandbox/fallback)
  const [localChatMessages, setLocalChatMessages] = useState<{ sender: string; text: string }[]>(
    []
  );
  const chatMessages = backendChatMessages || localChatMessages;

  const [chatInput, setChatInput] = useState('');
  const [choiceTimeRemainingMs, setChoiceTimeRemainingMs] = useState<number | null>(null);

  const choiceTimerRef = useRef<number | null>(null);
  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);
  const lastConnectionStatusRef = useRef<ConnectionStatus | null>(null);

  // Local sandbox render tick used to force re-renders for AI-vs-AI games
  // even when React state derived from GameState hasn’t otherwise changed.
  const [, setSandboxTurn] = useState(0);

  // Local setup state (used only when no gameId route param is provided)
  const [config, setConfig] = useState<LocalConfig>({
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai'],
  });
  const [isConfigured, setIsConfigured] = useState(false);
  const [backendSandboxError, setBackendSandboxError] = useState<string | null>(null);

  // Local-only sandbox state (legacy; retained for now as a fallback)
  const [localSandbox, setLocalSandbox] = useState<LocalSandboxState | null>(null);

  // Client-local sandbox engine (Stage 2 harness). When defined, this is the
  // source of truth for sandbox GameState. We keep it in a ref so methods are
  // stable across renders.
  const sandboxEngineRef = useRef<ClientSandboxEngine | null>(null);

  // Sandbox PlayerChoice state (used only in local sandbox mode). This mirrors
  // the backend pendingChoice flow but remains fully client-local.
  const [sandboxPendingChoice, setSandboxPendingChoice] = useState<PlayerChoice | null>(null);
  const sandboxChoiceResolverRef = useRef<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >(null);
  const [sandboxCaptureChoice, setSandboxCaptureChoice] = useState<PlayerChoice | null>(null);
  const [sandboxCaptureTargets, setSandboxCaptureTargets] = useState<Position[]>([]);

  // UI selection state (used in both modes)
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  // Sandbox stall/watchdog diagnostics for local AI games.
  const [sandboxLastProgressAt, setSandboxLastProgressAt] = useState<number | null>(null);
  const [sandboxStallWarning, setSandboxStallWarning] = useState<string | null>(null);
  const sandboxDiagnosticsEnabled = isSandboxAiStallDiagnosticsEnabled();

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
          const options = (choice as any).options as TChoice['options'];
          const optionsArray = (options as any[]) ?? [];
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

        // Human players
        if (choice.type === 'capture_direction') {
          // For capture_direction in the local sandbox, highlight the
          // possible landing squares on the board and let the user
          // choose by clicking one of them instead of showing a dialog.
          const anyChoice = choice as any;
          const options = (anyChoice.options ?? []) as any[];
          const targets: Position[] = options.map((opt) => opt.landingPosition as Position);
          setSandboxCaptureChoice(choice);
          setSandboxCaptureTargets(targets);
        } else {
          // Other choices (e.g. region_order) continue to use the
          // dialog-based ChoiceDialog UI.
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

  // When a :gameId is present in the route, connect to that backend game
  useEffect(() => {
    if (!routeGameId) {
      // Ensure any previous connection is torn down when leaving a game
      disconnect();
      setFatalGameError(null);
      return;
    }

    connectToGame(routeGameId);

    return () => {
      disconnect();
    };
  }, [routeGameId, connectToGame, disconnect]);

  // Listen for game_error events from the server
  useEffect(() => {
    if (!routeGameId) return;

    const handleGameError = (data: any) => {
      if (data && data.data) {
        setFatalGameError({
          message: data.data.message || 'An error occurred during the game.',
          technical: data.data.technical,
        });

        // Log technical details in development
        if (process.env.NODE_ENV === 'development' && data.data.technical) {
          console.error('[Game Error]', data.data.technical);
        }
      }
    };

    // Access socket through the game context
    // Note: This assumes the socket is exposed, or we use the error from context
    // For now, we'll rely on the error state from useGame context

    return () => {
      // Cleanup if needed
    };
  }, [routeGameId]);

  // Reset backend victory modal dismissal whenever the active game or victory state changes
  useEffect(() => {
    setIsVictoryModalDismissed(false);
  }, [routeGameId, victoryState]);

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig((prev) => ({
      ...prev,
      ...partial,
      playerTypes: partial.numPlayers
        ? prev.playerTypes.map((t, idx) => (idx < partial.numPlayers! ? t : prev.playerTypes[idx]))
        : prev.playerTypes,
    }));
  };

  const handlePlayerTypeChange = (index: number, type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      next[index] = type;
      return { ...prev, playerTypes: next };
    });
  };

  const setAllPlayerTypes = (type: LocalPlayerType) => {
    setConfig((prev) => {
      const next = [...prev.playerTypes];
      for (let i = 0; i < prev.numPlayers; i += 1) {
        next[i] = type;
      }
      return { ...prev, playerTypes: next };
    });
  };

  const handleStartLocalGame = async () => {
    const nextBoardType = config.boardType;

    // First, attempt to create a real backend game using the same
    // CreateGameRequest shape as the lobby. This keeps the sandbox
    // harness aligned with the server GameEngine and WebSocket layer
    // without duplicating rules client-side. If creation fails (e.g.
    // unauthenticated user, server down), we fall back to the
    // local-only board used previously.
    try {
      const payload: CreateGameRequest = {
        boardType: nextBoardType,
        maxPlayers: config.numPlayers,
        isRated: false,
        isPrivate: true,
        timeControl: {
          type: 'rapid',
          initialTime: 600,
          increment: 0,
        },
        // For now, derive a simple AI configuration from local
        // player types: any non-human seats become AI opponents
        // with a uniform difficulty. This keeps the harness
        // loosely in sync with LobbyPage without duplicating
        // its full form.
        aiOpponents: (() => {
          const aiSeats = config.playerTypes
            .slice(0, config.numPlayers)
            .filter((t) => t === 'ai').length;
          if (aiSeats <= 0) return undefined;
          return {
            count: aiSeats,
            difficulty: Array(aiSeats).fill(5),
            mode: 'service',
            aiType: 'heuristic',
          };
        })(),
      };

      const game = await gameApi.createGame(payload);
      // On success, immediately navigate into the real backend
      // game route so the sandbox uses the full GameEngine +
      // WebSocket + PlayerChoice/AI stack.
      navigate(`/game/${game.id}`);
      return;
    } catch (err) {
      console.error('Failed to create backend sandbox game, falling back to local-only board', err);
      setBackendSandboxError(
        'Backend sandbox game could not be created; falling back to local-only board only.'
      );
    }

    // Fallback: when backend game creation is unavailable, switch to a
    // client-local sandbox engine. This keeps the sandbox usable for quick
    // experiments while aligning it with the shared GameState model and
    // PlayerChoice semantics used by the backend GameEngine.
    const sandboxConfig: SandboxConfig = {
      boardType: nextBoardType,
      numPlayers: config.numPlayers,
      playerKinds: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[],
    };

    const interactionHandler = createSandboxInteractionHandler(
      config.playerTypes.slice(0, config.numPlayers)
    );

    sandboxEngineRef.current = new ClientSandboxEngine({
      config: sandboxConfig,
      interactionHandler,
    });

    setLocalSandbox(null);
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsConfigured(true);

    // If the first player is an AI, immediately start the sandbox AI turn
    // loop so AI-vs-AI games progress without any human clicks.
    const engine = sandboxEngineRef.current;
    if (engine) {
      const state = engine.getGameState();
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (current && current.type === 'ai') {
        void runSandboxAiTurnLoop();
      }
    }
  };

  const runSandboxAiTurnLoop = async () => {
    const engine = sandboxEngineRef.current;
    if (!engine) return;

    let safetyCounter = 0;
    // Allow a bounded number of consecutive AI turns per batch to avoid
    // accidental infinite loops, but drive progression one visible move at a
    // time so AI-vs-AI games feel continuous rather than "bursty".
    while (safetyCounter < 32) {
      const state = engine.getGameState();
      if (state.gameStatus !== 'active') break;
      const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
      if (!current || current.type !== 'ai') break;

      await engine.maybeRunAITurn();

      // After each AI move, clear any stale selection/highlights and bump the
      // sandboxTurn counter so BoardView re-renders with the latest state.
      setSelected(undefined);
      setValidTargets([]);
      setSandboxTurn((t) => t + 1);
      setSandboxLastProgressAt(Date.now());
      setSandboxStallWarning(null);

      safetyCounter += 1;

      // Small delay between moves so AI-only games progress in a smooth
      // sequence rather than a single visual burst of many moves.
      await new Promise((resolve) => window.setTimeout(resolve, 120));
    }

    // If the game is still active and the next player is an AI, schedule
    // another batch so AI-vs-AI games continue advancing without manual
    // clicks. The safety counter above still bounds each batch.
    const finalState = engine.getGameState();
    const next = finalState.players.find((p) => p.playerNumber === finalState.currentPlayer);
    if (finalState.gameStatus === 'active' && next && next.type === 'ai') {
      window.setTimeout(() => {
        void runSandboxAiTurnLoop();
      }, 200);
    }
  };

  const maybeRunSandboxAiIfNeeded = () => {
    const engine = sandboxEngineRef.current;
    if (!engine) return;

    const state = engine.getGameState();
    const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (state.gameStatus === 'active' && current && current.type === 'ai') {
      void runSandboxAiTurnLoop();
    }
  };

  const handleCopySandboxTrace = async () => {
    try {
      if (typeof window === 'undefined') {
        return;
      }

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

  // Unified sandbox click handler: prefer the ClientSandboxEngine when
  // available (Stage 2 harness), otherwise fall back to the legacy
  // LocalSandboxState controller.
  const handleSandboxCellClick = (pos: Position) => {
    // When a capture_direction choice is pending in the local sandbox,
    // interpret clicks as selecting one of the highlighted landing
    // squares instead of sending a normal click into the engine.
    if (sandboxCaptureChoice && sandboxCaptureChoice.type === 'capture_direction') {
      const currentChoice: any = sandboxCaptureChoice;
      const options: any[] = (currentChoice.options ?? []) as any[];
      const matching = options.find((opt) => positionsEqual(opt.landingPosition, pos));

      if (matching) {
        const resolver = sandboxChoiceResolverRef.current;
        if (resolver) {
          resolver({
            choiceId: currentChoice.id,
            playerNumber: currentChoice.playerNumber,
            choiceType: currentChoice.type,
            selectedOption: matching,
          } as PlayerChoiceResponseFor<PlayerChoice>);
        }
        sandboxChoiceResolverRef.current = null;
        setSandboxCaptureChoice(null);
        setSandboxCaptureTargets([]);

        // After resolving a capture_direction choice, the sandbox engine
        // continues the capture chain (possibly with additional automatic
        // segments). Bump sandboxTurn on the next tick so BoardView
        // re-reads the latest GameState once that chain has fully
        // resolved, and then trigger AI turns if the next player is an AI.
        window.setTimeout(() => {
          setSandboxTurn((t) => t + 1);
          maybeRunSandboxAiIfNeeded();
        }, 0);
      }
      // Ignore clicks that are not on a highlighted landing square.
      return;
    }

    const engine = sandboxEngineRef.current;
    if (engine) {
      const stateBefore = engine.getGameState();
      const current = stateBefore.players.find((p) => p.playerNumber === stateBefore.currentPlayer);

      // If it is currently an AI player's turn in the sandbox engine, ignore
      // human clicks and ensure the AI turn loop is running instead of placing
      // rings for the AI seat.
      if (stateBefore.gameStatus === 'active' && current && current.type === 'ai') {
        maybeRunSandboxAiIfNeeded();
        return;
      }

      // Ring-placement phase: a single click attempts a 1-ring placement
      // via the engine. On success, we immediately highlight the legal
      // movement targets for the newly placed/updated stack, and the
      // human must then move that stack; the AI will respond only after
      // the movement step completes.
      if (stateBefore.currentPhase === 'ring_placement') {
        void (async () => {
          const placed = await engine.tryPlaceRings(pos, 1);
          if (!placed) {
            return;
          }

          setSelected(pos);
          const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
          setValidTargets(targets);
          setSandboxTurn((t) => t + 1);
        })();
        return;
      }

      // Movement phase: mirror backend UX – first click selects a stack
      // and highlights its legal landing positions; second click on a
      // highlighted cell executes the move.
      if (!selected) {
        // Selection click: record selected cell and highlight valid targets.
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        // Inform the engine about the selection so its internal
        // movement state (_selectedStackKey) matches the UI.
        engine.handleHumanCellClick(pos);
        return;
      }

      // Clicking the same cell clears selection.
      if (positionsEqual(selected, pos)) {
        setSelected(undefined);
        setValidTargets([]);
        // Let the engine clear its internal selection as well.
        engine.clearSelection();
        return;
      }

      // If this click is on a highlighted target, treat it as executing
      // the move and then let the AI respond.
      const isTarget = validTargets.some((t) => positionsEqual(t, pos));
      if (isTarget) {
        engine.handleHumanCellClick(pos);
        setSelected(undefined);
        setValidTargets([]);
        setSandboxTurn((t) => t + 1);
        setSandboxStateVersion((v) => v + 1);
        return;
      }

      // Otherwise, ignore clicks on non-highlighted cells while a stack
      // is selected so that invalid landings cannot be executed. Users
      // can either click the selected stack again to clear selection, or
      // select a different stack by first clearing and then re-clicking.
      return;
    }

    if (!localSandbox) return;

    const next = handleLocalSandboxCellClick(localSandbox, pos);
    setLocalSandbox(next);
    setSelected(pos);
    setValidTargets([]); // movement/capture targets will be added in a later phase
  };

  /**
   * Sandbox double-click handler: implements the richer placement semantics
   * for the local sandbox during the ring_placement phase.
   *
   * - Empty cells: attempt a 2-ring placement (falling back to 1 ring if
   *   necessary) and then highlight movement targets from the new stack.
   * - Occupied cells: attempt a single-ring placement onto the stack and
   *   then highlight movement targets from that stack.
   */
  const handleSandboxCellDoubleClick = (pos: Position) => {
    const engine = sandboxEngineRef.current;
    if (!engine) return;

    const state = engine.getGameState();
    if (state.currentPhase !== 'ring_placement') {
      return;
    }

    const board = state.board;
    const key = positionToString(pos);
    const stack = board.stacks.get(key);
    const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (!player || player.ringsInHand <= 0) {
      return;
    }

    const isOccupied = !!stack && stack.rings.length > 0;
    const maxFromHand = player.ringsInHand;
    const maxPerPlacement = isOccupied ? 1 : maxFromHand;

    if (maxPerPlacement <= 0) {
      return;
    }

    void (async () => {
      let placed = false;

      if (!isOccupied) {
        // Empty cell: treat as a request to place 2 rings here in a single
        // placement action when possible.
        const desiredCount = Math.min(2, maxFromHand);
        placed = await engine.tryPlaceRings(pos, desiredCount);

        // If the desired multi-ring placement fails no-dead-placement checks,
        // fall back to a single-ring placement.
        if (!placed && desiredCount > 1) {
          placed = await engine.tryPlaceRings(pos, 1);
        }
      } else {
        // Existing stack: canonical rule is exactly 1 ring per placement.
        placed = await engine.tryPlaceRings(pos, 1);
      }

      if (!placed) {
        return;
      }

      // After a successful placement, we are now in the movement step for
      // this player, and the placed/updated stack must move. Highlight its
      // legal landing targets so the user can complete the turn.
      setSelected(pos);
      const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
      setValidTargets(targets);
      setSandboxTurn((t) => t + 1);
    })();
  };

  /**
   * Sandbox context-menu handler (right-click / long-press proxy): prompts
   * the user for a ring-count to place at the clicked position, then applies
   * that placement via tryPlaceRings when legal.
   */
  const handleSandboxCellContextMenu = (pos: Position) => {
    const engine = sandboxEngineRef.current;
    if (!engine) return;

    const state = engine.getGameState();
    if (state.currentPhase !== 'ring_placement') {
      return;
    }

    const board = state.board;
    const key = positionToString(pos);
    const stack = board.stacks.get(key);
    const player = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (!player || player.ringsInHand <= 0) {
      return;
    }

    const isOccupied = !!stack && stack.rings.length > 0;
    const maxFromHand = player.ringsInHand;
    const maxPerPlacement = isOccupied ? 1 : maxFromHand;

    if (maxPerPlacement <= 0) {
      return;
    }

    const promptLabel = isOccupied
      ? 'Place how many rings on this stack? (canonical: 1)'
      : `Place how many rings on this empty cell? (1–${maxPerPlacement})`;

    const raw = window.prompt(promptLabel, Math.min(2, maxPerPlacement).toString());
    if (!raw) {
      return;
    }

    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > maxPerPlacement) {
      return;
    }

    void (async () => {
      const placed = await engine.tryPlaceRings(pos, parsed);
      if (!placed) {
        return;
      }

      setSelected(pos);
      const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
      setValidTargets(targets);
      setSandboxTurn((t) => t + 1);
    })();
  };

  /**
   * Backend game click handling.
   *
   * Placement phase:
   *   - Single-click on an empty cell sends a place_ring with placementCount = 1
   *     if the backend reports such a move as legal.
   *   - Single-click on a stack simply selects it (no placement yet), mirroring
   *     the sandbox UX where stacked placements use double/right-click.
   *
   * Movement/capture phases:
   *   - First click selects a source stack and highlights legal targets using
   *     validMoves from the backend.
   *   - Second click on a highlighted target submits the matching move.
   */
  const handleBackendCellClick = (pos: Position, board: BoardState) => {
    if (!gameState) return;

    if (!isPlayer) {
      toast.error('Spectators cannot submit moves', { id: 'interaction-locked' });
      return;
    }

    if (!isConnectionActive) {
      toast.error('Moves paused while disconnected', { id: 'interaction-locked' });
      return;
    }

    // Ring placement phase: attempt a canonical 1-ring placement on empties
    // using the backend-reported validMoves. We never synthesize moves that
    // the backend hasn't already declared legal.
    if (gameState.currentPhase === 'ring_placement') {
      if (!Array.isArray(validMoves) || validMoves.length === 0) {
        return;
      }

      const key = positionToString(pos);
      const hasStack = !!board.stacks.get(key);

      if (!hasStack) {
        const placeMovesAtPos = validMoves.filter(
          (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
        );
        if (placeMovesAtPos.length === 0) {
          toast.error('Invalid placement position');
          return;
        }

        const preferred =
          placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];

        submitMove({
          type: 'place_ring',
          to: preferred.to,
          placementCount: preferred.placementCount,
          placedOnStack: preferred.placedOnStack,
        } as any);

        setSelected(undefined);
        setValidTargets([]);
        return;
      }

      // Clicking stacks in placement phase just selects them for now.
      setSelected(pos);
      setValidTargets([]);
      return;
    }

    // Movement/capture phases: simple "select source, then target" flow.

    // No existing selection: select this cell and highlight its valid targets.
    if (!selected) {
      setSelected(pos);
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const targets = validMoves
          .filter((m) => m.from && positionsEqual(m.from, pos))
          .map((m) => m.to);
        setValidTargets(targets);
      } else {
        setValidTargets([]);
      }
      return;
    }

    // Clicking the same cell clears selection.
    if (positionsEqual(selected, pos)) {
      setSelected(undefined);
      setValidTargets([]);
      return;
    }

    // If this click is one of the currently highlighted targets and we
    // have a matching valid move from the backend, submit that move.
    if (Array.isArray(validMoves) && validMoves.length > 0) {
      const matching = validMoves.find(
        (m) => m.from && positionsEqual(m.from, selected) && positionsEqual(m.to, pos)
      );

      if (matching) {
        submitMove({
          type: matching.type,
          from: matching.from,
          to: matching.to,
        } as any);

        setSelected(undefined);
        setValidTargets([]);
        return;
      }
    }

    // Otherwise treat this as a new selection.
    setSelected(pos);
    if (Array.isArray(validMoves) && validMoves.length > 0) {
      const targets = validMoves
        .filter((m) => m.from && positionsEqual(m.from, pos))
        .map((m) => m.to);
      setValidTargets(targets);

      if (targets.length === 0) {
        // Optional: could be annoying if just exploring, but helpful for feedback
        // toast('No valid moves from here', { icon: '🚫' });
      }
    } else {
      setValidTargets([]);
    }
  };

  /**
   * Backend double-click handling: in placement phase, prefer a 2-ring placement
   * on empty cells, falling back to a 1-ring placement if needed. On stacks,
   * double-click attempts a 1-ring placement onto the stack when the backend
   * reports such a move.
   */
  const handleBackendCellDoubleClick = (pos: Position, board: BoardState) => {
    if (!gameState) return;
    if (!isPlayer || !isConnectionActive) {
      toast.error('Cannot modify placements while disconnected or spectating', {
        id: 'interaction-locked',
      });
      return;
    }
    if (gameState.currentPhase !== 'ring_placement') {
      return;
    }

    if (!Array.isArray(validMoves) || validMoves.length === 0) {
      return;
    }

    const key = positionToString(pos);
    const hasStack = !!board.stacks.get(key);

    const placeMovesAtPos = validMoves.filter(
      (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
    );
    if (placeMovesAtPos.length === 0) {
      return;
    }

    let chosen: any | undefined;

    if (!hasStack) {
      const twoRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 2);
      const oneRing = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1);
      chosen = twoRing || oneRing || placeMovesAtPos[0];
    } else {
      chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === 1) || placeMovesAtPos[0];
    }

    if (!chosen) {
      return;
    }

    submitMove({
      type: 'place_ring',
      to: chosen.to,
      placementCount: chosen.placementCount,
      placedOnStack: chosen.placedOnStack,
    } as any);

    setSelected(undefined);
    setValidTargets([]);
  };

  /**
   * Backend context-menu handling (right-click / long-press proxy): prompt for
   * a ring count and submit the corresponding place_ring move if the backend
   * has advertised it via validMoves.
   */
  const handleBackendCellContextMenu = (pos: Position, board: BoardState) => {
    if (!gameState) return;
    if (!isPlayer || !isConnectionActive) {
      toast.error('Cannot modify placements while disconnected or spectating', {
        id: 'interaction-locked',
      });
      return;
    }
    if (gameState.currentPhase !== 'ring_placement') {
      return;
    }

    if (!Array.isArray(validMoves) || validMoves.length === 0) {
      return;
    }

    const key = positionToString(pos);
    const hasStack = !!board.stacks.get(key);

    const placeMovesAtPos = validMoves.filter(
      (m) => m.type === 'place_ring' && positionsEqual(m.to, pos)
    );
    if (placeMovesAtPos.length === 0) {
      return;
    }

    const counts = placeMovesAtPos.map((m) => m.placementCount ?? 1);
    const maxCount = Math.max(...counts);

    const promptLabel = hasStack
      ? 'Place how many rings on this stack? (canonical: 1)'
      : `Place how many rings on this empty cell? (1–${maxCount})`;

    const raw = window.prompt(promptLabel, Math.min(2, maxCount).toString());
    if (!raw) {
      return;
    }

    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > maxCount) {
      return;
    }

    const chosen = placeMovesAtPos.find((m) => (m.placementCount ?? 1) === parsed);
    if (!chosen) {
      return;
    }

    submitMove({
      type: 'place_ring',
      to: chosen.to,
      placementCount: chosen.placementCount,
      placedOnStack: chosen.placedOnStack,
    } as any);

    setSelected(undefined);
    setValidTargets([]);
  };

  // Auto-highlight valid placement positions during ring_placement
  useEffect(() => {
    if (!gameState) return;

    if (gameState.currentPhase === 'ring_placement') {
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const placementTargets = validMoves.filter((m) => m.type === 'place_ring').map((m) => m.to);

        setValidTargets((prev) => {
          // Simple length check first
          if (prev.length !== placementTargets.length) return placementTargets;
          // Deep check
          const allMatch = prev.every((p, i) =>
            placementTargets.some((pt) => positionsEqual(p, pt))
          );
          return allMatch ? prev : placementTargets;
        });
      } else {
        setValidTargets([]);
      }
    }
  }, [gameState?.currentPhase, validMoves]);

  // Track phase / player / choice changes for diagnostics
  useEffect(() => {
    if (!gameState) {
      lastPhaseRef.current = null;
      lastCurrentPlayerRef.current = null;
      return;
    }

    const events: string[] = [];

    if (gameState.currentPhase !== lastPhaseRef.current) {
      if (lastPhaseRef.current !== null) {
        events.push(`Phase changed: ${lastPhaseRef.current} → ${gameState.currentPhase}`);
      } else {
        events.push(`Phase: ${gameState.currentPhase}`);
      }
      lastPhaseRef.current = gameState.currentPhase;
    }

    if (gameState.currentPlayer !== lastCurrentPlayerRef.current) {
      events.push(`Current player: P${gameState.currentPlayer}`);
      lastCurrentPlayerRef.current = gameState.currentPlayer;
    }

    if (pendingChoice && pendingChoice.id !== lastChoiceIdRef.current) {
      events.push(`Choice requested: ${pendingChoice.type} for P${pendingChoice.playerNumber}`);
      lastChoiceIdRef.current = pendingChoice.id;
    } else if (!pendingChoice && lastChoiceIdRef.current) {
      events.push('Choice resolved');
      lastChoiceIdRef.current = null;
    }

    if (events.length > 0) {
      setEventLog((prev) => {
        const next = [...events, ...prev];
        return next.slice(0, 50);
      });
    }
  }, [gameState, pendingChoice]);

  useEffect(() => {
    if (!connectionStatus || lastConnectionStatusRef.current === connectionStatus) {
      lastConnectionStatusRef.current = connectionStatus;
      return;
    }

    const label =
      connectionStatus === 'connected'
        ? 'Connection restored'
        : connectionStatus === 'reconnecting'
          ? 'Connection interrupted – reconnecting'
          : connectionStatus === 'connecting'
            ? 'Connecting to server…'
            : 'Disconnected from server';

    setEventLog((prev) => [label, ...prev].slice(0, 50));
    lastConnectionStatusRef.current = connectionStatus;
  }, [connectionStatus]);

  // Maintain a live countdown for the current choice (if any)
  useEffect(() => {
    if (!pendingChoice || !choiceDeadline) {
      setChoiceTimeRemainingMs(null);
      if (choiceTimerRef.current !== null) {
        window.clearInterval(choiceTimerRef.current);
        choiceTimerRef.current = null;
      }
      return;
    }

    const update = () => {
      const remaining = choiceDeadline - Date.now();
      setChoiceTimeRemainingMs(remaining > 0 ? remaining : 0);
    };

    update();
    const id = window.setInterval(update, 250);
    choiceTimerRef.current = id as unknown as number;

    return () => {
      if (choiceTimerRef.current !== null) {
        window.clearInterval(choiceTimerRef.current);
        choiceTimerRef.current = null;
      }
    };
  }, [pendingChoice, choiceDeadline]);

  // State version counter to trigger AI turn checks after human moves.
  // Must be declared before any conditional returns to satisfy Rules of Hooks.
  const [sandboxStateVersion, setSandboxStateVersion] = useState(0);

  // Auto-trigger AI turns when state version changes (after human moves).
  // This effect checks the sandboxEngineRef directly to avoid prop dependency issues.
  useEffect(() => {
    if (!isConfigured || !sandboxEngineRef.current) {
      return;
    }

    const engine = sandboxEngineRef.current;
    const state = engine.getGameState();
    const current = state.players.find((p) => p.playerNumber === state.currentPlayer);

    // Only trigger if it's an active AI turn
    if (state.gameStatus === 'active' && current && current.type === 'ai') {
      // Update progress timestamp to prevent false stall warnings
      setSandboxLastProgressAt(Date.now());
      setSandboxStallWarning(null);

      // Small delay to allow React state to settle, then start AI turn loop
      const timeoutId = window.setTimeout(() => {
        void runSandboxAiTurnLoop();
      }, 50);

      return () => {
        window.clearTimeout(timeoutId);
      };
    }
  }, [isConfigured, sandboxStateVersion]);

  // Local sandbox AI-only stall watchdog. This runs independently of the
  // internal sandbox AI diagnostics and focuses on scheduler-level stalls
  // (situations where an AI player is to move but the local game state has
  // not advanced for an extended period).
  useEffect(() => {
    if (!isConfigured) {
      return;
    }

    const STALL_TIMEOUT_MS = 8000;
    const POLL_INTERVAL_MS = 1000;

    const id = window.setInterval(() => {
      setSandboxStallWarning((prevWarning) => {
        const last = sandboxLastProgressAt;
        if (last === null) {
          return prevWarning;
        }

        const engine = sandboxEngineRef.current;
        if (!engine) {
          return null;
        }

        const state = engine.getGameState();
        const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
        const now = Date.now();

        if (state.gameStatus !== 'active' || !current || current.type !== 'ai') {
          // Clear any previous warning when there is no active AI turn pending.
          return null;
        }

        if (now - last > STALL_TIMEOUT_MS) {
          return (
            prevWarning ??
            'Potential AI stall detected: sandbox AI has not advanced the game state for several seconds while an AI player is to move.'
          );
        }

        // Below threshold but still in AI turn: preserve any existing warning
        // (it may have been set by the diagnostics watcher or a previous poll).
        return prevWarning;
      });
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(id);
    };
  }, [isConfigured, sandboxLastProgressAt]);

  // Structural AI stall diagnostics watcher: when enabled via
  // RINGRIFT_ENABLE_SANDBOX_AI_STALL_DIAGNOSTICS, poll the sandbox AI trace
  // buffer and surface any "stall" entries as a UI banner so that AI-vs-AI
  // stalls are visible and debuggable from the /sandbox route.
  useEffect(() => {
    if (!sandboxDiagnosticsEnabled) {
      return;
    }

    if (typeof window === 'undefined') {
      return;
    }

    const POLL_INTERVAL_MS = 1000;
    const anyWindow = window as any;
    let lastSeenStallTimestamp = 0;

    const id = window.setInterval(() => {
      const trace = (anyWindow.__RINGRIFT_SANDBOX_TRACE__ ?? []) as any[];
      if (!Array.isArray(trace) || trace.length === 0) {
        return;
      }

      const latestStall = [...trace].reverse().find((entry) => entry && entry.kind === 'stall');
      if (!latestStall) {
        return;
      }

      const ts = typeof latestStall.timestamp === 'number' ? latestStall.timestamp : Date.now();
      if (ts <= lastSeenStallTimestamp) {
        return;
      }

      lastSeenStallTimestamp = ts;

      setSandboxStallWarning(
        (prev) =>
          prev ??
          'Sandbox AI stall detected by diagnostics: consecutive AI turns are not changing the game state. Use “Copy AI trace” for detailed debugging.'
      );
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(id);
    };
  }, [sandboxDiagnosticsEnabled]);

  // === Backend game mode ===
  if (routeGameId) {
    if (isConnecting && !gameState) {
      return (
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-2xl font-bold mb-2">Connecting to game…</h1>
          <p className="text-sm text-gray-500">Game ID: {routeGameId}</p>
        </div>
      );
    }

    // Reconnection banner
    const reconnectionBanner =
      connectionStatus !== 'connected' && gameState ? (
        <div className="bg-amber-500/20 border border-amber-500/50 text-amber-200 px-4 py-2 rounded mb-4 flex items-center justify-between">
          <span>
            {connectionStatus === 'reconnecting'
              ? 'Connection lost. Attempting to reconnect…'
              : connectionStatus === 'connecting'
                ? 'Connecting to game server…'
                : 'Disconnected from server. Moves are paused.'}
          </span>
          <div className="animate-spin h-4 w-4 border-2 border-amber-500 border-t-transparent rounded-full"></div>
        </div>
      ) : null;

    if (error && !gameState) {
      return (
        <div className="container mx-auto px-4 py-8 space-y-3">
          <h1 className="text-2xl font-bold mb-2">Unable to load game</h1>
          <p className="text-sm text-red-400">{error}</p>
          <p className="text-xs text-gray-500">Game ID: {routeGameId}</p>
        </div>
      );
    }

    if (!gameState || !gameId) {
      return (
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-2xl font-bold mb-2">Game not available</h1>
          <p className="text-sm text-gray-500">No game state received from server.</p>
        </div>
      );
    }

    const board = gameState.board;
    const boardType = gameState.boardType;

    const gameOverBannerText =
      victoryState && isVictoryModalDismissed && victoryState.reason
        ? getGameOverBannerText(victoryState.reason)
        : null;

    // Approximate must-move stack highlighting by inspecting backend-valid
    // movement/capture moves: if all such moves originate from the same
    // stack, we treat that stack as the must-move origin and highlight it
    // when the user has not made their own selection yet.
    const backendMustMoveFrom: Position | undefined = (() => {
      if (!Array.isArray(validMoves) || validMoves.length === 0) return undefined;
      if (gameState.currentPhase !== 'movement' && gameState.currentPhase !== 'capture') {
        return undefined;
      }

      const origins = validMoves
        .filter(
          (m) =>
            m.from &&
            (m.type === 'move_stack' ||
              m.type === 'move_ring' ||
              m.type === 'build_stack' ||
              m.type === 'overtaking_capture')
        )
        .map((m) => m.from as Position);

      if (origins.length === 0) return undefined;
      const first = origins[0];
      const allSame = origins.every((p) => positionsEqual(p, first));
      return allSame ? first : undefined;
    })();

    return (
      <div className="container mx-auto px-4 py-8 space-y-4">
        {reconnectionBanner}

        {gameOverBannerText && (
          <div className="bg-emerald-900/30 border border-emerald-500/60 text-emerald-100 px-4 py-2 rounded mb-2 text-sm">
            {gameOverBannerText}
          </div>
        )}

        {/* Fatal game error banner */}
        {fatalGameError && (
          <div className="bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-3 rounded">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <p className="font-semibold mb-1">{fatalGameError.message}</p>
                {process.env.NODE_ENV === 'development' && fatalGameError.technical && (
                  <p className="text-xs text-red-300 font-mono mt-2">
                    Technical: {fatalGameError.technical}
                  </p>
                )}
              </div>
              <button
                onClick={() => setFatalGameError(null)}
                className="ml-4 text-red-300 hover:text-red-100 transition"
                aria-label="Dismiss error"
              >
                ✕
              </button>
            </div>
          </div>
        )}

        <header className="flex items-center justify-between">
          <div>
            {renderGameHeader(gameState)}
            {!isPlayer && (
              <span className="ml-2 px-2 py-0.5 bg-purple-900/50 border border-purple-500/50 text-purple-200 text-xs rounded-full uppercase tracking-wider font-bold">
                Spectating
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2 text-xs text-gray-400">
            <span>Status: {gameState.gameStatus}</span>
            <span>• Phase: {gameState.currentPhase}</span>
            <span>• Current player: P{gameState.currentPlayer}</span>
          </div>
        </header>

        {/* Victory modal overlays the rest of the UI when the game is over. */}
        <VictoryModal
          isOpen={!!victoryState && !isVictoryModalDismissed}
          gameResult={victoryState}
          players={gameState.players}
          gameState={gameState}
          onClose={() => {
            setIsVictoryModalDismissed(true);
          }}
          onReturnToLobby={() => navigate('/lobby')}
          onRematch={() => {
            // TODO: Implement rematch request via WebSocket
            console.log('Rematch requested');
          }}
          currentUserId={user?.id}
        />

        <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
          <section>
            <BoardView
              boardType={boardType}
              board={board}
              selectedPosition={selected || backendMustMoveFrom}
              validTargets={validTargets}
              onCellClick={(pos) => handleBackendCellClick(pos, board)}
              onCellDoubleClick={(pos) => handleBackendCellDoubleClick(pos, board)}
              onCellContextMenu={(pos) => handleBackendCellContextMenu(pos, board)}
              isSpectator={!isPlayer}
            />
          </section>

          {/*
            In backend mode, choices are driven by the server via the
            GameContext. We render the ChoiceDialog overlay only when a
            pendingChoice exists. For now we do not cancel from the client;
            choices time out or are cleared by the server.
          */}
          {isPlayer && (
            <ChoiceDialog
              choice={pendingChoice}
              deadline={choiceDeadline}
              timeRemainingMs={choiceTimeRemainingMs}
              onSelectOption={(choice, option) => respondToChoice(choice, option)}
            />
          )}

          <aside className="w-full md:w-72 space-y-3 text-sm text-slate-100">
            <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
              <h2 className="font-semibold mb-2">Selection</h2>
              {selected ? (
                <div>
                  <div>
                    Selected: ({selected.x}, {selected.y}
                    {selected.z !== undefined ? `, ${selected.z}` : ''})
                  </div>
                  <div className="text-xs text-slate-300 mt-1">
                    Click a source stack, then click a highlighted destination to send a move to the
                    server. The backend GameEngine is the source of truth for legality and state.
                  </div>
                </div>
              ) : (
                <div className="text-slate-200">Click a cell to inspect it.</div>
              )}
              {boardInteractionMessage && (
                <div className="mt-3 text-xs text-amber-300">{boardInteractionMessage}</div>
              )}
            </div>

            <GameHUD
              gameState={gameState}
              currentPlayer={currentPlayer}
              instruction={getInstruction()}
              connectionStatus={connectionStatus}
              isSpectator={!isPlayer}
              lastHeartbeatAt={lastHeartbeatAt}
              currentUserId={user?.id}
            />

            <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1">
              <span>Log view</span>
              <button
                type="button"
                onClick={() => setShowSystemEventsInLog((prev) => !prev)}
                className="px-2 py-0.5 rounded border border-slate-600 bg-slate-900/70 text-xs hover:border-emerald-400 hover:text-emerald-200 transition"
              >
                {showSystemEventsInLog ? 'Moves + system' : 'Moves only'}
              </button>
            </div>

            <GameEventLog
              history={gameState.history}
              systemEvents={showSystemEventsInLog ? eventLog : []}
              victoryState={victoryState}
            />

            {/* Chat UI */}
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
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!chatInput.trim()) return;

                  if (sendChatMessage) {
                    sendChatMessage(chatInput);
                  } else {
                    // Fallback for local sandbox
                    setLocalChatMessages((prev) => [...prev, { sender: 'You', text: chatInput }]);
                  }
                  setChatInput('');
                }}
                className="flex gap-2"
              >
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type a message..."
                  className="flex-1 bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-emerald-500"
                />
                <button
                  type="submit"
                  className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-1 rounded text-xs font-medium"
                >
                  Send
                </button>
              </form>
            </div>
          </aside>
        </main>
      </div>
    );
  }

  // === Local sandbox mode (no gameId in route) ===

  // Render setup form before game starts
  const activePlayerTypes = config.playerTypes.slice(0, config.numPlayers);
  const humanSeatCount = activePlayerTypes.filter((t) => t === 'human').length;
  const aiSeatCount = activePlayerTypes.length - humanSeatCount;
  const selectedBoardPreset =
    BOARD_PRESETS.find((preset) => preset.value === config.boardType) ?? BOARD_PRESETS[0];

  if (!isConfigured || (!localSandbox && !sandboxEngineRef.current)) {
    return (
      <div className="container mx-auto px-4 py-8 space-y-6">
        <header>
          <h1 className="text-3xl font-bold mb-1">Start a RingRift Game (Local Sandbox)</h1>
          <p className="text-sm text-gray-500">
            This mode runs entirely in the browser using a local board. To view or play a real
            server-backed game, navigate to a URL with a game ID (e.g.
            <code className="ml-1 text-xs">/game/:gameId</code>).
          </p>
        </header>

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
                      className={`rounded-xl border bg-slate-900/60 px-4 py-3 flex items-center justify-between gap-4 ${
                        meta.accent
                      }`}
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
                <span className="font-semibold">{humanSeatCount}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">AI opponents</span>
                <span className="font-semibold">{aiSeatCount}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-300">Total seats</span>
                <span className="font-semibold">{config.numPlayers}</span>
              </div>
            </div>

            <div className="space-y-2">
              <p className="text-xs text-slate-400">
                We first attempt to stand up a backend game with these settings. If that fails, we
                fall back to a purely client-local sandbox so you can still test moves offline.
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
      </div>
    );
  }

  // Game view once configured (local sandbox)
  const sandboxEngine = sandboxEngineRef.current;
  const sandboxGameState: GameState | null = sandboxEngine
    ? sandboxEngine.getGameState()
    : localSandbox
      ? ({
          // Minimal projection when falling back to legacy LocalSandboxState
          id: 'sandbox-legacy',
          boardType: config.boardType,
          board: localSandbox.board,
          players: localSandbox.players,
          currentPhase: localSandbox.currentPhase,
          currentPlayer: localSandbox.currentPlayer,
          moveHistory: [],
          history: [],
          timeControl: { type: 'rapid', initialTime: 600, increment: 0 },
          spectators: [],
          gameStatus: 'active',
          createdAt: new Date(),
          lastMoveAt: new Date(),
          isRated: false,
          maxPlayers: config.numPlayers,
          totalRingsInPlay: 0,
          totalRingsEliminated: 0,
          victoryThreshold: 0,
          territoryVictoryThreshold: 0,
        } as GameState)
      : null;

  const sandboxBoardState: BoardState | null = sandboxGameState?.board ?? null;
  const sandboxVictoryResult = sandboxEngine ? sandboxEngine.getVictoryResult() : null;
  const sandboxGameOverBannerText =
    sandboxVictoryResult && isSandboxVictoryModalDismissed && sandboxVictoryResult.reason
      ? getGameOverBannerText(sandboxVictoryResult.reason)
      : null;
  const boardTypeValue = sandboxBoardState?.type ?? config.boardType;
  const boardPresetInfo = BOARD_PRESETS.find((preset) => preset.value === boardTypeValue);
  const boardDisplayLabel = boardPresetInfo?.label ?? boardTypeValue;
  const boardDisplaySubtitle = boardPresetInfo?.subtitle ?? 'Custom configuration';
  const boardDisplayBlurb =
    boardPresetInfo?.blurb ?? 'Custom layout selected for this local sandbox match.';
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
  const sandboxCurrentPlayer =
    sandboxPlayersList.find((p) => p.playerNumber === sandboxCurrentPlayerNumber) ??
    sandboxPlayersList[0];
  const sandboxPhaseKey = sandboxGameState?.currentPhase ?? 'ring_placement';
  const sandboxPhaseDetails = PHASE_COPY[sandboxPhaseKey] ?? PHASE_COPY.ring_placement;
  const sandboxCurrentPlayerLabel =
    sandboxCurrentPlayer?.username || `Player ${sandboxCurrentPlayer?.playerNumber ?? '?'}`;
  const displayedValidTargets =
    sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets;
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
  const sandboxModeNotes = [
    `Board: ${boardDisplayLabel}`,
    `${humanSeatCount} human seat${humanSeatCount === 1 ? '' : 's'} · ${aiSeatCount} AI`,
    sandboxEngine
      ? 'Engine parity mode with local AI and choice handler.'
      : 'Legacy local sandbox fallback (no backend).',
    'Runs entirely in-browser; use "Change Setup" to switch configurations.',
  ];

  return (
    <div className="container mx-auto px-4 py-8 space-y-4">
      {sandboxStallWarning && (
        <div className="p-3 rounded-xl border border-amber-500/70 bg-amber-900/40 text-amber-100 text-xs flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <span>{sandboxStallWarning}</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleCopySandboxTrace}
              className="px-3 py-1 rounded-lg border border-amber-300 bg-amber-800/70 text-[11px] font-semibold hover:border-amber-100 hover:bg-amber-700/80"
            >
              Copy AI trace
            </button>
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

      {/* Local sandbox victory modal, reusing the shared VictoryModal UI. */}
      {sandboxGameState && (
        <VictoryModal
          isOpen={!!sandboxVictoryResult && !isSandboxVictoryModalDismissed}
          gameResult={sandboxVictoryResult}
          players={sandboxGameState.players}
          gameState={sandboxGameState}
          onClose={() => {
            setIsSandboxVictoryModalDismissed(true);
          }}
          onReturnToLobby={() => {
            setIsConfigured(false);
            setLocalSandbox(null);
            sandboxEngineRef.current = null;
            setSelected(undefined);
            setValidTargets([]);
            setBackendSandboxError(null);
            setSandboxPendingChoice(null);
            setIsSandboxVictoryModalDismissed(false);
          }}
          currentUserId={user?.id}
        />
      )}

      <ChoiceDialog
        choice={sandboxPendingChoice}
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
        }}
      />

      <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
        <section className="flex justify-center md:block">
          {sandboxBoardState && (
            <div className="inline-block space-y-3">
              <div className="p-4 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Local Sandbox</p>
                    <h1 className="text-2xl font-bold text-white">Game – {boardDisplayLabel}</h1>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setIsConfigured(false);
                      setLocalSandbox(null);
                      sandboxEngineRef.current = null;
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
                </div>
              </div>

              <BoardView
                boardType={sandboxBoardState.type}
                board={sandboxBoardState}
                selectedPosition={selected}
                validTargets={displayedValidTargets}
                onCellClick={(pos) => handleSandboxCellClick(pos)}
                onCellDoubleClick={(pos) => handleSandboxCellDoubleClick(pos)}
                onCellContextMenu={(pos) => handleSandboxCellContextMenu(pos)}
                showMovementGrid
                showCoordinateLabels={
                  sandboxBoardState.type === 'square8' || sandboxBoardState.type === 'square19'
                }
              />

              {/* Sandbox game summary bar directly below the board */}
              <section className="mt-1 p-3 rounded-2xl border border-slate-700 bg-slate-900/70 shadow-lg flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 text-xs text-slate-200">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-2 py-1 rounded-full bg-slate-800/80 border border-slate-600">
                    {boardDisplaySubtitle}
                  </span>
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
            </div>
          )}
        </section>

        <aside className="w-full md:w-80 space-y-4 text-sm text-slate-100">
          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
            <h2 className="font-semibold">Players</h2>
            <div className="space-y-2">
              {sandboxPlayersList.map((player) => {
                const isCurrent = player.playerNumber === sandboxCurrentPlayerNumber;
                return (
                  <div
                    key={player.playerNumber}
                    className={`rounded-xl border px-3 py-2 text-xs flex items-center justify-between ${
                      isCurrent
                        ? 'border-emerald-400 bg-emerald-900/20'
                        : 'border-slate-700 bg-slate-900/40'
                    }`}
                  >
                    <div>
                      <p className="font-semibold text-white">
                        P{player.playerNumber} {player.username ? `• ${player.username}` : ''}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {player.type === 'ai' ? 'Computer' : 'Human'}
                      </p>
                    </div>
                    <div className="flex gap-3 text-right">
                      <div>
                        <p className="text-sm font-bold text-white">{player.ringsInHand}</p>
                        <p className="text-[11px] text-slate-400">in hand</p>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">{player.territorySpaces}</p>
                        <p className="text-[11px] text-slate-400">territory</p>
                      </div>
                      <div>
                        <p className="text-sm font-bold text-white">{player.eliminatedRings}</p>
                        <p className="text-[11px] text-slate-400">eliminated</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60">
            <GameEventLog
              history={sandboxGameState?.history ?? []}
              systemEvents={[]}
              victoryState={sandboxVictoryResult}
            />
          </div>

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
            <div className="flex items-center justify-between">
              <h2 className="font-semibold">Selection</h2>
              {displayedValidTargets && (
                <span className="text-xs text-slate-400">
                  Targets: {displayedValidTargets?.length ?? 0}
                </span>
              )}
            </div>
            {selected ? (
              <div className="space-y-2">
                <div className="text-lg font-mono font-semibold text-white">
                  ({selected.x}, {selected.y}
                  {selected.z !== undefined ? `, ${selected.z}` : ''})
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
              <div className="text-slate-200">
                Click a cell to inspect stacks and available moves.
              </div>
            )}
          </div>

          <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-2">
            <h2 className="font-semibold">Phase Guide</h2>
            <p className="text-xs uppercase tracking-wide text-slate-400">
              {sandboxPhaseDetails.label}
            </p>
            <p className="text-sm text-slate-200">{sandboxPhaseDetails.summary}</p>
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
    </div>
  );
}
