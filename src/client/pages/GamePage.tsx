import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameHUD } from '../components/GameHUD';
import { LocalSandboxState, handleLocalSandboxCellClick } from '../sandbox/localSandboxController';
import { ClientSandboxEngine, SandboxConfig, SandboxInteractionHandler } from '../sandbox/ClientSandboxEngine';
import {
  BoardState,
  BoardType,
  GameState,
  Position,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
  positionsEqual,
  CreateGameRequest
} from '../../shared/types/game';
import { useGame } from '../contexts/GameContext';
import { gameApi } from '../services/api';

type LocalPlayerType = 'human' | 'ai';

interface LocalConfig {
  numPlayers: number;
  boardType: BoardType;
  playerTypes: LocalPlayerType[]; // indexed 0..3 for players 1..4
}

function renderGameHeader(gameState: GameState) {
  const playerSummary = gameState.players
    .sort((a, b) => a.playerNumber - b.playerNumber)
    .map(p => `${p.username || `P${p.playerNumber}`} (${p.type})`)
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
    submitMove
  } = useGame();

  // Choice/phase diagnostics
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [choiceTimeRemainingMs, setChoiceTimeRemainingMs] = useState<number | null>(null);
  const choiceTimerRef = useRef<number | null>(null);
  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);

  // Local setup state (used only when no gameId route param is provided)
  const [config, setConfig] = useState<LocalConfig>({
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai']
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

  // UI selection state (used in both modes)
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

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
          const selectedOption =
            optionsArray[Math.floor(Math.random() * optionsArray.length)] as TChoice['options'][number];

          return {
            choiceId: choice.id,
            playerNumber: choice.playerNumber,
            choiceType: choice.type,
            selectedOption
          } as PlayerChoiceResponseFor<TChoice>;
        }

        // Human players: surface the choice to the sandbox UI and wait for a selection.
        setSandboxPendingChoice(choice);
        return new Promise<PlayerChoiceResponseFor<TChoice>>(resolve => {
          sandboxChoiceResolverRef.current = ((
            response: PlayerChoiceResponseFor<PlayerChoice>
          ) => {
            resolve(response as PlayerChoiceResponseFor<TChoice>);
          }) as (response: PlayerChoiceResponseFor<PlayerChoice>) => void;
        });
      }
    };
  };

  // When a :gameId is present in the route, connect to that backend game
  useEffect(() => {
    if (!routeGameId) {
      // Ensure any previous connection is torn down when leaving a game
      disconnect();
      return;
    }

    connectToGame(routeGameId);

    return () => {
      disconnect();
    };
  }, [routeGameId, connectToGame, disconnect]);

  const handleSetupChange = (partial: Partial<LocalConfig>) => {
    setConfig(prev => ({
      ...prev,
      ...partial,
      playerTypes: partial.numPlayers
        ? prev.playerTypes.map((t, idx) => (idx < partial.numPlayers! ? t : prev.playerTypes[idx]))
        : prev.playerTypes
    }));
  };

  const handlePlayerTypeChange = (index: number, type: LocalPlayerType) => {
    setConfig(prev => {
      const next = [...prev.playerTypes];
      next[index] = type;
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
          increment: 0
        },
        // For now, derive a simple AI configuration from local
        // player types: any non-human seats become AI opponents
        // with a uniform difficulty. This keeps the harness
        // loosely in sync with LobbyPage without duplicating
        // its full form.
        aiOpponents: (() => {
          const aiSeats = config.playerTypes
            .slice(0, config.numPlayers)
            .filter(t => t === 'ai').length;
          if (aiSeats <= 0) return undefined;
          return {
            count: aiSeats,
            difficulty: Array(aiSeats).fill(5),
            mode: 'service',
            aiType: 'heuristic'
          };
        })()
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
      playerKinds: config.playerTypes.slice(0, config.numPlayers) as LocalPlayerType[]
    };

    const interactionHandler = createSandboxInteractionHandler(
      config.playerTypes.slice(0, config.numPlayers)
    );

    sandboxEngineRef.current = new ClientSandboxEngine({
      config: sandboxConfig,
      interactionHandler
    });

    setLocalSandbox(null);
    setSelected(undefined);
    setValidTargets([]);
    setSandboxPendingChoice(null);
    setIsConfigured(true);
  };

  // Unified sandbox click handler: prefer the ClientSandboxEngine when
  // available (Stage 2 harness), otherwise fall back to the legacy
  // LocalSandboxState controller.
  const handleSandboxCellClick = (pos: Position) => {
    const engine = sandboxEngineRef.current;
    if (engine) {
      engine.handleHumanCellClick(pos);
      // In future iterations we will call engine.maybeRunAITurn() here when
      // the next player is AI, and re-render from engine.getGameState().
      setSelected(pos);
      setValidTargets([]);
      return;
    }

    if (!localSandbox) return;

    const next = handleLocalSandboxCellClick(localSandbox, pos);
    setLocalSandbox(next);
    setSelected(pos);
    setValidTargets([]); // movement/capture targets will be added in a later phase
  };

  /**
   * Backend game click handling: simple "select source, then select target" flow.
   *
   * - First click selects a source position and highlights legal targets
   *   using the validMoves array from the backend (if available).
   * - Second click on a highlighted target constructs a partial Move and
   *   calls submitMove, letting the server-side GameEngine validate and
   *   apply the move.
   */
  const handleBackendCellClick = (pos: Position, board: BoardState) => {
    if (!gameState) return;

    // No existing selection: select this cell and highlight its valid targets.
    if (!selected) {
      setSelected(pos);
      if (Array.isArray(validMoves) && validMoves.length > 0) {
        const targets = validMoves
          .filter(m => m.from && positionsEqual(m.from, pos))
          .map(m => m.to);
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
        m => m.from && positionsEqual(m.from, selected) && positionsEqual(m.to, pos)
      );

      if (matching) {
        submitMove({
          type: matching.type,
          from: matching.from,
          to: matching.to
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
        .filter(m => m.from && positionsEqual(m.from, pos))
        .map(m => m.to);
      setValidTargets(targets);
    } else {
      setValidTargets([]);
    }
  };

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
      events.push(
        `Choice requested: ${pendingChoice.type} for P${pendingChoice.playerNumber}`
      );
      lastChoiceIdRef.current = pendingChoice.id;
    } else if (!pendingChoice && lastChoiceIdRef.current) {
      events.push('Choice resolved');
      lastChoiceIdRef.current = null;
    }

    if (events.length > 0) {
      setEventLog(prev => {
        const next = [...events, ...prev];
        return next.slice(0, 50);
      });
    }
  }, [gameState, pendingChoice]);

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

    return (
      <div className="container mx-auto px-4 py-8 space-y-4">
        <header className="flex items-center justify-between">
          <div>{renderGameHeader(gameState)}</div>
          <div className="flex items-center space-x-2 text-xs text-gray-400">
            <span>Status: {gameState.gameStatus}</span>
            <span>• Phase: {gameState.currentPhase}</span>
            <span>• Current player: P{gameState.currentPlayer}</span>
          </div>
        </header>

        {/* Victory modal overlays the rest of the UI when the game is over. */}
        <VictoryModal
          isOpen={!!victoryState}
          gameState={gameState}
          result={victoryState}
          onClose={() => navigate('/lobby')}
        />

        <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
          <section>
            <BoardView
              boardType={boardType}
              board={board}
              selectedPosition={selected}
              validTargets={validTargets}
              onCellClick={pos => handleBackendCellClick(pos, board)}
            />
          </section>

          {/*
            In backend mode, choices are driven by the server via the
            GameContext. We render the ChoiceDialog overlay only when a
            pendingChoice exists. For now we do not cancel from the client;
            choices time out or are cleared by the server.
          */}
          <ChoiceDialog
            choice={pendingChoice}
            deadline={choiceDeadline}
            onSelectOption={(choice, option) => respondToChoice(choice, option)}
          />

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
                    Click a source stack, then click a highlighted destination to send a
                    move to the server. The backend GameEngine is the source of truth for
                    legality and state.
                  </div>
                </div>
              ) : (
                <div className="text-slate-200">Click a cell to inspect it.</div>
              )}
            </div>

            <GameHUD
              gameState={gameState}
              isConnecting={isConnecting}
              pendingChoice={pendingChoice}
              choiceTimeRemainingMs={choiceTimeRemainingMs}
            />

            <div className="p-3 border border-slate-700 rounded bg-slate-900/50 max-h-48 overflow-y-auto">
              <h2 className="font-semibold mb-2">Recent events</h2>
              {eventLog.length === 0 ? (
                <div className="text-slate-300 text-xs">No events yet.</div>
              ) : (
                <ul className="list-disc list-inside text-slate-200 space-y-1 text-xs">
                  {eventLog.map((entry, idx) => (
                    <li key={idx}>{entry}</li>
                  ))}
                </ul>
              )}
            </div>
          </aside>
        </main>
      </div>
    );
  }

  // === Local sandbox mode (no gameId in route) ===

  // Render setup form before game starts
  if (!isConfigured || (!localSandbox && !sandboxEngineRef.current)) {
    return (
      <div className="container mx-auto px-4 py-8 space-y-6">
        <header>
          <h1 className="text-3xl font-bold mb-1">Start a RingRift Game (Local Sandbox)</h1>
          <p className="text-sm text-gray-500">
            This mode runs entirely in the browser using a local board. To view or play
            a real server-backed game, navigate to a URL with a game ID (e.g.
            <code className="ml-1 text-xs">/game/:gameId</code>).
          </p>
        </header>

        <section className="max-w-xl p-4 rounded-md bg-slate-800 border border-slate-700 space-y-4 text-slate-100">
          {backendSandboxError && (
            <div className="p-2 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded">
              {backendSandboxError}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium mb-1 text-slate-100">Number of players</label>
            <select
              className="w-full max-w-xs px-2 py-1 rounded bg-slate-700 border border-slate-500 text-sm text-slate-100"
              value={config.numPlayers}
              onChange={e => handleSetupChange({ numPlayers: Number(e.target.value) })}
            >
              {[2, 3, 4].map(n => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1 text-slate-100">Board type</label>
            <select
              className="w-full max-w-xs px-2 py-1 rounded bg-slate-700 border border-slate-500 text-sm text-slate-100"
              value={config.boardType}
              onChange={e => handleSetupChange({ boardType: e.target.value as BoardType })}
            >
              <option value="square8">8x8 (compact)</option>
              <option value="square19">19x19 (full)</option>
              <option value="hexagonal">Hexagonal</option>
            </select>
          </div>

          <div>
            <p className="block text-sm font-medium mb-2 text-slate-100">Players</p>
            <div className="space-y-2 text-sm">
              {Array.from({ length: config.numPlayers }, (_, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-slate-100">Player {i + 1}</span>
                  <select
                    className="px-2 py-1 rounded bg-slate-700 border border-slate-500 text-slate-100"
                    value={config.playerTypes[i]}
                    onChange={e => handlePlayerTypeChange(i, e.target.value as LocalPlayerType)}
                  >
                    <option value="human">Human</option>
                    <option value="ai">Computer (AI)</option>
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleStartLocalGame}
              className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white"
            >
              Start Local Game
            </button>
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
        territoryVictoryThreshold: 0
      } as GameState)
    : null;

  const sandboxBoardState: BoardState | null = sandboxGameState?.board ?? null;
  const sandboxVictoryResult = sandboxEngine ? sandboxEngine.getVictoryResult() : null;

  return (
    <div className="container mx-auto px-4 py-8 space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-1">Game (Local Sandbox)</h1>
          <p className="text-sm text-gray-500">
            Board type: {sandboxBoardState?.type ?? config.boardType} • Players: {config.numPlayers} ({config.playerTypes
              .slice(0, config.numPlayers)
              .join(', ')})
          </p>
        </div>
        <div className="flex items-center space-x-2 text-sm">
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
            }}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-gray-200"
          >
            Change Setup
          </button>
        </div>
      </header>

      {/* Local sandbox victory modal, reusing the shared VictoryModal UI. */}
      <VictoryModal
        isOpen={!!sandboxVictoryResult}
        gameState={sandboxGameState}
        result={sandboxVictoryResult}
        onClose={() => {
          setIsConfigured(false);
          setLocalSandbox(null);
          sandboxEngineRef.current = null;
          setSelected(undefined);
          setValidTargets([]);
          setBackendSandboxError(null);
          setSandboxPendingChoice(null);
        }}
      />

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
              selectedOption: option
            } as PlayerChoiceResponseFor<PlayerChoice>);
            sandboxChoiceResolverRef.current = null;
          }
          setSandboxPendingChoice(null);
        }}
      />

      <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
        <section>
          {sandboxBoardState && (
            <BoardView
              boardType={sandboxBoardState.type}
              board={sandboxBoardState}
              selectedPosition={selected}
              validTargets={validTargets}
              onCellClick={pos => handleSandboxCellClick(pos)}
              showMovementGrid
            />
          )}
        </section>

        <aside className="w-full md:w-64 space-y-3 text-sm text-slate-100">
          <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
            <h2 className="font-semibold mb-2">Selection</h2>
            {selected ? (
              <div>
                <div>
                  Selected: ({selected.x}, {selected.y}
                  {selected.z !== undefined ? `, ${selected.z}` : ''})
                </div>
                <div className="text-xs text-slate-300 mt-1">Valid targets not yet wired.</div>
              </div>
            ) : (
              <div className="text-slate-200">Click a cell to select it.</div>
            )}
          </div>

          <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
            <h2 className="font-semibold mb-2">Status</h2>
            <ul className="list-disc list-inside text-slate-200 space-y-1">
              <li>
                Board rendered for{' '}
                {sandboxBoardState?.type === 'square8'
                  ? '8x8'
                  : sandboxBoardState?.type === 'square19'
                  ? '19x19'
                  : sandboxBoardState?.type === 'hexagonal'
                  ? 'hexagonal'
                  : 'unknown'}{' '}
                layout.
              </li>
              <li>This mode is currently local-only (no backend moves yet).</li>
              <li>Next: connect moves to GameEngine and AI/choice flow.</li>
            </ul>
          </div>
        </aside>
      </main>
    </div>
  );
}
