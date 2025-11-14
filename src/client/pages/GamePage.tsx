import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import {
  BOARD_CONFIGS,
  BoardState,
  BoardType,
  GameState,
  Position,
  RingStack
} from '../../shared/types/game';
import { useGame } from '../contexts/GameContext';

function createEmptyBoard(boardType: BoardType): BoardState {
  const config = BOARD_CONFIGS[boardType];
  return {
    stacks: new Map<string, RingStack>(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: config.size,
    type: boardType
  };
}

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
  const params = useParams<{ gameId?: string }>();
  const routeGameId = params.gameId;

  // Backend game context (used when viewing a real server game)
  const {
    gameId,
    gameState,
    isConnecting,
    error,
    connectToGame,
    disconnect,
    pendingChoice,
    choiceDeadline,
    respondToChoice,
    submitMove
  } = useGame();

  // Local setup state (used only when no gameId route param is provided)
  const [config, setConfig] = useState<LocalConfig>({
    numPlayers: 2,
    boardType: 'square8',
    playerTypes: ['human', 'human', 'ai', 'ai']
  });
  const [isConfigured, setIsConfigured] = useState(false);

  // Local-only board state (for sandbox mode without a backend game)
  const [localBoardType, setLocalBoardType] = useState<BoardType>('square8');
  const [localBoard, setLocalBoard] = useState<BoardState | null>(null);

  // UI selection state (used in both modes)
  const [selected, setSelected] = useState<Position | undefined>();
  const [validTargets, setValidTargets] = useState<Position[]>([]);

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

  const handleStartLocalGame = () => {
    const nextBoardType = config.boardType;
    setLocalBoardType(nextBoardType);
    setLocalBoard(createEmptyBoard(nextBoardType));
    setSelected(undefined);
    setValidTargets([]);
    setIsConfigured(true);
  };

  const handleCellClick = (pos: Position, board: BoardState | null | undefined) => {
    if (!board) return;
    const isSame =
      selected &&
      selected.x === pos.x &&
      selected.y === pos.y &&
      selected.z === pos.z;

    if (isSame) {
      setSelected(undefined);
      setValidTargets([]);
    } else {
      setSelected(pos);
      setValidTargets([]); // will later be populated from RuleEngine.getValidMoves
    }
  };

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

        <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
          <section>
            <BoardView
              boardType={boardType}
              board={board}
              selectedPosition={selected}
              validTargets={validTargets}
              onCellClick={pos => handleCellClick(pos, board)}
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

          <aside className="w-full md:w-72 space-y-3 text-sm">
            <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
              <h2 className="font-semibold mb-2">Selection</h2>
              {selected ? (
                <div>
                  <div>
                    Selected: ({selected.x}, {selected.y}
                    {selected.z !== undefined ? `, ${selected.z}` : ''})
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    Move execution is not yet wired to the backend; this view is currently
                    read-only.
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">Click a cell to inspect it.</div>
              )}
            </div>

            <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
              <h2 className="font-semibold mb-2">Status</h2>
              <ul className="list-disc list-inside text-gray-400 space-y-1">
                <li>Board type: {boardType}</li>
                <li>Backend game engine provides the authoritative state.</li>
                <li>WebSocket connection: {isConnecting ? 'connecting' : 'connected'}.</li>
              </ul>
            </div>
          </aside>
        </main>
      </div>
    );
  }

  // === Local sandbox mode (no gameId in route) ===

  // Render setup form before game starts
  if (!isConfigured || !localBoard) {
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

        <section className="max-w-xl p-4 rounded-md bg-slate-900 border border-slate-700 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-200">Number of players</label>
            <select
              className="w-full max-w-xs px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
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
            <label className="block text-sm font-medium mb-1 text-gray-200">Board type</label>
            <select
              className="w-full max-w-xs px-2 py-1 rounded bg-slate-800 border border-slate-600 text-sm"
              value={config.boardType}
              onChange={e => handleSetupChange({ boardType: e.target.value as BoardType })}
            >
              <option value="square8">8x8 (compact)</option>
              <option value="square19">19x19 (full)</option>
              <option value="hexagonal">Hexagonal</option>
            </select>
          </div>

          <div>
            <p className="block text-sm font-medium mb-2 text-gray-200">Players</p>
            <div className="space-y-2 text-sm">
              {Array.from({ length: config.numPlayers }, (_, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="text-gray-200">Player {i + 1}</span>
                  <select
                    className="px-2 py-1 rounded bg-slate-800 border border-slate-600"
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
  return (
    <div className="container mx-auto px-4 py-8 space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-1">Game (Local Sandbox)</h1>
          <p className="text-sm text-gray-500">
            Board type: {localBoardType} • Players: {config.numPlayers} ({config.playerTypes
              .slice(0, config.numPlayers)
              .join(', ')})
          </p>
        </div>
        <div className="flex items-center space-x-2 text-sm">
          <button
            type="button"
            onClick={() => {
              setIsConfigured(false);
              setLocalBoard(null);
            }}
            className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 border border-slate-600 text-gray-200"
          >
            Change Setup
          </button>
        </div>
      </header>

      <main className="flex flex-col md:flex-row md:space-x-8 space-y-4 md:space-y-0">
        <section>
          <BoardView
            boardType={localBoardType}
            board={localBoard}
            selectedPosition={selected}
            validTargets={validTargets}
            onCellClick={pos => handleCellClick(pos, localBoard)}
          />
        </section>

        <aside className="w-full md:w-64 space-y-3 text-sm">
          <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
            <h2 className="font-semibold mb-2">Selection</h2>
            {selected ? (
              <div>
                <div>
                  Selected: ({selected.x}, {selected.y}
                  {selected.z !== undefined ? `, ${selected.z}` : ''})
                </div>
                <div className="text-xs text-gray-400 mt-1">Valid targets not yet wired.</div>
              </div>
            ) : (
              <div className="text-gray-500">Click a cell to select it.</div>
            )}
          </div>

          <div className="p-3 border border-slate-700 rounded bg-slate-900/50">
            <h2 className="font-semibold mb-2">Status</h2>
            <ul className="list-disc list-inside text-gray-400 space-y-1">
              <li>
                Board rendered for {localBoardType === 'square8'
                  ? '8x8'
                  : localBoardType === 'square19'
                  ? '19x19'
                  : 'hexagonal'}{' '}
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
