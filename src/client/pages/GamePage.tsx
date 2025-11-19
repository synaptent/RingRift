import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { BoardView } from '../components/BoardView';
import { ChoiceDialog } from '../components/ChoiceDialog';
import { VictoryModal } from '../components/VictoryModal';
import { GameHUD } from '../components/GameHUD';
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
  Position,
  PlayerChoice,
  PlayerChoiceResponseFor,
  positionToString,
  positionsEqual,
  CreateGameRequest,
} from '../../shared/types/game';
import { toast } from 'react-hot-toast';
import { useGame } from '../contexts/GameContext';
import { useAuth } from '../contexts/AuthContext';
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
    .map((p) => `${p.username || `P${p.playerNumber}`} (${p.type})`)
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
  } = useGame();

  const { user } = useAuth();

  // Derived state for HUD
  const currentPlayer = gameState?.players.find((p) => p.playerNumber === gameState.currentPlayer);
  const isPlayer = gameState?.players.some((p) => p.id === user?.id);
  const isMyTurn = currentPlayer?.id === user?.id;

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
  const [chatMessages, setChatMessages] = useState<{ sender: string; text: string }[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [choiceTimeRemainingMs, setChoiceTimeRemainingMs] = useState<number | null>(null);
  const choiceTimerRef = useRef<number | null>(null);
  const lastPhaseRef = useRef<string | null>(null);
  const lastCurrentPlayerRef = useRef<number | null>(null);
  const lastChoiceIdRef = useRef<string | null>(null);

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
      return;
    }

    connectToGame(routeGameId);

    return () => {
      disconnect();
    };
  }, [routeGameId, connectToGame, disconnect]);

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
        // resolved.
        window.setTimeout(() => {
          setSandboxTurn((t) => t + 1);
        }, 0);
      }
      // Ignore clicks that are not on a highlighted landing square.
      return;
    }

    const engine = sandboxEngineRef.current;
    if (engine) {
      const stateBefore = engine.getGameState();

      // Ring-placement phase: a single click attempts a 1-ring placement
      // via the engine. On success, we immediately highlight the legal
      // movement targets for the newly placed/updated stack, and the
      // human must then move that stack; the AI will respond only after
      // the movement step completes.
      if (stateBefore.currentPhase === 'ring_placement') {
        engine.handleHumanCellClick(pos);

        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        setSandboxTurn((t) => t + 1);
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
        void runSandboxAiTurnLoop();
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

    let placed = false;

    if (!isOccupied) {
      // Empty cell: treat as a request to place 2 rings here in a single
      // placement action when possible.
      const desiredCount = Math.min(2, maxPerPlacement);
      placed = engine.tryPlaceRings(pos, desiredCount);

      // If the desired multi-ring placement fails no-dead-placement checks,
      // fall back to a single-ring placement.
      if (!placed && desiredCount > 1) {
        placed = engine.tryPlaceRings(pos, 1);
      }
    } else {
      // Existing stack: canonical rule is exactly 1 ring per placement.
      placed = engine.tryPlaceRings(pos, 1);
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

    const placed = engine.tryPlaceRings(pos, parsed);
    if (!placed) {
      return;
    }

    setSelected(pos);
    const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
    setValidTargets(targets);
    setSandboxTurn((t) => t + 1);
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

    // Reconnection banner
    const reconnectionBanner =
      isConnecting && gameState ? (
        <div className="bg-amber-500/20 border border-amber-500/50 text-amber-200 px-4 py-2 rounded mb-4 flex items-center justify-between">
          <span>Connection lost. Reconnecting...</span>
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
          isOpen={!!victoryState}
          gameResult={victoryState}
          players={gameState.players}
          onClose={() => {
            /* Optional: allow closing to view board */
          }}
          onReturnToLobby={() => navigate('/lobby')}
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
                    Click a source stack, then click a highlighted destination to send a move to the
                    server. The backend GameEngine is the source of truth for legality and state.
                  </div>
                </div>
              ) : (
                <div className="text-slate-200">Click a cell to inspect it.</div>
              )}
            </div>

            <GameHUD
              gameState={gameState}
              currentPlayer={currentPlayer}
              instruction={getInstruction()}
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
                  // TODO: Wire up to backend socket
                  setChatMessages((prev) => [...prev, { sender: 'You', text: chatInput }]);
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

        <section className="max-w-xl p-4 rounded-md bg-slate-800 border border-slate-700 space-y-4 text-slate-100">
          {backendSandboxError && (
            <div className="p-2 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded">
              {backendSandboxError}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium mb-1 text-slate-100">
              Number of players
            </label>
            <select
              className="w-full max-w-xs px-2 py-1 rounded bg-slate-700 border border-slate-500 text-sm text-slate-100"
              value={config.numPlayers}
              onChange={(e) => handleSetupChange({ numPlayers: Number(e.target.value) })}
            >
              {[2, 3, 4].map((n) => (
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
              onChange={(e) => handleSetupChange({ boardType: e.target.value as BoardType })}
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
                    onChange={(e) => handlePlayerTypeChange(i, e.target.value as LocalPlayerType)}
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

  return (
    <div className="container mx-auto px-4 py-8 space-y-4">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-1">Game (Local Sandbox)</h1>
          <p className="text-sm text-gray-500">
            Board type: {sandboxBoardState?.type ?? config.boardType} • Players: {config.numPlayers}{' '}
            ({config.playerTypes.slice(0, config.numPlayers).join(', ')})
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
      {sandboxGameState && (
        <VictoryModal
          isOpen={!!sandboxVictoryResult}
          gameResult={sandboxVictoryResult}
          players={sandboxGameState.players}
          onClose={() => {
            /* Optional: allow closing to view board */
          }}
          onReturnToLobby={() => {
            setIsConfigured(false);
            setLocalSandbox(null);
            sandboxEngineRef.current = null;
            setSelected(undefined);
            setValidTargets([]);
            setBackendSandboxError(null);
            setSandboxPendingChoice(null);
          }}
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
        <section>
          {sandboxBoardState && (
            <BoardView
              boardType={sandboxBoardState.type}
              board={sandboxBoardState}
              selectedPosition={selected}
              validTargets={sandboxCaptureTargets.length > 0 ? sandboxCaptureTargets : validTargets}
              onCellClick={(pos) => handleSandboxCellClick(pos)}
              onCellDoubleClick={(pos) => handleSandboxCellDoubleClick(pos)}
              onCellContextMenu={(pos) => handleSandboxCellContextMenu(pos)}
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
                <div className="text-xs text-slate-300 mt-1">
                  Valid targets are highlighted on the board; click a highlighted cell to move.
                </div>
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
