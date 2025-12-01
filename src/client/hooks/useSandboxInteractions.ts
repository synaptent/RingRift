import { useEffect, useState } from 'react';
import type { Position, PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';

interface UseSandboxInteractionsOptions {
  selected: Position | undefined;
  setSelected: React.Dispatch<React.SetStateAction<Position | undefined>>;
  validTargets: Position[];
  setValidTargets: React.Dispatch<React.SetStateAction<Position[]>>;
  choiceResolverRef: React.MutableRefObject<
    ((response: PlayerChoiceResponseFor<PlayerChoice>) => void) | null
  >;
}

export function useSandboxInteractions({
  selected,
  setSelected,
  validTargets,
  setValidTargets,
  choiceResolverRef,
}: UseSandboxInteractionsOptions) {
  const {
    sandboxEngine,
    isConfigured,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    sandboxCaptureTargets,
    setSandboxCaptureTargets,
    setSandboxLastProgressAt,
    setSandboxStallWarning,
    sandboxStateVersion,
    setSandboxStateVersion,
  } = useSandbox();

  // Local render tick used to force re-renders for AI-vs-AI games even when
  // React state derived from GameState hasn’t otherwise changed.
  const [, setSandboxTurn] = useState(0);

  const runSandboxAiTurnLoop = async () => {
    const engine = sandboxEngine;
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
      // eslint-disable-next-line no-await-in-loop
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
    const engine = sandboxEngine;
    if (!engine) return;

    const state = engine.getGameState();
    const current = state.players.find((p) => p.playerNumber === state.currentPlayer);
    if (state.gameStatus === 'active' && current && current.type === 'ai') {
      void runSandboxAiTurnLoop();
    }
  };

  // Unified sandbox click handler: prefer the ClientSandboxEngine when
  // available (Stage 2 harness), otherwise fall back to the legacy
  // LocalSandboxState controller.
  const handleCellClick = (pos: Position) => {
    // When a capture_direction choice is pending in the local sandbox,
    // interpret clicks as selecting one of the highlighted landing
    // squares instead of sending a normal click into the engine.
    if (sandboxCaptureChoice && sandboxCaptureChoice.type === 'capture_direction') {
      const currentChoice: any = sandboxCaptureChoice;
      const options: any[] = (currentChoice.options ?? []) as any[];
      const matching = options.find((opt) => positionsEqual(opt.landingPosition, pos));

      if (matching) {
        const resolver = choiceResolverRef.current;
        if (resolver) {
          resolver({
            choiceId: currentChoice.id,
            playerNumber: currentChoice.playerNumber,
            choiceType: currentChoice.type,
            selectedOption: matching,
          } as PlayerChoiceResponseFor<PlayerChoice>);
        }
        choiceResolverRef.current = null;
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

    const engine = sandboxEngine;
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
  const handleCellDoubleClick = (pos: Position) => {
    const engine = sandboxEngine;
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
  const handleCellContextMenu = (pos: Position) => {
    const engine = sandboxEngine;
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
   * Explicit selection clearer used by touch-centric sandbox controls.
   * This keeps BoardView highlights and the sandbox engine's internal
   * selection state (_selectedStackKey) in sync without embedding any
   * rules logic in the host.
   */
  const clearSelection = () => {
    setSelected(undefined);
    setValidTargets([]);
    if (sandboxEngine) {
      sandboxEngine.clearSelection();
    }
  };

  // Auto-trigger AI turns when state version changes (after human moves).
  useEffect(() => {
    if (!isConfigured || !sandboxEngine) {
      return;
    }

    const engine = sandboxEngine;
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

  return {
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection,
  };
}
