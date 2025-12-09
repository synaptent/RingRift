import { useState } from 'react';
import type { Position, PlayerChoice, PlayerChoiceResponseFor } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import { useSandbox } from '../contexts/SandboxContext';
import { useInvalidMoveFeedback } from './useInvalidMoveFeedback';

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
    isConfigured: _isConfigured,
    sandboxPendingChoice,
    setSandboxPendingChoice,
    sandboxCaptureChoice,
    setSandboxCaptureChoice,
    setSandboxCaptureTargets,
    setSandboxLastProgressAt,
    setSandboxStallWarning,
    sandboxStateVersion: _sandboxStateVersion,
    setSandboxStateVersion,
  } = useSandbox();

  // Local render tick used to force re-renders for AI-vs-AI games even when
  // React state derived from GameState hasn’t otherwise changed.
  const [, setSandboxTurn] = useState(0);

  // Invalid move feedback for sandbox: mirrors backend UX with cell shake
  // animation and contextual toasts when the user clicks illegal sources
  // or targets during movement/capture/chain_capture phases.
  const { shakingCellKey, triggerInvalidMove, analyzeInvalidMove } = useInvalidMoveFeedback();

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
    // When a territory region_order choice is pending, treat clicks inside
    // any highlighted region as selecting that region. This mirrors the
    // board-driven UX for ring_elimination and capture_direction choices.
    if (sandboxPendingChoice && sandboxPendingChoice.type === 'region_order') {
      const currentChoice = sandboxPendingChoice;
      const options = (currentChoice.options ?? []) as Array<{
        regionId: string;
        size: number;
        representativePosition: Position;
        moveId: string;
      }>;

      if (options.length === 0) {
        return;
      }

      const engine = sandboxEngine;
      const state = engine?.getGameState();
      const territories = state?.board.territories;

      if (!engine || !state || !territories || territories.size === 0) {
        return;
      }

      // Identify which territory region(s) contain the clicked cell using
      // the engine's regionId keys for stability, then map that back to
      // the concrete RegionOrderChoice option list.
      const clickedRegionIds: string[] = [];
      territories.forEach((territory, regionId) => {
        const spaces = territory.spaces ?? [];
        if (spaces.some((space) => positionsEqual(space, pos))) {
          clickedRegionIds.push(regionId);
        }
      });

      let selectedOption: (typeof options)[number] | undefined;

      if (clickedRegionIds.length > 0) {
        // Prefer the first option whose regionId owns the clicked cell,
        // preserving the option ordering from the underlying choice.
        selectedOption = options.find((opt) => clickedRegionIds.includes(opt.regionId));
      }

      // Fallback: if regionId-based mapping fails (for example, in older
      // fixtures), fall back to the representative-position heuristic so
      // clicks still resolve sensibly.
      if (!selectedOption) {
        territories.forEach((territory, regionId) => {
          if (selectedOption) return;
          const spaces = territory.spaces ?? [];
          const containsClick = spaces.some((space) => positionsEqual(space, pos));
          if (!containsClick) return;

          const hasRepresentative = spaces.some((space) =>
            options.some((opt) => positionsEqual(opt.representativePosition, space))
          );
          if (hasRepresentative) {
            selectedOption = options.find((opt) =>
              spaces.some((space) => positionsEqual(space, opt.representativePosition))
            );
          } else {
            // As a final safety net, match purely on regionId when the
            // click lies in a known territory but no representative
            // position overlaps.
            selectedOption = options.find((opt) => opt.regionId === regionId);
          }
        });
      }

      if (selectedOption) {
        const resolver = choiceResolverRef.current;
        if (resolver) {
          resolver({
            choiceId: currentChoice.id,
            playerNumber: currentChoice.playerNumber,
            choiceType: currentChoice.type,
            selectedOption,
          } as PlayerChoiceResponseFor<PlayerChoice>);
        }
        choiceResolverRef.current = null;
        window.setTimeout(() => {
          setSandboxPendingChoice(null);
          setSandboxStateVersion((v) => v + 1);
          maybeRunSandboxAiIfNeeded();
        }, 0);
      }

      // Ignore clicks that are not inside a highlighted territory region.
      return;
    }

    // When a ring_elimination choice is pending, treat clicks on highlighted
    // stacks as selecting the corresponding elimination option instead of
    // sending a normal click into the engine. This keeps elimination UX
    // board-driven and avoids an extra blocking dialog in sandbox mode.
    if (sandboxPendingChoice && sandboxPendingChoice.type === 'ring_elimination') {
      const currentChoice = sandboxPendingChoice;
      const options = (currentChoice.options ?? []) as Array<{ stackPosition: Position }>;
      const matching = options.find((opt) => positionsEqual(opt.stackPosition, pos));

      if (matching) {
        const resolver = choiceResolverRef.current;
        if (resolver) {
          // Resolve on the next tick so the orchestrator can finish
          // applying the elimination decision before we trigger any
          // follow-up AI turns or re-renders.
          resolver({
            choiceId: currentChoice.id,
            playerNumber: currentChoice.playerNumber,
            choiceType: currentChoice.type,
            selectedOption: matching,
          } as PlayerChoiceResponseFor<PlayerChoice>);
        }
        choiceResolverRef.current = null;
        // Clear the pending choice and advance sandbox state version on
        // the next macrotask to give the engine time to settle into the
        // post-decision state.
        window.setTimeout(() => {
          setSandboxPendingChoice(null);
          setSandboxStateVersion((v) => v + 1);
          // After resolving elimination, immediately check whether the
          // turn has advanced to an AI seat and, if so, start the AI loop.
          maybeRunSandboxAiIfNeeded();
        }, 0);
      }
      // Ignore clicks that are not on a highlighted elimination stack.
      return;
    }

    // When a capture_direction choice is pending in the local sandbox,
    // interpret clicks as selecting one of the highlighted landing
    // squares instead of sending a normal click into the engine.
    if (sandboxCaptureChoice && sandboxCaptureChoice.type === 'capture_direction') {
      const currentChoice = sandboxCaptureChoice;
      const options = (currentChoice.options ?? []) as Array<{ landingPosition: Position }>;
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
    if (!engine) {
      return;
    }

    const stateBefore = engine.getGameState();
    const current = stateBefore.players.find((p) => p.playerNumber === stateBefore.currentPlayer);

    // If it is currently an AI player's turn in the sandbox engine, ignore
    // human clicks and ensure the AI turn loop is running instead of placing
    // rings for the AI seat.
    if (stateBefore.gameStatus === 'active' && current && current.type === 'ai') {
      maybeRunSandboxAiIfNeeded();
      return;
    }

    const phaseBefore = stateBefore.currentPhase;
    const board = stateBefore.board;

    if (phaseBefore === 'ring_placement') {
      // 1) Selected stack + click on a valid landing → synthesize skip_placement + movement
      if (selected) {
        const selectedKey = positionToString(selected);
        const selectedStack = board.stacks.get(selectedKey);

        if (selectedStack && selectedStack.controllingPlayer === stateBefore.currentPlayer) {
          const landingPositions = engine.getValidLandingPositionsForCurrentPlayer(selected);
          const isValidLanding = landingPositions.some((t) => positionsEqual(t, pos));

          if (isValidLanding && !positionsEqual(selected, pos)) {
            void (async () => {
              const skipMove = {
                id: `skip-placement-${Date.now()}`,
                type: 'skip_placement' as const,
                player: stateBefore.currentPlayer,
                to: { x: 0, y: 0 },
                timestamp: new Date(),
                thinkTime: 0,
                moveNumber: stateBefore.history.length + 1,
              };
              await engine.applyCanonicalMove(skipMove);

              const afterSkip = engine.getGameState();
              if (afterSkip.gameStatus !== 'active') {
                setSelected(undefined);
                setValidTargets([]);
                setSandboxTurn((t) => t + 1);
                setSandboxStateVersion((v) => v + 1);
                return;
              }

              const validMoves = engine.getValidMoves(afterSkip.currentPlayer);
              const moveFromSelected = validMoves.find(
                (m) =>
                  m.from && positionsEqual(m.from, selected) && m.to && positionsEqual(m.to, pos)
              );

              if (moveFromSelected) {
                await engine.applyCanonicalMove(moveFromSelected);
              }

              setSelected(undefined);
              setValidTargets([]);
              setSandboxTurn((t) => t + 1);
              setSandboxStateVersion((v) => v + 1);
              maybeRunSandboxAiIfNeeded();
            })();
            return;
          }
        }
      }

      // For all other clicks in ring_placement (empty cells or stacks that are
      // not being used for skip_placement+moved), delegate to the sandbox
      // engine's canonical handler and then mirror its selection by treating
      // the clicked cell as selected and highlighting its movement targets.
      void (async () => {
        await engine.handleHumanCellClick(pos);

        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);

        setSandboxTurn((t) => t + 1);
        setSandboxStateVersion((v) => v + 1);
        maybeRunSandboxAiIfNeeded();
      })();
      return;
    }

    // Precompute whether this click is on a currently-highlighted target.
    const isTarget = validTargets.some((t) => positionsEqual(t, pos));
    const validMoves = engine.getValidMoves(stateBefore.currentPlayer);

    // Chain-capture phase: only allow clicks on canonical continuation
    // landings exposed by the orchestrator via getValidMoves().
    if (phaseBefore === 'chain_capture') {
      if (!isTarget) {
        // Surface invalid-move feedback when clicking outside canonical
        // continuation landings during a mandatory chain capture.
        const reason = analyzeInvalidMove(stateBefore, pos, {
          isPlayer: true,
          isMyTurn: true,
          isConnected: true,
          selectedPosition: selected ?? null,
          validMoves,
        });
        triggerInvalidMove(pos, reason);
        return;
      }

      // Clear overlays while the continuation segment is processed; if the
      // chain persists, highlights will be recomputed from canonical moves.
      setSelected(undefined);
      setValidTargets([]);

      void (async () => {
        await engine.handleHumanCellClick(pos);

        const after = engine.getGameState();
        if (after.gameStatus === 'active' && after.currentPhase === 'chain_capture') {
          const ctx = engine.getChainCaptureContextForCurrentPlayer();
          if (ctx) {
            setSelected(ctx.from);
            setValidTargets(ctx.landings);
          }
        }

        setSandboxTurn((t) => t + 1);
        setSandboxStateVersion((v) => v + 1);
        // After a successful human move (which may trigger line/territory
        // processing and advance the turn), immediately check whether it is
        // now an AI player's turn and, if so, start the AI loop.
        maybeRunSandboxAiIfNeeded();
      })();

      return;
    }

    // Movement/capture phase: mirror backend UX – first click selects a stack
    // and highlights its legal landing positions; second click on a
    // highlighted cell executes the move. In capture phase, also allow
    // clicking directly on a highlighted capture landing (with no prior
    // stack selection) to apply the overtaking_capture.
    if (!selected) {
      // For non-movement/capture phases, retain the previous behaviour
      // and let the engine interpret the click without invalid-move UX.
      if (phaseBefore !== 'movement' && phaseBefore !== 'capture') {
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        engine.handleHumanCellClick(pos);
        return;
      }

      // Capture phase: when no stack is currently selected, treat clicks on
      // canonical capture landings (derived from getValidMoves) as executing
      // the overtaking_capture directly. This keeps sandbox UX aligned with
      // the board-level capture highlights, where landing cells pulse even
      // before the attacking stack is explicitly selected.
      if (phaseBefore === 'capture') {
        const captureMoves = validMoves.filter((m) => m.type === 'overtaking_capture');
        const isCaptureLanding = captureMoves.some(
          (m) => m.to && positionsEqual(m.to as Position, pos)
        );

        if (isCaptureLanding) {
          setSelected(undefined);
          setValidTargets([]);

          void (async () => {
            await engine.handleHumanCellClick(pos);

            const after = engine.getGameState();
            if (after.gameStatus === 'active' && after.currentPhase === 'chain_capture') {
              const ctx = engine.getChainCaptureContextForCurrentPlayer();
              if (ctx) {
                setSelected(ctx.from);
                setValidTargets(ctx.landings);
              }
            }

            setSandboxTurn((t) => t + 1);
            setSandboxStateVersion((v) => v + 1);
            maybeRunSandboxAiIfNeeded();
          })();

          return;
        }
      }

      const key = positionToString(pos);
      const hasStack = !!board.stacks.get(key);
      const hasMovesFromHere = validMoves.some(
        (m) => m.from && positionsEqual(m.from as Position, pos)
      );

      if (hasStack && hasMovesFromHere) {
        // Selection click: record selected cell and highlight valid targets.
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        // Inform the engine about the selection so its internal
        // movement state (_selectedStackKey) matches the UI.
        engine.handleHumanCellClick(pos);
      } else {
        const reason = analyzeInvalidMove(stateBefore, pos, {
          isPlayer: true,
          isMyTurn: true,
          isConnected: true,
          selectedPosition: null,
          validMoves,
        });
        triggerInvalidMove(pos, reason);
      }
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
    if (isTarget) {
      // Clear selection/highlights immediately so movement overlays do not
      // persist while the orchestrator processes post-move consequences
      // (lines, territory, and any ring_elimination decisions).
      setSelected(undefined);
      setValidTargets([]);
      void (async () => {
        await engine.handleHumanCellClick(pos);

        const after = engine.getGameState();
        if (after.gameStatus === 'active' && after.currentPhase === 'chain_capture') {
          const ctx = engine.getChainCaptureContextForCurrentPlayer();
          if (ctx) {
            setSelected(ctx.from);
            setValidTargets(ctx.landings);
          }
        }

        setSandboxTurn((t) => t + 1);
        setSandboxStateVersion((v) => v + 1);
        // After a successful human move (which may trigger line/territory
        // processing and advance the turn), immediately check whether it is
        // now an AI player's turn and, if so, start the AI loop.
        maybeRunSandboxAiIfNeeded();
      })();
      return;
    }

    // Otherwise, treat clicks on non-highlighted cells while a stack is
    // selected as invalid landings and surface feedback instead of silently
    // ignoring them.
    if (phaseBefore === 'movement' || phaseBefore === 'capture') {
      const reason = analyzeInvalidMove(stateBefore, pos, {
        isPlayer: true,
        isMyTurn: true,
        isConnected: true,
        selectedPosition: selected,
        validMoves,
      });
      triggerInvalidMove(pos, reason);
    }
    return;
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

  return {
    shakingCellKey,
    handleCellClick,
    handleCellDoubleClick,
    handleCellContextMenu,
    maybeRunSandboxAiIfNeeded,
    clearSelection,
  };
}
