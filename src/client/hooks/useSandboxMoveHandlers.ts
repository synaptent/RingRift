/**
 * @fileoverview useSandboxMoveHandlers Hook - ADAPTER, NOT CANONICAL
 *
 * SSoT alignment: This hook is a **React adapter** over the sandbox engine.
 * It handles UI for movement and capture clicks, not rules logic.
 *
 * Canonical SSoT:
 * - Sandbox engine: `src/client/sandbox/ClientSandboxEngine.ts`
 * - Movement: `src/shared/engine/aggregates/Movement.ts`
 * - Capture: `src/shared/engine/aggregates/Capture.ts`
 *
 * This adapter:
 * - Ring placement phase clicks
 * - Movement/capture phase selection and execution
 * - Chain capture continuation
 * - Recovery slide handling
 *
 * DO NOT add rules logic here - it belongs in `src/shared/engine/`.
 *
 * @see docs/architecture/FSM_MIGRATION_STATUS_2025_12.md
 * @see docs/rules/SSOT_BANNER_GUIDE.md
 */

import { useCallback } from 'react';
import type { Position } from '../../shared/types/game';
import { positionToString, positionsEqual } from '../../shared/types/game';
import type { InvalidMoveReason, AnalyzeInvalidMoveOptions } from './useInvalidMoveFeedback';
import type { GameState, Move } from '../../shared/types/game';
import type { ClientSandboxEngine } from '../sandbox/ClientSandboxEngine';

export interface UseSandboxMoveHandlersOptions {
  sandboxEngine: ClientSandboxEngine | null;
  selected: Position | undefined;
  setSelected: React.Dispatch<React.SetStateAction<Position | undefined>>;
  validTargets: Position[];
  setValidTargets: React.Dispatch<React.SetStateAction<Position[]>>;
  bumpSandboxTurn: () => void;
  setSandboxStateVersion: (fn: (v: number) => number) => void;
  maybeRunSandboxAiIfNeeded: () => void;
  requestRecoveryChoice: () => Promise<'option1' | 'option2' | null>;
  analyzeInvalidMove: (state: GameState, pos: Position, opts: AnalyzeInvalidMoveOptions) => InvalidMoveReason;
  triggerInvalidMove: (pos: Position, reason: InvalidMoveReason) => void;
}

export interface UseSandboxMoveHandlersReturn {
  handleRingPlacementClick: (pos: Position) => void;
  handleChainCaptureClick: (pos: Position) => void;
  handleFirstClick: (pos: Position) => void;
  handleTargetClick: (pos: Position) => void;
}

export function useSandboxMoveHandlers({
  sandboxEngine,
  selected,
  setSelected,
  validTargets,
  setValidTargets,
  bumpSandboxTurn,
  setSandboxStateVersion,
  maybeRunSandboxAiIfNeeded,
  requestRecoveryChoice,
  analyzeInvalidMove,
  triggerInvalidMove,
}: UseSandboxMoveHandlersOptions): UseSandboxMoveHandlersReturn {
  const handleRingPlacementClick = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine) return;

      const stateBefore = engine.getGameState();
      const board = stateBefore.board;

      // Check for skip_placement + movement shortcut
      if (selected) {
        const selectedKey = positionToString(selected);
        const selectedStack = board.stacks.get(selectedKey);

        if (selectedStack && selectedStack.controllingPlayer === stateBefore.currentPlayer) {
          const landingPositions = engine.getValidLandingPositionsForCurrentPlayer(selected);
          const isValidLanding = landingPositions.some((t: Position) => positionsEqual(t, pos));

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
                bumpSandboxTurn();
                setSandboxStateVersion((v) => v + 1);
                return;
              }

              const validMoves = engine.getValidMoves(afterSkip.currentPlayer);
              const moveFromSelected = validMoves.find(
                (m: Move) =>
                  m.from && positionsEqual(m.from, selected) && m.to && positionsEqual(m.to, pos)
              );
              if (moveFromSelected) await engine.applyCanonicalMove(moveFromSelected);

              setSelected(undefined);
              setValidTargets([]);
              bumpSandboxTurn();
              setSandboxStateVersion((v) => v + 1);
              maybeRunSandboxAiIfNeeded();
            })();
            return;
          }
        }
      }

      // Standard ring placement click
      void (async () => {
        await engine.handleHumanCellClick(pos);
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        bumpSandboxTurn();
        setSandboxStateVersion((v) => v + 1);
        maybeRunSandboxAiIfNeeded();
      })();
    },
    [
      sandboxEngine,
      selected,
      setSelected,
      setValidTargets,
      bumpSandboxTurn,
      setSandboxStateVersion,
      maybeRunSandboxAiIfNeeded,
    ]
  );

  const handleChainCaptureClick = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine) return;

      const stateBefore = engine.getGameState();
      const validMoves = engine.getValidMoves(stateBefore.currentPlayer);
      const isTarget = validTargets.some((t) => positionsEqual(t, pos));

      if (!isTarget) {
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
        bumpSandboxTurn();
        setSandboxStateVersion((v) => v + 1);
        maybeRunSandboxAiIfNeeded();
      })();
    },
    [
      sandboxEngine,
      selected,
      validTargets,
      setSelected,
      setValidTargets,
      bumpSandboxTurn,
      setSandboxStateVersion,
      maybeRunSandboxAiIfNeeded,
      analyzeInvalidMove,
      triggerInvalidMove,
    ]
  );

  const handleFirstClick = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine) return;

      const stateBefore = engine.getGameState();
      const phaseBefore = stateBefore.currentPhase;
      const board = stateBefore.board;
      const validMoves = engine.getValidMoves(stateBefore.currentPlayer);

      if (phaseBefore !== 'movement' && phaseBefore !== 'capture') {
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
        engine.handleHumanCellClick(pos);
        return;
      }

      // Capture phase: direct landing click
      if (phaseBefore === 'capture') {
        const captureMoves = validMoves.filter((m: Move) => m.type === 'overtaking_capture');
        const isCaptureLanding = captureMoves.some(
          (m: Move) => m.to && positionsEqual(m.to as Position, pos)
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
          })();
          return;
        }
      }

      const key = positionToString(pos);
      const hasStack = !!board.stacks.get(key);
      const hasMovesFromHere = validMoves.some((m: Move) => m.from && positionsEqual(m.from, pos));
      const hasRecoveryFromHere = validMoves.some(
        (m: Move) => m.type === 'recovery_slide' && m.from && positionsEqual(m.from, pos)
      );

      if ((hasStack && hasMovesFromHere) || hasRecoveryFromHere) {
        setSelected(pos);
        const targets = engine.getValidLandingPositionsForCurrentPlayer(pos);
        setValidTargets(targets);
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
    },
    [sandboxEngine, setSelected, setValidTargets, analyzeInvalidMove, triggerInvalidMove]
  );

  const handleTargetClick = useCallback(
    (pos: Position) => {
      const engine = sandboxEngine;
      if (!engine || !selected) return;

      const sourcePos = selected;
      const validMoves = engine.getValidMoves(engine.getGameState().currentPlayer);

      setSelected(undefined);
      setValidTargets([]);

      void (async () => {
        // Check for recovery_slide moves
        const matchingRecoveryMoves = validMoves.filter(
          (m: Move) =>
            m.type === 'recovery_slide' &&
            m.from &&
            positionsEqual(m.from, sourcePos) &&
            m.to &&
            positionsEqual(m.to, pos)
        );

        if (matchingRecoveryMoves.length > 0) {
          let selectedMove = matchingRecoveryMoves[0];

          if (matchingRecoveryMoves.length > 1) {
            const option1Move = matchingRecoveryMoves.find((m: Move) => m.recoveryOption === 1);
            const option2Move = matchingRecoveryMoves.find((m: Move) => m.recoveryOption === 2);
            if (option1Move && option2Move) {
              const choice = await requestRecoveryChoice();
              if (choice === 'option2') selectedMove = option2Move;
              else if (choice === 'option1') selectedMove = option1Move;
              else {
                setSelected(sourcePos);
                const targets = engine.getValidLandingPositionsForCurrentPlayer(sourcePos);
                setValidTargets(targets);
                return;
              }
            }
          }

          await engine.applyCanonicalMove(selectedMove);
        } else {
          await engine.handleHumanCellClick(pos);
        }

        const after = engine.getGameState();
        if (after.gameStatus === 'active' && after.currentPhase === 'chain_capture') {
          const ctx = engine.getChainCaptureContextForCurrentPlayer();
          if (ctx) {
            setSelected(ctx.from);
            setValidTargets(ctx.landings);
          }
        }

        bumpSandboxTurn();
        setSandboxStateVersion((v) => v + 1);
        maybeRunSandboxAiIfNeeded();
      })();
    },
    [
      sandboxEngine,
      selected,
      setSelected,
      setValidTargets,
      bumpSandboxTurn,
      setSandboxStateVersion,
      maybeRunSandboxAiIfNeeded,
      requestRecoveryChoice,
    ]
  );

  return {
    handleRingPlacementClick,
    handleChainCaptureClick,
    handleFirstClick,
    handleTargetClick,
  };
}

export default useSandboxMoveHandlers;
