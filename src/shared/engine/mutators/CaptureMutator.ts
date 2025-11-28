import type {
  GameState as EngineGameState,
  OvertakingCaptureAction,
  ContinueChainAction,
} from '../types';
import type { GameState as SharedGameState } from '../../types/game';
import { mutateCapture as mutateCaptureAggregate } from '../aggregates/CaptureAggregate';

/**
 * Legacy capture mutator shim.
 *
 * For P1.2 the canonical capture mutation logic lives in
 * {@link CaptureAggregate.mutateCapture}. This module now delegates
 * to that implementation while keeping the existing Engine GameState
 * type used by the shared GameEngine wiring.
 *
 * The two GameState types (`shared/types/game` vs `shared/engine/types`)
 * are structurally compatible at runtime for the fields touched by
 * capture, so this adapter performs a narrow cast between them.
 */
export function mutateCapture(
  state: EngineGameState,
  action: OvertakingCaptureAction | ContinueChainAction
): EngineGameState {
  const nextShared = mutateCaptureAggregate(state as unknown as SharedGameState, action as any);
  return nextShared as unknown as EngineGameState;
}
