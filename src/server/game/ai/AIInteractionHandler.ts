import {
  PlayerChoice,
  PlayerChoiceResponse,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice
} from '../../../shared/types/game';
import { PlayerInteractionHandler } from '../PlayerInteractionManager';
import { globalAIEngine } from './AIEngine';

/**
 * AIInteractionHandler
 *
 * Implements PlayerInteractionHandler for AI-controlled players using
 * simple, deterministic heuristics based solely on the information
 * contained in the PlayerChoice payload. This keeps the implementation
 * independent of transport and full GameState while still providing
 * reasonable, rules-respecting defaults for AI decisions.
 *
 * The heuristics here are deliberately lightweight and can be gradually
 * replaced or augmented by Python AI service endpoints in the future.
 */
export class AIInteractionHandler implements PlayerInteractionHandler {
  async requestChoice(choice: PlayerChoice): Promise<PlayerChoiceResponse<unknown>> {
    const selectedOption = await this.selectOption(choice);

    return {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      selectedOption
    };
  }

  /**
   * Core selection logic for all PlayerChoice variants.
   *
   * For most choice types this is a pure, synchronous heuristic. For
   * line_reward_option, we first attempt to delegate to the Python AI
   * service via globalAIEngine.getLineRewardChoice (when an AI
   * configuration exists), and fall back to the local heuristic when
   * the service is unavailable or unconfigured.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async selectOption(choice: PlayerChoice): Promise<any> {
    switch (choice.type) {
      case 'line_order':
        return this.selectLineOrderOption(choice as LineOrderChoice);
      case 'line_reward_option':
        return this.selectLineRewardOption(choice as LineRewardChoice);
      case 'ring_elimination':
        return this.selectRingEliminationOption(choice as RingEliminationChoice);
      case 'region_order':
        return this.selectRegionOrderOption(choice as RegionOrderChoice);
      case 'capture_direction':
        return this.selectCaptureDirectionOption(choice as CaptureDirectionChoice);
      default:
        // Fallback: first option, if present. This mirrors the engine's
        // historical "first option" behaviour and ensures we always
        // return a valid member of choice.options.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const anyChoice = choice as any;
        return anyChoice.options?.[0];
    }
  }

  /**
   * Line order heuristic: prefer the line with the greatest number of
   * markers (longest line), falling back to the first option.
   */
  private selectLineOrderOption(choice: LineOrderChoice): LineOrderChoice['options'][number] {
    if (!choice.options.length) {
      // Should not happen in practice, but keep behaviour well-defined.
      return { lineId: '0', markerPositions: [] };
    }

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.markerPositions.length > best.markerPositions.length) {
        best = opt;
      }
    }

    return best;
  }

  /**
   * Line reward heuristic:
   * - For now, prefer preserving rings by defaulting to Option 2
   *   (minimum collapse, no elimination) when available.
   * - If only one option is present, take it.
   */
  private async selectLineRewardOption(
    choice: LineRewardChoice
  ): Promise<LineRewardChoice['options'][number]> {
    if (!choice.options.length) {
      // Fallback to Option 2 semantics when nothing is provided.
      return 'option_2_min_collapse_no_elimination';
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player. Any errors (including missing config or service
    // failures) are swallowed and we fall back to the local
    // heuristic below.
    try {
      const selected = await globalAIEngine.getLineRewardChoice(
        choice.playerNumber,
        null,
        choice.options
      );

      // Defensive: ensure the returned option is one of the original
      // options before accepting it.
      if (choice.options.includes(selected)) {
        return selected;
      }
    } catch {
      // Ignore and fall back to heuristic behaviour.
    }

    const hasOption2 = choice.options.includes('option_2_min_collapse_no_elimination');
    if (hasOption2) {
      return 'option_2_min_collapse_no_elimination';
    }

    return choice.options[0];
  }

  /**
   * Ring elimination heuristic:
   * - Prefer eliminating from the stack with the smallest capHeight to
   *   minimise immediate material loss.
   * - If tied, break ties by choosing the stack with the smallest
   *   totalHeight.
   */
  private async selectRingEliminationOption(
    choice: RingEliminationChoice
  ): Promise<RingEliminationChoice['options'][number]> {
    if (!choice.options.length) {
      // Synthetic, never used directly by engine; keeps type safety.
      return {
        stackPosition: { x: 0, y: 0 },
        capHeight: 0,
        totalHeight: 0
      };
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player. Any errors (including missing config or service
    // failures) are swallowed and we fall back to the local
    // heuristic below.
    try {
      const selected = await globalAIEngine.getRingEliminationChoice(
        choice.playerNumber,
        null,
        choice.options
      );

      // Defensive: ensure the returned option is one of the original
      // options before accepting it.
      if (choice.options.includes(selected)) {
        return selected;
      }
    } catch {
      // Ignore and fall back to heuristic behaviour.
    }

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.capHeight < best.capHeight) {
        best = opt;
      } else if (opt.capHeight === best.capHeight && opt.totalHeight < best.totalHeight) {
        best = opt;
      }
    }

    return best;
  }

  /**
   * Region order heuristic:
   * - Prefer processing the largest disconnected region first, on the
   *   assumption that larger regions represent more impactful swings in
   *   territory and ring elimination.
   * - When available, delegate to the Python AI service via
   *   globalAIEngine.getRegionOrderChoice, falling back to this local
   *   heuristic on error or missing configuration.
   */
  private async selectRegionOrderOption(
    choice: RegionOrderChoice
  ): Promise<RegionOrderChoice['options'][number]> {
    if (!choice.options.length) {
      return {
        regionId: '0',
        size: 0,
        representativePosition: { x: 0, y: 0 }
      };
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player. Any errors (including missing config or service
    // failures) are swallowed and we fall back to the local
    // heuristic below.
    try {
      const selected = await globalAIEngine.getRegionOrderChoice(
        choice.playerNumber,
        null,
        choice.options
      );

      // Defensive: ensure the returned option is one of the original
      // options before accepting it.
      if (choice.options.includes(selected)) {
        return selected;
      }
    } catch {
      // Ignore and fall back to heuristic behaviour.
    }

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.size > best.size) {
        best = opt;
      }
    }

    return best;
  }

  /**
   * Capture direction heuristic:
   * - Prefer the capture option that removes the largest captured
   *   capHeight (maximising immediate material gain).
   * - If tied, prefer the landing position that is closest to the
   *   centre of the board (using a simple Manhattan distance in
   *   coordinate space) as a proxy for central control.
   */
  private selectCaptureDirectionOption(
    choice: CaptureDirectionChoice
  ): CaptureDirectionChoice['options'][number] {
    if (!choice.options.length) {
      return {
        targetPosition: { x: 0, y: 0 },
        landingPosition: { x: 0, y: 0 },
        capturedCapHeight: 0
      };
    }

    // If only one option, no need to compute distances.
    if (choice.options.length === 1) {
      return choice.options[0];
    }

    const centre = this.estimateBoardCentre(choice.options[0].landingPosition);

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.capturedCapHeight > best.capturedCapHeight) {
        best = opt;
        continue;
      }

      if (opt.capturedCapHeight === best.capturedCapHeight) {
        const bestDist = this.manhattanDistance(best.landingPosition, centre);
        const optDist = this.manhattanDistance(opt.landingPosition, centre);
        if (optDist < bestDist) {
          best = opt;
        }
      }
    }

    return best;
  }

  /**
   * Roughly estimate the board centre based on any position; this avoids
   * needing full BoardState/BoardConfig while still giving us a stable
   * reference point for distance-based heuristics.
   */
  private estimateBoardCentre(reference: { x: number; y: number; z?: number }): {
    x: number;
    y: number;
    z?: number;
  } {
    // For now, assume coordinates are roughly centred around (0, 0) or
    // positive ranges starting at 0. We use a symmetric centre guess
    // based on the reference position as a no-op baseline; this is
    // mostly to provide a consistent Manhattan metric between options.
    const centre: { x: number; y: number; z?: number } = {
      x: reference.x,
      y: reference.y
    };

    if (reference.z !== undefined) {
      centre.z = reference.z;
    }

    return centre;
  }

  private manhattanDistance(a: { x: number; y: number; z?: number }, b: { x: number; y: number; z?: number }): number {
    const dzA = a.z ?? 0;
    const dzB = b.z ?? 0;
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(dzA - dzB);
  }
}
