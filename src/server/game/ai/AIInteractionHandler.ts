import {
  PlayerChoice,
  PlayerChoiceResponse,
  LineOrderChoice,
  LineRewardChoice,
  RingEliminationChoice,
  RegionOrderChoice,
  CaptureDirectionChoice,
} from '../../../shared/types/game';
import { PlayerInteractionHandler } from '../PlayerInteractionManager';
import { globalAIEngine } from './AIEngine';
import {
  createLinkedCancellationSource,
  type CancellationToken,
} from '../../../shared/utils/cancellation';
import { logger } from '../../utils/logger';
import { getMetricsService, type AIChoiceOutcome } from '../../services/MetricsService';

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
  private readonly sessionToken: CancellationToken | null;

  constructor(sessionToken?: CancellationToken) {
    this.sessionToken = sessionToken ?? null;
  }
  async requestChoice(choice: PlayerChoice): Promise<PlayerChoiceResponse<unknown>> {
    const selectedOption = await this.selectOption(choice);

    return {
      choiceId: choice.id,
      playerNumber: choice.playerNumber,
      selectedOption,
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
   *
   * Returns the selected option from the choice's options array, typed
   * according to the specific choice variant.
   */
  private async selectOption(choice: PlayerChoice): Promise<PlayerChoice['options'][number]> {
    switch (choice.type) {
      case 'line_order':
        return this.selectLineOrderOption(choice);
      case 'line_reward_option':
        return this.selectLineRewardOption(choice);
      case 'ring_elimination':
        return this.selectRingEliminationOption(choice);
      case 'region_order':
        return this.selectRegionOrderOption(choice);
      case 'capture_direction':
        return this.selectCaptureDirectionOption(choice);
      default: {
        // TypeScript exhaustiveness check: if this line is reached,
        // a new choice type was added without updating this switch.
        // The never type assertion ensures compile-time errors for
        // unhandled cases.
        const exhaustiveCheck: never = choice;
        logger.error('AIInteractionHandler received unknown choice type', {
          choiceId: (exhaustiveCheck as PlayerChoice).id,
          choiceType: (exhaustiveCheck as PlayerChoice).type,
          playerNumber: (exhaustiveCheck as PlayerChoice).playerNumber,
        });
        throw new Error(`Unhandled PlayerChoice type: ${(exhaustiveCheck as PlayerChoice).type}`);
      }
    }
  }

  /**
   * Line order heuristic: prefer the line with the greatest number of
   * markers (longest line). A line_order choice with no options is a
   * protocol violation and is treated as a hard error.
   */
  private async selectLineOrderOption(
    choice: LineOrderChoice
  ): Promise<LineOrderChoice['options'][number]> {
    const startedAt = Date.now();
    let outcome: AIChoiceOutcome = 'success';

    if (!choice.options.length) {
      logger.error('AIInteractionHandler received line_order choice with no options', {
        choiceId: choice.id,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
      });
      this.recordChoiceMetrics(choice.type, 'error', startedAt);
      throw new Error('PlayerChoice[line_order] must have at least one option');
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player and the AI is running in `service` mode. Any errors
    // (including missing config or service failures) are swallowed
    // and we fall back to the local heuristic below.
    const config = globalAIEngine.getAIConfig(choice.playerNumber);
    const mode = config?.mode ?? 'service';

    if (mode === 'service') {
      try {
        const tokenSource =
          this.sessionToken != null ? createLinkedCancellationSource(this.sessionToken) : null;

        if (tokenSource) {
          tokenSource.syncFromParent();
        }

        const selected = await globalAIEngine.getLineOrderChoice(
          choice.playerNumber,
          null,
          choice.options,
          tokenSource ? { token: tokenSource.token } : undefined
        );

        if (choice.options.includes(selected)) {
          this.recordChoiceMetrics(choice.type, outcome, startedAt);
          return selected;
        }

        logger.warn(
          'AI service returned invalid option for line_order; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            optionsCount: choice.options.length,
            invalidOption: selected,
          }
        );
        outcome = 'fallback';
      } catch (error) {
        const aiErrorType = (error as any)?.aiErrorType;
        outcome = aiErrorType === 'timeout' ? 'timeout' : 'fallback';
        logger.warn('AI service unavailable for line_order; falling back to local heuristic', {
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
          choiceType: choice.type,
          error: error instanceof Error ? error.message : String(error),
          aiErrorType,
        });
      }
    }

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.markerPositions.length > best.markerPositions.length) {
        best = opt;
      }
    }

    this.recordChoiceMetrics(choice.type, outcome, startedAt);
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
    const startedAt = Date.now();
    let outcome: AIChoiceOutcome = 'success';

    if (!choice.options.length) {
      logger.error('AIInteractionHandler received line_reward_option choice with no options', {
        choiceId: choice.id,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
      });
      this.recordChoiceMetrics(choice.type, 'error', startedAt);
      throw new Error('PlayerChoice[line_reward_option] must have at least one option');
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player and the AI is running in `service` mode. Any errors
    // (including missing config or service failures) are swallowed
    // and we fall back to the local heuristic below.
    const config = globalAIEngine.getAIConfig(choice.playerNumber);
    const mode = config?.mode ?? 'service';

    if (mode === 'service') {
      try {
        const selected = await globalAIEngine.getLineRewardChoice(
          choice.playerNumber,
          null,
          choice.options
        );

        // Defensive: ensure the returned option is one of the original
        // options before accepting it.
        if (choice.options.includes(selected)) {
          this.recordChoiceMetrics(choice.type, outcome, startedAt);
          return selected;
        }

        logger.warn(
          'AI service returned invalid option for line_reward_option; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            optionsCount: choice.options.length,
            invalidOption: selected,
          }
        );
      } catch (error) {
        // Service is unavailable or misconfigured for this player. Log a
        // structured warning and fall back to the local heuristic; this is
        // treated as a degraded AI mode for non-move decisions.
        const aiErrorType = (error as any)?.aiErrorType;
        outcome = aiErrorType === 'timeout' ? 'timeout' : 'fallback';
        logger.warn(
          'AI service unavailable for line_reward_option; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            error: error instanceof Error ? error.message : String(error),
            aiErrorType,
          }
        );
      }
    }

    const hasOption2 = choice.options.includes('option_2_min_collapse_no_elimination');
    if (hasOption2) {
      this.recordChoiceMetrics(choice.type, outcome, startedAt);
      return 'option_2_min_collapse_no_elimination';
    }

    const selected = choice.options[0];
    this.recordChoiceMetrics(choice.type, outcome, startedAt);
    return selected;
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
    const startedAt = Date.now();
    let outcome: AIChoiceOutcome = 'success';

    if (!choice.options.length) {
      logger.error('AIInteractionHandler received ring_elimination choice with no options', {
        choiceId: choice.id,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
      });
      this.recordChoiceMetrics(choice.type, 'error', startedAt);
      throw new Error('PlayerChoice[ring_elimination] must have at least one option');
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player and the AI is running in `service` mode. Any errors
    // (including missing config or service failures) are swallowed
    // and we fall back to the local heuristic below.
    const config = globalAIEngine.getAIConfig(choice.playerNumber);
    const mode = config?.mode ?? 'service';

    if (mode === 'service') {
      try {
        const tokenSource =
          this.sessionToken != null ? createLinkedCancellationSource(this.sessionToken) : null;

        if (tokenSource) {
          tokenSource.syncFromParent();
        }

        const selected = await globalAIEngine.getRingEliminationChoice(
          choice.playerNumber,
          null,
          choice.options,
          tokenSource ? { token: tokenSource.token } : undefined
        );

        // Defensive: ensure the returned option is one of the original
        // options before accepting it.
        if (choice.options.includes(selected)) {
          this.recordChoiceMetrics(choice.type, outcome, startedAt);
          return selected;
        }

        logger.warn(
          'AI service returned invalid option for ring_elimination; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            optionsCount: choice.options.length,
            invalidOption: selected,
          }
        );
      } catch (error) {
        // Service is unavailable or misconfigured for this player. Log a
        // structured warning and fall back to the local heuristic.
        const aiErrorType = (error as any)?.aiErrorType;
        outcome = aiErrorType === 'timeout' ? 'timeout' : 'fallback';
        logger.warn(
          'AI service unavailable for ring_elimination; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            error: error instanceof Error ? error.message : String(error),
            aiErrorType,
          }
        );
      }
    }

    let best = choice.options[0];

    for (const opt of choice.options) {
      if (opt.capHeight < best.capHeight) {
        best = opt;
      } else if (opt.capHeight === best.capHeight && opt.totalHeight < best.totalHeight) {
        best = opt;
      }
    }

    this.recordChoiceMetrics(choice.type, outcome, startedAt);
    return best;
  }

  /**
   * Region order heuristic:
   * - Prefer processing the largest disconnected region first, on the
   *   assumption that larger regions represent more impactful swings in
   *   territory and ring elimination.
   * - When a canonical skip option is present (regionId === 'skip' or
   *   size <= 0) and no concrete regions remain, fall back to that
   *   skip option so the AI can explicitly decline further processing.
   * - When available, delegate to the Python AI service via
   *   globalAIEngine.getRegionOrderChoice, falling back to this local
   *   heuristic on error or missing configuration.
   */
  private async selectRegionOrderOption(
    choice: RegionOrderChoice
  ): Promise<RegionOrderChoice['options'][number]> {
    const startedAt = Date.now();
    let outcome: AIChoiceOutcome = 'success';

    if (!choice.options.length) {
      logger.error('AIInteractionHandler received region_order choice with no options', {
        choiceId: choice.id,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
      });
      this.recordChoiceMetrics(choice.type, 'error', startedAt);
      throw new Error('PlayerChoice[region_order] must have at least one option');
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player and the AI is running in `service` mode. Any errors
    // (including missing config or service failures) are swallowed
    // and we fall back to the local heuristic below.
    const config = globalAIEngine.getAIConfig(choice.playerNumber);
    const mode = config?.mode ?? 'service';

    if (mode === 'service') {
      try {
        const tokenSource =
          this.sessionToken != null ? createLinkedCancellationSource(this.sessionToken) : null;

        if (tokenSource) {
          tokenSource.syncFromParent();
        }

        const selected = await globalAIEngine.getRegionOrderChoice(
          choice.playerNumber,
          null,
          choice.options,
          tokenSource ? { token: tokenSource.token } : undefined
        );

        // Defensive: ensure the returned option is one of the original
        // options before accepting it.
        if (choice.options.includes(selected)) {
          this.recordChoiceMetrics(choice.type, outcome, startedAt);
          return selected;
        }

        logger.warn(
          'AI service returned invalid option for region_order; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            optionsCount: choice.options.length,
            invalidOption: selected,
          }
        );
      } catch (error) {
        // Service is unavailable or misconfigured for this player. Log a
        // structured warning and fall back to the local heuristic.
        const aiErrorType = (error as any)?.aiErrorType;
        outcome = aiErrorType === 'timeout' ? 'timeout' : 'fallback';
        logger.warn('AI service unavailable for region_order; falling back to local heuristic', {
          gameId: choice.gameId,
          playerNumber: choice.playerNumber,
          choiceId: choice.id,
          choiceType: choice.type,
          error: error instanceof Error ? error.message : String(error),
          aiErrorType,
        });
      }
    }

    // Partition options into concrete regions vs skip/meta options. In both
    // backend and sandbox flows, skip_territory_processing is represented as
    // a RegionOrderChoice option with regionId === 'skip' and size <= 0.
    const regionOptions = choice.options.filter((opt) => opt.regionId !== 'skip' && opt.size > 0);
    const skipOptions = choice.options.filter((opt) => opt.regionId === 'skip' || opt.size <= 0);

    // If for some reason only skip-like options are present (no concrete
    // regions), prefer skipping further processing rather than throwing.
    if (regionOptions.length === 0 && skipOptions.length > 0) {
      const selectedSkip = skipOptions[0];
      this.recordChoiceMetrics(choice.type, outcome, startedAt);
      return selectedSkip;
    }

    const candidates = regionOptions.length > 0 ? regionOptions : choice.options;

    let best = candidates[0];

    for (const opt of candidates) {
      if (opt.size > best.size) {
        best = opt;
      }
    }

    this.recordChoiceMetrics(choice.type, outcome, startedAt);
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
  private async selectCaptureDirectionOption(
    choice: CaptureDirectionChoice
  ): Promise<CaptureDirectionChoice['options'][number]> {
    const startedAt = Date.now();
    let outcome: AIChoiceOutcome = 'success';

    if (!choice.options.length) {
      logger.error('AIInteractionHandler received capture_direction choice with no options', {
        choiceId: choice.id,
        choiceType: choice.type,
        playerNumber: choice.playerNumber,
      });
      this.recordChoiceMetrics(choice.type, 'error', startedAt);
      throw new Error('PlayerChoice[capture_direction] must have at least one option');
    }

    // First, attempt to delegate to the Python AI service via the
    // global AI engine when an AI configuration exists for this
    // player and the AI is running in `service` mode. Any errors
    // (including missing config or service failures) are swallowed
    // and we fall back to the local heuristic below.
    const config = globalAIEngine.getAIConfig(choice.playerNumber);
    const mode = config?.mode ?? 'service';

    if (mode === 'service') {
      try {
        const tokenSource =
          this.sessionToken != null ? createLinkedCancellationSource(this.sessionToken) : null;

        if (tokenSource) {
          tokenSource.syncFromParent();
        }

        const selected = await globalAIEngine.getCaptureDirectionChoice(
          choice.playerNumber,
          null,
          choice.options,
          tokenSource ? { token: tokenSource.token } : undefined
        );

        if (choice.options.includes(selected)) {
          this.recordChoiceMetrics(choice.type, outcome, startedAt);
          return selected;
        }

        logger.warn(
          'AI service returned invalid option for capture_direction; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            optionsCount: choice.options.length,
            invalidOption: selected,
          }
        );
      } catch (error) {
        const aiErrorType = (error as any)?.aiErrorType;
        outcome = aiErrorType === 'timeout' ? 'timeout' : 'fallback';
        logger.warn(
          'AI service unavailable for capture_direction; falling back to local heuristic',
          {
            gameId: choice.gameId,
            playerNumber: choice.playerNumber,
            choiceId: choice.id,
            choiceType: choice.type,
            error: error instanceof Error ? error.message : String(error),
            aiErrorType,
          }
        );
      }
    }

    // If only one option, no need to compute distances.
    if (choice.options.length === 1) {
      const selected = choice.options[0];
      this.recordChoiceMetrics(choice.type, outcome, startedAt);
      return selected;
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

    this.recordChoiceMetrics(choice.type, outcome, startedAt);
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
      y: reference.y,
    };

    if (reference.z !== undefined) {
      centre.z = reference.z;
    }

    return centre;
  }

  private manhattanDistance(
    a: { x: number; y: number; z?: number },
    b: { x: number; y: number; z?: number }
  ): number {
    const dzA = a.z ?? 0;
    const dzB = b.z ?? 0;
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(dzA - dzB);
  }

  private recordChoiceMetrics(
    choiceType: PlayerChoice['type'],
    outcome: AIChoiceOutcome,
    startedAtMs: number
  ): void {
    const durationMs = Math.max(0, Date.now() - startedAtMs);
    const metrics = getMetricsService();

    // Optional chaining so legacy/mocked metrics services do not break tests.
    metrics?.recordAIChoiceRequest?.(choiceType, outcome);
    metrics?.recordAIChoiceLatencyMs?.(choiceType, durationMs, outcome);
  }
}
