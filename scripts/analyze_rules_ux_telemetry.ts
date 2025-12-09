import * as fs from 'fs';
import * as path from 'path';
import type {
  RulesUxAggregatesRoot,
  RulesUxContextAggregate,
  RulesUxSourceAggregate,
} from '../src/shared/telemetry/rulesUxHotspotTypes';

/**
 * Rules‑UX hotspot analyzer for pre‑aggregated `ringrift_rules_ux_events_total` snapshots.
 *
 * This script:
 *   1. Loads a JSON file that matches the RulesUxAggregatesRoot shape.
 *   2. Computes per‑context and per‑source metrics (help opens per 100 games,
 *      reopen rates, resign‑after‑weird rates, etc.).
 *   3. Writes a machine‑readable JSON hotspot summary and a concise Markdown
 *      report suitable for pasting into UX iteration notes.
 *
 * Expected event keys inside the aggregated RulesUxSourceAggregate.events map
 * include (when present):
 *
 *   - "help_open"
 *   - "help_reopen"
 *   - "rules_help_repeat"              (legacy alias for help reopen)
 *   - "weird_state_banner_impression"
 *   - "weird_state_details_open"
 *   - "resign_after_weird_state"
 *   - "rules_weird_state_resign"       (legacy alias for resign‑after‑weird)
 *
 * The analyzer is resilient to additional keys and will ignore them.
 *
 * See `tests/fixtures/rules_ux_hotspots/rules_ux_aggregates.square8_2p.sample.json`
 * for a dry-run fixture that exercises expected inputs/outputs in tests.
 */

export type HotspotSeverity = 'HIGH' | 'MEDIUM' | 'LOW';

export interface CliOptions {
  inputPath: string;
  outputJsonPath?: string;
  outputMdPath?: string | null;
  minEvents: number;
  topK: number;
}

export interface SourceHotspotMetrics {
  source: string;
  helpOpens: number;
  helpReopens: number;
  weirdImpressions: number;
  weirdDetailsOpens: number;
  resignsAfterWeird: number;
  helpOpensPer100Games: number;
  helpReopenRate: number;
  resignAfterWeirdRate: number;
  sampleOk: boolean;
}

export interface ContextHotspotSummary {
  rulesContext: string;
  sumHelpOpens: number;
  helpOpensPer100Games: number;
  maxHelpReopenRate: number;
  maxResignAfterWeirdRate: number;
  hotspotSeverity: HotspotSeverity;
  sources: SourceHotspotMetrics[];
}

export interface HotspotSummaryJson {
  board: string;
  num_players: number;
  window: {
    start: string;
    end: string;
    label: string;
  };
  games: {
    started: number;
    completed: number;
  };
  contexts: ContextHotspotSummary[];
  topByHelpOpensPer100Games: string[];
}

/**
 * Minimal CLI parser for this script.
 *
 * Supported flags:
 *   --input PATH          (required)
 *   --output-json PATH    (optional; default derived from board/num_players)
 *   --output-md PATH      (optional; default derived from board/num_players;
 *                          if set to empty string, Markdown output is skipped)
 *   --min-events N        (optional; default 20)
 *   --top-k N             (optional; default 5)
 */
export function parseArgs(argv: string[]): CliOptions {
  const args = argv.slice(2);
  const partial: Partial<CliOptions> = {
    minEvents: 20,
    topK: 5,
  };

  for (let i = 0; i < args.length; i += 1) {
    const raw = args[i];

    if (!raw.startsWith('--')) {
      // Positional args are ignored for now.

      continue;
    }

    const [flag, inlineValue] = raw.split('=', 2);
    let value = inlineValue;

    if (value === undefined) {
      if (i + 1 < args.length && !args[i + 1].startsWith('--')) {
        value = args[i + 1];
        i += 1;
      }
    }

    switch (flag) {
      case '--input': {
        if (!value) {
          throw new Error('Flag --input requires a PATH value');
        }
        partial.inputPath = value;
        break;
      }
      case '--output-json': {
        // Empty string is treated as "use default name".
        if (value && value.length > 0) {
          partial.outputJsonPath = value;
        }
        break;
      }
      case '--output-md': {
        // Empty string is a sentinel for "skip Markdown output".
        partial.outputMdPath = value !== undefined ? value : '';
        break;
      }
      case '--min-events': {
        if (!value) {
          throw new Error('Flag --min-events requires a numeric value');
        }
        const parsed = Number.parseInt(value, 10);
        if (!Number.isFinite(parsed)) {
          throw new Error('Flag --min-events must be an integer');
        }
        partial.minEvents = parsed;
        break;
      }
      case '--top-k': {
        if (!value) {
          throw new Error('Flag --top-k requires a numeric value');
        }
        const parsed = Number.parseInt(value, 10);
        if (!Number.isFinite(parsed) || parsed <= 0) {
          throw new Error('Flag --top-k must be a positive integer');
        }
        partial.topK = parsed;
        break;
      }
      default:
        // Ignore unknown flags for forwards compatibility.
        break;
    }
  }

  if (!partial.inputPath) {
    throw new Error('Missing required --input PATH argument');
  }

  const inputPath = partial.inputPath;

  const options: CliOptions = {
    inputPath,
    minEvents: partial.minEvents ?? 20,
    topK: partial.topK ?? 5,
  };

  if (partial.outputJsonPath !== undefined) {
    options.outputJsonPath = partial.outputJsonPath;
  }
  if (partial.outputMdPath !== undefined) {
    options.outputMdPath = partial.outputMdPath;
  }

  return options;
}

/**
 * Validate a raw JSON object as a RulesUxAggregatesRoot.
 * Ensures the analyzer only operates on square8 2‑player snapshots
 * with at least one context entry.
 */
export function validateAggregatesRoot(raw: unknown): RulesUxAggregatesRoot {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Input JSON is not an object');
  }

  const root = raw as RulesUxAggregatesRoot;

  if (root.board !== 'square8') {
    throw new Error(`Unsupported board "${(root as any).board}". Expected "square8".`);
  }

  if (root.num_players !== 2) {
    throw new Error(
      `Unsupported num_players "${(root as any).num_players}". Expected 2 for this analyzer.`
    );
  }

  if (!root.games || typeof root.games !== 'object') {
    throw new Error('Missing or invalid "games" block in input JSON');
  }

  if (typeof root.games.completed !== 'number' || root.games.completed < 0) {
    throw new Error('"games.completed" must be a non-negative number');
  }

  if (!root.window || typeof root.window !== 'object') {
    throw new Error('Missing "window" block in input JSON');
  }

  if (typeof root.window.start !== 'string' || typeof root.window.end !== 'string') {
    throw new Error('"window.start" and "window.end" must be strings');
  }

  if (!Array.isArray(root.contexts) || root.contexts.length === 0) {
    throw new Error('Input JSON must contain a non-empty "contexts" array');
  }

  return root;
}

export function computeWindowLabel(start: string, end: string): string {
  const fromStart = /^(\d{4}-\d{2})/.exec(start);
  if (fromStart) {
    return fromStart[1];
  }

  const fromEnd = /^(\d{4}-\d{2})/.exec(end);
  if (fromEnd) {
    return fromEnd[1];
  }

  return 'unknown';
}

export function safeNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function sumEventCounts(events: RulesUxSourceAggregate['events']): number {
  return Object.values(events).reduce((acc, v) => acc + safeNumber(v), 0);
}

export function computeSourceMetrics(
  sourceAgg: RulesUxSourceAggregate,
  gamesCompleted: number,
  minEvents: number
): SourceHotspotMetrics {
  const events = sourceAgg.events ?? {};
  const totalEvents = sumEventCounts(events);

  const helpOpens = safeNumber((events as any).help_open);
  const helpReopens = safeNumber(
    (events as any).help_reopen !== undefined
      ? (events as any).help_reopen
      : (events as any).rules_help_repeat
  );
  const weirdImpressions = safeNumber((events as any).weird_state_banner_impression);
  const weirdDetailsOpens = safeNumber((events as any).weird_state_details_open);
  const resignsAfterWeird = safeNumber(
    (events as any).resign_after_weird_state !== undefined
      ? (events as any).resign_after_weird_state
      : (events as any).rules_weird_state_resign
  );

  const gamesDenominator = gamesCompleted > 0 ? gamesCompleted : 1;
  const helpOpensPer100Games = (helpOpens / gamesDenominator) * 100;
  const helpReopenRate = helpReopens / Math.max(helpOpens, 1);
  const resignAfterWeirdRate = resignsAfterWeird / Math.max(weirdImpressions, 1);

  const sampleOk = totalEvents >= minEvents && gamesCompleted >= 1;

  return {
    source: String(sourceAgg.source),
    helpOpens,
    helpReopens,
    weirdImpressions,
    weirdDetailsOpens,
    resignsAfterWeird,
    helpOpensPer100Games,
    helpReopenRate,
    resignAfterWeirdRate,
    sampleOk,
  };
}

export function computeContextSummary(
  contextAgg: RulesUxContextAggregate,
  gamesCompleted: number,
  minEvents: number
): ContextHotspotSummary {
  const sources = (contextAgg.sources ?? []).map((s) =>
    computeSourceMetrics(s, gamesCompleted, minEvents)
  );

  const sumHelpOpens = sources.reduce((acc, s) => acc + s.helpOpens, 0);
  const gamesDenominator = gamesCompleted > 0 ? gamesCompleted : 1;
  const helpOpensPer100Games = (sumHelpOpens / gamesDenominator) * 100;
  const maxHelpReopenRate = sources.reduce(
    (max, s) => (s.helpReopenRate > max ? s.helpReopenRate : max),
    0
  );
  const maxResignAfterWeirdRate = sources.reduce(
    (max, s) => (s.resignAfterWeirdRate > max ? s.resignAfterWeirdRate : max),
    0
  );

  return {
    rulesContext: String(contextAgg.rulesContext),
    sumHelpOpens,
    helpOpensPer100Games,
    maxHelpReopenRate,
    maxResignAfterWeirdRate,
    hotspotSeverity: 'LOW',
    sources,
  };
}

export function classifyHotspotSeverity(
  ctx: ContextHotspotSummary,
  isInTopKByHelpOpens: boolean
): HotspotSeverity {
  const { helpOpensPer100Games, maxHelpReopenRate, maxResignAfterWeirdRate } = ctx;

  // HIGH severity: among the top-K by help/100 games AND shows strong churn indicators.
  if (
    isInTopKByHelpOpens &&
    helpOpensPer100Games >= 1 &&
    (maxHelpReopenRate >= 0.4 || maxResignAfterWeirdRate >= 0.2)
  ) {
    return 'HIGH';
  }

  // MEDIUM severity: elevated help opens (at least ~1 per 100 games) or moderate churn,
  // regardless of rank.
  if (helpOpensPer100Games >= 1 || maxHelpReopenRate >= 0.25 || maxResignAfterWeirdRate >= 0.1) {
    return 'MEDIUM';
  }

  return 'LOW';
}

export function formatRate(value: number, decimals = 2): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : '0.00';
}

export function buildMarkdownReport(summary: HotspotSummaryJson): string {
  const lines: string[] = [];

  lines.push(
    `## Rules UX Hotspots – ${summary.board} ${summary.num_players}-player, window ${summary.window.label}`
  );
  lines.push('');
  lines.push(`Completed games: ${summary.games.completed}`);
  lines.push('');

  if (summary.topByHelpOpensPer100Games.length > 0) {
    lines.push('### Top contexts by help opens per 100 games');
    lines.push('');
    summary.topByHelpOpensPer100Games.forEach((ctxName, index) => {
      const ctx = summary.contexts.find((c) => c.rulesContext === ctxName);
      if (!ctx) return;

      lines.push(
        `${index + 1}. **${ctx.rulesContext}** – ${formatRate(
          ctx.helpOpensPer100Games
        )} help opens / 100 games (severity: ${ctx.hotspotSeverity})  `
      );
    });
    lines.push('');
  }

  // Detailed sections for each context in the top list.
  for (const ctxName of summary.topByHelpOpensPer100Games) {
    const ctx = summary.contexts.find((c) => c.rulesContext === ctxName);
    if (!ctx) continue;

    lines.push(`### Context: ${ctx.rulesContext}`);
    lines.push('');
    lines.push(`- Severity: **${ctx.hotspotSeverity}**`);
    lines.push(`- Help opens per 100 games: ${formatRate(ctx.helpOpensPer100Games)}`);
    lines.push(`- Max help reopen rate: ${formatRate(ctx.maxHelpReopenRate)}`);
    lines.push(`- Max resign-after-weird rate: ${formatRate(ctx.maxResignAfterWeirdRate)}`);
    lines.push('');
    lines.push('By source:');
    lines.push('');

    for (const src of ctx.sources) {
      lines.push(`- **${src.source}**`);
      lines.push(
        `  - help_open: ${src.helpOpens} (${formatRate(src.helpOpensPer100Games)} / 100 games)`
      );
      lines.push(
        `  - help_reopen: ${src.helpReopens} (reopen rate: ${formatRate(src.helpReopenRate)})`
      );
      lines.push(`  - weird_state_banner_impression: ${src.weirdImpressions}`);
      lines.push(`  - weird_state_details_open: ${src.weirdDetailsOpens}`);
      lines.push(
        `  - resign_after_weird_state: ${src.resignsAfterWeird} (${formatRate(
          src.resignAfterWeirdRate
        )} of impressions)`
      );
      lines.push('');
    }
  }

  return `${lines.join('\n')}\n`;
}

/**
 * Main entry point for programmatic and CLI usage.
 *
 * Returns a numeric exit code (0 on success, non-zero on failure).
 */
export async function main(argv: string[] = process.argv): Promise<number> {
  let options: CliOptions;

  try {
    options = parseArgs(argv);
  } catch (err) {
    console.error('[rules-ux-hotspots] Failed to parse CLI arguments:', err);
    return 1;
  }

  let rawJson: string;
  try {
    rawJson = await fs.promises.readFile(options.inputPath, 'utf8');
  } catch (err) {
    console.error(
      `[rules-ux-hotspots] Failed to read input JSON from "${options.inputPath}":`,
      err
    );
    return 1;
  }

  let root: RulesUxAggregatesRoot;
  try {
    const parsed = JSON.parse(rawJson);
    root = validateAggregatesRoot(parsed);
  } catch (err) {
    console.error('[rules-ux-hotspots] Invalid input JSON:', err);
    return 1;
  }

  const gamesCompleted = root.games.completed;

  const contextSummaries: ContextHotspotSummary[] = root.contexts.map((ctx) =>
    computeContextSummary(ctx, gamesCompleted, options.minEvents)
  );

  // Rank contexts by help opens per 100 games.
  const sortedByHelpOpens = [...contextSummaries].sort(
    (a, b) => b.helpOpensPer100Games - a.helpOpensPer100Games
  );
  const topList = sortedByHelpOpens.slice(0, options.topK);
  const topSet = new Set(topList.map((c) => c.rulesContext));

  // Classify severity now that we know the ranking.
  for (const ctx of contextSummaries) {
    ctx.hotspotSeverity = classifyHotspotSeverity(ctx, topSet.has(ctx.rulesContext));
  }

  const windowLabel = computeWindowLabel(root.window.start, root.window.end);

  const summary: HotspotSummaryJson = {
    board: root.board,
    num_players: root.num_players,
    window: {
      start: root.window.start,
      end: root.window.end,
      label: windowLabel,
    },
    games: {
      started: root.games.started,
      completed: root.games.completed,
    },
    contexts: contextSummaries,
    topByHelpOpensPer100Games: topList.map((c) => c.rulesContext),
  };

  // Derive default output paths from the snapshot metadata if not provided.
  const safeBoard = root.board || 'unknown';
  const safePlayers = root.num_players || 0;
  const baseName = `rules_ux_hotspots.${safeBoard}_${safePlayers}p`;

  const outputJsonPath =
    options.outputJsonPath && options.outputJsonPath.length > 0
      ? options.outputJsonPath
      : path.resolve(`${baseName}.json`);

  let outputMdPath: string | null;
  if (options.outputMdPath === '') {
    outputMdPath = null;
  } else if (!options.outputMdPath) {
    outputMdPath = path.resolve(`${baseName}.md`);
  } else {
    outputMdPath = options.outputMdPath;
  }

  try {
    await fs.promises.writeFile(outputJsonPath, JSON.stringify(summary, null, 2), 'utf8');
  } catch (err) {
    console.error(`[rules-ux-hotspots] Failed to write JSON summary to "${outputJsonPath}":`, err);
    return 1;
  }

  if (outputMdPath) {
    const markdown = buildMarkdownReport(summary);
    try {
      await fs.promises.writeFile(outputMdPath, markdown, 'utf8');
    } catch (err) {
      console.error(
        `[rules-ux-hotspots] Failed to write Markdown report to "${outputMdPath}":`,
        err
      );
      return 1;
    }
  }

  return 0;
}

if (require.main === module) {
  main().then((code) => {
    if (code !== 0) {
      console.error(`[rules-ux-hotspots] Exiting with code ${code}`);
      process.exitCode = code;
    }
  });
}
