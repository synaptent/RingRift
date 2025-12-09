import fs from 'fs';
import os from 'os';
import path from 'path';

import { main } from '../../scripts/analyze_rules_ux_telemetry';

describe('RulesUxHotspotAnalysis – happy path', () => {
  it('produces hotspot JSON and Markdown with expected metrics and rankings', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rules-ux-hotspots-happy-'));
    const inputPath = path.join(tmpDir, 'rules_ux_aggregates.json');
    const jsonOutPath = path.join(tmpDir, 'rules_ux_hotspots.square8_2p.json');
    const mdOutPath = path.join(tmpDir, 'rules_ux_hotspots.square8_2p.md');

    const aggregate = {
      board: 'square8',
      num_players: 2,
      window: {
        start: '2025-11-01T00:00:00Z',
        end: '2025-11-29T23:59:59Z',
      },
      games: {
        started: 12345,
        completed: 11000,
      },
      contexts: [
        {
          rulesContext: 'anm_forced_elimination',
          sources: [
            {
              source: 'hud',
              events: {
                help_open: 200,
                help_reopen: 70,
                weird_state_banner_impression: 120,
                weird_state_details_open: 90,
                resign_after_weird_state: 25,
              },
            },
            {
              source: 'victory_modal',
              events: {
                help_open: 50,
                help_reopen: 10,
                weird_state_banner_impression: 40,
                weird_state_details_open: 30,
                resign_after_weird_state: 5,
              },
            },
          ],
        },
        {
          rulesContext: 'structural_stalemate',
          sources: [
            {
              source: 'hud',
              events: {
                help_open: 150,
                help_reopen: 30,
                weird_state_banner_impression: 60,
                weird_state_details_open: 40,
                resign_after_weird_state: 6,
              },
            },
          ],
        },
        {
          rulesContext: 'territory_mini_region',
          sources: [
            {
              source: 'hud',
              events: {
                help_open: 100,
                help_reopen: 20,
                weird_state_banner_impression: 50,
                weird_state_details_open: 30,
                resign_after_weird_state: 5,
              },
            },
          ],
        },
      ],
    };

    fs.writeFileSync(inputPath, JSON.stringify(aggregate), 'utf8');

    const exitCode = await main([
      'node',
      'analyze_rules_ux_telemetry.ts',
      '--input',
      inputPath,
      '--output-json',
      jsonOutPath,
      '--output-md',
      mdOutPath,
      '--min-events',
      '20',
      '--top-k',
      '3',
    ]);

    expect(exitCode).toBe(0);
    expect(fs.existsSync(jsonOutPath)).toBe(true);
    expect(fs.existsSync(mdOutPath)).toBe(true);

    const jsonRaw = fs.readFileSync(jsonOutPath, 'utf8');
    const summary = JSON.parse(jsonRaw) as any;

    // Basic shape
    expect(summary.board).toBe('square8');
    expect(summary.num_players).toBe(2);
    expect(summary.games.completed).toBe(11000);
    expect(Array.isArray(summary.contexts)).toBe(true);
    expect(summary.contexts).toHaveLength(3);

    const anmCtx = summary.contexts.find((c: any) => c.rulesContext === 'anm_forced_elimination');
    const stalemateCtx = summary.contexts.find(
      (c: any) => c.rulesContext === 'structural_stalemate'
    );
    const territoryCtx = summary.contexts.find(
      (c: any) => c.rulesContext === 'territory_mini_region'
    );

    expect(anmCtx).toBeDefined();
    expect(stalemateCtx).toBeDefined();
    expect(territoryCtx).toBeDefined();

    // At least one HIGH severity context (the ANM forced elimination hotspot).
    expect(anmCtx.hotspotSeverity).toBe('HIGH');

    // Ordering by help opens per 100 games.
    expect(summary.topByHelpOpensPer100Games).toEqual([
      'anm_forced_elimination',
      'structural_stalemate',
      'territory_mini_region',
    ]);

    // Per-source metrics for the HUD source under anm_forced_elimination.
    const anmHud = (anmCtx.sources as any[]).find((s: any) => s.source === 'hud');
    expect(anmHud).toBeDefined();
    expect(anmHud.helpOpens).toBe(200);
    expect(anmHud.helpReopens).toBe(70);
    expect(anmHud.weirdImpressions).toBe(120);
    expect(anmHud.weirdDetailsOpens).toBe(90);
    expect(anmHud.resignsAfterWeird).toBe(25);
    expect(anmHud.sampleOk).toBe(true);

    // Numerical rates: allow small floating‑point differences.
    expect(anmCtx.sumHelpOpens).toBe(250);
    expect(anmCtx.helpOpensPer100Games).toBeCloseTo((250 / 11000) * 100, 2);
    expect(anmCtx.maxHelpReopenRate).toBeCloseTo(70 / 200, 3);
    expect(anmCtx.maxResignAfterWeirdRate).toBeCloseTo(25 / 120, 3);

    expect(anmHud.helpOpensPer100Games).toBeCloseTo((200 / 11000) * 100, 2);
    expect(anmHud.helpReopenRate).toBeCloseTo(70 / 200, 3);
    expect(anmHud.resignAfterWeirdRate).toBeCloseTo(25 / 120, 3);

    // Markdown content: header, context section, and key bullets.
    const md = fs.readFileSync(mdOutPath, 'utf8');
    expect(md).toContain('## Rules UX Hotspots – square8 2-player, window 2025-11');
    expect(md).toContain('### Context: anm_forced_elimination');
    expect(md).toContain('Help opens per 100 games: 2.27');
    expect(md).toContain('help_reopen: 70 (reopen rate: 0.35)');
  });

  it('smoke fixture: respects top-k ranking and min-events gating for dry-run usage', async () => {
    const fixturePath = path.resolve(
      __dirname,
      '../fixtures/rules_ux_hotspots/rules_ux_aggregates.square8_2p.sample.json'
    );
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rules-ux-hotspots-smoke-'));
    const jsonOutPath = path.join(tmpDir, 'summary.json');
    const mdOutPath = path.join(tmpDir, 'summary.md');

    const exitCode = await main([
      'node',
      'analyze_rules_ux_telemetry.ts',
      '--input',
      fixturePath,
      '--output-json',
      jsonOutPath,
      '--output-md',
      mdOutPath,
      '--min-events',
      '15',
      '--top-k',
      '2',
    ]);

    expect(exitCode).toBe(0);
    expect(fs.existsSync(jsonOutPath)).toBe(true);
    expect(fs.existsSync(mdOutPath)).toBe(true);

    const summary = JSON.parse(fs.readFileSync(jsonOutPath, 'utf8')) as any;

    // Basic shape derived from the fixture metadata.
    expect(summary.board).toBe('square8');
    expect(summary.num_players).toBe(2);
    expect(summary.games.started).toBe(600);
    expect(summary.games.completed).toBe(500);
    expect(summary.window.label).toBe('2025-12');
    expect(summary.contexts).toHaveLength(4);

    // Ranking should surface the two highest-help contexts.
    expect(summary.topByHelpOpensPer100Games).toEqual([
      'anm_forced_elimination',
      'structural_stalemate',
    ]);

    const territoryCtx = summary.contexts.find(
      (c: any) => c.rulesContext === 'territory_mini_region'
    );
    expect(territoryCtx).toBeDefined();
    expect(territoryCtx.sources[0].sampleOk).toBe(false); // below minEvents threshold
    expect(territoryCtx.hotspotSeverity).toBe('LOW');

    const anmCtx = summary.contexts.find((c: any) => c.rulesContext === 'anm_forced_elimination');
    expect(anmCtx.sources.every((s: any) => s.sampleOk)).toBe(true);
    expect(anmCtx.hotspotSeverity).toBe('HIGH');
    expect(anmCtx.sumHelpOpens).toBe(70);
    expect(anmCtx.helpOpensPer100Games).toBeCloseTo((70 / 500) * 100, 4);
    expect(anmCtx.maxHelpReopenRate).toBeCloseTo(20 / 60, 4);
    expect(anmCtx.maxResignAfterWeirdRate).toBeCloseTo(24 / 120, 4);

    const structuralCtx = summary.contexts.find(
      (c: any) => c.rulesContext === 'structural_stalemate'
    );
    expect(structuralCtx).toBeDefined();
    expect(structuralCtx.hotspotSeverity).toBe('MEDIUM');
    expect(structuralCtx.helpOpensPer100Games).toBeCloseTo((35 / 500) * 100, 4);

    const lineProcessing = summary.contexts.find((c: any) => c.rulesContext === 'line_processing');
    expect(lineProcessing).toBeDefined();
    expect(lineProcessing!.hotspotSeverity).toBe('MEDIUM');
    expect(lineProcessing!.sources[0].sampleOk).toBe(true);
    expect(lineProcessing!.helpOpensPer100Games).toBeCloseTo((8 / 500) * 100, 4);

    const md = fs.readFileSync(mdOutPath, 'utf8');
    expect(md).toContain('Rules UX Hotspots – square8 2-player');
    expect(md).toContain('Context: anm_forced_elimination');
    expect(md).toContain('Context: structural_stalemate');
  });

  it('skips Markdown output when --output-md is an empty string', async () => {
    const fixturePath = path.resolve(
      __dirname,
      '../fixtures/rules_ux_hotspots/rules_ux_aggregates.square8_2p.sample.json'
    );
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rules-ux-hotspots-nomd-'));
    const jsonOutPath = path.join(tmpDir, 'summary.json');
    const mdOutPath = path.join(tmpDir, 'summary.md');

    const exitCode = await main([
      'node',
      'analyze_rules_ux_telemetry.ts',
      '--input',
      fixturePath,
      '--output-json',
      jsonOutPath,
      '--output-md',
      '',
      '--min-events',
      '10',
      '--top-k',
      '3',
    ]);

    expect(exitCode).toBe(0);
    expect(fs.existsSync(jsonOutPath)).toBe(true);
    expect(fs.existsSync(mdOutPath)).toBe(false);

    const summary = JSON.parse(fs.readFileSync(jsonOutPath, 'utf8')) as any;
    expect(summary.topByHelpOpensPer100Games.length).toBeGreaterThan(0);
  });
});

describe('RulesUxHotspotAnalysis – validation failures', () => {
  it('returns non-zero exit code and writes no outputs for unsupported board', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rules-ux-hotspots-invalid-board-'));
    const inputPath = path.join(tmpDir, 'rules_ux_aggregates_hex.json');
    const jsonOutPath = path.join(tmpDir, 'hotspots.json');
    const mdOutPath = path.join(tmpDir, 'hotspots.md');

    const aggregate = {
      board: 'hex',
      num_players: 2,
      window: {
        start: '2025-11-01T00:00:00Z',
        end: '2025-11-29T23:59:59Z',
      },
      games: {
        started: 100,
        completed: 80,
      },
      contexts: [
        {
          rulesContext: 'anm_forced_elimination',
          sources: [
            {
              source: 'hud',
              events: {
                help_open: 10,
              },
            },
          ],
        },
      ],
    };

    fs.writeFileSync(inputPath, JSON.stringify(aggregate), 'utf8');

    const exitCode = await main([
      'node',
      'analyze_rules_ux_telemetry.ts',
      '--input',
      inputPath,
      '--output-json',
      jsonOutPath,
      '--output-md',
      mdOutPath,
    ]);

    expect(exitCode).not.toBe(0);
    expect(fs.existsSync(jsonOutPath)).toBe(false);
    expect(fs.existsSync(mdOutPath)).toBe(false);
  });

  it('returns non-zero exit code and writes no outputs when contexts array is missing', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rules-ux-hotspots-missing-contexts-'));
    const inputPath = path.join(tmpDir, 'rules_ux_aggregates_missing_contexts.json');
    const jsonOutPath = path.join(tmpDir, 'hotspots.json');
    const mdOutPath = path.join(tmpDir, 'hotspots.md');

    const aggregate: any = {
      board: 'square8',
      num_players: 2,
      window: {
        start: '2025-11-01T00:00:00Z',
        end: '2025-11-29T23:59:59Z',
      },
      games: {
        started: 100,
        completed: 80,
      },
      // contexts intentionally omitted
    };

    fs.writeFileSync(inputPath, JSON.stringify(aggregate), 'utf8');

    const exitCode = await main([
      'node',
      'analyze_rules_ux_telemetry.ts',
      '--input',
      inputPath,
      '--output-json',
      jsonOutPath,
      '--output-md',
      mdOutPath,
    ]);

    expect(exitCode).not.toBe(0);
    expect(fs.existsSync(jsonOutPath)).toBe(false);
    expect(fs.existsSync(mdOutPath)).toBe(false);
  });
});
