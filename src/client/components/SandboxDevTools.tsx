import React from 'react';
import { EvaluationPanel } from './EvaluationPanel';
import { GameState } from '../../shared/types/game';
import type { GameEndExplanation } from '../../shared/engine/gameEndExplanation';
import type { ClientSandboxEngine } from '../sandbox/ClientSandboxEngine';
import { getSandboxAiDiagnostics } from '../sandbox/sandboxAiDiagnostics';
import { DEFAULT_AI_DIFFICULTY } from '../contexts/SandboxContext';
import type { PositionEvaluationPayload } from '../../shared/types/websocket';

/**
 * Props for the SandboxDevTools component.
 *
 * This component encapsulates all developer/debug-mode UI panels:
 * - AI Evaluation panel
 * - AI Ladder Diagnostics panel (per-player AI metadata)
 * - AI Service Ladder Health panel (tier health from /internal/ladder/health)
 */
export interface SandboxDevToolsProps {
  // AI tracking state from useSandboxAITracking
  aiLadderHealth: Record<string, unknown> | null;
  aiLadderHealthError: string | null;
  aiLadderHealthLoading: boolean;
  onRefreshLadderHealth: () => void;
  onCopyLadderHealth: () => void;

  // Diagnostics handlers from useSandboxDiagnostics
  onCopyAiTrace: () => void;
  onCopyAiMeta: () => void;
  onExportScenario: () => void;
  onCopyTestFixture: () => void;

  // AI Evaluation state and handlers
  evaluationHistory: PositionEvaluationPayload['data'][];
  evaluationError: string | null;
  isEvaluating: boolean;
  onRequestEvaluation: () => void;

  // Game state for debugging
  gameState: GameState | null;
  sandboxEngine: ClientSandboxEngine | null;

  // Config for AI difficulties
  aiDifficulties: number[];

  // Optional game end explanation for debug panel
  gameEndExplanation?: GameEndExplanation | null;
}

/**
 * Helper to safely cast unknown to Record<string, unknown>
 */
function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

/**
 * SandboxDevTools component - encapsulates all developer/debug UI panels.
 *
 * This component is only rendered when:
 * - `!isBeginnerMode` (debug mode is active)
 * - `developerToolsEnabled` is true
 * - There is an active game state
 *
 * The component renders:
 * 1. AI Evaluation panel with request button
 * 2. AI Ladder Diagnostics panel showing per-player AI metadata
 * 3. AI Service Ladder Health panel showing tier health from the AI service
 */
export function SandboxDevTools({
  aiLadderHealth,
  aiLadderHealthError,
  aiLadderHealthLoading,
  onRefreshLadderHealth,
  onCopyLadderHealth,
  onCopyAiMeta,
  evaluationHistory,
  evaluationError,
  isEvaluating,
  onRequestEvaluation,
  gameState,
  sandboxEngine,
  aiDifficulties,
}: SandboxDevToolsProps) {
  return (
    <>
      {/* AI Evaluation Panel */}
      <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <h2 className="font-semibold">AI Evaluation (sandbox)</h2>
          <button
            type="button"
            onClick={onRequestEvaluation}
            disabled={!sandboxEngine || !gameState || isEvaluating}
            className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 disabled:opacity-60 disabled:cursor-not-allowed transition"
          >
            {isEvaluating ? 'Evaluating…' : 'Request evaluation'}
          </button>
        </div>
        <EvaluationPanel evaluationHistory={evaluationHistory} players={gameState?.players ?? []} />
        {evaluationError && (
          <p className="text-xs text-amber-400" data-testid="sandbox-evaluation-error">
            {evaluationError}
          </p>
        )}
      </div>

      {/* AI Ladder Diagnostics Panel */}
      {gameState && (
        <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="font-semibold">AI Ladder Diagnostics (sandbox)</h2>
            <button
              type="button"
              onClick={onCopyAiMeta}
              className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
            >
              Copy
            </button>
          </div>

          {(() => {
            const byPlayer = getSandboxAiDiagnostics();
            const aiPlayers = gameState.players.filter((p) => p.type === 'ai');

            if (aiPlayers.length === 0) {
              return <p className="text-xs text-slate-400">No AI players in this sandbox.</p>;
            }

            return (
              <div className="space-y-2">
                {aiPlayers.map((player) => {
                  const configuredDifficulty =
                    aiDifficulties[player.playerNumber - 1] ?? DEFAULT_AI_DIFFICULTY;
                  const meta = byPlayer[player.playerNumber];
                  const timeLabel =
                    meta && typeof meta.timestamp === 'number'
                      ? new Date(meta.timestamp).toLocaleTimeString()
                      : '';

                  return (
                    <div
                      key={player.playerNumber}
                      className="rounded-xl border border-slate-700 bg-slate-950/40 p-3"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold">
                          P{player.playerNumber} · D{configuredDifficulty}
                        </span>
                        <span className="text-[10px] text-slate-500">{timeLabel}</span>
                      </div>

                      {!meta ? (
                        <p className="mt-2 text-[10px] text-slate-400">
                          No sandbox AI metadata recorded yet (trigger an AI turn).
                        </p>
                      ) : (
                        <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                          <span className="text-slate-400">Source</span>
                          <span className="text-slate-200">{meta.source}</span>
                          <span className="text-slate-400">AI Type</span>
                          <span className="text-slate-200">{meta.aiType ?? '—'}</span>
                          <span className="text-slate-400">useNeuralNet</span>
                          <span className="text-slate-200">
                            {meta.useNeuralNet === null || meta.useNeuralNet === undefined
                              ? '—'
                              : meta.useNeuralNet
                                ? 'true'
                                : 'false'}
                          </span>
                          <span className="text-slate-400">Heuristic Profile</span>
                          <span className="text-slate-200">{meta.heuristicProfileId ?? '—'}</span>
                          <span className="text-slate-400">NN Model</span>
                          <span className="text-slate-200">{meta.nnModelId ?? '—'}</span>
                          <span className="text-slate-400">NN Checkpoint</span>
                          <span className="text-slate-200">{meta.nnCheckpoint ?? '—'}</span>
                          <span className="text-slate-400">NNUE Checkpoint</span>
                          <span className="text-slate-200">{meta.nnueCheckpoint ?? '—'}</span>
                          {meta.error && (
                            <>
                              <span className="text-slate-400">Error</span>
                              <span className="text-amber-300">{meta.error}</span>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })()}
        </div>
      )}

      {/* AI Service Ladder Health Panel */}
      {gameState && (
        <div className="p-4 border border-slate-700 rounded-2xl bg-slate-900/60 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <h2 className="font-semibold">AI Service Ladder Health</h2>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onRefreshLadderHealth}
                disabled={aiLadderHealthLoading}
                className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 disabled:opacity-60 disabled:cursor-not-allowed transition"
              >
                {aiLadderHealthLoading ? 'Loading…' : 'Refresh'}
              </button>
              <button
                type="button"
                onClick={onCopyLadderHealth}
                disabled={!aiLadderHealth}
                className="px-3 py-1 rounded-full border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 disabled:opacity-60 disabled:cursor-not-allowed transition"
              >
                Copy
              </button>
            </div>
          </div>

          {aiLadderHealthError && (
            <p className="text-xs text-amber-400" data-testid="sandbox-ai-ladder-health-error">
              {aiLadderHealthError}
            </p>
          )}

          {!aiLadderHealth ? (
            <p className="text-xs text-slate-400">
              Click Refresh to query `/internal/ladder/health` from the AI service.
            </p>
          ) : (
            (() => {
              const summary = asRecord(aiLadderHealth['summary']) ?? {};
              const tiersRaw = aiLadderHealth['tiers'];
              const tiers: unknown[] = Array.isArray(tiersRaw) ? tiersRaw : [];

              return (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
                    <span className="text-slate-400">Missing heuristic profiles</span>
                    <span className="text-slate-200">
                      {String(summary['missing_heuristic_profiles'] ?? '—')}
                    </span>
                    <span className="text-slate-400">Missing NNUE checkpoints</span>
                    <span className="text-slate-200">
                      {String(summary['missing_nnue_checkpoints'] ?? '—')}
                    </span>
                    <span className="text-slate-400">Missing NN checkpoints</span>
                    <span className="text-slate-200">
                      {String(summary['missing_neural_checkpoints'] ?? '—')}
                    </span>
                    <span className="text-slate-400">Overridden tiers</span>
                    <span className="text-slate-200">
                      {String(summary['overridden_tiers'] ?? '—')}
                    </span>
                  </div>

                  <div className="overflow-auto border border-slate-800 rounded-xl bg-slate-950/40">
                    <table className="w-full text-[10px]">
                      <thead className="text-slate-400">
                        <tr className="border-b border-slate-800">
                          <th className="text-left font-semibold px-2 py-1">D</th>
                          <th className="text-left font-semibold px-2 py-1">AI</th>
                          <th className="text-left font-semibold px-2 py-1">NN</th>
                          <th className="text-left font-semibold px-2 py-1">Heuristic</th>
                          <th className="text-left font-semibold px-2 py-1">Model</th>
                          <th className="text-left font-semibold px-2 py-1">Artifacts</th>
                        </tr>
                      </thead>
                      <tbody className="text-slate-200">
                        {tiers.map((tierRaw, index) => {
                          const tier = asRecord(tierRaw) ?? {};

                          const difficulty =
                            typeof tier['difficulty'] === 'number' ? tier['difficulty'] : undefined;
                          const aiType =
                            typeof tier['ai_type'] === 'string' ? tier['ai_type'] : undefined;
                          const useNeuralNet =
                            typeof tier['use_neural_net'] === 'boolean'
                              ? tier['use_neural_net']
                              : undefined;
                          const heuristicProfile =
                            typeof tier['heuristic_profile_id'] === 'string'
                              ? tier['heuristic_profile_id']
                              : undefined;
                          const modelId =
                            typeof tier['model_id'] === 'string' ? tier['model_id'] : undefined;

                          const artifacts = asRecord(tier['artifacts']) ?? {};

                          const heuristicProfileArtifact =
                            asRecord(artifacts['heuristic_profile']) ?? {};
                          const heuristicOk =
                            typeof heuristicProfileArtifact['available'] === 'boolean'
                              ? heuristicProfileArtifact['available']
                              : undefined;

                          const nnueArtifact = asRecord(artifacts['nnue']) ?? {};
                          const nnueFileArtifact = asRecord(nnueArtifact['file']) ?? {};
                          const nnueOk =
                            typeof nnueFileArtifact['exists'] === 'boolean'
                              ? nnueFileArtifact['exists']
                              : undefined;

                          const neuralNetArtifact = asRecord(artifacts['neural_net']) ?? {};
                          const neuralNetChosen = asRecord(neuralNetArtifact['chosen']) ?? {};
                          const nnOk =
                            typeof neuralNetChosen['exists'] === 'boolean'
                              ? neuralNetChosen['exists']
                              : undefined;

                          const artifactLabelParts: string[] = [];
                          if (typeof heuristicOk === 'boolean') {
                            artifactLabelParts.push(`H:${heuristicOk ? 'ok' : 'missing'}`);
                          }
                          if (typeof nnueOk === 'boolean') {
                            artifactLabelParts.push(`NNUE:${nnueOk ? 'ok' : 'missing'}`);
                          }
                          if (typeof nnOk === 'boolean') {
                            artifactLabelParts.push(`NN:${nnOk ? 'ok' : 'missing'}`);
                          }

                          return (
                            <tr
                              key={`${String(difficulty ?? 'unknown')}_${index}`}
                              className="border-b border-slate-900/60"
                            >
                              <td className="px-2 py-1">{difficulty}</td>
                              <td className="px-2 py-1">{aiType ?? '—'}</td>
                              <td className="px-2 py-1">
                                {useNeuralNet === true
                                  ? 'yes'
                                  : useNeuralNet === false
                                    ? 'no'
                                    : '—'}
                              </td>
                              <td className="px-2 py-1">{heuristicProfile ?? '—'}</td>
                              <td className="px-2 py-1">{modelId ?? '—'}</td>
                              <td className="px-2 py-1">
                                {artifactLabelParts.length > 0 ? artifactLabelParts.join(' ') : '—'}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })()
          )}
        </div>
      )}
    </>
  );
}

/**
 * Debug buttons that appear in the board header when in debug mode.
 * These include: Export Scenario JSON, Copy Test Fixture, and GameEndExplanation debug.
 */
export interface SandboxDevToolsHeaderButtonsProps {
  onExportScenario: () => void;
  onCopyTestFixture: () => void;
  gameEndExplanation?: GameEndExplanation | null;
}

export function SandboxDevToolsHeaderButtons({
  onExportScenario,
  onCopyTestFixture,
  gameEndExplanation,
}: SandboxDevToolsHeaderButtonsProps) {
  return (
    <>
      <button
        type="button"
        onClick={onExportScenario}
        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
      >
        Export Scenario JSON
      </button>
      <button
        type="button"
        onClick={onCopyTestFixture}
        className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-emerald-400 hover:text-emerald-200 transition"
      >
        Copy Test Fixture
      </button>
      {gameEndExplanation && (
        <details
          className="absolute top-full right-0 mt-2 w-96 p-4 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 text-xs font-mono overflow-auto max-h-96"
          data-testid="sandbox-game-end-explanation-debug"
        >
          <summary className="cursor-pointer text-slate-400 hover:text-slate-200 mb-2">
            Debug: GameEndExplanation
          </summary>
          <pre className="whitespace-pre-wrap text-emerald-300">
            {JSON.stringify(gameEndExplanation, null, 2)}
          </pre>
        </details>
      )}
    </>
  );
}

/**
 * Stall warning action button for copying AI trace.
 * This appears in the stall warning banner when developer tools are enabled.
 */
export interface SandboxDevToolsStallWarningActionProps {
  onCopyAiTrace: () => void;
}

export function SandboxDevToolsStallWarningAction({
  onCopyAiTrace,
}: SandboxDevToolsStallWarningActionProps) {
  return (
    <button
      type="button"
      onClick={onCopyAiTrace}
      className="px-3 py-1 rounded-lg border border-slate-600 text-xs font-semibold text-slate-100 hover:border-sky-400 hover:text-sky-200 transition"
    >
      Copy AI trace
    </button>
  );
}
