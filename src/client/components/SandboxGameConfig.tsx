import React, { useState } from 'react';
import { BoardType } from '../../shared/types/game';
import { LocalConfig, LocalPlayerType } from '../contexts/SandboxContext';
import { AIDifficultySelector, AIDifficultyBadge } from './AIDifficultySelector';
import { getDifficultyDescriptor } from '../utils/difficultyUx';
import { InlineAlert } from './ui/InlineAlert';

/**
 * Board preset configuration for display in the configuration UI
 */
export interface BoardPreset {
  value: BoardType;
  label: string;
  subtitle: string;
  blurb: string;
}

/**
 * Quick-start preset configuration
 */
export interface QuickStartPreset {
  id: string;
  label: string;
  description: string;
  learnMoreText?: string;
  icon: string;
  badge?: string;
  config: {
    boardType: BoardType;
    numPlayers: number;
    playerTypes: LocalPlayerType[];
  };
}

/**
 * Board type display labels for UI
 */
const BOARD_TYPE_LABELS: Record<BoardType, string> = {
  square8: 'sq8',
  hex8: 'hex8',
  square19: 'sq19',
  hexagonal: 'hex24',
};

/**
 * Board type row headers for grouping
 */
const BOARD_TYPE_ROW_HEADERS: Record<BoardType, string> = {
  square8: '8×8 Square',
  hex8: '8-Hex Compact',
  square19: '19×19 Square',
  hexagonal: '24-Hex Full',
};

/**
 * Sandbox time control configuration
 */
export interface SandboxTimeControl {
  initialTimeMs: number;
  incrementMs: number;
}

/**
 * Player type metadata for display
 */
const PLAYER_TYPE_META: Record<
  LocalPlayerType,
  { label: string; description: string; accent: string; chip: string }
> = {
  human: {
    label: 'Human',
    description: 'You control every move',
    accent: 'border-emerald-500 text-emerald-200',
    chip: 'bg-emerald-900/40 text-emerald-200',
  },
  ai: {
    label: 'Computer',
    description: 'Local heuristic AI',
    accent: 'border-sky-500 text-sky-200',
    chip: 'bg-sky-900/40 text-sky-200',
  },
};

/**
 * Default board presets
 */
export const BOARD_PRESETS: BoardPreset[] = [
  {
    value: 'square8',
    label: '8×8 Compact',
    subtitle: 'Fast tactical battles',
    blurb: 'Ideal for quick tests, fewer territories, emphasizes captures.',
  },
  {
    value: 'square19',
    label: '19×19 Classic',
    subtitle: 'Full RingRift experience',
    blurb: 'All line lengths and ring counts enabled for marathon sessions.',
  },
  {
    value: 'hex8',
    label: 'Hex 8 Compact',
    subtitle: 'Fast hex tactics',
    blurb: 'Smaller hex board for quick games with 6-direction movement.',
  },
  {
    value: 'hexagonal',
    label: 'Full Hex',
    subtitle: 'High-mobility frontier',
    blurb: 'Hex adjacency, sweeping captures, and large territory swings.',
  },
];

/**
 * Default quick-start presets (5 per board type)
 */
export const QUICK_START_PRESETS: QuickStartPreset[] = [
  // ===== Special presets =====
  {
    id: 'learn-basics',
    label: 'Learn the Basics',
    description: 'Tutorial mode',
    learnMoreText:
      'Perfect for new players! Play against an AI on a compact board to learn movement, captures, and territory.',
    icon: '🎓',
    badge: 'New Player',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  // ===== sq8 presets (5) =====
  {
    id: 'sq8-1h-1ai',
    label: 'Human vs AI',
    description: '1 human, 1 AI',
    learnMoreText:
      'Standard 1v1 match against the AI. Great for learning and practicing strategies.',
    icon: '🤖',
    badge: 'Recommended',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'sq8-2h',
    label: 'Hotseat',
    description: '2 humans',
    learnMoreText:
      'Pass-and-play mode for two players sharing a device. Take turns and compete face-to-face.',
    icon: '👥',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'sq8-2ai',
    label: 'AI Battle',
    description: '2 AIs',
    learnMoreText: 'Watch two AIs compete. Great for learning tactics and understanding game flow.',
    icon: '⚔️',
    config: {
      boardType: 'square8',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'sq8-2h-1ai',
    label: 'Co-op vs AI',
    description: '2 humans, 1 AI',
    learnMoreText: 'Team up with a friend against one AI opponent. Coordinate your strategies!',
    icon: '🤝',
    config: {
      boardType: 'square8',
      numPlayers: 3,
      playerTypes: ['human', 'human', 'ai', 'human'],
    },
  },
  {
    id: 'sq8-1h-2ai',
    label: 'Challenge',
    description: '1 human, 2 AIs',
    learnMoreText: 'Test your skills against two AI opponents. Can you outplay them both?',
    icon: '🎯',
    config: {
      boardType: 'square8',
      numPlayers: 3,
      playerTypes: ['human', 'ai', 'ai', 'human'],
    },
  },
  {
    id: 'sq8-1h-3ai',
    label: '4-Player FFA',
    description: '1 human, 3 AIs',
    learnMoreText: 'Epic 4-player chaos on the compact board. Can you survive against three AIs?',
    icon: '🎲',
    config: {
      boardType: 'square8',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
  // ===== hex8 presets (7) =====
  {
    id: 'learn-basics-hex8',
    label: 'Learn the Basics',
    description: 'Tutorial mode',
    learnMoreText:
      'Perfect for new players! Learn hexagonal movement with 6 directions on a compact board.',
    icon: '🎓',
    badge: 'New Player',
    config: {
      boardType: 'hex8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hex8-1h-1ai',
    label: 'Human vs AI',
    description: '1 human, 1 AI',
    learnMoreText:
      'A smaller hex board for faster games. Same 6-way movement as full hex, but quicker.',
    icon: '🤖',
    config: {
      boardType: 'hex8',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hex8-2h',
    label: 'Hotseat',
    description: '2 humans',
    learnMoreText: 'Pass-and-play mode on compact hex. Take turns with 6-directional movement.',
    icon: '👥',
    config: {
      boardType: 'hex8',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'hex8-2ai',
    label: 'AI Battle',
    description: '2 AIs',
    learnMoreText: 'Watch two AIs compete on compact hex. Fast-paced 6-directional gameplay.',
    icon: '⚔️',
    config: {
      boardType: 'hex8',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hex8-2h-1ai',
    label: 'Co-op vs AI',
    description: '2 humans, 1 AI',
    learnMoreText: 'Team up against the AI on compact hex. Fast co-op battles!',
    icon: '🤝',
    config: {
      boardType: 'hex8',
      numPlayers: 3,
      playerTypes: ['human', 'human', 'ai', 'human'],
    },
  },
  {
    id: 'hex8-1h-2ai',
    label: 'Challenge',
    description: '1 human, 2 AIs',
    learnMoreText: 'Face two AI opponents on compact hex. Quick and intense!',
    icon: '🎯',
    config: {
      boardType: 'hex8',
      numPlayers: 3,
      playerTypes: ['human', 'ai', 'ai', 'human'],
    },
  },
  {
    id: 'hex8-1h-3ai',
    label: '4-Player FFA',
    description: '1 human, 3 AIs',
    learnMoreText: 'Chaotic 4-player hex battle. Fast and furious 6-way combat!',
    icon: '🎲',
    config: {
      boardType: 'hex8',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
  // ===== sq19 presets (6) =====
  {
    id: 'sq19-1h-1ai',
    label: 'Human vs AI',
    description: '1 human, 1 AI',
    learnMoreText: 'Full-length RingRift on 19×19. Best once you are comfortable on 8×8.',
    icon: '🤖',
    config: {
      boardType: 'square19',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'sq19-2h',
    label: 'Hotseat',
    description: '2 humans',
    learnMoreText: 'Pass-and-play on the full 19×19 board. Epic long-form competition.',
    icon: '👥',
    config: {
      boardType: 'square19',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'sq19-2ai',
    label: 'AI Battle',
    description: '2 AIs',
    learnMoreText:
      'Watch two AIs play on 19×19. Great for observing territory and elimination patterns.',
    icon: '⚔️',
    config: {
      boardType: 'square19',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'sq19-2h-1ai',
    label: 'Co-op vs AI',
    description: '2 humans, 1 AI',
    learnMoreText: 'Team up on the full board against one AI. Epic cooperative strategy!',
    icon: '🤝',
    config: {
      boardType: 'square19',
      numPlayers: 3,
      playerTypes: ['human', 'human', 'ai', 'human'],
    },
  },
  {
    id: 'sq19-1h-2ai',
    label: 'Challenge',
    description: '1 human, 2 AIs',
    learnMoreText: 'Face two AIs on the full 19×19 board. Territory control is critical.',
    icon: '🎯',
    config: {
      boardType: 'square19',
      numPlayers: 3,
      playerTypes: ['human', 'ai', 'ai', 'human'],
    },
  },
  {
    id: 'sq19-1h-3ai',
    label: '4-Player FFA',
    description: '1 human, 3 AIs',
    learnMoreText: 'Massive 4-player warfare on 19×19. Strategic depth meets chaos!',
    icon: '🎲',
    config: {
      boardType: 'square19',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
  // ===== hexagonal presets (6) =====
  {
    id: 'hex24-1h-1ai',
    label: 'Human vs AI',
    description: '1 human, 1 AI',
    learnMoreText: 'Full hex board with 6-way movement. Unique tactical possibilities.',
    icon: '🤖',
    config: {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerTypes: ['human', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hex24-2h',
    label: 'Hotseat',
    description: '2 humans',
    learnMoreText: 'Pass-and-play on the full hex board. 6-directional strategic depth.',
    icon: '👥',
    config: {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerTypes: ['human', 'human', 'human', 'human'],
    },
  },
  {
    id: 'hex24-2ai',
    label: 'AI Battle',
    description: '2 AIs',
    learnMoreText: 'Watch two AIs compete on full hex. See 6-directional strategies unfold.',
    icon: '⚔️',
    config: {
      boardType: 'hexagonal',
      numPlayers: 2,
      playerTypes: ['ai', 'ai', 'human', 'human'],
    },
  },
  {
    id: 'hex24-2h-1ai',
    label: 'Co-op vs AI',
    description: '2 humans, 1 AI',
    learnMoreText: 'Team up on the full hex board against one AI. Coordinate 6-way attacks!',
    icon: '🤝',
    config: {
      boardType: 'hexagonal',
      numPlayers: 3,
      playerTypes: ['human', 'human', 'ai', 'human'],
    },
  },
  {
    id: 'hex24-1h-2ai',
    label: 'Challenge',
    description: '1 human, 2 AIs',
    learnMoreText: 'Face two AIs on the full hex board. Balance aggression and defense!',
    icon: '🎯',
    config: {
      boardType: 'hexagonal',
      numPlayers: 3,
      playerTypes: ['human', 'ai', 'ai', 'human'],
    },
  },
  {
    id: 'hex24-1h-3ai',
    label: '4-Player FFA',
    description: '1 human, 3 AIs',
    learnMoreText: 'Epic 4-player battle on the full hex board. Chaotic 6-way warfare!',
    icon: '🎲',
    config: {
      boardType: 'hexagonal',
      numPlayers: 4,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
    },
  },
];

export interface SandboxGameConfigProps {
  // Configuration state
  config: LocalConfig;
  onConfigChange: (partial: Partial<LocalConfig>) => void;
  onPlayerTypeChange: (index: number, type: LocalPlayerType) => void;
  onAIDifficultyChange: (index: number, difficulty: number) => void;

  // Clock configuration
  clockEnabled: boolean;
  onClockEnabledChange: (enabled: boolean) => void;
  timeControl: SandboxTimeControl;
  onResetPlayerTimes: () => void;

  // Actions
  onStartGame: () => void;
  onQuickStartPreset: (preset: QuickStartPreset) => void;
  onShowScenarioPicker: () => void;
  onShowSelfPlayBrowser: () => void;

  // Mode and user info
  isBeginnerMode: boolean;
  onModeChange: (mode: 'beginner' | 'debug') => void;
  developerToolsEnabled: boolean;
  isFirstTimePlayer: boolean;
  isLoggedIn: boolean;

  // Error state
  backendSandboxError: string | null;
}

/**
 * SandboxGameConfig - Pre-game configuration UI for sandbox mode
 *
 * Renders the game configuration screen shown before a sandbox game starts,
 * including board type selection, player configuration, AI difficulty,
 * quick-start presets, clock configuration, and start game button.
 */
export const SandboxGameConfig: React.FC<SandboxGameConfigProps> = ({
  config,
  onConfigChange,
  onPlayerTypeChange,
  onAIDifficultyChange,
  clockEnabled,
  onClockEnabledChange,
  timeControl,
  onResetPlayerTimes,
  onStartGame,
  onQuickStartPreset,
  onShowScenarioPicker,
  onShowSelfPlayBrowser,
  isBeginnerMode,
  onModeChange,
  developerToolsEnabled,
  isFirstTimePlayer,
  isLoggedIn,
  backendSandboxError,
}) => {
  // Show/hide advanced options - collapsed by default for first-time players
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(!isFirstTimePlayer);

  const activePlayerTypes = config.playerTypes.slice(0, config.numPlayers);
  const setupHumanSeatCount = activePlayerTypes.filter((t) => t === 'human').length;
  const setupAiSeatCount = activePlayerTypes.length - setupHumanSeatCount;
  const selectedBoardPreset =
    BOARD_PRESETS.find((preset) => preset.value === config.boardType) ?? BOARD_PRESETS[0];

  const setAllPlayerTypes = (type: LocalPlayerType) => {
    const next = [...config.playerTypes];
    for (let i = 0; i < config.numPlayers; i += 1) {
      next[i] = type;
    }
    onConfigChange({ playerTypes: next });
  };

  return (
    <div className="container mx-auto px-2 sm:px-4 py-4 sm:py-8 space-y-4 sm:space-y-6">
      <header className="flex flex-col gap-2 sm:gap-3 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold mb-1 flex items-center gap-2">
            <img
              src="/ringrift-icon.png"
              alt="RingRift"
              className="w-6 h-6 sm:w-8 sm:h-8 flex-shrink-0"
            />
            <span>RingRift – Start a Game (Sandbox)</span>
          </h1>
          <p className="text-sm text-slate-400">
            This mode runs entirely in the browser using a local board. To view or play a real
            server-backed game, navigate to a URL with a game ID (e.g.
            <code className="ml-1 text-xs text-slate-300">/game/:gameId</code>).
          </p>
        </div>
        {/* Beginner/Debug Mode Toggle */}
        <div className="flex items-center gap-3">
          <div className="inline-flex rounded-lg border border-slate-600 p-0.5 bg-slate-900/60">
            <button
              type="button"
              onClick={() => onModeChange('beginner')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition ${
                isBeginnerMode ? 'bg-emerald-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
              aria-pressed={isBeginnerMode}
            >
              <span className="hidden sm:inline">🎓 </span>Beginner
            </button>
            <button
              type="button"
              onClick={() => onModeChange('debug')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition ${
                !isBeginnerMode ? 'bg-sky-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
              aria-pressed={!isBeginnerMode}
            >
              <span className="hidden sm:inline">🔧 </span>Debug
            </button>
          </div>
          <span className="text-[10px] text-slate-500 hidden lg:inline">
            {isBeginnerMode ? 'Clean learning experience' : 'Developer tools & diagnostics'}
          </span>
        </div>
      </header>

      {/* Quick-start presets */}
      <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-slate-400">Quick Start</p>
            <h2 className="text-lg font-semibold text-white">Choose a preset</h2>
            <p className="text-xs text-slate-400 mt-1">
              {isLoggedIn
                ? 'Click a preset to start a game immediately (backend first, then local fallback).'
                : 'Click a preset to launch a local sandbox game immediately.'}
            </p>
          </div>
          {isFirstTimePlayer && (
            <span className="text-sm text-emerald-400 animate-pulse flex items-center gap-1">
              <span aria-hidden="true">👇</span> Start here
            </span>
          )}
        </div>
        {/* Group presets by board type in rows */}
        <div className="space-y-3">
          {(['square8', 'hex8', 'square19', 'hexagonal'] as BoardType[]).map((boardType) => {
            const presetsForType = QUICK_START_PRESETS.filter(
              (p) => p.config.boardType === boardType
            );
            if (presetsForType.length === 0) return null;

            return (
              <div key={boardType}>
                <p className="text-[10px] uppercase tracking-wide text-slate-500 mb-1.5">
                  {BOARD_TYPE_ROW_HEADERS[boardType]}
                </p>
                <div className="flex flex-wrap gap-2">
                  {presetsForType.map((preset) => {
                    const isLearnBasics =
                      preset.id === 'learn-basics' || preset.id === 'learn-basics-hex8';
                    const shouldHighlight = isLearnBasics && isFirstTimePlayer;
                    const boardLabel = BOARD_TYPE_LABELS[preset.config.boardType];

                    return (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => onQuickStartPreset(preset)}
                        title={preset.learnMoreText}
                        className={`relative flex items-center gap-2 px-3 py-2 rounded-xl border text-slate-200 transition text-sm ${
                          shouldHighlight
                            ? 'border-emerald-400 bg-emerald-900/40 ring-2 ring-emerald-500/50 ring-offset-2 ring-offset-slate-900 animate-pulse hover:animate-none hover:bg-emerald-900/50'
                            : preset.badge
                              ? 'border-emerald-500/50 bg-emerald-900/20 hover:border-emerald-400 hover:bg-emerald-900/30'
                              : 'border-slate-600 bg-slate-900/60 hover:border-emerald-400'
                        } hover:text-emerald-200`}
                      >
                        {preset.badge && (
                          <span
                            className={`absolute -top-2 -right-2 px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wide ${
                              shouldHighlight
                                ? 'bg-emerald-400 text-slate-950'
                                : 'bg-emerald-500 text-slate-950'
                            }`}
                          >
                            {shouldHighlight ? '✨ Start Here' : preset.badge}
                          </span>
                        )}
                        <span className="text-lg" role="img" aria-hidden="true">
                          {preset.icon}
                        </span>
                        <div className="text-left">
                          <p className="font-semibold">
                            {preset.label}
                            <span className="ml-1.5 text-[10px] font-normal text-slate-500">
                              {boardLabel}
                            </span>
                          </p>
                          <p className="text-xs text-slate-400">{preset.description}</p>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Show/Hide Advanced Options Toggle */}
      <button
        type="button"
        onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
        className="flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200 transition"
      >
        <span
          className={`transform transition-transform ${showAdvancedOptions ? 'rotate-90' : ''}`}
        >
          ▶
        </span>
        {showAdvancedOptions ? 'Hide advanced options' : 'Show advanced options'}
        {!showAdvancedOptions && (
          <span className="text-xs text-slate-500">(scenarios, manual setup, AI training)</span>
        )}
      </button>

      {showAdvancedOptions && (
        <>
          {/* Load Scenario section - hidden in beginner mode */}
          {!isBeginnerMode && (
            <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-400">Scenarios</p>
                  <h2 className="text-lg font-semibold text-white">Load a saved scenario</h2>
                </div>
              </div>
              <p className="text-sm text-slate-400 mb-3">
                Load test vectors, curated learning scenarios, or your own saved game states.
              </p>
              <button
                type="button"
                onClick={onShowScenarioPicker}
                className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition text-sm font-medium"
              >
                Browse Scenarios
              </button>
            </section>
          )}

          {/* Self-Play Games section (debug mode only) */}
          {!isBeginnerMode && developerToolsEnabled && (
            <section className="p-4 rounded-2xl bg-slate-800/50 border border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-slate-400">AI Training</p>
                  <h2 className="text-lg font-semibold text-white">Browse self-play games</h2>
                </div>
              </div>
              <p className="text-sm text-slate-400 mb-3">
                Load and replay games recorded during CMA-ES training, self-play soaks, and other AI
                training activities.
              </p>
              <button
                type="button"
                onClick={onShowSelfPlayBrowser}
                className="px-4 py-2 rounded-xl border border-slate-600 bg-slate-900/60 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition text-sm font-medium"
              >
                Browse Self-Play Games
              </button>
            </section>
          )}

          <section className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <div className="p-5 rounded-2xl bg-slate-800/70 border border-slate-700 space-y-6 text-slate-100 shadow-lg">
              {backendSandboxError && (
                <InlineAlert variant="error">{backendSandboxError}</InlineAlert>
              )}

              <div className="space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Players</p>
                    <h2 className="text-lg font-semibold text-white">Seats & control</h2>
                  </div>
                  <div className="flex gap-2 text-xs">
                    {[2, 3, 4].map((count) => (
                      <button
                        key={count}
                        type="button"
                        onClick={() => onConfigChange({ numPlayers: count })}
                        className={`px-2 py-1 rounded-full border ${
                          config.numPlayers === count
                            ? 'border-emerald-400 text-emerald-200 bg-emerald-900/30'
                            : 'border-slate-600 text-slate-300 hover:border-slate-400'
                        }`}
                      >
                        {count} Players
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3">
                  {Array.from({ length: config.numPlayers }, (_, i) => {
                    const type = config.playerTypes[i];
                    const meta = PLAYER_TYPE_META[type];
                    const difficulty = config.aiDifficulties[i];
                    const difficultyDesc = getDifficultyDescriptor(difficulty);
                    return (
                      <div
                        key={i}
                        className={`rounded-xl border bg-slate-900/60 px-4 py-3 ${meta.accent}`}
                      >
                        <div className="flex items-center justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-white">Player {i + 1}</p>
                            <p className="text-xs text-slate-300">
                              {type === 'ai' && difficultyDesc
                                ? `AI • ${difficultyDesc.name.split('(')[0].trim()}`
                                : meta.description}
                            </p>
                          </div>
                          <div className="flex items-center gap-2">
                            {/* AI Difficulty selector - only shown for AI players */}
                            {type === 'ai' && (
                              <AIDifficultyBadge
                                value={difficulty}
                                onChange={(d) => onAIDifficultyChange(i, d)}
                              />
                            )}
                            {/* Player type toggles */}
                            <div className="flex gap-1">
                              {(['human', 'ai'] as LocalPlayerType[]).map((candidate) => {
                                const isActive = type === candidate;
                                return (
                                  <button
                                    key={candidate}
                                    type="button"
                                    onClick={() => onPlayerTypeChange(i, candidate)}
                                    aria-pressed={isActive}
                                    className={`px-3 py-1 rounded-full border text-xs font-semibold transition ${
                                      isActive
                                        ? 'border-white/80 text-white bg-white/10'
                                        : 'border-slate-600 text-slate-300 hover:border-slate-400'
                                    }`}
                                  >
                                    {PLAYER_TYPE_META[candidate].label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        </div>
                        {/* Expanded difficulty selector for AI players */}
                        {type === 'ai' && (
                          <div className="mt-3 pt-3 border-t border-slate-700/50">
                            <AIDifficultySelector
                              value={difficulty}
                              onChange={(d) => onAIDifficultyChange(i, d)}
                              playerNumber={i + 1}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                <div className="flex flex-wrap gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => setAllPlayerTypes('human')}
                    className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-emerald-400 hover:text-emerald-200 transition"
                  >
                    All Human
                  </button>
                  <button
                    type="button"
                    onClick={() => setAllPlayerTypes('ai')}
                    className="px-3 py-1 rounded-full border border-slate-500 text-slate-200 hover:border-sky-400 hover:text-sky-200 transition"
                  >
                    All AI
                  </button>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">Board</p>
                    <h2 className="text-lg font-semibold text-white">Choose a layout</h2>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  {BOARD_PRESETS.map((preset) => {
                    const isSelected = preset.value === config.boardType;
                    return (
                      <button
                        key={preset.value}
                        type="button"
                        onClick={() => onConfigChange({ boardType: preset.value })}
                        className={`p-4 text-left rounded-2xl border transition shadow-sm ${
                          isSelected
                            ? 'border-emerald-400 bg-emerald-900/20 text-white'
                            : 'border-slate-600 bg-slate-900/60 text-slate-200 hover:border-slate-400'
                        }`}
                      >
                        <span className="text-xs uppercase tracking-wide text-slate-400">
                          {preset.subtitle}
                        </span>
                        <p className="text-lg font-semibold">{preset.label}</p>
                        <p className="text-xs text-slate-300 mt-1">{preset.blurb}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-700 text-slate-100 shadow-lg space-y-4">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Summary</p>
                <h2 className="text-xl font-bold text-white">{selectedBoardPreset.label}</h2>
                <p className="text-sm text-slate-300">{selectedBoardPreset.blurb}</p>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Humans</span>
                  <span className="font-semibold">{setupHumanSeatCount}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">AI opponents</span>
                  <span className="font-semibold">{setupAiSeatCount}</span>
                </div>
                {setupAiSeatCount > 0 && (
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">AI strength</span>
                    <span className="font-semibold text-sky-300">
                      {Array.from({ length: config.numPlayers }, (_, i) =>
                        config.playerTypes[i] === 'ai' ? `D${config.aiDifficulties[i]}` : null
                      )
                        .filter(Boolean)
                        .join(', ')}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Total seats</span>
                  <span className="font-semibold">{config.numPlayers}</span>
                </div>
              </div>

              {/* Clock settings */}
              <div className="space-y-3 pt-3 border-t border-slate-700/50">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">Time Control</span>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={clockEnabled}
                      onChange={(e) => {
                        onClockEnabledChange(e.target.checked);
                        // Reset player times when toggling (hook will reinitialize them)
                        if (e.target.checked) {
                          onResetPlayerTimes();
                        }
                      }}
                    />
                    <div className="w-9 h-5 bg-slate-700 peer-focus:ring-2 peer-focus:ring-emerald-400 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-emerald-600" />
                  </label>
                </div>
                {clockEnabled && (
                  <div className="text-xs text-slate-400 space-y-2">
                    <div className="flex items-center justify-between">
                      <span>Initial time</span>
                      <span className="font-mono text-slate-200">
                        {Math.round(timeControl.initialTimeMs / 60000)} min
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span>Increment</span>
                      <span className="font-mono text-slate-200">
                        +{Math.round(timeControl.incrementMs / 1000)}s
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-500">
                      Each player starts with {Math.round(timeControl.initialTimeMs / 60000)}{' '}
                      minutes and gains {Math.round(timeControl.incrementMs / 1000)} seconds after
                      each move.
                    </p>
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <p className="text-xs text-slate-400">
                  We first attempt to stand up a backend game with these settings. If that fails, we
                  fall back to a purely client-local sandbox so you can still test moves offline.
                </p>
                <button
                  type="button"
                  onClick={onStartGame}
                  className="w-full px-4 py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold text-white shadow-lg shadow-emerald-900/40 transition"
                >
                  Launch Game
                </button>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
};
