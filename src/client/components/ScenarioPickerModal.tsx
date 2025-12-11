import React, { useState, useEffect, useRef, useCallback } from 'react';
import type { LoadableScenario, ScenarioCategory } from '../sandbox/scenarioTypes';
import { CATEGORY_LABELS, DIFFICULTY_LABELS } from '../sandbox/scenarioTypes';
import {
  loadVectorScenarios,
  loadCuratedScenarios,
  loadCustomScenarios,
  deleteCustomScenario,
  filterScenarios,
} from '../sandbox/scenarioLoader';
import { importScenarioFromFile, exportScenarioToFile } from '../sandbox/statePersistence';

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

type TabType = 'curated' | 'vectors' | 'custom';

export interface ScenarioPickerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectScenario: (scenario: LoadableScenario) => void;
  /**
   * When true, surfaces advanced scenario-management affordances
   * (import/export/delete, RulesMatrix tags, etc.). Defaults to
   * false so the default sandbox UX remains player-first.
   */
  developerToolsEnabled?: boolean;
}

export const ScenarioPickerModal: React.FC<ScenarioPickerModalProps> = ({
  isOpen,
  onClose,
  onSelectScenario,
  developerToolsEnabled = false,
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('curated');
  const [scenarios, setScenarios] = useState<{
    curated: LoadableScenario[];
    vectors: LoadableScenario[];
    custom: LoadableScenario[];
  }>({ curated: [], vectors: [], custom: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<ScenarioCategory | 'all'>('all');
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Load scenarios when modal opens
  useEffect(() => {
    if (!isOpen) return;

    const loadAll = async () => {
      setLoading(true);
      setError(null);
      try {
        const [curated, vectors] = await Promise.all([
          loadCuratedScenarios(),
          loadVectorScenarios(),
        ]);
        const custom = loadCustomScenarios();
        setScenarios({ curated, vectors, custom });
      } catch (err) {
        setError('Failed to load scenarios');
        if (process.env.NODE_ENV !== 'test') {
          console.error('Failed to load scenarios:', err);
        }
      } finally {
        setLoading(false);
      }
    };

    loadAll();
  }, [isOpen]);

  // Focus trap
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    if (!dialogEl) return;

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (first) {
      first.focus();
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key !== 'Tab' || focusable.length === 0) return;

      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      const isShift = event.shiftKey;

      if (isShift && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!isShift && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    dialogEl.addEventListener('keydown', handleKeyDown);
    return () => dialogEl.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleDelete = useCallback((id: string) => {
    deleteCustomScenario(id);
    setScenarios((prev) => ({
      ...prev,
      custom: prev.custom.filter((s) => s.id !== id),
    }));
  }, []);

  const handleExport = useCallback((scenario: LoadableScenario) => {
    exportScenarioToFile(scenario);
  }, []);

  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const scenario = await importScenarioFromFile(file);
      // Add to custom scenarios
      setScenarios((prev) => ({
        ...prev,
        custom: [scenario, ...prev.custom.filter((s) => s.id !== scenario.id)],
      }));
      setActiveTab('custom');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import scenario');
    }

    // Reset file input
    event.target.value = '';
  }, []);

  if (!isOpen) return null;

  const currentScenarios = scenarios[activeTab];
  const filteredScenarios = filterScenarios(currentScenarios, {
    category: categoryFilter,
    searchQuery,
  });

  const onboardingScenarios =
    activeTab === 'curated' ? filteredScenarios.filter((scenario) => scenario.onboarding) : [];
  const nonOnboardingScenarios =
    activeTab === 'curated'
      ? filteredScenarios.filter((scenario) => !scenario.onboarding)
      : filteredScenarios;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="scenario-picker-title"
    >
      <div
        ref={dialogRef}
        className="bg-slate-900 rounded-2xl border border-slate-700 w-full max-w-3xl max-h-[80vh] flex flex-col shadow-2xl"
      >
        {/* Header */}
        <div className="p-4 border-b border-slate-700 flex justify-between items-center">
          <h2 id="scenario-picker-title" className="text-xl font-bold text-white">
            Load Scenario
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors p-1"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          {(['curated', 'vectors', 'custom'] as TabType[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab
                  ? 'text-emerald-400 border-b-2 border-emerald-400'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {tab === 'curated'
                ? 'Learning (Rules / FAQ)'
                : tab === 'vectors'
                  ? 'Test Scenarios'
                  : 'My Scenarios'}
              <span className="ml-1 text-xs opacity-60">({scenarios[tab].length})</span>
            </button>
          ))}
        </div>

        {/* Filters */}
        <div className="p-3 border-b border-slate-700 flex gap-3 flex-wrap">
          <input
            type="text"
            placeholder="Search scenarios..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 min-w-[200px] px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value as ScenarioCategory | 'all')}
            className="px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-600 text-sm text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            <option value="all">All Categories</option>
            {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
          {activeTab === 'custom' && developerToolsEnabled && (
            <>
              <button
                onClick={handleImportClick}
                className="px-3 py-1.5 rounded-lg bg-slate-700 border border-slate-600 text-sm text-white hover:bg-slate-600 transition-colors"
              >
                Import JSON
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileChange}
                className="hidden"
                aria-hidden="true"
              />
            </>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mx-4 mt-3 p-3 rounded-lg bg-red-900/30 border border-red-700 text-red-300 text-sm">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-red-400 hover:text-red-200">
              Dismiss
            </button>
          </div>
        )}

        {/* Scenario List */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="text-center text-slate-400 py-8">Loading scenarios...</div>
          ) : filteredScenarios.length === 0 ? (
            <div className="text-center text-slate-400 py-8">
              {activeTab === 'custom'
                ? 'No saved scenarios yet. Save a game state or import a JSON file to see it here.'
                : activeTab === 'curated'
                  ? 'No curated scenarios available yet.'
                  : 'No scenarios match your filters.'}
            </div>
          ) : (
            <div className="space-y-4">
              {activeTab === 'curated' && onboardingScenarios.length > 0 && (
                <section className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-wide text-emerald-400">Onboarding</p>
                      <p className="text-xs text-slate-400">
                        Recommended rules and FAQ scenarios for first-time players.
                      </p>
                    </div>
                  </div>
                  <div className="grid gap-3">
                    {onboardingScenarios.map((scenario) => (
                      <ScenarioCard
                        key={scenario.id}
                        scenario={scenario}
                        onSelect={() => {
                          try {
                            onSelectScenario(scenario);
                          } finally {
                            onClose();
                          }
                        }}
                        developerToolsEnabled={developerToolsEnabled}
                        showRulesSnippet={!!scenario.rulesSnippet}
                      />
                    ))}
                  </div>
                </section>
              )}

              <div className="grid gap-3">
                {nonOnboardingScenarios.map((scenario) => (
                  <ScenarioCard
                    key={scenario.id}
                    scenario={scenario}
                    onSelect={() => {
                      try {
                        onSelectScenario(scenario);
                      } finally {
                        onClose();
                      }
                    }}
                    onDelete={
                      activeTab === 'custom' && developerToolsEnabled
                        ? () => handleDelete(scenario.id)
                        : undefined
                    }
                    onExport={
                      activeTab === 'custom' && developerToolsEnabled
                        ? () => handleExport(scenario)
                        : undefined
                    }
                    developerToolsEnabled={developerToolsEnabled}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

interface ScenarioCardProps {
  scenario: LoadableScenario;
  onSelect: () => void;
  onDelete?: (() => void) | undefined;
  onExport?: (() => void) | undefined;
  developerToolsEnabled?: boolean;
  /** When true, renders the scenario.rulesSnippet callout if present. */
  showRulesSnippet?: boolean;
}

const ScenarioCard: React.FC<ScenarioCardProps> = ({
  scenario,
  onSelect,
  onDelete,
  onExport,
  developerToolsEnabled = false,
  showRulesSnippet = false,
}) => {
  const rulesTags = developerToolsEnabled
    ? scenario.tags.filter((tag) => tag.startsWith('Rules_'))
    : [];
  const otherTags = developerToolsEnabled
    ? scenario.tags.filter((tag) => !tag.startsWith('Rules_'))
    : [];

  return (
    <div className="p-3 rounded-xl border border-slate-700 bg-slate-800/50 flex justify-between items-start gap-3 hover:border-slate-600 transition-colors">
      <div className="flex-1 min-w-0">
        <h3 className="font-medium text-white truncate">{scenario.name}</h3>
        <p className="text-sm text-slate-400 line-clamp-2 mt-1">{scenario.description}</p>
        {showRulesSnippet && scenario.rulesSnippet && (
          <div className="mt-2 px-3 py-2 rounded-lg bg-emerald-900/40 border border-emerald-700/60 text-[11px] text-emerald-100">
            <span className="font-semibold">Rules context:</span>{' '}
            <span>{scenario.rulesSnippet}</span>
          </div>
        )}
        <div className="flex flex-wrap gap-1 mt-2">
          <span className="px-2 py-0.5 text-xs rounded-full bg-slate-700 text-slate-300">
            {CATEGORY_LABELS[scenario.category] || scenario.category}
          </span>
          <span className="px-2 py-0.5 text-xs rounded-full bg-slate-700 text-slate-300">
            {scenario.boardType}
          </span>
          <span className="px-2 py-0.5 text-xs rounded-full bg-slate-700 text-slate-300">
            {scenario.playerCount}P
          </span>
          {scenario.difficulty && (
            <span
              className={`px-2 py-0.5 text-xs rounded-full ${
                scenario.difficulty === 'beginner'
                  ? 'bg-green-900/50 text-green-300'
                  : scenario.difficulty === 'intermediate'
                    ? 'bg-yellow-900/50 text-yellow-300'
                    : 'bg-red-900/50 text-red-300'
              }`}
            >
              {DIFFICULTY_LABELS[scenario.difficulty]}
            </span>
          )}
          {developerToolsEnabled && rulesTags.length > 0 && (
            <span className="px-2 py-0.5 text-[10px] rounded-full bg-emerald-900/60 text-emerald-300">
              {rulesTags.length === 1
                ? `RulesMatrix: ${rulesTags[0]}`
                : `RulesMatrix: ${rulesTags.join(', ')}`}
            </span>
          )}
          {developerToolsEnabled && otherTags.length > 0 && (
            <span className="px-2 py-0.5 text-[10px] rounded-full bg-slate-700 text-slate-300">
              Tags: {otherTags.join(', ')}
            </span>
          )}
        </div>
      </div>
      <div className="flex gap-2 flex-shrink-0">
        {onExport && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onExport();
            }}
            className="px-2 py-1 text-xs rounded border border-slate-600 text-slate-400 hover:bg-slate-700 hover:text-white transition-colors"
            title="Export as JSON"
          >
            Export
          </button>
        )}
        {onDelete && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="px-2 py-1 text-xs rounded border border-red-600 text-red-400 hover:bg-red-900/30 transition-colors"
            title="Delete scenario"
          >
            Delete
          </button>
        )}
        <button
          onClick={onSelect}
          className="px-3 py-1 text-sm rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors"
        >
          Load
        </button>
      </div>
    </div>
  );
};

export default ScenarioPickerModal;
