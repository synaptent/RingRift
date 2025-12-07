/**
 * AccessibilitySettingsPanel - UI for configuring accessibility preferences
 *
 * Provides toggle controls for:
 * - High contrast mode
 * - Colorblind-friendly palettes (deuteranopia, protanopia, tritanopia)
 * - Reduced motion
 * - Large text mode
 */

import { useAccessibility, type ColorVisionMode } from '../contexts/AccessibilityContext';

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ToggleProps {
  id: string;
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

function Toggle({ id, label, description, checked, onChange }: ToggleProps) {
  return (
    <div className="flex items-start justify-between gap-4 py-3">
      <div className="flex-1">
        <label htmlFor={id} className="text-sm font-medium text-slate-200 cursor-pointer">
          {label}
        </label>
        <p className="text-xs text-slate-400 mt-0.5">{description}</p>
      </div>
      <button
        id={id}
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-800 ${
          checked ? 'bg-emerald-500' : 'bg-slate-600'
        }`}
      >
        <span
          aria-hidden="true"
          className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  );
}

interface SelectProps {
  id: string;
  label: string;
  description: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

function SelectOption({ id, label, description, value, options, onChange }: SelectProps) {
  return (
    <div className="py-3">
      <label htmlFor={id} className="text-sm font-medium text-slate-200 block mb-1">
        {label}
      </label>
      <p className="text-xs text-slate-400 mb-2">{description}</p>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-slate-700 border border-slate-600 rounded-md px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Color Vision Preview
// ---------------------------------------------------------------------------

interface ColorPreviewProps {
  mode: ColorVisionMode;
}

function ColorPreview({ mode }: ColorPreviewProps) {
  const { getPlayerColor } = useAccessibility();
  // The preview currently uses the global accessibility state; keep the mode
  // prop for future per-mode previews and mark it as used for type-checking.
  void mode;

  return (
    <div className="flex gap-2 mt-2">
      {[0, 1, 2, 3].map((playerIndex) => (
        <div
          key={playerIndex}
          className="w-8 h-8 rounded-full border-2 border-slate-600 flex items-center justify-center text-xs font-bold text-white"
          style={{ backgroundColor: getPlayerColor(playerIndex) }}
          title={`Player ${playerIndex + 1}`}
        >
          {playerIndex + 1}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

interface AccessibilitySettingsPanelProps {
  /** Optional callback when settings change */
  onSettingsChange?: () => void;
  /** Whether to show in compact mode (fewer descriptions) */
  compact?: boolean;
}

export function AccessibilitySettingsPanel({
  onSettingsChange,
  compact = false,
}: AccessibilitySettingsPanelProps) {
  const {
    highContrastMode,
    colorVisionMode,
    reducedMotion,
    largeText,
    systemPrefersReducedMotion,
    setPreference,
    resetPreferences,
  } = useAccessibility();

  const handleChange = <
    K extends 'highContrastMode' | 'colorVisionMode' | 'reducedMotion' | 'largeText',
  >(
    key: K,
    value: K extends 'colorVisionMode' ? ColorVisionMode : boolean
  ) => {
    setPreference(key, value as never);
    onSettingsChange?.();
  };

  const colorVisionOptions = [
    { value: 'normal', label: 'Standard Colors' },
    { value: 'deuteranopia', label: 'Deuteranopia (Red-Green)' },
    { value: 'protanopia', label: 'Protanopia (Red-Green)' },
    { value: 'tritanopia', label: 'Tritanopia (Blue-Yellow)' },
  ];

  return (
    <div className="bg-slate-800 rounded-lg p-4 space-y-1">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-slate-100">Accessibility</h3>
        <button
          onClick={() => {
            resetPreferences();
            onSettingsChange?.();
          }}
          className="text-xs text-slate-400 hover:text-slate-200 underline focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 rounded"
        >
          Reset to defaults
        </button>
      </div>

      <div className="divide-y divide-slate-700">
        {/* High Contrast Mode */}
        <Toggle
          id="high-contrast"
          label="High Contrast Mode"
          description={
            compact
              ? 'Stronger borders and colors'
              : 'Increases visual distinction with thicker borders, stronger colors, and more prominent focus indicators.'
          }
          checked={highContrastMode}
          onChange={(checked) => handleChange('highContrastMode', checked)}
        />

        {/* Reduced Motion */}
        <div>
          <Toggle
            id="reduced-motion"
            label="Reduce Motion"
            description={
              compact
                ? systemPrefersReducedMotion
                  ? 'System preference detected'
                  : 'Disable animations'
                : systemPrefersReducedMotion
                  ? 'Your system prefers reduced motion. This setting is automatically enabled.'
                  : 'Disables non-essential animations and transitions for a calmer experience.'
            }
            checked={reducedMotion || systemPrefersReducedMotion}
            onChange={(checked) => handleChange('reducedMotion', checked)}
          />
          {systemPrefersReducedMotion && !reducedMotion && (
            <p className="text-xs text-amber-400 -mt-2 mb-2 pl-0">
              ⚠️ System reduced motion is enabled
            </p>
          )}
        </div>

        {/* Large Text */}
        <Toggle
          id="large-text"
          label="Large Text"
          description={
            compact
              ? 'Increase font sizes'
              : 'Increases base font sizes by approximately 25% for better readability.'
          }
          checked={largeText}
          onChange={(checked) => handleChange('largeText', checked)}
        />

        {/* Color Vision Mode */}
        <SelectOption
          id="color-vision"
          label="Color Vision Mode"
          description={
            compact
              ? 'Adjust colors for color blindness'
              : 'Select a color palette optimized for different types of color vision deficiency. Players are also distinguished by patterns.'
          }
          value={colorVisionMode}
          options={colorVisionOptions}
          onChange={(value) => handleChange('colorVisionMode', value as ColorVisionMode)}
        />

        {/* Color Preview */}
        <div className="py-3">
          <p className="text-xs text-slate-400 mb-1">Player color preview:</p>
          <ColorPreview mode={colorVisionMode} />
        </div>
      </div>

      {/* Keyboard shortcuts hint */}
      <div className="mt-4 pt-3 border-t border-slate-700">
        <p className="text-xs text-slate-500">
          Press <kbd className="px-1.5 py-0.5 bg-slate-700 rounded text-slate-300">?</kbd> during
          gameplay for keyboard shortcuts
        </p>
      </div>
    </div>
  );
}

export default AccessibilitySettingsPanel;
