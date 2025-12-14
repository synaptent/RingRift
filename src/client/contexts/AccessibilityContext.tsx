/**
 * AccessibilityContext - Manages user accessibility preferences
 *
 * Provides settings for:
 * - High contrast mode (increased borders, stronger colors)
 * - Colorblind-friendly palette (patterns + distinct colors)
 * - Reduced motion (respects prefers-reduced-motion, plus manual override)
 *
 * Preferences are persisted to localStorage and exposed via useAccessibility hook.
 */

import React, { createContext, useContext, useCallback, useEffect, useState, useMemo } from 'react';
import {
  getPlayerColorClass as getPlayerColorClassFromTheme,
  getPlayerColorHex,
  PLAYER_COLOR_CLASSES,
  PLAYER_COLOR_PALETTES,
  type ColorVisionMode,
} from '../utils/playerTheme';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AccessibilityPreferences {
  /** High contrast mode - stronger borders, higher color contrast */
  highContrastMode: boolean;
  /** Color vision deficiency mode */
  colorVisionMode: ColorVisionMode;
  /** Reduced motion - disables animations */
  reducedMotion: boolean;
  /** Large text mode - increases base font sizes */
  largeText: boolean;
}

export interface AccessibilityContextValue extends AccessibilityPreferences {
  /** Update a single preference */
  setPreference: <K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => void;
  /** Reset all preferences to defaults */
  resetPreferences: () => void;
  /** Whether system prefers reduced motion */
  systemPrefersReducedMotion: boolean;
  /** Effective reduced motion (user setting OR system preference) */
  effectiveReducedMotion: boolean;
  /** Get player color class based on current color vision mode */
  getPlayerColorClass: (playerIndex: number, type: 'bg' | 'text' | 'border' | 'ring') => string;
  /** Get player color for SVG/canvas (hex value) */
  getPlayerColor: (playerIndex: number) => string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = 'ringrift-accessibility-preferences';

const DEFAULT_PREFERENCES: AccessibilityPreferences = {
  highContrastMode: false,
  colorVisionMode: 'normal',
  reducedMotion: false,
  largeText: false,
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const DEFAULT_CONTEXT_VALUE: AccessibilityContextValue = {
  ...DEFAULT_PREFERENCES,
  setPreference: () => {},
  resetPreferences: () => {},
  systemPrefersReducedMotion: false,
  effectiveReducedMotion: false,
  getPlayerColorClass: (playerIndex: number, type: 'bg' | 'text' | 'border' | 'ring') =>
    getPlayerColorClassFromTheme(playerIndex, DEFAULT_PREFERENCES.colorVisionMode, type),
  getPlayerColor: (playerIndex: number) =>
    getPlayerColorHex(playerIndex, DEFAULT_PREFERENCES.colorVisionMode),
};

const AccessibilityContext = createContext<AccessibilityContextValue>(DEFAULT_CONTEXT_VALUE);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export function AccessibilityProvider({
  children,
}: AccessibilityProviderProps): React.ReactElement {
  // Load initial preferences from localStorage
  const [preferences, setPreferences] = useState<AccessibilityPreferences>(() => {
    if (typeof window === 'undefined') return DEFAULT_PREFERENCES;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<AccessibilityPreferences>;
        return { ...DEFAULT_PREFERENCES, ...parsed };
      }
    } catch {
      // Ignore parse errors
    }
    return DEFAULT_PREFERENCES;
  });

  // Track system preference for reduced motion
  const [systemPrefersReducedMotion, setSystemPrefersReducedMotion] = useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  // Listen for system preference changes
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e: MediaQueryListEvent) => setSystemPrefersReducedMotion(e.matches);

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    }

    mediaQuery.addListener(handler);
    return () => mediaQuery.removeListener(handler);
  }, []);

  // Persist preferences to localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch {
      // Ignore storage errors
    }
  }, [preferences]);

  // Apply CSS class to document root for global styling
  useEffect(() => {
    if (typeof document === 'undefined') return;

    const root = document.documentElement;

    // High contrast mode
    root.classList.toggle('high-contrast', preferences.highContrastMode);

    // Color vision mode
    root.dataset.colorVision = preferences.colorVisionMode;

    // Reduced motion (explicit user preference overrides system)
    const effectiveReducedMotion = preferences.reducedMotion || systemPrefersReducedMotion;
    root.classList.toggle('reduce-motion', effectiveReducedMotion);

    // Large text
    root.classList.toggle('large-text', preferences.largeText);
  }, [preferences, systemPrefersReducedMotion]);

  const setPreference = useCallback(
    <K extends keyof AccessibilityPreferences>(key: K, value: AccessibilityPreferences[K]) => {
      setPreferences((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const resetPreferences = useCallback(() => {
    setPreferences(DEFAULT_PREFERENCES);
  }, []);

  const effectiveReducedMotion = preferences.reducedMotion || systemPrefersReducedMotion;

  const getPlayerColorClass = useCallback(
    (playerIndex: number, type: 'bg' | 'text' | 'border' | 'ring'): string => {
      return getPlayerColorClassFromTheme(playerIndex, preferences.colorVisionMode, type);
    },
    [preferences.colorVisionMode]
  );

  const getPlayerColor = useCallback(
    (playerIndex: number): string => {
      return getPlayerColorHex(playerIndex, preferences.colorVisionMode);
    },
    [preferences.colorVisionMode]
  );

  const value = useMemo<AccessibilityContextValue>(
    () => ({
      ...preferences,
      setPreference,
      resetPreferences,
      systemPrefersReducedMotion,
      effectiveReducedMotion,
      getPlayerColorClass,
      getPlayerColor,
    }),
    [
      preferences,
      setPreference,
      resetPreferences,
      systemPrefersReducedMotion,
      effectiveReducedMotion,
      getPlayerColorClass,
      getPlayerColor,
    ]
  );

  return <AccessibilityContext.Provider value={value}>{children}</AccessibilityContext.Provider>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Access accessibility preferences and helpers.
 *
 * @example
 * ```tsx
 * const { highContrastMode, setPreference, getPlayerColorClass } = useAccessibility();
 *
 * // Toggle high contrast
 * setPreference('highContrastMode', !highContrastMode);
 *
 * // Get player color class
 * const bgClass = getPlayerColorClass(playerIndex, 'bg');
 * ```
 */
export function useAccessibility(): AccessibilityContextValue {
  return useContext(AccessibilityContext);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export type { ColorVisionMode } from '../utils/playerTheme';
export { PLAYER_COLOR_PALETTES, PLAYER_COLOR_CLASSES };
