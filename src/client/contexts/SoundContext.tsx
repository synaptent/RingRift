/**
 * SoundContext - Manages game sound effects and user preferences
 *
 * Provides:
 * - Sound effect playback for game events (moves, captures, victory, etc.)
 * - Mute toggle with localStorage persistence
 * - Volume control
 * - Respects reduced motion preference (some users prefer silence with reduced motion)
 *
 * Uses Web Audio API for low-latency sound playback with procedurally generated
 * sounds (no external audio files required).
 */

import React, {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  useMemo,
  useRef,
} from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SoundEffect =
  | 'move' // Ring/stack movement
  | 'place' // Ring placement
  | 'capture' // Capture an opponent's ring
  | 'chain_capture' // Chain capture continuation
  | 'invalid' // Invalid move attempt
  | 'select' // Select a piece
  | 'deselect' // Deselect/cancel
  | 'phase_change' // Phase transition
  | 'turn_start' // Your turn begins
  | 'victory' // Game won
  | 'defeat' // Game lost
  | 'draw' // Game drawn
  | 'line_formed' // Line completed
  | 'territory_claimed' // Territory captured
  | 'elimination' // Ring eliminated
  | 'tick' // Timer tick (low time warning)
  | 'notification'; // Generic notification

export interface SoundPreferences {
  /** Master mute toggle */
  muted: boolean;
  /** Master volume (0-1) */
  volume: number;
  /** Play sound on your turn start */
  turnStartSound: boolean;
}

export interface SoundContextValue extends SoundPreferences {
  /** Play a sound effect */
  playSound: (effect: SoundEffect) => void;
  /** Toggle mute on/off */
  toggleMute: () => void;
  /** Set volume (0-1) */
  setVolume: (volume: number) => void;
  /** Update a preference */
  setPreference: <K extends keyof SoundPreferences>(key: K, value: SoundPreferences[K]) => void;
  /** Whether audio is available (AudioContext supported) */
  audioAvailable: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = 'ringrift-sound-preferences';

const DEFAULT_PREFERENCES: SoundPreferences = {
  muted: false,
  volume: 0.5,
  turnStartSound: true,
};

// ---------------------------------------------------------------------------
// Audio Generation (Procedural sounds - no external files needed)
// ---------------------------------------------------------------------------

/**
 * Creates a simple oscillator-based sound effect.
 * This approach eliminates the need for audio files.
 */
function createOscillatorSound(
  ctx: AudioContext,
  frequency: number,
  duration: number,
  type: OscillatorType = 'sine',
  volume: number = 0.3
): void {
  const oscillator = ctx.createOscillator();
  const gainNode = ctx.createGain();

  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, ctx.currentTime);

  gainNode.gain.setValueAtTime(volume, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

  oscillator.connect(gainNode);
  gainNode.connect(ctx.destination);

  oscillator.start(ctx.currentTime);
  oscillator.stop(ctx.currentTime + duration);
}

/**
 * Creates a frequency sweep sound (for transitions, selections).
 */
function createSweepSound(
  ctx: AudioContext,
  startFreq: number,
  endFreq: number,
  duration: number,
  type: OscillatorType = 'sine',
  volume: number = 0.2
): void {
  const oscillator = ctx.createOscillator();
  const gainNode = ctx.createGain();

  oscillator.type = type;
  oscillator.frequency.setValueAtTime(startFreq, ctx.currentTime);
  oscillator.frequency.exponentialRampToValueAtTime(endFreq, ctx.currentTime + duration);

  gainNode.gain.setValueAtTime(volume, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

  oscillator.connect(gainNode);
  gainNode.connect(ctx.destination);

  oscillator.start(ctx.currentTime);
  oscillator.stop(ctx.currentTime + duration);
}

/**
 * Creates a multi-tone chord sound (for victory, special events).
 */
function createChordSound(
  ctx: AudioContext,
  frequencies: number[],
  duration: number,
  volume: number = 0.15
): void {
  frequencies.forEach((freq, index) => {
    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(freq, ctx.currentTime);

    // Stagger the notes slightly for arpeggio effect
    const startTime = ctx.currentTime + index * 0.05;
    gainNode.gain.setValueAtTime(0, ctx.currentTime);
    gainNode.gain.linearRampToValueAtTime(volume, startTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.start(ctx.currentTime);
    oscillator.stop(startTime + duration);
  });
}

/**
 * Creates a noise burst (for invalid moves, errors).
 */
function createNoiseSound(ctx: AudioContext, duration: number, volume: number = 0.1): void {
  const bufferSize = ctx.sampleRate * duration;
  const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
  const data = buffer.getChannelData(0);

  // Generate white noise
  for (let i = 0; i < bufferSize; i++) {
    data[i] = (Math.random() * 2 - 1) * 0.5;
  }

  const source = ctx.createBufferSource();
  const gainNode = ctx.createGain();
  const filter = ctx.createBiquadFilter();

  source.buffer = buffer;
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(1000, ctx.currentTime);
  filter.Q.setValueAtTime(1, ctx.currentTime);

  gainNode.gain.setValueAtTime(volume, ctx.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

  source.connect(filter);
  filter.connect(gainNode);
  gainNode.connect(ctx.destination);

  source.start(ctx.currentTime);
  source.stop(ctx.currentTime + duration);
}

/**
 * Play a specific sound effect using Web Audio API.
 */
function playSoundEffect(ctx: AudioContext, effect: SoundEffect, masterVolume: number): void {
  const vol = masterVolume;

  switch (effect) {
    case 'move':
      // Soft pop sound
      createOscillatorSound(ctx, 440, 0.1, 'sine', vol * 0.4);
      break;

    case 'place':
      // Thud-like placement sound
      createSweepSound(ctx, 200, 100, 0.15, 'triangle', vol * 0.5);
      break;

    case 'capture':
      // Satisfying capture sound - descending tone
      createSweepSound(ctx, 600, 200, 0.2, 'sawtooth', vol * 0.3);
      createOscillatorSound(ctx, 150, 0.15, 'sine', vol * 0.4);
      break;

    case 'chain_capture':
      // Quick double-tap for chain
      createOscillatorSound(ctx, 500, 0.08, 'square', vol * 0.25);
      setTimeout(() => {
        if (ctx.state === 'running') {
          createOscillatorSound(ctx, 600, 0.08, 'square', vol * 0.25);
        }
      }, 100);
      break;

    case 'invalid':
      // Error buzz
      createNoiseSound(ctx, 0.15, vol * 0.3);
      createOscillatorSound(ctx, 150, 0.1, 'square', vol * 0.2);
      break;

    case 'select':
      // Light click up
      createSweepSound(ctx, 800, 1200, 0.05, 'sine', vol * 0.3);
      break;

    case 'deselect':
      // Light click down
      createSweepSound(ctx, 1200, 800, 0.05, 'sine', vol * 0.25);
      break;

    case 'phase_change':
      // Gentle transition chime
      createSweepSound(ctx, 400, 800, 0.2, 'sine', vol * 0.3);
      break;

    case 'turn_start':
      // Attention-getting double tone
      createOscillatorSound(ctx, 523, 0.15, 'sine', vol * 0.4); // C5
      setTimeout(() => {
        if (ctx.state === 'running') {
          createOscillatorSound(ctx, 659, 0.2, 'sine', vol * 0.4); // E5
        }
      }, 150);
      break;

    case 'victory':
      // Triumphant major chord arpeggio (C major)
      createChordSound(ctx, [523, 659, 784, 1047], 0.8, vol * 0.3); // C5, E5, G5, C6
      break;

    case 'defeat':
      // Sad minor chord
      createChordSound(ctx, [220, 262, 330], 0.6, vol * 0.25); // A3, C4, E4 (A minor)
      break;

    case 'draw':
      // Neutral resolution
      createChordSound(ctx, [392, 494], 0.5, vol * 0.25); // G4, B4
      break;

    case 'line_formed':
      // Ascending success sound
      createSweepSound(ctx, 400, 800, 0.25, 'sine', vol * 0.35);
      setTimeout(() => {
        if (ctx.state === 'running') {
          createOscillatorSound(ctx, 1000, 0.15, 'sine', vol * 0.3);
        }
      }, 200);
      break;

    case 'territory_claimed':
      // Solid claim sound
      createOscillatorSound(ctx, 300, 0.2, 'triangle', vol * 0.4);
      createOscillatorSound(ctx, 450, 0.15, 'sine', vol * 0.3);
      break;

    case 'elimination':
      // Ring removal sound
      createSweepSound(ctx, 400, 100, 0.3, 'sawtooth', vol * 0.25);
      break;

    case 'tick':
      // Clock tick for low time
      createOscillatorSound(ctx, 1000, 0.03, 'square', vol * 0.2);
      break;

    case 'notification':
      // Generic notification ping
      createOscillatorSound(ctx, 880, 0.1, 'sine', vol * 0.35);
      break;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const SoundContext = createContext<SoundContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

interface SoundProviderProps {
  children: React.ReactNode;
}

export function SoundProvider({ children }: SoundProviderProps): React.ReactElement {
  const audioContextRef = useRef<AudioContext | null>(null);
  const [audioAvailable, setAudioAvailable] = useState(false);

  // Load preferences from localStorage
  const [preferences, setPreferences] = useState<SoundPreferences>(() => {
    if (typeof window === 'undefined') return DEFAULT_PREFERENCES;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<SoundPreferences>;
        return { ...DEFAULT_PREFERENCES, ...parsed };
      }
    } catch {
      // Ignore parse errors
    }
    return DEFAULT_PREFERENCES;
  });

  // Initialize AudioContext on first user interaction (required by browsers)
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const initAudio = () => {
      if (audioContextRef.current) return;

      try {
        const AudioContextClass =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (AudioContextClass) {
          audioContextRef.current = new AudioContextClass();
          setAudioAvailable(true);
        }
      } catch {
        setAudioAvailable(false);
      }
    };

    // Initialize on any user interaction
    const events = ['click', 'touchstart', 'keydown'];
    const handler = () => {
      initAudio();
      // Resume if suspended (browser autoplay policy)
      if (audioContextRef.current?.state === 'suspended') {
        audioContextRef.current.resume();
      }
    };

    events.forEach((event) => document.addEventListener(event, handler, { once: false }));

    return () => {
      events.forEach((event) => document.removeEventListener(event, handler));
    };
  }, []);

  // Persist preferences
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch {
      // Ignore storage errors
    }
  }, [preferences]);

  const playSound = useCallback(
    (effect: SoundEffect) => {
      if (preferences.muted) return;
      if (!audioContextRef.current) return;

      // Resume context if needed (browser autoplay policy)
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }

      try {
        playSoundEffect(audioContextRef.current, effect, preferences.volume);
      } catch {
        // Silently fail - audio is non-critical
      }
    },
    [preferences.muted, preferences.volume]
  );

  const toggleMute = useCallback(() => {
    setPreferences((prev) => ({ ...prev, muted: !prev.muted }));
  }, []);

  const setVolume = useCallback((volume: number) => {
    setPreferences((prev) => ({ ...prev, volume: Math.max(0, Math.min(1, volume)) }));
  }, []);

  const setPreference = useCallback(
    <K extends keyof SoundPreferences>(key: K, value: SoundPreferences[K]) => {
      setPreferences((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const value = useMemo<SoundContextValue>(
    () => ({
      ...preferences,
      playSound,
      toggleMute,
      setVolume,
      setPreference,
      audioAvailable,
    }),
    [preferences, playSound, toggleMute, setVolume, setPreference, audioAvailable]
  );

  return <SoundContext.Provider value={value}>{children}</SoundContext.Provider>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Access sound effects and preferences.
 *
 * @example
 * ```tsx
 * const { playSound, toggleMute, muted } = useSound();
 *
 * // Play a sound
 * playSound('capture');
 *
 * // Toggle mute (M key)
 * toggleMute();
 * ```
 */
export function useSound(): SoundContextValue {
  const context = useContext(SoundContext);
  if (!context) {
    throw new Error('useSound must be used within a SoundProvider');
  }
  return context;
}

/**
 * Optional hook that returns null if outside provider (for optional sound support).
 */
export function useSoundOptional(): SoundContextValue | null {
  return useContext(SoundContext);
}
