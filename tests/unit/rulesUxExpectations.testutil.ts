// Test-only helper module: canonical rules-aware UX phrases used by regression tests.
// These snippets are derived from UX_RULES_COPY_SPEC.md and related rules docs.
// Tests use case-insensitive substring/regex checks rather than strict equality.
export const RulesUxPhrases = {
  victory: {
    ringElimination: [
      // Ring Elimination – winning by eliminating more than half of all rings globally.
      // Shared substring between VictoryModal copy (“in play”) and curated scenarios (“in the game”).
      'eliminating more than half of all rings',
      // Used primarily by TeachingOverlay and curated scenarios.
      'Eliminated rings are permanently removed',
    ],
    territory: [
      // Territory Control – owning more than half of all board spaces as Territory.
      'more than half of all board spaces as Territory',
    ],
    structuralStalemate: [
      // Structural stalemate / plateau – no legal moves or forced eliminations remain.
      'structural stalemate',
      // Tiebreak ladder enumeration in VictoryModal shows territory and eliminated rings separately.
      'territory spaces',
      'eliminated rings',
    ],
  },
  movement: {
    stackHeight: [
      // Movement semantics – minimum distance based on stack height.
      // Use a shared substring that appears in both TeachingOverlay and curated scenarios,
      // regardless of whether they say "its height" or "the stack’s height".
      'at least as many spaces as',
    ],
  },
  capture: {
    basic: [
      // Basic overtaking capture – captured rings stay in play (distinct from elimination).
      'captured rings stay in play',
    ],
    chainMandatory: [
      // Chain capture semantics – once the chain begins you must keep capturing while any capture exists.
      'once the chain begins you must keep capturing as long as any capture is available',
    ],
  },
  feAnm: {
    activeNoMoves: [
      // Active–No–Moves (ANM) – no real moves available this turn.
      'no legal placements, movements, or captures',
      'forced elimination will now resolve automatically',
    ],
    forcedElimination: [
      // Forced Elimination (FE) – caps removed automatically until a real move is available.
      'Forced Elimination',
      'caps are removed from your stacks automatically until either a real move becomes available or your stacks are gone',
      'does not count as a “real move” for Last Player Standing',
    ],
  },
  lines: {
    overlengthOption2: [
      // Overlength line option 2 – collapse a shorter scoring segment and skip elimination.
      'overlength lines',
      'collapse a minimal scoring segment',
      'skip elimination',
    ],
  },
  territory: {
    basicRegion: [
      // Territory processing – disconnected regions and elimination in the region.
      'disconnected region you control is processed',
      'rings in the region are eliminated',
    ],
    miniRegionQ23: [
      // Q23 mini-region edge case phrasing; currently used for docs / future scenarios.
      'disconnected groups of empty cells',
      'rings inside the region are eliminated',
    ],
  },
} as const;
