import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TeachingOverlay } from '../../src/client/components/TeachingOverlay';
import { RulesUxPhrases } from './rulesUxExpectations.testutil';

describe('TeachingOverlay UX regression – rules-aware copy', () => {
  it('explains Active–No–Moves without treating forced elimination as a real move', () => {
    render(
      <TeachingOverlay topic="active_no_moves" isOpen={true} onClose={() => {}} position="center" />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // Canonical ANM phrasing: no legal placements, movements, or captures.
    expect(dialog).toHaveTextContent(/no legal placements, movements, or captures/i);

    // Canonical ANM label: explicitly called out as an Active–No–Moves state.
    expect(dialog).toHaveTextContent(/Active.?No.?Moves state/i);

    // Real-move vs FE distinction for LPS must be explicit.
    expect(dialog).toHaveTextContent(/Real moves.*placements, movements, and captures/i);
    expect(text).toMatch(/do not count as real moves? for Last Player Standing/i);

    // Sanity: ensure the overlay does not accidentally claim FE is a real move.
    expect(text).not.toMatch(/forced elimination.*counts as a real move/i);
  });

  it('describes Forced Elimination semantics and LPS relationship canonically', () => {
    render(
      <TeachingOverlay
        topic="forced_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // Header/title should contain the canonical label.
    expect(dialog).toHaveTextContent(/Forced Elimination/i);

    // Forced Elimination happens when you control stacks but have no real moves.
    expect(dialog).toHaveTextContent(/no legal placements, movements, or captures/i);

    // Caps are removed until a real move becomes available or stacks are gone.
    // (Copy updated: "automatically" removed for conciseness)
    expect(text).toMatch(
      /caps are removed from your stacks.*until either a real move becomes available or your stacks are gone/i
    );

    // FE eliminations are permanent and count toward Ring Elimination.
    expect(text).toMatch(/permanently eliminated and count toward.*Ring Elimination/i);

    // FE does not count as a real move for LPS (canonical snippet from RulesUxPhrases).
    expect(text).toMatch(new RegExp(RulesUxPhrases.feAnm.forcedElimination[2], 'i'));
  });

  it('Victory: Elimination topic matches global Ring Elimination semantics', () => {
    render(
      <TeachingOverlay
        topic="victory_elimination"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // Elimination victory is based on reaching ringsPerPlayer (starting ring supply).
    expect(text).toMatch(/eliminating a number of rings equal to the starting ring supply/i);

    // Eliminated rings are permanently removed and distinct from captured rings.
    expect(text).toMatch(/Eliminated rings are permanently removed/i);
    expect(text).toMatch(/captured rings you carry in stacks do not count toward this threshold/i);
  });

  it('Victory: Territory topic matches Territory Control semantics', () => {
    render(
      <TeachingOverlay
        topic="victory_territory"
        isOpen={true}
        onClose={() => {}}
        position="center"
      />
    );

    const dialog = screen.getByRole('dialog');
    const text = dialog.textContent || '';

    // Territory victory is based on owning more than half of all board spaces as Territory.
    expect(text).toMatch(/owning more than half of all board spaces as Territory/i);

    // Territory is permanent once claimed.
    expect(text).toMatch(/once a space becomes Territory it can.?t be captured back/i);
  });
});
