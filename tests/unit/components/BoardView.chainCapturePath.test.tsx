import React from 'react';
import { render } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import type { BoardState, Position } from '../../../src/shared/types/game';

function createBoard(): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };
}

describe('BoardView chain capture path overlay', () => {
  it('renders chain capture arrow overlay when path has multiple positions', () => {
    const rectSpy = jest
      .spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockImplementation(function getRect(this: HTMLElement) {
        const xAttr = this.getAttribute?.('data-x');
        const yAttr = this.getAttribute?.('data-y');
        const base = 20;
        const left = xAttr != null ? Number(xAttr) * base : 0;
        const top = yAttr != null ? Number(yAttr) * base : 0;
        const width = 20;
        const height = 20;

        return {
          x: left,
          y: top,
          width,
          height,
          top,
          left,
          right: left + width,
          bottom: top + height,
          toJSON: () => ({}),
        } as DOMRect;
      });

    const board = createBoard();
    // Provide two positions so the overlay draws at least one segment.
    const chainCapturePath: Position[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 2, y: 1 },
    ];

    const { container, rerender } = render(
      <BoardView boardType="square8" board={board} chainCapturePath={chainCapturePath} />
    );

    // Rerender to allow refs to be set before overlay computation.
    rerender(<BoardView boardType="square8" board={board} chainCapturePath={chainCapturePath} />);

    const overlay = container.querySelector('svg');
    expect(overlay).toBeInTheDocument();
    // Expect at least one line segment and a highlight circle.
    expect(overlay?.querySelectorAll('line').length).toBeGreaterThan(0);
    expect(overlay?.querySelectorAll('circle').length).toBeGreaterThan(0);

    rectSpy.mockRestore();
  });
});
