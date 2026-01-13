import React from 'react';
import { Dialog } from './ui/Dialog';
import { Button } from './ui/Button';
import type { Position } from '../../shared/types/game';

export interface TerritoryRegionOption {
  regionId: string;
  size: number;
  representativePosition: Position;
  moveId: string;
  // RR-FIX-2026-01-13: Include spaces for highlighting successive territories correctly.
  spaces?: Position[];
}

export interface TerritoryRegionChoiceDialogProps {
  isOpen: boolean;
  options: TerritoryRegionOption[];
  onClose: () => void;
  onConfirm: (option: TerritoryRegionOption) => void;
}

/**
 * Dialog for disambiguating which territory region to process when the user
 * clicks on a cell that belongs to multiple overlapping disconnected regions.
 */
export function TerritoryRegionChoiceDialog({
  isOpen,
  options,
  onClose,
  onConfirm,
}: TerritoryRegionChoiceDialogProps) {
  // Color classes matching the territory-region-a, territory-region-b, etc. CSS
  const regionColors = [
    'bg-cyan-500/20 border-cyan-400 hover:bg-cyan-500/30',
    'bg-pink-500/20 border-pink-400 hover:bg-pink-500/30',
    'bg-amber-500/20 border-amber-400 hover:bg-amber-500/30',
    'bg-violet-500/20 border-violet-400 hover:bg-violet-500/30',
  ];

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      labelledBy="territory-region-choice-title"
      describedBy="territory-region-choice-description"
      className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-sm p-6 space-y-4"
      overlayTestId="territory-region-choice-overlay"
    >
      <div className="space-y-1">
        <h2 id="territory-region-choice-title" className="text-lg font-bold text-slate-100">
          Multiple Territories
        </h2>
        <p id="territory-region-choice-description" className="text-sm text-slate-300">
          This cell belongs to multiple territory regions. Choose which one to process:
        </p>
      </div>

      <div className="space-y-2">
        {options.map((option, index) => {
          const colorClass = regionColors[index % regionColors.length];
          return (
            <button
              key={option.regionId}
              onClick={() => onConfirm(option)}
              className={`w-full px-4 py-3 rounded-lg border-2 text-left transition-colors ${colorClass}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-slate-100">Region {index + 1}</span>
                <span className="text-sm text-slate-300">{option.size} spaces</span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="flex justify-end pt-2">
        <Button type="button" variant="secondary" size="sm" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </Dialog>
  );
}
