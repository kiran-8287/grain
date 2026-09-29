import React from 'react';
import { GrainInstance } from '../types';

interface GrainTableProps {
  grains: GrainInstance[];
  selectedGrainId: number | null;
  onSelectGrain: (id: number) => void;
  unit: string;
}

const confidenceBadgeClass = (label: 'HIGH' | 'MEDIUM' | 'LOW' | undefined) => {
  switch (label) {
    case 'HIGH':
      return 'bg-success-bg text-success border-success/20';
    case 'MEDIUM':
      return 'bg-warning-bg text-warning border-warning/20';
    case 'LOW':
      return 'bg-error-bg text-error border-error/20';
    default:
      return 'bg-surface-subtle text-text-muted border border-border';
  }
};

export const GrainTable: React.FC<GrainTableProps> = ({
  grains,
  selectedGrainId,
  onSelectGrain,
  unit,
}) => {
  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-semibold text-brand uppercase tracking-wider">Per-Grain Table</span>
          <h3 className="text-base font-bold text-text-primary">Grain Population Overview</h3>
        </div>
        <span className="text-xs text-text-muted">{grains.length} accepted instances</span>
      </div>

      <div className="overflow-x-auto max-h-72 overflow-y-auto border border-border rounded-xl">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="sticky top-0 bg-surface-subtle text-text-secondary font-semibold uppercase text-[10px] tracking-wider z-10">
            <tr>
              <th className="py-2 px-3">Grain #</th>
              <th className="py-2 px-3">Confidence</th>
              <th className="py-2 px-3">Length ({unit})</th>
              <th className="py-2 px-3">Breadth ({unit})</th>
              <th className="py-2 px-3">L/B Ratio</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {grains.map((g) => {
              const isSelected = g.id === selectedGrainId;
              const len = g.geometry.length_mm !== null && g.geometry.length_mm !== undefined
                ? g.geometry.length_mm
                : g.geometry.length_pixels;
              const brd = g.geometry.breadth_mm !== null && g.geometry.breadth_mm !== undefined
                ? g.geometry.breadth_mm
                : g.geometry.breadth_pixels;

              return (
                <tr
                  key={g.id}
                  onClick={() => onSelectGrain(g.id)}
                  className={`cursor-pointer transition ${
                    isSelected
                      ? 'bg-brand-light border-l-4 border-brand'
                      : 'hover:bg-surface-subtle'
                  }`}
                >
                  <td className="py-2.5 px-3 font-semibold text-text-primary">#{g.id}</td>
                  <td className="py-2.5 px-3">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${confidenceBadgeClass(g.confidence_label)}`}>
                      {g.confidence_label ?? `${Math.round(g.confidence * 100)}%`}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-text-secondary">{len}</td>
                  <td className="py-2.5 px-3 font-mono text-text-secondary">{brd}</td>
                  <td className="py-2.5 px-3 font-mono text-brand font-medium">
                    {g.geometry.lb_ratio ?? 'N/A'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
