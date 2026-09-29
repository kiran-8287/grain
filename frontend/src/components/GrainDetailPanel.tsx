import React from 'react';
import { GrainInstance } from '../types';
import { Info } from 'lucide-react';

interface GrainDetailPanelProps {
  grain: GrainInstance | null;
  unit: string;
}

export const GrainDetailPanel: React.FC<GrainDetailPanelProps> = ({ grain, unit }) => {
  if (!grain) {
    return (
      <div className="bg-surface border border-border rounded-2xl p-6 text-center text-text-muted">
        <Info className="w-8 h-8 mx-auto text-text-muted mb-2" />
        <h3 className="text-sm font-semibold text-text-primary">No Grain Selected</h3>
        <p className="text-xs text-text-secondary mt-1">
          Select any grain from the image or grain table to inspect its mask-based geometric measurements.
        </p>
      </div>
    );
  }

  const { geometry } = grain;

  const lengthDisplay = geometry.length_mm !== null && geometry.length_mm !== undefined
    ? `${geometry.length_mm} mm`
    : `${geometry.length_pixels} px`;

  const breadthDisplay = geometry.breadth_mm !== null && geometry.breadth_mm !== undefined
    ? `${geometry.breadth_mm} mm`
    : `${geometry.breadth_pixels} px`;

  return (
    <div className="bg-surface border border-border rounded-2xl p-5 shadow-sm space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div>
          <span className="text-xs uppercase font-semibold text-brand tracking-wider">Per-Grain Inspector</span>
          <h2 className="text-xl font-bold text-text-primary">Grain #{grain.id}</h2>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-medium border ${
              grain.segmentation_quality === 'good'
                ? 'bg-success-bg text-success border-success/20'
                : 'bg-warning-bg text-warning border-warning/20'
            }`}
          >
            Quality: {grain.segmentation_quality}
          </span>
          {grain.is_touching && (
            <span className="text-xs px-2.5 py-1 rounded-full font-medium bg-error-bg text-error border border-error/20">
              Touching / Overlapping
            </span>
          )}
        </div>
      </div>

      {/* Geometry Metrics Grid */}
      <div>
        <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-2">Geometric Measurements</h4>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Length</span>
            <span className="font-bold text-text-primary text-sm">{lengthDisplay}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Breadth</span>
            <span className="font-bold text-text-primary text-sm">{breadthDisplay}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">L/B Ratio</span>
            <span className="font-bold text-text-primary text-sm">{geometry.lb_ratio ?? 'N/A'}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Mask Area</span>
            <span className="font-bold text-text-primary text-sm">{geometry.area_pixels.toLocaleString()} px²</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Solidity</span>
            <span className="font-bold text-text-primary text-sm">{geometry.solidity.toFixed(4)}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Confidence</span>
            <span className="font-bold text-success text-sm">{Math.round(grain.confidence * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Segmentation Metadata */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
          <span className="text-text-muted block text-[11px]">Bounding Box</span>
          <span className="font-mono text-text-primary text-[11px]">
            [{grain.bbox.map((v) => Math.round(v)).join(', ')}]
          </span>
        </div>
        <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
          <span className="text-text-muted block text-[11px]">Centroid</span>
          <span className="font-mono text-text-primary text-[11px]">
            ({grain.centroid.map((v) => Math.round(v)).join(', ')})
          </span>
        </div>
      </div>

    </div>
  );
};
