import React from 'react';
import { GrainInstance } from '../types';
import { Info, AlertTriangle, CheckCircle, ShieldAlert, Sparkles, Scale } from 'lucide-react';

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
          Select any grain from the image or grain table to inspect its individual 14-parameter multi-label defect analysis.
        </p>
      </div>
    );
  }

  const { geometry, defects } = grain;

  const defectRows = [
    {
      title: 'Broken Grain',
      status: defects.broken.broken_label === 'broken' ? 'Yes' : defects.broken.broken_label === 'whole' ? 'No' : 'Undetermined',
      isDefect: defects.broken.broken_label === 'broken',
      confidence: defects.broken.confidence ? `${Math.round(defects.broken.confidence * 100)}%` : undefined,
      method: defects.broken.method,
      note: defects.broken.reason,
    },
    {
      title: 'Damaged / Slightly Damaged',
      status: defects.damaged.damaged_label === 'damaged' ? 'Yes' : 'No',
      statusLabel: `Prediction: ${defects.damaged.damaged_label === 'damaged' ? 'Yes' : 'No'}`,
      isDefect: defects.damaged.damaged_label === 'damaged',
      confidence: (defects.damaged.confidence ?? defects.damaged.damaged_probability) != null
        ? `${Math.round((defects.damaged.confidence ?? defects.damaged.damaged_probability) * 100)}%`
        : undefined,
      confidenceLabel: 'Confidence:',
      modelStatus: defects.damaged.model_status === 'experimental'
        ? 'Status: Experimental / uncalibrated'
        : undefined,
      confidenceBasis: defects.damaged.confidence_basis,
      method: defects.damaged.method,
      limitation: defects.damaged.limitation,
    },
    {
      title: 'Discoloured',
      status: defects.discoloured.discoloured_label === 'discoloured' ? 'Yes' : 'No',
      isDefect: defects.discoloured.discoloured_label === 'discoloured',
      confidence: defects.discoloured.discoloured_confidence ? `${Math.round(defects.discoloured.discoloured_confidence * 100)}%` : undefined,
      method: defects.discoloured.method,
    },
    {
      title: 'Chalky',
      status: defects.chalky.chalky_label === 'chalky'
        ? 'Yes'
        : defects.chalky.chalky_label === 'not_chalky'
        ? 'No'
        : 'Undetermined',
      isDefect: defects.chalky.chalky_label === 'chalky',
      confidence: defects.chalky.chalky_probability != null ? `${Math.round(defects.chalky.chalky_probability * 100)}%` : undefined,
      method: defects.chalky.method,
      limitation: defects.chalky.limitation,
    },
    {
      title: 'Red Grain',
      status: defects.red.red_label === 'red' ? 'Yes' : 'No',
      isDefect: defects.red.red_label === 'red',
      confidence: defects.red.red_confidence ? `${Math.round(defects.red.red_confidence * 100)}%` : undefined,
      method: defects.red.red_method,
    },
    {
      title: 'Dehusked (Visual Proxy)',
      status: defects.dehusked.dehusked_label === 'dehusked' ? 'Yes' : 'No',
      isDefect: defects.dehusked.dehusked_label === 'dehusked',
      confidence: defects.dehusked.dehusked_confidence ? `${Math.round(defects.dehusked.dehusked_confidence * 100)}%` : undefined,
      method: defects.dehusked.method,
      limitation: defects.dehusked.limitation,
    },
    {
      title: 'Immature / Shrunken',
      status: defects.immature_shrunken.immature_shrunken_status === 'immature_shrunken' ? 'Yes' : 'No',
      isDefect: defects.immature_shrunken.immature_shrunken_status === 'immature_shrunken',
      confidence: defects.immature_shrunken.confidence ? `${Math.round(defects.immature_shrunken.confidence * 100)}%` : undefined,
      method: defects.immature_shrunken.method,
    },
    {
      title: 'Sprouted / Weevilled',
      status: defects.sprouted_weevilled.sprouted_weevilled_label === 'sprouted_weevilled'
        ? 'Yes'
        : defects.sprouted_weevilled.sprouted_weevilled_label === 'normal'
        ? 'No'
        : 'Undetermined',
      statusLabel: `Prediction: ${defects.sprouted_weevilled.sprouted_weevilled_label === 'sprouted_weevilled' ? 'Yes' : defects.sprouted_weevilled.sprouted_weevilled_label === 'normal' ? 'No' : 'Undetermined'}`,
      isDefect: defects.sprouted_weevilled.sprouted_weevilled_label === 'sprouted_weevilled',
      confidence: (defects.sprouted_weevilled.confidence ?? defects.sprouted_weevilled.probability) != null
        ? `${Math.round((defects.sprouted_weevilled.confidence ?? defects.sprouted_weevilled.probability) * 100)}%`
        : undefined,
      confidenceLabel: 'Confidence:',
      modelStatus: defects.sprouted_weevilled.model_status,
      confidenceBasis: defects.sprouted_weevilled.confidence_basis,
      method: defects.sprouted_weevilled.method,
      limitation: defects.sprouted_weevilled.limitation,
    },
  ];

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
            <span className="font-bold text-warning text-sm">{geometry.lb_ratio ?? 'N/A'}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Mask Area</span>
            <span className="font-bold text-text-primary">{geometry.area_pixels} px</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Solidity</span>
            <span className="font-bold text-text-primary">{geometry.solidity}</span>
          </div>
          <div className="bg-surface-subtle p-2.5 rounded-lg border border-border">
            <span className="text-text-muted block text-[11px]">Confidence</span>
            <span className="font-bold text-success">{Math.round(grain.confidence * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Multi-Label Defect Classification */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Multi-Label Defect Analysis</h4>
          <span className="text-[10px] text-text-muted">Can possess multiple defects simultaneously</span>
        </div>
        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border text-xs">
          {defectRows.map((row) => (
            <div key={row.title} className="p-2.5 flex items-center justify-between hover:bg-surface-subtle transition">
              <div>
                <span className="font-medium text-text-primary block">{row.title}</span>
                <span className="text-[10px] text-text-muted">{row.method}</span>
                {row.modelStatus && (
                  <span className="text-[10px] text-warning block mt-0.5">{row.modelStatus}</span>
                )}
                {row.confidenceBasis && (
                  <span className="text-[10px] text-text-muted block mt-0.5">{row.confidenceBasis}</span>
                )}
                {row.limitation && (
                  <span className="text-[10px] text-warning block mt-0.5">{row.limitation}</span>
                )}
                {row.note && (
                  <span className="text-[10px] text-text-muted block mt-0.5">{row.note}</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {row.confidence && (
                  <span className="text-[10px] text-text-muted font-mono">{row.confidenceLabel || 'Confidence:'} {row.confidence}</span>
                )}
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    row.status === 'Yes'
                      ? 'bg-error-bg text-error border border-error/30'
                      : row.status === 'No'
                      ? 'bg-success-bg text-success border border-success/20'
                      : 'bg-surface-subtle text-text-muted border border-border'
                  }`}
                >
                  {row.statusLabel || row.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
