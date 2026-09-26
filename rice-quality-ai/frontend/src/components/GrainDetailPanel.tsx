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
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 text-center text-slate-400">
        <Info className="w-8 h-8 mx-auto text-slate-600 mb-2" />
        <h3 className="text-sm font-semibold text-slate-300">No Grain Selected</h3>
        <p className="text-xs text-slate-400 mt-1">
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
      isDefect: defects.damaged.damaged_label === 'damaged',
      confidence: defects.damaged.damaged_probability ? `${Math.round(defects.damaged.damaged_probability * 100)}%` : undefined,
      method: defects.damaged.method,
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
      status: defects.chalky.chalky_label === 'chalky' ? 'Yes' : 'No',
      isDefect: defects.chalky.chalky_label === 'chalky',
      confidence: defects.chalky.chalky_probability ? `${Math.round(defects.chalky.chalky_probability * 100)}%` : undefined,
      method: defects.chalky.method,
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
      status: defects.sprouted_weevilled.sprouted_weevilled_label === 'sprouted_weevilled' ? 'Yes' : 'No',
      isDefect: defects.sprouted_weevilled.sprouted_weevilled_label === 'sprouted_weevilled',
      confidence: defects.sprouted_weevilled.probability ? `${Math.round(defects.sprouted_weevilled.probability * 100)}%` : undefined,
      method: defects.sprouted_weevilled.method,
    },
  ];

  const lengthDisplay = geometry.length_mm !== null && geometry.length_mm !== undefined
    ? `${geometry.length_mm} mm`
    : `${geometry.length_pixels} px`;

  const breadthDisplay = geometry.breadth_mm !== null && geometry.breadth_mm !== undefined
    ? `${geometry.breadth_mm} mm`
    : `${geometry.breadth_pixels} px`;

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs uppercase font-semibold text-amber-400 tracking-wider">Per-Grain Inspector</span>
          <h2 className="text-xl font-bold text-white">Grain #{grain.id}</h2>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`text-xs px-2.5 py-1 rounded-full font-medium border ${
              grain.segmentation_quality === 'good'
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
            }`}
          >
            Quality: {grain.segmentation_quality}
          </span>
          {grain.is_touching && (
            <span className="text-xs px-2.5 py-1 rounded-full font-medium bg-rose-500/10 text-rose-400 border border-rose-500/20">
              Touching / Overlapping
            </span>
          )}
        </div>
      </div>

      {/* Geometry Metrics Grid */}
      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Geometric Measurements</h4>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">Length</span>
            <span className="font-bold text-slate-100 text-sm">{lengthDisplay}</span>
          </div>
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">Breadth</span>
            <span className="font-bold text-slate-100 text-sm">{breadthDisplay}</span>
          </div>
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">L/B Ratio</span>
            <span className="font-bold text-amber-400 text-sm">{geometry.lb_ratio ?? 'N/A'}</span>
          </div>
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">Mask Area</span>
            <span className="font-bold text-slate-100">{geometry.area_pixels} px</span>
          </div>
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">Solidity</span>
            <span className="font-bold text-slate-100">{geometry.solidity}</span>
          </div>
          <div className="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60">
            <span className="text-slate-400 block text-[11px]">Confidence</span>
            <span className="font-bold text-emerald-400">{Math.round(grain.confidence * 100)}%</span>
          </div>
        </div>
      </div>

      {/* Multi-Label Defect Classification */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Multi-Label Defect Analysis</h4>
          <span className="text-[10px] text-slate-400">Can possess multiple defects simultaneously</span>
        </div>
        <div className="border border-slate-800 rounded-xl overflow-hidden divide-y divide-slate-800/60 text-xs">
          {defectRows.map((row) => (
            <div key={row.title} className="p-2.5 flex items-center justify-between hover:bg-slate-800/30 transition">
              <div>
                <span className="font-medium text-slate-200 block">{row.title}</span>
                <span className="text-[10px] text-slate-400">{row.method}</span>
                {row.limitation && (
                  <span className="text-[10px] text-amber-400/80 block mt-0.5">{row.limitation}</span>
                )}
                {row.note && (
                  <span className="text-[10px] text-slate-400 block mt-0.5">{row.note}</span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {row.confidence && (
                  <span className="text-[10px] text-slate-400 font-mono">{row.confidence}</span>
                )}
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                    row.status === 'Yes'
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                      : row.status === 'No'
                      ? 'bg-emerald-500/10 text-emerald-400'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {row.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sample-Level Parameters Distinction (Section 2 of tasks.txt) */}
      <div className="p-3 bg-slate-800/40 border border-slate-800 rounded-xl text-xs space-y-2">
        <span className="text-[11px] font-semibold text-slate-300 block">Sample-Level Parameters (Not Per-Grain):</span>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-slate-900/60 p-2 rounded border border-slate-700/50">
            <span className="text-[10px] text-slate-400 block">Foreign Matter</span>
            <span className="text-[11px] font-medium text-amber-400">Sample-level parameter</span>
          </div>
          <div className="bg-slate-900/60 p-2 rounded border border-slate-700/50">
            <span className="text-[10px] text-slate-400 block">Admixture</span>
            <span className="text-[11px] font-medium text-amber-400">Sample-level parameter</span>
          </div>
          <div className="bg-slate-900/60 p-2 rounded border border-slate-700/50">
            <span className="text-[10px] text-slate-400 block">Total Count</span>
            <span className="text-[11px] font-medium text-amber-400">Sample-level parameter</span>
          </div>
        </div>
      </div>
    </div>
  );
};
