import React from 'react';
import { ImageQuality } from '../types';
import { ShieldAlert, AlertTriangle, CheckCircle, HelpCircle, FileCheck, Layers } from 'lucide-react';

interface QualityWarningsPanelProps {
  quality?: ImageQuality;
  warnings: string[];
  calibrationMode: string;
  pixelsPerMm?: number | null;
  megapixels: number;
}

export const QualityWarningsPanel: React.FC<QualityWarningsPanelProps> = ({
  quality,
  warnings,
  calibrationMode,
  pixelsPerMm,
  megapixels,
}) => {
  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">Provenance & Quality</span>
          <h3 className="text-lg font-bold text-white mt-0.5">Image Quality & Pipeline Diagnostics</h3>
        </div>
        <div className="text-xs bg-slate-800 px-3 py-1 rounded-lg border border-slate-700 text-slate-300">
          Resolution: <span className="font-semibold text-white">{megapixels} MP</span> (Informational only)
        </div>
      </div>

      {/* Warnings List */}
      {warnings.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-amber-400">
            <AlertTriangle className="w-4 h-4" />
            <span>Active Pipeline Warnings ({warnings.length}):</span>
          </div>
          <ul className="list-disc list-inside text-xs text-amber-200/90 space-y-1">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* 12-MP Rule Scientific Rationale (Section 6 of tasks.txt) */}
      <div className="p-3.5 bg-slate-800/40 border border-slate-800 rounded-xl text-xs space-y-1">
        <span className="font-semibold text-slate-200 block text-[11px]">
          Engineering Principle: No Fixed 12-Megapixel Gate
        </span>
        <p className="text-slate-400 leading-relaxed text-[11px]">
          Total image megapixels does not equal pixels-per-grain. A 2 MP close-up of 1 grain contains far more detail
          than a 12 MP image with 1,000 tiny grains. The system assesses quality post-segmentation based on median grain pixel area
          ({quality?.median_grain_pixels ?? 0} px/grain) and Laplacian blur variance ({Math.round(quality?.blur_score ?? 0)}).
        </p>
      </div>

      {/* Methodology Provenance Matrix */}
      <div>
        <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
          Parameter Provenance & Scientific Basis
        </h4>
        <div className="border border-slate-800 rounded-xl overflow-hidden divide-y divide-slate-800/60 text-xs">
          <div className="grid grid-cols-4 p-2.5 bg-slate-800/60 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
            <div>Parameter</div>
            <div>Classification Basis</div>
            <div>Methodology / Model</div>
            <div>Scientific Class</div>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Broken</span>
            <span>Length &lt; 0.75 * L_whole</span>
            <span>Iterative robust median estimator</span>
            <span className="text-emerald-400">Official Concept</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Chalky</span>
            <span>LAB brightness + GLCM texture</span>
            <span>Logistic Regression (models/chalky)</span>
            <span className="text-blue-400">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Damaged</span>
            <span>Aspect-padded crop 224x224</span>
            <span>VGG-19 Transfer Learning</span>
            <span className="text-blue-400">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Discoloured</span>
            <span>Adaptive LAB DeltaE distance</span>
            <span>Population LAB Euclidean DeltaE</span>
            <span className="text-amber-400">Engineering Heuristic</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Dehusked</span>
            <span>Bran-like brown surface coverage</span>
            <span>HSV color proxy (non-chemical)</span>
            <span className="text-purple-400">Visual Proxy (Not Staining)</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Foreign Matter</span>
            <span>Full-image object detection</span>
            <span>YOLO11n non-rice detector</span>
            <span className="text-emerald-400">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Admixture</span>
            <span>Sample-level geometric outlier</span>
            <span>Mahalanobis distance on whole grains</span>
            <span className="text-amber-400">Statistical Proxy</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-slate-300">
            <span className="font-medium text-white">Moisture</span>
            <span>Physical/laboratory measurement</span>
            <span>Explicitly Not Measured</span>
            <span className="text-rose-400">Excluded (Non-RGB)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
