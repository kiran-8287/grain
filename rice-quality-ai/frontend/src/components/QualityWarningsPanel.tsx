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
    <div className="bg-surface border border-border rounded-2xl p-6 shadow-sm space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div>
          <span className="text-xs font-semibold text-brand uppercase tracking-wider">Provenance & Quality</span>
          <h3 className="text-lg font-bold text-text-primary mt-0.5">Image Quality & Pipeline Diagnostics</h3>
        </div>
        <div className="text-xs bg-surface-subtle px-3 py-1 rounded-lg border border-border text-text-secondary">
          Resolution: <span className="font-semibold text-text-primary">{megapixels} MP</span> (Informational only)
        </div>
      </div>

      {/* Warnings List */}
      {warnings.length > 0 && (
        <div className="bg-warning-bg border border-warning/30 rounded-xl p-4 space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-warning">
            <AlertTriangle className="w-4 h-4" />
            <span>Active Pipeline Warnings ({warnings.length}):</span>
          </div>
          <ul className="list-disc list-inside text-xs text-text-secondary space-y-1">
            {warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* 12-MP Rule Scientific Rationale */}
      <div className="p-3.5 bg-surface-subtle border border-border rounded-xl text-xs space-y-1">
        <span className="font-semibold text-text-primary block text-[11px]">
          Engineering Principle: No Fixed 12-Megapixel Gate
        </span>
        <p className="text-text-secondary leading-relaxed text-[11px]">
          Total image megapixels does not equal pixels-per-grain. A 2 MP close-up of 1 grain contains far more detail
          than a 12 MP image with 1,000 tiny grains. The system assesses quality post-segmentation based on median grain pixel area
          ({quality?.median_grain_pixels ?? 0} px/grain) and Laplacian blur variance ({Math.round(quality?.blur_score ?? 0)}).
        </p>
      </div>

      {/* Methodology Provenance Matrix */}
      <div>
        <h4 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-2">
          Parameter Provenance & Scientific Basis
        </h4>
        <div className="border border-border rounded-xl overflow-hidden divide-y divide-border text-xs">
          <div className="grid grid-cols-4 p-2.5 bg-surface-subtle text-[10px] font-semibold text-text-muted uppercase tracking-wider">
            <div>Parameter</div>
            <div>Classification Basis</div>
            <div>Methodology / Model</div>
            <div>Scientific Class</div>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Broken</span>
            <span>Length &lt; 0.75 * L_whole</span>
            <span>Iterative robust median estimator</span>
            <span className="text-success">Official Concept</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Chalky</span>
            <span>LAB brightness + GLCM texture</span>
            <span>Logistic Regression (models/chalky)</span>
            <span className="text-info">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Damaged</span>
            <span>Aspect-padded crop 224x224</span>
            <span>VGG-19 Transfer Learning</span>
            <span className="text-info">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Discoloured</span>
            <span>Adaptive LAB DeltaE distance</span>
            <span>Population LAB Euclidean DeltaE</span>
            <span className="text-warning">Engineering Heuristic</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Dehusked</span>
            <span>Bran-like brown surface coverage</span>
            <span>HSV color proxy (non-chemical)</span>
            <span className="text-text-secondary">Visual Proxy (Not Staining)</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Foreign Matter</span>
            <span>Full-image object detection</span>
            <span>YOLO11n non-rice detector</span>
            <span className="text-success">Literature ML</span>
          </div>
          <div className="grid grid-cols-4 p-2.5 text-text-secondary">
            <span className="font-medium text-text-primary">Admixture</span>
            <span>Sample-level geometric outlier</span>
            <span>Mahalanobis distance on whole grains</span>
            <span className="text-warning">Statistical Proxy</span>
          </div>

        </div>
      </div>
    </div>
  );
};
