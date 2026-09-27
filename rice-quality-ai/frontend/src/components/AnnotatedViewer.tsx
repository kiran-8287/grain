import React, { useState } from 'react';
import { Eye, EyeOff, ZoomIn, ZoomOut, Maximize2, Crosshair } from 'lucide-react';
import { GrainInstance } from '../types';

interface AnnotatedViewerProps {
  annotatedImageUrl?: string;
  grains: GrainInstance[];
  selectedGrainId: number | null;
  onSelectGrain: (id: number) => void;
}

export const AnnotatedViewer: React.FC<AnnotatedViewerProps> = ({
  annotatedImageUrl,
  grains,
  selectedGrainId,
  onSelectGrain,
}) => {
  const [zoom, setZoom] = useState(1);

  if (!annotatedImageUrl) {
    return (
      <div className="bg-surface-subtle border border-border rounded-2xl p-12 text-center text-text-muted">
        No annotated image available
      </div>
    );
  }

  return (
    <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex flex-col">
      {/* Controls Bar */}
      <div className="px-4 py-3 border-b border-border flex flex-wrap items-center justify-between gap-3 bg-surface-subtle">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-text-primary">Annotated Inspection</span>
          <span className="text-[10px] bg-surface border border-border text-text-muted px-2 py-0.5 rounded-full">
            {grains.length} Grains Mapped
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Grain Selector Dropdown */}
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Crosshair className="w-3.5 h-3.5 text-brand" />
            <select
              value={selectedGrainId ?? ''}
              onChange={(e) => onSelectGrain(Number(e.target.value))}
              className="bg-surface border border-border rounded-md px-2 py-1 text-text-primary text-xs focus:outline-none focus:border-brand"
            >
              <option value="">Select Grain #ID...</option>
              {grains.map((g) => (
                <option key={g.id} value={g.id}>
                  Grain #{g.id} {g.is_touching ? '(Touching)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Zoom controls */}
          <div className="flex items-center bg-surface rounded-lg p-0.5 border border-border">
            <button
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
              className="p-1 hover:text-text-primary text-text-muted transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] px-1.5 text-text-secondary font-mono">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
              className="p-1 hover:text-text-primary text-text-muted transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1)}
              className="p-1 hover:text-text-primary text-text-muted transition ml-0.5"
              title="Reset Zoom"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Image Canvas Container */}
      <div className="relative overflow-auto p-4 flex items-center justify-center min-h-[380px] bg-surface-subtle">
        <div
          style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
          className="transition-transform duration-200 w-full"
        >
          <img
            src={annotatedImageUrl}
            alt="Annotated Rice Grains"
            className="block w-full max-w-full h-auto object-contain rounded-lg shadow-sm border border-border"
          />
        </div>
      </div>

      {/* Footer Instructions */}
      <div className="px-4 py-2 border-t border-border bg-surface-subtle text-[11px] text-text-muted flex items-center justify-between">
        <span>Grains tagged with colored instance masks and #ID tags. Foreign matter shown with red boxes.</span>
        <span>Use dropdown or table to select a grain</span>
      </div>
    </div>
  );
};
