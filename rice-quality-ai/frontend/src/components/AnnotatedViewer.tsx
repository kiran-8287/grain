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
      <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-12 text-center text-slate-500">
        No annotated image available
      </div>
    );
  }

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col">
      {/* Controls Bar */}
      <div className="px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3 bg-slate-900/60">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-slate-200">Annotated Inspection</span>
          <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full border border-slate-700">
            {grains.length} Grains Mapped
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Grain Selector Dropdown */}
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Crosshair className="w-3.5 h-3.5 text-amber-400" />
            <select
              value={selectedGrainId ?? ''}
              onChange={(e) => onSelectGrain(Number(e.target.value))}
              className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-slate-200 text-xs focus:outline-none focus:border-amber-500"
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
          <div className="flex items-center bg-slate-800 rounded-lg p-0.5 border border-slate-700">
            <button
              onClick={() => setZoom((z) => Math.max(0.75, z - 0.25))}
              className="p-1 hover:text-white text-slate-400 transition"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] px-1.5 text-slate-300 font-mono">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom((z) => Math.min(2.5, z + 0.25))}
              className="p-1 hover:text-white text-slate-400 transition"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setZoom(1)}
              className="p-1 hover:text-white text-slate-400 transition ml-0.5"
              title="Reset Zoom"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Image Canvas Container */}
      <div className="relative overflow-auto p-4 flex items-center justify-center min-h-[380px] max-h-[560px] bg-slate-950/70">
        <div
          style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
          className="transition-transform duration-200"
        >
          <img
            src={annotatedImageUrl}
            alt="Annotated Rice Grains"
            className="rounded-lg shadow-2xl max-w-full h-auto object-contain border border-slate-800/80"
          />
        </div>
      </div>

      {/* Footer Instructions */}
      <div className="px-4 py-2 border-t border-slate-800/60 bg-slate-900/40 text-[11px] text-slate-400 flex items-center justify-between">
        <span>Grains tagged with colored instance masks and #ID tags. Foreign matter shown with red boxes.</span>
        <span>Use dropdown or table to select a grain</span>
      </div>
    </div>
  );
};
